# Once the full transformer architecture is up I'll put it here
# right now just have it generate the skeleton -> once naanobot is in we add in its functionality to this
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit.Chem.Scaffolds import MurckoScaffold

from src.paths import *
from src.nano_maker.modules.nAAno.smiles_handler import *
from src.nano_maker.modules.nAAno.radialseeker import RadialSeeker
from src.nano_maker.skeleton import Skeleton
from src.nano_maker.naanobot import NAAnoBot

class NanoMaker:
    """
    Acts as the command centre for protein binding pocket sythesis
    """
    def __init__(self, version_code, skeleton_weight_filename, skeleton_cfg, naanobot_weight_filename, naanobot_cfg, radial_cfg):
        # INITIALIZE SKELETON =========================================================================================
        sk_cfg = skeleton_cfg.copy()

        self._SkeletonPrototype = Skeleton(n_embd=sk_cfg['n_embd'], n_head=sk_cfg['n_head'],
                                                n_layers=sk_cfg['n_layers'],
                                                block_size=sk_cfg['block_size'],
                                                map4_res=sk_cfg['map4_res'], max_angstroms=sk_cfg['max_angstroms'],
                                                dropout=sk_cfg['dropout'])

        skltn_prototype_weights = torch.load(CONTAINER / skeleton_weight_filename, map_location="cpu")
        self._SkeletonPrototype.load_state_dict(skltn_prototype_weights["model_state_dict"])
        self._SkeletonPrototype.eval()


        self.angstrom_cutoff = float(sk_cfg['max_angstroms'] / sk_cfg['radial_resolution'])

        # INITIALIZE NAANOBOT ==========================================================================================
        nb_cfg = naanobot_cfg.copy()
        self._NAAnoBotPrototype = NAAnoBot(n_embd=nb_cfg["n_embd"], n_head=nb_cfg["n_head"],
                                           n_layers=nb_cfg["n_layers"],
                                           block_size=nb_cfg["block_size"],
                                           map4_res=nb_cfg["map4_res"],
                                           n_spatial_features=nb_cfg["n_spatial_features"],
                                           max_angstroms=nb_cfg["max_angstroms"],
                                           dropout=nb_cfg["dropout"],
                                           loss_temperature=nb_cfg["loss_temperature"])

        nnbot_prototype_weights = torch.load(CONTAINER / naanobot_weight_filename, map_location="cpu")
        self._NAAnoBotPrototype.load_state_dict(nnbot_prototype_weights["model_state_dict"])
        self._NAAnoBotPrototype.eval()

        # global fingerprint variable for repeated generations
        self._smiles = None
        self._scaffold_smiles = None
        self._map4_fingerprint = None

        # radial  ======================================================================================================
        self._RadialSeeker = RadialSeeker(radial_resolution=radial_cfg['radial_resolution'],
                                          max_angstroms=radial_cfg['max_angstroms'],
                                          verbose=False)

        # naanoeng is internally called in NAAnoBot for generation ease

        self._version = version_code
        # good to have if down the line this becomes a bigger thing, allows us to check differences across iterations


    def ingest_chemical(self, smiles):
        """easier to overwrite"""
        self._smiles = smiles
        rules_passed = eval_lipinski(smiles)  # preliminary check -> for future info
        scaffold = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smiles))
        scaffold_smiles = Chem.MolToSmiles(scaffold)

        # class variable assignment / update
        self._scaffold_smiles = scaffold_smiles
        self._smiles = smiles
        self._map4_fingerprint = torch.tensor(smiles_fingerprint(scaffold_smiles), dtype=torch.float32).unsqueeze(0)

        print(f"Chemical Ingested: {smiles}")
        print(f"Scaffold: {scaffold_smiles}")
        print(f"Drug Likeness Rules Passed: {rules_passed} / 4")

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # Base -> generate protein pocket data | coords and biochemical environments
    # allow option to save data as a "radial" file and load it as a radial sequence
    # -> file must contain target SMILES as title up top as a header ">" kinda like fasta idk
    def generate_nanopkt_data(self, sampling_temp, sph_coordinates=None):
        """
        Accepts chemical in smiles format, roughly screens it for drug likeness then outputs a 3D coordinate map for proteins
        Use this for data generation for visualization or file saving

        \n If the user has a specific set of spherical coordinates in mind already or wants to place amino acids along
         a predetermined skeleton, they can provide it and naanobot will build on the provided skeleton
        :return:
        """
        if self._map4_fingerprint is not None:   # make sure NanoMaker has ingested something to work with
            if sph_coordinates is None:  # if the user provided nothing -> generate own skeleton
                pocket_sph_skeleton = self._pocket_sph_skeleton()
            else:
                pocket_sph_skeleton = sph_coordinates  # user gave something -> use those
                                                       # down the line need error handling for these
        else:
            return ValueError("Run function: ingest_chemical prior to attempting to generate protein cage")

        raw_skeleton = self._pocket_xyz_skeleton(pocket_sph_skeleton)
        skeleton = [[round(num, 4) for num in coords] for coords in raw_skeleton]
        aa_sequence = self._NAAnoBotPrototype.generate(self._map4_fingerprint, pocket_sph_skeleton, sampling_temperature=sampling_temp)

        nano_pocket_data = {
            "version_code": self._version,
            "SMILES": self._smiles,
            "Scaffold": self._scaffold_smiles,
            "Sampling_temperature": sampling_temp,
            "3D_skeleton": skeleton,
            "aa_sequence": aa_sequence}
        return nano_pocket_data


    def _pocket_sph_skeleton(self):
        raw_skeleton = self._SkeletonPrototype.generate(self._map4_fingerprint)   # initial processing -> turn into a regular python list
        process_skeleton = [vector.detach().flatten().tolist() for vector in raw_skeleton]

        # locates stop radius "token" -> anything lower than angstrom cutoff (0.33)
        sph_skeleton = []
        for vector in process_skeleton:
            if abs(vector[0]) <= self.angstrom_cutoff:
                break
            sph_skeleton.append(vector)

        return sph_skeleton

    def _pocket_xyz_skeleton(self, sph_skeleton):
        # translate from spherical coordinate system to xyz coordinates
        pocket_skeleton = [self._RadialSeeker.radial_to_xyz(vector) for vector in sph_skeleton]
        return pocket_skeleton