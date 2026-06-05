# Once the full transformer architecture is up I'll put it here
# right now just have it generate the skeleton -> once naanobot is in we add in its functionality to this
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.paths import *
from src.nano_maker.modules.nAAno.smiles_handler import *
from src.nano_maker.modules.nAAno.radialseeker import RadialSeeker
from src.nano_maker.skeleton import Skeleton
from src.nano_maker.naanobot import NAAnoBot

class NanoMaker:
    """
    Acts as the command centre for protein binding pocket sythesis
    """

    def __init__(self, skeleton_weight_filename, skeleton_cfg, naanobot_weight_filename, naanobot_config, radial_cfg):
        # INITIALIZE SKELETON =========================================================================================
        sk_cfg = skeleton_cfg.copy()

        self._SkeletonPrototype = Skeleton(n_embd=sk_cfg['n_embd'], n_head=sk_cfg['n_head'],
                                                n_layers=sk_cfg['n_layers'],
                                                block_size=sk_cfg['block_size'],
                                                map4_res=sk_cfg['map4_res'], max_angstroms=sk_cfg['max_angstroms'],
                                                dropout=sk_cfg['dropout'])

        skltn_prototype_weights = torch.load(CONTAINER / skeleton_weight_filename, map_location="cpu")
        self._SkeletonPrototype.load_state_dict(skltn_prototype_weights["model_state_dict"])

        self.angstrom_cutoff = float(sk_cfg['max_angstroms'] / sk_cfg['radial_resolution'])

        # INITIALIZE NAANOBOT ==========================================================================================
        nb_cfg = naanobot_config.copy()
        self._NAAnoBotPrototype = NAAnoBot(n_embd=nb_cfg["n_embd"], n_head=nb_cfg["n_head"],
                                           n_layers=nb_cfg["n_layers"],
                                           block_size=nb_cfg["block_size"],
                                           map4_res=nb_cfg["map4_res"],
                                           n_spatial_features=nb_cfg["n_spatial_features"],
                                           max_angstroms=nb_cfg["max_angstroms"],
                                           dropout=nb_cfg["dropout"])

        nnbot_prototype_weights = torch.load(CONTAINER / naanobot_weight_filename, map_location="cpu")
        self._NAAnoBotPrototype.load_state_dict(nnbot_prototype_weights["model_state_dict"])

        # global fingerprint variable for repeated generations
        self._smiles = None
        self._map4_fingerprint = None

        # radial  ======================================================================================================
        self._RadialSeeker = RadialSeeker(radial_resolution=radial_cfg['radial_resolution'],
                                          max_angstroms=radial_cfg['max_angstroms'],
                                          verbose=False)

        # naanoeng is internally called in NAAnoBot for generation ease


    def ingest_chemical(self, smiles):
        """easier to overwrite"""
        self._smiles = smiles
        rules_passed = eval_lipinski(smiles)
        self._map4_fingerprint = torch.tensor(smiles_fingerprint(smiles), dtype=torch.float32).unsqueeze(0)
        print(f"Chemical Ingested: {smiles}")
        print(f"Drug Likeness Rules Passed: {rules_passed} / 4")

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # Base -> generate protein pocket data | coords and biochemical environments
    # allow option to save data as a "radial" file and load it as a radial sequence
    # -> file must contain target SMILES as title up top as a header ">" kinda like fasta idk
    # allow option to visualize a "radial file"
    def generate_pocket_data(self, temperature):
        """
        Accepts chemical in smiles format, roughly screens it for drug likeness then outputs a 3D coordinate map for proteins
        Use this for data generation for visualization or file saving
        :return:
        """
        if self._map4_fingerprint is not None:   # make sure NanoMaker has ingested something
            pocket_sph_skeleton = self._pocket_sph_skeleton()
        else:
            return ValueError("Run function: ingest_chemical prior to attempting to generate protein cage")

        skeleton = self._pocket_xyz_skeleton(pocket_sph_skeleton)
        aa_ids = self._NAAnoBotPrototype.generate(self._map4_fingerprint, pocket_sph_skeleton, temperature=temperature)

        pocket_data = {"SMILES": self._smiles,
                       "Temperature": temperature,
                       "3D_skeleton": skeleton,
                       "aa_ids": aa_ids}
        # radial sequence

        return pocket_data


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


    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # the idea is that once the pocket has been generated by the user
    # they inspect it, visualize it then save it and load it up again for visualization

    def visualize_pocket(self, pocket_data):
        pass

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    def visualize_radial_file(self, filepath):
        """Returns pocket data and smiles"""
        # load up pocket data then pass on to visualize_pocket
        pass

    def save_radial_file(self, pocket_data, filepath):
        pass