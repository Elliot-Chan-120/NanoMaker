# Once the full transformer architecture is up I'll put it here
# right now just have it generate the skeleton -> once naanobot is in we add in its functionality to this
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.paths import *
from src.nano_maker.modules.nAAno.smiles_handler import *
from src.nano_maker.modules.nAAno.radialseeker import RadialSeeker
from src.nano_maker.modules.nAAno.naanoeng import NAAnoEng
from src.nano_maker.skeleton import Skeleton

class NanoMaker:

    def __init__(self, skeleton_weight_filename, skeleton_cfg, radial_cfg):
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
        self.map4_fingerprint = None

        # MODULES ======================================================================================================
        self._RadialSeeker = RadialSeeker(radial_resolution=radial_cfg['radial_resolution'],
                                          max_angstroms=radial_cfg['max_angstroms'],
                                          verbose=False)
        self._NAAnoEng = NAAnoEng(verbose=False)


    def ingest_chemical(self, smiles):
        """easier to overwrite"""
        rules_passed = eval_lipinski(smiles)
        self.map4_fingerprint = torch.tensor(smiles_fingerprint(smiles), dtype=torch.float32).unsqueeze(0)

    def generate(self):
        """
        Accepts chemical in smiles format, roughly screens it for drug likeness then outputs a 3D coordinate map for proteins
        :return:
        """
        if self.map4_fingerprint is not None:   # make sure NanoMaker has ingested something
            pocket_skeleton = self.pocket_skeleton()
        else:
            return ValueError("Run function: ingest_chemical prior to attempting to generate protein cage")

        return pocket_skeleton


    def pocket_skeleton(self):
        raw_skeleton = self._SkeletonPrototype.generate(self.map4_fingerprint)   # initial processing -> turn into a regular python list
        process_skeleton = [vector.detach().flatten().tolist() for vector in raw_skeleton]

        # locates stop radius "token" -> anything lower than angstrom cutoff (0.33)
        pruned_skeleton = []
        for vector in process_skeleton:
            if abs(vector[0]) <= self.angstrom_cutoff:
                break
            pruned_skeleton.append(vector)

        # translate from spherical coordinate system to xyz coordinates
        pocket_skeleton = [self._RadialSeeker.radial_to_xyz(vector) for vector in pruned_skeleton]
        return pocket_skeleton
