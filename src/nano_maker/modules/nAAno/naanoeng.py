# protein feature engineering:
# spatial data + physicochemical data
# add parts of GEM to append a physiochemical vector on top of spatial one
# - charge, hydrophobicity, pKa, H-bond capacity, .etc
# GEM properties = nAAno_token

# feature vectors as "tokens" per amino acid
# put stuff in nAAno_library for physicochemical analysis

import torch
from src.nano_maker.modules.nAAno.naanolibrary import *

class NAAnoEng:
    """Run this everytime we need a new set of feature vectors"""
    def __init__(self, max_angstroms, block_size, verbose=False):
        self.max_angstroms = max_angstroms
        self.block_size = block_size
        self.verbose = verbose

    def initialize(self):
        self.nAAno_vectors = {aa_id: self.build_nAAnoVector(aa_id) for aa_id in AA_IDS}

        if self.verbose:
            print("NAAnoEng initialized")
        return True

    def n_features(self):
        return len(self.nAAno_vectors['A'])

    def get_aa_id(self, naano_vector):
        aa_id = None
        for code, key in self.nAAno_vectors.items():
            if key == naano_vector:
                aa_id = code
                break

        if aa_id is None:
            raise ValueError(f"Feature vector {naano_vector} presented does not exist")

        return aa_id

    def get_nAAnovector(self, aa_id):
        return self.nAAno_vectors[aa_id]

    @staticmethod
    def build_nAAnoVector(aa_id: str):
        """
        use this in these two scenarios:
        \n    - generating embeddings after updating feature vector
        \n    - inference
        :param aa_id: single letter amino acid reference code
        :returns: feature vector representing a given (valid) amino acid
        """
        # sanity check
        if aa_id not in AA_IDS:
            raise ValueError(f"{aa_id} not in valid AA ID list")

        # embedding scheme, MAKE SURE TO UPDATE THIS IF YOU EVER UPDATE NAANOLIBRARY
        naano_vector = [
            MOLECULAR_WEIGHTS[aa_id],
            NET_CHARGES[aa_id],
            ISOELECTRIC_PTS[aa_id],
            HYDROPHOBICITY_IDXS[aa_id],
            HALF_LIFE[aa_id],
        ]
        naano_vector += FUNCTIONAL_FP[aa_id]
        naano_vector += PROPENSITIES[aa_id]
        return naano_vector

    # generation + training data processing
    def get_nAAno_X(self, coord_context, bioch_context, coord_Y):
        # go through coordinate context (iterate through range)
        # calculate and add relative coordinates concat. with nAAnovectors
        naano_X = []
        coord_Y = torch.tensor(coord_Y, dtype=torch.float32)

        for idx in range(self.block_size):
            coord_X = torch.tensor(coord_context[idx], dtype=torch.float32)
            naanovector_X = torch.tensor(bioch_context[idx])
            # construct augmented relative coordinate vector then concat with nAAno_token
            # XYZ
            relative_vect = self.sph_to_xyz(coord_X) - self.sph_to_xyz(coord_Y)  # 3
            euclidean_dist = (torch.norm(relative_vect) + 1e-8).unsqueeze(0)   # 1
            unit_dir = (relative_vect / euclidean_dist)   # 3

            # spherical
            Xr, Xaz, Xpl = coord_X
            Yr, Yaz, Ypl = coord_Y
            r_diff = Yr - Xr.unsqueeze(0)    # 1
            az_diff = self.angle_diff(Xaz, Yaz).unsqueeze(0)   # 1
            pl_diff = self.angle_diff(Xpl, Ypl).unsqueeze(0)  # 1

            naano_X.append(torch.cat([naanovector_X, relative_vect, euclidean_dist, unit_dir,
                                        r_diff, az_diff, pl_diff]))

        return torch.stack(naano_X)  # block_size, 32

    def approx_id(self, vector):
        min_error = None
        approximate_identity = None
        for aa_id, n_v in self.nAAno_vectors.items():
            error = vector - n_v
            if error < min_error:
                min_error = error
                approximate_identity = aa_id

        return approximate_identity


    @staticmethod
    def sph_to_xyz(spherical_vector):
        v = torch.tensor(spherical_vector, dtype=torch.float32)
        r, az, pl = v[0], v[1], v[2]
        x = r * torch.sin(pl) * torch.cos(az)
        y = r * torch.sin(pl) * torch.sin(az)
        z = r * torch.cos(pl)
        return torch.stack([x, y, z])

    @staticmethod
    def angle_diff(aX, aY):
        return ((torch.cos(aX) - torch.cos(aY))**2 + (torch.sin(aX) - torch.sin(aY))**2).mean()

def encoder_check(verbose=True):
    module = NAAnoEng(max_angstroms=42, block_size=42, verbose=False)
    module.initialize()
    for aa_code, aa_vect in module.nAAno_vectors.items():
        print(f"{aa_code} -- {aa_vect}")
    print(f"{module.n_features()} features")

    # check decoder and encoder
    for aa in AA_IDS:
        aa_str = aa
        aa_emb = module.get_nAAnovector(aa)
        if aa_str == module.get_aa_id(aa_emb):
            if verbose:
                print(f"{aa_str}: str <-> vect aligned")
        else:
            raise ValueError(f"Ensure {aa} in nAAno_library is up to date")
# encoder_check()  # note: all good