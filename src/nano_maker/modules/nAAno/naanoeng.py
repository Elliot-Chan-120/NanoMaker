# protein feature engineering:
# spatial data + physicochemical data
# add parts of GEM to append a physiochemical vector on top of spatial one
# - charge, hydrophobicity, pKa, H-bond capacity, .etc
# GEM properties = nAAno_token

# feature vectors as "tokens" per amino acid
# put stuff in nAAno_library for physicochemical analysis
from src.nano_maker.modules.nAAno.naanolibrary import *
import torch

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
            MOLECULAR_WEIGHTS[aa_id] * 100,  # scale up molecular weights by a lot, all in 0.1 range
            NET_CHARGES[aa_id] * 5,  # scale up charge
            ISOELECTRIC_PTS[aa_id],
            HYDROPHOBICITY_IDXS[aa_id] * 10, # same thing
            # generally we want all of them to be on a whole number ish scale
        ]
        naano_vector += FUNCTIONAL_FP[aa_id]
        naano_vector += [p * 10 for p in PROPENSITIES[aa_id]]
        return naano_vector

    # generation + training data processing
    def get_nAAno_X(self, coord_context, bioch_context, coord_Y):
        """
        Generates contextual data for naano dataset class and NAAnoBot's amino acid cage generation
        :param coord_context:
        :param bioch_context:
        :param coord_Y:
        :return:
        """
        # go through coordinate context (iterate through range) <- vectorized now
        # calculate and add relative coordinates concat. with nAAnovectors
        # construct augmented relative coordinate vector then concat with nAAno_token
        coord_X_tensor = torch.tensor(coord_context, dtype=torch.float32)
        bioch_tensor = torch.tensor(bioch_context, dtype=torch.float32)
        coord_Y_tensor = torch.tensor(coord_Y, dtype=torch.float32)

        # XYZ
        xyz_X = self.sph_to_xyz(coord_X_tensor)
        # workaround the batch processing nature -> add dimension at 0th then undo
        xyz_Y = self.sph_to_xyz(coord_Y_tensor.unsqueeze(0)).squeeze(0)

        relative_vect = xyz_X - xyz_Y.unsqueeze(0)   # 3
        euclidean = (torch.norm(relative_vect, dim=-1, keepdim=True) + 1e-8) # 1
        unit_dir = relative_vect / euclidean  # 3

        r_diff = (coord_Y_tensor[0] - coord_X_tensor[:, 0]).unsqueeze(1) # 1
        az_diff = self.angle_diff(coord_X_tensor[:, 1], coord_Y_tensor[1]).unsqueeze(1) # 1
        pl_diff = self.angle_diff(coord_X_tensor[:, 2], coord_Y_tensor[2]).unsqueeze(1) # 1

        # nAAno "token" = 22
        # spatial + 10
        # total features is 32 -> output 22 on linear head

        return torch.cat([bioch_tensor, relative_vect, euclidean,
                          unit_dir, r_diff, az_diff, pl_diff], dim=-1)

    def approx_id(self, vector):
        min_error = float('inf')
        approximate_identity = None
        vector = vector.detach().float().squeeze()
        for aa_id, n_v in self.nAAno_vectors.items():
            n_v_tensor = torch.tensor(n_v, dtype=torch.float32)
            error = torch.norm(vector - n_v_tensor).item()
            if error < min_error:
                min_error = error
                approximate_identity = aa_id

        return approximate_identity


    @staticmethod
    def sph_to_xyz(spherical_vector_batch):
        v = spherical_vector_batch
        r, az, pl = v[:, 0], v[:, 1], v[:, 2]
        x = r * torch.sin(pl) * torch.cos(az)
        y = r * torch.sin(pl) * torch.sin(az)
        z = r * torch.cos(pl)
        return torch.stack([x, y, z], dim=-1)

    @staticmethod
    def angle_diff(aX, aY):
        return (torch.cos(aX) - torch.cos(aY))**2 + (torch.sin(aX) - torch.sin(aY))**2

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