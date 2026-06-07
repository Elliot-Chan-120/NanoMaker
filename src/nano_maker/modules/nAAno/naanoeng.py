# protein feature engineering:
# spatial data + physicochemical data
# add parts of GEM to append a physiochemical vector on top of spatial one
# - charge, hydrophobicity, pKa, H-bond capacity, .etc
# GEM properties = nAAno_token

# feature vectors as "tokens" per amino acid
# put stuff in nAAno_library for physicochemical analysis
from src.nano_maker.modules.nAAno.naanolibrary import *
import torch
import torch.nn.functional as F

class NAAnoEng:
    """Run this everytime we need a new set of feature vectors"""
    def __init__(self, max_angstroms, block_size, verbose=False):
        self.max_angstroms = max_angstroms
        self.block_size = block_size
        self.verbose = verbose
        self.nAAno_vectors = None
        self.nAAno_tensors = None


    def initialize(self):
        self.nAAno_vectors = {aa_id: self.build_nAAnoVector(aa_id) for aa_id in AA_IDS}
        self.nAAno_tensors = torch.tensor([v for v in self.nAAno_vectors.values()], dtype=torch.float32)

        if self.verbose:
            print("NAAnoEng initialized")
        return True

    def n_features(self):
        length = len(self.nAAno_vectors['A'])
        for aa, vector in self.nAAno_vectors.items():
            if len(vector) != length:
                raise ValueError(f"nAAno token lengths are not aligned. Check naanolibrary for inconsistent entries.")
        return length

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
            MOLECULAR_WEIGHTS[aa_id] * 5,
            NET_CHARGES[aa_id],
            ISOELECTRIC_PTS[aa_id] / 14,
            HYDROPHOBICITY_IDXS[aa_id],
            RSA_THEORETICAL[aa_id],
            VOL_A[aa_id]
            # generally we want all of them to be on the same scale
        ]
        naano_vector += [sc_fp for sc_fp in SIDE_CHAIN_FINGERPRINT[aa_id]]
        naano_vector += [p / 1.9 for p in PROPENSITIES[aa_id]]  # divide by the max - keep it within -1 <-> 1
        naano_vector += [rel_con for rel_con in RELATIVE_CONFORMATIONAL[aa_id]]
        naano_vector += [abs_con for abs_con in ABSOLUTE_CONFORMATIONAL[aa_id]]
        naano_vector += [con_soft for con_soft in CONFORMATIONAL_SOFTNESS[aa_id]]
        naano_vector += [pka_fp for pka_fp in PKAs[aa_id]]
        naano_vector += [h_pro for h_pro in H_PROFILE[aa_id]]

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

        relative_vect = xyz_X - xyz_Y.unsqueeze(0)  # 3
        euclidean = (torch.norm(relative_vect, dim=-1, keepdim=True) + 1e-8)  # 1
        unit_dir = relative_vect / euclidean  # 3

        r_diff = (coord_Y_tensor[0] - coord_X_tensor[:, 0]).unsqueeze(1)  # 1
        az_diff = self.angle_diff(coord_X_tensor[:, 1], coord_Y_tensor[1]).unsqueeze(1)  # 1
        pl_diff = self.angle_diff(coord_X_tensor[:, 2], coord_Y_tensor[2]).unsqueeze(1)  # 1

        # nAAno "token" = 22
        # spatial + 10
        # total features is 32 -> output 22 on linear head
        return torch.cat([bioch_tensor, relative_vect / self.max_angstroms, euclidean / self.max_angstroms,
                          unit_dir, r_diff / self.max_angstroms, az_diff / 4, pl_diff / 4], dim=-1)
        # normalize angstrom-related metrics by self.max_angstroms
        # normalize angular metrics by 4 -> thats the max range, brings everything down to 0-1


    # INFERENCE / PROTEIN POCKET SYNTHESIS
    def approx_id(self, pred_vector, sampling_temperature):
        """
        amino acid picker, converts error between amino acids into probabilities that are used to sample amino acids
        :param pred_vector:
        :param sampling_temperature:
        :return:
        """
        # note -> add in temperature sampling | getting monotonous outputs right now
        with torch.no_grad():
            aa_ids = [aa_id for aa_id in AA_IDS if aa_id != "VOID"]  # take out the VOID token
            n_v_tensor = torch.tensor([self.nAAno_vectors[aa_id] for aa_id in aa_ids], dtype=torch.float32)


            pred_vector = pred_vector.detach().float().squeeze()
            # mimic loss function in naanobot for amino acid selection
            pred_norm = F.normalize(pred_vector.unsqueeze(0), dim=-1)
            aa_norm = F.normalize(n_v_tensor, dim=-1)

            affinities = (pred_norm @ aa_norm.T).squeeze(0)
            logits = affinities / sampling_temperature

            # I want error to dicate the probability of being chosen
            # less error = more likely % vice versa
            # during testing without temp sampling -> just choosing the most "optimal" amino acid
            # monotonous, no variety that is biochemically observable irl
            # need to introduce diversity to mimic real life pockets

            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1).item()

        return aa_ids[idx]


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
    # check decoder and encoder
    for aa in AA_IDS:
        aa_str = aa
        aa_emb = module.get_nAAnovector(aa)
        if aa_str == module.get_aa_id(aa_emb):
            if verbose:
                print(f"{aa_str}: str <-> vect aligned")
        else:
            raise ValueError(f"Ensure {aa} in nAAno_library is up to date")

    print(f"{module.n_features()} features")
encoder_check()  # note: all good