# protein feature engineering:
# spatial data + physicochemical data
# when you pass an AA seq through ESM2, you get back a matrix of shape:
# [sequence length x embedding dimension] - big problem
# each row is a per-residue embedding, a vector that encodes that AA's...
# - identity, local chemical environment, inferred structural context as a vector
# add parts of GEM to append a physiochemical vector on top of spatial one
# - charge, hydrophobicity, pKa, H-bond capacity, .etc
# GEM properties = nAAno_token

# I don't think we need a module for tokenizing, this isn't NLP
# put stuff in nAAno_library for physicochemical analysis

# TODO: ramachandran freedom feature

from src.nano_maker.modules.nAAno.naanolibrary import *

class NAAnoEng:
    """Run this everytime we need a new set of feature vectors"""
    def __init__(self, verbose=False):
        self.verbose = verbose

    def initialize(self):
        self.nAAno_vectors = {aa_id: build_nAAnoVector(aa_id) for aa_id in AA_IDS}

        if self.verbose:
            print("NAAnoEng initialized")
        return True

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
    return naano_vector


def encoder_check(verbose=True):
    module = NAAnoEng(verbose=True)
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
# encoder_check()  # note: all good