# protein tokenizing:
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

# save this
from nAAno_library import *

class NAAnoEnc:
    """Run this everytime we need a new set of embeddings"""
    def __init__(self, verbose=False):
        self.nAAno_emb = {}   # basically just make it easier to embed down the line
        self.verbose = verbose

    def initialize(self):
        self.nAAno_emb = {aa_id: get_embedding(aa_id) for aa_id in AA_IDS}

        if self.verbose:
            print("NAAnoEnc initialized")
        return True

    def get_aa_id(self, emb_vector):
        aa_id = None
        for code, key in self.nAAno_emb.items():
            if key == emb_vector:
                aa_id = code
                break

        if aa_id is None:
            raise ValueError(f"Embedding vector {emb_vector} presented does not exist")

        return aa_id


def get_embedding(aa_id: str):
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
    embedding = [
        MOLECULAR_WEIGHTS[aa_id],
        NET_CHARGES[aa_id],
        ISOELECTRIC_PTS[aa_id],
        HYDROPHOBICITY_IDXS[aa_id],
        HALF_LIFE[aa_id],
    ]
    embedding += FUNCTIONAL_FP[aa_id]
    return embedding


def encoder_check():
    encoder = NAAnoEnc(verbose=True)
    encoder.initialize()
    for aa_code, aa_vect in encoder.nAAno_emb.items():
        print(f"{aa_code} -- {aa_vect}")

    # check decoder and encoder
    for aa in AA_IDS:
        aa_str = aa
        aa_emb = get_embedding(aa)
        if aa_str == encoder.get_aa_id(aa_emb):
            print(f"{aa_str}: str <-> vect aligned")

# encoder_check()  # note: all good