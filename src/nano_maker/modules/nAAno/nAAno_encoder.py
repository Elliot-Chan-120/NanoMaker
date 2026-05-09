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

class NAAnoEncoder:
    """Run this everytime we need a new set of embeddings"""
    def __init__(self, verbose=False):
        self.nAAno_emb = {}   # basically just make it easier to embed down the line
        self.verbose = verbose

    def initialize(self):
        AA_embs = {}
        for aa_id in AA_IDS:
            AA_embs[aa_id] = get_embedding(aa_id)
        self.nAAno_emb = AA_embs

        if self.verbose:
            print("NAAnoEncoder initialized")
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
    return embedding


def encoder_check():
    encoder = NAAnoEncoder(verbose=True)
    encoder.initialize()
    print(encoder.nAAno_emb)

    # check decoder and encoder
    for aa in AA_IDS:
        aa_str = aa
        aa_emb = get_embedding(aa)
        if aa_str == encoder.get_aa_id(aa_emb):
            print(f"{aa_str}: str <-> vect aligned")

encoder_check()