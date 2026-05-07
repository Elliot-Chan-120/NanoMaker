from src.utility.data_modules import *
from src.utility.path_modules import *

# protein tokenizing: 1 token = spatial data + physicochemical data
# when you pass an AA seq through ESM2, you get back a matrix of shape:
# [sequence length x embedding dimension]
# each row is a per-residue embedding, a vector that encodes that AA's...
# - identity, local chemical environment, inferred structural context as a CLS token
# add parts of GEM to append a physiochemical token onto CLS token
# - charge, hydrophobicity, pKa, H-bond capacity, .etc
# GEM properties = nAAno_token