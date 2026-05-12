AA_IDS = [
    'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','VOID',
]

# molecular weights are in kilodaltons
MOLECULAR_WEIGHTS = {
    'A': 0.09,
    'C': 0.12,
    'D': 0.13,
    'E': 0.15,
    'F': 0.17,
    'G': 0.08,
    'H': 0.16,
    'I': 0.13,
    'K': 0.15,
    'L': 0.13,
    'M': 0.15,
    'N': 0.13,
    'P': 0.12,
    'Q': 0.15,
    'R': 0.17,
    'S': 0.11,
    'T': 0.12,
    'V': 0.12,
    'W': 0.20,
    'Y': 0.18,
    'VOID': 0,
}

# net charges at physiological pH ~7.4
NET_CHARGES = {
    'A': 0,
    'C': 0,
    'D': -1,
    'E': -1,
    'F': 0,
    'G': 0,
    'H': 0,
    'I': 0,
    'K': +1,
    'L': 0,
    'M': 0,
    'N': 0,
    'P': 0,
    'Q': 0,
    'R': +1,
    'S': 0,
    'T': 0,
    'V': 0,
    'W': 0,
    'Y': 0,
    'VOID': 0,
}

# isoelectric point pHs from peptide web
# NOTE: will retrain model since I was using old textbook values with one outdated value -> shouldn't affect model accuracy much
ISOELECTRIC_PTS = {
    'A': 6.02,
    'C': 5.02,
    'D': 2.98,
    'E': 3.22,
    'F': 5.48,
    'G': 5.97,
    'H': 7.59,
    'I': 5.98,
    'K': 9.47,
    'L': 5.98,
    'M': 5.75,
    'N': 5.41,
    'P': 6.30,
    'Q': 5.65,
    'R': 10.76,
    'S': 5.68,
    'T': 5.60,
    'V': 5.97,
    'W': 5.94,
    'Y': 5.66,
    'VOID': 0,
}

# hydrophobicity levels - pH 7
# +ve values and up = hydrophobic
# around 0 = neutral
# -ve values = hydrophilic
HYDROPHOBICITY_IDXS = {
    'A': 41,
    'C': 49,
    'D': -55,
    'E': -31,
    'F': 100,
    'G': 0,
    'H': 8,
    'I': 100,
    'K': -23,
    'L': 97,
    'M': 74,
    'N': -28,
    'P': -46,
    'Q': -10,
    'R': -14,
    'S': -5,
    'T': 13,
    'V': 76,
    'W': 97,
    'Y': 63,
    'VOID': 0,
}


# individual half lives
HALF_LIFE = {
    'A': 4.4,
    'C': 1.2,
    'D': 1.1,
    'E': 1,
    'F': 1,
    'G': 30,
    'H': 3.5,
    'I': 20,
    'K': 1.3,
    'L': 5.5,
    'M': 30,
    'N': 1.4,
    'P': 21,
    'Q': 0.8,
    'R': 1,
    'S': 1.9,
    'T': 7.2,
    'V': 100,
    'W': 2.8,
    'Y': 2.8,
    'VOID': 0,
}


# pseudo molecular fingerprint accounting for biochemical / functional group standout features
# check function_fp_scheme.txt for more understanding
FUNCTIONAL_FP = {
    "A": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "G": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "D": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "E": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "K": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "R": [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "H": [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "N": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "Q": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "C": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    "M": [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    "F": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "Y": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    "W": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    "L": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "I": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "V": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "S": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "T": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "P": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    "VOID": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}