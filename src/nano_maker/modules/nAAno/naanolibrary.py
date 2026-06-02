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
    'VOID': -0.1
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
    'VOID': 0
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
    'VOID': 0.00,
}

# hydrophobicity levels - pH 7
# +ve values and up = hydrophobic
# around 0 = neutral
# -ve values = hydrophilic
HYDROPHOBICITY_IDXS = {
    'A': 0.41,
    'C': 0.49,
    'D': -0.55,
    'E': -0.31,
    'F': 1,
    'G': 0.00,
    'H': 0.08,
    'I': 1.00,
    'K': -0.23,
    'L': 0.97,
    'M': 0.74,
    'N': -0.28,
    'P': -0.46,
    'Q': -0.10,
    'R': -0.14,
    'S': -0.5,
    'T': 0.13,
    'V': 0.76,
    'W': 0.97,
    'Y': 0.63,
    'VOID': -2,
}
# reference for the above: see GEM README.md


# individual half lives -> divided by 100
HALF_LIFE = {
    'A': 0.044,
    'C': 0.012,
    'D': 0.011,
    'E': 0.01,
    'F': 0.01,
    'G': 0.30,
    'H': 0.035,
    'I': 0.20,
    'K': 0.013,
    'L': 0.055,
    'M': 0.30,
    'N': 0.014,
    'P': 0.21,
    'Q': 0.08,
    'R': 0.01,
    'S': 0.019,
    'T': 0.072,
    'V': 1,
    'W': 0.028,
    'Y': 0.028,
    'VOID': -1,
}
# Reference: Biochemistry: Paul Jay Friedman (Fifth Edition), + biochemistry notes


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
    "VOID": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
}


# [P_alpha, P_beta, P_loops, P_hinge]
PROPENSITIES = {
    'A': [1.46, 0.87, 0.71, 1.00],
    'C': [0.93, 1.37, 0.84, 0.97],
    'D': [0.96, 0.54, 1.31, 0.87],
    'E': [1.46, 0.65, 0.85, 0.89],
    'F': [1.04, 1.44, 0.70, 0.88],
    'G': [0.42, 0.36, 1.85, 0.96],
    'H': [0.85, 1.16, 1.02, 0.89],
    'I': [0.86, 1.85, 0.59, 0.76],
    'K': [1.12, 0.86, 0.99, 1.24],
    'L': [1.32, 1.15, 0.65, 0.76],
    'M': [1.39, 0.76, 0.83, 0.41],
    'N': [0.75, 0.66, 1.40, 1.21],
    'P': [0.55, 0.35, 1.75, 1.83],
    'Q': [1.25, 0.81, 0.92, 1.10],
    'R': [1.17, 0.88, 0.94, 1.10],
    'S': [0.68, 0.90, 1.31, 1.41],
    'T': [0.78, 1.29, 0.99, 1.02],
    'V': [0.84, 1.90, 0.58, 0.82],
    'W': [1.13, 1.19, 0.78, 0.86],
    'Y': [0.98, 1.39, 0.78, 1.02],
    'VOID': [0, 0, 0, 0],
}

# Analysis of Domain-Swapped Oligomers Reveals Local Sequence Preferences and Structural Imprints at the Linker Regions and Swapped Interfaces - Scientific Figure on ResearchGate.
# Available from: https://www.researchgate.net/figure/Propensities-of-amino-acids-to-be-in-alpha-helix-beta-sheets-loops-and-hinge_fig12_230589968 [accessed 1 Jun 2026]