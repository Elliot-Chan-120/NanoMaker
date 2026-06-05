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
    'VOID': -0.2
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
    'VOID': 0,
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
    'VOID': -1,
}
# reference for the above: see GEM's README.md


# Reference: Biochemistry: Paul Jay Friedman (Fifth Edition), + my biochemistry notes
# pseudo molecular fingerprint accounting for biochemical / functional group standout features
# check function_fp_scheme.txt for more understanding
# n_C, n_O, n_N, n_S, n_H, n_Aromatic, n_resonancepairs (lone pairs and pi bonds)
# max = 9, 2, 3, 1, 11, 2, 5
SIDE_CHAIN_FINGERPRINT = {
    "A": [1/9, 0/2, 0/3, 0,  3/11, 0/2, 0/5],
    "G": [0/9, 0/2, 0/3, 0,  1/11, 0/2, 0/5],
    "D": [2/9, 2/2, 0/3, 0,  2/11, 0/2, 5/5],
    "E": [3/9, 2/2, 0/3, 0,  4/11, 0/2, 5/5],
    "K": [4/9, 0/2, 1/3, 0, 11/11, 0/2, 1/5],
    "R": [4/9, 0/2, 3/3, 0, 11/11, 0/2, 3/5],   #NH neutral , NH2+, pi bond on guanidinium
    "H": [4/9, 0/2, 2/3, 0,  6/11, 1/2, 3/5],
    "N": [2/9, 1/2, 1/3, 0,  4/11, 0/2, 4/5],
    "Q": [3/9, 1/2, 1/3, 0,  6/11, 0/2, 4/5],
    "C": [1/9, 0/2, 0/3, 1,  3/11, 0/2, 2/5],
    "M": [3/9, 0/2, 0/3, 1,  7/11, 0/2, 1/5],
    "F": [7/9, 0/2, 0/3, 0,  7/11, 1/2, 3/5],
    "Y": [7/9, 1/2, 0/3, 0,  7/11, 1/2, 5/5],
    "W": [9/9, 0/2, 1/3, 0,  8/11, 2/2, 5/5],
    "L": [4/9, 0/2, 0/3, 0,  9/11, 0/2, 0/5],
    "I": [4/9, 0/2, 0/3, 0,  9/11, 0/2, 0/5],
    "V": [3/9, 0/2, 0/3, 0,  7/11, 0/2, 0/5],
    "S": [1/9, 1/2, 0/3, 0,  3/11, 0/2, 2/5],
    "T": [2/9, 1/2, 0/3, 0,  4/11, 0/2, 2/5],
    "P": [4/9, 0/2, 1/3, 0,  9/11, 0/2, 0/5],  # does have a lone pair but its tertiary -> prob not involved in resonance and NH3 is already in + state
    "VOID": [-1/9, -1/2, -1/3, -1, -1/11, -1/2, -1/5],
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
    'VOID': [-1, -1, -1, -1],
}

# Analysis of Domain-Swapped Oligomers Reveals Local Sequence Preferences and Structural Imprints at the Linker Regions and Swapped Interfaces - Scientific Figure on ResearchGate.
# Available from: https://www.researchgate.net/figure/Propensities-of-amino-acids-to-be-in-alpha-helix-beta-sheets-loops-and-hinge_fig12_230589968 [accessed 1 Jun 2026]