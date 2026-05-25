import numpy as np
import torch

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

# save this
from naanolibrary import *

# protein feature engineering:
# spatial data + physicochemical data
# when you pass an AA seq through ESM2, you get back a matrix of shape:
# [sequence length x embedding dimension] - big problem
# each row is a per-residue embedding, a vector that encodes that AA's...
# - identity, local chemical environment, inferred structural context as a vector
# add parts of GEM to append a physiochemical vector on top of spatial one
# - charge, hydrophobicity, pKa, H-bond capacity, .etc
# GEM properties = nAAno_token

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

class RadialSeeker:

    def __init__(self, radial_resolution, max_angstroms,
                 verbose=False):
        self.radial_resolution = radial_resolution  # how fine we want the order to be
        self.angstrom_lim = max_angstroms  # maximum angstrom range found + some extra for context enrichment
        # can always edit this later on
        self.angstrom_inc = float(max_angstroms / radial_resolution)
        self.threshold = self.angstrom_inc / 2  # standardize how we determine radial sequences
        # getting half the angstrom increment will let us finely separate them

        #                           smallest distance is the base increment, not 0
        self.radius_levels = torch.arange(self.angstrom_lim, self.angstrom_inc, step=-1 * self.angstrom_inc)

        self.verbose = verbose

        self.nAAno_mod = NAAnoEng(verbose=False)
        self.nAAno_mod.initialize()

    def radial_sequence(self, aa_seq, vect_seq):
        # first order them by absolute distance to centroid
        # radial resolution determines how finely we want to order them

        radial_seq = []  # lookup table
        seen = set()

        # sanity
        if len(aa_seq) != len(vect_seq):
            raise ValueError(f"string and vector sequences of {aa_seq} are different lengths")

        # iterate up resolution increments, if a molecule's coordinate is within like 1/resolution of an angstrom, append it's info
        # its (radius, pos index) is now the unique ID, so if we stumble on it again its not duplicated
        for level in self.radius_levels:  # go through all possible radius levels
            for i in range(len(aa_seq)):  # go through all amino acids in that sequence
                num_id = i
                if num_id not in seen and self._dist_check(np.linalg.norm(vect_seq[i]), level):
                    idx_vect = self.xyz_to_radial(vect_seq[i])
                    radial_seq.append([self.nAAno_mod.get_nAAnovector(aa_seq[i]), idx_vect])
                    seen.add(num_id)

        # create VOID token (0,0,0) at the end of the sequence to denote a stop
        return radial_seq + [[['VOID'], [0, 0, 0]]]  # we want to go from outside inward

    def _dist_check(self, ang_radius, level):
        if abs(ang_radius - level) <= self.threshold:
            return True
        else:
            return False

    def xyz_to_radial(self, xyz_vector):
        """
        ENCODE: Converts a tensor of 3D vectors into spherical coordinate vector
        """
        x, y, z = xyz_vector
        radius = np.linalg.norm(xyz_vector)
        azimuthal = np.arctan2(y, x)
        polar = np.arccos(z / radius)
        return [radius.item(), azimuthal.item(), polar.item()]

    def radial_to_xyz(self, spherical_vector):
        """DECODE: Converts a tensor of spherical coordinates into raw angstrom values"""
        radius, azimuthal, polar = spherical_vector
        x = radius * np.sin(polar) * np.cos(azimuthal)
        y = radius * np.sin(polar) * np.sin(azimuthal)
        z = radius * np.cos(polar)
        return [x.item(), y.item(), z.item()]


def test_resolution():
    module = RadialSeeker(radial_resolution=100, max_angstroms=33)
    for level in module.radius_levels:
        print(level)

    test_vector = [6.54, -9.2, 23.69]
    t_v2ix = module.xyz_to_radial(test_vector)
    t_ix4v = module.radial_to_xyz(t_v2ix)
    print(f"{test_vector} --Encode-- {t_v2ix}")
    print(f"{t_v2ix} --Decode-- {t_ix4v}")
# test_resolution()  # all good

# TODO: normalize to decimals later