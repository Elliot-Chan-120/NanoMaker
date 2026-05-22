import numpy as np
import torch

class RadialSeeker:

    def __init__(self, radial_resolution, intrashell_resolution, max_angstroms,
                 verbose=False):
        self.radial_resolution = radial_resolution
        self.angstrom_lim = max_angstroms  # maximum angstrom range found + some extra for context enrichment
        # can always edit this later on
        self.angstrom_inc = float(max_angstroms / radial_resolution)
        self.threshold = self.angstrom_inc / 2  # standardize how we determine radial sequences
        # getting half the angstrom increment will let us finely separate them

        #                           smallest distance is the base increment, not 0
        self.radius_levels = torch.arange(self.angstrom_lim, self.angstrom_inc, step=-1 * self.angstrom_inc)

        self.intrashell_resolution = intrashell_resolution
        self.intrashell_inc = max_angstroms / intrashell_resolution

        self.verbose = verbose

    def radial_sequence(self, aa_seq, vect_seq):
        # first order them by absolute distance to centroid
        # then convert them to indices according to intra-shell resolution

        # radial resolution determines how finely we want to order them
        # shell resolution determines how finely in 3D space we represent them

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
                    idx_vect = self.vect2idx(vect_seq[i])
                    radial_seq.append([[aa_seq[i]], idx_vect])
                    seen.add(num_id)

        # create two VOID tokens (0,0,0) on both sides to denote stops and starts
        return radial_seq + [[['VOID'], [0, 0, 0]]]  # we want to go from outside inward

    def _dist_check(self, dist, ang_radius):
        if abs(dist - ang_radius) <= self.threshold:
            return True
        else:
            return False

    def vect2idx(self, vector):
        """
        ENCODE: Converts a tensor of 3D vectors into their nearest shell-resolution's index
        """
        vector_a = np.array(vector) + self.angstrom_lim
        idxs = np.round(vector_a / (2 * self.intrashell_inc))
        idxs = np.clip(idxs, 0, self.intrashell_resolution)
        return idxs.astype(int).tolist()

    def num2vect(self, idxs):
        """DECODE: Converts a tensor of indexes into their max_Angstrom-scaled 3D vectors"""
        vector = []
        for number in idxs:
            vector.append(float(number) * 2 * self.intrashell_inc - self.angstrom_lim)
        return vector


def test_resolution():
    module = RadialSeeker(radial_resolution=100, intrashell_resolution=100, max_angstroms=33)
    for level in module.radius_levels:
        print(level)

    test_vector = [0.6, 1.2, 23.69]
    t_v2ix = module.vect2idx(test_vector)
    t_ix4v = module.num2vect(t_v2ix)
    print(f"{test_vector} --Encode-- {t_v2ix}")
    print(f"{t_v2ix} --Decode-- {t_ix4v}")
# test_resolution()  # all good

# TODO: normalize to decimals later