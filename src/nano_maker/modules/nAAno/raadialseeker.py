import numpy as np
import torch

class RAAdialSeeker:

    def __init__(self, resolution, verbose=False):
        self.resolution = resolution
        self.verbose = verbose

        self.angstrom_lim = 33   # maximum angstrom range found + some extra for context enrichment
                                 # can always edit this later on
        self.angstrom_inc = float(33 / resolution)
        self.threshold = float(1 / resolution)  # standardize how we determine radial sequences
        #                           smallest distance is the base increment, not 0
        self.radius_levels = torch.arange(self.angstrom_inc, self.angstrom_lim + self.angstrom_inc, step=self.angstrom_inc)

        self.coord_layer = None # [1, 3] -> 3D vector

    def init_spAAtial(self):
        self._make_coord_layer()

    def _make_coord_layer(self):
        self.coord_layer = torch.randn(size=(1, 3))

    def radial_sequence(self, aa_seq, vect_seq):
        radial_seq = {}  # lookup table
        seen = []

        # sanity
        if len(aa_seq) != len(vect_seq):
            raise ValueError(f"string and vector sequences of {aa_seq} are different lengths")

        # iterate up resolution increments, if a molecule's coordinate is within like 1/resolution of an angstrom, append it's info
        # its radius is now the unique ID, so if we stumble on it again its not duplicated
        for level in self.radius_levels:
            radial_seq[level] = []
            for i in range(len(aa_seq)):
                num_id = (np.sqrt(sum(vect_seq[i] ** 2)), i)
                if self._dist_check(vect_seq[i], level) and (num_id not in seen):
                    radial_seq[level].append([aa_seq[i], vect_seq[i]])
                    seen.append(num_id)

        # create two VOID tokens (0,0,0) on both sides to denote stops and starts
        return [0, 0, 0] + radial_seq[::-1] + [0, 0, 0]  # we want to go from outside inward

    def _dist_check(self, vector, ang_radius):
        if abs(ang_radius - np.sqrt(sum(vector ** 2))) <= self.threshold:
            return True
        else:
            return False

def test_resolution():
    module = RAAdialSeeker(resolution = 100)
    module.init_spAAtial()
    for level in module.radius_levels:
        print(level)

test_resolution()