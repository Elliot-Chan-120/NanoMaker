import numpy as np
import torch
import random
import torch.nn.functional as F
import ast

class RAAdialSeeker:

    def __init__(self, resolution, verbose=False):
        self.resolution = resolution
        self.verbose = verbose

        self.angstrom_lim = 33   # maximum angstrom range found + some extra for context enrichment
                                 # can always edit this later on
        self.angstrom_inc = float(8 / resolution)
        self.threshold = float(1 / resolution)  # standardize how we determine radial sequences
        #                           smallest distance is the base increment, not 0
        self.radius_levels = torch.arange(self.angstrom_inc, self.angstrom_lim + self.angstrom_inc, step=self.angstrom_inc)

        self.hit_layer = None  # [resolution, 2] -> on and off
        self.coord_layer = None # [1, 3] -> 3D vector

    def init_spAAtial(self):
        self._make_hit_layer()
        self._make_coord_layer()

    def _make_hit_layer(self):
        self.hit_layer = torch.rand(size=(self.resolution, 2))    # literally just a tensor with a hit or no hit cell

    def _make_coord_layer(self):
        self.coord_layer = torch.randn(size=(1, 3))

    def radial_sequence(self, aa_seq, vect_seq):
        radial_seq = {}  # lookup table
        seen = []

        # sanity
        if len(aa_seq) != len(vect_seq):
            raise ValueError(f"string and vector sequences of {aa_seq} are different lengths")

        # iterate down resolution increments, if a molecule's coordinate is within like 1/resolution of an angstrom, append it's info
        # to that layer then remove it's index from that list
        for level in self.radius_levels:
            radial_seq[level] = []
            for i in range(len(aa_seq)):
                unique_radius = np.sqrt(sum(vect_seq[i] ** 2))
                if self._dist_check(vect_seq[i], level) and (unique_radius not in seen):
                    radial_seq[level].append([aa_seq[i], vect_seq[i]])
                    seen.append(unique_radius)
            if not radial_seq[level]:
                radial_seq[level].append(['VOID', [0,0,0]])

        return radial_seq

    def _dist_check(self, vector, ang_radius):
        if abs(ang_radius - np.sqrt(sum(vector ** 2))) <= self.threshold:
            return True
        else:
            return False

def test_resolution():
    module = RAAdialSeeker(resolution = 1000)
    module.init_spAAtial()
    print(module.hit_layer)
    for level in module.radius_levels:
        print(level)

test_resolution()