import numpy as np
import torch

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
                    radial_seq.append([aa_seq[i], idx_vect])
                    seen.add(num_id)

        # create VOID token (0,0,0) at the end of the sequence to denote a stop
        return radial_seq + [[['VOID'], [0, 0, 0]]]  # we want to go from outside inward

    def _dist_check(self, ang_radius, level):
        if abs(ang_radius - level) <= self.threshold:
            return True
        else:
            return False

    @staticmethod
    def xyz_to_radial(xyz_vector):
        """
        ENCODE: Converts a tensor of 3D vectors into spherical coordinate vector
        """
        x, y, z = xyz_vector
        radius = np.linalg.norm(xyz_vector)
        azimuthal = np.arctan2(y, x)
        polar = np.arccos(z / radius)
        return [radius.item(), azimuthal.item(), polar.item()]

    @staticmethod
    def radial_to_xyz(spherical_vector):
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


