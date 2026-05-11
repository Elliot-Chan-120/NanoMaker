import torch

# global functions to help out with index encoding and index-to 3D vector decoding

def vect2idx(inc, vector):
    """Converts a torch.tensor of 3D vectors into their nearest shell-resolution cell index"""
    return torch.round(vector / inc).long()

def idx2vect(inc, idxs):
    """Converts a torch.tensor of indexes back into their original 3D vectors"""
    return idxs * inc


class SHELL:
    """Will hold our probability distributions for each individual 3D vector for that shell: so X, Y, and Z individually"""
    def __init__(self, radius, shell_resolution, smooth, max_angstroms):
        self.radius = radius
        self.shell_resolution = shell_resolution + 1  # one more slot for the 0
        # i guess the 0 will be at the bottom: wrong its at the top
        self.smooth = smooth

        # EACH AXIS IS STRUCTURED AS SUCH [0, max_angstroms ....... -max_angstroms]
        self.x_axis = torch.zeros(shell_resolution)
        self.y_axis = torch.zeros(shell_resolution)
        self.z_axis = torch.zeros(shell_resolution)

        self.max_angstroms = max_angstroms
        self.shell_increment = max_angstroms / shell_resolution

    def log_hit(self, vector):
        vector = torch.tensor(vector, dtype=torch.float32)
        x_ix, y_ix, z_ix = vect2idx(self.shell_increment, vector)

        self.x_axis[x_ix] += 1
        self.y_axis[y_ix] += 1
        self.z_axis[z_ix] += 1

    def end_training(self):
        self.x_axis = self.normalize(self.x_axis)
        self.x_axis = self.normalize(self.y_axis)
        self.x_axis = self.normalize(self.z_axis)

    def normalize(self, t):
        t = t + self.smooth
        t  = t / t.sum()
        return t