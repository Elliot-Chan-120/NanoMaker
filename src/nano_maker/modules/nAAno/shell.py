import torch

class Shell:
    """
    Basically a geometric tokenizer
    Performs coordinate discretization for any given 3D vector
    """
    def __init__(self, shell_resolution, max_angstroms, smooth):
        self.shell_resolution = shell_resolution + 1  # one more slot for the 0
        self.max_angstroms = max_angstroms  # make sure this is the same in RAAdialSeeker
        self.shell_increment = max_angstroms / shell_resolution
        self.smooth = smooth

        # EACH AXIS IS STRUCTURED AS SUCH [0, max_angstroms ....... -max_angstroms]
        self.x_axis = torch.zeros(shell_resolution)
        self.y_axis = torch.zeros(shell_resolution)
        self.z_axis = torch.zeros(shell_resolution)


    def vect2idx(self, vector):
        """
        Converts a torch.tensor of 3D vectors into their nearest shell-resolution's index
        """
        idxs = torch.round(vector / self.shell_increment)
        idxs = torch.clamp(idxs, -self.shell_resolution, self.shell_resolution).long()
        return idxs

    def idx2vect(self, idxs):
        """Converts a torch.tensor of indexes back into their original 3D vectors"""
        return idxs * self.shell_increment
