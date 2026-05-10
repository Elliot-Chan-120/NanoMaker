import numpy as np
import torch
import random
import torch.nn.functional as F
import ast

class SpAAtialEnc:

    def __init__(self, resolution):
        self.resolution = resolution
        self.angstrom_lim = 8  # this is the spatial limit we used during AA pocket data mining
        self.angstrom_inc = float(8 / resolution)

        self.hit_layer = None  # [resolution, 2] -> on and off
        self.coord_layer = None # [1, 3] -> 3D vector

    def init_spAAtial(self):
        self.make_hit_layer()
        self.make_coord_layer()

    def make_hit_layer(self):
        self.hit_layer = torch.rand(size=(self.resolution, 2))    # literally just a tensor with a hit or no hit cell

    def make_coord_layer(self):
        self.coord_layer = torch.randn(size=(1, 3))

    def radial_sequence(self):
        pass




# utility helpers for this module
def clean(raw_pocket_AA_data):
    tmp_dict = eval(raw_pocket_AA_data, {"array": np.array})

    AA_IDs = tmp_dict["AA_IDs"]
    AA_3Ds = [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in tmp_dict["3d_struct"]]

    return {
        "AA_IDs": AA_IDs,
        "AA_3Ds": AA_3Ds,
    }


def cleaner_test():
    sample = ("{'AA_IDs': ['E', 'K', 'I', 'G', 'E', 'G', 'T', 'G', 'V', 'V', 'Y', 'K', 'V', 'V', 'A',"
              " 'L', 'K', 'V', 'K', 'L', 'V', 'F', 'E', 'F', 'L', 'H', 'Q', 'D', 'L', 'K', 'K', 'K', 'P',"
              " 'Q', 'N', 'L', 'L', 'I', 'K', 'L', 'A', 'D', 'L', 'E'], "
              "'3d_struct': [array([ -1.41055556,  -5.20638964, -10.3391666 ]), array([-2.91055553, "
              "-6.43038866, -7.08816667]), array([-0.54055558, -4.75238917, -4.62616642]), array([-1.58455555,"
              " -7.00138781, -1.71916684]), array([-4.40555565, -7.52238962,  0.8278331 ]), array([-4.99455564, "
              "-4.7833893 ,  3.35083381]), array([-7.31355564, -4.91838953,  6.34983341]), array([-9.84455578, "
              "-5.79238817,  1.03683368]), array([-7.53355543, -6.13238833, -1.91016682]), array([-5.25055544, "
              "-3.12938807, -2.61016663]), array([-4.4865555 , -2.53838846, -6.2571667 ]), array([-2.01055555, "
              "-0.45138857, -8.16116663]), array([ -0.94755556,   6.0746105 , -10.76916655]), array([-2.91355555,  "
              "5.4476116 , -7.59116658]), array([-3.44855563,  2.69861105, -5.02816661]), array([-7.06655543,  "
              "1.58561209, -5.04616674]), array([-8.55455534, -0.103389  , -2.03016671]), array([-1.13155556,  "
              "9.12460974,  2.60583297]), array([-3.54555552, 11.34461096,  0.68383304]), array([-10.87755577,   "
              "4.37461164,  -1.78216656]), array([-8.35655539,  6.07560995, -4.08416661]), array([-5.12855546,  "
              "7.02661016, -2.40616663]), array([-1.83555554,  8.54461172, -3.40316685]), "
              "array([ 0.5414445 ,  5.79661062, -4.52016648]), array([ 3.77744444,  5.4156101 , -2.56816681]), "
              "array([ 6.62944467,  3.18561056, -3.65616663]), array([ 7.25144489,  0.78361204, -0.87116679]), "
              "array([ 5.7554446 , -0.17338869,  2.44783298]), array([ 7.81544407, -0.27338907,  5.63083355]), "
              "array([ 7.49444397, -4.04038927,  6.14383308]), array([ 9.00344475, -4.67638895,  2.71083347]), "
              "array([ 1.41244443, -1.46738932, 11.5618333 ]), array([ 4.47944458, -1.6223881 ,  9.22883312]), "
              "array([ 2.30044444, -2.5353882 ,  6.24283306]), array([0.39744444, 0.78261068, 6.52083294]), "
              "array([3.52444465, 2.97561147, 6.78883354]), array([4.59944446, 3.91161039, 3.26183311]), "
              "array([7.90344436, 5.28460958, 2.0788335 ]), array([3.52044447, 8.44360998, 5.47883312]), "
              "array([0.66044445, 6.73861196, 7.30383388]), array([-1.97555558,  4.91961172,  5.2038335 ]), "
              "array([-5.52655546,  4.09361151,  6.5078334 ]), array([-10.66355602,   4.12361219,   8.61583416]), "
              "array([ -5.99055569, -12.91838858,   6.67283336])]}")
    print(clean(sample))
# cleaner_test()  # need to just make sure it outputs stuff as intended

def test_resolution():
    spatial_module = SpAAtialEnc(resolution = 1000)
    spatial_module.make_hit_layer()
    print(spatial_module.hit_layer)
# test_resolution()  # good