from src.paths import *
from shell import *
from raadialseeker import *
import ast
import re

import pandas as pd

datapath = PROJECT_ROOT / "nano_notebooks" / "notebook_database" / "AA3D_df.csv"


radial_module = RAAdialSeeker(resolution=100, verbose=True)
shell = Shell(shell_resolution=100, max_angstroms=33, smooth=1)

