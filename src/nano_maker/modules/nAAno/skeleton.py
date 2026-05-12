from src.paths import *
from raadialseeker import *
import ast
import re

import pandas as pd

datapath = PROJECT_ROOT / "nano_notebooks" / "notebook_database" / "AA3D_df.csv"

radial_module = RadialSeeker(radial_resolution=100, intrashell_resolution=100, max_angstroms=33)
