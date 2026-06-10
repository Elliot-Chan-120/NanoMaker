import torch
import matplotlib.pyplot as plt

from src.paths import *
from src.nano_maker.utility import *

class PocketWatcher:
    """visualizes ingested pocket data and can output biochemical and structural data (reports)"""
    def __init__(self):
        self._nanopkt_data = None


    # pt1 -> get it to retrieve data either by passing the data or the file
    def ingest_file(self, pocket_filepath):
        self._nanopkt_data = load_nano_pocket(pocket_filepath)
        return True

    def ingest_data(self, pocket_data):
        #TODO: data sanity checks before setting pocket_data to class variable
        self._nanopkt_data = pocket_data


    # pt2 -> choose between visualizing just the skeleton or the entire pocket
    def visualize_skeleton(self):
        pass

    def visualize_pocket(self):
        pass


    # pt3 -> perform statistical analysis on the amino acid pocket
    # geometry and biochemical analysis
    def pocket_report(self):
        pass

