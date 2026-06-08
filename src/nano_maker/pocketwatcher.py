import torch
import matplotlib.pyplot as plt
from src.paths import *

class PocketWatcher:
    """visualizes ingested pocket data and can output biochemical and structural data (reports)"""
    def __init__(self):
        pass

    # pt1 -> get it to retrieve data either by passing the data or the file
    def ingest_file(self, pocket_filepath):
        pass

    def ingest_data(self, pocket_data):
        pass


    # pt2 -> choose between visualizing just the skeleton or the entire pocket
    def visualize_skeleton(self):
        pass

    def visualize_pocket(self):
        pass


    # pt3 -> perform statistical analysis on the amino acid pocket
    # geometry and biochemical analysis
    def pocket_report(self):
        pass

