import torch
import plotly.express as px
import pandas as pd
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


    def visualize_pocket(self, identifier="centroid"):
        """
        :param identifier: color codes AAs by properties
        \n default setting = centroid -> amino acid vs ligand centroid
        \n net_charge
        \n isoelectric
        \n hydrophobicity
        \n net_size
        \n aromatic
        \n net_flexibility
        :return:
        """
        raw_coords = self._nanopkt_data["3D_skeleton"]
        aa_sequence = self._nanopkt_data["aa_sequence"]
        # convert to dataframe
        coordinate_dataframe = [{  # centroid entry
                'x_A': 0,
                'y_A': 0,
                'z_A': 0,
                'ID': 'VOID',
                'centroid': True,
            }]
        for idx in range(len(raw_coords)):
            coord = raw_coords[idx]
            aa_id = aa_sequence[idx]
            coordinate_dataframe.append({
                'x_A': coord[0],
                'y_A': coord[1],
                'z_A': coord[2],
                'ID': aa_id,
                'centroid': False,
            })

        c_df = pd.DataFrame(coordinate_dataframe)

        fig = px.scatter_3d(c_df, x='x_A', y='y_A', z='z_A', color=identifier, hover_data='ID')
        fig.show()



    # pt3 -> perform statistical analysis on the amino acid pocket
    # geometry and biochemical analysis
    def pocket_report(self):
        pass

