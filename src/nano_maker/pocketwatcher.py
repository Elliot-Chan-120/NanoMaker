import torch
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.paths import *
from src.nano_maker.utility import *
from src.nano_maker.modules.nAAno.naanoeng import NAAnoEng

class PocketWatcher:
    """visualizes ingested pocket data and can output biochemical and structural data (reports)"""
    def __init__(self, naanoeng_config, pocket_config):
        self._nanopkt_data = None
        self._NAAnoEng_module = NAAnoEng(naanoeng_config["max_angstroms"], naanoeng_config["block_size"], verbose=False)
        self._NAAnoEng_module.initialize()
        self.summary_vectors = self._NAAnoEng_module.summary_vectors()

        self.pw_cfg = pocket_config



    # pt1 -> get it to retrieve data either by passing the data or the file
    def ingest_file(self, pocket_filepath):
        self._nanopkt_data = load_nano_pocket(pocket_filepath)
        return True

    def ingest_data(self, pocket_data):
        #TODO: data sanity checks before setting pocket_data to class variable
        self._nanopkt_data = pocket_data


    def visualize_pocket(self, identifier="skeleton"):
        """
        :param identifier: color codes AAs by properties
        \n default setting = skeleton -> amino acid vs ligand centroid but will still show the aa id
        \n charge_env
        \n hydrophobicity
        \n net_flex
        \n net_steric
        :return:
        """
        raw_coords = self._nanopkt_data["3D_skeleton"]
        aa_sequence = self._nanopkt_data["aa_sequence"]
        # convert to dataframe
        coordinate_dataframe = []

        for idx in range(len(raw_coords)):
            coord = raw_coords[idx]
            aa_id = aa_sequence[idx]
            aa_summary = self.summary_vectors[aa_id]
            coordinate_dataframe.append({
                'x_A': coord[0],
                'y_A': coord[1],
                'z_A': coord[2],
                'ID': aa_id,
                'skeleton': True,
                'charge_env': aa_summary['charge_env'],
                'hydrophobicity': aa_summary['hydrophobicity'],
                'net_flex': aa_summary['net_flex'],
                'net_steric': aa_summary['net_steric_profile']
            })

        c_df = pd.DataFrame(coordinate_dataframe)

        fig = px.scatter_3d(c_df, x='x_A', y='y_A', z='z_A',
                            color=identifier,
                            color_continuous_scale=self.pw_cfg['colorscales'][identifier],
                            hover_data='ID')

        fig.update_layout(
            paper_bgcolor=self.pw_cfg["paper_bgcolor"],
            scene=dict(
                bgcolor=self.pw_cfg["grid_bgcolor"],
                xaxis=dict(
                    gridcolor=self.pw_cfg["gridline_color"],
                    zerolinecolor=self.pw_cfg["gridline_color"],
                    backgroundcolor=self.pw_cfg["grid_bgcolor"],
                    color=self.pw_cfg["label_color"]
                ),
                yaxis=dict(
                    gridcolor=self.pw_cfg["gridline_color"],
                    zerolinecolor=self.pw_cfg["gridline_color"],
                    backgroundcolor=self.pw_cfg["grid_bgcolor"],
                    color=self.pw_cfg["label_color"]
                ),
                zaxis=dict(
                    gridcolor=self.pw_cfg["gridline_color"],
                    zerolinecolor=self.pw_cfg["gridline_color"],
                    backgroundcolor=self.pw_cfg["grid_bgcolor"],
                    color=self.pw_cfg["label_color"]
                ),
            ),
            legend=dict(
                x=0.01,
                y=0.99,
                xanchor='left',
                yanchor='top',
                font=dict(
                    size=12,
                    color=self.pw_cfg["label_color"],
                ),
            ),
            coloraxis_colorbar=dict(
                x=1.05,
                xanchor='left',
                tickfont=dict(color=self.pw_cfg["label_color"]),
                title=dict(font=dict(color=self.pw_cfg["label_color"])),
            ),
            legend_font_color=self.pw_cfg["label_color"],
            legend_title_font_color=self.pw_cfg["label_color"]
        )

        fig.add_trace(go.Scatter3d(
            x=[0.2],
            y=[0.2],
            z=[0.2],
            mode='markers',
            marker=dict(
                size=15,
                color='#8fbcbb',
            ),
            name='centroid',

        ))

        fig.show()



    # pt3 -> perform statistical analysis on the amino acid pocket
    # geometry and biochemical analysis
    def pocket_report(self):
        pass

