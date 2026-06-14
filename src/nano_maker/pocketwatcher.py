import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.nano_maker.utility import *
from src.nano_maker.modules.nAAno.naanoeng import NAAnoEng

class PocketWatcher:
    """visualizes ingested pocket data and can output biochemical and structural data (reports)"""
    def __init__(self, naanoeng_config, pocket_config):
        self._nanopkt_data = None
        self._NAAnoEng_module = NAAnoEng(naanoeng_config["max_angstroms"], naanoeng_config["block_size"], verbose=False)
        self._NAAnoEng_module.initialize()
        self.summary_vectors = self._NAAnoEng_module.summary_vectors()
        self.outpath = POCKET_DATA

        self.pw_cfg = pocket_config



    # pt1 -> get it to retrieve data either by passing the data or the file
    def ingest_file(self, pocket_filepath):
        self._nanopkt_data = load_nano_pocket(pocket_filepath)
        return True

    def ingest_data(self, pocket_data):
        #TODO: data sanity checks before setting pocket_data to class variable
        self._nanopkt_data = pocket_data


    def visualize_pocket(self, identifier="skeleton", outfile_name=None):
        """
        :param identifier: color codes AAs by properties
        \n default setting = skeleton -> amino acid vs ligand centroid but will still show the aa id
        \n polar_character
        \n hydrophobicity
        \n flexibility
        \n steric_accessibility
        :param outfile_name:
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
                'skeleton': 1.0,
                'color_code': list(self.summary_vectors).index(aa_id),
                'polar_character': aa_summary['polar_character'],
                'hydrophobicity': aa_summary['hydrophobicity'],
                'flexibility': aa_summary['flexibility'],
                'steric_accessibility': aa_summary['steric_accessibility']
            })

        c_df = pd.DataFrame(coordinate_dataframe)

        fig = px.scatter_3d(c_df, x='x_A', y='y_A', z='z_A',
                            color=identifier,
                            color_continuous_scale=self.pw_cfg['colorscales'][identifier],
                            hover_data='ID')

        fig.update_traces(marker_size=5)

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
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode='markers',
            marker=dict(
                size=15,
                color='#8fbcbb',
                symbol='circle',
            ),
            name='centroid',

        ))

        # verified it works through cli
        if outfile_name:
            output_path = self.outpath / f"{outfile_name}_{identifier}.html"
            fig.write_html(output_path)
            print(f"{outfile_name} visualization saved in {output_path}")


        fig.show()


    # pt3 -> perform statistical analysis on the amino acid pocket
    # geometry and biochemical analysis
    def pocket_report(self, outfile_name=None):
        raw_coords = self._nanopkt_data["3D_skeleton"]
        aa_sequence = self._nanopkt_data["aa_sequence"]
        # proximal_threshold = self.pw_cfg["proximal_threshold"]

        biochemical_sum, biochemical_note = self.biochemical_summary(aa_sequence)
        coordinate_sum, receptor_style = self.coordinate_summary(raw_coords)

        separator = "=" * 50
        # now piece everything together in a small summary
        content = "BINDING POCKET REPORT\n"
        content += separator
        content += f"\nSampling temperature: {self._nanopkt_data["Sampling_temperature"]}"
        content += "\nAmino acid sequence:"
        for aa_id in aa_sequence:
            content += f"{aa_id}"

        content += "\n"
        content += separator
        content += "\n\nSection 1: Biochemical Property Summary\n"
        for summary, value in biochemical_sum.items():
            content += f"|-- {summary}: {value:.3f}\n"

        content += "\nSection 2: Geometric Analysis\n"
        content += f"|-- X range: {coordinate_sum["X_coverage"]:.3f}\n"
        content += f"|-- Y range: {coordinate_sum["Y_coverage"]:.3f}\n"
        content += f"|-- Z range: {coordinate_sum["Z_coverage"]:.3f}\n\n"

        content += separator
        content += "\n\nNotable Binding Pocket Characteristics:\n"
        content += f"- {biochemical_note}{receptor_style}"

        if outfile_name:
            output_path = self.outpath / f"{outfile_name}_report.txt"
            with open (output_path, 'w', encoding='utf-8') as file:
                file.write(content)
            print(f"{outfile_name} report saved at {output_path}")

        return content



    def biochemical_summary(self, aa_sequence):
        tmp_charge = []
        tmp_hydro = []
        tmp_flex = []
        tmp_steric = []

        for aa in aa_sequence:
            aa_summary = self.summary_vectors[aa]
            tmp_charge.append(aa_summary['polar_character'])
            tmp_hydro.append(aa_summary['hydrophobicity'])
            tmp_flex.append(aa_summary['flexibility'])
            tmp_steric.append(aa_summary['steric_accessibility'])

        length = len(aa_sequence)
        avg_charge = float(sum(tmp_charge) / length)
        avg_hydro = float(sum(tmp_hydro) / length)
        avg_flex = float(sum(tmp_flex) / length)
        avg_steric = float(sum(tmp_steric) / length)

        # measure how much this pocket deviates from a global average for each of the summary properties
        avg_vector = self.summary_vectors['AVG']
        charge_dev = float(((avg_charge - avg_vector['polar_character']) / avg_vector['polar_character']) * 100)
        hydrophobicity_dev = float(((avg_hydro - avg_vector['hydrophobicity']) / avg_vector['hydrophobicity']) * 100)
        flex_dev = float(((avg_flex - avg_vector['flexibility']) / avg_vector['flexibility']) * 100)
        steric_dev = float(((avg_steric - avg_vector['steric_accessibility']) / avg_vector['steric_accessibility']) * 100)

        notable_features = self.property_status_human_interpretable({
            "polar character": charge_dev,
            "hydrophobic": hydrophobicity_dev,
            "flexible": flex_dev,
            "sterically accessible": steric_dev,
        })

        summary = {
            'average_polar_character': avg_charge,
            'polar_character_deviation_pct': charge_dev,
            'average_hydrophobicity': avg_hydro,
            'hydrophobicity_deviation_pct': hydrophobicity_dev,
            'average_flexibility': avg_flex,
            'flexibility_deviation_pct': flex_dev,
            'average_steric': avg_steric,
            'steric_deviation_pct': steric_dev,
        }

        return summary, notable_features

    @staticmethod
    def property_status_human_interpretable(properties):
        notes = ""
        for bio_property, deviation in properties.items():
            label = f"{bio_property}"
            magnitude = abs(deviation)
            if magnitude < 10:
                continue
            if magnitude < 20:
                adj = "slightly"
            elif magnitude < 40:
                adj = "moderately"
            elif magnitude < 60:
                adj = "highly"
            elif magnitude < 80:
                adj = "significantly"
            else:
                adj = "extremely"
            # TODO: refine the groupings later -> pretty arbitrary at the moment.

            if deviation < 0 and magnitude > 10:
                if bio_property == "polar character":
                    label = "non-polar character"
                elif bio_property == "hydrophobic":
                    label = "hydrophilic"
                elif bio_property == "flexible":
                    label = "rigid"
                else:  # sterically accessible
                    label = "sterically inaccessible"

            notes += f"{adj} {label}, "

        return notes

    def coordinate_summary(self, xyz_skeleton):
        """
        :param xyz_skeleton:
        :return:
        """
        # so we have 3 dimensions -> x y z
        # for each -> get the minimum and max values
        # if min is -ve and max is +ve AND max - min >= 15 -> dimension = "BROAD"
        # if...
        # 3/3 dimensions = BROAD -> cage-style binding pocket
        # 2/3 = BROAD -> chamber-style
        # 1/3 = BROAD -> vice-style binding
        # 0/3 = BROAD -> surface patch-style
        broad_count = 0
        x_coverage, x_is_broad = self.broad_check([coords[0] for coords in xyz_skeleton])
        y_coverage, y_is_broad = self.broad_check([coords[1] for coords in xyz_skeleton])
        z_coverage, z_is_broad = self.broad_check([coords[2] for coords in xyz_skeleton])

        if x_is_broad:
            broad_count += 1
        if y_is_broad:
            broad_count += 1
        if z_is_broad:
            broad_count += 1


        # summary metrics + shape inference
        if broad_count == 3:
            receptor_style = "cage-style binding pocket"
        elif broad_count == 2:
            receptor_style = "chamber-style binding pocket"
        elif broad_count == 1:
            receptor_style = "vice-style binding pocket"
        else: # broad count is still 0...
            receptor_style = "surface patch-style binding pocket"


        summary = {
            'X_coverage': x_coverage,
            'Y_coverage': y_coverage,
            'Z_coverage': z_coverage,
        }

        return summary, receptor_style


    def broad_check(self, numbers):
        min_range = 0
        max_range = 0
        for number in numbers:
            min_range = min(min_range, number)
            max_range = max(max_range, number)

        dim_coverage = max_range + abs(min_range)

        if min_range < 0 < max_range and (dim_coverage >= self.pw_cfg["broad_threshold"]):
                    return dim_coverage, True
        return dim_coverage, False