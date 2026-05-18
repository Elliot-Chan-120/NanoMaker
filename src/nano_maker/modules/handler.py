# task: review and allocate all data combinations within SMILES -> PDB
# 30k ish SMILES
# for each SMILES: many PDBs
# for each of those PDBs -> many context coordinates + Y
# need an allocator to generate training + test datasets for this task


# projected unpacking time = 41 minutes
# TODO: organize into train and test + batch processor

import torch
import re
import ast
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class Allocate:

    def __init__(self, smilespdb, pdb_radial, smiles_molfp,
                 train_pct, test_pct, block_size):
        smilespdb["PDB_hits"] = smilespdb["PDB_hits"].apply(ast.literal_eval)

        self.smilespdb = smilespdb
        self.radial_data = pdb_radial
        self.molfp_data = smiles_molfp

        self.train_pct = train_pct
        self.test_pct = test_pct
        self.block_size = block_size

        # these are going to be temporary -> holds lots of data
        self.expanded_pointer = []
        self.master_df = []

        self.total_pairs = None
        self.total_points = None


# notes on data columns
# print(SMILE_2_PDBHITS.columns)  # 'SMILES', 'PDB_hits'
# print(MOLECULAR_FINGERPRINTS.columns)  # 'smiles_str', 'map4_fp'
# print(RADIAL_SEQUENCES.columns)  # 'PDB_ID', 'radial_sequence'

    def expand_smilespdb(self):
        new = []
        for _, row in self.smilespdb.iterrows():
            SMILES = row['SMILES']
            for pdb_id in row['PDB_hits']:
                new.append({
                    "SMILES": SMILES,
                    "PDB_HIT": pdb_id
                })

        self.total_pairs = len(new)
        self.expanded_pointer = pd.DataFrame(new)
        return True


    def get_XXY(self):
        assert isinstance(self.expanded_pointer, pd.DataFrame), (f"Expected DataFrame, "
                                                                 f"make sure function expand_smilespdb is run")

        result_df = []

        for _, row in tqdm(self.expanded_pointer.iterrows(), total=len(self.expanded_pointer), desc="Creating Master DataFrame"):
            # get prelim data
            smiles_id = row["SMILES"]
            pdb_id = row["PDB_HIT"]

            # query for molecular fingerprint
            mol_hit = self.molfp_data.loc[self.molfp_data["smiles_str"]==smiles_id]
            molecular_fingerprint = mol_hit["map4_fp"]

            # query for pdb radial sequence
            pdb_hit = self.radial_data.loc[self.radial_data["PDB_ID"]==pdb_id]
            radial_sequence = pdb_hit["radial_sequence"]

            radial_X, radial_Y = self.radial_XY(radial_sequence)  # coordinate context + resulting coordinate

            for idx in range(len(radial_X)):
                result_df.append(
                    {
                        "molecular_fingerprint": molecular_fingerprint,     # context for cross-attention
                        "radial_X": radial_X[idx],                          # X - radial sequence context
                        "radial_Y": radial_Y[idx]                           # Y - next coordinate (Y vector)
                    }
                )

        result_df = pd.DataFrame(result_df)
        self.master_df = result_df

        return True


    def get_all_pairs(self):
        return self.total_pairs

    def get_all_points(self):
        return True

    def radial_XY(self, radial_seq):
        """
        Generates X and Y set for a given radial sequence + molecular fingerprint in the background?
        :param radial_seq:
        :return:
        """
        X, Y = [], []
        context = [[0, 0, 0]] * self.block_size
        for idx in radial_seq:
            coords = idx[1]  # = [X, Y, Z]
            X.append(context)
            Y.append(coords)
            context = context[1:] + [coords]

        return X, Y