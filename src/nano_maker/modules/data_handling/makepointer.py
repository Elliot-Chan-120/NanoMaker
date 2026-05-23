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
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class MakePointer:

    def __init__(self, smilespdb, pdb_radial, smiles_molfp,
                 train_pct, seed):
        smilespdb["PDB_hits"] = smilespdb["PDB_hits"].apply(ast.literal_eval)
        self.smilespdb = smilespdb
        self.radial_data = pdb_radial
        self.molfp_data = smiles_molfp

        if not 0 < train_pct < 1:
            raise ValueError(f"Train % split must be between 0.00 and 1.00: value submitted {train_pct}")
        self.train_pct = train_pct
        self.seed = seed

        self.train_rows = None
        self.test_rows = None


# notes on data columns
# print(SMILE_2_PDBHITS.columns)  # 'SMILES', 'PDB_hits'
# print(MOLECULAR_FINGERPRINTS.columns)  # 'smiles_str', 'map4_fp'
# print(RADIAL_SEQUENCES.columns)  # 'PDB_ID', 'radial_sequence'

    def make_pointerfile(self, out_folder):
        rng = np.random.default_rng(self.seed)

        all_smiles = self.smilespdb["SMILES"].tolist()
        rng.shuffle(all_smiles)

        n_train = int(len(all_smiles) * self.train_pct)
        train_set = set(all_smiles[:n_train])
        test_set = set(all_smiles[n_train:])  # unseen smiles -> ZeroBind methodology

        train_rows, test_rows = [], []

        for _, row in tqdm(self.smilespdb.iterrows(), total=len(self.smilespdb), desc="Building Pointer Files"):
            smiles = row["SMILES"]
            pdb_hits = row["PDB_hits"]
            bucket = train_rows if smiles in train_set else test_rows

            for pdb_id in pdb_hits:
                seq_data = self.radial_data.loc[self.radial_data["PDB_ID"]==pdb_id]
                if not len(seq_data):
                    continue
                seq_len = len(seq_data["radial_sequence"].values[0])
                n_windows = seq_len

                for window_idx in range(n_windows):
                    bucket.append({
                        "SMILES": smiles,
                        "PDB_HIT": pdb_id,
                        "WINDOW_IDX": window_idx,
                    })


        self.train_rows = pd.DataFrame(train_rows)
        self.test_rows = pd.DataFrame(test_rows)

        out_path = Path(str(out_folder))
        self.train_rows.to_parquet(out_path / "training_pointers.parquet", index=False)
        self.test_rows.to_parquet(out_path / "test_pointers.parquet", index=False)

        return True

# TODO: function to sample some molecular fingerprints and calculate tanimoto similarity (1000 at a time?)
