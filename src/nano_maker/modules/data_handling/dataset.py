import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import torch
from torch.utils.data import Dataset

from torch.utils.data import Dataset

class RadialDataset(Dataset):

    def __init__(self, pointer, smiles_molfp, pdb_radial,
                 block_size, max_angstroms):
        self.pointer = pointer
        self.smiles_molfp = smiles_molfp.set_index("smiles_str")       # <class 'torch.Tensor'>
        self.pdb_radial = pdb_radial.set_index("PDB_ID")               # <class 'list'>
        # set index moves "column label" column to row label, enabling O(1) .loc["target_ID"] lookups

        self.block_size = block_size
        self.max_angstroms = max_angstroms

    def __len__(self):
        return len(self.pointer)


    def __getitem__(self, idx):
        row = self.pointer.iloc[idx]
        smiles = row["SMILES"]
        pdb_id = row["PDB_HIT"]
        target_idx = row["WINDOW_IDX"]

        # context query
        molecular_fingerprint = self.smiles_molfp.loc[smiles, "map4_fp"]

        # radial sequence query
        radial_sequence = self.pdb_radial.loc[pdb_id, "radial_sequence"]

        radial_X, radial_Y = self.radial_XY(radial_sequence, target_idx)

        return (molecular_fingerprint,
                torch.tensor(radial_X, dtype=torch.float32),     # block_size, 3
                torch.tensor(radial_Y, dtype=torch.float32))     # 3,


    def radial_XY(self, radial_seq, target_idx):
        """
        Generates X and Y set for a given radial sequence + molecular fingerprint in the background?
        """
        r_pad = int(self.max_angstroms * 1.5)  # max range + 1
        az_pad = 0
        pl_pad = 0
        context = [[r_pad, az_pad, pl_pad] for _ in range(self.block_size)]
        for idx in range(len(radial_seq)):
            coords = radial_seq[idx][1]  # = [X, Y, Z]
            if idx == target_idx:
                return context, coords  # This is our XY data
            context = context[1:]+[coords]  # sliding window append -> context style in nanogpt