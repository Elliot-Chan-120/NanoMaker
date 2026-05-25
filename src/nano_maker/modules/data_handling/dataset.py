import torch
from torch.utils.data import Dataset

import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import torch
from torch.utils.data import Dataset

class RadialDataset(Dataset):

    def __init__(self, pointer_path, smiles_molfp, pdb_radial,
                 block_size, max_angstroms):
        self.pointer = pq.ParquetFile(pointer_path)
        self.table = pq.read_table(pointer_path, memory_map=True)

        self.smiles_col = self.table.column("SMILES").combine_chunks()
        self.pdb_col = self.table.column("PDB_HIT").combine_chunks()
        self.window_col = self.table.column("WINDOW_IDX").combine_chunks()

        self.smiles_molfp = smiles_molfp  # <class 'torch.Tensor'>
        self.pdb_radial = pdb_radial      # <class 'list'>
        # set index moves "column label" column to row label, enabling O(1) .loc["target_ID"] lookups

        self.block_size = block_size
        self.max_angstroms = max_angstroms

    def __len__(self):
        return len(self.table)


    def __getitem__(self, idx):
        smiles = self.smiles_col[idx].as_py()
        pdb_id = self.pdb_col[idx].as_py()
        target_idx = self.window_col[idx].as_py()

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
        o_num = int(self.max_angstroms * 1.5)  # max range + 1
        context = [[o_num, o_num, o_num] for _ in range(self.block_size)]
        for idx in range(len(radial_seq)):
            coords = radial_seq[idx][1]  # = [X, Y, Z]
            if idx == target_idx:
                return context, coords  # This is our XY data
            context = context[1:]+[coords]  # sliding window append -> context style in nanogpt

        raise ValueError(f"target idx out of range for {radial_seq}")