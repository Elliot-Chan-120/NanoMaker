import torch
from torch.utils.data import Dataset

class RadialDataset(Dataset):

    def __init__(self, test_pointer, train_pointer, smiles_molfp, pdb_radial,
                 block_size, mode='train'):
        self.test_pointer = test_pointer
        self.train_pointer = train_pointer                             # when extracted:
        self.smiles_molfp = smiles_molfp.set_index("smiles_str")       # <class 'torch.Tensor'>
        self.pdb_radial = pdb_radial.set_index("PDB_ID")               # <class 'list'>
        # set index moves "column label" column to row label, enabling O(1) .loc["target_ID"] lookups

        self.block_size = block_size
        self.mode = mode if mode =='train' else 'test'


    def __len__(self):
        return len(self.train_pointer) if self.mode == 'train' else len(self.test_pointer)


    def __getitem__(self, idx):
        row = self.train_pointer.iloc[idx] if self.mode == 'train' else self.test_pointer.iloc[idx]
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
        context = [[0, 0, 0] for _ in range(self.block_size)]
        for idx in range(len(radial_seq)):
            coords = radial_seq[idx][1]  # = [X, Y, Z]
            if idx == target_idx:
                return context, coords  # This is our XY data
            context = context[1:]+[coords]  # sliding window append -> context style in nanogpt

        raise ValueError(f"target idx out of range for {radial_seq}")