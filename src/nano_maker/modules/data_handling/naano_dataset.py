from torch.utils.data import Dataset
import pyarrow.parquet as pq
import torch

from src.nano_maker.modules.nAAno.naanoeng import NAAnoEng

# note: all coordinates are now on the spherical system
# doesn't need to predict an "end" since that's covered by Skeleton
# all it has to do is predict the next valid suitable biochemical environment

# USE THIS FOR WEBSITE TRAINING

class NAAnoDataset(Dataset):
    def __init__(self, pointer_path, smiles_molfp, pdb_radial,
                 block_size, max_angstroms):
        self.naano_module = NAAnoEng(max_angstroms=max_angstroms,
                                     block_size=block_size,
                                     verbose=False)
        self.naano_module.initialize()

        self.pointer = pq.ParquetFile(pointer_path)
        self.table = pq.read_table(pointer_path, memory_map=True)

        self.smiles_col = self.table.column("SMILES").combine_chunks()
        self.pdb_col = self.table.column("PDB_HIT").combine_chunks()
        self.window_col = self.table.column("WINDOW_IDX").combine_chunks()

        self.smiles_molfp = smiles_molfp
        self.pdb_radial = pdb_radial

        self.block_size = block_size
        self.max_angstroms = max_angstroms

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        smiles = self.smiles_col[idx].as_py()
        pdb_id = self.pdb_col[idx].as_py()
        target_idx = self.window_col[idx].as_py()

        # context
        molecular_fingerprint = self.smiles_molfp.loc[smiles, "map4_fp"]
        radial_sequence = self.pdb_radial.loc[pdb_id, "radial_sequence"]
        nAAno_X, nAAno_Y = self.naano_XY(radial_sequence, target_idx)

        return (molecular_fingerprint,
                nAAno_X,  # we're already receiving it as a tensor
                torch.tensor(nAAno_Y, dtype=torch.float32))


    def naano_XY(self, radial_sequence, target_idx):
        """
        ok think slightly harder about this
        - we want the relative vector to the target coordinate
        - get context first, record target Y when found and then break
        - calculate relative coordinates and append to biochemical vectors after
        :param radial_sequence:
        :param target_idx:
        :return:
        """
        r_pad = int(self.max_angstroms)
        az_pad = 0
        pl_pad = 0

        coord_context = [[r_pad, az_pad, pl_pad] for _ in range(self.block_size)]
        bioch_context = [self.naano_module.get_nAAnovector("VOID") for _ in range(self.block_size)]

        coord_Y = None  # what we use to calculate the relative vectors
        naano_Y = None  # this is what our model needs to predict

        for idx in range(len(radial_sequence)):
            aa_id = radial_sequence[idx][0][0]
            nAAno_token = self.naano_module.get_nAAnovector(aa_id)  # biochemical vector
            coords = radial_sequence[idx][1]  # convert back to [x, y, z]

            if idx == target_idx:
                coord_Y = coords
                naano_Y = nAAno_token
            else:
                coord_context = coord_context[1:]+[coords]
                bioch_context = bioch_context[1:]+[nAAno_token]

        # create final context matrix
        naano_X = self.naano_module.get_nAAno_X(coord_context, bioch_context, coord_Y)

        return naano_X, naano_Y

    # 22 biochemical features + 7 augmented relative spatial features
    # 29 total -> 29 in embedding -> 22 linear head output