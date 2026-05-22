# goal: prototype skeleton transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from map4 import MAP4
import re

import matplotlib.pyplot as plt

from pathlib import Path

from modules.nAAno.radialseeker import RadialSeeker
from modules.nano_printer.smiles_handler import smiles_fingerprint
from src.database.dataset import RadialDataset


# radial seeker module params
radial_resolution = 100
intrashell_resolution = 100
max_angstroms = 33

# model params
block_size = 50
batch_size = 500
dropout=0.1
n_embd = 256
n_head = 4
map4_res = 1024
learning_rate = 1e-3
n_epochs = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(311104)


# Load data
# print(SMILE_2_PDBHITS.columns)  # 'SMILES', 'PDB_hits'
# print(MOLECULAR_FINGERPRINTS.columns)  # 'smiles_str', 'map4_fp'
# print(RADIAL_SEQUENCES.columns)  # 'PDB_ID', 'radial_sequence'

from src.paths import *
RADIAL_SEQUENCES = pd.read_pickle(DATABASE / "radial_seq_df.pkl")
MOLECULAR_FINGERPRINTS = pd.read_pickle(DATABASE / "molfp_df.pkl")
TRAIN_POINTER = pd.read_parquet(DATABASE / "training_pointers.parquet")
TEST_POINTER = pd.read_parquet(DATABASE / "test_pointers.parquet")

# training dataset
training_dataset = RadialDataset(pointer=TRAIN_POINTER,
                                 smiles_molfp=MOLECULAR_FINGERPRINTS,
                                 pdb_radial=RADIAL_SEQUENCES,
                                 block_size=block_size)
from torch.utils.data import DataLoader
loader = DataLoader(
    training_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

# test / validation set
test_set = RadialDataset(pointer=TEST_POINTER,
                                 smiles_molfp=MOLECULAR_FINGERPRINTS,
                                 pdb_radial=RADIAL_SEQUENCES,
                                 block_size=block_size)
from torch.utils.data import DataLoader
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    num_workers=4,
    shuffle=False,
    drop_last=True
)


class NewGELU(nn.Module):  # might be worth looking into for this

    def forward(self, x):
        return 0.5, * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # compute affinities
        attn_wts = q @ k.transpose(-2, -1) * C**-0.5
        attn_wts = F.softmax(attn_wts, dim=-1)
        attn_wts = self.dropout(attn_wts)

        # weighed aggregation
        values = self.value(x)
        output = attn_wts @ values
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(n_head)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out


class CrossAttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, coordinate_emb, molecular_emb):
        q = self.query(coordinate_emb)
        k = self.key(molecular_emb)
        v = self.value(molecular_emb)

        head_size = q.shape[-1]

        c_attn = q @ k.transpose(-2, -1) * head_size**-0.5
        c_attn = F.softmax(c_attn, dim=-1)
        c_attn = self.dropout(c_attn)

        return c_attn @ v


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.2):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([CrossAttentionHead(n_embd,
                                                       head_size,
                                                       ) for _ in range(n_head)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, coord_emb, mol_emb):
        out = torch.cat([h(coord_emb, mol_emb) for h in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out


class Stack(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.satt_head = MultiHeadAttention(n_embd, n_head)
        self.ffwd = FeedForwardNet(n_embd)
        self.catt_head = MultiHeadCrossAttention(n_embd, n_head)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x, mol_emb):
        x = x + self.satt_head(self.ln1(x))
        x = x + self.catt_head(self.ln2(x), mol_emb)
        x = x + self.ffwd(self.ln3(x))
        return x


class FeedForwardNet(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SkeletonModel(nn.Module):
    def __init__(self, n_embd, n_head, block_size, map4_res, radial_resolution):
        super().__init__()
        self.block_size = block_size
        self.map4_res = map4_res
        self.radial_resolution = radial_resolution

        self.c_project = nn.Linear(3, n_embd)  # feed coordinates here
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.mol_encoder = nn.Sequential(
            nn.Linear(map4_res, int(map4_res//2)),
            nn.ReLU(),
            nn.Linear(int(map4_res//2), n_embd),
            nn.LayerNorm(n_embd)
        )


        self.stack = Stack(n_embd, n_head)

        # OUTPUT HEAD -> outputs coordinates
        self.c_head = nn.Linear(n_embd, 3)


    def forward(self, coord_hist, map4_enc, targets=None):
        """
        coord_hist will be [n_batch, block_size, 3]
        targets is [n_batch, 1, 3]
        map4_enc -> unsure how I'm going to encode this
        """
        B, T, C = coord_hist.shape
        coordinate_emb = self.c_project(coord_hist.float() / self.radial_resolution)
        pos_emb = self.pos_emb(torch.arange(T, device=coord_hist.device))
        x = coordinate_emb + pos_emb

        mol_emb = self.mol_encoder(map4_enc.float()).unsqueeze(1)

        x = self.stack(x, mol_emb)

        output_coords = self.c_head(x[:, -1, :])

        loss = None
        if targets is not None:
            loss = F.mse_loss(output_coords, (targets.float() / self.radial_resolution))

        return output_coords, loss

    def generate(self, map4_enc, max_steps=130):
        # largest protein pocket in dataset was 107
        map4_enc = map4_enc.to(device)
        coord_context = torch.zeros(1, block_size, 3, device=map4_enc.device)
        coord_out = []

        for _ in range(max_steps):
            next_coord, _ = self.forward(coord_context, map4_enc)
            coord_out.append(next_coord)
            coord_context = torch.cat([coord_context[:, 1:, :], next_coord.unsqueeze(1)], dim=1)


        return coord_out



# =====================================================================================================================
model = SkeletonModel(n_embd=n_embd, n_head=n_head, block_size=block_size,
                      map4_res=map4_res, radial_resolution=radial_resolution).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss(model, loader, device, radial_res, max_batches=None):
    model.eval()
    total_loss = 0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        map4_fp, radial_X, radial_Y = batch

        map4_fp = map4_fp.to(device)
        radial_X = radial_X.to(device)
        radial_Y = radial_Y.to(device)

        _, loss = model(radial_X, map4_fp, radial_Y)
        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / n_batches


def train(n_epochs):
    for epoch in range(n_epochs):  # start with few epochs
        total_loss = 0
        for batch_idx, batch in enumerate(loader):
            map4_fp, radial_X, radial_Y = batch

            map4_fp = map4_fp.to(device)
            radial_X = radial_X.to(device)
            radial_Y = radial_Y.to(device)

            optimizer.zero_grad()
            pred, loss = model(radial_X, map4_fp, radial_Y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch + 1} | Batch {batch_idx + 1} | Loss: {loss.item():.6f}")

        train_loss = total_loss / len(loader)
        val_loss = estimate_loss(model, test_loader, device, radial_res=radial_resolution, max_batches=batch_size)

        print(f"Epoch {epoch + 1} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Gap: {val_loss - train_loss:.6f}")

    return model


def generate(smiles):
    map4_tensor = torch.tensor(smiles_fingerprint(smiles), dtype=torch.float32).unsqueeze(0)
    coordinates = model.generate(map4_tensor)
    return coordinates


train(n_epochs=n_epochs)