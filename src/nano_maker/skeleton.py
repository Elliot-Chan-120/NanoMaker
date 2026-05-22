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


# RadialSeeker parameters used -> record these - might design a config file later
radial_resolution = 100
intrashell_resolution = 100
max_angstroms = 33

batch_size = 64  # how many data X X Ys we want to pass at a time
block_size = 50  # coordinate context
max_iters = 500
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 3
dropout = 0.2
map4_res = 1024



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
            nn.Linear(512, n_embd),
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
        pos_emb = self.pos_emb(torch.arange(T))
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
        coord_context = torch.zeros(1, block_size, 3)
        coord_out = []

        for _ in range(max_steps):
            next_coord, _ = self.forward(coord_context, map4_enc)
            coord_out.append(next_coord)
            coord_context = torch.cat([coord_context[:, 1:, :], next_coord.unsqueeze(1)], dim=1)


        return coord_out