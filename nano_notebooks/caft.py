import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
from pandas.tseries.frequencies import key


# start off with one layer of everything
# 1 self-attention on spatial context, one cross-attention to SMILES, one FFN


# notes:
# Chunk MAP4 -> nn.Linear(1024, n_embd)?
# maybe figure out a way to only pass in the active parts

# (10, 3) coordinate history -> predict next [X, Y, Z]
# 10 coordinates of 3D tensors

class NewGELU(nn.Module):

    def forward(self, x):
        return 0.5, * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class SmileEncoder(nn.Module):

    def __init__(self):
        pass


class RadialEncoder(nn.Module):

    def __init__(self):
        pass


# SelfAttentionHead
# MultiHeadSelfAttention
# DecoderBlock
# EncoderBlock
# CrossAttentionHead
# MultiHeadCrossAttention
# CrossAttentionTransformer


class CrossAttention(nn.Module):

    def __init__(self, n_embd, n_head, dropout=0.2):
        super().__init__()
        self.coord_projection = nn.Linear(3, n_embd)

        self.key = nn.Linear(n_embd, n_head, bias=False)
        self.value = nn.Linear(n_embd, n_head, bias=False)
        self.query = nn.Linear(n_embd, n_head, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, coord_hist, mol_emb):
        """
        coord_hist: (Batch, block_size, 3)
        mol_emb: (Batch, idk yet, idk yet)
        """
        query_x = self.coord_projection(coord_hist)   # B, 10, 3 -> B, 10, n_embd

        q = self.query(query_x)
        k = self.key(mol_emb)
        v = self.value(mol_emb)

        head_size = q.shape[-1]

        c_attn = q @ k.transpose(-2, -1) * head_size**-0.5
        c_attn = F.softmax(c_attn, dim=-1)
        c_attn = self.dropout(c_attn)

        return c_attn @ v

