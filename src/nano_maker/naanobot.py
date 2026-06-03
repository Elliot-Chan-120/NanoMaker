# Generates a suitable biochemical environment for a SMILES given nAAnoVector context
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from src.nano_maker.modules.nAAno.naanoeng import NAAnoEng

# NAANOBOT MODEL
class NAAnoBot(nn.Module):
    def __init__(self, n_embd, n_head, n_layers, block_size,
                 map4_res, max_angstroms,
                 n_nAAno_features, n_spatial_features,
                 dropout):
        super().__init__()
        self.block_size = block_size
        self.map4_res = map4_res
        self.n_nAAno_features = n_nAAno_features
        self.n_spatial_features = n_spatial_features
        total_features = self.n_nAAno_features + self.n_spatial_features

        self.max_angstroms = max_angstroms
        self.naano_module = NAAnoEng(max_angstroms=max_angstroms,
                                     block_size=block_size,
                                     verbose=False)
        self.naano_module.initialize()

        # layers + architecture
        self.nAAno_project = nn.Linear(total_features, n_embd)  # feed nAAno feature vector
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.mol_encoder = MolecularEncoder(n_embd, map4_res, dropout)

        self.stacks = nn.ModuleList([Stack(n_embd, n_head, block_size, dropout) for _ in range(n_layers)])

        # OUTPUT HEAD -> outputs feature vector
        self.ln_f = nn.LayerNorm(n_embd)
        self.nAAno_head = nn.Linear(n_embd, n_nAAno_features)


    def forward(self, nAAno_context, map4_enc, targets=None):
        """
        nAAno_context will be [n_batch, block_size, n_features + 3]
        targets is [n_batch, 1, n_features]
        map4_enc -> unsure how I'm going to encode this
        """
        B, T, C = nAAno_context.shape
        nAAno_emb = self.nAAno_project(nAAno_context.float())
        pos_emb = self.pos_emb(torch.arange(T, device=nAAno_context.device))
        x = nAAno_emb + pos_emb

        mol_emb = self.mol_encoder(map4_enc.float()).unsqueeze(1)

        for stack in self.stacks:
            x = stack(x, mol_emb)

        output = self.nAAno_head(self.ln_f(x[:, -1, :]))

        loss = None
        if targets is not None:
            # split up loss into various parts
            # notes:
            # first 4 -> physicochemical -> removed half life
            # 13 -> functional group fingerprint (one-hot)
            # 4 structural propensity
            physicochemical_loss = F.mse_loss(output[:, :4],
                                              targets[:, :4])
            functional_loss = F.binary_cross_entropy_with_logits(output[:, 4:17],
                                                                 targets[:, 4:17])
            structural_loss = F.mse_loss(output[:, 17:],
                                         targets[:, 17:])

            loss = physicochemical_loss * 0.333 + functional_loss * 0.333 + structural_loss * 0.333

        return output, loss


    def generate(self, map4_enc, sph_coordinates):
        """
        Feed it a list of spherical coordinates, and have them converted to what is done in
        :param map4_enc:
        :param sph_coordinates:
        :return:
        """
        map4_enc = map4_enc.to(next(self.parameters()).device)

        bioch_context = [self.naano_module.get_nAAnovector("VOID") for _ in range(self.block_size)]
        coord_context = [[self.max_angstroms, 0, 0] for _ in range(self.block_size)]

        aa_order = []

        for coordinate in sph_coordinates:   # go through spherical coordinates 1 by 1

            # converts to tensors internally
            naano_X = self.naano_module.get_nAAno_X(coord_context, bioch_context, coordinate).unsqueeze(0)

            output, _ = self.forward(naano_X, map4_enc)
            aa_id = self.naano_module.approx_id(output)

            aa_order.append(aa_id)

            next_nAAnovector = self.naano_module.get_nAAnovector(aa_id)
            bioch_context = bioch_context[1:] + [next_nAAnovector]
            coord_context = coord_context[1:] + [coordinate.tolist() if torch.is_tensor(coordinate) else coordinate]


        return aa_order


# most of the stuff below is from the nanoGPT file except the nanogpt and molecular encoder
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # added in a causal mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # compute affinities
        attn_wts = q @ k.transpose(-2, -1) * C**-0.5
        attn_wts = attn_wts.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        attn_wts = F.softmax(attn_wts, dim=-1)
        attn_wts = self.dropout(attn_wts)

        # weighed aggregation
        values = self.value(x)
        output = attn_wts @ values
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out


class CrossAttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, dropout):
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
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([CrossAttentionHead(n_embd,
                                                       head_size,
                                                       dropout) for _ in range(n_head)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, coord_emb, mol_emb):
        out = torch.cat([h(coord_emb, mol_emb) for h in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out


class FeedForwardNet(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Stack(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.satt_head = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForwardNet(n_embd, dropout)
        self.catt_head = MultiHeadCrossAttention(n_embd, n_head, dropout)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x, mol_emb):
        x = x + self.satt_head(self.ln1(x))
        x = x + self.catt_head(self.ln2(x), mol_emb)
        x = x + self.ffwd(self.ln3(x))
        return x


# previously was way too simple of an MLP, added in more layers
# consider upgrading to use ChemBERTa and save embeddings as vectors to feed in to cross-attention
class MolecularEncoder(nn.Module):
    def __init__(self, n_embd, map4_res, dropout):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Linear(map4_res, map4_res),
            nn.LayerNorm(map4_res),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.check1 = nn.Linear(map4_res, map4_res // 2)

        self.block2 = nn.Sequential(
            nn.Linear(map4_res // 2, map4_res // 2),
            nn.LayerNorm(map4_res // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.check2 = nn.Linear(map4_res // 2, n_embd)
        self.residual1 = nn.Linear(map4_res, map4_res // 2)

        self.block3 = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.LayerNorm(n_embd),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.residual2 = nn.Linear(map4_res // 2, n_embd)


        self.output = nn.LayerNorm(n_embd)

    def forward(self, x):
        x1 = self.block1(x) + x
        x1_down = self.check1(x1)

        x2 = self.block2(x1_down) + self.residual1(x1)
        x2_down = self.check2(x2)

        x3 = self.block3(x2_down) + self.residual2(x2)

        return self.output(x3)