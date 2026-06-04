import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# refer to notes to clear up gaps in understanding this

class Skeleton(nn.Module):
    def __init__(self, n_embd, n_head, n_layers, block_size,
                 map4_res, max_angstroms,
                 dropout):
        super().__init__()
        self.block_size = block_size
        self.map4_res = map4_res
        self.max_angstroms = max_angstroms

        self.c_project = nn.Linear(3, n_embd)  # feed coordinates here
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.mol_encoder = MolecularEncoder(n_embd, map4_res, dropout)


        self.stacks = nn.ModuleList([Stack(n_embd, n_head, block_size, dropout) for _ in range(n_layers)])

        # OUTPUT HEAD -> outputs coordinates
        self.ln_f = nn.LayerNorm(n_embd)
        self.c_head = nn.Linear(n_embd, 3)


    def forward(self, coord_hist, map4_enc, targets=None):
        """
        coord_hist will be [n_batch, block_size, 3]
        targets is [n_batch, 1, 3]
        map4_enc -> unsure how I'm going to encode this
        """
        B, T, C = coord_hist.shape
        coordinate_emb = self.c_project(coord_hist.float())
        pos_emb = self.pos_emb(torch.arange(T, device=coord_hist.device))
        x = coordinate_emb + pos_emb

        mol_emb = self.mol_encoder(map4_enc.float()).unsqueeze(1)

        for stack in self.stacks:
            x = stack(x, mol_emb)

        r, az, pl = self.c_head(self.ln_f(x[:, -1, :])).unbind(dim=1)
        r = F.softplus(r)                 # clip off negative values
        az = torch.tanh(az) * torch.pi    # prevent outputs from being nonsensical angles
        pl = torch.sigmoid(pl) * torch.pi

        output = torch.stack([r, az, pl], dim=1)

        # circular loss instead of regular coordinate MSE
        # doesn't make sense to convert to xyz so I'll use circular loss and radius MSE
        loss = None
        if targets is not None:
            Xrad, Xazm, Xplr = output.unbind(dim=1)
            Yrad, Yazm, Yplr = targets.unbind(dim=1)

            # normalize radius by max_angstroms to get on 0-1 scale, sin and cos are already on -1 to 1
            radial_loss = F.mse_loss(Xrad / self.max_angstroms, Yrad / self.max_angstroms)  # linear scale
            azm_loss = self.circle_loss(Xazm, Yazm)
            pol_loss = self.circle_loss(Xplr, Yplr)

            loss = 0.3 * radial_loss + 0.35 * azm_loss + 0.35 * pol_loss

        return output, loss

    def circle_loss(self, aX, aY):
        # this is calculating the distance between two angles on the unit circle
        return ((torch.cos(aX) - torch.cos(aY))**2 + (torch.sin(aX) - torch.sin(aY))**2).mean()


    def generate(self, map4_enc, max_AAs=130):
        # largest protein pocket in dataset was 107
        map4_enc = map4_enc.to(next(self.parameters()).device)

        # may 26 -> empty outer context pad / "token" = max_angstroms, neutral azimuth, neutral polar
        # -> inner stop "token" is 0 0 0
        # ok changing this after downloading the weights doesn't matter - separate from inference logic
        # max angstroms is actually ~5 angstroms past the max in the dataset, so the relationship is still clear
        coord_context = torch.tensor([[self.max_angstroms, 0, 0] for _ in range(self.block_size)]).unsqueeze(0).to(map4_enc.device)

        coord_out = []

        for _ in range(max_AAs):
            next_coord, _ = self.forward(coord_context, map4_enc)
            coord_out.append(next_coord.detach())
            coord_context = torch.cat([coord_context[:, 1:, :], next_coord.unsqueeze(1)], dim=1)

        return torch.stack(coord_out, dim=1).squeeze(0)

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