

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import os
from torch.utils.data import Dataset, DataLoader
import tiktoken

# from torch.cuda.amp import autocast, GradScaler
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

from datasets import load_dataset


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embedding, n_heads, dropout_p):
        super().__init__()
        assert n_embedding % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embedding // n_heads

        self.q_proj = nn.Linear(n_embedding, n_embedding)
        self.k_proj = nn.Linear(n_embedding, n_embedding)
        self.v_proj = nn.Linear(n_embedding, n_embedding)
        self.out_proj = nn.Linear(n_embedding, n_embedding)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape  # batch size, seq length, embedding dim

        q = self.q_proj(x).view(B, T, self.n_heads,
                                self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads,
                                self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads,
                                self.head_dim).transpose(1, 2)
        # c, embd dim split into n_heads x head_dim

        # built-in scaled dot product attention for efficiency
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)


class FeedForward(nn.Module):
    def __init__(self, n_embedding, dropout_p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding, 4 * n_embedding),
            nn.GELU(),
            nn.Linear(4 * n_embedding, n_embedding),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embedding, n_heads, dropout_p):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embedding)
        self.ln2 = nn.LayerNorm(n_embedding)
        self.attn = MultiHeadAttention(n_embedding, n_heads, dropout_p)
        self.ff = FeedForward(n_embedding, dropout_p)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ff(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_embedding, n_layers, n_heads, dropout_p, block_size):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embedding)
        self.pos_embed = nn.Embedding(block_size, n_embedding)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embedding, n_heads, dropout_p)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embedding)
        self.head = nn.Linear(n_embedding, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size, "Sequence exceeds block size."

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_embed(idx) + self.pos_embed(pos)
        x = self.dropout(x)

        # Causal mask for decoder: prevent attending to future tokens
        attn_mask = torch.ones(T, T, device=idx.device,
                               dtype=torch.bool).tril()

        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits
