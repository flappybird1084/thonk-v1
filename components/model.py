

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


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        self.head_dim = head_dim
        half = head_dim // 2

        freqs = torch.arange(half, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (freqs / half))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        angles = t[:, None] * inv_freq[None, :]
        sin = angles.sin()
        cos = angles.cos()

        # cached on CPU; will be moved to device with the module
        self.register_buffer("sin_cached", sin, persistent=False)
        self.register_buffer("cos_cached", cos, persistent=False)

    def forward(self, q, k):
        # q,k: (B, H, T, D)
        B, H, T, D = q.shape
        half = D // 2
        sin = self.sin_cached[:T].to(q.dtype).to(q.device)  # (T, half)
        cos = self.cos_cached[:T].to(q.dtype).to(q.device)

        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]

        q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
        return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embedding, n_heads, dropout_p, max_seq_len):
        super().__init__()
        assert n_embedding % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embedding // n_heads

        self.q_proj = nn.Linear(n_embedding, n_embedding)
        self.k_proj = nn.Linear(n_embedding, n_embedding)
        self.v_proj = nn.Linear(n_embedding, n_embedding)
        self.out_proj = nn.Linear(n_embedding, n_embedding)
        self.dropout = nn.Dropout(dropout_p)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape  # batch size, seq length, embedding dim

        q = self.q_proj(x).view(B, T, self.n_heads,
                                self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads,
                                self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads,
                                self.head_dim).transpose(1, 2)
        # c, embd dim split into n_heads x head_dim
        q, k = self.rope(q, k)

        # built-in scaled dot product attention for efficiency
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            # attn_mask=attn_mask,
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
    def __init__(self, n_embedding, n_heads, dropout_p, max_seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embedding)
        self.ln2 = nn.LayerNorm(n_embedding)
        self.attn = MultiHeadAttention(n_embedding, n_heads, dropout_p, max_seq_len)
        self.ff = FeedForward(n_embedding, dropout_p)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ff(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_embedding, n_layers, n_heads, dropout_p, block_size):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embedding)
        # self.pos_embed = nn.Embedding(block_size, n_embedding)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embedding, n_heads, dropout_p, block_size)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embedding)
        self.head = nn.Linear(n_embedding, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size, "Sequence exceeds block size."

        # pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        # x = self.token_embed(idx) + self.pos_embed(pos)
        x = self.token_embed(idx)
        x = self.dropout(x)

        # Causal mask for decoder: prevent attending to future tokens
        attn_mask = torch.ones(T, T, device=idx.device,
                               dtype=torch.bool).tril()

        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits
