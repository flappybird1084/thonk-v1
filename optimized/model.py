import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")
        half = head_dim // 2

        freqs = torch.arange(half, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (freqs / half))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        angles = t[:, None] * inv_freq[None, :]
        self.register_buffer("sin_cached", angles.sin(), persistent=False)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        seq_len = q.size(2)
        sin = self.sin_cached[:seq_len].to(dtype=q.dtype, device=q.device).unsqueeze(0).unsqueeze(0)
        cos = self.cos_cached[:seq_len].to(dtype=q.dtype, device=q.device).unsqueeze(0).unsqueeze(0)

        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        q = torch.cat((q1 * cos - q2 * sin, q1 * sin + q2 * cos), dim=-1)
        k = torch.cat((k1 * cos - k2 * sin, k1 * sin + k2 * cos), dim=-1)
        return q, k


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embedding: int, n_heads: int, dropout_p: float, max_seq_len: int):
        super().__init__()
        if n_embedding % n_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads.")
        self.n_heads = n_heads
        self.head_dim = n_embedding // n_heads
        self.qkv_proj = nn.Linear(n_embedding, 3 * n_embedding)
        self.out_proj = nn.Linear(n_embedding, n_embedding)
        self.dropout_p = dropout_p
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, channels = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq, channels)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, n_embedding: int, dropout_p: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding, 4 * n_embedding),
            nn.GELU(),
            nn.Linear(4 * n_embedding, n_embedding),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embedding: int, n_heads: int, dropout_p: float, max_seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embedding)
        self.ln2 = nn.LayerNorm(n_embedding)
        self.attn = MultiHeadAttention(n_embedding, n_heads, dropout_p, max_seq_len)
        self.ff = FeedForward(n_embedding, dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embedding: int,
        n_layers: int,
        n_heads: int,
        dropout_p: float,
        block_size: int,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embedding)
        self.dropout = nn.Dropout(dropout_p)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_embedding=n_embedding,
                    n_heads=n_heads,
                    dropout_p=dropout_p,
                    max_seq_len=block_size,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embedding)
        self.head = nn.Linear(n_embedding, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight
        self.block_size = block_size
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"Sequence length ({seq_len}) exceeds block size ({self.block_size}).")

        x = self.dropout(self.token_embed(idx))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)
