
import torch
import torch.nn as nn
from torch.nn import functional as F
import math, time, os
from torch.utils.data import Dataset, DataLoader
import tiktoken

# from torch.cuda.amp import autocast, GradScaler
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

from datasets import load_dataset
from components.model import GPTModel


tokenizer = tiktoken.get_encoding("gpt2")

base_encoding = tiktoken.get_encoding("gpt2")

special_tokens = {
    "[INST]": base_encoding.n_vocab,  # next available token id
    "[/INST]": base_encoding.n_vocab + 1,
}

# 3. Create a new encoding that merges GPT‑2’s tokens + your special tokens
tokenizer = tiktoken.Encoding(
    name="gpt2_with_inst",
    pat_str=base_encoding._pat_str,
    mergeable_ranks=base_encoding._mergeable_ranks,
    special_tokens={**base_encoding._special_tokens, **special_tokens},
)


def encode(text):
    return tokenizer.encode(text, allowed_special={"[INST]", "[/INST]"})


def decode(tokens):
    return tokenizer.decode(tokens)
