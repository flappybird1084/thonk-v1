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
from components.tokenizer import encode, decode, tokenizer


def decode(tokens):
    return tokenizer.decode(tokens)

class TextDataset(Dataset):
    def __init__(self, hf_dataset, block_size):
        self.dataset = hf_dataset
        # self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.dataset["train"])

    def __getitem__(self, idx):
        # Start with a random index sample
        rand_idx = torch.randint(0, len(self.dataset["train"]), (1,)).item()
        text = self.dataset["train"][rand_idx]["text"]
        tokens = encode(text)

        # Keep appending more samples if too short
        while len(tokens) < self.block_size + 1:
            next_idx = torch.randint(0, len(self.dataset["train"]), (1,)).item()
            next_text = self.dataset["train"][next_idx]["text"]
            tokens.extend(encode(" " + next_text))
            # Prevent runaway growth
            if len(tokens) > self.block_size * 2:
                break

        # Truncate to block_size + 1
        tokens = torch.tensor(tokens[: self.block_size + 1])

        x = tokens[: self.block_size]
        y = tokens[1 : self.block_size + 1]
        return x.long(), y.long()
