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

from itertools import islice


# class TextDataset(Dataset):
#     def __init__(self, hf_dataset, block_size):
#         self.dataset = hf_dataset
#         # self.tokenizer = tokenizer
#         self.block_size = block_size
#         self.token_cache = []
#         self.pull_idx = 0

#     def __len__(self):
#         return 1e10

#     def __getitem__(self, idx):
#         # Start with a random index sample
#         while len(self.token_cache) <= self.block_size:
#             text = next(islice(self.dataset['train']['text'], self.pull_idx, None))
#             self.pull_idx += 1
#             text = " [STARTOFTEXT] " + text + " [ENDOFTEXT] "
#             tokens = encode(text)
#             self.token_cache.append(tokens)

#         # Truncate to block_size + 1
#         tokens = torch.tensor(self.token_cache[: self.block_size + 1])

#         x = tokens[: self.block_size]
#         y = tokens[1 : self.block_size + 1]
#         self.token_cache.pop(0)
#         return x.long(), y.long()

class TextDataset(Dataset):
    def __init__(self, hf_dataset, block_size):
        self.dataset = hf_dataset
        self.block_size = block_size
        # Initialize as a flat list of integers (tokens), not a list of lists
        self.token_cache = []
        self.pull_idx = 0
        self.text_iter = None 

    def __len__(self):
        return 1_000_000_000  # Large number for infinite streaming

    def __getitem__(self, idx):
        # Ensure we have enough tokens to fill a block + the target token
        while len(self.token_cache) <= self.block_size:
            # Fix 1: Handle iterator creation correctly to avoid infinite islice creation
            if self.text_iter is None:
                self.text_iter = iter(self.dataset['train']['text'])
            
            try:
                text = next(self.text_iter)
            except StopIteration:
                # Reset if dataset runs out
                self.text_iter = iter(self.dataset['train']['text'])
                text = next(self.text_iter)
            
            text = " [STARTOFTEXT] " + text + " [ENDOFTEXT] "
            
            # Fix 2: Flatten the list. extend() adds individual integers to token_cache
            tokens = encode(text)
            self.token_cache.extend(tokens)
        
        # Grab the next block_size + 1 tokens from the flat cache
        block = self.token_cache[:self.block_size + 1]
        
        # Prepare inputs (x) and targets (y)
        x = torch.tensor(block[:-1], dtype=torch.long)
        y = torch.tensor(block[1:], dtype=torch.long)
        
        # Fix 3: Remove the consumed tokens from the cache (shift the window)
        # Note: popping one by one is slow, slicing is better. 
        # However, to mimic your specific logic of sliding window, we truncate:
        self.token_cache = self.token_cache[self.block_size:]
        
        return x, y