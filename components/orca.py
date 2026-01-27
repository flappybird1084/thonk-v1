
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

class OrcaSFTDataset(Dataset):
  def __init__(self, block_size, stream_dataset=True):
    self.dataset = load_dataset("Open-Orca/OpenOrca", streaming=False)
    self.block_size = block_size
    
    self.current_text_tokenized = []
    self.pull_idx = 0
    self.text_iter = None
    self.response_iter = None
    
  def __len__(self):
    return 1_000_000_000 #streamable dataset :(
      
  def __getitem__(self, idx):
    while len(self.current_text_tokenized) <= self.block_size:
      if self.text_iter is None:
        self.text_iter = iter(self.dataset['train']['question'])

      if self.response_iter is None:
        self.response_iter = iter(self.dataset['train']['response'])

      try:
        text = next(self.text_iter)
        response = next(self.response_iter)
      except StopIteration:
        self.text_iter = iter(self.dataset['train']['question'])
        self.response_iter = iter(self.dataset['train']['response'])
        text = next(self.text_iter)
        response = next(self.response_iter)
        
      tokens = encode(" [STARTOFTEXT] [INST] "+ text + " [/INST] " + response + " [ENDOFTEXT] ")
      self.current_text_tokenized.extend(tokens)

    block = self.current_text_tokenized[:self.block_size + 1]
    x = torch.tensor(block[:self.block_size]).long()
    y = torch.tensor(block[1:self.block_size + 1]).long()
    self.current_text_tokenized = self.current_text_tokenized[1:]
    return x, y