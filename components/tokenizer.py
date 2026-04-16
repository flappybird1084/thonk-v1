
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



base_encoding= tiktoken.get_encoding("r50k_base")
special_tokens={
  "[INST]": base_encoding.n_vocab,
  "[/INST]": base_encoding.n_vocab + 1,
  "[STARTOFTEXT]": base_encoding.n_vocab + 2,
  "[ENDOFTEXT]": base_encoding.n_vocab + 3,
}
tokenizer = tiktoken.Encoding(
  name="bob",
  pat_str=base_encoding._pat_str,
  mergeable_ranks=base_encoding._mergeable_ranks,
  special_tokens={**base_encoding._special_tokens, **special_tokens}
)
def encode(text):
  return tokenizer.encode(
    text,
    allowed_special={"[INST]", "[/INST]", "[STARTOFTEXT]", "[ENDOFTEXT]"},
    disallowed_special=(),
  )

def decode(text):
  return tokenizer.decode(text)
