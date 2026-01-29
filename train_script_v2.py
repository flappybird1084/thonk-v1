import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tiktoken
import contextlib
from datasets import load_dataset

import time
from components.dataset import TextDataset
from components.orca import OrcaSFTDataset
from components.model import GPTModel
from components.tokenizer import tokenizer as final_tokenizer, encode, decode
from tqdm import tqdm

# Using OrcaSFTDataset instead of TextDataset
block_size = 256
n_embedding = 256
n_layers = 8
n_heads = 8
dropout_p = 0.1
batch_size = 64  # Different batch size from v1
learning_rate = 1e-4
# max_iters = 100000
max_iters = 50
pbar_update_interval = 2
num_workers = 4
train_model = True
save_model = True
save_path = "checkpoints/v2.pth"
load_model = False
load_path = "checkpoints/v2.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Using OrcaSFTDataset instead of TextDataset
orcadataset = OrcaSFTDataset(block_size)

model = GPTModel(block_size=block_size, n_embedding=n_embedding, n_layers=n_layers,
                 n_heads=n_heads, dropout_p=dropout_p, vocab_size=final_tokenizer.n_vocab)
text_ctx = torch.tensor(encode("the output vector is B,T,logits")).unsqueeze(0)
model(text_ctx).shape

model.to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Using OrcaSFTDataset in the dataloader
dataloader = DataLoader(orcadataset, batch_size=batch_size,
                        num_workers=num_workers)
vocab_size = final_tokenizer.n_vocab
torch.set_float32_matmul_precision("high")


if load_model:
    print(f"loaded model from {load_path}")
    model.load_state_dict(torch.load(load_path))

if train_model:
    compiled_model = torch.compile(model)
    # scaler = torch.amp.GradScaler('cuda')
    pbar = tqdm(total=max_iters, ncols=100)
    data_iter = iter(dataloader)
    amp_ctx = torch.autocast(
        device_type='cuda', dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()
    for i in range(max_iters):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            xb, yb = next(data_iter)
        xb, yb = xb.to(device), yb.to(device)

        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        with amp_ctx:
            logits = compiled_model(xb)
            # logits = logits.argmax(dim=-1)
            logits = logits.transpose(1, 2)
            loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        # scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(
            compiled_model.parameters(), max_norm=1.0)
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()
        if i % pbar_update_interval == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            pbar.update(pbar_update_interval)
    if save_model:
        torch.save(model.state_dict(), save_path)
        print(f"model saved to {save_path}")


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=None):
    """Generate text from the model given a prompt."""
    model.eval()
    with torch.no_grad():
        # Encode prompt
        tokens = torch.tensor(encode(prompt)).unsqueeze(0).to(device)

        # Generation loop
        for _ in range(max_new_tokens):
            # Crop context to block_size
            tokens_cond = tokens if tokens.size(
                1) <= block_size else tokens[:, -block_size:]

            # Forward pass
            logits = model(tokens_cond)

            # Focus on last position
            logits = logits[:, -1, :] / temperature

            # Optional: top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Get probabilities and sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)

    model.train()
    return decode(tokens[0].tolist())


# After the training loop:
generated = generate_text(
    model, final_tokenizer, "[STARTOFTEXT] [INST] The mitochondria is known to most in the medical field as A. The powerhouse of the cell. B. Garbage bin dump. C. Tectonic plates. D. Camera of security in house. Write a neatly formatted response. [/INST] ", max_new_tokens=1000, top_k=50)
print("\n--- Generated Text ---")
print(generated)
