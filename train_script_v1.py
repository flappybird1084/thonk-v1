import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tiktoken
from datasets import load_dataset

import time
from components.dataset import TextDataset
from components.model import GPTModel
from components.tokenizer import tokenizer as final_tokenizer
from tqdm import tqdm

base_encoding = tiktoken.get_encoding("r50k_base")
special_tokens = {
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
    return tokenizer.encode(text, allowed_special={"[INST]", "[/INST]", "[STARTOFTEXT]", "[ENDOFTEXT]"})


def decode(text):
    return tokenizer.decode(text)


text = "[STARTOFTEXT] hello world [ENDOFTEXT]"
print(encode(text))
print(decode(encode(text)))

base_dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu", "default", streaming=True)
# next_text = next(iter(base_dataset['train']['text']))

block_size = 256
n_embedding = 256
n_layers = 8
n_heads = 8
dropout_p = 0.1
batch_size = 64
learning_rate = 1e-4
max_iters = 1000000
pbar_update_interval = 2
num_workers = 4
train_model = True
save_model = True
save_path = "checkpoints/v1.pth"
load_model = True
load_path = "checkpoints/v1.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = TextDataset(base_dataset, block_size)

# for i in range(10):
#   start = time.perf_counter()
#   x = (next(iter(dataset['train'])))
#   end = time.perf_counter()
# print(f"time {i}: {end-start}")

model = GPTModel(block_size=block_size, n_embedding=n_embedding, n_layers=n_layers,
                 n_heads=n_heads, dropout_p=dropout_p, vocab_size=final_tokenizer.n_vocab)
text_ctx = torch.tensor(encode("the output vector is B,T,logits")).unsqueeze(0)
model(text_ctx).shape

model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=num_workers)
vocab_size = final_tokenizer.n_vocab
torch.set_float32_matmul_precision("high")


if load_model:
    print(f"loaded model from {load_path}")
    model.load_state_dict(torch.load(load_path))

if train_model:
    compiled_model = torch.compile(model)
    # scaler = torch.amp.GradScaler('cuda')
    pbar = tqdm(total=max_iters)
    data_iter = iter(dataloader)
    for i in range(max_iters):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            xb, yb = next(data_iter)
        xb, yb = xb.to(device), yb.to(device)

        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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
    model, tokenizer, "The mitochondria is known to most in the medical field as ", max_new_tokens=1000, top_k=50)
print("\n--- Generated Text ---")
print(generated)
