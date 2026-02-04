import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tiktoken
from datasets import load_dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

from components.dataset import TextDataset
from components.model import GPTModel
from components.tokenizer import tokenizer as final_tokenizer

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
    special_tokens={**base_encoding._special_tokens, **special_tokens},
)


def encode(text):
    return tokenizer.encode(
        text, allowed_special={"[INST]", "[/INST]", "[STARTOFTEXT]", "[ENDOFTEXT]"}
    )


def decode(text):
    return tokenizer.decode(text)


text = "[STARTOFTEXT] hello world [ENDOFTEXT]"
print(encode(text))
print(decode(encode(text)))

base_dataset = load_dataset("HuggingFaceFW/fineweb-edu", "default", streaming=True)

block_size = 256
n_embedding = 256
n_layers = 8
n_heads = 8
dropout_p = 0.1
batch_size = 32
learning_rate = 1e-4
# max_iters = 1000000
max_iters = 1000
pbar_update_interval = 2
num_workers = 4
train_model = True
save_model = True
save_path = "checkpoints/v1-pl.pth"
load_model = False
load_path = "checkpoints/v1-pl.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "checkpoints/v1-pl.pth"
checkpoint_interval = 1000
tensorboard_logdir = "tb_logs"

torch.set_float32_matmul_precision("high")

dataset = TextDataset(base_dataset, block_size)


class LitGPT(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = GPTModel(
            block_size=block_size,
            n_embedding=n_embedding,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout_p=dropout_p,
            vocab_size=final_tokenizer.n_vocab,
        )
        self.loss_fn = nn.CrossEntropyLoss()

        if load_model:
            print(f"loaded model from {load_path}")
            self.model.load_state_dict(torch.load(load_path, map_location="cpu"))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        xb, yb = batch
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(self.device.type == "cuda"),
        ):
            logits = self.model(xb)
            logits = logits.transpose(1, 2)
            loss = self.loss_fn(logits, yb)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def on_train_end(self):
        if save_model:
            torch.save(self.model.state_dict(), save_path)
            print(f"model saved to {save_path}")


class TextDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


class SimpleCheckpointCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % checkpoint_interval == 0:
            torch.save(pl_module.model.state_dict(), checkpoint_path)


def generate_text(
    model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=None
):
    """Generate text from the model given a prompt."""
    model.eval()
    with torch.no_grad():
        tokens = torch.tensor(encode(prompt)).unsqueeze(0).to(device)
        for _ in range(max_new_tokens):
            tokens_cond = (
                tokens if tokens.size(1) <= block_size else tokens[:, -block_size:]
            )
            logits = model(tokens_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
    model.train()
    return decode(tokens[0].tolist())


if train_model:
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="v1-step{step}",
        save_top_k=-1,
        every_n_train_steps=checkpoint_interval,
    )
    trainer = pl.Trainer(
        max_steps=max_iters,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        log_every_n_steps=pbar_update_interval,
        enable_checkpointing=True,
        callbacks=[checkpoint_cb, SimpleCheckpointCallback()],
        logger=True,
        default_root_dir=tensorboard_logdir,
        gradient_clip_val=1.0,
    )
    model = LitGPT()
    data_module = TextDataModule()

    text_ctx = torch.tensor(encode("the output vector is B,T,logits")).unsqueeze(0)
    model(text_ctx).shape

    trainer.fit(model, datamodule=data_module)
else:
    model = LitGPT()

model = model.to(device)

generated = generate_text(
    model,
    tokenizer,
    "The mitochondria is known to most in the medical field as ",
    max_new_tokens=1000,
    top_k=50,
)
print("\n--- Generated Text ---")
print(generated)
