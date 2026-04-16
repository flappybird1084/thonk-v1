import contextlib
import os
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from tqdm import tqdm

from components.model import GPTModel
from components.tokenizer import encode, decode, tokenizer


@dataclass
class TrainConfig:
    block_size: int = 256
    n_embedding: int = 256
    n_layers: int = 16
    n_heads: int = 8
    dropout_p: float = 0.1

    pretrain_batch_size: int = 48
    sft_batch_size: int = 32
    pretrain_steps: int = 100000
    sft_steps: int = 20000
    pretrain_learning_rate: float = 3e-4
    sft_learning_rate: float = 5e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1
    pretrain_num_workers: int = 4
    sft_num_workers: int = 0

    compile_model: bool = True
    log_every: int = 10
    ckpt_dir: str = "checkpoints"
    pretrain_ckpt_name: str = "v3_pretrain.pth"
    sft_ckpt_name: str = "v3_sft.pth"
    resume_pretrain_if_available: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FineWebPretrainDataset(IterableDataset):
    """Streams web text and produces causal LM blocks."""

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", "default", streaming=True
        )["train"]

    def __len__(self):
        return 1_000_000_000

    def __iter__(self):
        worker = get_worker_info()
        stream = self.dataset
        if worker is not None:
            stream = stream.shard(
                num_shards=worker.num_workers, index=worker.id, contiguous=True
            )

        token_cache: list[int] = []
        for row in stream:
            packed = f" [STARTOFTEXT] {row['text']} [ENDOFTEXT] "
            token_cache.extend(encode(packed))

            # Non-overlapping block packing improves gradient diversity.
            while len(token_cache) >= self.block_size + 1:
                block = token_cache[: self.block_size + 1]
                x = torch.tensor(block[:-1], dtype=torch.long)
                y = torch.tensor(block[1:], dtype=torch.long)
                del token_cache[: self.block_size]
                yield x, y


class AlpacaSFTDataset(Dataset):
    """Alpaca SFT dataset with prompt-token masking (ignore instruction loss)."""

    def __init__(self, block_size: int):
        self.block_size = block_size
        ds = load_dataset("tatsu-lab/alpaca")
        self.train = ds["train"]
        self.examples: list[tuple[list[int], list[int]]] = []
        self._pretokenize_examples()
        self.size = len(self.examples)

    def __len__(self):
        return self.size

    @staticmethod
    def _format_prompt(instruction: str, input_text: str) -> str:
        if input_text.strip():
            user_prompt = (
                "### Instruction:\n"
                f"{instruction.strip()}\n\n"
                "### Input:\n"
                f"{input_text.strip()}\n"
            )
        else:
            user_prompt = f"### Instruction:\n{instruction.strip()}\n"

        return f" [STARTOFTEXT] [INST] {user_prompt} [/INST] "

    def _build_example(self, row):
        prompt_text = self._format_prompt(row["instruction"], row.get("input", ""))
        response_text = f"{row['output'].strip()} [ENDOFTEXT] "

        prompt_tokens = encode(prompt_text)
        response_tokens = encode(response_text)

        max_len = self.block_size + 1
        if len(prompt_tokens) >= max_len:
            # Keep instruction context and at least one supervised response token.
            prompt_tokens = prompt_tokens[: max_len - 1]
            response_tokens = response_tokens[:1]
        elif len(prompt_tokens) + len(response_tokens) > max_len:
            response_tokens = response_tokens[: max_len - len(prompt_tokens)]

        tokens = prompt_tokens + response_tokens
        labels = ([-100] * len(prompt_tokens)) + response_tokens

        if len(tokens) < 2:
            return None

        # Force fixed-length samples so default DataLoader collation can stack safely.
        pad_token = getattr(tokenizer, "eot_token", 0)
        if len(tokens) < max_len:
            pad_len = max_len - len(tokens)
            tokens.extend([pad_token] * pad_len)
            labels.extend([-100] * pad_len)
        elif len(tokens) > max_len:
            tokens = tokens[:max_len]
            labels = labels[:max_len]

        x_tokens = tokens[:-1]
        y_tokens = labels[1:]

        if not any(tok != -100 for tok in y_tokens):
            return None
        return x_tokens, y_tokens

    def _pretokenize_examples(self):
        for row in self.train:
            example = self._build_example(row)
            if example is not None:
                self.examples.append(example)

    def __getitem__(self, idx):
        x_tokens, y_tokens = self.examples[idx]
        x = torch.tensor(x_tokens, dtype=torch.long)
        y = torch.tensor(y_tokens, dtype=torch.long)
        return x, y


def build_model(cfg: TrainConfig) -> GPTModel:
    return GPTModel(
        vocab_size=tokenizer.n_vocab,
        n_embedding=cfg.n_embedding,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout_p=cfg.dropout_p,
        block_size=cfg.block_size,
    )


def run_stage(
    stage_name: str,
    train_model: nn.Module,
    grad_model: GPTModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: nn.Module,
    steps: int,
    cfg: TrainConfig,
):
    train_model.train()

    use_amp = cfg.device == "cuda"

    pbar = tqdm(total=steps, ncols=110, desc=stage_name)
    data_iter = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, steps + 1):
        running_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            try:
                xb, yb = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                xb, yb = next(data_iter)

            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp
                else contextlib.nullcontext()
            )
            with amp_ctx:
                logits = train_model(xb).transpose(1, 2)
                loss = loss_fn(logits, yb)
                running_loss += loss.item()
                loss = loss / cfg.grad_accum_steps

            loss.backward()

        torch.nn.utils.clip_grad_norm_(grad_model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        pbar.update(1)
        if step % cfg.log_every == 0:
            pbar.set_postfix(
                loss=f"{(running_loss / cfg.grad_accum_steps):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )


def build_optimizer(
    model: nn.Module, learning_rate: float, weight_decay: float, device: str
):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.dim() < 2
            or name.endswith("bias")
            or "ln" in name.lower()
            or "norm" in name.lower()
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=(device == "cuda"),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model: GPTModel, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def generate_text(
    model: GPTModel,
    prompt: str,
    device: str,
    block_size: int,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int | None = 50,
) -> str:
    model.eval()
    tokens = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx = tokens[:, -block_size:]
            logits = model(ctx)[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

    model.train()
    return decode(tokens[0].tolist())


def main():
    cfg = TrainConfig()
    torch.set_float32_matmul_precision("high")
    if cfg.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_model(cfg).to(cfg.device)
    train_model = torch.compile(model) if cfg.compile_model else model
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"Using device: {cfg.device}")
    pretrain_ckpt = os.path.join(cfg.ckpt_dir, cfg.pretrain_ckpt_name)
    if cfg.resume_pretrain_if_available and os.path.exists(pretrain_ckpt):
        print(
            f"Found pretrain checkpoint, loading and skipping pretrain: {pretrain_ckpt}"
        )
        model.load_state_dict(torch.load(pretrain_ckpt, map_location=cfg.device))
    else:
        print("Stage 1: pretraining on FineWeb-EDU")

        pretrain_dataset = FineWebPretrainDataset(cfg.block_size)
        pretrain_optimizer = build_optimizer(
            model,
            learning_rate=cfg.pretrain_learning_rate,
            weight_decay=cfg.weight_decay,
            device=cfg.device,
        )
        pretrain_scheduler = build_scheduler(
            optimizer=pretrain_optimizer,
            total_steps=cfg.pretrain_steps,
            warmup_ratio=cfg.warmup_ratio,
            min_lr_ratio=cfg.min_lr_ratio,
        )
        pretrain_loader_kwargs = {
            "batch_size": cfg.pretrain_batch_size,
            "num_workers": cfg.pretrain_num_workers,
            "pin_memory": (cfg.device == "cuda"),
        }
        if cfg.pretrain_num_workers > 0:
            pretrain_loader_kwargs["persistent_workers"] = True
            pretrain_loader_kwargs["prefetch_factor"] = 2

        pretrain_loader = DataLoader(
            pretrain_dataset,
            **pretrain_loader_kwargs,
        )
        run_stage(
            stage_name="pretrain",
            train_model=train_model,
            grad_model=model,
            dataloader=pretrain_loader,
            optimizer=pretrain_optimizer,
            scheduler=pretrain_scheduler,
            loss_fn=loss_fn,
            steps=cfg.pretrain_steps,
            cfg=cfg,
        )

        save_checkpoint(model, pretrain_ckpt)
        print(f"Saved pretrain checkpoint: {pretrain_ckpt}")

    print("Stage 2: SFT finetuning on Alpaca with prompt loss masked")
    sft_dataset = AlpacaSFTDataset(cfg.block_size)
    sft_optimizer = build_optimizer(
        model,
        learning_rate=cfg.sft_learning_rate,
        weight_decay=cfg.weight_decay,
        device=cfg.device,
    )
    sft_scheduler = build_scheduler(
        optimizer=sft_optimizer,
        total_steps=cfg.sft_steps,
        warmup_ratio=cfg.warmup_ratio,
        min_lr_ratio=cfg.min_lr_ratio,
    )
    sft_loader_kwargs = {
        "batch_size": cfg.sft_batch_size,
        "shuffle": True,
        "num_workers": cfg.sft_num_workers,
        "pin_memory": (cfg.device == "cuda"),
    }
    if cfg.sft_num_workers > 0:
        sft_loader_kwargs["persistent_workers"] = True
        sft_loader_kwargs["prefetch_factor"] = 2

    sft_loader = DataLoader(
        sft_dataset,
        **sft_loader_kwargs,
    )
    run_stage(
        stage_name="alpaca_sft",
        train_model=train_model,
        grad_model=model,
        dataloader=sft_loader,
        optimizer=sft_optimizer,
        scheduler=sft_scheduler,
        loss_fn=loss_fn,
        steps=cfg.sft_steps,
        cfg=cfg,
    )

    sft_ckpt = os.path.join(cfg.ckpt_dir, cfg.sft_ckpt_name)
    save_checkpoint(model, sft_ckpt)
    print(f"Saved SFT checkpoint: {sft_ckpt}")

    sample_prompt = (
        " [STARTOFTEXT] [INST] ### Instruction:\n"
        "Explain why masking prompt tokens is used in SFT.\n"
        "[/INST] "
    )
    print("\\n--- Sample Generation ---")
    print(generate_text(model, sample_prompt, cfg.device, cfg.block_size))


if __name__ == "__main__":
    main()
