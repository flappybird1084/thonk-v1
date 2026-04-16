import contextlib
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from tqdm import tqdm

from components.model import GPTModel
from components.tokenizer import decode, encode, tokenizer


@dataclass
class TrainConfig:
    block_size: int = 256
    n_embedding: int = 256
    n_layers: int = 16
    n_heads: int = 8
    dropout_p: float = 0.1

    pretrain_batch_size: int = 48
    sft_batch_size: int = 32
    pretrain_steps: int = 100_000
    sft1_steps: int = 20_000
    sft2_steps: int = 20_000
    pretrain_learning_rate: float = 3e-4
    sft1_learning_rate: float = 5e-5
    sft2_learning_rate: float = 3e-5
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
    pretrain_ckpt_name: str = "v4_pretrain.pth"
    sft1_ckpt_name: str = "v4_sft1_chatbot_instruction_prompts.pth"
    sft2_ckpt_name: str = "v4_sft2_alpaca.pth"
    resume_pretrain_if_available: bool = True
    resume_sft1_if_available: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FineWebPretrainDataset(IterableDataset):
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "default",
            streaming=True,
        )["train"]

    def __iter__(self):
        worker = get_worker_info()
        stream = self.dataset
        if worker is not None:
            stream = stream.shard(
                num_shards=worker.num_workers,
                index=worker.id,
                contiguous=True,
            )

        token_cache: list[int] = []
        for row in stream:
            packed = f" [STARTOFTEXT] {row['text']} [ENDOFTEXT] "
            token_cache.extend(encode(packed))
            while len(token_cache) >= self.block_size + 1:
                block = token_cache[: self.block_size + 1]
                x = torch.tensor(block[:-1], dtype=torch.long)
                y = torch.tensor(block[1:], dtype=torch.long)
                del token_cache[: self.block_size]
                yield x, y


class PromptResponseSFTDataset(Dataset):
    def __init__(self, dataset_name: str, block_size: int):
        self.block_size = block_size
        ds = load_dataset(dataset_name)
        self.train = ds["train"]
        self.examples: list[tuple[list[int], list[int]]] = []
        self._pretokenize_examples()
        self.size = len(self.examples)

    def __len__(self):
        return self.size

    @staticmethod
    def _format_prompt(prompt_text: str) -> str:
        return f" [STARTOFTEXT] [INST] {prompt_text.strip()} [/INST] "

    def _build_example(self, prompt_text: str, response_text: str):
        prompt_tokens = encode(self._format_prompt(prompt_text))
        response_tokens = encode(f"{response_text.strip()} [ENDOFTEXT] ")

        max_len = self.block_size + 1
        if len(prompt_tokens) >= max_len:
            prompt_tokens = prompt_tokens[: max_len - 1]
            response_tokens = response_tokens[:1]
        elif len(prompt_tokens) + len(response_tokens) > max_len:
            response_tokens = response_tokens[: max_len - len(prompt_tokens)]

        tokens = prompt_tokens + response_tokens
        labels = ([-100] * len(prompt_tokens)) + response_tokens
        if len(tokens) < 2:
            return None

        pad_token = getattr(tokenizer, "eot_token", 0)
        if len(tokens) < max_len:
            pad_len = max_len - len(tokens)
            tokens.extend([pad_token] * pad_len)
            labels.extend([-100] * pad_len)

        x_tokens = tokens[:-1]
        y_tokens = labels[1:]
        if not any(tok != -100 for tok in y_tokens):
            return None
        return x_tokens, y_tokens

    def _iter_prompt_response_pairs(self):
        for row in self.train:
            prompt = row.get("prompt", "")
            response = row.get("response", "")
            if not isinstance(prompt, str) or not isinstance(response, str):
                continue
            if not prompt.strip() or not response.strip():
                continue
            yield prompt, response

    def _pretokenize_examples(self):
        for prompt, response in self._iter_prompt_response_pairs():
            example = self._build_example(prompt, response)
            if example is not None:
                self.examples.append(example)

    def __getitem__(self, idx):
        x_tokens, y_tokens = self.examples[idx]
        x = torch.tensor(x_tokens, dtype=torch.long)
        y = torch.tensor(y_tokens, dtype=torch.long)
        return x, y


class AlpacaSFTDataset(PromptResponseSFTDataset):
    def __init__(self, block_size: int):
        self.block_size = block_size
        ds = load_dataset("tatsu-lab/alpaca")
        self.train = ds["train"]
        self.examples: list[tuple[list[int], list[int]]] = []
        self._pretokenize_examples()
        self.size = len(self.examples)

    @staticmethod
    def _alpaca_prompt(instruction: str, input_text: str) -> str:
        if input_text.strip():
            return (
                "### Instruction:\n"
                f"{instruction.strip()}\n\n"
                "### Input:\n"
                f"{input_text.strip()}\n"
            )
        return f"### Instruction:\n{instruction.strip()}\n"

    def _iter_prompt_response_pairs(self):
        for row in self.train:
            prompt = self._alpaca_prompt(row["instruction"], row.get("input", ""))
            response = row["output"]
            if not prompt.strip() or not response.strip():
                continue
            yield prompt, response


def build_model(cfg: TrainConfig) -> GPTModel:
    return GPTModel(
        vocab_size=tokenizer.n_vocab,
        n_embedding=cfg.n_embedding,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout_p=cfg.dropout_p,
        block_size=cfg.block_size,
    )


def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float, device: str):
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


def save_checkpoint(model: GPTModel, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def run_sft_stage(
    stage_name: str,
    model: GPTModel,
    train_model: nn.Module,
    cfg: TrainConfig,
    loss_fn: nn.Module,
    dataset: Dataset,
    learning_rate: float,
    steps: int,
):
    optimizer = build_optimizer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=cfg.weight_decay,
        device=cfg.device,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=steps,
        warmup_ratio=cfg.warmup_ratio,
        min_lr_ratio=cfg.min_lr_ratio,
    )
    loader_kwargs = {
        "batch_size": cfg.sft_batch_size,
        "shuffle": True,
        "num_workers": cfg.sft_num_workers,
        "pin_memory": (cfg.device == "cuda"),
    }
    if cfg.sft_num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    dataloader = DataLoader(dataset, **loader_kwargs)
    run_stage(
        stage_name=stage_name,
        train_model=train_model,
        grad_model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        steps=steps,
        cfg=cfg,
    )


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

    pretrain_ckpt = os.path.join(cfg.ckpt_dir, cfg.pretrain_ckpt_name)
    sft1_ckpt = os.path.join(cfg.ckpt_dir, cfg.sft1_ckpt_name)
    sft2_ckpt = os.path.join(cfg.ckpt_dir, cfg.sft2_ckpt_name)

    print(f"Using device: {cfg.device}")

    if cfg.resume_pretrain_if_available and os.path.exists(pretrain_ckpt):
        print(f"Found pretrain checkpoint, loading and skipping pretrain: {pretrain_ckpt}")
        model.load_state_dict(torch.load(pretrain_ckpt, map_location=cfg.device))
    else:
        print("Stage 1: pretraining on FineWeb-EDU")
        pretrain_dataset = FineWebPretrainDataset(cfg.block_size)
        pretrain_optimizer = build_optimizer(
            model=model,
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
        pretrain_loader = DataLoader(pretrain_dataset, **pretrain_loader_kwargs)
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

    if cfg.resume_sft1_if_available and os.path.exists(sft1_ckpt):
        print(f"Found SFT1 checkpoint, loading and skipping SFT1: {sft1_ckpt}")
        model.load_state_dict(torch.load(sft1_ckpt, map_location=cfg.device))
    else:
        print("Stage 2: SFT1 on alespalla/chatbot_instruction_prompts")
        sft1_dataset = PromptResponseSFTDataset(
            dataset_name="alespalla/chatbot_instruction_prompts",
            block_size=cfg.block_size,
        )
        run_sft_stage(
            stage_name="sft1_chatbot_instruction_prompts",
            model=model,
            train_model=train_model,
            cfg=cfg,
            loss_fn=loss_fn,
            dataset=sft1_dataset,
            learning_rate=cfg.sft1_learning_rate,
            steps=cfg.sft1_steps,
        )
        save_checkpoint(model, sft1_ckpt)
        print(f"Saved SFT1 checkpoint: {sft1_ckpt}")

    print("Stage 3: SFT2 on Alpaca")
    sft2_dataset = AlpacaSFTDataset(cfg.block_size)
    run_sft_stage(
        stage_name="sft2_alpaca",
        model=model,
        train_model=train_model,
        cfg=cfg,
        loss_fn=loss_fn,
        dataset=sft2_dataset,
        learning_rate=cfg.sft2_learning_rate,
        steps=cfg.sft2_steps,
    )
    save_checkpoint(model, sft2_ckpt)
    print(f"Saved SFT2 checkpoint: {sft2_ckpt}")

    sample_prompt = (
        " [STARTOFTEXT] [INST] "
        "Explain the difference between pretraining and instruction finetuning."
        " [/INST] "
    )
    print("\n--- Sample Generation ---")
    print(generate_text(model, sample_prompt, cfg.device, cfg.block_size))


if __name__ == "__main__":
    main()
