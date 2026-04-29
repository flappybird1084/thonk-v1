import math
import os
import copy
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    get_worker_info,
)

from components.model import GPTModel
from components.tokenizer import decode, encode, tokenizer


@dataclass
class TrainConfig:
    block_size: int = 1024
    n_embedding: int = 1024
    n_layers: int = 8
    n_heads: int = 8
    dropout_p: float = 0.1

    pretrain_batch_size: int = 16
    sft_batch_size: int = 16
    dpo_batch_size: int = 8

    pretrain_steps: int = 500000
    sft1_steps: int = 80000
    sft2_steps: int = 80000
    dpo_steps: int = 30000

    pretrain_learning_rate: float = 3e-4
    sft1_learning_rate: float = 5e-5
    sft2_learning_rate: float = 3e-5
    dpo_learning_rate: float = 2e-5

    dpo_beta: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1

    pretrain_num_workers: int = 4
    sft_num_workers: int = 0
    prefetch_factor: int = 2

    compile_model: bool = True
    compile_mode: str = "default"
    log_every: int = 10

    ckpt_dir: str = "checkpoints"
    pretrain_ckpt_name: str = "v4.4_pretrain.pth"
    sft1_ckpt_name: str = "v4.4_sft1_chatbot_instruction_prompts_lora.pth"
    sft2_ckpt_name: str = "v4.4_sft2_alpaca_lora.pth"
    dpo_ckpt_name: str = "v4.4_dpo_ultrafeedback_lora.pth"

    resume_pretrain_if_available: bool = True
    resume_sft1_if_available: bool = True
    resume_sft2_if_available: bool = True
    resume_dpo_if_available: bool = True

    lightning_ckpt_every_n_steps: int = 500
    lora_enable: bool = True
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")


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


def _extract_text(value, preferred_roles: set[str] | None = None) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    texts.append(text)
                continue
            if not isinstance(item, dict):
                continue
            content = item.get("content", "")
            role = str(item.get("role", "")).strip().lower()
            if not isinstance(content, str):
                continue
            content = content.strip()
            if not content:
                continue
            if preferred_roles is None or role in preferred_roles:
                texts.append(content)
        if texts:
            return "\n".join(texts).strip()
    return ""


class UltraFeedbackDPODataset(Dataset):
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.data = load_dataset(
            "HuggingFaceH4/ultrafeedback_binarized", split="train_prefs"
        )
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def _build_pair(self, prompt_text: str, response_text: str):
        prompt_tokens = encode(f" [STARTOFTEXT] [INST] {prompt_text.strip()} [/INST] ")
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

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)
        if (y != -100).sum().item() == 0:
            return None
        return x, y

    def _build_item(self, row):
        prompt = _extract_text(
            row.get("prompt", ""), preferred_roles={"user", "system"}
        )
        chosen = _extract_text(row.get("chosen", ""), preferred_roles={"assistant"})
        rejected = _extract_text(row.get("rejected", ""), preferred_roles={"assistant"})
        if not prompt:
            prompt = _extract_text(row.get("prompt", ""))
        if not chosen:
            chosen = _extract_text(row.get("chosen", ""))
        if not rejected:
            rejected = _extract_text(row.get("rejected", ""))
        if not prompt or not chosen or not rejected:
            return None

        chosen_pair = self._build_pair(prompt, chosen)
        rejected_pair = self._build_pair(prompt, rejected)
        if chosen_pair is None or rejected_pair is None:
            return None

        chosen_x, chosen_y = chosen_pair
        rejected_x, rejected_y = rejected_pair
        return chosen_x, chosen_y, rejected_x, rejected_y

    def __getitem__(self, idx):
        for shift in range(32):
            row = self.data[(idx + shift) % self.size]
            item = self._build_item(row)
            if item is not None:
                return item

        zero_x = torch.zeros(self.block_size, dtype=torch.long)
        zero_y = torch.full((self.block_size,), -100, dtype=torch.long)
        return zero_x, zero_y, zero_x.clone(), zero_y.clone()


def build_model(cfg: TrainConfig) -> GPTModel:
    return GPTModel(
        vocab_size=tokenizer.n_vocab,
        n_embedding=cfg.n_embedding,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout_p=cfg.dropout_p,
        block_size=cfg.block_size,
    )


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0.")

        self.base = base_linear
        self.r = r
        self.scaling = alpha / float(r)
        self.dropout = nn.Dropout(dropout)

        self.lora_A = nn.Parameter(torch.empty(r, self.base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + (lora_out * self.scaling)


def apply_lora_adapters(module: nn.Module, cfg: TrainConfig):
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear) and name in cfg.lora_target_modules:
            setattr(
                module,
                name,
                LoRALinear(
                    base_linear=child,
                    r=cfg.lora_r,
                    alpha=cfg.lora_alpha,
                    dropout=cfg.lora_dropout,
                ),
            )
        else:
            apply_lora_adapters(child, cfg)


def has_lora(module: nn.Module) -> bool:
    return any(isinstance(m, LoRALinear) for m in module.modules())


def mark_only_lora_trainable(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)


def maybe_enable_lora_for_finetuning(model: nn.Module, cfg: TrainConfig):
    if not cfg.lora_enable:
        return
    if not has_lora(model):
        apply_lora_adapters(model, cfg)
    mark_only_lora_trainable(model)


def print_parameter_stats(model: nn.Module, prefix: str):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (100.0 * trainable / total) if total else 0.0
    print(f"{prefix} trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")


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
        # fused=(device == "cuda"),
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


def sequence_logprob_from_logits(logits: torch.Tensor, labels: torch.Tensor):
    log_probs = F.log_softmax(logits, dim=-1)
    valid_mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~valid_mask, 0)
    token_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(
        -1
    )
    token_log_probs = token_log_probs * valid_mask
    return token_log_probs.sum(dim=-1)


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: GPTModel,
        cfg: TrainConfig,
        stage_name: str,
        learning_rate: float,
        total_steps: int,
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.stage_name = stage_name
        self.learning_rate = learning_rate
        self.total_steps = total_steps

    def configure_optimizers(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        optimizer = build_optimizer(
            model=self.model,
            learning_rate=self.learning_rate,
            weight_decay=self.cfg.weight_decay,
            device=device,
        )
        scheduler = build_scheduler(
            optimizer=optimizer,
            total_steps=self.total_steps,
            warmup_ratio=self.cfg.warmup_ratio,
            min_lr_ratio=self.cfg.min_lr_ratio,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class CausalLMLightningModule(BaseLightningModule):
    def __init__(
        self,
        model: GPTModel,
        cfg: TrainConfig,
        stage_name: str,
        learning_rate: float,
        total_steps: int,
    ):
        super().__init__(model, cfg, stage_name, learning_rate, total_steps)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def training_step(self, batch, batch_idx):
        xb, yb = batch
        logits = self.model(xb).transpose(1, 2)
        loss = self.loss_fn(logits, yb)
        self.log(
            f"{self.stage_name}/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=xb.size(0),
        )
        return loss


class DPOLightningModule(BaseLightningModule):
    def __init__(
        self,
        model: GPTModel,
        cfg: TrainConfig,
        stage_name: str,
        total_steps: int,
    ):
        super().__init__(
            model=model,
            cfg=cfg,
            stage_name=stage_name,
            learning_rate=cfg.dpo_learning_rate,
            total_steps=total_steps,
        )
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    def training_step(self, batch, batch_idx):
        chosen_x, chosen_y, rejected_x, rejected_y = batch

        policy_chosen_logits = self.model(chosen_x)
        policy_rejected_logits = self.model(rejected_x)
        policy_chosen_logp = sequence_logprob_from_logits(
            policy_chosen_logits, chosen_y
        )
        policy_rejected_logp = sequence_logprob_from_logits(
            policy_rejected_logits, rejected_y
        )

        with torch.no_grad():
            ref_chosen_logits = self.ref_model(chosen_x)
            ref_rejected_logits = self.ref_model(rejected_x)
            ref_chosen_logp = sequence_logprob_from_logits(ref_chosen_logits, chosen_y)
            ref_rejected_logp = sequence_logprob_from_logits(
                ref_rejected_logits, rejected_y
            )

        policy_logratio = policy_chosen_logp - policy_rejected_logp
        ref_logratio = ref_chosen_logp - ref_rejected_logp
        logits = self.cfg.dpo_beta * (policy_logratio - ref_logratio)
        loss = -F.logsigmoid(logits).mean()
        pref_acc = (policy_logratio > 0).float().mean()

        self.log(
            f"{self.stage_name}/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=chosen_x.size(0),
        )
        self.log(
            f"{self.stage_name}/pref_acc",
            pref_acc,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=chosen_x.size(0),
        )
        return loss


def save_checkpoint(model: GPTModel, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def make_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    device: str,
    shuffle: bool,
    prefetch_factor: int,
    total_steps: int,
):
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": (device == "cuda"),
    }
    if isinstance(dataset, IterableDataset):
        loader_kwargs["shuffle"] = False
    else:
        if shuffle:
            loader_kwargs["sampler"] = RandomSampler(
                dataset,
                replacement=True,
                num_samples=batch_size * total_steps,
            )
        else:
            loader_kwargs["shuffle"] = False
    if device == "cuda":
        loader_kwargs["pin_memory_device"] = "cuda"
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def make_trainer(cfg: TrainConfig, stage_name: str, max_steps: int):
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = "bf16-mixed" if device == "cuda" else "32-true"

    stage_dir = os.path.join(cfg.ckpt_dir, "lightning", stage_name)
    os.makedirs(stage_dir, exist_ok=True)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=stage_dir,
            filename=f"{stage_name}" + "-{step}",
            monitor=f"{stage_name}/loss",
            mode="min",
            save_last=True,
            save_top_k=5,
            every_n_train_steps=cfg.lightning_ckpt_every_n_steps,
            save_on_train_epoch_end=False,
        ),
    ]

    logger = TensorBoardLogger(
        save_dir=cfg.ckpt_dir, name="lightning_logs", version=stage_name
    )

    return pl.Trainer(
        accelerator=accelerator,
        devices=1,
        precision=precision,
        max_steps=max_steps,
        max_epochs=1,
        limit_train_batches=max_steps,
        accumulate_grad_batches=cfg.grad_accum_steps,
        gradient_clip_val=cfg.grad_clip,
        gradient_clip_algorithm="norm",
        log_every_n_steps=cfg.log_every,
        benchmark=(device == "cuda"),
        num_sanity_val_steps=0,
        enable_model_summary=False,
        enable_checkpointing=True,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=cfg.ckpt_dir,
        deterministic=False,
        use_distributed_sampler=False,
    )


def maybe_compile_lightning_module(module: pl.LightningModule, cfg: TrainConfig):
    if cfg.compile_model and torch.cuda.is_available():
        return torch.compile(module, mode=cfg.compile_mode)
    return module


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
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1_000_000:.2f}M")
    print_parameter_stats(model, prefix="Pretrain")
    print(f"Using device: {device}")

    pretrain_ckpt = os.path.join(cfg.ckpt_dir, cfg.pretrain_ckpt_name)
    sft1_ckpt = os.path.join(cfg.ckpt_dir, cfg.sft1_ckpt_name)
    sft2_ckpt = os.path.join(cfg.ckpt_dir, cfg.sft2_ckpt_name)
    dpo_ckpt = os.path.join(cfg.ckpt_dir, cfg.dpo_ckpt_name)

    if cfg.resume_pretrain_if_available and os.path.exists(pretrain_ckpt):
        print(
            f"Found pretrain checkpoint, loading and skipping pretrain: {pretrain_ckpt}"
        )
        model.load_state_dict(torch.load(pretrain_ckpt, map_location=device))
    else:
        print("Stage 1: pretraining on FineWeb-EDU (Lightning)")
        pretrain_dataset = FineWebPretrainDataset(cfg.block_size)
        pretrain_loader = make_loader(
            dataset=pretrain_dataset,
            batch_size=cfg.pretrain_batch_size,
            num_workers=cfg.pretrain_num_workers,
            device=device,
            shuffle=False,
            prefetch_factor=cfg.prefetch_factor,
            total_steps=cfg.pretrain_steps,
        )
        pretrain_module = CausalLMLightningModule(
            model=model,
            cfg=cfg,
            stage_name="pretrain",
            learning_rate=cfg.pretrain_learning_rate,
            total_steps=cfg.pretrain_steps,
        )
        pretrain_module = maybe_compile_lightning_module(pretrain_module, cfg)
        pretrain_trainer = make_trainer(cfg, "pretrain", cfg.pretrain_steps)
        pretrain_trainer.fit(pretrain_module, train_dataloaders=pretrain_loader)
        save_checkpoint(model, pretrain_ckpt)
        print(f"Saved pretrain checkpoint: {pretrain_ckpt}")

    # LoRA applies only to SFT and DPO stages.
    maybe_enable_lora_for_finetuning(model, cfg)
    print_parameter_stats(model, prefix="SFT/DPO (LoRA)")

    if cfg.resume_sft1_if_available and os.path.exists(sft1_ckpt):
        print(f"Found SFT1 checkpoint, loading and skipping SFT1: {sft1_ckpt}")
        model.load_state_dict(torch.load(sft1_ckpt, map_location=device))
    else:
        print("Stage 2: SFT1 on alespalla/chatbot_instruction_prompts (Lightning)")
        sft1_dataset = PromptResponseSFTDataset(
            dataset_name="alespalla/chatbot_instruction_prompts",
            block_size=cfg.block_size,
        )
        sft1_loader = make_loader(
            dataset=sft1_dataset,
            batch_size=cfg.sft_batch_size,
            num_workers=cfg.sft_num_workers,
            device=device,
            shuffle=True,
            prefetch_factor=cfg.prefetch_factor,
            total_steps=cfg.sft1_steps,
        )
        sft1_module = CausalLMLightningModule(
            model=model,
            cfg=cfg,
            stage_name="sft1",
            learning_rate=cfg.sft1_learning_rate,
            total_steps=cfg.sft1_steps,
        )
        sft1_module = maybe_compile_lightning_module(sft1_module, cfg)
        sft1_trainer = make_trainer(cfg, "sft1", cfg.sft1_steps)
        sft1_trainer.fit(sft1_module, train_dataloaders=sft1_loader)
        save_checkpoint(model, sft1_ckpt)
        print(f"Saved SFT1 checkpoint: {sft1_ckpt}")

    if cfg.resume_sft2_if_available and os.path.exists(sft2_ckpt):
        print(f"Found SFT2 checkpoint, loading and skipping SFT2: {sft2_ckpt}")
        model.load_state_dict(torch.load(sft2_ckpt, map_location=device))
    else:
        print("Stage 3: SFT2 on Alpaca (Lightning)")
        sft2_dataset = AlpacaSFTDataset(cfg.block_size)
        sft2_loader = make_loader(
            dataset=sft2_dataset,
            batch_size=cfg.sft_batch_size,
            num_workers=cfg.sft_num_workers,
            device=device,
            shuffle=True,
            prefetch_factor=cfg.prefetch_factor,
            total_steps=cfg.sft2_steps,
        )
        sft2_module = CausalLMLightningModule(
            model=model,
            cfg=cfg,
            stage_name="sft2",
            learning_rate=cfg.sft2_learning_rate,
            total_steps=cfg.sft2_steps,
        )
        sft2_module = maybe_compile_lightning_module(sft2_module, cfg)
        sft2_trainer = make_trainer(cfg, "sft2", cfg.sft2_steps)
        sft2_trainer.fit(sft2_module, train_dataloaders=sft2_loader)
        save_checkpoint(model, sft2_ckpt)
        print(f"Saved SFT2 checkpoint: {sft2_ckpt}")

    if cfg.resume_dpo_if_available and os.path.exists(dpo_ckpt):
        print(f"Found DPO checkpoint, loading and skipping DPO: {dpo_ckpt}")
        model.load_state_dict(torch.load(dpo_ckpt, map_location=device))
    else:
        print("Stage 4: DPO on UltraFeedback (Lightning)")
        dpo_dataset = UltraFeedbackDPODataset(cfg.block_size)
        dpo_loader = make_loader(
            dataset=dpo_dataset,
            batch_size=cfg.dpo_batch_size,
            num_workers=cfg.sft_num_workers,
            device=device,
            shuffle=True,
            prefetch_factor=cfg.prefetch_factor,
            total_steps=cfg.dpo_steps,
        )
        dpo_module = DPOLightningModule(
            model=model,
            cfg=cfg,
            stage_name="dpo",
            total_steps=cfg.dpo_steps,
        )
        dpo_module = maybe_compile_lightning_module(dpo_module, cfg)
        dpo_trainer = make_trainer(cfg, "dpo", cfg.dpo_steps)
        dpo_trainer.fit(dpo_module, train_dataloaders=dpo_loader)
        save_checkpoint(model, dpo_ckpt)
        print(f"Saved DPO checkpoint: {dpo_ckpt}")

    sample_prompt = (
        " [STARTOFTEXT] [INST] "
        "Explain the difference between pretraining and instruction finetuning."
        " [/INST] "
    )
    print("\n--- Sample Generation ---")
    print(generate_text(model, sample_prompt, device, cfg.block_size))


if __name__ == "__main__":
    main()
