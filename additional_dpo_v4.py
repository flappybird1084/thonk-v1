import contextlib
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from components.tokenizer import encode, tokenizer
from train_script_v4 import TrainConfig, build_model, build_optimizer, build_scheduler


@dataclass
class DPOConfig:
    dpo_dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    dpo_split: str = "train_prefs"
    dpo_batch_size: int = 8
    dpo_steps: int = 30_000
    dpo_learning_rate: float = 2e-5
    beta: float = 0.1
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1
    num_workers: int = 0
    compile_model: bool = True
    log_every: int = 10
    ckpt_dir: str = "checkpoints"
    out_ckpt_name: str = "v4_dpo_ultrafeedback.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _extract_text(value, preferred_roles: set[str] | None = None) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            if isinstance(item, str):
                item_text = item.strip()
                if item_text:
                    texts.append(item_text)
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

        fallback_texts: list[str] = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                text = item["content"].strip()
                if text:
                    fallback_texts.append(text)
        return "\n".join(fallback_texts).strip()
    return ""


def _build_pair_tokens(prompt_text: str, response_text: str, block_size: int):
    prompt_formatted = f" [STARTOFTEXT] [INST] {prompt_text.strip()} [/INST] "
    prompt_tokens = encode(prompt_formatted)
    response_tokens = encode(f"{response_text.strip()} [ENDOFTEXT] ")

    max_len = block_size + 1
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


class UltraFeedbackDPODataset(Dataset):
    def __init__(self, dataset_name: str, split: str, block_size: int):
        self.block_size = block_size
        self.data = load_dataset(dataset_name, split=split)
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def _build_item(self, row):
        prompt_text = _extract_text(row.get("prompt", ""), preferred_roles={"user", "system"})
        chosen_text = _extract_text(row.get("chosen", ""), preferred_roles={"assistant"})
        rejected_text = _extract_text(row.get("rejected", ""), preferred_roles={"assistant"})

        if not prompt_text:
            prompt_text = _extract_text(row.get("prompt", ""))
        if not chosen_text:
            chosen_text = _extract_text(row.get("chosen", ""))
        if not rejected_text:
            rejected_text = _extract_text(row.get("rejected", ""))

        if not prompt_text or not chosen_text or not rejected_text:
            return None

        chosen_pair = _build_pair_tokens(prompt_text, chosen_text, self.block_size)
        rejected_pair = _build_pair_tokens(prompt_text, rejected_text, self.block_size)
        if chosen_pair is None or rejected_pair is None:
            return None

        chosen_x, chosen_y = chosen_pair
        rejected_x, rejected_y = rejected_pair
        return chosen_x, chosen_y, rejected_x, rejected_y

    def __getitem__(self, idx):
        # Probe nearby items when malformed rows appear.
        for shift in range(32):
            row = self.data[(idx + shift) % self.size]
            item = self._build_item(row)
            if item is not None:
                return item

        zero_x = torch.zeros(self.block_size, dtype=torch.long)
        zero_y = torch.full((self.block_size,), -100, dtype=torch.long)
        return zero_x, zero_y, zero_x.clone(), zero_y.clone()


def sequence_logprob(model: torch.nn.Module, input_ids: torch.Tensor, labels: torch.Tensor):
    logits = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    valid_mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~valid_mask, 0)
    token_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * valid_mask
    return token_log_probs.sum(dim=-1)


def dpo_loss(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    chosen_x: torch.Tensor,
    chosen_y: torch.Tensor,
    rejected_x: torch.Tensor,
    rejected_y: torch.Tensor,
    beta: float,
    amp_ctx,
):
    with amp_ctx:
        policy_chosen_logp = sequence_logprob(policy_model, chosen_x, chosen_y)
        policy_rejected_logp = sequence_logprob(policy_model, rejected_x, rejected_y)

    with torch.no_grad():
        ref_chosen_logp = sequence_logprob(ref_model, chosen_x, chosen_y)
        ref_rejected_logp = sequence_logprob(ref_model, rejected_x, rejected_y)

    policy_logratio = policy_chosen_logp - policy_rejected_logp
    ref_logratio = ref_chosen_logp - ref_rejected_logp
    logits = beta * (policy_logratio - ref_logratio)
    loss = -F.logsigmoid(logits).mean()

    with torch.no_grad():
        pref_acc = (policy_logratio > 0).float().mean()
    return loss, pref_acc


def resolve_start_ckpt(train_cfg: TrainConfig):
    candidates = [
        os.path.join(train_cfg.ckpt_dir, train_cfg.sft2_ckpt_name),
        os.path.join(train_cfg.ckpt_dir, train_cfg.sft1_ckpt_name),
        os.path.join(train_cfg.ckpt_dir, train_cfg.pretrain_ckpt_name),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No v4 checkpoint found. Tried: {candidates}")


def save_checkpoint(model: torch.nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def main():
    train_cfg = TrainConfig()
    dpo_cfg = DPOConfig()

    torch.set_float32_matmul_precision("high")
    if dpo_cfg.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    start_ckpt = resolve_start_ckpt(train_cfg)
    print(f"Using device: {dpo_cfg.device}")
    print(f"Loading base checkpoint: {start_ckpt}")

    policy_model = build_model(train_cfg).to(dpo_cfg.device)
    policy_model.load_state_dict(torch.load(start_ckpt, map_location=dpo_cfg.device))
    train_model = torch.compile(policy_model) if dpo_cfg.compile_model else policy_model

    ref_model = build_model(train_cfg).to(dpo_cfg.device)
    ref_model.load_state_dict(torch.load(start_ckpt, map_location=dpo_cfg.device))
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    total_params = sum(p.numel() for p in policy_model.parameters())
    print(f"Model parameters: {total_params / 1_000_000:.2f}M")

    dataset = UltraFeedbackDPODataset(
        dataset_name=dpo_cfg.dpo_dataset_name,
        split=dpo_cfg.dpo_split,
        block_size=train_cfg.block_size,
    )
    print(f"DPO dataset rows: {len(dataset)}")

    loader_kwargs = {
        "batch_size": dpo_cfg.dpo_batch_size,
        "shuffle": True,
        "num_workers": dpo_cfg.num_workers,
        "pin_memory": (dpo_cfg.device == "cuda"),
    }
    if dpo_cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    dataloader = DataLoader(dataset, **loader_kwargs)

    optimizer = build_optimizer(
        model=policy_model,
        learning_rate=dpo_cfg.dpo_learning_rate,
        weight_decay=train_cfg.weight_decay,
        device=dpo_cfg.device,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=dpo_cfg.dpo_steps,
        warmup_ratio=dpo_cfg.warmup_ratio,
        min_lr_ratio=dpo_cfg.min_lr_ratio,
    )

    use_amp = dpo_cfg.device == "cuda"
    pbar = tqdm(total=dpo_cfg.dpo_steps, ncols=120, desc="dpo_ultrafeedback")
    data_iter = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, dpo_cfg.dpo_steps + 1):
        running_loss = 0.0
        running_acc = 0.0
        for _ in range(dpo_cfg.grad_accum_steps):
            try:
                chosen_x, chosen_y, rejected_x, rejected_y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                chosen_x, chosen_y, rejected_x, rejected_y = next(data_iter)

            chosen_x = chosen_x.to(dpo_cfg.device, non_blocking=True)
            chosen_y = chosen_y.to(dpo_cfg.device, non_blocking=True)
            rejected_x = rejected_x.to(dpo_cfg.device, non_blocking=True)
            rejected_y = rejected_y.to(dpo_cfg.device, non_blocking=True)

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp
                else contextlib.nullcontext()
            )
            loss, pref_acc = dpo_loss(
                policy_model=train_model,
                ref_model=ref_model,
                chosen_x=chosen_x,
                chosen_y=chosen_y,
                rejected_x=rejected_x,
                rejected_y=rejected_y,
                beta=dpo_cfg.beta,
                amp_ctx=amp_ctx,
            )
            running_loss += loss.item()
            running_acc += pref_acc.item()
            (loss / dpo_cfg.grad_accum_steps).backward()

        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), dpo_cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        pbar.update(1)
        if step % dpo_cfg.log_every == 0:
            pbar.set_postfix(
                loss=f"{(running_loss / dpo_cfg.grad_accum_steps):.4f}",
                pref_acc=f"{(running_acc / dpo_cfg.grad_accum_steps):.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

    out_path = os.path.join(dpo_cfg.ckpt_dir, dpo_cfg.out_ckpt_name)
    save_checkpoint(policy_model, out_path)
    print(f"Saved DPO checkpoint: {out_path}")


if __name__ == "__main__":
    main()
