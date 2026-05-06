import json
import os
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import IterableDataset, get_worker_info

from components.tokenizer import encode, tokenizer


def load_v44_module():
    module_path = Path(__file__).resolve().parent / "train_script_v4_4-lightning.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing dependency script: {module_path}")

    spec = importlib.util.spec_from_file_location(
        "train_script_v4_4_lightning", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V44 = load_v44_module()


@dataclass
class TrainConfig(V44.TrainConfig):
    cot_batch_size: int = 6
    cot_grad_accum_steps: int = 16
    cot_steps: int = 400000
    cot_learning_rate: float = 2e-5
    cot_ckpt_name: str = "v4.5_cot_collection_on_dpo_lora.pth"
    resume_cot_if_available: bool = True

    cot_dataset_name: str = "kaist-ai/CoT-Collection"
    cot_dataset_json_fallback: str = "data/CoT_collection_en.json"


class CoTCollectionSFTDataset(IterableDataset):
    def __init__(self, cfg: TrainConfig):
        self.block_size = cfg.block_size
        self.path = hf_hub_download(
            repo_id=cfg.cot_dataset_name,
            repo_type="dataset",
            filename=cfg.cot_dataset_json_fallback,
        )

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

    @staticmethod
    def _stream_top_level_object_values(path: str):
        decoder = json.JSONDecoder()
        chunk_size = 1 << 20
        with open(path, "r", encoding="utf-8") as f:
            buf = ""
            pos = 0
            started = False
            done = False
            eof = False

            while not done:
                if pos >= len(buf) and not eof:
                    chunk = f.read(chunk_size)
                    if chunk:
                        buf = buf[pos:] + chunk
                        pos = 0
                    else:
                        eof = True
                if pos >= len(buf):
                    break

                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if pos >= len(buf):
                    continue

                if not started:
                    if buf[pos] != "{":
                        raise RuntimeError("Expected JSON object at top level.")
                    started = True
                    pos += 1
                    continue

                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if pos >= len(buf):
                    continue
                if buf[pos] == ",":
                    pos += 1
                    continue
                if buf[pos] == "}":
                    done = True
                    continue
                if buf[pos] != '"':
                    if eof:
                        context = buf[max(0, pos - 40) : min(len(buf), pos + 40)]
                        raise RuntimeError(
                            "Malformed JSON object: expected a quoted key. "
                            f"Found {buf[pos]!r} at pos {pos}. Context: {context!r}"
                        )
                    chunk = f.read(chunk_size)
                    if not chunk:
                        context = buf[max(0, pos - 40) : min(len(buf), pos + 40)]
                        eof = True
                        raise RuntimeError(
                            "Malformed JSON object: expected a quoted key. "
                            f"Found {buf[pos]!r} at pos {pos}. Context: {context!r}"
                        )
                    else:
                        buf += chunk
                    continue

                while True:
                    try:
                        _, pos = decoder.raw_decode(buf, pos)
                        break
                    except json.JSONDecodeError:
                        if eof:
                            raise
                        chunk = f.read(chunk_size)
                        if not chunk:
                            eof = True
                        else:
                            buf += chunk

                while True:
                    while pos < len(buf) and buf[pos].isspace():
                        pos += 1
                    if pos < len(buf):
                        break
                    if eof:
                        raise RuntimeError("Malformed JSON object: unexpected EOF.")
                    chunk = f.read(chunk_size)
                    if not chunk:
                        eof = True
                    else:
                        buf += chunk

                if buf[pos] != ":":
                    if eof:
                        raise RuntimeError("Malformed JSON object: expected ':'.")
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise RuntimeError("Malformed JSON object: expected ':'.")
                    buf += chunk
                    continue
                pos += 1

                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                while True:
                    try:
                        value, pos = decoder.raw_decode(buf, pos)
                        break
                    except json.JSONDecodeError:
                        if eof:
                            raise
                        chunk = f.read(chunk_size)
                        if not chunk:
                            eof = True
                        else:
                            buf += chunk

                yield value

                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if pos < len(buf) and buf[pos] == ",":
                    pos += 1
                elif pos < len(buf) and buf[pos] == "}":
                    done = True

                if pos > (1 << 20):
                    buf = buf[pos:]
                    pos = 0

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        for idx, row in enumerate(self._stream_top_level_object_values(self.path)):
            if idx % num_workers != worker_id:
                continue
            if not isinstance(row, dict):
                continue
            source = str(row.get("source", "")).strip()
            response = str(row.get("response", row.get("rationale", ""))).strip()
            target = str(row.get("target", "")).strip()
            if not source or not response or not target:
                continue

            completion = f"{response} <answer> {target}"
            example = self._build_example(source, completion)
            if example is None:
                continue

            x_tokens, y_tokens = example
            x = torch.tensor(x_tokens, dtype=torch.long)
            y = torch.tensor(y_tokens, dtype=torch.long)
            yield x, y


def main():
    cfg = TrainConfig()

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = V44.build_model(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1_000_000:.2f}M")
    print(f"Using device: {device}")

    dpo_ckpt = os.path.join(cfg.ckpt_dir, cfg.dpo_ckpt_name)
    sft2_ckpt = os.path.join(cfg.ckpt_dir, cfg.sft2_ckpt_name)
    sft1_ckpt = os.path.join(cfg.ckpt_dir, cfg.sft1_ckpt_name)
    pretrain_ckpt = os.path.join(cfg.ckpt_dir, cfg.pretrain_ckpt_name)
    cot_ckpt = os.path.join(cfg.ckpt_dir, cfg.cot_ckpt_name)

    # Pick highest-stage available v4.4 checkpoint.
    base_candidates = [dpo_ckpt, sft2_ckpt, sft1_ckpt, pretrain_ckpt]
    base_ckpt = next((path for path in base_candidates if os.path.exists(path)), None)
    if base_ckpt is None:
        raise FileNotFoundError(f"No base checkpoint found. Tried: {base_candidates}")

    base_state = torch.load(base_ckpt, map_location=device)
    has_lora_weights = any(
        k.endswith(".lora_A") or k.endswith(".lora_B") for k in base_state.keys()
    )
    print(f"Loading base checkpoint for v4.5 stage: {base_ckpt}")
    if has_lora_weights:
        # LoRA checkpoints need adapter modules present before loading.
        V44.maybe_enable_lora_for_finetuning(model, cfg)
        model.load_state_dict(base_state)
    else:
        # Plain checkpoints load into plain model first, then adapters are attached.
        model.load_state_dict(base_state)
        V44.maybe_enable_lora_for_finetuning(model, cfg)

    if cfg.resume_cot_if_available and os.path.exists(cot_ckpt):
        print(f"Found CoT v4.5 checkpoint, loading and skipping stage: {cot_ckpt}")
        model.load_state_dict(torch.load(cot_ckpt, map_location=device))
    else:
        print("Stage: CoT-Collection SFT on top of DPO (Lightning)")
        dataset = CoTCollectionSFTDataset(cfg)
        print("Streaming CoT-Collection examples from JSON...")

        train_loader = V44.make_loader(
            dataset=dataset,
            batch_size=cfg.cot_batch_size,
            num_workers=cfg.sft_num_workers,
            device=device,
            shuffle=True,
            prefetch_factor=cfg.prefetch_factor,
            total_steps=cfg.cot_steps,
        )

        module = V44.CausalLMLightningModule(
            model=model,
            cfg=cfg,
            stage_name="cot_collection",
            learning_rate=cfg.cot_learning_rate,
            total_steps=cfg.cot_steps,
        )
        module = V44.maybe_compile_lightning_module(module, cfg)
        cfg.grad_accum_steps = cfg.cot_grad_accum_steps
        trainer = V44.make_trainer(cfg, "cot_collection", cfg.cot_steps)
        trainer.fit(module, train_dataloaders=train_loader)

        V44.save_checkpoint(model, cot_ckpt)
        print(f"Saved CoT v4.5 checkpoint: {cot_ckpt}")

    sample_prompt = (
        " [STARTOFTEXT] [INST] "
        "If a train leaves at 9am and travels 60 miles in 1.5 hours, what is the speed?"
        " [/INST] "
    )
    print("\n--- Sample Generation ---")
    print(V44.generate_text(model, sample_prompt, device, cfg.block_size))


if __name__ == "__main__":
    main()
