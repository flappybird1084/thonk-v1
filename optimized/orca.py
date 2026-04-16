import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from .tokenizer import encode


class OrcaSFTDataset(Dataset):
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.dataset = load_dataset("Open-Orca/OpenOrca", split="train")
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._build_examples()

    def __len__(self):
        return len(self.examples)

    def _build_examples(self):
        max_len = self.block_size + 1
        for row in self.dataset:
            question = (row.get("question") or "").strip()
            response = (row.get("response") or "").strip()
            if not question or not response:
                continue

            prompt_text = f" [STARTOFTEXT] [INST] {question} [/INST] "
            response_text = f"{response} [ENDOFTEXT] "

            prompt_tokens = encode(prompt_text)
            response_tokens = encode(response_text)
            if len(prompt_tokens) >= max_len:
                prompt_tokens = prompt_tokens[: max_len - 1]
                response_tokens = response_tokens[:1]
            elif len(prompt_tokens) + len(response_tokens) > max_len:
                response_tokens = response_tokens[: max_len - len(prompt_tokens)]

            tokens = prompt_tokens + response_tokens
            labels = ([-100] * len(prompt_tokens)) + response_tokens
            if len(tokens) < 2:
                continue

            x = torch.tensor(tokens[:-1], dtype=torch.long)
            y = torch.tensor(labels[1:], dtype=torch.long)
            if (y != -100).any():
                self.examples.append((x, y))

    def __getitem__(self, idx: int):
        return self.examples[idx]
