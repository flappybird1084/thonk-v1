import torch
from datasets import Dataset as HFDataset
from torch.utils.data import IterableDataset, get_worker_info

from .tokenizer import encode


class TextDataset(IterableDataset):
    def __init__(self, hf_dataset: HFDataset, block_size: int):
        super().__init__()
        self.dataset = hf_dataset
        self.block_size = block_size

    def __iter__(self):
        stream = self.dataset
        worker = get_worker_info()
        if worker is not None:
            stream = stream.shard(num_shards=worker.num_workers, index=worker.id, contiguous=True)

        token_cache: list[int] = []
        for row in stream:
            text = f" [STARTOFTEXT] {row['text']} [ENDOFTEXT] "
            token_cache.extend(encode(text))

            while len(token_cache) >= self.block_size + 1:
                block = token_cache[: self.block_size + 1]
                x = torch.tensor(block[:-1], dtype=torch.long)
                y = torch.tensor(block[1:], dtype=torch.long)
                del token_cache[: self.block_size]
                yield x, y
