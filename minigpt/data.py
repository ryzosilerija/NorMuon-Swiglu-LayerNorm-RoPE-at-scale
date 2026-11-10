"""Character-level data in the numpy-llm format (train.bin / val.bin / test.bin
as uint16 ids, meta.json with the character list), so step 1 uses exactly
the same data as the study. Step 2 replaces this with a BPE tokenizer."""
import json
import os

import numpy as np
import torch


class CharData:
    def __init__(self, data_dir):
        meta = json.load(open(os.path.join(data_dir, "meta.json"), encoding="utf-8"))
        self.chars = meta["chars"]
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.splits = {}
        for split in ("train", "val", "test"):
            path = os.path.join(data_dir, f"{split}.bin")
            if os.path.exists(path):
                self.splits[split] = np.memmap(path, dtype=np.uint16, mode="r")

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def batch(self, split, batch_size, ctx, rng, device):
        """Random windows; rng is a numpy Generator so batches match nllm for the same seed."""
        src = self.splits[split]
        starts = rng.integers(0, len(src) - ctx - 1, size=batch_size)
        x = np.stack([src[s:s + ctx] for s in starts]).astype(np.int64)
        y = np.stack([src[s + 1:s + ctx + 1] for s in starts]).astype(np.int64)
        return torch.from_numpy(x).to(device, non_blocking=True), torch.from_numpy(y).to(device, non_blocking=True)
