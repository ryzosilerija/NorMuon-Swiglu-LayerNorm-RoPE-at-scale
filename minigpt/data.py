"""Training data.

CharData   character-level data in the numpy-llm format (step 1, the port check)
TokenData  BPE token files written by scripts/prepare_pretrain.py (step 3 onwards)
load_data  picks the right one from the folder's meta.json
"""
import json
import os

import numpy as np
import torch


class _Windows:
    """Random training windows from a 1-D token array."""

    def batch(self, split, batch_size, ctx, rng, device):
        src = self.splits[split]
        starts = rng.integers(0, len(src) - ctx - 1, size=batch_size)
        x = np.stack([src[s:s + ctx] for s in starts]).astype(np.int64)
        y = np.stack([src[s + 1:s + ctx + 1] for s in starts]).astype(np.int64)
        return torch.from_numpy(x).to(device, non_blocking=True), torch.from_numpy(y).to(device, non_blocking=True)


class CharData(_Windows):
    def __init__(self, data_dir):
        meta = json.load(open(os.path.join(data_dir, "meta.json"), encoding="utf-8"))
        self.chars = meta["chars"]
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.splits = {s: np.memmap(os.path.join(data_dir, f"{s}.bin"), dtype=np.uint16, mode="r")
                       for s in ("train", "val", "test") if os.path.exists(os.path.join(data_dir, f"{s}.bin"))}

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


class TokenData(_Windows):
    def __init__(self, data_dir):
        from .tokenizer import BPETokenizer
        meta = json.load(open(os.path.join(data_dir, "meta.json"), encoding="utf-8"))
        self.tokenizer_path = os.path.join(data_dir, meta["tokenizer"])
        self.tok = BPETokenizer.load(self.tokenizer_path)
        self.vocab_size = self.tok.vocab_size
        self.splits = {s: np.memmap(os.path.join(data_dir, f"{s}.bin"), dtype=np.uint16, mode="r")
                       for s in ("train", "val") if os.path.exists(os.path.join(data_dir, f"{s}.bin"))}

    def encode(self, text):
        return self.tok.encode(text, allow_special=False)

    def decode(self, ids):
        return self.tok.decode(ids)


def load_data(data_dir):
    meta = json.load(open(os.path.join(data_dir, "meta.json"), encoding="utf-8"))
    return TokenData(data_dir) if "tokenizer" in meta else CharData(data_dir)
