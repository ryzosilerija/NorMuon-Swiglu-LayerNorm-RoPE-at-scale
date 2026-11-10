"""Load a model trained with the numpy-llm framework (model.npz + config.json)
into the PyTorch GPT. Used to *prove* the port is exact: same weights in,
same predictions out (python scripts/check_port.py).

Only the winning configuration (SwiGLU / LayerNorm / RoPE) is supported.
"""
import json
import os

import numpy as np
import torch

from .model import GPT, GPTConfig

# nllm Linear weights are (in, out); torch Linear weights are (out, in)
LINEAR_SUFFIXES = ("wqkv.weight", "wo.weight", "w_gate.weight", "w_up.weight", "w_down.weight", "head.weight")


def load_nllm(run_dir, weights=None):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    want = {"activation": "swiglu", "norm": "layernorm", "positional": "rope"}
    bad = {k: cfg.get(k) for k, v in want.items() if cfg.get(k) != v}
    if bad:
        raise ValueError(f"only swiglu/layernorm/rope runs can be converted, this run has {bad}")
    blob = np.load(weights or os.path.join(run_dir, "model.npz"))
    chars = json.loads(str(blob["__chars__"]))
    gcfg = GPTConfig(vocab_size=len(chars), ctx=cfg["ctx"], n_layer=cfg["n_layer"], d_model=cfg["d_model"],
                     n_head=cfg["n_head"])
    model = GPT(gcfg)
    state = {}
    for name, p in model.named_parameters():
        if name not in blob:
            raise KeyError(f"{name} missing from {run_dir}/model.npz")
        a = blob[name]
        if name.endswith(LINEAR_SUFFIXES):
            a = a.T
        if tuple(a.shape) != tuple(p.shape):
            raise ValueError(f"{name}: shape {a.shape} does not fit {tuple(p.shape)}")
        state[name] = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    model.load_state_dict(state)
    return model, cfg, chars
