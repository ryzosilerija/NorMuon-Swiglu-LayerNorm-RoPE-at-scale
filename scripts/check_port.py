"""Prove the PyTorch port is exact.

  python scripts/check_port.py ../LLM/results/runs/45_normuon_swiglu_layernorm_rope_s1 --data ../LLM/data/tinystories
  python scripts/check_port.py ../LLM/results/runs/45_normuon_swiglu_layernorm_rope_s1 --data ../LLM/data/tinystories --nllm ../LLM

Loads weights trained by the numpy-llm framework into the PyTorch model and
  1. recomputes the validation loss on the study's fixed validation batches:
     it must match the loss the old framework recorded in summary.json;
  2. with --nllm (path to the numpy-llm repo): runs the old framework on the
     same input and compares every output logit directly.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minigpt.convert import load_nllm  # noqa: E402
from minigpt.data import CharData  # noqa: E402

VAL_SEED = 12345


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data", required=True, help="the dataset the run was evaluated on")
    ap.add_argument("--nllm", default=None, help="path to the numpy-llm repo, for a direct logit comparison")
    args = ap.parse_args()

    model, cfg, chars = load_nllm(args.run)
    model.eval()
    data = CharData(args.data)
    assert data.chars == chars, "vocabulary differs from the run's"
    summ = json.load(open(os.path.join(args.run, "summary.json")))

    # 1. validation loss on the identical fixed batches
    rng = np.random.default_rng(VAL_SEED)
    losses = []
    with torch.no_grad():
        for _ in range(cfg.get("eval_batches", 20)):
            x, y = data.batch("val", cfg.get("batch_size", 32), cfg["ctx"], rng, "cpu")
            losses.append(model(x, y)[1].item())
    mine, theirs = float(np.mean(losses)), summ["final_val_loss"]
    ok1 = abs(mine - theirs) < 1e-3
    print(f"validation loss  PyTorch {mine:.5f}  numpy-llm {theirs:.5f}  diff {abs(mine - theirs):.1e}  "
          f"{'OK' if ok1 else 'MISMATCH'}")

    # 2. direct logit comparison against the old framework
    ok2 = True
    if args.nllm:
        sys.path.insert(0, os.path.abspath(args.nllm))
        from nllm.backend import to_numpy  # noqa: E402
        from nllm.train import load_model  # noqa: E402
        old, _, _ = load_model(args.run)
        x, _ = data.batch("val", 4, cfg["ctx"], np.random.default_rng(0), "cpu")
        a = to_numpy(old(x.numpy()).data).astype(np.float64)
        with torch.no_grad():
            b = model(x)[0].double().numpy()
        diff = float(np.abs(a - b).max())
        ok2 = diff < 1e-3
        print(f"logits           max difference {diff:.1e} over {a.size:,} values  {'OK' if ok2 else 'MISMATCH'}")

    print("port is exact" if ok1 and ok2 else "port differs - check the layer mapping")
    sys.exit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    main()
