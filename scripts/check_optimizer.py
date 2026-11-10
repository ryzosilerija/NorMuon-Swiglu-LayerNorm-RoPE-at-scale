"""Prove NorMuon + the training step match the numpy-llm framework.

  python scripts/check_optimizer.py --nllm ../LLM --data ../LLM/data/tinystories

Starts both frameworks from the *same* initial weights, feeds them the *same*
batches with the same learning-rate schedule and gradient clipping, and
compares the loss at every step and the weights at the end.
"""
import argparse
import json
import math
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minigpt.convert import load_nllm  # noqa: E402
from minigpt.data import CharData  # noqa: E402
from minigpt.optim import NorMuon, lr_multiplier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nllm", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.02)
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.nllm))
    from nllm import tensor as T  # noqa: E402
    from nllm.backend import to_numpy, xp  # noqa: E402
    from nllm.nn import GPT as OldGPT  # noqa: E402
    from nllm.optim import build_optimizer  # noqa: E402

    data = CharData(args.data)
    cfg = {"optimizer": "normuon", "activation": "swiglu", "norm": "layernorm", "positional": "rope",
           "n_layer": 2, "d_model": 64, "n_head": 4, "ctx": 128, "batch_size": 16}
    old = OldGPT(cfg, data.vocab_size, seed=1)
    named = old.named_parameters()

    # hand the old model's initial weights to the PyTorch model
    tmp = tempfile.mkdtemp()
    json.dump(cfg, open(os.path.join(tmp, "config.json"), "w"))
    np.savez(os.path.join(tmp, "model.npz"), **{n: to_numpy(p.data) for n, p in named},
             __chars__=np.array(json.dumps(data.chars)))
    new, _, _ = load_nllm(tmp)

    opt_old = build_optimizer("normuon", named, args.lr, 0.1)
    opt_new = NorMuon(new.named_parameters(), lr=args.lr, weight_decay=0.1)
    rng = np.random.default_rng(7)
    worst = 0.0
    for step in range(args.steps):
        lr = args.lr * lr_multiplier(step, args.steps, 5)
        x, y = data.batch("train", cfg["batch_size"], cfg["ctx"], rng, "cpu")

        # old framework: forward, backward, clip at 1.0, NorMuon step
        old.zero_grad()
        logits = old(xp.asarray(x.numpy()))
        V = logits.shape[-1]
        loss_old = T.cross_entropy(logits.reshape(-1, V), xp.asarray(y.numpy().reshape(-1)))
        loss_old.backward()
        g = math.sqrt(sum(float((p.grad * p.grad).sum()) for _, p in named))
        if g > 1.0:
            for _, p in named:
                p.grad = p.grad * (1.0 / (g + 1e-12))
        opt_old.lr = lr
        opt_old.step()

        # PyTorch
        opt_new.zero_grad(set_to_none=True)
        _, loss_new = new(x, y)
        loss_new.backward()
        torch.nn.utils.clip_grad_norm_(new.parameters(), 1.0)
        for grp in opt_new.param_groups:
            grp["lr"] = lr
        opt_new.step()

        d = abs(loss_old.item() - loss_new.item())
        worst = max(worst, d)
        if step % 5 == 0 or step == args.steps - 1:
            print(f"step {step:3d}  loss numpy-llm {loss_old.item():.6f}  PyTorch {loss_new.item():.6f}  diff {d:.1e}")

    sd = new.state_dict()
    wdiff = 0.0
    for n, p in named:
        a = to_numpy(p.data)
        b = sd[n].numpy()
        if n.endswith(("wqkv.weight", "wo.weight", "w_gate.weight", "w_up.weight", "w_down.weight", "head.weight")):
            b = b.T
        wdiff = max(wdiff, float(np.abs(a - b).max()))
    ok = worst < 1e-3 and wdiff < 1e-3
    print(f"largest loss difference {worst:.1e}, largest weight difference after {args.steps} steps {wdiff:.1e}  "
          f"{'OK: NorMuon and the training step match' if ok else 'MISMATCH'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
