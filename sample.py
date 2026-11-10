"""Continue a story with a trained model.

  python sample.py runs/port_check "Once upon a time"
  python sample.py runs/port_check "Lily saw a big dog" --chars 400 --temperature 0.7
"""
import argparse
import os

import torch

from minigpt.model import GPT, GPTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("prompt")
    ap.add_argument("--chars", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    blob = torch.load(os.path.join(args.run, "model.pt"), map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**blob["config"]))
    model.load_state_dict(blob["model"])
    model.eval()
    chars = blob["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    idx = torch.tensor([[stoi[c] for c in args.prompt if c in stoi]])
    out = model.generate(idx, args.chars, args.temperature, args.top_k,
                         generator=torch.Generator().manual_seed(args.seed))
    print("".join(chars[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
