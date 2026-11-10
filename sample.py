"""Continue text with a trained model.

  python sample.py runs/port_check "Once upon a time"
  python sample.py runs/pretrain "The water cycle is" --tokens 150 --temperature 0.7
"""
import argparse
import os

import torch

from minigpt.model import GPT, GPTConfig
from minigpt.tokenizer import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("prompt")
    ap.add_argument("--tokens", "--chars", type=int, default=300, help="how many tokens (characters for char models)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    blob = torch.load(os.path.join(args.run, "model.pt"), map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**blob["config"]))
    model.load_state_dict(blob["model"])
    model.eval()
    if "tokenizer" in blob:
        tok = BPETokenizer.load(os.path.join(args.run, blob["tokenizer"]))
        encode, decode = (lambda s: tok.encode(s, allow_special=False)), tok.decode
    else:
        chars = blob["chars"]
        stoi = {c: i for i, c in enumerate(chars)}
        encode, decode = (lambda s: [stoi[c] for c in s if c in stoi]), (lambda ids: "".join(chars[i] for i in ids))
    idx = torch.tensor([encode(args.prompt)])
    out = model.generate(idx, args.tokens, args.temperature, args.top_k,
                         generator=torch.Generator().manual_seed(args.seed))
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
