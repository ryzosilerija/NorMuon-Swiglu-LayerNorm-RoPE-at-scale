"""Train the BPE tokenizer on a text file and show how it splits text.

  python scripts/train_tokenizer.py ../LLM/data/tinystories/raw.txt --vocab-size 4096 --out tokenizer.json
  python scripts/train_tokenizer.py some_corpus.txt --vocab-size 16384 --max-chars 50000000

Prints training progress, the compression it achieves (characters per token,
vs 1.0 for your character-level models), the longest tokens it learned, and an
example sentence split into tokens.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minigpt.tokenizer import BPETokenizer  # noqa: E402

EXAMPLE = "Once upon a time, there was a little girl named Lily. She loved to play outside!"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text_file")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--max-chars", type=int, default=20_000_000, help="train on at most this many characters")
    ap.add_argument("--out", default="tokenizer.json")
    args = ap.parse_args()

    text = open(args.text_file, encoding="utf-8", errors="replace").read(args.max_chars)
    text = text.replace("<|endoftext|>", "\n")
    print(f"training on {len(text):,} characters, target vocabulary {args.vocab_size:,}")
    t0 = time.time()
    tok = BPETokenizer().train(text, args.vocab_size)
    print(f"trained in {time.time() - t0:.0f} s: {len(tok.merges):,} merges + 256 bytes + "
          f"{len(tok.special)} special tokens = {tok.vocab_size:,}")
    tok.save(args.out)
    print(f"saved {args.out}")

    sample = text[-200_000:]
    ids = tok.encode(sample, allow_special=False)
    assert tok.decode(ids) == sample, "round trip failed"
    print(f"\ncompression: {len(sample) / len(ids):.2f} characters per token "
          f"(your character-level models: 1.00), round trip OK")

    longest = sorted((i for i in tok.vocab if i >= 256), key=lambda i: -len(tok.vocab[i]))[:15]
    print("longest tokens:", ", ".join(repr(tok.token_str(i)) for i in longest))
    ex = tok.encode(EXAMPLE)
    print(f"\n{EXAMPLE}\n-> {len(ex)} tokens: " + " | ".join(tok.token_str(i) for i in ex))


if __name__ == "__main__":
    main()
