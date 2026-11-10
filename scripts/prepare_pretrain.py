"""Step 3 data: download text, train the BPE tokenizer on it, and save the
whole corpus as token files for fast training.

  # on Kaggle (internet on): ~700M training tokens of FineWeb-Edu
  python scripts/prepare_pretrain.py --out data/fineweb --tokens 700000000

  # from your own text file(s) instead (one document per paragraph / blank line)
  python scripts/prepare_pretrain.py --out data/mytext --text-file corpus.txt

Output folder:
  tokenizer.json   the BPE tokenizer (vocabulary --vocab-size, default 16384)
  train.bin        training tokens (uint16), documents separated by <|endoftext|>
  val.bin          held-out tokens (the first documents, never trained on)
  meta.json        sizes and settings

FineWeb-Edu: high-quality educational web pages selected by a classifier
(HuggingFaceFW/fineweb-edu, the "sample-10BT" subset, read as a stream, so
nothing larger than needed is downloaded).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minigpt.tokenizer import BPETokenizer  # noqa: E402

_TOK = None


def _init(path):
    global _TOK
    _TOK = BPETokenizer.load(path)


def _encode_batch(docs):
    eot = _TOK.special["<|endoftext|>"]
    out = []
    for d in docs:
        out.extend(_TOK.encode(d, allow_special=False))   # web text must never create special tokens
        out.append(eot)
    return np.array(out, dtype=np.uint16)


def documents(args):
    """Yield documents (strings) from FineWeb-Edu or local text files."""
    if args.text_file:
        for path in args.text_file:
            buf = []
            for line in open(path, encoding="utf-8", errors="replace"):
                if line.strip():
                    buf.append(line)
                elif buf:
                    yield "".join(buf).strip()
                    buf = []
            if buf:
                yield "".join(buf).strip()
        return
    from datasets import load_dataset  # pip install datasets
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=args.subset, split="train", streaming=True)
    for row in ds:
        yield row["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fineweb")
    ap.add_argument("--tokens", type=int, default=700_000_000, help="training tokens to write")
    ap.add_argument("--val-tokens", type=int, default=5_000_000)
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--tokenizer-chars", type=int, default=100_000_000, help="characters used to train the tokenizer")
    ap.add_argument("--subset", default="sample-10BT")
    ap.add_argument("--text-file", nargs="*", default=None, help="use local text file(s) instead of FineWeb-Edu")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--batch-docs", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok_path = os.path.join(args.out, "tokenizer.json")
    t0 = time.time()
    docs = documents(args)

    # 1. tokenizer: train on the first --tokenizer-chars characters (kept in memory, then reused)
    head, n = [], 0
    for d in docs:
        head.append(d)
        n += len(d)
        if n >= args.tokenizer_chars:
            break
    if os.path.exists(tok_path):
        print(f"using existing {tok_path}")
    else:
        print(f"training tokenizer (vocab {args.vocab_size:,}) on {n:,} characters from {len(head):,} documents...")
        BPETokenizer().train("\n".join(head), args.vocab_size).save(tok_path)
        print(f"  done [{time.time() - t0:.0f} s]")
    vocab = BPETokenizer.load(tok_path).vocab_size
    assert vocab <= 65536, "uint16 token files need vocab <= 65536"

    # 2. encode everything in parallel: the first documents become validation, the rest training
    def batches():
        b = []
        for d in head:
            b.append(d)
            if len(b) == args.batch_docs:
                yield b
                b = []
        for d in docs:
            b.append(d)
            if len(b) == args.batch_docs:
                yield b
                b = []
        if b:
            yield b

    files = {"val": open(os.path.join(args.out, "val.bin"), "wb"), "train": open(os.path.join(args.out, "train.bin"), "wb")}
    counts = {"val": 0, "train": 0}
    t1, last = time.time(), 0
    with Pool(args.workers, initializer=_init, initargs=(tok_path,)) as pool:
        for arr in pool.imap(_encode_batch, batches(), chunksize=4):
            split = "val" if counts["val"] < args.val_tokens else "train"
            arr.tofile(files[split])
            counts[split] += len(arr)
            if counts["train"] - last >= 50_000_000:
                last = counts["train"]
                rate = (counts["train"] + counts["val"]) / (time.time() - t1)
                print(f"  {counts['train'] / 1e6:,.0f}M training tokens  ({rate / 1e6:.2f}M tokens/s)", flush=True)
            if counts["train"] >= args.tokens:
                pool.terminate()
                break
    for f in files.values():
        f.close()
    meta = {"tokenizer": "tokenizer.json", "vocab_size": vocab, "train_tokens": counts["train"],
            "val_tokens": counts["val"], "source": "local text" if args.text_file else f"fineweb-edu/{args.subset}"}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    print(f"wrote {args.out}: {counts['train']:,} training + {counts['val']:,} validation tokens, "
          f"vocab {vocab:,} [{(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
