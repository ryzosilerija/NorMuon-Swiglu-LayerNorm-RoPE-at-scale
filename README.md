# mini-chatgpt

A small ChatGPT-style assistant built end to end: tokenizer → pretraining →
instruction tuning → preference tuning → evaluation → serving.

It starts from the winning configuration of my
[language-model study](https://github.com/ryzosilerija/LLM) (62 configurations,
[results](https://ryzosilerija.github.io/LLM/)): a GPT trained with **NorMuon**,
using **SwiGLU**, **LayerNorm** and **RoPE**.

## Roadmap

| step | what | status |
|---|---|---|
| 1 | Port the winning model + NorMuon to PyTorch, prove it matches the old framework | done |
| 2 | Byte-level BPE tokenizer, written from scratch | in progress |
| 3 | Pretraining on diverse text (e.g. FineWeb-Edu), resumable across Kaggle sessions | |
| 4 | Instruction tuning (SFT): chat template, loss only on assistant replies | |
| 5 | Preference tuning (DPO) | |
| 6 | Evaluation: HellaSwag / ARC style multiple choice + chat test set | |
| 7 | Serving: KV-cache, streaming, FastAPI, Docker, quantization | |

## Step 1: PyTorch port

```
minigpt/model.py    GPT: pre-norm blocks, LayerNorm, SwiGLU (hidden 8d/3), RoPE, fused attention
minigpt/optim.py    NorMuon (Newton-Schulz + per-neuron normalization) with AdamW for the rest
minigpt/data.py     character-level data in the study's format (replaced by BPE in step 2)
minigpt/convert.py  load weights trained with the old numpy framework
train.py            training: fp16 on GPU, gradient clipping, cosine schedule, resumable checkpoints
sample.py           continue a story with a trained model
scripts/check_port.py  proves the port is exact
```

Checks:

1. **Same weights, same outputs:** load a model trained by the numpy framework and
   compare its validation loss (and logits) with the old framework's.
2. **Same training result:** train config 45 at the study's small size and compare
   with the study's result (validation loss 0.9993 ± 0.0001 over 3 seeds).

```
python scripts/check_port.py ../LLM/results/runs/45_normuon_swiglu_layernorm_rope_s1 --data ../LLM/data/tinystories --nllm ../LLM
python train.py --data ../LLM/data/tinystories --out runs/port_check
python sample.py runs/port_check "Once upon a time"
```

## Step 2: BPE tokenizer

```
minigpt/tokenizer.py        byte-level BPE: GPT-4 regex pre-split, fast incremental training, special tokens
scripts/train_tokenizer.py  train on a text file, show compression and example splits
tests/test_tokenizer.py     round trips (unicode, emoji), special tokens, and proof that the fast
                            training learns exactly the same merges as textbook BPE
```

```
python tests/test_tokenizer.py
python scripts/train_tokenizer.py ../LLM/data/tinystories/raw.txt --vocab-size 4096 --out tokenizer.json
```
