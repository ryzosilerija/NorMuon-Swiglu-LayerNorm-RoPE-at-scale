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
| 2 | Byte-level BPE tokenizer, written from scratch | done |
| 3 | Pretraining a ~42M model on 0.7B tokens of FineWeb-Edu, 2 GPUs, resumable | in progress |
| 4 | Instruction tuning (SFT): chat template, loss only on assistant replies | |
| 5 | Preference tuning (DPO) | |
| 6 | Evaluation: HellaSwag / ARC style multiple choice + chat test set | |
| 7 | Serving: KV-cache, streaming, FastAPI, Docker, quantization | |

## Step 1: PyTorch port

```
minigpt/model.py    GPT: pre-norm blocks, LayerNorm, SwiGLU (hidden 8d/3), RoPE, fused attention
minigpt/optim.py    NorMuon (Newton-Schulz + per-neuron normalization) with AdamW for the rest
minigpt/data.py     training windows: character data (study format) or BPE token files
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

## Step 3: pretraining

```
scripts/prepare_pretrain.py  stream FineWeb-Edu, train a 16,384-token BPE on 100M characters,
                             encode ~700M tokens in parallel into train.bin / val.bin (uint16)
train.py                     + token data, multi-GPU (torchrun / DistributedDataParallel),
                             gradient accumulation, samples printed at every eval
```

Model: 8 layers x 512, 8 heads, context 512 -> ~42M parameters (~25M outside the embeddings).
Batch: 16 sequences x 512 tokens x 4 accumulation x 2 GPUs = 65,536 tokens per step, 10,700 steps.

Kaggle notebook (GPU T4 x2, Internet on), one cell, then Save Version -> Save & Run All:

```
!pip install -q regex datasets
!git clone https://github.com/ryzosilerija/NorMuon-Swiglu-LayerNorm-RoPE-at-scale.git mini-chatgpt
%cd mini-chatgpt
!python scripts/prepare_pretrain.py --out /kaggle/temp/fineweb --tokens 700000000
!torchrun --nproc_per_node 2 train.py --data /kaggle/temp/fineweb --out /kaggle/working/runs/pretrain \
    --n-layer 8 --d-model 512 --n-head 8 --ctx 512 --batch-size 16 --grad-accum 4 \
    --max-steps 10700 --warmup 300 --lr 0.002 --eval-every 500 --time-limit 11.5
```

Then: `python sample.py runs/pretrain "The water cycle is" --tokens 150`
