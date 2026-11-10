"""Train the GPT with NorMuon in PyTorch.

  # step 1 check: same setup as config 45 in the numpy-llm study (2 layers x 64, 2000 steps)
  python train.py --data data/tinystories --out runs/port_check

  # the big model from Part 2, on a Kaggle GPU (stops cleanly before Kaggle's 12 h limit; rerun to continue)
  python train.py --data data/tinystories_big --out /kaggle/working/runs/big --n-layer 8 --d-model 384 \
      --n-head 8 --batch-size 64 --max-steps 34800 --warmup 1000 --lr 0.0022 --eval-every 1000 --time-limit 11.5

Resumable: if --out already has ckpt.pt, training continues from it
(model, optimizer, step and data position), so long runs can span several
12-hour Kaggle sessions. Mixed precision (fp16) is used automatically on GPU.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from minigpt.data import CharData
from minigpt.model import GPT, GPTConfig
from minigpt.optim import NorMuon, lr_multiplier

VAL_SEED = 12345  # same fixed validation batches as the numpy-llm study


@torch.no_grad()
def evaluate(model, data, args, device, amp):
    model.eval()
    rng = np.random.default_rng(VAL_SEED)
    losses, correct, count = [], 0, 0
    for _ in range(args.eval_batches):
        x, y = data.batch("val", args.batch_size, args.ctx, rng, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits, loss = model(x, y)
        losses.append(loss.item())
        correct += (logits.argmax(-1) == y).sum().item()
        count += y.numel()
    model.train()
    loss = float(np.mean(losses))
    return {"loss": loss, "bpc": loss / math.log(2), "acc": correct / count}


def save_ckpt(path, model, opt, step, rng, args):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "rng": rng.bit_generator.state, "args": vars(args)}, tmp)
    os.replace(tmp, path)  # never leave a half-written checkpoint behind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tinystories")
    ap.add_argument("--out", default="runs/port_check")
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.02, help="study's tuned NorMuon LR at small size; 0.0022 at 14M")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 mixed precision on GPU")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (faster on recent GPUs)")
    ap.add_argument("--time-limit", type=float, default=0.0,
                    help="hours; save a checkpoint and stop cleanly after this long (e.g. 11.5 on Kaggle)")
    ap.add_argument("--stop-at-step", type=int, default=0, help="(testing) save and stop at this step")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    amp = device.type == "cuda" and not args.no_amp
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    data = CharData(args.data)
    cfg = GPTConfig(vocab_size=data.vocab_size, ctx=args.ctx, n_layer=args.n_layer, d_model=args.d_model,
                    n_head=args.n_head)
    model = GPT(cfg).to(device)
    opt = NorMuon(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    rng = np.random.default_rng(args.seed + 1000)  # same training batches as the numpy-llm study
    step = 0

    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]
        step = ck["step"]
        print(f"resumed from {ckpt_path} at step {step}")
    else:
        json.dump({"model": vars(cfg), "train": vars(args)}, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    fwd = torch.compile(model) if args.compile else model
    print(f"{model.n_params():,} parameters on {device}{' (fp16 mixed precision)' if amp else ''}")
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    tokens_per_step = args.batch_size * args.ctx
    t0, t_last, best = time.time(), time.time(), None

    while step < args.max_steps:
        lr = args.lr * lr_multiplier(step, args.max_steps, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = data.batch("train", args.batch_size, args.ctx, rng, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            _, loss = fwd(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)  # gradients back to real scale before clipping and NorMuon
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        scaler.step(opt)      # skipped automatically if fp16 produced inf/nan gradients
        scaler.update()
        step += 1

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t_last
            n_steps = args.log_every if step > 1 else 1
            print(f"step {step:6d}  loss {loss.item():.4f}  lr {lr:.2e}  |g| {gnorm:.3f}  "
                  f"{n_steps * tokens_per_step / max(dt, 1e-9):,.0f} tok/s", flush=True)
            t_last = time.time()
        if step % args.eval_every == 0 or step == args.max_steps:
            ev = evaluate(model, data, args, device, amp)
            best = ev["loss"] if best is None else min(best, ev["loss"])
            print(f"  eval step {step}: val loss {ev['loss']:.4f}  BPC {ev['bpc']:.3f}  acc {ev['acc']:.1%}", flush=True)
            log.write(json.dumps({"step": step, "lr": lr, "train_loss": loss.item(), **ev,
                                  "elapsed_s": time.time() - t0}) + "\n")
            log.flush()
        if step % args.ckpt_every == 0 or step == args.max_steps:
            save_ckpt(ckpt_path, model, opt, step, rng, args)
        out_of_time = args.time_limit and time.time() - t0 > args.time_limit * 3600
        if (out_of_time or step == args.stop_at_step) and step < args.max_steps:
            save_ckpt(ckpt_path, model, opt, step, rng, args)
            print(f"stopped at step {step} ({'time limit' if out_of_time else 'stop-at-step'}); "
                  f"run the same command again to continue")
            return

    torch.save({"model": model.state_dict(), "config": vars(cfg), "chars": data.chars},
               os.path.join(args.out, "model.pt"))
    final = evaluate(model, data, args, device, amp)
    json.dump({"final_val_loss": final["loss"], "final_bpc": final["bpc"], "final_acc": final["acc"],
               "best_val_loss": best, "n_params": model.n_params(), "train_time_s": time.time() - t0},
              open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(f"done: val loss {final['loss']:.4f}  BPC {final['bpc']:.3f}  [{(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
