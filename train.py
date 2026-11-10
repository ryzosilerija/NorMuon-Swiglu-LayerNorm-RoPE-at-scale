"""Train the GPT with NorMuon in PyTorch.

  # step 1 check: same setup as config 45 in the numpy-llm study (2 layers x 64, 2000 steps)
  python train.py --data data/tinystories --out runs/port_check

  # step 3 pretraining (~42M parameters) on both Kaggle T4 GPUs; stops before the 12 h limit, rerun to continue
  torchrun --nproc_per_node 2 train.py --data data/fineweb --out runs/pretrain --n-layer 8 --d-model 512 \
      --n-head 8 --ctx 512 --batch-size 16 --grad-accum 4 --max-steps 10700 --warmup 300 --lr 0.002 \
      --eval-every 500 --time-limit 11.5

Works with character data (step 1) and BPE token data from scripts/prepare_pretrain.py (step 3).
Multi-GPU: start with torchrun; each GPU trains on different random windows and gradients are averaged
(DistributedDataParallel). Tokens per optimizer step = batch-size x ctx x grad-accum x GPUs.

Resumable: if --out already has ckpt.pt, training continues from it (model, optimizer, step and every
GPU's data position). Mixed precision (fp16) is used automatically on GPU.
"""
import argparse
import json
import math
import os
import shutil
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist

from minigpt.data import TokenData, load_data
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
    ev = {"loss": loss, "acc": correct / count}
    if isinstance(data, TokenData):
        ev["ppl"] = math.exp(loss)               # perplexity: "how many tokens it is choosing between"
    else:
        ev["bpc"] = loss / math.log(2)           # bits per character (the study's metric)
    return ev


def fmt_eval(ev):
    extra = f"perplexity {ev['ppl']:.1f}" if "ppl" in ev else f"BPC {ev['bpc']:.3f}"
    return f"val loss {ev['loss']:.4f}  {extra}  acc {ev['acc']:.1%}"


def gather_rng(rng, world):
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, rng.bit_generator.state)
    else:
        states[0] = rng.bit_generator.state
    return states


def save_ckpt(path, model, opt, scaler, step, rng_states, args):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "rng": rng_states, "args": vars(args)}, tmp)
    os.replace(tmp, path)  # never leave a half-written checkpoint behind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tinystories")
    ap.add_argument("--out", default="runs/port_check")
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32, help="sequences per GPU per micro-step")
    ap.add_argument("--grad-accum", type=int, default=1, help="micro-steps per optimizer step")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.02, help="study's tuned NorMuon LR at small size; 0.0022 at 14M")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--sample-every", type=int, default=0, help="print a short sample every N steps (0 = at evals)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 mixed precision on GPU")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (faster on recent GPUs)")
    ap.add_argument("--time-limit", type=float, default=0.0,
                    help="hours; save a checkpoint and stop cleanly after this long (e.g. 11.5 on Kaggle)")
    ap.add_argument("--stop-at-step", type=int, default=0, help="(testing) save and stop at this step")
    args = ap.parse_args()

    # ---- processes: one per GPU under torchrun, otherwise a single process
    ddp = "RANK" in os.environ
    rank, world = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])) if ddp else (0, 1)
    local = int(os.environ.get("LOCAL_RANK", 0))
    if args.device == "auto":
        device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if ddp:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    main_proc = rank == 0
    say = print if main_proc else (lambda *a, **k: None)
    amp = device.type == "cuda" and not args.no_amp
    torch.manual_seed(args.seed)             # same initial weights on every GPU
    if main_proc:
        os.makedirs(args.out, exist_ok=True)

    data = load_data(args.data)
    cfg = GPTConfig(vocab_size=data.vocab_size, ctx=args.ctx, n_layer=args.n_layer, d_model=args.d_model,
                    n_head=args.n_head)
    model = GPT(cfg).to(device)
    opt = NorMuon(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    # data order: GPU 0 sees exactly the study's batches; other GPUs get their own streams
    rng = np.random.default_rng(args.seed + 1000 + 7919 * rank)
    step = 0

    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        states = ck["rng"] if isinstance(ck["rng"], list) else [ck["rng"]]
        if len(states) == world:
            rng.bit_generator.state = states[rank]
        else:
            say(f"note: checkpoint was made with {len(states)} process(es), now {world}: data streams restart")
        step = ck["step"]
        say(f"resumed from {ckpt_path} at step {step}")
    elif main_proc:
        json.dump({"model": vars(cfg), "train": vars(args), "world_size": world},
                  open(os.path.join(args.out, "config.json"), "w"), indent=2)
        if isinstance(data, TokenData):
            shutil.copy(data.tokenizer_path, os.path.join(args.out, "tokenizer.json"))

    raw, start_step = model, step
    fwd = torch.compile(model) if args.compile else model
    if ddp:
        fwd = torch.nn.parallel.DistributedDataParallel(fwd, device_ids=[local] if device.type == "cuda" else None)
    tokens_per_step = args.batch_size * args.ctx * args.grad_accum * world
    say(f"{raw.n_params():,} parameters on {world} x {device.type}{' (fp16 mixed precision)' if amp else ''}; "
        f"{tokens_per_step:,} tokens per step, {tokens_per_step * args.max_steps / 1e6:,.0f}M tokens in total")
    log = open(os.path.join(args.out, "log.jsonl"), "a") if main_proc else None
    t0, t_last, best = time.time(), time.time(), None
    prompt = data.encode("The most important thing about" if isinstance(data, TokenData) else "Once upon a time")

    while step < args.max_steps:
        lr = args.lr * lr_multiplier(step, args.max_steps, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(args.grad_accum):
            x, y = data.batch("train", args.batch_size, args.ctx, rng, device)
            sync = (not ddp) or micro == args.grad_accum - 1      # average gradients across GPUs only once
            with (nullcontext() if sync else fwd.no_sync()):
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    _, loss = fwd(x, y)
                scaler.scale(loss / args.grad_accum).backward()
            loss_sum += loss.item()
        scaler.unscale_(opt)  # gradients back to real scale before clipping and NorMuon
        gnorm = torch.nn.utils.clip_grad_norm_(raw.parameters(), args.grad_clip).item()
        scaler.step(opt)      # skipped automatically if fp16 produced inf/nan gradients
        scaler.update()
        step += 1
        train_loss = loss_sum / args.grad_accum

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t_last
            n_steps = args.log_every if step > 1 else 1
            eta = (time.time() - t0) / max(step - start_step, 1) * (args.max_steps - step)
            say(f"step {step:6d}  loss {train_loss:.4f}  lr {lr:.2e}  |g| {gnorm:.3f}  "
                f"{n_steps * tokens_per_step / max(dt, 1e-9):,.0f} tok/s  eta {eta / 3600:.1f} h", flush=True)
            t_last = time.time()
        if main_proc and (step % args.eval_every == 0 or step == args.max_steps):
            ev = evaluate(raw, data, args, device, amp)
            best = ev["loss"] if best is None else min(best, ev["loss"])
            say(f"  eval step {step}: {fmt_eval(ev)}", flush=True)
            raw.eval()
            out = raw.generate(torch.tensor([prompt], device=device), 40, 0.8, 20,
                               generator=torch.Generator(device).manual_seed(0))
            raw.train()
            sample = data.decode(out[0].tolist())
            say(f"  sample: {sample!r}", flush=True)
            log.write(json.dumps({"step": step, "lr": lr, "train_loss": train_loss, **ev, "sample": sample,
                                  "tokens": step * tokens_per_step, "elapsed_s": time.time() - t0}) + "\n")
            log.flush()

        # stop / checkpoint decisions are made by GPU 0 and shared, so every process agrees
        out_of_time = bool(args.time_limit and time.time() - t0 > args.time_limit * 3600)
        if ddp:
            flag = torch.tensor([int(out_of_time)], device=device)
            dist.broadcast(flag, 0)
            out_of_time = bool(flag.item())
        stopping = (out_of_time or step == args.stop_at_step) and step < args.max_steps
        if step % args.ckpt_every == 0 or step == args.max_steps or stopping:
            states = gather_rng(rng, world)
            if main_proc:
                save_ckpt(ckpt_path, raw, opt, scaler, step, states, args)
        if stopping:
            say(f"stopped at step {step} ({'time limit' if out_of_time else 'stop-at-step'}); "
                f"run the same command again to continue")
            if ddp:
                dist.barrier()
                dist.destroy_process_group()
            return

    if main_proc:
        blob = {"model": raw.state_dict(), "config": vars(cfg)}
        if isinstance(data, TokenData):
            blob["tokenizer"] = "tokenizer.json"
        else:
            blob["chars"] = data.chars
        torch.save(blob, os.path.join(args.out, "model.pt"))
        final = evaluate(raw, data, args, device, amp)
        json.dump({"final_val_loss": final["loss"], "final_acc": final["acc"], **{k: final[k] for k in ("bpc", "ppl") if k in final},
                   "final_bpc": final.get("bpc"), "best_val_loss": best, "n_params": raw.n_params(),
                   "tokens": step * tokens_per_step, "train_time_s": time.time() - t0},
                  open(os.path.join(args.out, "summary.json"), "w"), indent=2)
        say(f"done: {fmt_eval(final)}  [{(time.time() - t0) / 60:.1f} min]")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
