"""NorMuon for PyTorch, ported from nllm/optim.py.

Hidden 2-D weight matrices (inside the transformer blocks) get NorMuon:
  1. Nesterov momentum
  2. Newton-Schulz orthogonalization (all singular values -> ~1)
  3. per-output-neuron second-moment normalization (the "Nor" in NorMuon)
  4. rescale to 0.2 * sqrt(max(m, n)) so the learning rate is on AdamW's scale
Everything else (embeddings, output head, norm gains/biases) gets AdamW with
the same learning rate. Weight decay is decoupled and applies to 2-D params only.

Layout note: nllm stores Linear weights as (in, out); PyTorch stores them as
(out, in). So an output neuron is a *column* in nllm and a *row* here. The
math is identical; only the axis changes (see _normalize_neurons).
"""
import math

import torch

NS_COEFFS = (3.4445, -4.7750, 2.0315)  # quintic Newton-Schulz (Keller Jordan's Muon)


@torch.no_grad()
def newton_schulz(G, steps=5):
    """Approximately orthogonalize G. Runs in float32 for stability."""
    a, b, c = NS_COEFFS
    X = G.float()
    X = X / (X.norm() + 1e-7)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


class NorMuon(torch.optim.Optimizer):
    def __init__(self, named_params, lr, weight_decay=0.1, momentum=0.95, beta2=0.95, ns_steps=5,
                 adam_betas=(0.9, 0.95), eps=1e-8):
        named_params = list(named_params)
        muon = [p for n, p in named_params if p.ndim == 2 and n.startswith("blocks.")]
        adam = [p for n, p in named_params if not (p.ndim == 2 and n.startswith("blocks."))]
        groups = [dict(params=muon, kind="normuon"), dict(params=adam, kind="adamw")]
        super().__init__(groups, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2,
                                      ns_steps=ns_steps, adam_betas=adam_betas, eps=eps))

    @staticmethod
    def _normalize_neurons(O, state, beta2):
        # PyTorch Linear weight is (out, in): one output neuron = one row
        row_ms = (O * O).mean(dim=1)
        if "v" not in state:
            state["v"] = row_ms.clone()
        else:
            state["v"].mul_(beta2).add_(row_ms, alpha=1 - beta2)
        On = O / (state["v"].sqrt()[:, None] + 1e-8)
        return On * (O.norm() / (On.norm() + 1e-12))  # restore the orthogonalized update's overall size

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if group["kind"] == "normuon":
                    if "buf" not in state:
                        state["buf"] = g.clone()
                    else:
                        state["buf"].mul_(group["momentum"]).add_(g)
                    u = g + group["momentum"] * state["buf"]          # Nesterov
                    O = newton_schulz(u, group["ns_steps"])
                    O = self._normalize_neurons(O, state, group["beta2"])
                    m, n = p.shape
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.add_(O.to(p.dtype), alpha=-lr * 0.2 * math.sqrt(max(m, n)))
                else:  # AdamW, bias-corrected
                    b1, b2 = group["adam_betas"]
                    if "m" not in state:
                        state["m"], state["v"], state["t"] = torch.zeros_like(g), torch.zeros_like(g), 0
                    state["t"] += 1
                    state["m"].mul_(b1).add_(g, alpha=1 - b1)
                    state["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                    mh = state["m"] / (1 - b1 ** state["t"])
                    vh = state["v"] / (1 - b2 ** state["t"])
                    if wd and p.ndim >= 2:
                        p.mul_(1 - lr * wd)
                    p.add_((mh / (vh.sqrt() + group["eps"])).to(p.dtype), alpha=-lr)


def lr_multiplier(step, max_steps, warmup, min_ratio=0.1):
    """Linear warmup, then cosine decay to min_ratio (nllm/schedule.py, 'cosine')."""
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
