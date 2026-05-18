"""
precompute_inner_loop_sens.py — Inner-loop step-count sensitivity sweep for
MAML/FOMAML/Reptile on held-out sinusoidal tasks (cell 71 of the notebook).

Writes to BOTH:
  - src/data/sampleData/meta-learning/inner_loop_sens.json
  - public/sample-data/meta-learning/inner_loop_sens.json

Output schema:
{
  "N_test": [0, 1, 2, 3, 5, 8, 12],
  "MAML":    [...7 floats],   // mean test MSE
  "FOMAML":  [...7 floats],
  "Reptile": [...7 floats]
}

Runs in ~2-3 minutes on CPU (re-trains all three methods, then 20 held-out
tasks × 3 methods × 7 N_test values).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

SEED = 20260518
DEVICE = torch.device("cpu")
INNER_LR = 0.01
META_LR = 1e-3
N_INNER = 5
BATCH = 4
N_META = 100
REPTILE_BETA = 0.1

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "meta-learning",
    REPO_ROOT / "public" / "sample-data" / "meta-learning",
]


class SmallMLP(nn.Module):
    def __init__(self, hidden=40):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


def sample_sinusoid_batch(rng, B=4, K=5, M=15, x_range=(-5.0, 5.0)):
    A = rng.uniform(0.1, 5.0, size=B)
    phi = rng.uniform(0.0, 2 * np.pi, size=B)
    x_s = rng.uniform(*x_range, size=(B, K))
    x_q = rng.uniform(*x_range, size=(B, M))
    y_s = A[:, None] * np.sin(x_s + phi[:, None])
    y_q = A[:, None] * np.sin(x_q + phi[:, None])
    to_t = lambda a: torch.tensor(a, dtype=torch.float32).unsqueeze(-1)
    return to_t(x_s), to_t(y_s), to_t(x_q), to_t(y_q)


def train_maml():
    torch.manual_seed(SEED)
    model = SmallMLP(hidden=40).to(DEVICE)
    params = {n: p.detach().clone().requires_grad_(True) for n, p in model.named_parameters()}

    def loss_fn(p, x, y):
        return F.mse_loss(functional_call(model, p, (x,)), y)

    def adapt(p, sx, sy):
        for _ in range(N_INNER):
            g = grad(loss_fn)(p, sx, sy)
            p = {k: pp - INNER_LR * g[k] for k, pp in p.items()}
        return p

    def task_meta_loss(p, sx, sy, qx, qy):
        adapted = adapt(p, sx, sy)
        return loss_fn(adapted, qx, qy)

    batched = vmap(task_meta_loss, in_dims=(None, 0, 0, 0, 0))
    opt = torch.optim.Adam(list(params.values()), lr=META_LR)
    rng = np.random.default_rng(SEED)
    for _ in range(N_META):
        sx, sy, qx, qy = sample_sinusoid_batch(rng, B=BATCH)
        opt.zero_grad()
        per_task = batched(params, sx, sy, qx, qy)
        loss = per_task.mean()
        loss.backward()
        opt.step()
    return params, model


def adapt_detached(model, params, sx, sy, alpha, n_inner):
    p = {k: v.detach().clone() for k, v in params.items()}

    def loss_fn(p, x, y):
        return F.mse_loss(functional_call(model, p, (x,)), y)

    for _ in range(n_inner):
        p = {k: v.requires_grad_(True) for k, v in p.items()}
        ls = loss_fn(p, sx, sy)
        gs = torch.autograd.grad(ls, list(p.values()))
        p = {k: (v.detach() - alpha * g) for (k, v), g in zip(p.items(), gs)}
    return p


def train_fomaml():
    torch.manual_seed(SEED)
    model = SmallMLP(hidden=40).to(DEVICE)
    params = {n: p.detach().clone().requires_grad_(True) for n, p in model.named_parameters()}
    opt = torch.optim.Adam(list(params.values()), lr=META_LR)

    def loss_fn(p, x, y):
        return F.mse_loss(functional_call(model, p, (x,)), y)

    rng = np.random.default_rng(SEED)
    for _ in range(N_META):
        sx, sy, qx, qy = sample_sinusoid_batch(rng, B=BATCH)
        opt.zero_grad()
        for p in params.values():
            if p.grad is not None:
                p.grad.zero_()
        for b in range(BATCH):
            adapted = adapt_detached(model, params, sx[b], sy[b], INNER_LR, N_INNER)
            adapted = {k: v.requires_grad_(True) for k, v in adapted.items()}
            ls = loss_fn(adapted, qx[b], qy[b])
            gs = torch.autograd.grad(ls, list(adapted.values()))
            for (k, mp), g in zip(params.items(), gs):
                if mp.grad is None:
                    mp.grad = g.detach() / BATCH
                else:
                    mp.grad = mp.grad + g.detach() / BATCH
        opt.step()
    return params, model


def train_reptile():
    torch.manual_seed(SEED)
    model = SmallMLP(hidden=40).to(DEVICE)
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    rng = np.random.default_rng(SEED)
    for _ in range(N_META):
        sx, sy, qx, qy = sample_sinusoid_batch(rng, B=BATCH)
        for b in range(BATCH):
            x_b = torch.cat([sx[b], qx[b]], dim=0)
            y_b = torch.cat([sy[b], qy[b]], dim=0)
            adapted = adapt_detached(model, params, x_b, y_b, INNER_LR, N_INNER)
            with torch.no_grad():
                for k in params:
                    params[k] += (REPTILE_BETA / BATCH) * (adapted[k] - params[k])
    return params, model


def evaluate_sensitivity(method_params, model, sens_rng, N_test_values, n_held_out=20):
    held_tasks = []
    for _ in range(n_held_out):
        A_h = sens_rng.uniform(0.5, 5.0)
        phi_h = sens_rng.uniform(0, 2 * np.pi)
        xs_h = sens_rng.uniform(-5, 5, size=5)
        xq_h = sens_rng.uniform(-5, 5, size=50)
        ys_h = A_h * np.sin(xs_h + phi_h)
        yq_h = A_h * np.sin(xq_h + phi_h)
        held_tasks.append((
            torch.tensor(xs_h, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(ys_h, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(xq_h, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(yq_h, dtype=torch.float32).unsqueeze(-1),
        ))

    def loss_fn(p, x, y):
        return F.mse_loss(functional_call(model, p, (x,)), y)

    results = []
    for N_test in N_test_values:
        mse_per_task = []
        for (xs_h, ys_h, xq_h, yq_h) in held_tasks:
            p = {k: v.detach().clone() for k, v in method_params.items()}
            for _ in range(N_test):
                p_grad = {k: v.detach().clone().requires_grad_(True) for k, v in p.items()}
                ls = loss_fn(p_grad, xs_h, ys_h)
                gs = torch.autograd.grad(ls, list(p_grad.values()))
                p = {k: v - INNER_LR * g for (k, v), g in zip(p.items(), gs)}
            with torch.no_grad():
                pred = functional_call(model, p, (xq_h,))
                mse = F.mse_loss(pred, yq_h).item()
            mse_per_task.append(mse)
        results.append(float(np.mean(mse_per_task)))
    return results


def main():
    print("Training MAML…", file=sys.stderr)
    maml_params, maml_model = train_maml()
    print("Training FOMAML…", file=sys.stderr)
    fomaml_params, fomaml_model = train_fomaml()
    print("Training Reptile…", file=sys.stderr)
    reptile_params, reptile_model = train_reptile()

    N_test_values = [0, 1, 2, 3, 5, 8, 12]
    print("Sweeping N_test…", file=sys.stderr)
    maml_mse = evaluate_sensitivity(maml_params, maml_model, np.random.default_rng(SEED + 700), N_test_values)
    fomaml_mse = evaluate_sensitivity(fomaml_params, fomaml_model, np.random.default_rng(SEED + 701), N_test_values)
    reptile_mse = evaluate_sensitivity(reptile_params, reptile_model, np.random.default_rng(SEED + 702), N_test_values)

    payload = {
        "N_test": N_test_values,
        "MAML": maml_mse,
        "FOMAML": fomaml_mse,
        "Reptile": reptile_mse,
    }
    for outdir in OUT_DIRS:
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "inner_loop_sens.json", "w") as f:
            json.dump(payload, f, allow_nan=False)
        print(f"wrote {outdir / 'inner_loop_sens.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
