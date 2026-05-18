"""
precompute_maml_loss.py — Generate MAML meta-training loss curves + the
adaptation-evolution curve for a held-out sinusoidal task, snapshotted at the
notebook's default hyperparameters (alpha=0.01, N=5, B=4) AND at a small grid
of alternative (alpha, N) settings so the MamlMetaLossExplorer slider can
select among them.

Writes to BOTH:
  - src/data/sampleData/meta-learning/maml_loss.json
  - public/sample-data/meta-learning/maml_loss.json

Output schema:
{
  "alphaGrid": [0.005, 0.01, 0.02],
  "NGrid":     [3, 5, 8],
  "curves": [
    { "alpha": 0.005, "N": 3, "metaLoss": [...100 floats], "lossAtEnd": float },
    ...
  ],
  "defaultIndex": index into curves with the notebook's (alpha=0.01, N=5),
  "adaptationXSupport": [...5 floats],
  "adaptationYSupport": [...5 floats],
  "adaptationXDense":   [...200 floats],
  "adaptationYTrue":    [...200 floats],
  "adaptationPredsByStep": { "0": [...200], "1": [...200], "3": [...200], "5": [...200] }
}

Runs in ~3-4 minutes on CPU (9 meta-training runs of 100 iters each).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

SEED = 20260518
DEVICE = torch.device("cpu")

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "meta-learning",
    REPO_ROOT / "public" / "sample-data" / "meta-learning",
]


# --- Model -------------------------------------------------------------------
class SmallMLP(nn.Module):
    def __init__(self, hidden=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


# --- Sinusoidal task sampler -------------------------------------------------
def sample_sinusoid_batch(rng, B=4, K=5, M=15, x_range=(-5.0, 5.0)):
    A = rng.uniform(0.1, 5.0, size=B)
    phi = rng.uniform(0.0, 2 * np.pi, size=B)
    x_s = rng.uniform(*x_range, size=(B, K))
    x_q = rng.uniform(*x_range, size=(B, M))
    y_s = A[:, None] * np.sin(x_s + phi[:, None])
    y_q = A[:, None] * np.sin(x_q + phi[:, None])
    to_t = lambda a: torch.tensor(a, dtype=torch.float32).unsqueeze(-1)
    return to_t(x_s), to_t(y_s), to_t(x_q), to_t(y_q)


# --- One full MAML training run ----------------------------------------------
def train_maml(alpha, N_inner, n_meta=100, seed=SEED, B=4):
    torch.manual_seed(seed)
    model = SmallMLP(hidden=40).to(DEVICE)
    params = {n: p.detach().clone().requires_grad_(True) for n, p in model.named_parameters()}

    def loss_fn(p, x, y):
        return F.mse_loss(functional_call(model, p, (x,)), y)

    def adapt(p, sx, sy):
        for _ in range(N_inner):
            g = grad(loss_fn)(p, sx, sy)
            p = {k: pp - alpha * g[k] for k, pp in p.items()}
        return p

    def task_meta_loss(p, sx, sy, qx, qy):
        adapted = adapt(p, sx, sy)
        return loss_fn(adapted, qx, qy)

    batched = vmap(task_meta_loss, in_dims=(None, 0, 0, 0, 0))
    opt = torch.optim.Adam(list(params.values()), lr=1e-3)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(n_meta):
        sx, sy, qx, qy = sample_sinusoid_batch(rng, B=B)
        opt.zero_grad()
        per_task = batched(params, sx, sy, qx, qy)
        loss = per_task.mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return params, model, losses


# --- Adaptation evolution on a held-out task ---------------------------------
def adaptation_evolution(params, model, alpha, x_s, y_s, x_dense, n_steps_list):
    """Re-run the inner loop from `params` for various step counts; return predictions."""
    x_s_t = torch.tensor(x_s, dtype=torch.float32).unsqueeze(-1)
    y_s_t = torch.tensor(y_s, dtype=torch.float32).unsqueeze(-1)
    x_dense_t = torch.tensor(x_dense, dtype=torch.float32).unsqueeze(-1)

    def loss_fn(p, x, y):
        return F.mse_loss(functional_call(model, p, (x,)), y)

    grad_loss = grad(loss_fn)
    out = {}
    for n in n_steps_list:
        p = {k: v.detach().clone() for k, v in params.items()}
        for _ in range(n):
            g = grad_loss(p, x_s_t, y_s_t)
            p = {k: v - alpha * g[k] for k, v in p.items()}
        with torch.no_grad():
            yhat = functional_call(model, p, (x_dense_t,)).squeeze(-1).numpy().tolist()
        out[str(n)] = yhat
    return out


def main():
    alpha_grid = [0.005, 0.01, 0.02]
    N_grid = [3, 5, 8]
    curves = []
    default_index = 0
    print("Training 9 MAML configurations…", file=sys.stderr)
    for a in alpha_grid:
        for N in N_grid:
            print(f"  alpha={a}, N={N}…", file=sys.stderr)
            params, model, losses = train_maml(a, N)
            curve = {
                "alpha": float(a),
                "N": int(N),
                "metaLoss": [float(x) for x in losses],
                "lossAtEnd": float(losses[-1]),
            }
            curves.append(curve)
            if abs(a - 0.01) < 1e-9 and N == 5:
                default_index = len(curves) - 1
                # Held-out adaptation evolution at the default config
                # Use the same held-out task as cell 17: A=3.2, phi=1.7, fixed support.
                eval_rng = np.random.default_rng(SEED + 1)
                A_eval, phi_eval = 3.2, 1.7
                x_s = eval_rng.uniform(-5, 5, size=5)
                y_s = A_eval * np.sin(x_s + phi_eval)
                x_dense = np.linspace(-5, 5, 200)
                y_true = A_eval * np.sin(x_dense + phi_eval)
                preds = adaptation_evolution(params, model, a, x_s, y_s, x_dense, [0, 1, 3, 5])
                adaptation = {
                    "adaptationXSupport": x_s.tolist(),
                    "adaptationYSupport": y_s.tolist(),
                    "adaptationXDense": x_dense.tolist(),
                    "adaptationYTrue": y_true.tolist(),
                    "adaptationPredsByStep": preds,
                }

    payload = {
        "alphaGrid": alpha_grid,
        "NGrid": N_grid,
        "curves": curves,
        "defaultIndex": default_index,
        **adaptation,
    }

    for outdir in OUT_DIRS:
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "maml_loss.json", "w") as f:
            json.dump(payload, f, allow_nan=False)
        print(f"wrote {outdir / 'maml_loss.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
