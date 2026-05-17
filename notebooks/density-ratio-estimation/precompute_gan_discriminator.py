"""precompute_gan_discriminator.py

Self-contained PyTorch precompute for the §8.4 GAN-discriminator viz. Reproduces
the notebook's BCE training loop (treating q as a frozen 'generator') and
serializes the discriminator trajectory + final grid evaluation to JSON.

Verifies the identity D* = sigmoid(log r) at the BCE optimum.

Output: writes `gan_discriminator.json` to BOTH
  - src/data/sampleData/density-ratio-estimation/
  - public/sample-data/density-ratio-estimation/
(per CLAUDE.md dual-write rule for fetch-able sample data).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "density-ratio-estimation",
    REPO_ROOT / "public" / "sample-data" / "density-ratio-estimation",
]


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SEED = 20260616
MU_Q = 1.0
SIGMA = 1.0
N_POOL = 500
N_ITERS = 1000
GRID_LO = -3.0
GRID_HI = 5.0
N_GRID = 200


# -----------------------------------------------------------------------------
# Model (reused from §8.3)
# -----------------------------------------------------------------------------


class WitnessMLP(nn.Module):
    def __init__(self, d_in: int = 1, hidden: int = 32, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d_in, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# -----------------------------------------------------------------------------
# JSON helpers
# -----------------------------------------------------------------------------


def _to_jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def _round_floats(obj, ndigits: int = 6):
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return 0.0
        return round(obj, ndigits)
    if isinstance(obj, list):
        return [_round_floats(x, ndigits) for x in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    return obj


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))


def main() -> None:
    torch.manual_seed(SEED)

    x_p_gan = torch.randn(N_POOL, 1) * SIGMA + 0.0
    x_q_gan = torch.randn(N_POOL, 1) * SIGMA + MU_Q
    x_pool = torch.cat([x_p_gan, x_q_gan], dim=0)
    y_pool = torch.cat([torch.ones(N_POOL), torch.zeros(N_POOL)])

    D = WitnessMLP(d_in=1, hidden=32, depth=2)
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    snapshot_iters = [0, 50, 100, 200, 400, 700, N_ITERS - 1]
    x_grid = np.linspace(GRID_LO, GRID_HI, N_GRID, dtype=np.float32)
    x_grid_t = torch.from_numpy(x_grid).unsqueeze(1)
    snapshots: dict[int, list[float]] = {}
    loss_trace: list[float] = []

    for it in range(N_ITERS):
        opt_D.zero_grad()
        logits = D(x_pool)
        loss = bce(logits, y_pool)
        loss.backward()
        opt_D.step()
        loss_trace.append(float(loss.item()))
        if it in snapshot_iters:
            with torch.no_grad():
                snapshots[it] = D(x_grid_t).cpu().numpy().tolist()

    with torch.no_grad():
        logits_grid = D(x_grid_t).cpu().numpy()
    D_grid = 1.0 / (1.0 + np.exp(-logits_grid))

    # Closed-form D*(x) = p(x) / (p(x) + q(x)), log r(x) = log(p/q) = 1/2 - x
    p_pdf = normal_pdf(x_grid, 0.0, SIGMA)
    q_pdf = normal_pdf(x_grid, MU_Q, SIGMA)
    D_star = p_pdf / (p_pdf + q_pdf)
    log_r_true_grid = np.log(p_pdf / q_pdf)

    r_hat_disc = D_grid / np.clip(1.0 - D_grid, 1e-9, None)
    r_true_grid_pdf = p_pdf / q_pdf

    pearson_logit_logr = float(np.corrcoef(logits_grid.flatten(), log_r_true_grid)[0, 1])

    payload = {
        "config": {
            "seed": SEED,
            "muQ": MU_Q,
            "sigma": SIGMA,
            "n_pool": N_POOL,
            "n_iters": N_ITERS,
            "lr": 1e-3,
            "hidden": 32,
            "depth": 2,
            "snapshot_iters": snapshot_iters,
        },
        "x_grid": x_grid.tolist(),
        "loss_trace": loss_trace,
        "logits_grid_final": logits_grid.tolist(),
        "D_grid_final": D_grid.tolist(),
        "D_star_grid": D_star.tolist(),
        "log_r_true_grid": log_r_true_grid.tolist(),
        "r_true_grid_pdf": r_true_grid_pdf.tolist(),
        "r_hat_disc_grid": r_hat_disc.tolist(),
        "snapshots": {str(k): v for k, v in snapshots.items()},
        "p_pdf": p_pdf.tolist(),
        "q_pdf": q_pdf.tolist(),
        "pearson_logit_logr": pearson_logit_logr,
        "final_bce": loss_trace[-1],
    }
    payload = _round_floats(_to_jsonable(payload), ndigits=6)

    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "gan_discriminator.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, allow_nan=False, separators=(",", ":"))
        print(f"Wrote {out_path}")

    print(f"\nFinal BCE: {loss_trace[-1]:.4f}")
    print(f"Pearson(logit D, closed-form log r): {pearson_logit_logr:.4f}")


if __name__ == "__main__":
    main()
