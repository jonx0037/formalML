"""precompute_neural_dre.py

Self-contained PyTorch precompute for the §8.3 Neural-DRE viz. Reproduces the
notebook's WitnessMLP + NWJ-KL training loop and serializes the trajectory +
final grid evaluation to JSON.

Output: writes `neural_dre.json` to BOTH
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


# -----------------------------------------------------------------------------
# Paths (matches the dual-write convention from other precompute scripts).
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "density-ratio-estimation",
    REPO_ROOT / "public" / "sample-data" / "density-ratio-estimation",
]


# -----------------------------------------------------------------------------
# DGP
# -----------------------------------------------------------------------------

SEED = 20260615
MU_Q = 1.0
SIGMA = 1.0
N_P = 300
N_Q = 300
N_ITERS = 500
GRID_LO = -3.0
GRID_HI = 5.0
N_GRID = 200


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class WitnessMLP(nn.Module):
    """Faithful copy of the notebook's §8.3 MLP: 1 → 32 → 32 → 1."""

    def __init__(self, d_in: int = 1, hidden: int = 32, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d_in, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def nwj_kl_objective(T_p: torch.Tensor, T_q: torch.Tensor, clip_q: float = 10.0) -> torch.Tensor:
    """Empirical NWJ-KL lower bound; clamp on q-side to avoid exp overflow."""
    T_q_safe = torch.clamp(T_q, max=clip_q)
    return T_p.mean() - torch.exp(T_q_safe - 1.0).mean()


# -----------------------------------------------------------------------------
# Serialization
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


def main() -> None:
    torch.manual_seed(SEED)
    np_rng = np.random.default_rng(SEED)

    # Draw the running toy (NumPy → tensors so the JSON anchors to the same draw).
    x_p = np_rng.normal(0.0, SIGMA, size=N_P).astype(np.float32)
    x_q = np_rng.normal(MU_Q, SIGMA, size=N_Q).astype(np.float32)
    x_grid = np.linspace(GRID_LO, GRID_HI, N_GRID, dtype=np.float32)

    x_p_t = torch.from_numpy(x_p).unsqueeze(1)
    x_q_t = torch.from_numpy(x_q).unsqueeze(1)
    x_grid_t = torch.from_numpy(x_grid).unsqueeze(1)

    net = WitnessMLP(d_in=1, hidden=32, depth=2)
    n_params = sum(p.numel() for p in net.parameters())
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    loss_trace: list[float] = []
    # Record T(x_grid) every k iterations so the viz can replay the trajectory.
    snapshot_iters = [0, 50, 100, 200, 300, 400, N_ITERS - 1]
    snapshots: dict[int, list[float]] = {}

    for it in range(N_ITERS):
        optimizer.zero_grad()
        T_p = net(x_p_t)
        T_q = net(x_q_t)
        obj = nwj_kl_objective(T_p, T_q, clip_q=10.0)
        loss = -obj  # maximize obj == minimize -obj
        loss.backward()
        optimizer.step()
        loss_trace.append(float(obj.detach().item()))
        if it in snapshot_iters:
            with torch.no_grad():
                snapshots[it] = net(x_grid_t).cpu().numpy().tolist()

    # Final evaluation
    with torch.no_grad():
        T_grid = net(x_grid_t).cpu().numpy()
    r_hat_grid = np.exp(T_grid - 1.0)

    # Closed-form references
    log_r_true_grid = (MU_Q * MU_Q) / 2 - MU_Q * x_grid  # 1/2 - x for muQ=sigma=1
    r_true_grid = np.exp(log_r_true_grid)
    kl_truth = (MU_Q ** 2) / (2 * SIGMA ** 2)

    final_nwj = loss_trace[-1]
    pearson_true = float(np.corrcoef(r_hat_grid, r_true_grid)[0, 1])

    payload = {
        "config": {
            "seed": SEED,
            "muQ": MU_Q,
            "sigma": SIGMA,
            "n_p": N_P,
            "n_q": N_Q,
            "n_iters": N_ITERS,
            "snapshot_iters": snapshot_iters,
            "n_params": n_params,
            "lr": 1e-3,
            "hidden": 32,
            "depth": 2,
        },
        "x_grid": x_grid.tolist(),
        "x_p": x_p.tolist(),
        "x_q": x_q.tolist(),
        "loss_trace": loss_trace,
        "T_grid_final": T_grid.tolist(),
        "r_hat_grid": r_hat_grid.tolist(),
        "snapshots": {str(k): v for k, v in snapshots.items()},
        "log_r_true_grid": log_r_true_grid.tolist(),
        "r_true_grid": r_true_grid.tolist(),
        "kl_truth": kl_truth,
        "final_nwj": final_nwj,
        "pearson_true_r": pearson_true,
    }
    payload = _round_floats(_to_jsonable(payload), ndigits=6)

    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "neural_dre.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, allow_nan=False, separators=(",", ":"))
        print(f"Wrote {out_path}")

    print(f"\nFinal NWJ-KL lower bound: {final_nwj:.4f}  (ceiling KL = {kl_truth:.4f})")
    print(f"Pearson(neural r-hat, true r): {pearson_true:.4f}")


if __name__ == "__main__":
    main()
