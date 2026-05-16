"""Precompute pre-trained MLPRegressor coefficients for the §7/§9/§11
interactive viz on the live site.

The notebook is read-only; this side script rebuilds the §2 HETERO_TOY data
(seed-aligned with the notebook) and the §7.4 deep ensemble at M=50, then
serializes per-member ``coefs_`` / ``intercepts_`` plus the aleatoric
heteroscedastic interpolator to JSON.

Output: written to BOTH
  * ``public/sample-data/uncertainty-quantification/ensemble.json`` (Astro
    serves this at /sample-data/uncertainty-quantification/ensemble.json)
  * ``src/data/sampleData/uncertainty-quantification/ensemble.json`` (build
    target — tracked but not served)

Run:
    cd notebooks/uncertainty-quantification && .venv/bin/python precompute_ensemble.py
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Seed must match notebook setup cell for the ensemble to be trained on the
# same HETERO_TOY draw as fig_07_*.png and fig_09_*.png.
RANDOM_SEED = 20260514
rng = np.random.default_rng(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Notebook-aligned config (see §2 setup and §7.4).
N_RUNNING = 200
M_FULL = 50
HIDDEN = (16, 16)
ACTIVATION = "tanh"
MAX_ITER = 1000


def f_true(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


def sigma_true(x: np.ndarray) -> np.ndarray:
    return 0.1 + 0.5 * np.abs(x)


def aleatoric_from_residuals(X: np.ndarray, y: np.ndarray, f_mean: np.ndarray, n_bins: int = 15):
    """Replicates the notebook helper for the served viz; emits the bin centers
    + smoothed bin values so the TS interpolator on the client can rebuild it."""
    resid_sq = (y - f_mean) ** 2
    edges = np.linspace(X.min(), X.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    vals = np.zeros(n_bins)
    for k in range(n_bins):
        mask = (X >= edges[k]) & (X <= edges[k + 1])
        vals[k] = resid_sq[mask].mean() if mask.any() else np.nan
    valid = ~np.isnan(vals)
    vals = vals[valid]
    centers = centers[valid]
    if len(vals) >= 3:
        vals = np.convolve(vals, np.ones(3) / 3, mode="same")
    return centers.tolist(), vals.tolist()


def to_jsonable(obj: Any) -> Any:
    """Recursively cast numpy arrays/scalars to plain Python lists/numbers."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        v = obj.item()
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"non-finite value in payload: {obj}")
        return v
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"non-finite float: {obj}")
        return obj
    return obj


def main() -> None:
    # Rebuild HETERO_TOY exactly as the notebook does.
    X = rng.uniform(-3, 3, N_RUNNING)
    y = f_true(X) + rng.normal(0, sigma_true(X))

    # Train M_FULL independent MLP regressors.
    print(f"Training {M_FULL} MLP members on HETERO_TOY (n={N_RUNNING})...")
    seeds = rng.integers(0, 2**31 - 1, size=M_FULL)
    members = []
    for k, s in enumerate(seeds):
        mlp = MLPRegressor(
            hidden_layer_sizes=HIDDEN,
            activation=ACTIVATION,
            solver="adam",
            learning_rate_init=0.01,
            max_iter=MAX_ITER,
            random_state=int(s),
            tol=1e-6,
            n_iter_no_change=50,
        )
        mlp.fit(X.reshape(-1, 1), y)
        members.append({
            "seed": int(s),
            # coefs_[i] has shape (h_i, h_{i+1}); each is a numpy array.
            "coefs": [W.tolist() for W in mlp.coefs_],
            "intercepts": [b.tolist() for b in mlp.intercepts_],
        })
        if (k + 1) % 10 == 0:
            print(f"  trained {k + 1}/{M_FULL} members")

    # Per-member predictions on the training set so the client can rebuild
    # the §6 aleatoric interpolator without retraining.
    preds_train = np.array([
        # Replay sklearn predict via the serialized coefs to keep the
        # NumPy-forward and sklearn-trained outputs in lockstep on the
        # client side.
        members_predict_sklearn(X, m)
        for m in members
    ])
    mean_train = preds_train.mean(axis=0)
    centers, vals = aleatoric_from_residuals(X, y, mean_train)

    payload = {
        "seed": RANDOM_SEED,
        "n_running": N_RUNNING,
        "hidden": list(HIDDEN),
        "activation": ACTIVATION,
        "M": M_FULL,
        "X": X.tolist(),
        "y": y.tolist(),
        "members": members,
        # Heteroscedastic σ²(x) interpolator: linearly interpolate (centers, vals).
        "aleatoric": {"centers": centers, "vals": vals},
    }

    out_payload = to_jsonable(payload)

    out_dirs = [
        Path(__file__).resolve().parents[2] / "public" / "sample-data" / "uncertainty-quantification",
        Path(__file__).resolve().parents[2] / "src" / "data" / "sampleData" / "uncertainty-quantification",
    ]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        out = d / "ensemble.json"
        with out.open("w") as fh:
            json.dump(out_payload, fh, allow_nan=False)
        size_kb = out.stat().st_size / 1024
        print(f"Wrote {out} ({size_kb:.1f} KB)")


def members_predict_sklearn(X: np.ndarray, member: dict) -> np.ndarray:
    """Replay the NumPy forward pass that the TS client also uses, so the
    serialized aleatoric interpolator is bit-aligned with what the viz sees."""
    h = X.reshape(-1, 1)
    coefs = [np.asarray(W) for W in member["coefs"]]
    intercepts = [np.asarray(b) for b in member["intercepts"]]
    L = len(coefs)
    for ell in range(L - 1):
        h = h @ coefs[ell] + intercepts[ell]
        h = np.tanh(h)
    h = h @ coefs[L - 1] + intercepts[L - 1]
    return h.ravel()


if __name__ == "__main__":
    main()
