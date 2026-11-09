"""Precompute pre-trained MLPRegressor coefficients for the §7/§9/§11
interactive viz on the live site.

The notebook is read-only; this side script rebuilds the §2 HETERO_TOY data
(seed-aligned with the notebook) and the §7.4 deep ensemble at M=50, then
serializes per-member ``coefs_`` / ``intercepts_`` plus the aleatoric
heteroscedastic interpolator to JSON.

It also builds the §8 conformal payload (``conformal_ensemble.json``) for
``ConformalUQCalibrator``. Split conformal is only valid when the calibration
points are exchangeable with test points, so they must not have been used to
train the base predictor. Mirroring notebook cell 43, HETERO_TOY is split into a
100-point train fold and a 50-point calibration fold. A fresh 20-member
ensemble, plus its aleatoric interpolator, is fit on the train fold only, and
the calibration fold is shipped alongside it. (The full ensemble above is
trained on all 200 points, so calibrating on any of those points would shrink
q̂ and under-cover.)

Output: each payload is written to BOTH
  * ``public/sample-data/uncertainty-quantification/<name>.json`` (Astro
    serves this at /sample-data/uncertainty-quantification/<name>.json)
  * ``src/data/sampleData/uncertainty-quantification/<name>.json`` (build
    target — tracked but not served)

Run (``--only`` regenerates one payload and leaves the other file untouched):
    cd notebooks/uncertainty-quantification && .venv/bin/python precompute_ensemble.py [--only full|conformal]
"""
from __future__ import annotations

import argparse
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

# §8 conformal fold (notebook cell 43): 100 train / 50 calibration, M = 20.
# The split uses its own generator so the full-ensemble stream above is
# unchanged by this payload.
CONFORMAL_SPLIT_SEED = RANDOM_SEED + 43
N_TR_CONF = 100
N_CAL_CONF = 50
M_CONF = 20

OUT_DIRS = [
    Path(__file__).resolve().parents[2] / "public" / "sample-data" / "uncertainty-quantification",
    Path(__file__).resolve().parents[2] / "src" / "data" / "sampleData" / "uncertainty-quantification",
]


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


def fit_members(X: np.ndarray, y: np.ndarray, seeds: np.ndarray, label: str) -> list[dict]:
    """Train one MLP per seed on (X, y) and serialize its weights."""
    print(f"Training {len(seeds)} MLP members on {label} (n={len(X)})...")
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
            print(f"  trained {k + 1}/{len(seeds)} members")
    return members


def ensemble_aleatoric(X: np.ndarray, y: np.ndarray, members: list[dict]):
    """Aleatoric bins from the ensemble-mean residuals on (X, y), using the
    NumPy forward pass the TS client also uses."""
    preds = np.array([members_predict_sklearn(X, m) for m in members])
    return aleatoric_from_residuals(X, y, preds.mean(axis=0))


def write_payload(name: str, payload: dict) -> None:
    out_payload = to_jsonable(payload)
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        out = d / f"{name}.json"
        with out.open("w") as fh:
            json.dump(out_payload, fh, allow_nan=False)
        size_kb = out.stat().st_size / 1024
        print(f"Wrote {out} ({size_kb:.1f} KB)")


def build_full_payload(X: np.ndarray, y: np.ndarray) -> dict:
    """§7/§9/§11 payload: M_FULL members trained on all of HETERO_TOY."""
    seeds = rng.integers(0, 2**31 - 1, size=M_FULL)
    members = fit_members(X, y, seeds, "HETERO_TOY")
    # Aleatoric interpolator from the training-set residuals, so the client
    # can rebuild the §6 interpolator without retraining.
    centers, vals = ensemble_aleatoric(X, y, members)
    return {
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


def build_conformal_payload(X: np.ndarray, y: np.ndarray) -> dict:
    """§8 payload: M_CONF members + aleatoric bins fit on the train fold only;
    the calibration fold is disjoint from everything the predictor saw."""
    split_rng = np.random.default_rng(CONFORMAL_SPLIT_SEED)
    perm = split_rng.permutation(N_RUNNING)
    train_idx = np.sort(perm[:N_TR_CONF])
    cal_idx = np.sort(perm[N_TR_CONF:N_TR_CONF + N_CAL_CONF])
    assert not set(train_idx.tolist()) & set(cal_idx.tolist())
    X_tr, y_tr = X[train_idx], y[train_idx]
    seeds = split_rng.integers(0, 2**31 - 1, size=M_CONF)
    members = fit_members(X_tr, y_tr, seeds, "HETERO_TOY train fold")
    centers, vals = ensemble_aleatoric(X_tr, y_tr, members)
    return {
        "seed": RANDOM_SEED,
        "split_seed": CONFORMAL_SPLIT_SEED,
        "n_running": N_RUNNING,
        "hidden": list(HIDDEN),
        "activation": ACTIVATION,
        "M": M_CONF,
        # Indices into HETERO_TOY (ensemble.json X / y), kept for provenance.
        "train_idx": train_idx.tolist(),
        "cal_idx": cal_idx.tolist(),
        "X_train": X_tr.tolist(),
        "y_train": y_tr.tolist(),
        "X_cal": X[cal_idx].tolist(),
        "y_cal": y[cal_idx].tolist(),
        "members": members,
        "aleatoric": {"centers": centers, "vals": vals},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", choices=["full", "conformal"],
                        help="regenerate a single payload (default: both)")
    args = parser.parse_args()

    # Rebuild HETERO_TOY exactly as the notebook does.
    X = rng.uniform(-3, 3, N_RUNNING)
    y = f_true(X) + rng.normal(0, sigma_true(X))

    if args.only in (None, "full"):
        write_payload("ensemble", build_full_payload(X, y))
    if args.only in (None, "conformal"):
        write_payload("conformal_ensemble", build_conformal_payload(X, y))


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
