"""precompute_dvg_crossover.py — Ng-Jordan discriminative-vs-generative crossover sweep.

Mirrors notebook cell 53's setup but on a richer grid (d in {10, 50, 200}, more
sample sizes, n_reps = 50) for the §7.4 DvGCrossoverPlot viz. The notebook does
d = 100 only with n_reps = 15 to stay in the <60s runtime budget; this script
is the one-shot precompute that powers the interactive viz.

Writes to BOTH:
  - src/data/sampleData/naive-bayes/dvg_crossover.json
  - public/sample-data/naive-bayes/dvg_crossover.json

Gated on file existence: re-run by deleting the JSON or passing --force.

Setup: two-class Gaussian, class +1 -> +mu, class 0 -> -mu, with mu a unit-norm
random direction shared between train and test within each replicate. Per-coord
noise is N(0, 1). Both GNB (correctly specified — diagonal cov) and unregularized
logistic regression are fit per replicate; mean test error and SE are tracked.

Runtime: ~2-3 min on a 2020-era laptop (CPU-only, vectorized NumPy via sklearn).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

SEED = 42
D_VALUES = [10, 50, 200]
N_VALUES = [20, 40, 80, 160, 320, 640, 1280, 2560, 5000]
N_REPS = 50
N_TEST = 2000

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "naive-bayes",
    REPO_ROOT / "public" / "sample-data" / "naive-bayes",
]


def ng_jordan_sample(n: int, d: int, mu: np.ndarray, rng: np.random.Generator):
    """Two-class Gaussian along shared direction mu. Class +1 -> +mu, class 0 -> -mu."""
    y = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, d)) + (2 * y - 1)[:, None] * mu
    return X, y


def evaluate(n_train: int, d: int, rng: np.random.Generator) -> dict:
    """Mean GNB and LR test error over N_REPS replicates; mu shared per replicate."""
    err_gnb, err_lr = [], []
    for _ in range(N_REPS):
        mu = rng.normal(size=d)
        mu /= np.linalg.norm(mu)
        Xtr, ytr = ng_jordan_sample(n_train, d, mu, rng)
        Xte, yte = ng_jordan_sample(N_TEST, d, mu, rng)
        if len(np.unique(ytr)) < 2:
            continue
        gnb = GaussianNB().fit(Xtr, ytr)
        err_gnb.append(1 - accuracy_score(yte, gnb.predict(Xte)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr = LogisticRegression(C=1e4, max_iter=500, solver="liblinear").fit(Xtr, ytr)
        err_lr.append(1 - accuracy_score(yte, lr.predict(Xte)))
    n_eff = max(len(err_gnb), 1)
    return {
        "n": n_train,
        "gnb_mean": float(np.mean(err_gnb)),
        "gnb_se": float(np.std(err_gnb, ddof=1) / np.sqrt(n_eff)),
        "lr_mean": float(np.mean(err_lr)),
        "lr_se": float(np.std(err_lr, ddof=1) / np.sqrt(n_eff)),
    }


def main(force: bool = False) -> None:
    target = OUT_DIRS[0] / "dvg_crossover.json"
    if target.exists() and not force:
        print(f"precompute_dvg_crossover.py: {target} exists, skipping (use --force to recompute)")
        return

    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    results: dict[str, dict[str, list[dict]]] = {"gaussianNB": {}, "logisticRegression": {}}
    for d in D_VALUES:
        print(f"  d = {d}: ", end="", flush=True)
        per_d_gnb, per_d_lr = [], []
        for n in N_VALUES:
            row = evaluate(n, d, rng)
            per_d_gnb.append({"n": row["n"], "mean_err": row["gnb_mean"], "se_err": row["gnb_se"]})
            per_d_lr.append({"n": row["n"], "mean_err": row["lr_mean"], "se_err": row["lr_se"]})
            print(f"n={n} (GNB {row['gnb_mean']:.3f}/LR {row['lr_mean']:.3f}) ", end="", flush=True)
        results["gaussianNB"][str(d)] = per_d_gnb
        results["logisticRegression"][str(d)] = per_d_lr
        print()
    elapsed = time.perf_counter() - t0

    payload = {
        "seed": SEED,
        "d_values": D_VALUES,
        "n_values": N_VALUES,
        "n_reps": N_REPS,
        "n_test": N_TEST,
        "models": ["gaussianNB", "logisticRegression"],
        "results": results,
        "wall_seconds": float(elapsed),
    }
    serialized = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "dvg_crossover.json"
        out_path.write_text(serialized)
        print(f"  wrote {out_path}")
    print(f"  total wall time: {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute even if output exists")
    args = parser.parse_args()
    main(force=args.force)
