"""Precompute SVM-on-two-moons data for §10's MarginBoundDemo.

Outputs (dual-written to src/data/sampleData/generalization-bounds/ and
public/sample-data/generalization-bounds/):

  margin_bound.json: {
    train: { X: [[x1, x2], ...], y: [-1|+1, ...] },
    decision_grid: { xs, ys, Z },                  # 2D grid of decision_function values
    margins: { gamma_grid, emp_margin_loss, bound, w_norm, R_kernel, n, delta },
  }

Runtime: ~3 s.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

SEED = 42
SLUG = "generalization-bounds"
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / SLUG,
    REPO_ROOT / "public" / "sample-data" / SLUG,
]


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def _round_floats(obj: Any, ndigits: int = 5) -> Any:
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(x, ndigits) for x in obj]
    return obj


def margin_loss(y_true: np.ndarray, scores: np.ndarray, gamma: float) -> float:
    margins = y_true * scores
    return float(np.clip(1 - margins / gamma, 0, 1).mean())


def margin_bound(B: float, R: float, gamma: float, n: int, delta: float) -> float:
    return 2 * B * R / (gamma * np.sqrt(n)) + 3 * np.sqrt(np.log(2 / delta) / (2 * n))


def main() -> None:
    X_train, y_raw = make_moons(n_samples=200, noise=0.20, random_state=SEED)
    y_train = 2 * y_raw - 1  # {0, 1} -> {-1, +1}

    # gamma is set explicitly so we can reuse the same value below — avoiding
    # the private `clf._gamma` attribute (which can change between sklearn
    # versions).  Keep the SVC `gamma=` argument and the rbf_kernel call in sync.
    rbf_gamma = 1.0
    clf = SVC(kernel="rbf", C=10.0, gamma=rbf_gamma)
    clf.fit(X_train, y_train)
    scores_train = clf.decision_function(X_train)

    # w-norm in RKHS via dual coefs.  Public API:
    # clf.dual_coef_ has shape (1, n_sv) for binary; squeeze to 1-D.
    dual_coefs = clf.dual_coef_.ravel()
    sv = clf.support_vectors_
    K_sv = rbf_kernel(sv, sv, gamma=rbf_gamma)  # (n_sv, n_sv) Gram
    w_norm = float(np.sqrt(dual_coefs @ K_sv @ dual_coefs))
    R_kernel = 1.0  # RBF kernel: K(x, x) = 1

    # 2D decision-function grid
    xs = np.linspace(-2.0, 3.0, 100)
    ys = np.linspace(-1.5, 2.0, 70)
    XX, YY = np.meshgrid(xs, ys)
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)

    n = len(X_train)
    delta = 0.05
    gamma_grid = np.linspace(0.05, 1.5, 40)
    emp_margin_loss = np.array([margin_loss(y_train, scores_train, g) for g in gamma_grid])
    bounds = np.array([margin_bound(w_norm, R_kernel, g, n, delta) for g in gamma_grid])

    # Find the optimal gamma (minimum of emp_margin_loss + bound)
    total = emp_margin_loss + bounds
    optimal_gamma = float(gamma_grid[int(np.argmin(total))])

    # Median positive margin (notebook reference value)
    positive_margins = scores_train[y_train * scores_train > 0]
    median_margin = float(np.median(positive_margins)) if positive_margins.size else 0.0

    payload = {
        "config": {"n": n, "delta": delta, "C": 10.0, "rbf_gamma": rbf_gamma, "seed": SEED},
        "train": {"X": X_train, "y": y_train.astype(int)},
        "decision_grid": {"xs": xs, "ys": ys, "Z": Z},
        "margins": {
            "gamma_grid": gamma_grid,
            "emp_margin_loss": emp_margin_loss,
            "bound": bounds,
            "w_norm": w_norm,
            "R_kernel": R_kernel,
            "median_positive_margin": median_margin,
            "optimal_gamma": optimal_gamma,
        },
    }
    payload = _round_floats(_to_jsonable(payload), ndigits=5)

    serialized = json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n"
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "margin_bound.json"
        out_path.write_text(serialized)
        size_kb = out_path.stat().st_size / 1024
        print(f"wrote {out_path}  ({size_kb:.1f} KB)")

    # Reference values printed for the verify suite
    print(f"  w_norm (RKHS): {w_norm:.4f}")
    print(f"  median positive margin: {median_margin:.4f}")
    print(f"  optimal gamma: {optimal_gamma:.4f}")
    print(f"  empirical 0-1 risk: {(y_train * scores_train <= 0).mean():.4f}")


if __name__ == "__main__":
    main()
