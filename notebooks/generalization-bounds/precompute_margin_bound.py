"""Precompute SVM-on-two-moons data for §10's MarginBoundDemo.

Outputs (dual-written to src/data/sampleData/generalization-bounds/ and
public/sample-data/generalization-bounds/):

  margin_bound.json: {
    train: { X: [[x1, x2], ...], y: [-1|+1, ...] },
    decision_grid: { xs, ys, Z },                  # 2D grid of decision_function values
    margins: { gamma_grid, emp_margin_loss, bound, w_norm, intercept, B,
               R_kernel, median_positive_margin, gamma_min_total, min_total },
  }

SVC fits an intercept b, so f(x) = <w, phi(x)> + b is not in F_B for the RBF
kernel alone. We absorb b into the feature map: f(x) = <(w, b), (phi(x), 1)>,
which lives in the RKHS of K + 1 with norm B = sqrt(||w||^2 + b^2) and
R = sqrt(sup_x K(x, x) + 1) = sqrt(2).

Theorem 6 holds for a gamma and B fixed before seeing the data. Here both are
read off the fitted model and gamma is scanned, so a rigorous certificate needs
a union bound over the grid. The scan is descriptive: at n = 200 the total stays
above 1 (vacuous) at every gamma.

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
    # Absorb the intercept via the augmented kernel K + 1 (see module docstring).
    intercept = float(clf.intercept_[0])
    B = float(np.hypot(w_norm, intercept))
    R_kernel = float(np.sqrt(2.0))  # sup_x K(x, x) + 1 = 2 for the RBF kernel

    # 2D decision-function grid
    xs = np.linspace(-2.0, 3.0, 100)
    ys = np.linspace(-1.5, 2.0, 70)
    XX, YY = np.meshgrid(xs, ys)
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)

    n = len(X_train)
    delta = 0.05
    # The grid extends past the minimum of the total (near gamma = 3.5), so the
    # reported minimizer is interior rather than the grid's right edge.
    gamma_grid = np.linspace(0.05, 5.0, 100)
    emp_margin_loss = np.array([margin_loss(y_train, scores_train, g) for g in gamma_grid])
    bounds = np.array([margin_bound(B, R_kernel, g, n, delta) for g in gamma_grid])

    # Where the total (emp_margin_loss + bound) is smallest on this grid. This
    # is not a certified optimal margin: gamma is chosen after seeing the data.
    total = emp_margin_loss + bounds
    i_min = int(np.argmin(total))
    gamma_min_total = float(gamma_grid[i_min])
    min_total = float(total[i_min])

    # Median positive margin y_i f(x_i) (notebook reference value)
    signed_margins = y_train * scores_train
    positive_margins = signed_margins[signed_margins > 0]
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
            "intercept": intercept,
            "B": B,
            "R_kernel": R_kernel,
            "median_positive_margin": median_margin,
            "gamma_min_total": gamma_min_total,
            "min_total": min_total,
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
    print(f"  intercept b: {intercept:.4f}")
    print(f"  B = sqrt(w_norm^2 + b^2): {B:.4f}  (R = sqrt(2))")
    print(f"  median positive margin: {median_margin:.4f}")
    print(f"  gamma minimizing the total: {gamma_min_total:.4f}  (total {min_total:.4f}, vacuous if > 1)")
    print(f"  empirical 0-1 risk: {(y_train * scores_train <= 0).mean():.4f}")


if __name__ == "__main__":
    main()
