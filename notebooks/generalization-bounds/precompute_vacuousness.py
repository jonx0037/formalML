"""Precompute the §12 vacuousness demo: binary-MNIST MLP + classical Rademacher bound.

Outputs (dual-written):

  vacuousness.json: {
    config: { n_train, n_test, hidden_width, seed },
    results: {
      train_accuracy, test_accuracy, empirical_gap,
      W1_op_norm, w2_l2_norm, R_X,
      classical_rademacher_bound, corollary3_bound,
      ratio_bound_over_gap,
    },
    train_probs: [..n_train..],
    test_probs:  [..n_test..],
    y_train: [..n_train..],
    y_test:  [..n_test..],
    comparison: {
      threshold_class: {
        n_grid: [...], emp_gap_mean: [...], bound: [...]
      }
    }
  }

Runtime: ~10 s (with cached MNIST), up to ~60 s on first fetch_openml.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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


def fetch_binary_mnist_subset(n_train: int, n_test: int, rng: np.random.Generator):
    """Fetch MNIST and filter to digits 0 and 1, then subsample."""
    try:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    except Exception as exc:
        print(f"ERROR: failed to fetch MNIST: {exc}", file=sys.stderr)
        print("This script requires internet on first run.  After successful fetch", file=sys.stderr)
        print("sklearn caches the data under ~/scikit_learn_data/.", file=sys.stderr)
        sys.exit(1)

    X_all = mnist["data"].astype(np.float64) / 255.0
    y_all = mnist["target"].astype(int)
    mask = (y_all == 0) | (y_all == 1)
    X, y = X_all[mask], y_all[mask]
    idx = rng.choice(len(X), size=n_train + n_test, replace=False)
    X, y = X[idx], y[idx]
    return train_test_split(X, y, train_size=n_train, random_state=SEED)


def mlp_layer_norm_rademacher_bound(W1: np.ndarray, w2: np.ndarray, R_X: float, n: int, w_hidden: int) -> float:
    """Bartlett 1998 / Neyshabur et al. 2017 layer-norm-based Rademacher upper bound.

    R_n <= R_X * ||W_1||_op * ||w_2||_2 * sqrt(2 log w_hidden) / sqrt(n).
    """
    op_norm_W1 = float(np.linalg.svd(W1, compute_uv=False).max())
    l2_norm_w2 = float(np.linalg.norm(w2))
    factor_log = float(np.sqrt(2 * np.log(max(w_hidden, 2))))
    return R_X * op_norm_W1 * l2_norm_w2 * factor_log / np.sqrt(n)


def threshold_class_baseline(n_grid: list[int], delta: float = 0.05) -> dict:
    """For the §12 comparison panel: empirical gap & bound for the threshold class.

    Closed-form heuristic: emp gap ≈ 1.5/sqrt(n), Rademacher bound ≈ sqrt(log(n+1)/n).
    """
    out_emp = []
    out_bnd = []
    for n in n_grid:
        out_emp.append(float(1.5 / np.sqrt(n)))
        rad = float(np.sqrt(np.log(n + 1) / n))
        out_bnd.append(float(rad + 3 * np.sqrt(np.log(2 / delta) / (2 * n))))
    return {"n_grid": n_grid, "emp_gap_mean": out_emp, "bound": out_bnd}


def main() -> None:
    rng = np.random.default_rng(SEED)
    n_train, n_test, hidden_width = 1000, 1000, 64

    print(f"fetching binary MNIST (digits 0 vs 1), n_train={n_train}, n_test={n_test}...")
    X_train, X_test, y_train, y_test = fetch_binary_mnist_subset(n_train, n_test, rng)

    print(f"training 2-layer MLP (hidden_width={hidden_width})...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(hidden_width,),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=SEED,
        learning_rate_init=0.001,
    )
    mlp.fit(X_train, y_train)
    train_acc = float(mlp.score(X_train, y_train))
    test_acc = float(mlp.score(X_test, y_test))
    empirical_gap = train_acc - test_acc

    # Extract layer norms
    W1 = mlp.coefs_[0]  # (784, hidden)
    w2 = mlp.coefs_[1].ravel()  # (hidden,)
    R_X = float(np.sqrt(784))  # worst-case input L2 norm under [0, 1] pixel normalization
    classical_rad = mlp_layer_norm_rademacher_bound(W1.T, w2, R_X, n_train, hidden_width)
    delta = 0.05
    cor3_bound = classical_rad + 3 * np.sqrt(np.log(2 / delta) / (2 * n_train))
    ratio = cor3_bound / max(abs(empirical_gap), 1e-3)

    train_probs = mlp.predict_proba(X_train)[:, 1]
    test_probs = mlp.predict_proba(X_test)[:, 1]

    payload = {
        "config": {"n_train": n_train, "n_test": n_test, "hidden_width": hidden_width, "seed": SEED, "delta": delta},
        "results": {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "empirical_gap": empirical_gap,
            "W1_op_norm": float(np.linalg.svd(W1, compute_uv=False).max()),
            "w2_l2_norm": float(np.linalg.norm(w2)),
            "R_X": R_X,
            "classical_rademacher_bound": classical_rad,
            "corollary3_bound": cor3_bound,
            "ratio_bound_over_gap": ratio,
        },
        "train_probs": train_probs,
        "test_probs": test_probs,
        "y_train": y_train.astype(int),
        "y_test": y_test.astype(int),
        "comparison": {"threshold_class": threshold_class_baseline([30, 100, 300, 1000, 3000])},
    }
    payload = _round_floats(_to_jsonable(payload), ndigits=5)

    serialized = json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n"
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "vacuousness.json"
        out_path.write_text(serialized)
        size_kb = out_path.stat().st_size / 1024
        print(f"wrote {out_path}  ({size_kb:.1f} KB)")

    print(f"  train accuracy: {train_acc:.4f}")
    print(f"  test  accuracy: {test_acc:.4f}")
    print(f"  empirical gap:  {empirical_gap:.4f}")
    print(f"  ||W1||_op:      {payload['results']['W1_op_norm']:.4f}")
    print(f"  ||w2||_2:       {payload['results']['w2_l2_norm']:.4f}")
    print(f"  classical Rademacher upper bound: {classical_rad:.2f}")
    print(f"  Corollary 3 bound at delta=0.05:  {cor3_bound:.2f}")
    print(f"  ratio bound / gap:                {ratio:.0f}x")


if __name__ == "__main__":
    main()
