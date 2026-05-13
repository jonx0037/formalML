"""Precompute the §6.5 FTSL bound vs empirical generalization gap.

Outputs (dual-written):
  ftsl_envelope.json: {
    config: { seed, n_test, replicates, ns, d, delta },
    ftsl_curves: { d=1: [...], d=3: [...], d=5: [...], d=10: [...] },
    empirical_half_planes: [{ n, gap_mean, gap_std, gap_samples: [...] }, ...]
  }

Runtime: ~15 s on a 2020-era laptop.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.svm import LinearSVC

SEED = 20260513
SLUG = "vc-dimension"
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / SLUG,
    REPO_ROOT / "public" / "sample-data" / SLUG,
]


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"non-finite float encountered: {obj}")
        return obj
    return obj


def ftsl_bound(n: int, d: int, delta: float) -> float:
    """FTSL upper bound from eq. (6.2)."""
    return math.sqrt(8 * (d * math.log(2 * math.e * n / d) + math.log(4 / delta)) / n)


def sample_half_plane(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample n points from U[0,1]^2 with labels y = 1[x2 > x1 + 0.2]."""
    X = rng.uniform(0.0, 1.0, size=(n, 2))
    y = (X[:, 1] > X[:, 0] + 0.2).astype(np.int8)
    return X, y


def main() -> None:
    rng = np.random.default_rng(SEED)
    ns = [30, 60, 120, 250, 500, 1000]
    d = 3
    delta = 0.05
    replicates = 15
    n_test = 20_000

    # FTSL closed-form curves for d in {1, 3, 5, 10}.
    n_grid = list(range(20, 1001, 10))
    ftsl_curves: dict[str, list[dict[str, float]]] = {}
    for dd in (1, 3, 5, 10):
        ftsl_curves[f"d={dd}"] = [{"n": n, "epsilon": ftsl_bound(n, dd, delta)} for n in n_grid]

    # Empirical Monte Carlo on half-planes.
    X_test, y_test = sample_half_plane(rng, n_test)
    empirical: list[dict[str, Any]] = []
    for n in ns:
        gaps: list[float] = []
        bound = ftsl_bound(n, d, delta)
        for _rep in range(replicates):
            X_tr, y_tr = sample_half_plane(rng, n)
            # ERM via LinearSVC at high C (approx-realizable: data has zero noise).
            if len(np.unique(y_tr)) < 2:
                # Degenerate sample (all one class) — skip.
                continue
            clf = LinearSVC(C=1e6, dual=False, max_iter=10_000)
            clf.fit(X_tr, y_tr)
            train_err = float(np.mean(clf.predict(X_tr) != y_tr))
            test_err = float(np.mean(clf.predict(X_test) != y_test))
            gaps.append(abs(test_err - train_err))
        gaps_arr = np.array(gaps)
        empirical.append(
            {
                "n": n,
                "bound": bound,
                "gap_mean": float(gaps_arr.mean()),
                "gap_std": float(gaps_arr.std(ddof=1) if len(gaps_arr) > 1 else 0.0),
                "gap_samples": gaps,
            }
        )

    payload = {
        "config": {
            "seed": SEED,
            "n_test": n_test,
            "replicates": replicates,
            "ns": ns,
            "d_VC_half_plane": d,
            "delta": delta,
        },
        "ftsl_curves": ftsl_curves,
        "empirical_half_planes": empirical,
    }

    payload = _to_jsonable(payload)
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "ftsl_envelope.json"
        out_file.write_text(json.dumps(payload, allow_nan=False, indent=2))
        print(f"  wrote {out_file}")

    print("\n=== Summary ===")
    print(f"  FTSL curves: d in {{1, 3, 5, 10}}, n in [{min(n_grid)}, {max(n_grid)}]")
    print(f"  Empirical: {len(empirical)} sample sizes, {replicates} replicates each")
    print(f"  Sample value: n=30 bound={empirical[0]['bound']:.3f} emp_mean={empirical[0]['gap_mean']:.3f}")
    print(f"               n=1000 bound={empirical[-1]['bound']:.3f} emp_mean={empirical[-1]['gap_mean']:.3f}")


if __name__ == "__main__":
    main()
