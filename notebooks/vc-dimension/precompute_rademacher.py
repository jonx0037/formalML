"""Precompute the §9.5 empirical Rademacher complexity vs Massart/Sauer–Shelah bounds.

Outputs (dual-written):
  rademacher_axis_rectangles.json: {
    config: { seed, ns, d_VC, B_rademacher, n_samples_per_n },
    rows: [{ n, restricted_size, empirical, empirical_std, massart, ss_bound }, ...]
  }

Notebook cell 75 prints the headline values; we reproduce them with the same seed
and parameters.

Runtime: ~6 s.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

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
            raise ValueError(f"non-finite float in payload: {obj}")
        return obj
    return obj


def enumerate_rectangle_dichotomies(points: np.ndarray) -> list[np.ndarray]:
    """Brute-force enumerate H|_S for axis-aligned rectangles.

    For each binary labeling, the positive class's bounding box is the
    minimum-enclosing rectangle. The labeling is realized iff that
    bounding box contains no negative-labeled point.
    """
    n = len(points)
    realized: list[np.ndarray] = []
    for mask in range(1 << n):
        labels = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.int8)
        pos = points[labels == 1]
        neg = points[labels == 0]
        if len(pos) == 0:
            realized.append(labels)
            continue
        x_lo, y_lo = pos.min(axis=0)
        x_hi, y_hi = pos.max(axis=0)
        if len(neg) == 0:
            realized.append(labels)
            continue
        in_box = (neg[:, 0] >= x_lo) & (neg[:, 0] <= x_hi) & (neg[:, 1] >= y_lo) & (neg[:, 1] <= y_hi)
        if not in_box.any():
            realized.append(labels)
    return realized


def empirical_rademacher(realized_labelings: list[np.ndarray], B: int, rng: np.random.Generator) -> tuple[float, float]:
    """Compute empirical Rademacher complexity by averaging sup over B random sign vectors."""
    n = len(realized_labelings[0])
    # Center labels to {-1, +1} for the sup; H|_S contains 0/1 vectors, so center to (2*labels - 1)/2.
    # Actually the empirical Rademacher uses h(x_i) directly, so labels are {0, 1}.
    H = np.array(realized_labelings)  # (M, n)
    sigmas = rng.choice([-1, 1], size=(B, n)).astype(np.float64)
    # sup_h (1/n) sum_i sigma_i h(x_i)
    dot = sigmas @ H.T / n  # (B, M)
    sup = dot.max(axis=1)
    return float(sup.mean()), float(sup.std(ddof=1))


def main() -> None:
    rng = np.random.default_rng(SEED)
    ns = [5, 8, 12, 16, 20]
    d_vc = 4
    B = 2000

    rows: list[dict[str, Any]] = []
    for n in ns:
        # Sample n points from U[0, 1]^2.
        points = rng.uniform(0, 1, size=(n, 2))
        realized = enumerate_rectangle_dichotomies(points)
        m = len(realized)
        emp_mean, emp_std = empirical_rademacher(realized, B, rng)
        massart = math.sqrt(2 * math.log(m) / n) if m > 1 else 0.0
        ss_bound = math.sqrt(2 * d_vc * math.log(math.e * n / d_vc) / n)
        rows.append({
            "n": n,
            "restricted_size": m,
            "empirical": emp_mean,
            "empirical_std": emp_std,
            "massart": massart,
            "ss_bound": ss_bound,
        })
        print(f"  n={n:>3d}  |H|_S|={m:>6d}  emp={emp_mean:.4f}  Massart={massart:.4f}  SS={ss_bound:.4f}")

    payload = {
        "config": {
            "seed": SEED,
            "ns": ns,
            "d_VC": d_vc,
            "B_rademacher": B,
            "domain": "[0, 1]^2 uniform",
            "class": "axis-aligned rectangles",
        },
        "rows": rows,
    }

    payload = _to_jsonable(payload)
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "rademacher_axis_rectangles.json"
        out_file.write_text(json.dumps(payload, allow_nan=False, indent=2))
        print(f"  wrote {out_file}")


if __name__ == "__main__":
    main()
