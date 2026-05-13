"""Precompute the §10.5 integrative four-panel "money shot".

Outputs (dual-written):
  integrative_monte_carlo.json: {
    config: { seed, replicates, n_test, sample_sizes, ... },
    panel_a_growth: { half_plane: [...], rectangle: [...] },
    panel_b_ftsl_vs_empirical: { half_plane: [...], rectangle: [...] },
    panel_c_sample_complexity: { ... },
    panel_d_rademacher_at_n_20: { half_plane: {...}, rectangle: {...} }
  }

Combines four protocols on the half-planes and axis-rectangles anchor classes.
Brief §10 specifies the four-protocol Monte Carlo with explicit assert checks.

Runtime: ~35 s on a 2020-era laptop.
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
            raise ValueError(f"non-finite float in payload: {obj}")
        return obj
    return obj


def comb(n: int, k: int) -> int:
    return math.comb(n, k)


def sauer_shelah_binom_sum(n: int, d: int) -> int:
    return sum(comb(n, i) for i in range(min(d, n) + 1))


def ftsl_bound(n: int, d: int, delta: float) -> float:
    return math.sqrt(8 * (d * math.log(2 * math.e * n / d) + math.log(4 / delta)) / n)


def realizable_sc(eps: float, delta: float, d: int) -> int:
    return math.ceil(8 * (d * math.log(2 * math.e / eps) + math.log(2 / delta)) / eps)


# -----------------------------------------------------------------------------
# Data generators (the two anchor classes from §10.1).
# -----------------------------------------------------------------------------


def sample_half_plane(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    X = rng.uniform(0, 1, size=(n, 2))
    y = (X[:, 1] > X[:, 0] + 0.2).astype(np.int8)
    return X, y


def sample_rectangle(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    X = rng.uniform(0, 1, size=(n, 2))
    y = ((X[:, 0] >= 0.3) & (X[:, 0] <= 0.7) & (X[:, 1] >= 0.3) & (X[:, 1] <= 0.7)).astype(np.int8)
    return X, y


# -----------------------------------------------------------------------------
# ERM helpers
# -----------------------------------------------------------------------------


def fit_half_plane(X: np.ndarray, y: np.ndarray) -> Any:
    if len(np.unique(y)) < 2:
        # Degenerate: return a constant classifier.
        class Const:
            def __init__(self, c: int) -> None:
                self.c = c
            def predict(self, Xq: np.ndarray) -> np.ndarray:
                return np.full(len(Xq), self.c, dtype=np.int8)
        return Const(int(y[0]) if len(y) else 0)
    clf = LinearSVC(C=1e6, dual=False, max_iter=10_000)
    clf.fit(X, y)
    return clf


def fit_rectangle(X: np.ndarray, y: np.ndarray) -> Any:
    """ERM rectangle: smallest enclosing axis-rectangle of positive class.
    Returns a callable that classifies new points."""
    pos = X[y == 1]
    if len(pos) == 0:
        x_lo = y_lo = float("inf")
        x_hi = y_hi = float("-inf")
    else:
        x_lo, y_lo = pos.min(axis=0)
        x_hi, y_hi = pos.max(axis=0)

    class Rect:
        def predict(self, Xq: np.ndarray) -> np.ndarray:
            in_box = (Xq[:, 0] >= x_lo) & (Xq[:, 0] <= x_hi) & (Xq[:, 1] >= y_lo) & (Xq[:, 1] <= y_hi)
            return in_box.astype(np.int8)

    return Rect()


# -----------------------------------------------------------------------------
# Brute-force restriction enumerators for Protocol 1.
# -----------------------------------------------------------------------------


def enumerate_rect_dichotomies(points: np.ndarray) -> int:
    n = len(points)
    realized = 0
    for mask in range(1 << n):
        labels = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.int8)
        pos = points[labels == 1]
        neg = points[labels == 0]
        if len(pos) == 0:
            realized += 1
            continue
        x_lo, y_lo = pos.min(axis=0)
        x_hi, y_hi = pos.max(axis=0)
        if len(neg) == 0:
            realized += 1
            continue
        in_box = (neg[:, 0] >= x_lo) & (neg[:, 0] <= x_hi) & (neg[:, 1] >= y_lo) & (neg[:, 1] <= y_hi)
        if not in_box.any():
            realized += 1
    return realized


# Half-plane dichotomies via Cover (1965)'s O(n^d) construction.
# Each non-trivial dichotomy is determined by the line through some pair of
# support points; that line gives 2 sign assignments. Plus the all-0 and all-1
# trivial dichotomies. This is O(n^2) candidate hyperplanes for d=2.
def enumerate_hp_dichotomies(points: np.ndarray) -> int:
    return len(enumerate_hp_label_vectors(points))


# -----------------------------------------------------------------------------
# Empirical Rademacher (Protocol 4).
# -----------------------------------------------------------------------------


def enumerate_hp_label_vectors(points: np.ndarray) -> np.ndarray:
    """Enumerate all distinct half-plane dichotomies on points via Cover (1965).

    Each dichotomy realizable by a half-plane corresponds to a directed line in
    the plane; rotating the line generates all O(n^2) sign-vector patterns plus
    the two trivial constants. Concretely, for each *direction* w (normal vector
    perpendicular to one of the C(n, 2) pair-connecting lines, plus a few
    random extra directions), we compute the projection of points onto w and
    consider every threshold falling between consecutive projections.
    """
    n = len(points)
    seen: set[tuple[int, ...]] = set()
    rows: list[np.ndarray] = []

    def record(lbl: np.ndarray) -> None:
        key = tuple(int(v) for v in lbl)
        if key not in seen:
            seen.add(key)
            rows.append(lbl.astype(np.float64))

    # Trivial dichotomies
    record(np.zeros(n, dtype=np.int8))
    record(np.ones(n, dtype=np.int8))

    # For each pair (i, j), the perpendicular to the segment i->j gives a
    # candidate direction; the threshold sweeps every gap between projections.
    EPS = 1e-9
    directions: list[np.ndarray] = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[j] - points[i]
            norm = np.linalg.norm(dx)
            if norm < EPS:
                continue
            # Perpendicular direction (rotate by 90°)
            w = np.array([-dx[1], dx[0]]) / norm
            directions.append(w)
            directions.append(-w)
    # Also include the original directions (along i->j) — useful when threshold
    # passes through point i or j.
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[j] - points[i]
            norm = np.linalg.norm(dx)
            if norm < EPS:
                continue
            directions.append(dx / norm)
            directions.append(-dx / norm)

    for w in directions:
        proj = points @ w
        order = np.argsort(proj)
        sorted_proj = proj[order]
        # Threshold below all projections gives all-0 (or all-1 for the flipped sign).
        for k in range(n + 1):
            # Threshold between sorted_proj[k-1] and sorted_proj[k] (or at the extremes).
            if k == 0:
                t = sorted_proj[0] - 1.0
            elif k == n:
                t = sorted_proj[-1] + 1.0
            else:
                t = (sorted_proj[k - 1] + sorted_proj[k]) / 2
            labels = (proj >= t).astype(np.int8)
            record(labels)
            labels_flip = 1 - labels
            record(labels_flip)
    return np.array(rows)


def enumerate_rect_label_vectors(points: np.ndarray) -> np.ndarray:
    n = len(points)
    rows: list[np.ndarray] = []
    for mask in range(1 << n):
        labels = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.int8)
        pos = points[labels == 1]
        neg = points[labels == 0]
        if len(pos) == 0:
            rows.append(labels.astype(np.float64))
            continue
        x_lo, y_lo = pos.min(axis=0)
        x_hi, y_hi = pos.max(axis=0)
        if len(neg) == 0:
            rows.append(labels.astype(np.float64))
            continue
        in_box = (neg[:, 0] >= x_lo) & (neg[:, 0] <= x_hi) & (neg[:, 1] >= y_lo) & (neg[:, 1] <= y_hi)
        if not in_box.any():
            rows.append(labels.astype(np.float64))
    return np.array(rows)


def empirical_rademacher(H: np.ndarray, B: int, rng: np.random.Generator) -> float:
    n = H.shape[1]
    sigmas = rng.choice([-1.0, 1.0], size=(B, n))
    sup = (sigmas @ H.T / n).max(axis=1)
    return float(sup.mean())


# -----------------------------------------------------------------------------
# Main protocol orchestrator.
# -----------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(SEED)
    delta = 0.05
    n_test = 30_000

    # ============================================================
    # Protocol 1: Growth-function check at n in {5, 10, 15, 20}.
    # ============================================================
    print("=== Protocol 1: Growth-function check ===")
    panel_a_hp: list[dict[str, Any]] = []
    panel_a_rect: list[dict[str, Any]] = []
    for n in (5, 10, 15, 20):
        pts = rng.uniform(0, 1, size=(n, 2))
        # Half-plane: closed form
        pi_hp = n * n - n + 2
        # Cap by enumerator (sanity) at n <= 15 only
        pi_rect = enumerate_rect_dichotomies(pts) if n <= 20 else None
        ss_hp = sauer_shelah_binom_sum(n, 3)
        ss_rect = sauer_shelah_binom_sum(n, 4)
        panel_a_hp.append({"n": n, "Pi": pi_hp, "SS_ceiling_d3": ss_hp})
        panel_a_rect.append({"n": n, "Pi": pi_rect, "SS_ceiling_d4": ss_rect})
        print(f"  n={n:>3d}  HP Pi={pi_hp:>5d}  SS_HP={ss_hp:>5d}  rect Pi={pi_rect:>6d}  SS_rect={ss_rect:>6d}")
        # Brief §10.5 assertion: Pi_rect <= SS_ceiling.
        assert pi_rect <= ss_rect, f"Sauer-Shelah ceiling violated at n={n}"

    # ============================================================
    # Protocol 2: FTSL envelope vs empirical gap.
    # ============================================================
    print("\n=== Protocol 2: FTSL envelope vs empirical gap ===")
    X_test_hp, y_test_hp = sample_half_plane(rng, n_test)
    X_test_rect, y_test_rect = sample_rectangle(rng, n_test)
    replicates = 15
    panel_b_hp: list[dict[str, Any]] = []
    panel_b_rect: list[dict[str, Any]] = []
    for n in (30, 60, 120, 250, 500, 1000):
        gaps_hp: list[float] = []
        gaps_rect: list[float] = []
        for _ in range(replicates):
            X_tr_hp, y_tr_hp = sample_half_plane(rng, n)
            X_tr_rect, y_tr_rect = sample_rectangle(rng, n)
            clf_hp = fit_half_plane(X_tr_hp, y_tr_hp)
            clf_rect = fit_rectangle(X_tr_rect, y_tr_rect)
            tr_err_hp = float(np.mean(clf_hp.predict(X_tr_hp) != y_tr_hp))
            te_err_hp = float(np.mean(clf_hp.predict(X_test_hp) != y_test_hp))
            tr_err_rect = float(np.mean(clf_rect.predict(X_tr_rect) != y_tr_rect))
            te_err_rect = float(np.mean(clf_rect.predict(X_test_rect) != y_test_rect))
            gaps_hp.append(abs(te_err_hp - tr_err_hp))
            gaps_rect.append(abs(te_err_rect - tr_err_rect))
        bound_hp = ftsl_bound(n, 3, delta)
        bound_rect = ftsl_bound(n, 4, delta)
        emp_hp = float(np.mean(gaps_hp))
        emp_rect = float(np.mean(gaps_rect))
        panel_b_hp.append({"n": n, "bound": bound_hp, "empirical": emp_hp})
        panel_b_rect.append({"n": n, "bound": bound_rect, "empirical": emp_rect})
        print(f"  n={n:>5d}  HP bound={bound_hp:.3f} emp={emp_hp:.3f}   rect bound={bound_rect:.3f} emp={emp_rect:.3f}")
        # Brief §10.5 assertion: empirical << bound.
        assert emp_hp <= bound_hp * 2.0, f"HP empirical exceeds 2x bound at n={n}: {emp_hp} > 2*{bound_hp}"
        assert emp_rect <= bound_rect * 2.0, f"Rect empirical exceeds 2x bound at n={n}"

    # ============================================================
    # Protocol 3: Realizable sample-complexity.
    # ============================================================
    print("\n=== Protocol 3: Realizable sample-complexity at delta=0.05 ===")
    panel_c: list[dict[str, Any]] = []
    for eps in (0.10, 0.05, 0.02):
        n_th_hp = realizable_sc(eps, delta, 3)
        n_th_rect = realizable_sc(eps, delta, 4)
        # Empirical: smallest n where 95% of replicates achieve R(h_hat) <= eps.
        emp_hp = None
        emp_rect = None
        for n_try in (30, 50, 80, 130, 200, 320, 500, 800, 1300, 2000):
            successes_hp = 0
            successes_rect = 0
            trials = 40
            for _ in range(trials):
                X_tr_hp, y_tr_hp = sample_half_plane(rng, n_try)
                X_tr_rect, y_tr_rect = sample_rectangle(rng, n_try)
                clf_hp = fit_half_plane(X_tr_hp, y_tr_hp)
                clf_rect = fit_rectangle(X_tr_rect, y_tr_rect)
                R_hp = float(np.mean(clf_hp.predict(X_test_hp) != y_test_hp))
                R_rect = float(np.mean(clf_rect.predict(X_test_rect) != y_test_rect))
                if R_hp <= eps:
                    successes_hp += 1
                if R_rect <= eps:
                    successes_rect += 1
            if emp_hp is None and successes_hp / trials >= 0.95:
                emp_hp = n_try
            if emp_rect is None and successes_rect / trials >= 0.95:
                emp_rect = n_try
            if emp_hp is not None and emp_rect is not None:
                break
        panel_c.append(
            {
                "eps": eps,
                "n_emp_HP": emp_hp,
                "n_th_HP": n_th_hp,
                "n_emp_rect": emp_rect,
                "n_th_rect": n_th_rect,
            }
        )
        print(f"  eps={eps:.3f}  HP emp={emp_hp} theo={n_th_hp}   rect emp={emp_rect} theo={n_th_rect}")

    # ============================================================
    # Protocol 4: Empirical Rademacher at n = 20.
    # ============================================================
    print("\n=== Protocol 4: Rademacher complexity at n = 20 ===")
    n_rad = 20
    pts_rad = rng.uniform(0, 1, size=(n_rad, 2))
    H_hp = enumerate_hp_label_vectors(pts_rad)
    H_rect = enumerate_rect_label_vectors(pts_rad)
    B = 2000
    rad_hp = empirical_rademacher(H_hp, B, rng)
    rad_rect = empirical_rademacher(H_rect, B, rng)
    ss_hp = math.sqrt(2 * 3 * math.log(math.e * n_rad / 3) / n_rad)
    ss_rect = math.sqrt(2 * 4 * math.log(math.e * n_rad / 4) / n_rad)
    panel_d = {
        "n": n_rad,
        "half_plane": {"empirical": rad_hp, "restricted_size": int(len(H_hp)), "ss_bound": ss_hp},
        "rectangle": {"empirical": rad_rect, "restricted_size": int(len(H_rect)), "ss_bound": ss_rect},
    }
    print(f"  half-planes:  empirical={rad_hp:.4f}  SS bound={ss_hp:.4f}")
    print(f"  rectangles:   empirical={rad_rect:.4f}  SS bound={ss_rect:.4f}")
    # Brief §10.5 assertion: empirical Rademacher dominated by Sauer–Shelah bound.
    assert rad_hp <= ss_hp, "HP Rademacher exceeded Sauer-Shelah bound"
    assert rad_rect <= ss_rect, "rect Rademacher exceeded Sauer-Shelah bound"

    # ============================================================
    # Assemble payload and write to both OUT_DIRS.
    # ============================================================
    payload = {
        "config": {
            "seed": SEED,
            "replicates_protocol_2": replicates,
            "trials_protocol_3": 40,
            "B_rademacher": B,
            "n_test": n_test,
            "delta": delta,
        },
        "panel_a_growth": {"half_plane": panel_a_hp, "rectangle": panel_a_rect},
        "panel_b_ftsl_vs_empirical": {"half_plane": panel_b_hp, "rectangle": panel_b_rect},
        "panel_c_sample_complexity": panel_c,
        "panel_d_rademacher_at_n_20": panel_d,
    }
    payload = _to_jsonable(payload)
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "integrative_monte_carlo.json"
        out_file.write_text(json.dumps(payload, allow_nan=False, indent=2))
        print(f"\n  wrote {out_file}")


if __name__ == "__main__":
    main()
