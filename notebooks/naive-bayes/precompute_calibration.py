"""precompute_calibration.py — Raw NB scores + Platt/isotonic fit data for §8.3.

Produces the JSON payload the CalibrationReliabilityDiagram viz consumes. The
viz refits Platt and isotonic client-side from the calibration scores so the
reader can adjust bin counts and see ECE/Brier update live.

Writes to BOTH:
  - src/data/sampleData/naive-bayes/calibration_scores.json
  - public/sample-data/naive-bayes/calibration_scores.json

Held-out calibration split per CLAUDE.md "Recalibration viz: never fit-and-
evaluate on the same sample." Fit MNB on (train minus cal), score on (cal) and
(test); ship cal-scores + test-scores + true labels.

Runtime: ~5-10 seconds (single sklearn fit, no Monte Carlo).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

SEED = 42
ALPHA = 1.0
CATEGORIES = ["rec.sport.baseball", "sci.med"]
DATASET_NAME = "20newsgroups-rec.sport.baseball-vs-sci.med"

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "naive-bayes",
    REPO_ROOT / "public" / "sample-data" / "naive-bayes",
]


def main(force: bool = False) -> None:
    target = OUT_DIRS[0] / "calibration_scores.json"
    if target.exists() and not force:
        print(f"precompute_calibration.py: {target} exists, skipping (use --force to recompute)")
        return

    print(f"  loading 20-Newsgroups ({CATEGORIES})...")
    data = fetch_20newsgroups(
        subset="all",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        random_state=SEED,
    )
    Xtr_t, Xte_t, ytr, yte = train_test_split(
        data.data, data.target, test_size=0.4, random_state=SEED, stratify=data.target
    )
    vec = CountVectorizer(min_df=5, stop_words="english").fit(Xtr_t)
    Xtr = vec.transform(Xtr_t)
    Xte = vec.transform(Xte_t)
    n_train, vocab = Xtr.shape
    n_test = Xte.shape[0]
    print(f"  train: {n_train}, test: {n_test}, vocab: {vocab}")

    # Held-out calibration split: 30% of train.
    Xtr2, Xcal, ytr2, ycal = train_test_split(
        Xtr, ytr, test_size=0.3, random_state=SEED, stratify=ytr
    )
    n_train_fit, n_cal = Xtr2.shape[0], Xcal.shape[0]

    mnb = MultinomialNB(alpha=ALPHA).fit(Xtr2, ytr2)
    # log-odds scores: log P(y=1 | x) - log P(y=0 | x). Numerically stable.
    log_proba_cal = mnb.predict_log_proba(Xcal)
    log_proba_test = mnb.predict_log_proba(Xte)
    scores_cal = (log_proba_cal[:, 1] - log_proba_cal[:, 0]).astype(float)
    scores_test = (log_proba_test[:, 1] - log_proba_test[:, 0]).astype(float)
    raw_proba_test = mnb.predict_proba(Xte)[:, 1].astype(float)

    # Fit Platt (logistic on scores) once on the calibration split — the viz can
    # refit if it wants, but we ship the canonical params for sanity-checking.
    platt = LogisticRegression(C=1e10, solver="lbfgs").fit(scores_cal.reshape(-1, 1), ycal)
    # Platt's sigmoid: P(y=1 | s) = 1 / (1 + exp(A * s + B)) with A < 0, B = -A * threshold.
    A = float(-platt.coef_[0, 0])
    B = float(-platt.intercept_[0])

    iso = IsotonicRegression(out_of_bounds="clip").fit(scores_cal, ycal)
    # Sample iso at the unique calibration-set x positions for a faithful step
    # function (linear interpolation between these xy pairs in JS reproduces sklearn).
    iso_x = np.sort(np.unique(np.asarray(scores_cal, dtype=float)))
    iso_y = iso.predict(iso_x).astype(float)

    payload = {
        "seed": SEED,
        "dataset": DATASET_NAME,
        "categories": CATEGORIES,
        "alpha": ALPHA,
        "n_train_fit": int(n_train_fit),
        "n_calibration": int(n_cal),
        "n_test": int(n_test),
        "scores_cal": scores_cal.tolist(),
        "labels_cal": ycal.tolist(),
        "scores_test": scores_test.tolist(),
        "raw_proba_test": raw_proba_test.tolist(),
        "labels_test": yte.tolist(),
        "platt_params": {"A": A, "B": B},
        "isotonic_breakpoints": [
            [float(x), float(y)] for x, y in zip(iso_x.tolist(), iso_y.tolist())
        ],
    }
    serialized = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "calibration_scores.json"
        out_path.write_text(serialized)
        print(f"  wrote {out_path} ({len(serialized):,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute even if output exists")
    args = parser.parse_args()
    main(force=args.force)
