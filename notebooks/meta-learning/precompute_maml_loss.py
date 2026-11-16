"""
precompute_maml_loss.py — MAML meta-training loss curves + the
adaptation-evolution curve for a held-out sinusoidal task, at the notebook's
default hyperparameters (alpha=0.01, N=5, B=4) AND at a small grid of
alternative (alpha, N) settings so the MamlMetaLossExplorer slider can select
among them.

Executes the notebook's code cells 0-14 via _notebook_exec.py, snapshots the
module-level `rng` exactly as cell 15 finds it, and then runs cell 15's own
source once per (alpha, N) with only `INNER_LR` and `N_INNER` substituted,
restoring that rng snapshot before each run. Every grid point therefore trains
on the notebook's task stream, and the default (0.01, 5) curve is the
notebook's run itself (meta-loss 4.456 / 5.462 / 4.152 at iters 0 / 50 / 99).
The adaptation evolution is cell 17's `predict_after_n_steps` on that default
θ₀, so it reproduces Figure 6. An earlier version trained every configuration
from a fresh `default_rng(SEED)`, so the explorer showed a separate run.

Writes to BOTH:
  - src/data/sampleData/meta-learning/maml_loss.json
  - public/sample-data/meta-learning/maml_loss.json

Output schema:
{
  "alphaGrid": [0.005, 0.01, 0.02],
  "NGrid":     [3, 5, 8],
  "curves": [
    { "alpha": 0.005, "N": 3, "metaLoss": [...100 floats], "lossAtEnd": float },
    ...
  ],
  "defaultIndex": index into curves with the notebook's (alpha=0.01, N=5),
  "adaptationXSupport": [...5 floats],
  "adaptationYSupport": [...5 floats],
  "adaptationXDense":   [...200 floats],
  "adaptationYTrue":    [...200 floats],
  "adaptationPredsByStep": { "0": [...200], "1": [...200], "3": [...200], "5": [...200] }
}

Runs in ~30 s on CPU (9 meta-training runs of 100 iters each).
"""

import sys

import numpy as np

from _notebook_exec import code_cells, floats, run_cell, run_notebook_through, write_json

TRAIN_CELL = 15
EVAL_CELL = 17
ALPHA_GRID = [0.005, 0.01, 0.02]
N_GRID = [3, 5, 8]
DEFAULT = (0.01, 5)


def main():
    cells = code_cells()
    train_src = cells[TRAIN_CELL]
    for line in ("INNER_LR = 0.01\n", "N_INNER = 5\n"):
        if train_src.count(line) != 1:
            sys.exit(f"cell {TRAIN_CELL} no longer defines {line.strip()!r} exactly once")

    print(f"Executing notebook cells 0-{TRAIN_CELL - 1}…", file=sys.stderr)
    ns = run_notebook_through(TRAIN_CELL - 1)
    rng_state = ns["rng"].bit_generator.state

    curves, default_index, adaptation = [], None, None
    for a in ALPHA_GRID:
        for N in N_GRID:
            src = (train_src.replace("INNER_LR = 0.01\n", f"INNER_LR = {a!r}\n")
                            .replace("N_INNER = 5\n", f"N_INNER = {N}\n"))
            ns["rng"] = np.random.default_rng()
            ns["rng"].bit_generator.state = rng_state
            run_cell(ns, TRAIN_CELL, src)
            losses = ns["meta_losses"]
            print(f"  alpha={a}, N={N}: {losses[0]:.3f} -> {losses[-1]:.3f}",
                  file=sys.stderr)
            curves.append({
                "alpha": float(a),
                "N": int(N),
                "metaLoss": floats(losses),
                "lossAtEnd": float(losses[-1]),
            })
            if (a, N) == DEFAULT:
                default_index = len(curves) - 1
                run_cell(ns, EVAL_CELL, cells[EVAL_CELL])
                preds = {str(n): floats(ns["predict_after_n_steps"](
                    ns["meta_params"], ns["x_dense_t"], n_steps=n))
                    for n in (0, 1, 3, 5)}
                adaptation = {
                    "adaptationXSupport": floats(ns["x_s"]),
                    "adaptationYSupport": floats(ns["y_s"]),
                    "adaptationXDense": floats(ns["x_dense"]),
                    "adaptationYTrue": floats(ns["y_dense"]),
                    "adaptationPredsByStep": preds,
                }

    payload = {
        "alphaGrid": ALPHA_GRID,
        "NGrid": N_GRID,
        "curves": curves,
        "defaultIndex": default_index,
        **adaptation,
    }
    write_json("maml_loss.json", payload)


if __name__ == "__main__":
    main()
