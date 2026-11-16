"""
precompute_protonet_training.py — ProtoNet training curves + held-out decision
regions, from cells 44-46 of the notebook (Figures 10-11).

Executes the notebook's code cells 0-44 in order via _notebook_exec.py, so the
ProtoNet is the one trained in cell 44 on the module-level `rng` after the
earlier cells have advanced it. The two held-out tasks are then recomputed
exactly as cell 46 does (its own `default_rng(SEED + 100)`, no other RNG). An
earlier version trained from a fresh `default_rng(SEED)` and did not match the
notebook's run (first-iteration loss 0.977 instead of the notebook's 0.888).

Writes to BOTH:
  - src/data/sampleData/meta-learning/protonet_training.json
  - public/sample-data/meta-learning/protonet_training.json

Output schema:
{
  "metaIters": 500,
  "lossCurve": [...500 floats],
  "accCurve":  [...500 floats],
  "smoothedLoss": [...476 floats],
  "smoothedAcc":  [...476 floats],
  "gridX": [...220 floats],     // linspace(-4, 4, 220) (same for gridY)
  "gridY": [...220 floats],
  "heldOutTasks": [
    { "Xs": [[x,y], ...], "ys": [...], "Xq": [[x,y], ...], "yq": [...],
      "queryAcc": float,
      "predGrid": [...48400 ints in {0..4}]  // pred[iy*220 + ix], meshgrid order
    }, ...2 tasks
  ]
}

Runs in ~5 s on CPU.
"""

import sys

import numpy as np
import torch

from _notebook_exec import floats, run_notebook_through, write_json

LAST_CELL = 44
SEED = 20260518


def smooth(a, w=25):
    a = np.asarray(a)
    kernel = np.ones(w) / w
    return np.convolve(a, kernel, mode="valid")


def main():
    print(f"Executing notebook cells 0-{LAST_CELL}…", file=sys.stderr)
    ns = run_notebook_through(LAST_CELL)
    proto = ns["proto"]
    sample_protonet_task = ns["sample_protonet_task"]
    losses, accs = ns["proto_losses"], ns["proto_accs"]
    print(f"loss {losses[0]:.3f} -> {losses[-1]:.3f}; "
          f"acc {accs[0]:.2f} -> {accs[-1]:.2f}", file=sys.stderr)

    # Held-out tasks: the same computation as cell 46.
    held_rng = np.random.default_rng(SEED + 100)
    gx = np.linspace(-4, 4, 220)
    gy = np.linspace(-4, 4, 220)
    XX, YY = np.meshgrid(gx, gy)
    grid_t = torch.tensor(np.column_stack([XX.ravel(), YY.ravel()]),
                          dtype=torch.float32)
    held_tasks = []
    for _ in range(2):
        Xs, ys_np, Xq, yq_np, _means = sample_protonet_task(held_rng)
        Xs_t = torch.tensor(Xs, dtype=torch.float32)
        ys_t = torch.tensor(ys_np, dtype=torch.long)
        Xq_t = torch.tensor(Xq, dtype=torch.float32)
        with torch.no_grad():
            sup_emb = proto(Xs_t)
            prototypes = torch.stack([sup_emb[ys_t == k].mean(dim=0)
                                      for k in range(5)])
            q_dists = torch.cdist(proto(Xq_t), prototypes, p=2) ** 2
            q_preds = q_dists.argmin(dim=-1).numpy()
            g_dists = torch.cdist(proto(grid_t), prototypes, p=2) ** 2
            pred_grid = g_dists.argmin(dim=-1).numpy()  # flat, meshgrid ravel order
        q_acc = float((q_preds == yq_np).mean())
        print(f"held-out query acc {q_acc:.2f}", file=sys.stderr)
        held_tasks.append({
            "Xs": Xs.tolist(),
            "ys": ys_np.tolist(),
            "Xq": Xq.tolist(),
            "yq": yq_np.tolist(),
            "queryAcc": q_acc,
            "predGrid": pred_grid.astype(int).tolist(),
        })

    payload = {
        "metaIters": int(ns["N_EPOCHS_PROTO"]),
        "lossCurve": floats(losses),
        "accCurve": floats(accs),
        "smoothedLoss": floats(smooth(losses)),
        "smoothedAcc": floats(smooth(accs)),
        "gridX": floats(gx),
        "gridY": floats(gy),
        "heldOutTasks": held_tasks,
    }
    write_json("protonet_training.json", payload)


if __name__ == "__main__":
    main()
