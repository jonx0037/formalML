"""
precompute_protonet_training.py — ProtoNet training curves + held-out decision
regions, matching cells 44-46 of the notebook.

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
  "heldOutTasks": [
    { "Xs": [[x,y], ...], "ys": [...], "Xq": [[x,y], ...], "yq": [...],
      "queryAcc": float,
      "gridX":    [...220 floats],  // linspace(-4, 4, 220) (same for gridY)
      "predGrid": [...48400 ints in {0..4}]  // pred[i*220 + j]
    }, ...2 tasks
  ]
}

Runs in ~30-60s on CPU.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 20260518
DEVICE = torch.device("cpu")

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "meta-learning",
    REPO_ROOT / "public" / "sample-data" / "meta-learning",
]


def sample_protonet_task(rng, n_classes=5, k_shot=5, m_query=15, sigma=0.35, radius=2.5):
    base = rng.uniform(0, 2 * np.pi)
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False) + base
    means = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    Xs, ys, Xq, yq = [], [], [], []
    for c, mu in enumerate(means):
        Xs.append(mu + sigma * rng.standard_normal((k_shot, 2)))
        ys.extend([c] * k_shot)
        Xq.append(mu + sigma * rng.standard_normal((m_query, 2)))
        yq.extend([c] * m_query)
    return np.vstack(Xs), np.array(ys), np.vstack(Xq), np.array(yq), means


class ProtoNet(nn.Module):
    def __init__(self, in_dim=2, h_dim=64, emb_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, emb_dim),
        )

    def forward(self, x):
        return self.net(x)


def proto_loss_and_acc(model, support_x, support_y, query_x, query_y, n_classes=5):
    sup_emb = model(support_x)
    qry_emb = model(query_x)
    prototypes = torch.stack([
        sup_emb[support_y == k].mean(dim=0) for k in range(n_classes)
    ])
    dists = torch.cdist(qry_emb, prototypes, p=2) ** 2
    logits = -dists
    loss = F.cross_entropy(logits, query_y)
    acc = (logits.argmax(dim=-1) == query_y).float().mean()
    return loss, acc, prototypes


def smooth(a, w=25):
    a = np.asarray(a)
    kernel = np.ones(w) / w
    return np.convolve(a, kernel, mode='valid')


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    proto = ProtoNet().to(DEVICE)
    opt = torch.optim.Adam(proto.parameters(), lr=1e-3)
    n_iters = 500
    losses, accs = [], []
    print(f"Training ProtoNet for {n_iters} iters…", file=sys.stderr)
    for ep in range(n_iters):
        Xs, ys_np, Xq, yq_np, _ = sample_protonet_task(rng)
        Xs_t = torch.tensor(Xs, dtype=torch.float32)
        ys_t = torch.tensor(ys_np, dtype=torch.long)
        Xq_t = torch.tensor(Xq, dtype=torch.float32)
        yq_t = torch.tensor(yq_np, dtype=torch.long)
        opt.zero_grad()
        loss, acc, _ = proto_loss_and_acc(proto, Xs_t, ys_t, Xq_t, yq_t)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        accs.append(float(acc.item()))

    # Held-out tasks
    held_rng = np.random.default_rng(SEED + 100)
    held_tasks = []
    gx = np.linspace(-4, 4, 220)
    gy = np.linspace(-4, 4, 220)
    XX, YY = np.meshgrid(gx, gy)
    grid_t = torch.tensor(np.column_stack([XX.ravel(), YY.ravel()]), dtype=torch.float32)
    print("Computing held-out decision regions…", file=sys.stderr)
    for _ in range(2):
        Xs, ys_np, Xq, yq_np, _ = sample_protonet_task(held_rng)
        Xs_t = torch.tensor(Xs, dtype=torch.float32)
        ys_t = torch.tensor(ys_np, dtype=torch.long)
        Xq_t = torch.tensor(Xq, dtype=torch.float32)
        with torch.no_grad():
            sup_emb = proto(Xs_t)
            protos = torch.stack([sup_emb[ys_t == k].mean(dim=0) for k in range(5)])
            qry_emb = proto(Xq_t)
            q_dists = torch.cdist(qry_emb, protos, p=2) ** 2
            q_preds = q_dists.argmin(dim=-1).numpy()
            emb_grid = proto(grid_t)
            grid_dists = torch.cdist(emb_grid, protos, p=2) ** 2
            pred_grid = grid_dists.argmin(dim=-1).numpy().astype(int)
        q_acc = float((q_preds == yq_np).mean())
        held_tasks.append({
            "Xs": Xs.tolist(),
            "ys": ys_np.tolist(),
            "Xq": Xq.tolist(),
            "yq": yq_np.tolist(),
            "queryAcc": q_acc,
            "predGrid": pred_grid.tolist(),
        })

    payload = {
        "metaIters": n_iters,
        "lossCurve": [float(x) for x in losses],
        "accCurve": [float(x) for x in accs],
        "smoothedLoss": [float(x) for x in smooth(losses).tolist()],
        "smoothedAcc": [float(x) for x in smooth(accs).tolist()],
        "gridX": gx.tolist(),
        "gridY": gy.tolist(),
        "heldOutTasks": held_tasks,
    }

    for outdir in OUT_DIRS:
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "protonet_training.json", "w") as f:
            json.dump(payload, f, allow_nan=False)
        print(f"wrote {outdir / 'protonet_training.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
