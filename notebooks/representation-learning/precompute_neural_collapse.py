"""precompute_neural_collapse.py — supervised classifier collapse trajectory.

Trains a small supervised classifier on the 8-class fixture from §4 to
interpolation and beyond (~300 epochs). At each epoch, logs:
  • NC1:  tr(Σ_W) / tr(Σ_B)  (within / between class variance ratio)
  • NC2:  mean off-diagonal cosine of class means after centering+normalizing
  • train-set accuracy
At the end, saves the final last-layer features (PCA-2 projection per class) and
the final class-mean cosine-similarity matrix.

The NeuralCollapseSimplex viz reads this and shows:
  • the NC1 trajectory collapsing to ~0
  • the NC2 trajectory converging to -1/(K-1) = -1/7
  • the final class-mean cosine matrix as a heatmap (diagonal = 1, off = ETF target)

Writes to BOTH:
  - src/data/sampleData/representation-learning/neural_collapse_trajectory.json
  - public/sample-data/representation-learning/neural_collapse_trajectory.json

Runtime: ~30-60 s on CPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
N_PER_CLASS = 100
K = 8
D = 16
HIDDEN = 64
FEAT_DIM = 16    # last-layer feature dimension (before the K-way classifier head)
EPOCHS = 400
BATCH = 64
LR = 1e-2
WEIGHT_DECAY = 1e-4

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "representation-learning",
    REPO_ROOT / "public" / "sample-data" / "representation-learning",
]


def make_fixture(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """8 isotropic 16-D Gaussians at corners of a high-d simplex-like configuration."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 4.0
    centers = np.concatenate([centers, np.zeros((K, D - 2))], axis=1)
    X = np.zeros((N_PER_CLASS * K, D), dtype=np.float32)
    y = np.zeros(N_PER_CLASS * K, dtype=np.int64)
    for k in range(K):
        X[k * N_PER_CLASS : (k + 1) * N_PER_CLASS] = centers[k] + rng.normal(scale=0.6, size=(N_PER_CLASS, D))
        y[k * N_PER_CLASS : (k + 1) * N_PER_CLASS] = k
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


class Classifier(nn.Module):
    def __init__(self, d_in: int = D, hidden: int = HIDDEN, feat: int = FEAT_DIM, K: int = K):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat),
        )
        self.head = nn.Linear(feat, K, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logits = self.head(h)
        return logits, h


def collapse_metrics(features: np.ndarray, labels: np.ndarray, K_classes: int) -> tuple[float, float, np.ndarray]:
    """Compute NC1 (within/between trace ratio) and NC2 (mean off-diagonal cosine)."""
    global_mean = features.mean(0)
    class_means = np.zeros((K_classes, features.shape[1]))
    within_tr = 0.0
    n_total = 0
    for k in range(K_classes):
        mask = labels == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        mu_k = features[mask].mean(0)
        class_means[k] = mu_k
        diffs = features[mask] - mu_k
        within_tr += float((diffs * diffs).sum())
        n_total += nk
    within_tr /= max(n_total, 1)
    between_tr = 0.0
    for k in range(K_classes):
        nk = int((labels == k).sum())
        if nk == 0:
            continue
        dmu = class_means[k] - global_mean
        between_tr += (nk / n_total) * float((dmu * dmu).sum())
    nc1 = within_tr / max(between_tr, 1e-12)
    # NC2: cosine similarity of centered + normalized class means.
    centered = class_means - global_mean
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    centered_n = centered / np.maximum(norms, 1e-12)
    G = centered_n @ centered_n.T
    K2 = K_classes
    nc2 = float((G.sum() - np.trace(G)) / (K2 * (K2 - 1)))
    return nc1, nc2, G


def main(force: bool = False) -> None:
    target = OUT_DIRS[0] / "neural_collapse_trajectory.json"
    if target.exists() and not force:
        print(f"precompute_neural_collapse.py: {target} exists, skipping (use --force)")
        return
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"  building fixture: {K} classes, n = {N_PER_CLASS * K}, d = {D}")
    X, y = make_fixture(SEED)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    n = len(X)

    model = Classifier()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    nc1_hist: list[float] = []
    nc2_hist: list[float] = []
    acc_hist: list[float] = []
    loss_hist: list[float] = []

    print(f"  training: epochs = {EPOCHS}, ETF target = {-1 / (K - 1):.4f}")
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            logits, _ = model(Xt[idx])
            loss = F.cross_entropy(logits, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach())
        with torch.no_grad():
            logits, h = model(Xt)
            pred = logits.argmax(dim=1)
            acc = float((pred == yt).float().mean())
            h_np = h.numpy().astype(float)
        nc1, nc2, _ = collapse_metrics(h_np, y, K)
        nc1_hist.append(nc1)
        nc2_hist.append(nc2)
        acc_hist.append(acc)
        loss_hist.append(total_loss / max(1, n // BATCH))
        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch + 1:3d}  acc = {acc:.3f}  NC1 = {nc1:.4f}  NC2 = {nc2:.4f}")

    # Final-state outputs.
    with torch.no_grad():
        _, h = model(Xt)
        h_np = h.numpy().astype(float)
    _, _, G_final = collapse_metrics(h_np, y, K)

    # PCA-2 of the final-layer features for the scatter.
    Xc = h_np - h_np.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / Xc.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    top2 = eigvecs[:, -2:][:, ::-1]
    feat_2d = Xc @ top2

    payload = {
        "seed": SEED,
        "K": int(K),
        "n_per_class": int(N_PER_CLASS),
        "feat_dim": int(FEAT_DIM),
        "epochs": int(EPOCHS),
        "etf_target": float(-1 / (K - 1)),
        "nc1_history": nc1_hist,
        "nc2_history": nc2_hist,
        "acc_history": acc_hist,
        "loss_history": loss_hist,
        "final_cosine_matrix": G_final.tolist(),
        "final_features_2d": feat_2d.tolist(),
        "labels": y.tolist(),
    }
    serialized = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "neural_collapse_trajectory.json"
        out_path.write_text(serialized)
        print(f"  wrote {out_path} ({len(serialized):,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute even if output exists")
    args = parser.parse_args()
    main(force=args.force)
