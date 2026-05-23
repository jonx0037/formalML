"""precompute_simclr_embeddings.py — SimCLR encoder + projector embeddings.

Trains a small SimCLR-style contrastive encoder + projection head on the
4-label × 4-color synthetic fixture from §6.2 (heavy noise on the color dims
makes the contrastive task color-invariant). Serializes:
  • the encoder output h for the full fixture (16-D → projected to 2D via PCA)
  • the projector output z (also PCA-projected to 2D)
  • the label and color metadata for each sample
  • PCA-2 of the raw input X (for comparison)
  • the LDA / Fisher direction projection of the raw X (sufficient-statistic ref)
  • the simplex-ETF cosine target for K = 4 labels

Two Tier-2 viz read this file:
  • RepresentationGeometryViewer (§1.2 / §12.1) — toggle raw/PCA/h/z views
  • SufficiencyToContrastiveBridge (§12.4) — show that the InfoNCE-optimal
    encoder's class means and the closed-form sufficient statistic for the
    Gaussian-mixture class converge to the same simplex ETF geometry.

Writes to BOTH:
  - src/data/sampleData/representation-learning/simclr_embeddings.json
  - public/sample-data/representation-learning/simclr_embeddings.json

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
N_LABELS = 4
N_COLORS = 4
N_PER_CELL = 60
D = 16
LABEL_NOISE = 0.4
COLOR_NOISE = 2.0
ENC_HIDDEN = 32
ENC_OUT = 16
PROJ_OUT = 8
TAU = 0.5
EPOCHS = 200
BATCH = 64
LR = 1e-2

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "representation-learning",
    REPO_ROOT / "public" / "sample-data" / "representation-learning",
]


def make_fixture(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label × color fixture: 4 labels x 4 colors x N_PER_CELL samples in 16-D.

    Labels live in dims 0-7 (each label has a unique mean pattern on these).
    Colors live in dims 8-15 (each color has a unique mean pattern there).
    Label noise is small (so the contrastive task can succeed on label structure),
    color noise is large (so color is the dominant variance axis — like
    Sec. §6.2's projector-head experiment, the contrastive loss is forced to be
    color-invariant under augmentations that resample the color dims).
    """
    rng = np.random.default_rng(seed)
    n = N_LABELS * N_COLORS * N_PER_CELL
    X = np.zeros((n, D), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    colors = np.zeros(n, dtype=np.int64)
    # Random per-label / per-color mean vectors in the corresponding half-dims.
    label_means = rng.normal(size=(N_LABELS, 8)) * 3.0
    color_means = rng.normal(size=(N_COLORS, 8)) * 3.0
    idx = 0
    for ell in range(N_LABELS):
        for c in range(N_COLORS):
            for _ in range(N_PER_CELL):
                X[idx, :8] = label_means[ell] + rng.normal(scale=LABEL_NOISE, size=8)
                X[idx, 8:] = color_means[c] + rng.normal(scale=COLOR_NOISE, size=8)
                labels[idx] = ell
                colors[idx] = c
                idx += 1
    perm = rng.permutation(n)
    return X[perm], labels[perm], colors[perm]


def augment(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Augmentation: resample the color half-dims (8..15) ~ N(0, COLOR_NOISE²).

    This is the augmentation that defines "same instance" for the contrastive
    task — two views of the same anchor differ only in their color half.
    The encoder is forced to be color-invariant if it wants to align positives.
    """
    out = x.clone()
    n = x.shape[0]
    eps = torch.randn(n, 8, generator=rng) * COLOR_NOISE
    # Re-center each row's color half on the random base around 0.
    out[:, 8:] = eps
    return out


class SimCLREncoder(nn.Module):
    def __init__(self, d_in: int = D, d_h: int = ENC_HIDDEN, d_out: int = ENC_OUT, d_proj: int = PROJ_OUT):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, d_out)
        )
        self.proj = nn.Sequential(
            nn.Linear(d_out, d_h), nn.ReLU(), nn.Linear(d_h, d_proj)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        z = self.proj(h)
        z = F.normalize(z, dim=1)
        return h, z


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    """Symmetric NT-Xent (SimCLR) loss. z1, z2 are (B, d_proj), unit-normed."""
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)               # (2B, d)
    sim = (z @ z.t()) / tau                      # (2B, 2B)
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))
    # Positive partner for row i: i + B (mod 2B).
    target = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, target)


def project_to_2d(X: np.ndarray) -> np.ndarray:
    """PCA-2 projection by direct top-2 eigendecomposition."""
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / Xc.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvals ascending; take last two.
    top2 = eigvecs[:, -2:][:, ::-1]
    return Xc @ top2


def lda_1d_direction(X: np.ndarray, y: np.ndarray, K: int) -> np.ndarray:
    """Fisher LDA direction (rank-1 reduction) for K-class data."""
    Xc = X - X.mean(0, keepdims=True)
    Sw = np.zeros((X.shape[1], X.shape[1]))
    means = np.zeros((K, X.shape[1]))
    for k in range(K):
        mask = y == k
        means[k] = X[mask].mean(0)
        Xkc = X[mask] - means[k]
        Sw += Xkc.T @ Xkc
    Sw /= X.shape[0]
    Sb = np.zeros_like(Sw)
    mu = X.mean(0)
    for k in range(K):
        nk = int((y == k).sum())
        dmu = (means[k] - mu).reshape(-1, 1)
        Sb += (nk / X.shape[0]) * (dmu @ dmu.T)
    eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(Sw) @ Sb)
    direction = eigvecs[:, -1]
    direction = direction / np.linalg.norm(direction)
    return Xc @ direction.reshape(-1, 1)


def class_means(Z: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    means = np.zeros((K, Z.shape[1]))
    for k in range(K):
        means[k] = Z[labels == k].mean(0)
    return means


def pairwise_cosine(M: np.ndarray) -> float:
    """Mean off-diagonal cosine similarity of K centered + unit-normed rows."""
    Mc = M - M.mean(0, keepdims=True)
    norms = np.linalg.norm(Mc, axis=1, keepdims=True)
    Mn = Mc / np.maximum(norms, 1e-12)
    G = Mn @ Mn.T
    K = M.shape[0]
    off = (G.sum() - np.trace(G)) / (K * (K - 1))
    return float(off)


def main(force: bool = False) -> None:
    target = OUT_DIRS[0] / "simclr_embeddings.json"
    if target.exists() and not force:
        print(f"precompute_simclr_embeddings.py: {target} exists, skipping (use --force)")
        return
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"  building fixture: {N_LABELS} labels × {N_COLORS} colors × {N_PER_CELL}/cell, d = {D}")
    X, labels, colors = make_fixture(SEED)
    Xt = torch.from_numpy(X)
    n = len(X)
    gen = torch.Generator()
    gen.manual_seed(SEED + 1)

    print(f"  training SimCLR: epochs = {EPOCHS}, batch = {BATCH}, τ = {TAU}")
    model = SimCLREncoder()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_hist: list[float] = []
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        total = 0.0
        steps = 0
        for i in range(0, n - BATCH, BATCH):
            idx = perm[i : i + BATCH]
            x = Xt[idx]
            x1 = augment(x, gen)
            x2 = augment(x, gen)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = nt_xent(z1, z2, TAU)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            steps += 1
        loss_hist.append(total / max(steps, 1))
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch + 1:3d}  L_NCE = {loss_hist[-1]:.3f}")

    # Encoder pass on the full set (no augmentation) — gives the "natural" h, z.
    with torch.no_grad():
        h_full, z_full = model(Xt)
        h_np = h_full.numpy().astype(float)
        z_np = z_full.numpy().astype(float)

    h2d = project_to_2d(h_np)
    z2d = project_to_2d(z_np)
    x2d = project_to_2d(X)
    lda1d = lda_1d_direction(X, labels, N_LABELS)

    # Sufficiency-to-contrastive bridge: compare the class-mean geometry.
    h_means = class_means(h_np, labels, N_LABELS)
    z_means = class_means(z_np, labels, N_LABELS)
    x_means = class_means(X, labels, N_LABELS)
    etf_target = -1.0 / (N_LABELS - 1)

    payload = {
        "seed": SEED,
        "n": int(n),
        "K_labels": int(N_LABELS),
        "K_colors": int(N_COLORS),
        "d_input": int(D),
        "tau": float(TAU),
        "loss_history": [float(v) for v in loss_hist],
        "labels": labels.tolist(),
        "colors": colors.tolist(),
        # 2D projections (used by RepresentationGeometryViewer).
        "raw_2d": x2d.tolist(),
        "h_2d": h2d.tolist(),
        "z_2d": z2d.tolist(),
        "lda_1d": [float(v) for v in lda1d.flatten()],
        # Class-mean cosines (used by SufficiencyToContrastiveBridge).
        "h_class_means": h_means.tolist(),
        "z_class_means": z_means.tolist(),
        "x_class_means": x_means.tolist(),
        "h_mean_cosine": pairwise_cosine(h_means),
        "z_mean_cosine": pairwise_cosine(z_means),
        "x_mean_cosine": pairwise_cosine(x_means),
        "etf_target": float(etf_target),
    }
    serialized = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "simclr_embeddings.json"
        out_path.write_text(serialized)
        print(f"  wrote {out_path} ({len(serialized):,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute even if output exists")
    args = parser.parse_args()
    main(force=args.force)
