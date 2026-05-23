"""precompute_vae_decoder.py — VAE latent space + decoder grid for §4.3 viz.

Trains a small variational autoencoder (β = 1) on the 8-class synthetic
fixture used throughout §4, then serializes:
  • the encoder's posterior means μ_z(x) for the training set (with class labels)
  • the decoder's mean-prediction surface μ_x(z) on a regular z-grid

The VAELatentSpaceExplorer viz overlays the training scatter on the decoder
heatmap and lets the reader click anywhere in latent space to see what the
decoder would draw.

Writes to BOTH:
  - src/data/sampleData/representation-learning/vae_decoder.json
  - public/sample-data/representation-learning/vae_decoder.json

Runtime: ~10-20 s on CPU.
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
N_PER_CLASS = 200
K = 8
D = 8           # input dim
Z = 2           # latent dim (so we can plot the latent space directly)
HIDDEN = 32
EPOCHS = 60
BATCH = 64
LR = 1e-2
BETA = 1.0
GRID = 24       # 24 × 24 z-grid for the decoder surface

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "representation-learning",
    REPO_ROOT / "public" / "sample-data" / "representation-learning",
]


def make_fixture(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """8 isotropic 8-D Gaussians on a ring."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 4.0
    centers = np.concatenate([centers, np.zeros((K, D - 2))], axis=1)
    X = np.zeros((N_PER_CLASS * K, D), dtype=np.float32)
    y = np.zeros(N_PER_CLASS * K, dtype=np.int64)
    for k in range(K):
        X[k * N_PER_CLASS : (k + 1) * N_PER_CLASS] = (
            centers[k] + rng.normal(scale=0.5, size=(N_PER_CLASS, D))
        )
        y[k * N_PER_CLASS : (k + 1) * N_PER_CLASS] = k
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


class VAE(nn.Module):
    def __init__(self, d: int = D, z: int = Z, hidden: int = HIDDEN):
        super().__init__()
        self.enc_h = nn.Linear(d, hidden)
        self.enc_mu = nn.Linear(hidden, z)
        self.enc_ls = nn.Linear(hidden, z)
        self.dec_h = nn.Linear(z, hidden)
        self.dec_mu = nn.Linear(hidden, d)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.enc_h(x))
        return self.enc_mu(h), self.enc_ls(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.dec_h(z))
        return self.dec_mu(h)

    def forward(self, x: torch.Tensor):
        mu, ls = self.encode(x)
        sigma = (0.5 * ls).exp()
        eps = torch.randn_like(mu)
        zsamp = mu + sigma * eps
        xhat = self.decode(zsamp)
        return xhat, mu, ls


def main(force: bool = False) -> None:
    target = OUT_DIRS[0] / "vae_decoder.json"
    if target.exists() and not force:
        print(f"precompute_vae_decoder.py: {target} exists, skipping (use --force)")
        return
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"  building fixture: {K} classes, n = {N_PER_CLASS * K}, d = {D}, z = {Z}")
    X, y = make_fixture(SEED)
    Xt = torch.from_numpy(X)

    print(f"  training VAE: epochs = {EPOCHS}, beta = {BETA}")
    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(X)
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        total_recon = 0.0
        total_kl = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            xb = Xt[idx]
            xhat, mu, ls = model(xb)
            recon = F.mse_loss(xhat, xb, reduction="sum")
            kl = -0.5 * (1 + ls - mu.pow(2) - ls.exp()).sum()
            loss = recon + BETA * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_recon += float(recon.detach())
            total_kl += float(kl.detach())
        if (epoch + 1) % 20 == 0:
            print(
                f"  epoch {epoch + 1:3d}  recon = {total_recon / n:.3f}  KL = {total_kl / n:.3f}"
            )

    # Encoder pass on the full training set — store posterior means.
    with torch.no_grad():
        mu_all, ls_all = model.encode(Xt)
        mu_all_np = mu_all.numpy().astype(float)
        sigma_all_np = (0.5 * ls_all).exp().numpy().astype(float)

    # Decoder evaluated on a regular z-grid (for the latent-space heatmap).
    z_range = float(np.max(np.abs(mu_all_np)) * 1.2)
    zs = np.linspace(-z_range, z_range, GRID)
    grid_z = np.array([[zx, zy] for zy in zs[::-1] for zx in zs], dtype=np.float32)
    with torch.no_grad():
        decoded = model.decode(torch.from_numpy(grid_z)).numpy().astype(float)

    payload = {
        "seed": SEED,
        "n_per_class": int(N_PER_CLASS),
        "K": int(K),
        "D": int(D),
        "Z": int(Z),
        "beta": float(BETA),
        "z_range": float(z_range),
        "grid": int(GRID),
        "scatter_mu": mu_all_np.tolist(),
        "scatter_sigma": sigma_all_np.tolist(),
        "scatter_labels": y.tolist(),
        "grid_z": grid_z.tolist(),
        "grid_x": decoded.tolist(),
    }
    serialized = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "vae_decoder.json"
        out_path.write_text(serialized)
        print(f"  wrote {out_path} ({len(serialized):,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute even if output exists")
    args = parser.parse_args()
    main(force=args.force)
