"""precompute_alignment_uniformity_trajectory.py — Wang-Isola trajectory.

Trains a small contrastive encoder (linear projection from a 2D toy fixture
onto the 2D unit circle) for a long horizon. Logs (L_align, L_uniform, L_NCE)
at each epoch.

The AlignmentUniformityScatter viz reads this and plots the trajectory in
the (L_align, L_uniform) plane, showing that successful contrastive training
drives both metrics toward their joint minimizer (Wang-Isola 2020).

Writes to BOTH:
  - src/data/sampleData/representation-learning/alignment_uniformity_trajectory.json
  - public/sample-data/representation-learning/alignment_uniformity_trajectory.json

Runtime: ~5-10 s on CPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 17
N = 256
D_IN = 4
D_PROJ = 4   # project to 4-D unit hypersphere
TAU = 0.5
EPOCHS = 300
LR = 5e-2
ALIGN_RECORD_EVERY = 1

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "representation-learning",
    REPO_ROOT / "public" / "sample-data" / "representation-learning",
]


def make_pairs(seed: int) -> torch.Tensor:
    """Sample N anchor vectors in R^{D_IN} and N positives = anchor + small noise."""
    rng = np.random.default_rng(seed)
    anchors = rng.normal(size=(N, D_IN)).astype(np.float32)
    positives = anchors + 0.3 * rng.normal(size=(N, D_IN)).astype(np.float32)
    # Pack as (2N, D_IN) with rows [a_1, p_1, a_2, p_2, ...].
    pairs = np.empty((2 * N, D_IN), dtype=np.float32)
    pairs[0::2] = anchors
    pairs[1::2] = positives
    return torch.from_numpy(pairs)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D_IN, 16), nn.ReLU(), nn.Linear(16, D_PROJ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


def alignment_loss(z: torch.Tensor, tau: float) -> torch.Tensor:
    """E[-<z, z+>/tau] across N positive pairs in a (2N, d) batch."""
    K = z.shape[0] // 2
    za = z[0::2]
    zp = z[1::2]
    return -(za * zp).sum(dim=1).mean() / tau


def uniformity_loss(z: torch.Tensor, tau: float) -> torch.Tensor:
    """E_z log E_z' exp(<z, z'>/tau) averaged over the 2N batch."""
    N2 = z.shape[0]
    sim = (z @ z.t()) / tau
    mask = torch.eye(N2, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))
    return torch.logsumexp(sim, dim=1).mean() - float(np.log(N2 - 1))


def nt_xent_full(z: torch.Tensor, tau: float) -> torch.Tensor:
    """NT-Xent on a (2N, d) batch where row 2i is the positive partner of 2i+1."""
    K = z.shape[0] // 2
    sim = (z @ z.t()) / tau
    mask = torch.eye(2 * K, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))
    # Build the target index for each row.
    target = torch.empty(2 * K, dtype=torch.long, device=z.device)
    target[0::2] = torch.arange(1, 2 * K, 2)
    target[1::2] = torch.arange(0, 2 * K, 2)
    return F.cross_entropy(sim, target)


def main(force: bool = False) -> None:
    target_path = OUT_DIRS[0] / "alignment_uniformity_trajectory.json"
    if target_path.exists() and not force:
        print(f"precompute_alignment_uniformity_trajectory.py: {target_path} exists, skipping (use --force)")
        return
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"  building {N} positive pairs in R^{D_IN}")
    pairs = make_pairs(SEED)
    model = Encoder()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    align_hist: list[float] = []
    uniform_hist: list[float] = []
    nce_hist: list[float] = []
    print(f"  training: epochs = {EPOCHS}, tau = {TAU}")
    for epoch in range(EPOCHS):
        z = model(pairs)
        loss = nt_xent_full(z, TAU)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % ALIGN_RECORD_EVERY == 0:
            with torch.no_grad():
                z = model(pairs)
                a = float(alignment_loss(z, TAU))
                u = float(uniformity_loss(z, TAU))
                nce = float(nt_xent_full(z, TAU))
            align_hist.append(a)
            uniform_hist.append(u)
            nce_hist.append(nce)
        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch + 1:3d}  L_NCE = {nce_hist[-1]:.3f}  L_align = {align_hist[-1]:.3f}  L_uniform = {uniform_hist[-1]:.3f}")

    # Final-epoch embeddings (for the optional final-state scatter).
    with torch.no_grad():
        final_z = model(pairs).numpy().astype(float)

    payload = {
        "seed": SEED,
        "N_pairs": int(N),
        "d_in": int(D_IN),
        "d_proj": int(D_PROJ),
        "tau": float(TAU),
        "epochs": int(EPOCHS),
        "record_every": int(ALIGN_RECORD_EVERY),
        "align_history": align_hist,
        "uniform_history": uniform_hist,
        "nce_history": nce_hist,
        "final_z": final_z.tolist(),
    }
    serialized = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "alignment_uniformity_trajectory.json"
        out_path.write_text(serialized)
        print(f"  wrote {out_path} ({len(serialized):,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute even if output exists")
    args = parser.parse_args()
    main(force=args.force)
