"""
precompute_np_curves.py — Generate CNP and Latent NP training curves +
held-out posterior predictive snapshot, matching cells 34-37 of the
verified notebook.

Writes to BOTH:
  - src/data/sampleData/meta-learning/np_curves.json
  - public/sample-data/meta-learning/np_curves.json

Output schema:
{
  "epochs": 1000,
  "cnpLoss":  [...1000 floats],
  "lnpLoss":  [...1000 floats],
  "smoothedCnpLoss": [...976 floats],   // 25-pt moving average (matches notebook)
  "smoothedLnpLoss": [...976 floats],
  "smoothedXStart": 12,                  // start index of smoothed series in original grid
  "heldOut": {
    "ell": 1.0,
    "xFull":  [...100 floats],
    "yFull":  [...100 floats],
    "xContext": [...8 floats],
    "yContext": [...8 floats],
    "xDense":   [...200 floats],
    "muGp":     [...200 floats],
    "sdGp":     [...200 floats],
    "muCnp":    [...200 floats],
    "sdCnp":    [...200 floats],
    "muLnp":    [...200 floats],
    "sdLnp":    [...200 floats],
    "lnpSamples": [[...200], [...200], ...20 traces]
  }
}

Runs in ~15-20 s on CPU.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SEED = 20260518
DEVICE = torch.device("cpu")
NOISE_STD = 0.05
JITTER = 1e-5

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "meta-learning",
    REPO_ROOT / "public" / "sample-data" / "meta-learning",
]


def rbf_kernel(x1, x2, ell):
    d2 = (x1[:, None] - x2[None, :]) ** 2
    return np.exp(-0.5 * d2 / ell ** 2)


def sample_gp_task_full(rng, x_range=(-3.0, 3.0), n_total=100,
                       ell_range=(0.5, 2.0), noise=NOISE_STD):
    ell = rng.uniform(*ell_range)
    x = np.linspace(*x_range, n_total)
    K = rbf_kernel(x, x, ell) + (noise ** 2 + JITTER) * np.eye(n_total)
    L = np.linalg.cholesky(K)
    y = L @ rng.standard_normal(n_total)
    return x, y, ell


def split_ctx_tgt(x, y, n_ctx_min=5, n_ctx_max=15, n_tgt=30, rng=None):
    n_total = len(x)
    n_ctx = rng.integers(n_ctx_min, n_ctx_max + 1)
    idx = rng.permutation(n_total)
    return (x[idx[:n_ctx]], y[idx[:n_ctx]]), (x[idx[n_ctx:n_ctx + n_tgt]], y[idx[n_ctx:n_ctx + n_tgt]])


def to_col(a):
    return torch.tensor(a.reshape(-1, 1), dtype=torch.float32)


class NPEncoder(nn.Module):
    def __init__(self, x_dim=1, y_dim=1, h_dim=64, r_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, r_dim),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=-1)).mean(dim=0)


class NPDecoder(nn.Module):
    def __init__(self, x_dim=1, r_dim=64, h_dim=64, y_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + r_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, 2 * y_dim),
        )

    def forward(self, x_target, r):
        r_exp = r.unsqueeze(0).expand(x_target.size(0), -1)
        out = self.net(torch.cat([x_target, r_exp], dim=-1))
        mu, log_sigma = out.chunk(2, dim=-1)
        return mu, log_sigma.clamp(min=-5.0, max=2.0)


class CNP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = NPEncoder()
        self.decoder = NPDecoder()

    def forward(self, x_ctx, y_ctx, x_tgt):
        r = self.encoder(x_ctx, y_ctx)
        return self.decoder(x_tgt, r)


class LatentNPEncoder(nn.Module):
    def __init__(self, x_dim=1, y_dim=1, h_dim=64, z_dim=16):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(x_dim + y_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
        )
        self.to_mu = nn.Linear(h_dim, z_dim)
        self.to_log_sigma = nn.Linear(h_dim, z_dim)

    def forward(self, x, y):
        h = self.shared(torch.cat([x, y], dim=-1)).mean(dim=0)
        return self.to_mu(h), self.to_log_sigma(h).clamp(min=-5.0, max=2.0)


class LatentNP(nn.Module):
    def __init__(self, z_dim=16):
        super().__init__()
        self.encoder = LatentNPEncoder(z_dim=z_dim)
        self.decoder = NPDecoder(r_dim=z_dim)
        self.z_dim = z_dim

    def forward(self, x_ctx, y_ctx, x_tgt, x_full=None, y_full=None):
        mu_c, log_sigma_c = self.encoder(x_ctx, y_ctx)
        if x_full is None:
            z = mu_c + log_sigma_c.exp() * torch.randn_like(mu_c)
            mu, log_sigma = self.decoder(x_tgt, z)
            return mu, log_sigma, (mu_c, log_sigma_c), None
        mu_t, log_sigma_t = self.encoder(x_full, y_full)
        z = mu_t + log_sigma_t.exp() * torch.randn_like(mu_t)
        mu, log_sigma = self.decoder(x_tgt, z)
        return mu, log_sigma, (mu_c, log_sigma_c), (mu_t, log_sigma_t)


def cnp_neg_log_lik(mu, log_sigma, y):
    return -torch.distributions.Normal(mu, log_sigma.exp()).log_prob(y).sum()


def lnp_neg_elbo(mu_pred, log_sigma_pred, y_target, prior_params, post_params):
    log_lik = torch.distributions.Normal(mu_pred, log_sigma_pred.exp()).log_prob(y_target).sum()
    mu_c, log_sigma_c = prior_params
    mu_t, log_sigma_t = post_params
    sigma_c2 = (2 * log_sigma_c).exp()
    sigma_t2 = (2 * log_sigma_t).exp()
    kl = (log_sigma_c - log_sigma_t + (sigma_t2 + (mu_t - mu_c) ** 2) / (2 * sigma_c2) - 0.5).sum()
    return -log_lik + kl


def smooth(a, w=25):
    a = np.asarray(a)
    kernel = np.ones(w) / w
    return np.convolve(a, kernel, mode='valid')


def gp_posterior(x_ctx, y_ctx, x_tgt, ell, noise=NOISE_STD):
    K_cc = rbf_kernel(x_ctx, x_ctx, ell) + (noise ** 2 + JITTER) * np.eye(len(x_ctx))
    K_ct = rbf_kernel(x_ctx, x_tgt, ell)
    K_tt = rbf_kernel(x_tgt, x_tgt, ell)
    L = np.linalg.cholesky(K_cc)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_ctx))
    mu = K_ct.T @ alpha
    v = np.linalg.solve(L, K_ct)
    sigma2 = np.diag(K_tt) - np.sum(v ** 2, axis=0) + noise ** 2
    return mu, np.sqrt(sigma2)


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    cnp = CNP().to(DEVICE)
    cnp_opt = torch.optim.Adam(cnp.parameters(), lr=1e-3)

    torch.manual_seed(SEED + 1)
    lnp = LatentNP().to(DEVICE)
    lnp_opt = torch.optim.Adam(lnp.parameters(), lr=1e-3)

    N_EPOCHS = 1000
    cnp_losses, lnp_losses = [], []
    print(f"Training CNP + Latent NP for {N_EPOCHS} epochs…", file=sys.stderr)
    for ep in range(N_EPOCHS):
        x_full, y_full, ell = sample_gp_task_full(rng)
        (xc, yc), (xt, yt) = split_ctx_tgt(x_full, y_full, rng=rng)
        xc_t, yc_t = to_col(xc), to_col(yc)
        xt_t, yt_t = to_col(xt), to_col(yt)
        xfull_t = torch.cat([xc_t, xt_t], dim=0)
        yfull_t = torch.cat([yc_t, yt_t], dim=0)

        cnp_opt.zero_grad()
        mu, log_sigma = cnp(xc_t, yc_t, xt_t)
        loss_c = cnp_neg_log_lik(mu, log_sigma, yt_t)
        loss_c.backward()
        cnp_opt.step()
        cnp_losses.append(float(loss_c.item()))

        lnp_opt.zero_grad()
        mu_p, log_sigma_p, prior_p, post_p = lnp(xc_t, yc_t, xt_t, xfull_t, yfull_t)
        loss_l = lnp_neg_elbo(mu_p, log_sigma_p, yt_t, prior_p, post_p)
        loss_l.backward()
        lnp_opt.step()
        lnp_losses.append(float(loss_l.item()))

    # Held-out
    print("Computing held-out task predictions…", file=sys.stderr)
    held_rng = np.random.default_rng(SEED + 42)
    x_full, y_full, ell_true = sample_gp_task_full(held_rng, ell_range=(1.0, 1.0))
    ctx_idx = held_rng.choice(len(x_full), size=8, replace=False)
    xc, yc = x_full[ctx_idx], y_full[ctx_idx]
    x_dense = np.linspace(-3, 3, 200)
    mu_gp, sd_gp = gp_posterior(xc, yc, x_dense, ell_true)

    with torch.no_grad():
        mu_cnp, log_sigma_cnp = cnp(to_col(xc), to_col(yc), to_col(x_dense))
    mu_cnp_np = mu_cnp.squeeze(-1).numpy()
    sd_cnp_np = log_sigma_cnp.exp().squeeze(-1).numpy()

    torch.manual_seed(SEED + 7)
    lnp_samples = []
    with torch.no_grad():
        for _ in range(20):
            mu, log_sigma, _, _ = lnp(to_col(xc), to_col(yc), to_col(x_dense))
            lnp_samples.append(mu.squeeze(-1).numpy())
    lnp_samples_np = np.array(lnp_samples)
    mu_lnp_np = lnp_samples_np.mean(axis=0)
    sd_lnp_np = lnp_samples_np.std(axis=0)

    payload = {
        "epochs": N_EPOCHS,
        "cnpLoss": [float(x) for x in cnp_losses],
        "lnpLoss": [float(x) for x in lnp_losses],
        "smoothedCnpLoss": [float(x) for x in smooth(cnp_losses).tolist()],
        "smoothedLnpLoss": [float(x) for x in smooth(lnp_losses).tolist()],
        "smoothedXStart": 12,
        "heldOut": {
            "ell": float(ell_true),
            "xFull": [float(x) for x in x_full.tolist()],
            "yFull": [float(y) for y in y_full.tolist()],
            "xContext": [float(x) for x in xc.tolist()],
            "yContext": [float(y) for y in yc.tolist()],
            "xDense": [float(x) for x in x_dense.tolist()],
            "muGp": [float(x) for x in mu_gp.tolist()],
            "sdGp": [float(x) for x in sd_gp.tolist()],
            "muCnp": [float(x) for x in mu_cnp_np.tolist()],
            "sdCnp": [float(x) for x in sd_cnp_np.tolist()],
            "muLnp": [float(x) for x in mu_lnp_np.tolist()],
            "sdLnp": [float(x) for x in sd_lnp_np.tolist()],
            "lnpSamples": [[float(x) for x in row] for row in lnp_samples_np[:5].tolist()],
        },
    }

    for outdir in OUT_DIRS:
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "np_curves.json", "w") as f:
            json.dump(payload, f, allow_nan=False)
        print(f"wrote {outdir / 'np_curves.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
