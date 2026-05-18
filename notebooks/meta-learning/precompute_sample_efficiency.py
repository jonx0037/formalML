"""
precompute_sample_efficiency.py — Sample-efficiency sweep for CNP and Latent NP
on held-out GP tasks (cell 69 of the notebook). Requires trained CNP/LatentNP
which we re-train inline (matching cells 34-35) for self-containment.

Writes to BOTH:
  - src/data/sampleData/meta-learning/sample_efficiency.json
  - public/sample-data/meta-learning/sample_efficiency.json

Output schema:
{
  "K_ctx": [2, 4, 8, 12, 16, 20],
  "cnpNllMean": [...6 floats],
  "cnpNllSE":   [...6 floats],
  "lnpNllMean": [...6 floats],
  "lnpNllSE":   [...6 floats]
}

Runs in ~30-45s on CPU.
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


def sample_gp_task_full(rng, x_range=(-3.0, 3.0), n_total=100, ell_range=(0.5, 2.0), noise=NOISE_STD):
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
        self.net = nn.Sequential(nn.Linear(x_dim + y_dim, h_dim), nn.ReLU(),
                                 nn.Linear(h_dim, h_dim), nn.ReLU(),
                                 nn.Linear(h_dim, r_dim))

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=-1)).mean(dim=0)


class NPDecoder(nn.Module):
    def __init__(self, x_dim=1, r_dim=64, h_dim=64, y_dim=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(x_dim + r_dim, h_dim), nn.ReLU(),
                                 nn.Linear(h_dim, h_dim), nn.ReLU(),
                                 nn.Linear(h_dim, 2 * y_dim))

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
        self.shared = nn.Sequential(nn.Linear(x_dim + y_dim, h_dim), nn.ReLU(),
                                    nn.Linear(h_dim, h_dim), nn.ReLU())
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


def cnp_nll(mu, log_sigma, y):
    return -torch.distributions.Normal(mu, log_sigma.exp()).log_prob(y).sum()


def lnp_elbo(mu_pred, log_sigma_pred, y_target, prior_params, post_params):
    log_lik = torch.distributions.Normal(mu_pred, log_sigma_pred.exp()).log_prob(y_target).sum()
    mu_c, log_sigma_c = prior_params
    mu_t, log_sigma_t = post_params
    sigma_c2 = (2 * log_sigma_c).exp()
    sigma_t2 = (2 * log_sigma_t).exp()
    kl = (log_sigma_c - log_sigma_t + (sigma_t2 + (mu_t - mu_c) ** 2) / (2 * sigma_c2) - 0.5).sum()
    return -log_lik + kl


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    cnp = CNP().to(DEVICE)
    cnp_opt = torch.optim.Adam(cnp.parameters(), lr=1e-3)
    torch.manual_seed(SEED + 1)
    lnp = LatentNP().to(DEVICE)
    lnp_opt = torch.optim.Adam(lnp.parameters(), lr=1e-3)
    N_EPOCHS = 1000
    print(f"Training CNP + Latent NP for {N_EPOCHS} epochs…", file=sys.stderr)
    for _ in range(N_EPOCHS):
        x_full, y_full, ell = sample_gp_task_full(rng)
        (xc, yc), (xt, yt) = split_ctx_tgt(x_full, y_full, rng=rng)
        xc_t, yc_t = to_col(xc), to_col(yc)
        xt_t, yt_t = to_col(xt), to_col(yt)
        xfull_t = torch.cat([xc_t, xt_t], dim=0)
        yfull_t = torch.cat([yc_t, yt_t], dim=0)
        cnp_opt.zero_grad()
        mu, log_sigma = cnp(xc_t, yc_t, xt_t)
        loss_c = cnp_nll(mu, log_sigma, yt_t)
        loss_c.backward()
        cnp_opt.step()
        lnp_opt.zero_grad()
        mu_p, log_sigma_p, prior_p, post_p = lnp(xc_t, yc_t, xt_t, xfull_t, yfull_t)
        loss_l = lnp_elbo(mu_p, log_sigma_p, yt_t, prior_p, post_p)
        loss_l.backward()
        lnp_opt.step()

    eff_rng = np.random.default_rng(SEED + 500)
    K_ctx_values = [2, 4, 8, 12, 16, 20]
    n_held = 20
    n_target = 30
    cnp_mean, cnp_se, lnp_mean, lnp_se = [], [], [], []
    for K_ctx in K_ctx_values:
        cnp_nlls, lnp_nlls = [], []
        for _ in range(n_held):
            x_full, y_full, _ell = sample_gp_task_full(eff_rng)
            idx = eff_rng.permutation(len(x_full))
            xc_v = x_full[idx[:K_ctx]]
            yc_v = y_full[idx[:K_ctx]]
            xt_v = x_full[idx[K_ctx:K_ctx + n_target]]
            yt_v = y_full[idx[K_ctx:K_ctx + n_target]]
            xc_v_t, yc_v_t = to_col(xc_v), to_col(yc_v)
            xt_v_t, yt_v_t = to_col(xt_v), to_col(yt_v)
            with torch.no_grad():
                mu_c, log_sigma_c = cnp(xc_v_t, yc_v_t, xt_v_t)
                nll_c = -torch.distributions.Normal(mu_c, log_sigma_c.exp()).log_prob(yt_v_t).mean()
                cnp_nlls.append(float(nll_c.item()))
                lps = []
                for _ in range(20):
                    mu_l, log_sigma_l, _, _ = lnp(xc_v_t, yc_v_t, xt_v_t)
                    lp = torch.distributions.Normal(mu_l, log_sigma_l.exp()).log_prob(yt_v_t).mean()
                    lps.append(float(lp.item()))
                lnp_nlls.append(-np.mean(lps))
        cnp_mean.append(float(np.mean(cnp_nlls)))
        cnp_se.append(float(np.std(cnp_nlls) / np.sqrt(n_held)))
        lnp_mean.append(float(np.mean(lnp_nlls)))
        lnp_se.append(float(np.std(lnp_nlls) / np.sqrt(n_held)))

    payload = {
        "K_ctx": K_ctx_values,
        "cnpNllMean": cnp_mean,
        "cnpNllSE": cnp_se,
        "lnpNllMean": lnp_mean,
        "lnpNllSE": lnp_se,
    }
    for outdir in OUT_DIRS:
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "sample_efficiency.json", "w") as f:
            json.dump(payload, f, allow_nan=False)
        print(f"wrote {outdir / 'sample_efficiency.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
