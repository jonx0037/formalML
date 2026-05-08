#!/usr/bin/env python3
"""Precompute head_to_head.json for the §12 ESSPerSecondHeadToHead viz.

Re-runs the §12 head-to-head benchmark from cell 26 of
01_stochastic_gradient_mcmc.ipynb: Bayesian logistic regression with
D = 10 features, N ∈ {500, 2000, 10000, 50000} synthetic observations,
sampling via SGLD, SGHMC, and NUTS (PyMC). Reports ESS/sec, wall-clock
time, and posterior-mean estimates per (method, N).

Output schema (head_to_head.json, ~3 KB):
  {
    "true_beta_first5":  [1.0, -0.7, 0.5, 0.0, 0.0],
    "N_grid":            [500, 2000, 10000, 50000],
    "methods":           ["SGLD", "SGHMC", "NUTS"],
    "ess_per_sec":       {"SGLD": [...], "SGHMC": [...], "NUTS": [...]},
    "wall_clock_sec":    {"SGLD": [...], "SGHMC": [...], "NUTS": [...]},
    "posterior_means_at_N_max": {"SGLD": [...], "SGHMC": [...], "NUTS": [...]},
  }

Dual-writes to src/data/sampleData/stochastic-gradient-mcmc/ AND
public/sample-data/stochastic-gradient-mcmc/ from inside main() — the
canonical formalML pattern (see CLAUDE.md "Sample-data dual-location").

Usage (after `uv sync`):
    cd notebooks/stochastic-gradient-mcmc
    .venv/bin/python precompute_head_to_head.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)


SEED = 0
D_FEATURES = 10
N_GRID = [500, 2000, 10000, 50000]
TRUE_BETA = np.concatenate([np.array([1.0, -0.7, 0.5]), np.zeros(D_FEATURES - 3)])
PRIOR_SIGMA = 5.0


def make_blr_dataset(N: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((N, D_FEATURES))
    logits = X @ TRUE_BETA
    y = (rng.uniform(size=N) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    return X, y


def sgld_logreg(X, y, n_steps, eta, B, prior_sigma, rng):
    N, D = X.shape
    beta = np.zeros(D)
    chain = np.empty((n_steps, D))
    for n in range(n_steps):
        idx = rng.choice(N, size=B, replace=False)
        Xb, yb = X[idx], y[idx]
        p = 1.0 / (1.0 + np.exp(-Xb @ beta))
        g = beta / prior_sigma**2 + (N / B) * Xb.T @ (p - yb)
        beta = beta - eta * g + np.sqrt(2 * eta) * rng.standard_normal(D)
        chain[n] = beta
    return chain


def sghmc_logreg(X, y, n_steps, eta, B, C, prior_sigma, rng):
    N, D = X.shape
    beta = np.zeros(D)
    r = rng.standard_normal(D)
    chain = np.empty((n_steps, D))
    for n in range(n_steps):
        idx = rng.choice(N, size=B, replace=False)
        Xb, yb = X[idx], y[idx]
        p = 1.0 / (1.0 + np.exp(-Xb @ beta))
        g = beta / prior_sigma**2 + (N / B) * Xb.T @ (p - yb)
        r = (1 - eta * C) * r - eta * g + np.sqrt(2 * eta * C) * rng.standard_normal(D)
        beta = beta + eta * r
        chain[n] = beta
    return chain


def nuts_logreg(X, y, n_draws, prior_sigma):
    import pymc as pm  # noqa: PLC0415 — heavy import deferred

    with pm.Model():
        beta = pm.Normal("beta", mu=0.0, sigma=prior_sigma, shape=X.shape[1])
        logits = pm.math.dot(X, beta)
        pm.Bernoulli("y", logit_p=logits, observed=y)
        trace = pm.sample(
            draws=n_draws,
            tune=200,
            chains=1,
            cores=1,
            progressbar=False,
            random_seed=SEED + 999,
            target_accept=0.85,
        )
    return trace


def autocorr(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / max(n, 1)
    if var == 0.0:
        return np.zeros(min(n // 4, 500) + 1)
    max_lag = min(n // 4, 500)
    return np.array([np.dot(x[: n - k], x[k:]) / (n * var) for k in range(max_lag + 1)])


def integrated_autocorr_time(x: np.ndarray, c: float = 5.0) -> float:
    acf = autocorr(x)
    if len(acf) <= 1:
        return 1.0
    tau = 1.0 + 2.0 * np.cumsum(acf[1:])
    M = np.where(np.arange(1, len(tau) + 1) >= c * tau)[0]
    M = M[0] if len(M) else len(tau) - 1
    return float(tau[M])


def ess(chain: np.ndarray) -> float:
    """Per-coord ESS averaged."""
    return float(np.mean([len(chain) / max(integrated_autocorr_time(chain[:, j]), 1.0) for j in range(chain.shape[1])]))


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent  # notebooks/<slug>/.. /.. → repo root
    out_dirs = [
        repo_root / "src" / "data" / "sampleData" / "stochastic-gradient-mcmc",
        repo_root / "public" / "sample-data" / "stochastic-gradient-mcmc",
    ]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    ess_per_sec: dict[str, list[float]] = {"SGLD": [], "SGHMC": [], "NUTS": []}
    wall_clock_sec: dict[str, list[float]] = {"SGLD": [], "SGHMC": [], "NUTS": []}
    posterior_means_at_N_max: dict[str, list[float]] = {"SGLD": [], "SGHMC": [], "NUTS": []}

    for N in N_GRID:
        rng = np.random.default_rng(SEED + N)
        X, y = make_blr_dataset(N, rng)

        # Choose iteration counts so each method runs ≈ 1 s of useful sampling.
        # Smaller N → more steps; larger N → fewer.
        n_sgld = 5000
        n_sghmc = 5000
        # NUTS draws scale inversely with N (the cell's heuristic).
        n_nuts = max(300, int(750_000 / N))

        # SGLD
        t0 = time.perf_counter()
        sgld_chain = sgld_logreg(X, y, n_sgld, eta=1e-4, B=128, prior_sigma=PRIOR_SIGMA, rng=np.random.default_rng(SEED + 100 + N))
        t_sgld = time.perf_counter() - t0
        ess_sgld = ess(sgld_chain[1000:])  # drop warm-up
        ess_per_sec["SGLD"].append(ess_sgld / max(t_sgld, 1e-9))
        wall_clock_sec["SGLD"].append(t_sgld)

        # SGHMC
        t0 = time.perf_counter()
        sghmc_chain = sghmc_logreg(X, y, n_sghmc, eta=5e-5, B=128, C=0.05, prior_sigma=PRIOR_SIGMA, rng=np.random.default_rng(SEED + 200 + N))
        t_sghmc = time.perf_counter() - t0
        ess_sghmc = ess(sghmc_chain[1000:])
        ess_per_sec["SGHMC"].append(ess_sghmc / max(t_sghmc, 1e-9))
        wall_clock_sec["SGHMC"].append(t_sghmc)

        # NUTS
        t0 = time.perf_counter()
        nuts_trace = nuts_logreg(X, y, n_draws=n_nuts, prior_sigma=PRIOR_SIGMA)
        t_nuts = time.perf_counter() - t0
        nuts_samples = nuts_trace.posterior["beta"].values.reshape(-1, D_FEATURES)
        ess_nuts = ess(nuts_samples)
        ess_per_sec["NUTS"].append(ess_nuts / max(t_nuts, 1e-9))
        wall_clock_sec["NUTS"].append(t_nuts)

        if N == N_GRID[-1]:
            posterior_means_at_N_max["SGLD"] = list(map(float, sgld_chain[1000:].mean(axis=0)[:5]))
            posterior_means_at_N_max["SGHMC"] = list(map(float, sghmc_chain[1000:].mean(axis=0)[:5]))
            posterior_means_at_N_max["NUTS"] = list(map(float, nuts_samples.mean(axis=0)[:5]))

        print(
            f"N = {N:6d}: SGLD = {ess_per_sec['SGLD'][-1]:8.1f} ESS/s, "
            f"SGHMC = {ess_per_sec['SGHMC'][-1]:8.1f}, NUTS = {ess_per_sec['NUTS'][-1]:8.1f}  "
            f"(wall: {t_sgld:.1f}s / {t_sghmc:.1f}s / {t_nuts:.1f}s)"
        )

    payload = {
        "true_beta_first5": list(map(float, TRUE_BETA[:5])),
        "N_grid": list(N_GRID),
        "methods": ["SGLD", "SGHMC", "NUTS"],
        "ess_per_sec": {k: [float(v) for v in vs] for k, vs in ess_per_sec.items()},
        "wall_clock_sec": {k: [float(v) for v in vs] for k, vs in wall_clock_sec.items()},
        "posterior_means_at_N_max": posterior_means_at_N_max,
    }

    for d in out_dirs:
        out = d / "head_to_head.json"
        out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
