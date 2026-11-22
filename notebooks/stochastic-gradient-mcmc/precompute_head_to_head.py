#!/usr/bin/env python3
"""Precompute head_to_head.json for the §12 ESSPerSecondHeadToHead viz.

Runs the §12 head-to-head benchmark with exactly the settings of cell 26 of
01_stochastic_gradient_mcmc.ipynb: Bayesian logistic regression with D = 10
features, prior N(0, 2² I), N ∈ {500, 2000, 10000, 50000} synthetic
observations, sampled by SGLD, SGHMC (mini-batch B = 32, 5,000 steps, first
1,000 dropped), and NUTS (PyMC, 200 tuning steps, max(300, 1500·√(500/N))
draws, target_accept = 0.85). Reports ESS/sec, wall-clock time, and the
posterior mean and standard deviation of the first 5 coefficients at N_max.

Step sizes scale with N. The posterior curvature (Hessian of H) grows linearly
in N, so a fixed η crosses the Euler stability limit η·λ_max < 2 (§8.5) and the
gradient-noise heating η·V_B grows like N²/B (§7.5, eq. 8.3). Anchored at
N_REF = 500, where η_SGLD·λ_max ≈ 0.1 (the §11.5 10% rule), each scheme's step
scales inversely with curvature (§12.2a):
    SGLD   η = 1e-3 · 500/N
    SGHMC  η = 5e-3 · √(500/N),  C = 10 · √(N/500)   (η·C and η²·λ stay fixed)

ESS is (draws) / IAT of β₁ (Sokal automated windowing, the notebook's
`integrated_autocorr_time`); for NUTS the IAT is floored at 1. NUTS is warmed
up (a 5-draw run on the same model, which triggers the PyTensor compile)
before its timer starts, so the timings measure sampling, not compilation.

Seeds (SEED = 0): data SEED + N, SGLD SEED + 1000 + N, SGHMC SEED + 2000 + N,
NUTS SEED + 999. The chains, ESS, and posterior summaries are deterministic;
ess_per_sec and wall_clock_sec depend on the machine.

Output schema (head_to_head.json):
  {
    "true_beta_first5":          [1.0, -0.7, 0.5, 0.0, 0.0],
    "N_grid":                    [500, 2000, 10000, 50000],
    "methods":                   ["SGLD", "SGHMC", "NUTS"],
    "ess_per_sec":               {"SGLD": [...], "SGHMC": [...], "NUTS": [...]},
    "wall_clock_sec":            {"SGLD": [...], "SGHMC": [...], "NUTS": [...]},
    "posterior_means_at_N_max":  {"SGLD": [...], "SGHMC": [...], "NUTS": [...]},
    "posterior_sd_at_N_max":     {"SGLD": [...], "SGHMC": [...], "NUTS": [...]},
    "note":                      "..."
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
import logging
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)


SEED = 0
D_FEATURES = 10
N_GRID = [500, 2000, 10000, 50000]
TRUE_BETA = np.concatenate([np.array([1.0, -0.7, 0.5]), np.zeros(D_FEATURES - 3)])
PRIOR_SIGMA = 2.0
B = 32
N_STEPS = 5000
BURN = 1000
N_REF = 500
ETA_SGLD_REF, ETA_SGHMC_REF, C_SGHMC_REF = 1e-3, 5e-3, 10.0


def sgld_eta(N: int) -> float:
    return ETA_SGLD_REF * N_REF / N


def sghmc_eta_C(N: int) -> tuple[float, float]:
    return ETA_SGHMC_REF * np.sqrt(N_REF / N), C_SGHMC_REF * np.sqrt(N / N_REF)


def nuts_draws(N: int) -> int:
    return max(300, int(1500 * np.sqrt(500 / N)))


def make_blr_dataset(N: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((N, D_FEATURES))
    logits = X @ TRUE_BETA
    y = (rng.uniform(size=N) < 1 / (1 + np.exp(-logits))).astype(int)
    return X, y


def sgld_logreg(X, y, n_steps, eta, B, prior_sigma, rng):
    N, D = X.shape
    beta = np.zeros(D)
    chain = np.empty((n_steps, D))
    for n in range(n_steps):
        idx = rng.choice(N, size=B, replace=False)
        Xb, yb = X[idx], y[idx]
        p = 1 / (1 + np.exp(-Xb @ beta))
        g_data = (N / B) * Xb.T @ (p - yb)
        g = beta / prior_sigma**2 + g_data
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
        p = 1 / (1 + np.exp(-Xb @ beta))
        g_data = (N / B) * Xb.T @ (p - yb)
        g = beta / prior_sigma**2 + g_data
        r = (1 - eta * C) * r - eta * g + np.sqrt(2 * eta * C) * rng.standard_normal(D)
        beta = beta + eta * r
        chain[n] = beta
    return chain


def nuts_logreg(X, y, n_draws, prior_sigma):
    """NUTS via PyMC. Returns (draws, sampling wall-clock seconds).

    A 5-draw warm-up run on the same model triggers the PyTensor compile
    before the timer starts, so the returned time excludes compilation.
    """
    import pymc as pm  # noqa: PLC0415 — heavy import deferred

    with pm.Model():
        beta = pm.Normal("beta", mu=0, sigma=prior_sigma, shape=X.shape[1])
        logits = pm.math.dot(X, beta)
        pm.Bernoulli("y", logit_p=logits, observed=y)
        pm_logger = logging.getLogger("pymc")
        level = pm_logger.level
        pm_logger.setLevel(logging.ERROR)
        pm.sample(draws=5, tune=5, chains=1, cores=1, progressbar=False,
                  random_seed=SEED + 999, compute_convergence_checks=False)
        pm_logger.setLevel(level)
        t0 = time.perf_counter()
        trace = pm.sample(draws=n_draws, tune=200, chains=1, cores=1,
                          progressbar=False, random_seed=SEED + 999, target_accept=0.85)
        t_nuts = time.perf_counter() - t0
    return trace.posterior["beta"].values[0], t_nuts


def autocorr(x, max_lag=None):
    """Sample autocorrelation of a 1D chain, lags 0..max_lag (notebook cell 2)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if max_lag is None:
        max_lag = min(n // 4, 500)
    var = np.dot(x, x) / n
    if var == 0.0:
        return np.zeros(max_lag + 1)
    return np.array([np.dot(x[: n - k], x[k:]) / (n * var) for k in range(max_lag + 1)])


def integrated_autocorr_time(x, c=5.0):
    """Sokal's automated-windowing IAT estimator (notebook cell 2)."""
    acf = autocorr(x)
    tau = 1.0 + 2.0 * np.cumsum(acf[1:])
    M = np.where(np.arange(1, len(tau) + 1) >= c * tau)[0]
    M = M[0] if len(M) else len(tau) - 1
    return tau[M]


def run_benchmark(verbose: bool = True) -> dict:
    """Run the §12 sweep. Returns per-method results and the N_max chains' summaries."""
    results = {"SGLD": {}, "SGHMC": {}, "NUTS": {}}
    for N in N_GRID:
        X, y = make_blr_dataset(N, np.random.default_rng(SEED + N))

        t0 = time.perf_counter()
        chain_s = sgld_logreg(X, y, N_STEPS, eta=sgld_eta(N), B=B, prior_sigma=PRIOR_SIGMA,
                              rng=np.random.default_rng(SEED + 1000 + N))
        t_sgld = time.perf_counter() - t0
        ess_s = (N_STEPS - BURN) / integrated_autocorr_time(chain_s[BURN:, 0])
        results["SGLD"][N] = {"ess_per_sec": ess_s / t_sgld, "wall_sec": t_sgld}

        eta_h, C_h = sghmc_eta_C(N)
        t0 = time.perf_counter()
        chain_h = sghmc_logreg(X, y, N_STEPS, eta=eta_h, B=B, C=C_h, prior_sigma=PRIOR_SIGMA,
                               rng=np.random.default_rng(SEED + 2000 + N))
        t_sghmc = time.perf_counter() - t0
        ess_h = (N_STEPS - BURN) / integrated_autocorr_time(chain_h[BURN:, 0])
        results["SGHMC"][N] = {"ess_per_sec": ess_h / t_sghmc, "wall_sec": t_sghmc}

        chain_n, t_nuts = nuts_logreg(X, y, nuts_draws(N), PRIOR_SIGMA)
        ess_n = len(chain_n) / max(integrated_autocorr_time(chain_n[:, 0]), 1.0)
        results["NUTS"][N] = {"ess_per_sec": ess_n / t_nuts, "wall_sec": t_nuts}

        if verbose:
            print(f"N = {N:>6}: SGLD = {results['SGLD'][N]['ess_per_sec']:>7.1f} ESS/s, "
                  f"SGHMC = {results['SGHMC'][N]['ess_per_sec']:>7.1f}, "
                  f"NUTS = {results['NUTS'][N]['ess_per_sec']:>7.1f}  "
                  f"(wall: {t_sgld:4.2f}s/{t_sghmc:4.2f}s/{t_nuts:5.2f}s)")

    post = {"SGLD": chain_s[BURN:], "SGHMC": chain_h[BURN:], "NUTS": chain_n}
    return {
        "results": results,
        "post_mean": {k: v.mean(axis=0) for k, v in post.items()},
        "post_sd": {k: v.std(axis=0) for k, v in post.items()},
    }


NOTE = (
    "Generated by notebooks/stochastic-gradient-mcmc/precompute_head_to_head.py, whose code is "
    "notebook cell 26's: Bayesian logistic regression, D = 10, prior N(0, 2^2 I), B = 32, 5,000 SG "
    "steps (first 1,000 dropped); SGLD eta = 1e-3*500/N; SGHMC eta = 5e-3*sqrt(500/N), "
    "C = 10*sqrt(N/500); NUTS (PyMC) 200 tune + max(300, 1500*sqrt(500/N)) draws, "
    "target_accept 0.85, timed after a warm-up compile; ESS = draws / Sokal IAT of beta_1; "
    "seeds data SEED+N, SGLD SEED+1000+N, SGHMC SEED+2000+N, NUTS SEED+999, SEED = 0. "
    "Posterior means/sds are of the first 5 coefficients at N = 50,000. "
    "ess_per_sec and wall_clock_sec are machine-dependent."
)


def build_payload(bench: dict) -> dict:
    results = bench["results"]
    methods = ["SGLD", "SGHMC", "NUTS"]
    return {
        "true_beta_first5": [float(v) for v in TRUE_BETA[:5]],
        "N_grid": list(N_GRID),
        "methods": methods,
        "ess_per_sec": {m: [round(float(results[m][N]["ess_per_sec"]), 1) for N in N_GRID] for m in methods},
        "wall_clock_sec": {m: [round(float(results[m][N]["wall_sec"]), 3) for N in N_GRID] for m in methods},
        "posterior_means_at_N_max": {m: [round(float(v), 4) for v in bench["post_mean"][m][:5]] for m in methods},
        "posterior_sd_at_N_max": {m: [round(float(v), 4) for v in bench["post_sd"][m][:5]] for m in methods},
        "note": NOTE,
    }


def write_payload(payload: dict, repo_root: Path) -> None:
    out_dirs = [
        repo_root / "src" / "data" / "sampleData" / "stochastic-gradient-mcmc",
        repo_root / "public" / "sample-data" / "stochastic-gradient-mcmc",
    ]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        out = d / "head_to_head.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out} ({out.stat().st_size} bytes)")


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent  # notebooks/<slug>/.. /.. → repo root
    write_payload(build_payload(run_benchmark()), repo_root)


if __name__ == "__main__":
    main()
