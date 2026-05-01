"""Precompute centered eight-schools NUTS trace as JSON for B3 NealsFunnelExplorer.

Companion to ``01_probabilistic_programming.ipynb``. Fits the centered
parameterization of Rubin (1981)'s eight-schools model in PyMC at default
NUTS settings, extracts the joint draws of (mu, log tau, theta_1..theta_8)
along with divergence flags, and writes the trace to JSON under both
``src/data/sampleData/probabilistic-programming/`` and
``public/sample-data/probabilistic-programming/`` for the formalML viz
component to consume at runtime.

The centered fit is *expected* to produce divergent transitions clustering at
the funnel neck (low log-tau region). That is the §5.2 pedagogical payload —
B3 NealsFunnelExplorer scatters these draws with divergent transitions in red
so the reader watches the divergences cluster precisely where the topic's
prose says they should.

Usage::

    cd notebooks/probabilistic-programming
    .venv/bin/python precompute_neals_funnel.py

Runtime: ~30-90 seconds on a 2020-era laptop.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths and reproducibility
# --------------------------------------------------------------------------- #

SEED = 20260430
NOTEBOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NOTEBOOK_DIR.parents[1]  # formalML/

# Dual-location convention per CLAUDE.md "Sample-data dual-location":
# components fetch /sample-data/<slug>/<file>.json at runtime, so the file
# must live in public/. The src/data/sampleData/<slug>/ copy is the
# canonical source-controlled artifact.
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "probabilistic-programming",
    REPO_ROOT / "public" / "sample-data" / "probabilistic-programming",
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

OUT_FILENAME = "neals_funnel.json"

# --------------------------------------------------------------------------- #
# Eight-schools data (Rubin 1981)
# --------------------------------------------------------------------------- #

SCHOOLS = ["A", "B", "C", "D", "E", "F", "G", "H"]
Y_OBS = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
SIGMA_OBS = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
J = len(SCHOOLS)

# --------------------------------------------------------------------------- #
# JSON serialization helpers
# --------------------------------------------------------------------------- #


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def _round_floats(arr: list, ndigits: int = 4) -> list:
    """Round nested float lists to ndigits to keep JSON sizes reasonable.

    The viz tolerates 4 sig figs of jitter — these are sampler draws, not
    exact constants, so the visible scatter is unchanged.
    """
    out = []
    for v in arr:
        if isinstance(v, list):
            out.append(_round_floats(v, ndigits))
        elif isinstance(v, float):
            out.append(round(v, ndigits))
        else:
            out.append(v)
    return out


# --------------------------------------------------------------------------- #
# Fit the centered parameterization
# --------------------------------------------------------------------------- #


def fit_centered_eight_schools() -> az.InferenceData:
    """Fit the §5.2 model spec verbatim:

        mu     ~ N(0, 5^2)
        tau    ~ HalfNormal(5)
        theta  ~ N(mu, tau^2)
        y      ~ N(theta, sigma_obs^2)

    NUTS at default target_accept=0.8 to expose divergent transitions.
    """
    rng = np.random.default_rng(SEED)
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=5.0)
        tau = pm.HalfNormal("tau", sigma=5.0)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=J)
        pm.Normal("y_obs", mu=theta, sigma=SIGMA_OBS, observed=Y_OBS)

        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=1,  # serial for reproducibility across machines
            target_accept=0.8,
            random_seed=int(rng.integers(2**31)),
            progressbar=False,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": False},
        )
    return idata


def fit_noncentered_eight_schools() -> az.InferenceData:
    """Fit the §5.3 non-centered reparameterization:

        mu     ~ N(0, 5^2)
        tau    ~ HalfNormal(5)
        z      ~ N(0, 1)                 (standard-Normal auxiliary)
        theta  := mu + tau * z           (deterministic)
        y      ~ N(theta, sigma_obs^2)

    Same NUTS settings as the centered fit. Expected diagnostics:
    zero (or very few) divergences, R-hat ≈ 1.00 across the board, full
    effective sample size — because the (z_j, log tau) joint is rectangular
    instead of funnel-shaped.
    """
    rng = np.random.default_rng(SEED + 1)  # different seed for second fit
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=5.0)
        tau = pm.HalfNormal("tau", sigma=5.0)
        z = pm.Normal("z", mu=0.0, sigma=1.0, shape=J)
        theta = pm.Deterministic("theta", mu + tau * z)
        pm.Normal("y_obs", mu=theta, sigma=SIGMA_OBS, observed=Y_OBS)

        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=1,
            target_accept=0.8,
            random_seed=int(rng.integers(2**31)),
            progressbar=False,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": False},
        )
    return idata


def extract_noncentered_payload(idata: az.InferenceData) -> dict:
    """Flatten chains × draws → samples and pull z, theta, divergence flags.

    Returns a dict matching the centered payload's draws structure plus a
    ``z`` field (the standard-Normal auxiliary that NUTS actually saw).
    """
    posterior = idata.posterior
    sample_stats = idata.sample_stats

    n_chains = posterior.sizes["chain"]
    n_draws = posterior.sizes["draw"]
    n_total = n_chains * n_draws

    mu_flat = posterior["mu"].values.reshape(n_total)
    tau_flat = posterior["tau"].values.reshape(n_total)
    log_tau_flat = np.log(tau_flat)
    z_flat = posterior["z"].values.reshape(n_total, J)
    theta_flat = posterior["theta"].values.reshape(n_total, J)

    diverging = sample_stats["diverging"].values.reshape(n_total).astype(bool)
    n_divergent = int(diverging.sum())
    summary = az.summary(idata, var_names=["mu", "tau"], round_to=4)

    print(f"  Drew {n_total} samples (non-centered)")
    print(f"  Divergent transitions: {n_divergent} ({100.0 * n_divergent / n_total:.2f}%)")
    print(f"  R-hat (mu, tau): {float(summary.loc['mu', 'r_hat']):.4f}, "
          f"{float(summary.loc['tau', 'r_hat']):.4f}")
    print(f"  ESS bulk (mu, tau): {int(summary.loc['mu', 'ess_bulk'])}, "
          f"{int(summary.loc['tau', 'ess_bulk'])}")

    return {
        "metadata": {
            "model": "eight_schools_noncentered",
            "engine": "PyMC",
            "pymc_version": pm.__version__,
            "n_chains": n_chains,
            "n_draws_per_chain": n_draws,
            "n_total": n_total,
            "n_divergent": n_divergent,
            "divergence_rate": n_divergent / n_total,
            "target_accept": 0.8,
            "tune_steps": 1000,
            "seed": SEED + 1,
            "rhat": {
                "mu": float(summary.loc["mu", "r_hat"]),
                "tau": float(summary.loc["tau", "r_hat"]),
            },
            "ess_bulk": {
                "mu": int(summary.loc["mu", "ess_bulk"]),
                "tau": int(summary.loc["tau", "ess_bulk"]),
            },
        },
        "draws": {
            "mu": _round_floats(_to_jsonable(mu_flat)),
            "log_tau": _round_floats(_to_jsonable(log_tau_flat)),
            "z": _round_floats(_to_jsonable(z_flat)),
            "theta": _round_floats(_to_jsonable(theta_flat)),
            "divergent": _to_jsonable(diverging),
        },
    }


def extract_trace_payload(idata: az.InferenceData) -> dict:
    """Flatten chains × draws → samples and pull divergence flags."""
    posterior = idata.posterior
    sample_stats = idata.sample_stats

    n_chains = posterior.sizes["chain"]
    n_draws = posterior.sizes["draw"]
    n_total = n_chains * n_draws

    mu_flat = posterior["mu"].values.reshape(n_total)
    tau_flat = posterior["tau"].values.reshape(n_total)
    theta_flat = posterior["theta"].values.reshape(n_total, J)

    log_tau_flat = np.log(tau_flat)

    # Divergent transitions: chain × draw boolean array, flattened.
    diverging = sample_stats["diverging"].values.reshape(n_total).astype(bool)

    n_divergent = int(diverging.sum())
    summary = az.summary(idata, var_names=["mu", "tau"], round_to=4)

    print(f"  Drew {n_total} samples ({n_chains} chains × {n_draws} draws)")
    print(f"  Divergent transitions: {n_divergent} ({100.0 * n_divergent / n_total:.2f}%)")
    print(f"  R-hat (mu, tau): {float(summary.loc['mu', 'r_hat']):.4f}, "
          f"{float(summary.loc['tau', 'r_hat']):.4f}")
    print(f"  ESS bulk (mu, tau): {int(summary.loc['mu', 'ess_bulk'])}, "
          f"{int(summary.loc['tau', 'ess_bulk'])}")

    return {
        "metadata": {
            "model": "eight_schools_centered",
            "engine": "PyMC",
            "pymc_version": pm.__version__,
            "n_chains": n_chains,
            "n_draws_per_chain": n_draws,
            "n_total": n_total,
            "n_divergent": n_divergent,
            "divergence_rate": n_divergent / n_total,
            "target_accept": 0.8,
            "tune_steps": 1000,
            "seed": SEED,
            "rhat": {
                "mu": float(summary.loc["mu", "r_hat"]),
                "tau": float(summary.loc["tau", "r_hat"]),
            },
            "ess_bulk": {
                "mu": int(summary.loc["mu", "ess_bulk"]),
                "tau": int(summary.loc["tau", "ess_bulk"]),
            },
        },
        "schools": SCHOOLS,
        "y_obs": _round_floats(_to_jsonable(Y_OBS), 2),
        "sigma_obs": _round_floats(_to_jsonable(SIGMA_OBS), 2),
        "draws": {
            "mu": _round_floats(_to_jsonable(mu_flat)),
            "log_tau": _round_floats(_to_jsonable(log_tau_flat)),
            "theta": _round_floats(_to_jsonable(theta_flat)),
            "divergent": _to_jsonable(diverging),
        },
    }


def main() -> None:
    t0 = time.time()
    print("Fitting centered eight-schools …")
    idata = fit_centered_eight_schools()
    payload = extract_trace_payload(idata)

    print("Fitting non-centered eight-schools …")
    idata_nc = fit_noncentered_eight_schools()
    payload["non_centered"] = extract_noncentered_payload(idata_nc)

    body = json.dumps(payload, separators=(",", ":"))
    for d in OUT_DIRS:
        out_path = d / OUT_FILENAME
        out_path.write_text(body)
        print(f"  Wrote {out_path.relative_to(REPO_ROOT)} ({len(body) / 1024:.1f} KB)")

    print(f"Done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
