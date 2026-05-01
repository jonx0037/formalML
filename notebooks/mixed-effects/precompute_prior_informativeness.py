"""Precompute Bayesian non-centered fits at four τ-prior scales for A5.

Companion to ``01_mixed_effects.ipynb``. Generates the §1 six-classroom data
with NumPy PCG64 and seed 20260429 (matching the topic's notebook), then fits
the non-centered model from §5 four times — once each at HalfNormal scales
{0.5, 2, 10, 50} for τ. For each fit, extracts the τ posterior, σ posterior,
and per-classroom random-effects posterior summary, plus the REML estimates
from statsmodels as a non-Bayesian reference.

The four prior scales span "prior dominates" → "prior is irrelevant" by
orders of magnitude. The reader who slides between them watches the §5.5
claim "Bayesian agrees with REML when data is informative" become
falsifiable: at scale=0.5 the prior shrinks Bayesian estimates much harder
than REML; at scale=10 (default) and scale=50 (loose) the two views agree.

Usage::

    cd notebooks/mixed-effects
    .venv/bin/python precompute_prior_informativeness.py

Runtime: ~30-90 seconds (4 fits × ~10s each).
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import statsmodels.formula.api as smf
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths and reproducibility
# --------------------------------------------------------------------------- #

SEED = 20260429  # matches §1's notebook seed
NOTEBOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NOTEBOOK_DIR.parents[1]

OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "mixed-effects",
    REPO_ROOT / "public" / "sample-data" / "mixed-effects",
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

OUT_FILENAME = "prior_informativeness.json"

# --------------------------------------------------------------------------- #
# Six-classroom DGP (matches §1's setup)
# --------------------------------------------------------------------------- #

J = 6
GROUP_SIZES = np.array([4, 6, 8, 10, 12, 20])
N = int(GROUP_SIZES.sum())
ALPHA_TRUE = 50.0
BETA_TRUE = 5.0
TAU_TRUE = 5.0
SIGMA_TRUE = 8.0

# τ-prior HalfNormal scales to scan.
PRIOR_SCALES = [0.5, 2.0, 10.0, 50.0]


def generate_data():
    """Six-classroom synthetic data, NumPy PCG64 seed=20260429."""
    rng = np.random.default_rng(SEED)
    classroom = np.repeat(np.arange(J), GROUP_SIZES)
    a_true = rng.normal(0.0, TAU_TRUE, size=J)
    x = rng.uniform(0.0, 5.0, size=N)
    y = (
        ALPHA_TRUE
        + BETA_TRUE * x
        + a_true[classroom]
        + rng.normal(0.0, SIGMA_TRUE, size=N)
    )
    return classroom, x, y, a_true


# --------------------------------------------------------------------------- #
# REML reference fit (frequentist plug-in path)
# --------------------------------------------------------------------------- #


def reml_reference(classroom, x, y):
    """Fit a random-intercept LMM via statsmodels.MixedLM (REML).

    Returns alpha_hat, beta_hat, tau_sq_hat, sigma_sq_hat, and per-classroom
    BLUPs as dicts the JSON output can carry.
    """
    df = pd.DataFrame({"y": y, "x": x, "classroom": classroom})
    mlm = smf.mixedlm("y ~ x", df, groups="classroom").fit(
        reml=True, method=["powell"]
    )
    alpha_hat = float(mlm.params["Intercept"])
    beta_hat = float(mlm.params["x"])
    tau_sq_hat = float(mlm.cov_re.iloc[0, 0])
    sigma_sq_hat = float(mlm.scale)

    blups = mlm.random_effects  # dict {classroom_index: pandas Series}
    blup_a = np.array([float(blups[j].iloc[0]) for j in range(J)])

    # 95% Wald intervals on the BLUPs from the conditional posterior variance:
    #   Var(u_j | y) = (1 - λ_j) τ²  with  λ_j = τ²·n_j / (τ²·n_j + σ²)
    lambdas = tau_sq_hat * GROUP_SIZES / (tau_sq_hat * GROUP_SIZES + sigma_sq_hat)
    sd_a = np.sqrt(np.maximum((1 - lambdas) * tau_sq_hat, 0.0))
    z = 1.959963984540054
    blup_lo = blup_a - z * sd_a
    blup_hi = blup_a + z * sd_a

    return {
        "alpha_hat": alpha_hat,
        "beta_hat": beta_hat,
        "tau_sq_hat": tau_sq_hat,
        "sigma_sq_hat": sigma_sq_hat,
        "lambdas": lambdas.tolist(),
        "blup_a": blup_a.tolist(),
        "blup_a_lo": blup_lo.tolist(),
        "blup_a_hi": blup_hi.tolist(),
    }


# --------------------------------------------------------------------------- #
# Bayesian non-centered fit at one prior scale
# --------------------------------------------------------------------------- #


def bayesian_fit_at_scale(classroom, x, y, prior_scale: float) -> dict:
    """Fit the §5 non-centered model with τ ~ HalfNormal(prior_scale)."""
    print(f"  Fitting non-centered model with τ ~ HalfNormal({prior_scale}) …")
    rng = np.random.default_rng(SEED + int(prior_scale * 100))
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=50.0, sigma=20.0)
        beta = pm.Normal("beta", mu=0.0, sigma=10.0)
        tau = pm.HalfNormal("tau", sigma=prior_scale)
        sigma = pm.HalfNormal("sigma", sigma=10.0)
        z = pm.Normal("z", mu=0.0, sigma=1.0, shape=J)
        a = pm.Deterministic("a", tau * z)
        pm.Normal(
            "y_obs",
            mu=alpha + beta * x + a[classroom],
            sigma=sigma,
            observed=y,
        )
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=1,
            target_accept=0.9,
            random_seed=int(rng.integers(2**31)),
            progressbar=False,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": False},
        )

    posterior = idata.posterior
    n_chains = posterior.sizes["chain"]
    n_draws = posterior.sizes["draw"]
    n_total = n_chains * n_draws

    tau_flat = posterior["tau"].values.reshape(n_total)
    sigma_flat = posterior["sigma"].values.reshape(n_total)
    a_flat = posterior["a"].values.reshape(n_total, J)

    # Per-classroom posterior mean and 95% credible interval.
    a_mean = a_flat.mean(axis=0)
    a_lo = np.quantile(a_flat, 0.025, axis=0)
    a_hi = np.quantile(a_flat, 0.975, axis=0)

    # Posterior summaries for τ and σ (truncated to deciles for compact JSON).
    tau_samples_summary = {
        "mean": float(tau_flat.mean()),
        "median": float(np.median(tau_flat)),
        "lo": float(np.quantile(tau_flat, 0.025)),
        "hi": float(np.quantile(tau_flat, 0.975)),
    }
    sigma_samples_summary = {
        "mean": float(sigma_flat.mean()),
        "median": float(np.median(sigma_flat)),
        "lo": float(np.quantile(sigma_flat, 0.025)),
        "hi": float(np.quantile(sigma_flat, 0.975)),
    }

    # Sub-sample τ for visualization (full chain too large for JSON).
    rng2 = np.random.default_rng(SEED)
    sample_idx = rng2.choice(n_total, size=min(800, n_total), replace=False)
    tau_thinned = tau_flat[sample_idx]

    # Diagnostics.
    summary = az.summary(idata, var_names=["tau", "sigma"], round_to=4)

    return {
        "prior_scale": prior_scale,
        "n_chains": n_chains,
        "n_draws_per_chain": n_draws,
        "tau_summary": tau_samples_summary,
        "sigma_summary": sigma_samples_summary,
        "tau_samples": [round(float(v), 4) for v in tau_thinned.tolist()],
        "a_mean": [round(float(v), 4) for v in a_mean.tolist()],
        "a_lo": [round(float(v), 4) for v in a_lo.tolist()],
        "a_hi": [round(float(v), 4) for v in a_hi.tolist()],
        "rhat_tau": float(summary.loc["tau", "r_hat"]),
        "ess_bulk_tau": int(summary.loc["tau", "ess_bulk"]),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    t0 = time.time()
    classroom, x, y, a_true = generate_data()
    print(f"Generated six-classroom data: N={N}, J={J}, seed={SEED}")
    print(f"  True (α, β, τ, σ) = ({ALPHA_TRUE}, {BETA_TRUE}, {TAU_TRUE}, {SIGMA_TRUE})")

    print("Fitting REML reference …")
    reml = reml_reference(classroom, x, y)
    print(f"  REML α̂={reml['alpha_hat']:.3f}, β̂={reml['beta_hat']:.3f}")
    print(f"  REML τ̂²={reml['tau_sq_hat']:.3f}, σ̂²={reml['sigma_sq_hat']:.3f}")

    fits = []
    for ps in PRIOR_SCALES:
        fits.append(bayesian_fit_at_scale(classroom, x, y, ps))

    payload = {
        "metadata": {
            "model": "random_intercept_non_centered",
            "engine": "PyMC",
            "pymc_version": pm.__version__,
            "n_classrooms": J,
            "group_sizes": GROUP_SIZES.tolist(),
            "n_obs": N,
            "seed": SEED,
            "true_values": {
                "alpha": ALPHA_TRUE,
                "beta": BETA_TRUE,
                "tau": TAU_TRUE,
                "sigma": SIGMA_TRUE,
            },
            "prior_scales": PRIOR_SCALES,
        },
        "data": {
            "classroom": classroom.tolist(),
            "x": [round(float(v), 4) for v in x.tolist()],
            "y": [round(float(v), 4) for v in y.tolist()],
            "a_true": [round(float(v), 4) for v in a_true.tolist()],
        },
        "reml": reml,
        "fits": fits,
    }

    body = json.dumps(payload, separators=(",", ":"))
    for d in OUT_DIRS:
        out_path = d / OUT_FILENAME
        out_path.write_text(body)
        print(f"  Wrote {out_path.relative_to(REPO_ROOT)} ({len(body) / 1024:.1f} KB)")

    print(f"Done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
