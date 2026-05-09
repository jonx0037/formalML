"""Precompute §10 diabetes-benchmark fits as JSON for DiabetesPredictiveLogLoss viz.

Companion to ``01_sparse_bayesian_priors.ipynb``. Mirrors §10 of the notebook —
five-method comparison (ridge baseline + horseshoe + regularized horseshoe +
continuous spike-slab + R2-D2) on the Efron–Hastie–Johnstone–Tibshirani (2004)
diabetes dataset under a single 70/30 train/test split. Uses a bumped chain
budget (4 chains × 500 draws × 500 tune at target_accept=0.95) relative to the
in-notebook lightweight version (2 × 200 × 200) to demonstrate convergence at
ship quality.

Per CLAUDE.md "Sample-data dual-location": JSON is written to BOTH
``src/data/sampleData/sparse-bayesian-priors/`` (canonical, source-controlled)
and ``public/sample-data/sparse-bayesian-priors/`` (Astro-served at runtime).

Usage::

    cd notebooks/sparse-bayesian-priors
    .venv/bin/python precompute_diabetes.py

Runtime: ~3-8 minutes on a 2020-era laptop (5 fits at p=10, n_train=309).
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
from scipy import stats
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pytensor")

# --------------------------------------------------------------------------- #
# Paths and reproducibility
# --------------------------------------------------------------------------- #

SEED = 20260509
NOTEBOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NOTEBOOK_DIR.parents[1]

OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "sparse-bayesian-priors",
    REPO_ROOT / "public" / "sample-data" / "sparse-bayesian-priors",
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

OUT_FILENAME = "diabetes_predictive.json"

NUTS_KW = dict(
    draws=500,
    tune=500,
    chains=4,
    cores=2,  # PyMC parallel chains; cap at 2 to avoid laptop oversubscription
    target_accept=0.95,
    random_seed=SEED,
    progressbar=False,
    return_inferencedata=True,
)

# --------------------------------------------------------------------------- #
# JSON serialization helpers (mirrored from precompute_neals_funnel.py)
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


def _round_floats(arr, ndigits: int = 5):
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
# Model factories (mirror notebook §10 exactly)
# --------------------------------------------------------------------------- #


def fit_ridge(X, y, prior_scale: float = 1.0) -> az.InferenceData:
    p_loc = X.shape[1]
    with pm.Model():
        beta = pm.Normal("beta", 0.0, prior_scale, shape=p_loc)
        sigma = pm.HalfNormal("sigma", 2.0)
        pm.Normal("y", mu=pm.math.dot(X, beta), sigma=sigma, observed=y)
        return pm.sample(**NUTS_KW)


def fit_horseshoe(X, y) -> az.InferenceData:
    p_loc = X.shape[1]
    with pm.Model():
        tau = pm.HalfCauchy("tau", 1.0)
        lam = pm.HalfCauchy("lam", 1.0, shape=p_loc)
        z = pm.Normal("z", 0.0, 1.0, shape=p_loc)
        beta = pm.Deterministic("beta", z * lam * tau)
        sigma = pm.HalfNormal("sigma", 2.0)
        pm.Normal("y", mu=pm.math.dot(X, beta), sigma=sigma, observed=y)
        return pm.sample(**NUTS_KW)


def fit_regularized_horseshoe(X, y, m0: float = 3.0, c: float = 2.0) -> az.InferenceData:
    p_loc = X.shape[1]
    n_loc = X.shape[0]
    tau0 = (m0 / (p_loc - m0)) * (1.0 / np.sqrt(n_loc))
    with pm.Model():
        tau = pm.HalfCauchy("tau", tau0)
        lam = pm.HalfCauchy("lam", 1.0, shape=p_loc)
        lam_tilde2 = (c**2 * lam**2) / (c**2 + tau**2 * lam**2)
        z = pm.Normal("z", 0.0, 1.0, shape=p_loc)
        beta = pm.Deterministic("beta", z * pm.math.sqrt(lam_tilde2) * tau)
        sigma = pm.HalfNormal("sigma", 2.0)
        pm.Normal("y", mu=pm.math.dot(X, beta), sigma=sigma, observed=y)
        return pm.sample(**NUTS_KW)


def fit_spike_slab_continuous(
    X, y, pi_p: float = 0.3, tau_slab: float = 2.0
) -> az.InferenceData:
    """HMC-compatible continuous spike-and-slab via Mixture of Normals."""
    p_loc = X.shape[1]
    tau_spike = 0.01
    with pm.Model():
        components = [pm.Normal.dist(0.0, tau_spike), pm.Normal.dist(0.0, tau_slab)]
        beta = pm.Mixture(
            "beta", w=[1 - pi_p, pi_p], comp_dists=components, shape=p_loc
        )
        sigma = pm.HalfNormal("sigma", 2.0)
        pm.Normal("y", mu=pm.math.dot(X, beta), sigma=sigma, observed=y)
        return pm.sample(**NUTS_KW)


def fit_r2d2(X, y, a_r2: float = 0.5, b_r2: float = 5.0, xi: float = 0.5) -> az.InferenceData:
    p_loc = X.shape[1]
    with pm.Model():
        R2 = pm.Beta("R2", a_r2, b_r2)
        phi = pm.Dirichlet("phi", a=np.full(p_loc, xi))
        sigma = pm.HalfNormal("sigma", 2.0)
        var_per = sigma**2 * (R2 / (1.0 - R2)) * phi
        z = pm.Normal("z", 0.0, 1.0, shape=p_loc)
        beta = pm.Deterministic("beta", z * pm.math.sqrt(var_per))
        pm.Normal("y", mu=pm.math.dot(X, beta), sigma=sigma, observed=y)
        return pm.sample(**NUTS_KW)


# --------------------------------------------------------------------------- #
# Posterior summarization
# --------------------------------------------------------------------------- #


def summarize_predictive(idata: az.InferenceData, X_te, y_te, name: str) -> dict:
    """Compute test predictive log-likelihood + bootstrap CI + diagnostics."""
    posterior = idata.posterior
    sample_stats = idata.sample_stats

    # Every fit_* factory wraps β as pm.Deterministic("beta", ...), so beta is
    # always present in posterior — no fallback needed.
    beta_da = posterior["beta"]
    sigma_da = posterior["sigma"]

    n_chains = posterior.sizes["chain"]
    n_draws = posterior.sizes["draw"]
    n_total = n_chains * n_draws
    p_loc = beta_da.shape[-1]

    beta_samp = beta_da.values.reshape(n_total, p_loc)  # (S, P)
    sigma_samp = sigma_da.values.reshape(n_total)

    # Posterior moments per coefficient
    beta_mean = beta_samp.mean(axis=0)
    beta_q05 = np.quantile(beta_samp, 0.05, axis=0)
    beta_q95 = np.quantile(beta_samp, 0.95, axis=0)

    # Test predictive log-likelihood (per-observation, mean over posterior)
    n_eval = min(n_total, 400)
    idx_eval = np.linspace(0, n_total - 1, n_eval).astype(int)
    test_lls = np.empty(n_eval)
    for i, s in enumerate(idx_eval):
        pred_mu = X_te @ beta_samp[s]
        test_lls[i] = float(stats.norm.logpdf(y_te, pred_mu, sigma_samp[s]).mean())

    test_ll_mean = float(test_lls.mean())
    test_ll_se = float(test_lls.std(ddof=1) / np.sqrt(n_eval))

    # Bootstrap 95% CI on the per-observation log-likelihood.
    n_boot = 2000
    rng_boot = np.random.default_rng(SEED + abs(hash(name)) % 1024)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sel = rng_boot.choice(n_eval, size=n_eval, replace=True)
        boot_means[b] = test_lls[sel].mean()
    ci_low = float(np.quantile(boot_means, 0.025))
    ci_high = float(np.quantile(boot_means, 0.975))

    divs = int(sample_stats["diverging"].values.sum())
    summary = az.summary(idata, var_names=["sigma"], round_to=4)
    rhat_sigma = float(summary.loc["sigma", "r_hat"])
    ess_sigma = int(summary.loc["sigma", "ess_bulk"])

    return {
        "name": name,
        "test_ll_mean": test_ll_mean,
        "test_ll_se": test_ll_se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "divergences": divs,
        "rhat_sigma": rhat_sigma,
        "ess_sigma": ess_sigma,
        "beta_mean": _round_floats(_to_jsonable(beta_mean)),
        "beta_q05": _round_floats(_to_jsonable(beta_q05)),
        "beta_q95": _round_floats(_to_jsonable(beta_q95)),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    t0 = time.time()
    data = load_diabetes()
    X_diab = (data.data - data.data.mean(0)) / data.data.std(0)
    y_diab = (data.target - data.target.mean()) / data.target.std()
    feature_names = list(data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_diab, y_diab, test_size=0.3, random_state=SEED
    )
    print(
        f"Diabetes: n={len(y_diab)}, p={X_diab.shape[1]}, train={len(y_train)}, test={len(y_test)}"
    )

    fits_spec = [
        ("Ridge", "#7f7f7f", fit_ridge),
        ("Horseshoe", "#1f4e79", fit_horseshoe),
        ("Reg. horseshoe", "#3a6e3a", fit_regularized_horseshoe),
        ("Spike-slab", "#7b3c10", fit_spike_slab_continuous),
        ("R2-D2", "#2e7baa", fit_r2d2),
    ]

    results = []
    for name, color, fit_fn in fits_spec:
        t1 = time.time()
        print(f"  Fitting {name} ...")
        idata = fit_fn(X_train, y_train)
        record = summarize_predictive(idata, X_test, y_test, name)
        record["color"] = color
        elapsed = time.time() - t1
        record["elapsed_seconds"] = round(elapsed, 1)
        print(
            f"    {name:<18} ll = {record['test_ll_mean']:8.4f} "
            f"± {record['test_ll_se']:.4f} | divs = {record['divergences']:>3} "
            f"| Rhat(σ) = {record['rhat_sigma']:.3f} | {elapsed:.1f}s"
        )
        results.append(record)

    payload = {
        "metadata": {
            "dataset": "diabetes (Efron-Hastie-Johnstone-Tibshirani 2004)",
            "n": int(len(y_diab)),
            "p": int(X_diab.shape[1]),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "test_size": 0.3,
            "feature_names": feature_names,
            "seed": SEED,
            "pymc_version": pm.__version__,
            "nuts_kwargs": {
                k: v for k, v in NUTS_KW.items() if k != "random_seed"
            },
            "total_seconds": round(time.time() - t0, 1),
        },
        "methods": results,
    }

    # _to_jsonable converts non-finite floats (NaN/inf) to None — JSON spec
    # disallows NaN/Infinity literals, and browser JSON.parse rejects them.
    body = json.dumps(_to_jsonable(payload), separators=(",", ":"), allow_nan=False)
    for d in OUT_DIRS:
        out_path = d / OUT_FILENAME
        out_path.write_text(body)
        print(
            f"  Wrote {out_path.relative_to(REPO_ROOT)} ({len(body) / 1024:.1f} KB)"
        )

    print(f"Done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
