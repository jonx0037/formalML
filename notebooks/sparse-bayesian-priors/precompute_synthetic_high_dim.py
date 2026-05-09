"""Precompute §9 production-scale synthetic high-dim regression as JSON.

Companion to ``01_sparse_bayesian_priors.ipynb``. Mirrors §9 of the notebook
but at *production scaling*: ``p = 200, n_train = 100, n_test = 300,
k_true = 10`` versus the in-notebook lightweight ``p = 30, n = 60, k = 3``.
The increased dimensionality stresses the funnel pathology and the active-set
inclusion-uncertainty story the brief promises.

Fits the four sparse priors via PyMC NUTS at full chain budget
(4 chains × 1000 draws × 1000 tune at target_accept=0.95):

    1. Horseshoe (Carvalho-Polson-Scott 2010), non-centered parameterization
    2. Regularized horseshoe (Piironen-Vehtari 2017) with calibrated tau0
    3. Continuous spike-and-slab (mixture of N(0, tau_spike) and N(0, tau_slab))
    4. R2-D2 (Zhang-Reich-Bondell 2022)

Writes ``synthetic_high_dim.json`` with per-coefficient posterior summaries,
inclusion probabilities at multiple thresholds, divergence counts, R-hat /
ESS diagnostics, and test predictive log-likelihood. The
``ActiveSetRecovery.tsx`` viz consumes this JSON.

Per CLAUDE.md "Sample-data dual-location": JSON written to BOTH
``src/data/sampleData/sparse-bayesian-priors/`` and
``public/sample-data/sparse-bayesian-priors/``.

Usage::

    cd notebooks/sparse-bayesian-priors
    nohup .venv/bin/python precompute_synthetic_high_dim.py </dev/null \
        >precompute_synthetic.log 2>&1 &

Runtime: ~20-45 minutes on a 2020-era laptop (4 fits × 5-15 min each).
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

OUT_FILENAME = "synthetic_high_dim.json"

# Production-scale data dimensions
P = 200
N_TRAIN = 100
N_TEST = 300
K_TRUE = 10
SIGMA_Y = 1.0

# NUTS budget — full chain budget for production fits
NUTS_KW = dict(
    draws=1000,
    tune=1000,
    chains=4,
    cores=2,  # parallel pairs to stay under laptop oversubscription
    target_accept=0.95,
    random_seed=SEED,
    progressbar=False,
    return_inferencedata=True,
)

INCLUSION_THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.5]

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
# Synthetic data generation
# --------------------------------------------------------------------------- #


def make_synthetic_data():
    """Build the production-scale sparse regression problem.

    True coefficients: K_TRUE = 10 active coefficients with mixed magnitudes
    spanning the {moderate, large, signal-to-noise borderline} regimes —
    designed to stress sparse-prior recovery in the regime where horseshoe vs
    Bayesian-LASSO geometric differences (§5) become visible.
    """
    rng_data = np.random.default_rng(SEED)
    beta_true = np.zeros(P)
    # Spread active coordinates across the index range for visual distinctness
    active_idx = np.linspace(8, P - 8, K_TRUE).astype(int)
    # Magnitudes: mix of moderate / large / borderline; alternating signs
    magnitudes = np.array([2.5, -1.8, 1.4, -2.1, 1.0, -1.6, 2.3, -1.2, 1.7, -2.0])
    beta_true[active_idx] = magnitudes

    X_train = rng_data.standard_normal((N_TRAIN, P))
    y_train = X_train @ beta_true + rng_data.standard_normal(N_TRAIN) * SIGMA_Y
    X_test = rng_data.standard_normal((N_TEST, P))
    y_test = X_test @ beta_true + rng_data.standard_normal(N_TEST) * SIGMA_Y

    return X_train, y_train, X_test, y_test, beta_true, active_idx


# --------------------------------------------------------------------------- #
# Model factories — mirror notebook §9 with p replaced by P (=200)
# --------------------------------------------------------------------------- #


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


def fit_regularized_horseshoe(X, y, m0: float = K_TRUE, c: float = 2.0) -> az.InferenceData:
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
    X, y, pi_p: float = K_TRUE / P, tau_slab: float = 2.0
) -> az.InferenceData:
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


def summarize_recovery(
    idata: az.InferenceData, X_te, y_te, beta_true, name: str
) -> dict:
    posterior = idata.posterior
    sample_stats = idata.sample_stats

    if "beta" in posterior:
        beta_da = posterior["beta"]
    else:  # fallback for the no-deterministic horseshoe; not used in this script
        beta_da = posterior["z"] * posterior["lam"] * posterior["tau"]

    sigma_da = posterior["sigma"]

    n_chains = posterior.sizes["chain"]
    n_draws = posterior.sizes["draw"]
    n_total = n_chains * n_draws
    p_loc = beta_da.shape[-1]

    beta_samp = beta_da.values.reshape(n_total, p_loc)  # (S, P)
    sigma_samp = sigma_da.values.reshape(n_total)

    beta_mean = beta_samp.mean(axis=0)
    beta_q05 = np.quantile(beta_samp, 0.05, axis=0)
    beta_q95 = np.quantile(beta_samp, 0.95, axis=0)

    # Inclusion probabilities at multiple thresholds for the slider in
    # ActiveSetRecovery.tsx — store all as a lookup.
    abs_beta = np.abs(beta_samp)
    inclusion = {
        f"thr_{t:.2f}".replace(".", "_"): (abs_beta > t).mean(axis=0).tolist()
        for t in INCLUSION_THRESHOLDS
    }

    # Coefficient-recovery error — useful diagnostic
    rmse = float(np.sqrt(np.mean((beta_mean - beta_true) ** 2)))

    # Test predictive log-likelihood
    n_eval = min(n_total, 500)
    idx_eval = np.linspace(0, n_total - 1, n_eval).astype(int)
    test_lls = np.empty(n_eval)
    for i, s in enumerate(idx_eval):
        pred_mu = X_te @ beta_samp[s]
        test_lls[i] = float(stats.norm.logpdf(y_te, pred_mu, sigma_samp[s]).mean())
    test_ll_mean = float(test_lls.mean())
    test_ll_se = float(test_lls.std(ddof=1) / np.sqrt(n_eval))

    # Diagnostics
    divs = int(sample_stats["diverging"].values.sum())
    summary = az.summary(idata, var_names=["sigma"], round_to=4)
    rhat_sigma = float(summary.loc["sigma", "r_hat"])
    ess_sigma = int(summary.loc["sigma", "ess_bulk"])

    # Per-coefficient R-hat / ESS — take max R-hat and min ESS across the P
    # coefficients as scalar diagnostics.
    beta_summary = az.summary(idata, var_names=["beta"], round_to=4)
    rhat_beta_max = float(beta_summary["r_hat"].max())
    ess_beta_min = int(beta_summary["ess_bulk"].min())

    return {
        "name": name,
        "beta_mean": _round_floats(_to_jsonable(beta_mean)),
        "beta_q05": _round_floats(_to_jsonable(beta_q05)),
        "beta_q95": _round_floats(_to_jsonable(beta_q95)),
        "inclusion": {k: _round_floats(v) for k, v in inclusion.items()},
        "rmse": round(rmse, 5),
        "test_ll_mean": round(test_ll_mean, 5),
        "test_ll_se": round(test_ll_se, 5),
        "divergences": divs,
        "rhat_sigma": rhat_sigma,
        "ess_sigma": ess_sigma,
        "rhat_beta_max": rhat_beta_max,
        "ess_beta_min": ess_beta_min,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    t0 = time.time()
    print(f"Synthetic high-dim regression: P={P}, N_TRAIN={N_TRAIN}, N_TEST={N_TEST}, K_TRUE={K_TRUE}")
    X_train, y_train, X_test, y_test, beta_true, active_idx = make_synthetic_data()
    print(f"True active indices: {active_idx.tolist()}")
    print(f"True magnitudes:     {beta_true[active_idx].round(3).tolist()}")
    print(f"NUTS budget: {NUTS_KW['chains']} chains × {NUTS_KW['draws']} draws × "
          f"{NUTS_KW['tune']} tune (target_accept={NUTS_KW['target_accept']})")
    print()

    fits_spec = [
        ("Horseshoe", "#1f4e79", fit_horseshoe),
        ("Reg. horseshoe", "#3a6e3a", fit_regularized_horseshoe),
        ("Spike-slab", "#7b3c10", fit_spike_slab_continuous),
        ("R2-D2", "#2e7baa", fit_r2d2),
    ]

    results = []
    for name, color, fit_fn in fits_spec:
        t1 = time.time()
        print(f"  Fitting {name} ...", flush=True)
        idata = fit_fn(X_train, y_train)
        record = summarize_recovery(idata, X_test, y_test, beta_true, name)
        record["color"] = color
        elapsed = time.time() - t1
        record["elapsed_seconds"] = round(elapsed, 1)
        incl_at_active = [
            round(record["inclusion"]["thr_0_10"][i], 3) for i in active_idx
        ]
        print(
            f"    {name:<18} divs={record['divergences']:>3} "
            f"| Rhat_max={record['rhat_beta_max']:.3f} "
            f"| ESS_min={record['ess_beta_min']:>5} "
            f"| RMSE={record['rmse']:.3f} "
            f"| ll={record['test_ll_mean']:7.4f} "
            f"| {elapsed:.1f}s",
            flush=True,
        )
        print(f"      inclusion@0.10 at true active: {incl_at_active}", flush=True)
        results.append(record)

    payload = {
        "metadata": {
            "p": P,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "k_true": K_TRUE,
            "sigma_y": SIGMA_Y,
            "active_indices": _to_jsonable(active_idx),
            "beta_true": _round_floats(_to_jsonable(beta_true)),
            "seed": SEED,
            "pymc_version": pm.__version__,
            "nuts_kwargs": {
                k: v for k, v in NUTS_KW.items() if k != "random_seed"
            },
            "inclusion_thresholds": INCLUSION_THRESHOLDS,
            "total_seconds": round(time.time() - t0, 1),
        },
        "priors": results,
    }

    body = json.dumps(payload, separators=(",", ":"))
    for d in OUT_DIRS:
        out_path = d / OUT_FILENAME
        out_path.write_text(body)
        print(
            f"  Wrote {out_path.relative_to(REPO_ROOT)} ({len(body) / 1024:.1f} KB)"
        )

    print(f"Done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
