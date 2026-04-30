"""Precompute PyMC stacking pipeline + distillation grid outputs as JSON.

Companion to ``01_stacking_and_predictive_ensembles.ipynb``. This script mirrors
the notebook's data-generating process and four-candidate Bayesian fit
(BLR / BPR-4 / GP / BART) at four training sizes, then writes per-observation
LOO log densities, stacking + BB-pseudo-BMA weights, and held-out log-scores
to JSON files under ``src/data/sampleData/stacking-and-predictive-ensembles/``
for the formalML viz components to consume.

It also constructs a distillation grid: a 200-point predictive band from the
stacked teacher and five student fits (linear, BPR-3/6/10, MLP) at four
distillation-dataset sizes (50, 100, 200, 400 teacher samples).

Usage::

    cd notebooks/stacking-and-predictive-ensembles
    .venv/bin/python serialize_for_viz.py

Runtime: ~10-15 minutes on a 2020-era laptop.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pymc_bart as pmb
import scipy.optimize as sopt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths and reproducibility
# --------------------------------------------------------------------------- #

SEED = 20260430
NOTEBOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NOTEBOOK_DIR.parents[1]  # formalML/
OUT_DIR = REPO_ROOT / "src" / "data" / "sampleData" / "stacking-and-predictive-ensembles"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_NAMES = ["BLR", "BPR-4", "GP", "BART"]

# 200-point deterministic eval grid in [0, 1]. Used for held-out log-score
# *and* as the x-axis for the distillation predictive bands.
X_EVAL = np.linspace(0.0, 1.0, 200)
SIGMA = 0.25


def true_function(x: np.ndarray) -> np.ndarray:
    """DGP: smooth global sinusoid plus a localized higher-frequency wiggle past x = 0.5."""
    return np.sin(2 * np.pi * x) + 0.4 * (x > 0.5) * np.cos(6 * np.pi * x)


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
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def write_json(path: Path, payload: dict) -> int:
    with open(path, "w") as f:
        json.dump(_to_jsonable(payload), f)
    return path.stat().st_size


# --------------------------------------------------------------------------- #
# Posterior-predictive log density (Monte-Carlo over posterior samples)
# --------------------------------------------------------------------------- #

def predictive_logpdf_mc(idata, model_fn, x_eval, y_eval) -> np.ndarray:
    """Per-observation log p(y_eval[i] | x_eval[i], data) via MC over the posterior."""
    posterior = idata.posterior
    n_chains = posterior.sizes["chain"]
    n_draws = posterior.sizes["draw"]
    n_eval = len(x_eval)
    log_p = np.full((n_chains * n_draws, n_eval), -np.inf)
    s = 0
    for c in range(n_chains):
        for d in range(n_draws):
            params = {v: posterior[v].values[c, d] for v in posterior.data_vars}
            mean_s, std_s = model_fn(params, x_eval)
            log_p[s] = stats.norm.logpdf(y_eval, loc=mean_s, scale=std_s)
            s += 1
    return np.logaddexp.reduce(log_p, axis=0) - np.log(log_p.shape[0])


def gp_marginal_loo_pointwise(idata_gp, X_train, y_train) -> np.ndarray:
    """Closed-form Rasmussen & Williams (2006) eq. 5.12-5.13 leave-one-out predictive
    densities for a marginal GP, MC-averaged over posterior hyperparameter samples."""
    posterior = idata_gp.posterior
    eta = posterior["eta"].values.reshape(-1)
    ell = posterior["ell"].values.reshape(-1)
    sigma_y = posterior["sigma_y"].values.reshape(-1)
    S, n = len(eta), len(y_train)
    log_p = np.empty((S, n))
    pairwise_sq = (X_train[:, None] - X_train[None, :]) ** 2
    I_n = np.eye(n)
    for s in range(S):
        K = eta[s] ** 2 * np.exp(-0.5 * pairwise_sq / ell[s] ** 2)
        K_y = K + sigma_y[s] ** 2 * I_n
        K_y_inv = np.linalg.inv(K_y)
        alpha = K_y_inv @ y_train
        diag = np.diag(K_y_inv)
        loo_mean = y_train - alpha / diag
        loo_var = 1.0 / diag
        log_p[s] = stats.norm.logpdf(y_train, loc=loo_mean, scale=np.sqrt(loo_var))
    return np.logaddexp.reduce(log_p, axis=0) - np.log(S)


def gp_predictive_logpdf(idata_gp, x_train, y_train, x_eval, y_eval) -> np.ndarray:
    """Posterior predictive density for a marginal GP at held-out points,
    MC-averaged over posterior hyperparameter samples (closed-form per draw)."""
    posterior = idata_gp.posterior
    eta = posterior["eta"].values.reshape(-1)
    ell = posterior["ell"].values.reshape(-1)
    sigma_y = posterior["sigma_y"].values.reshape(-1)
    S, n_eval = len(eta), len(x_eval)
    n_train = len(x_train)
    log_p = np.empty((S, n_eval))
    pairwise_train = (x_train[:, None] - x_train[None, :]) ** 2
    cross = (x_eval[:, None] - x_train[None, :]) ** 2
    I_n = np.eye(n_train)
    for s in range(S):
        K = eta[s] ** 2 * np.exp(-0.5 * pairwise_train / ell[s] ** 2)
        K_y = K + sigma_y[s] ** 2 * I_n
        L = np.linalg.cholesky(K_y)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        K_star = eta[s] ** 2 * np.exp(-0.5 * cross / ell[s] ** 2)  # (n_eval, n_train)
        pred_mean = K_star @ alpha
        v = np.linalg.solve(L, K_star.T)
        pred_var_latent = eta[s] ** 2 - np.sum(v ** 2, axis=0)
        pred_var = np.maximum(pred_var_latent, 0.0) + sigma_y[s] ** 2
        log_p[s] = stats.norm.logpdf(y_eval, loc=pred_mean, scale=np.sqrt(pred_var))
    return np.logaddexp.reduce(log_p, axis=0) - np.log(S)


def gp_predictive_mean_var(idata_gp, x_train, y_train, x_eval) -> tuple[np.ndarray, np.ndarray]:
    """Posterior-mean of the GP predictive mean and predictive *latent* variance at x_eval,
    averaged over posterior hyperparameter samples. Used for figures/bands.
    Returns (mean, var_latent + noise) — the marginal predictive distribution variance."""
    posterior = idata_gp.posterior
    eta = posterior["eta"].values.reshape(-1)
    ell = posterior["ell"].values.reshape(-1)
    sigma_y = posterior["sigma_y"].values.reshape(-1)
    S, n_eval = len(eta), len(x_eval)
    n_train = len(x_train)
    means = np.empty((S, n_eval))
    variances = np.empty((S, n_eval))
    pairwise_train = (x_train[:, None] - x_train[None, :]) ** 2
    cross = (x_eval[:, None] - x_train[None, :]) ** 2
    I_n = np.eye(n_train)
    for s in range(S):
        K = eta[s] ** 2 * np.exp(-0.5 * pairwise_train / ell[s] ** 2)
        K_y = K + sigma_y[s] ** 2 * I_n
        L = np.linalg.cholesky(K_y)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        K_star = eta[s] ** 2 * np.exp(-0.5 * cross / ell[s] ** 2)
        pred_mean = K_star @ alpha
        v = np.linalg.solve(L, K_star.T)
        pred_var = np.maximum(eta[s] ** 2 - np.sum(v ** 2, axis=0), 0.0) + sigma_y[s] ** 2
        means[s] = pred_mean
        variances[s] = pred_var
    return means.mean(axis=0), variances.mean(axis=0)


# --------------------------------------------------------------------------- #
# Stacking + BB-pseudo-BMA weights from the LOO log-density matrix
# --------------------------------------------------------------------------- #

def stacking_weights_softmax(P_log: np.ndarray) -> np.ndarray:
    """Maximize sum_i log(sum_k w_k exp P_log[i,k]) over the simplex via L-BFGS-B
    with softmax reparameterization (matches notebook §4)."""
    K = P_log.shape[1]

    def neg_score(theta):
        w = np.exp(theta - theta.max())
        w = w / w.sum()
        log_mix = np.logaddexp.reduce(P_log + np.log(w + 1e-300), axis=1)
        return -log_mix.sum()

    res = sopt.minimize(neg_score, np.zeros(K), method="L-BFGS-B")
    w = np.exp(res.x - res.x.max())
    return w / w.sum()


def pseudo_bma_weights(elpds: np.ndarray) -> np.ndarray:
    """BB-pseudo-BMA weights: softmax over LOO ELPDs (uniform model prior)."""
    e = elpds - elpds.max()
    w = np.exp(e)
    return w / w.sum()


def k_eff_inverse_simpson(w: np.ndarray) -> float:
    """Effective sample size 1 / sum w_i^2 (per the brief's spec)."""
    return float(1.0 / np.sum(w ** 2))


# --------------------------------------------------------------------------- #
# Full PyMC pipeline at one training size
# --------------------------------------------------------------------------- #

def fit_full_pipeline(n_train: int, rng: np.random.Generator) -> dict:
    """Fit BLR / BPR-4 / GP / BART at training size ``n_train`` and return a payload
    matching the ``pymc_pipeline_n*.json`` schema."""
    print(f"\n{'='*70}\n[n={n_train}] generating data and fitting four PyMC candidates\n{'='*70}")

    x_train = rng.uniform(0.0, 1.0, size=n_train)
    y_train = true_function(x_train) + SIGMA * rng.standard_normal(n_train)

    # Held-out: deterministic eval grid + fixed-seed noise for reproducibility.
    eval_rng = np.random.default_rng(SEED + 7)
    y_eval = true_function(X_EVAL) + SIGMA * eval_rng.standard_normal(len(X_EVAL))

    x_mean, x_std = float(x_train.mean()), float(x_train.std())

    def std_x(x):
        return (x - x_mean) / x_std

    warning = None

    # ----- BLR -----
    print("  [BLR] fitting ...")
    with pm.Model() as m_blr:
        alpha = pm.Normal("alpha", mu=0, sigma=5)
        beta = pm.Normal("beta", mu=0, sigma=5)
        sigma_y = pm.HalfNormal("sigma_y", sigma=2)
        mu = alpha + beta * x_train
        pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y_train)
        idata_blr = pm.sample(
            1000, tune=1000, chains=4, random_seed=SEED,
            progressbar=False, idata_kwargs={"log_likelihood": True},
        )

    def blr_predict(params, x):
        return params["alpha"] + params["beta"] * x, params["sigma_y"]

    lp_eval_blr = predictive_logpdf_mc(idata_blr, blr_predict, X_EVAL, y_eval)

    # ----- BPR-4 (degree 4 -> 5 betas) -----
    print("  [BPR-4] fitting ...")
    xs_tr = std_x(x_train)
    with pm.Model() as m_bpr:
        betas = pm.Normal("betas", mu=0, sigma=5, shape=5)
        sigma_y = pm.HalfNormal("sigma_y", sigma=2)
        mu = sum(betas[j] * xs_tr ** j for j in range(5))
        pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y_train)
        idata_bpr = pm.sample(
            1000, tune=1000, chains=4, random_seed=SEED,
            progressbar=False, idata_kwargs={"log_likelihood": True},
        )

    def bpr_predict(params, x):
        xs = std_x(x)
        mu = sum(params["betas"][j] * xs ** j for j in range(5))
        return mu, params["sigma_y"]

    lp_eval_bpr = predictive_logpdf_mc(idata_bpr, bpr_predict, X_EVAL, y_eval)

    # ----- GP -----
    print("  [GP] fitting ...")
    with pm.Model() as m_gp:
        eta = pm.HalfNormal("eta", sigma=2)
        ell = pm.InverseGamma("ell", alpha=5, beta=1)
        sigma_y = pm.HalfNormal("sigma_y", sigma=2)
        cov = eta ** 2 * pm.gp.cov.ExpQuad(1, ell)
        gp = pm.gp.Marginal(cov_func=cov)
        gp.marginal_likelihood("y_obs", X=x_train[:, None], y=y_train, sigma=sigma_y)
        idata_gp = pm.sample(
            1000, tune=1000, chains=4, random_seed=SEED,
            progressbar=False, idata_kwargs={"log_likelihood": True},
        )

    lp_eval_gp = gp_predictive_logpdf(idata_gp, x_train, y_train, X_EVAL, y_eval)

    # ----- BART -----
    print("  [BART] fitting ...")
    m_bart = None
    idata_bart = None
    lp_eval_bart = np.full(len(X_EVAL), -np.inf)
    try:
        with pm.Model() as m_bart_local:
            X_bart = pm.Data("X_bart", x_train[:, None])
            sigma_y = pm.HalfNormal("sigma_y", sigma=2)
            mu = pmb.BART("mu", X=X_bart, Y=y_train, m=50)
            pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y_train, shape=mu.shape)
            idata_bart = pm.sample(
                1000, tune=1000, chains=4, random_seed=SEED,
                progressbar=False, idata_kwargs={"log_likelihood": True},
            )
        m_bart = m_bart_local

        with m_bart:
            pm.set_data({"X_bart": X_EVAL[:, None]})
            pp_bart = pm.sample_posterior_predictive(
                idata_bart, var_names=["mu"], random_seed=SEED, progressbar=False,
            )
        mu_bart_eval = pp_bart.posterior_predictive["mu"].values.reshape(-1, len(X_EVAL))
        sigma_post_bart = idata_bart.posterior["sigma_y"].values.reshape(-1)
        n_pp = mu_bart_eval.shape[0]
        n_sigma = len(sigma_post_bart)
        n_use = min(n_pp, n_sigma)
        # Vectorize over (posterior sample, eval point) via NumPy broadcasting.
        # log_per_sample has shape (n_use, len(y_eval)); logaddexp.reduce along
        # axis 0 averages over posterior samples in log-space.
        log_per_sample = stats.norm.logpdf(
            y_eval[None, :],
            loc=mu_bart_eval[:n_use, :],
            scale=sigma_post_bart[:n_use, None],
        )
        lp_eval_bart = np.logaddexp.reduce(log_per_sample, axis=0) - np.log(n_use)
    except Exception as e:  # pragma: no cover - capture any BART failure
        warning = f"BART fit failed: {e}"
        print(f"  [BART] WARNING: {e}")
        m_bart = None
        idata_bart = None
        lp_eval_bart = np.full(len(X_EVAL), -np.inf)

    # ----- LOO log-densities (per-observation) -----
    loo_blr = az.loo(idata_blr, pointwise=True)
    loo_bpr = az.loo(idata_bpr, pointwise=True)
    lp_loo_blr = loo_blr.loo_i.values
    lp_loo_bpr = loo_bpr.loo_i.values
    lp_loo_gp = gp_marginal_loo_pointwise(idata_gp, x_train, y_train)
    loo_bart = None
    if idata_bart is not None:
        loo_bart = az.loo(idata_bart, pointwise=True)
        lp_loo_bart = loo_bart.loo_i.values
        loo_bart_se = float(loo_bart.se)
    else:
        lp_loo_bart = np.full(n_train, -np.inf)
        loo_bart_se = 0.0

    P_log = np.stack([lp_loo_blr, lp_loo_bpr, lp_loo_gp, lp_loo_bart], axis=1)

    elpds_loo = P_log.sum(axis=0)
    elpd_se = np.array([
        float(loo_blr.se),
        float(loo_bpr.se),
        float(np.std(lp_loo_gp, ddof=1) * np.sqrt(len(lp_loo_gp))),
        loo_bart_se,
    ])

    # ----- Weights -----
    if not np.all(np.isfinite(P_log)):
        # Drop BART if it failed; fall back to three-candidate stacking.
        finite_cols = np.isfinite(P_log).all(axis=0)
        P_finite = P_log[:, finite_cols]
        w_partial = stacking_weights_softmax(P_finite)
        w_stack = np.zeros(4)
        w_stack[np.where(finite_cols)[0]] = w_partial
        w_bma = pseudo_bma_weights(np.where(finite_cols, elpds_loo, -1e18))
    else:
        w_stack = stacking_weights_softmax(P_log)
        w_bma = pseudo_bma_weights(elpds_loo)

    # ----- Held-out (eval grid) log-scores -----
    LP_eval = np.stack([lp_eval_blr, lp_eval_bpr, lp_eval_gp, lp_eval_bart], axis=1)
    finite_eval = np.isfinite(LP_eval).all(axis=1)
    log_w_stack = np.log(np.clip(w_stack, 1e-12, None))
    log_w_bma = np.log(np.clip(w_bma, 1e-12, None))
    lp_stack_eval = np.array([
        np.logaddexp.reduce(LP_eval[i] + log_w_stack) for i in range(len(X_EVAL))
    ])
    lp_bma_eval = np.array([
        np.logaddexp.reduce(LP_eval[i] + log_w_bma) for i in range(len(X_EVAL))
    ])

    # Single best by elpd_loo (across the four candidates' LOO ELPDs).
    best_idx = int(np.argmax(elpds_loo))
    lp_select_eval = LP_eval[:, best_idx]

    payload = {
        "n": n_train,
        "x_train": x_train,
        "y_train": y_train,
        "candidates": list(CANDIDATE_NAMES),
        "elpd_loo": dict(zip(CANDIDATE_NAMES, elpds_loo)),
        "elpd_loo_se": dict(zip(CANDIDATE_NAMES, elpd_se)),
        "lpd_per_obs": {
            name: P_log[:, i] for i, name in enumerate(CANDIDATE_NAMES)
        },
        "w_bma": dict(zip(CANDIDATE_NAMES, w_bma)),
        "w_stack": dict(zip(CANDIDATE_NAMES, w_stack)),
        "K_eff_bma": k_eff_inverse_simpson(w_bma),
        "K_eff_stack": k_eff_inverse_simpson(w_stack),
        "holdout_log_score": {
            "single_best": float(np.mean(lp_select_eval[finite_eval])) if finite_eval.any() else None,
            "bma_mixture": float(np.mean(lp_bma_eval[finite_eval])) if finite_eval.any() else None,
            "stacking_mixture": float(np.mean(lp_stack_eval[finite_eval])) if finite_eval.any() else None,
        },
    }
    if warning is not None:
        payload["warning"] = warning

    # Carry the n=100 artefacts back to the caller for Pareto-k + distillation.
    extras = {
        "loo_blr": loo_blr,
        "loo_bpr": loo_bpr,
        "loo_bart": loo_bart,
        "idata_blr": idata_blr,
        "idata_bpr": idata_bpr,
        "idata_gp": idata_gp,
        "idata_bart": idata_bart,
        "m_bart": m_bart,
        "x_train": x_train,
        "y_train": y_train,
        "x_mean": x_mean,
        "x_std": x_std,
        "w_stack": w_stack,
    }
    return payload, extras


# --------------------------------------------------------------------------- #
# Pareto-k (n=100)
# --------------------------------------------------------------------------- #

def build_pareto_k_payload(n_train: int, extras: dict) -> dict:
    n = n_train
    k_blr = extras["loo_blr"].pareto_k.values
    k_bpr = extras["loo_bpr"].pareto_k.values
    k_gp = np.full(n, np.nan)  # closed-form LOO -> Pareto-k undefined
    k_bart = extras["loo_bart"].pareto_k.values if extras["loo_bart"] is not None else np.full(n, np.nan)
    return {
        "n": n,
        "x_train": extras["x_train"],
        "y_train": extras["y_train"],
        "pareto_k": {
            "BLR": k_blr,
            "BPR-4": k_bpr,
            "GP": k_gp,
            "BART": k_bart,
        },
    }


# --------------------------------------------------------------------------- #
# Distillation: teacher predictive at X_EVAL + 5 students × 4 dataset sizes
# --------------------------------------------------------------------------- #

def teacher_predictive(extras: dict, rng: np.random.Generator, n_samples_per_x: int = 60):
    """Sample from the stacked teacher at the evaluation grid X_EVAL.

    Returns ``(mean, lower_95, upper_95, log_score, inference_cost, samples)``
    plus the cached BART mu samples at X_EVAL for reuse in distillation training.
    """
    x_train = extras["x_train"]
    y_train = extras["y_train"]
    idata_blr = extras["idata_blr"]
    idata_bpr = extras["idata_bpr"]
    idata_gp = extras["idata_gp"]
    idata_bart = extras["idata_bart"]
    m_bart = extras["m_bart"]
    x_mean = extras["x_mean"]
    x_std = extras["x_std"]
    w_stack = extras["w_stack"]

    # Pre-compute BART mu at X_EVAL (single batched call).
    if idata_bart is not None and m_bart is not None:
        with m_bart:
            pm.set_data({"X_bart": X_EVAL[:, None]})
            pp_bart = pm.sample_posterior_predictive(
                idata_bart, var_names=["mu"], random_seed=SEED + 200, progressbar=False,
            )
        mu_bart_eval = pp_bart.posterior_predictive["mu"].values.reshape(-1, len(X_EVAL))
        sigma_bart = idata_bart.posterior["sigma_y"].values.reshape(-1)
    else:
        mu_bart_eval = None
        sigma_bart = None

    # Posterior predictive mean+var at X_EVAL via closed-form GP (averaged over hypers).
    gp_eval_mean, gp_eval_var = gp_predictive_mean_var(idata_gp, x_train, y_train, X_EVAL)

    posterior_blr = idata_blr.posterior
    posterior_bpr = idata_bpr.posterior

    blr_alpha = posterior_blr["alpha"].values.reshape(-1)
    blr_beta = posterior_blr["beta"].values.reshape(-1)
    blr_sigma = posterior_blr["sigma_y"].values.reshape(-1)
    bpr_betas = posterior_bpr["betas"].values.reshape(-1, 5)
    bpr_sigma = posterior_bpr["sigma_y"].values.reshape(-1)

    n_eval = len(X_EVAL)
    samples = np.empty((n_eval, n_samples_per_x))
    # candidate index per (x_j, s)
    cand_choice = rng.choice(4, size=(n_eval, n_samples_per_x), p=w_stack)

    xs_eval = (X_EVAL - x_mean) / x_std

    for j in range(n_eval):
        for s in range(n_samples_per_x):
            c = cand_choice[j, s]
            if c == 0:  # BLR
                idx = rng.integers(len(blr_alpha))
                mu_s = blr_alpha[idx] + blr_beta[idx] * X_EVAL[j]
                samples[j, s] = rng.normal(mu_s, blr_sigma[idx])
            elif c == 1:  # BPR-4
                idx = rng.integers(len(bpr_sigma))
                betas_s = bpr_betas[idx]
                xs_j = xs_eval[j]
                mu_s = float(sum(betas_s[k] * xs_j ** k for k in range(5)))
                samples[j, s] = rng.normal(mu_s, bpr_sigma[idx])
            elif c == 2:  # GP
                # Use posterior-mean predictive mean + posterior-mean predictive std.
                samples[j, s] = rng.normal(gp_eval_mean[j], np.sqrt(gp_eval_var[j]))
            else:  # BART
                if mu_bart_eval is None:
                    # Fall back to posterior-mean from one of the others if BART failed.
                    samples[j, s] = rng.normal(gp_eval_mean[j], np.sqrt(gp_eval_var[j]))
                else:
                    sample_idx = rng.integers(mu_bart_eval.shape[0])
                    sigma_idx = rng.integers(len(sigma_bart))
                    samples[j, s] = rng.normal(
                        mu_bart_eval[sample_idx, j], sigma_bart[sigma_idx]
                    )

    teacher_mean = samples.mean(axis=1)
    lower = np.percentile(samples, 2.5, axis=1)
    upper = np.percentile(samples, 97.5, axis=1)

    # Held-out log-score for the teacher: density at fixed-seed y_eval points.
    eval_rng = np.random.default_rng(SEED + 7)
    y_eval = true_function(X_EVAL) + SIGMA * eval_rng.standard_normal(n_eval)
    lp_blr = predictive_logpdf_mc(idata_blr,
                                   lambda p, x: (p["alpha"] + p["beta"] * x, p["sigma_y"]),
                                   X_EVAL, y_eval)

    def _bpr_predict(params, x):
        xs = (x - x_mean) / x_std
        mu = sum(params["betas"][k] * xs ** k for k in range(5))
        return mu, params["sigma_y"]

    lp_bpr = predictive_logpdf_mc(idata_bpr, _bpr_predict, X_EVAL, y_eval)
    lp_gp = gp_predictive_logpdf(idata_gp, x_train, y_train, X_EVAL, y_eval)
    if mu_bart_eval is not None:
        n_use = min(mu_bart_eval.shape[0], len(sigma_bart))
        # Same broadcasting trick as in fit_full_pipeline: shape (n_use, n_eval).
        log_per_sample = stats.norm.logpdf(
            y_eval[None, :],
            loc=mu_bart_eval[:n_use, :],
            scale=sigma_bart[:n_use, None],
        )
        lp_bart = np.logaddexp.reduce(log_per_sample, axis=0) - np.log(n_use)
    else:
        lp_bart = np.full(n_eval, -np.inf)
    LP_eval = np.stack([lp_blr, lp_bpr, lp_gp, lp_bart], axis=1)
    log_w_stack = np.log(np.clip(w_stack, 1e-12, None))
    finite_eval = np.isfinite(LP_eval).all(axis=1)
    if finite_eval.any():
        teacher_log_score = float(np.mean([
            np.logaddexp.reduce(LP_eval[i] + log_w_stack)
            for i in range(n_eval) if finite_eval[i]
        ]))
    else:
        teacher_log_score = float("nan")

    teacher_inference_cost = int(4 * 4000)  # 4 candidates × 4000 posterior samples per prediction

    return {
        "mean": teacher_mean,
        "lower_95": lower,
        "upper_95": upper,
        "log_score": teacher_log_score,
        "inference_cost": teacher_inference_cost,
        "samples": samples,
        "mu_bart_eval": mu_bart_eval,
        "sigma_bart": sigma_bart,
        "gp_eval_mean": gp_eval_mean,
        "gp_eval_var": gp_eval_var,
        "y_eval": y_eval,
        "LP_eval_per_candidate": LP_eval,
    }


def sample_teacher_at_inputs(x_inputs, n_samples_each, extras, teacher_cache, rng):
    """Draw ``n_samples_each`` teacher samples per input. Returns an array (n_inputs, n_samples_each)."""
    x_train = extras["x_train"]
    y_train = extras["y_train"]
    idata_blr = extras["idata_blr"]
    idata_bpr = extras["idata_bpr"]
    idata_gp = extras["idata_gp"]
    idata_bart = extras["idata_bart"]
    m_bart = extras["m_bart"]
    x_mean = extras["x_mean"]
    x_std = extras["x_std"]
    w_stack = extras["w_stack"]

    # Pre-compute BART mu at the new inputs.
    if idata_bart is not None and m_bart is not None:
        with m_bart:
            pm.set_data({"X_bart": x_inputs[:, None]})
            pp_bart = pm.sample_posterior_predictive(
                idata_bart, var_names=["mu"], random_seed=SEED + 300, progressbar=False,
            )
        mu_bart_inputs = pp_bart.posterior_predictive["mu"].values.reshape(-1, len(x_inputs))
        sigma_bart = teacher_cache["sigma_bart"]
    else:
        mu_bart_inputs = None
        sigma_bart = None

    posterior_blr = idata_blr.posterior
    posterior_bpr = idata_bpr.posterior
    blr_alpha = posterior_blr["alpha"].values.reshape(-1)
    blr_beta = posterior_blr["beta"].values.reshape(-1)
    blr_sigma = posterior_blr["sigma_y"].values.reshape(-1)
    bpr_betas = posterior_bpr["betas"].values.reshape(-1, 5)
    bpr_sigma = posterior_bpr["sigma_y"].values.reshape(-1)

    # Closed-form GP predictive (mean + total variance) at x_inputs.
    gp_input_mean, gp_input_var = gp_predictive_mean_var(idata_gp, x_train, y_train, x_inputs)

    n_inputs = len(x_inputs)
    samples = np.empty((n_inputs, n_samples_each))
    cand_choice = rng.choice(4, size=(n_inputs, n_samples_each), p=w_stack)
    xs_inputs = (x_inputs - x_mean) / x_std

    # Vectorize per-candidate by grouping the (j, s) flat positions where this
    # candidate was drawn, then sampling all of them in one NumPy call. This
    # eliminates the (n_inputs × n_samples_each) Python double-loop while
    # preserving identical results given the same `rng` state. We process
    # candidates in deterministic order (BLR → BPR-4 → GP → BART) so the rng
    # advances reproducibly across calls.
    flat_choice = cand_choice.reshape(-1)
    flat_x = np.broadcast_to(x_inputs[:, None], (n_inputs, n_samples_each)).reshape(-1)
    flat_xs = np.broadcast_to(xs_inputs[:, None], (n_inputs, n_samples_each)).reshape(-1)
    flat_gp_mean = np.broadcast_to(gp_input_mean[:, None], (n_inputs, n_samples_each)).reshape(-1)
    flat_gp_std = np.broadcast_to(np.sqrt(gp_input_var)[:, None], (n_inputs, n_samples_each)).reshape(-1)
    flat_samples = np.empty(flat_choice.shape, dtype=float)

    # BLR.
    blr_mask = flat_choice == 0
    if blr_mask.any():
        n_blr = int(blr_mask.sum())
        post = rng.integers(len(blr_alpha), size=n_blr)
        mu = blr_alpha[post] + blr_beta[post] * flat_x[blr_mask]
        flat_samples[blr_mask] = rng.normal(mu, blr_sigma[post])

    # BPR-4 (degree-4 polynomial in standardized x).
    bpr_mask = flat_choice == 1
    if bpr_mask.any():
        n_bpr = int(bpr_mask.sum())
        post = rng.integers(len(bpr_sigma), size=n_bpr)
        b = bpr_betas[post]                          # shape (n_bpr, 5)
        xs = flat_xs[bpr_mask]
        # mu_s = sum_k b[s,k] * xs[s]**k for k=0..4
        powers = np.stack([xs ** k for k in range(5)], axis=1)  # (n_bpr, 5)
        mu = (b * powers).sum(axis=1)
        flat_samples[bpr_mask] = rng.normal(mu, bpr_sigma[post])

    # GP (closed-form predictive at fixed posterior-marginalized hypers).
    gp_mask = flat_choice == 2
    if gp_mask.any():
        flat_samples[gp_mask] = rng.normal(flat_gp_mean[gp_mask], flat_gp_std[gp_mask])

    # BART, with GP fallback when BART evaluations weren't materialized at the
    # distillation inputs.
    bart_mask = flat_choice == 3
    if bart_mask.any():
        if mu_bart_inputs is None:
            flat_samples[bart_mask] = rng.normal(flat_gp_mean[bart_mask], flat_gp_std[bart_mask])
        else:
            n_bart = int(bart_mask.sum())
            sample_idx = rng.integers(mu_bart_inputs.shape[0], size=n_bart)
            sigma_idx = rng.integers(len(sigma_bart), size=n_bart)
            # We need the input-index for each masked flat position to look up the right column.
            input_idx = np.broadcast_to(np.arange(n_inputs)[:, None], (n_inputs, n_samples_each)).reshape(-1)
            mu = mu_bart_inputs[sample_idx, input_idx[bart_mask]]
            flat_samples[bart_mask] = rng.normal(mu, sigma_bart[sigma_idx])

    samples = flat_samples.reshape(n_inputs, n_samples_each)
    return samples


def fit_student_linear(x_distill, y_distill):
    """OLS with intercept and slope (deterministic, returns mean + degenerate band)."""
    m = LinearRegression().fit(x_distill[:, None], y_distill)
    mean = m.predict(X_EVAL[:, None])
    resid = y_distill - m.predict(x_distill[:, None])
    sigma_hat = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.1
    lower = mean - 1.96 * sigma_hat
    upper = mean + 1.96 * sigma_hat

    def _log_score_fn(y_eval, _mean=mean, _sigma=sigma_hat):
        return stats.norm.logpdf(y_eval, loc=_mean, scale=_sigma)

    return {
        "mean": mean,
        "lower_95": lower,
        "upper_95": upper,
        "sigma_hat": sigma_hat,
        "log_score_fn": _log_score_fn,
        "inference_cost": 1,
    }


def fit_student_bayesian_poly(x_distill, y_distill, degree, x_mean, x_std):
    """Bayesian polynomial regression via NUTS at the given degree."""
    xs_distill = (x_distill - x_mean) / x_std
    K = degree + 1
    with pm.Model():
        betas = pm.Normal("betas", mu=0, sigma=5, shape=K)
        sigma_y = pm.HalfNormal("sigma_y", sigma=2)
        mu = sum(betas[k] * xs_distill ** k for k in range(K))
        pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y_distill)
        idata = pm.sample(
            500, tune=500, chains=2, random_seed=SEED,
            progressbar=False, target_accept=0.9,
        )
    posterior = idata.posterior
    betas_samples = posterior["betas"].values.reshape(-1, K)
    sigma_samples = posterior["sigma_y"].values.reshape(-1)
    xs_eval = (X_EVAL - x_mean) / x_std

    # Predictive mean over X_EVAL: vectorize the polynomial evaluation across
    # posterior samples. `powers` has shape (len(X_EVAL), K); the matrix product
    # with the posterior beta matrix gives means_per_post of shape (n_post, n_eval).
    n_post = len(sigma_samples)
    powers = np.stack([xs_eval ** k for k in range(K)], axis=1)  # (n_eval, K)
    means_per_post = betas_samples @ powers.T                     # (n_post, n_eval)

    pred_mean = means_per_post.mean(axis=0)
    # Predictive band: marginalize noise via NumPy broadcasting in a single call.
    rng_local = np.random.default_rng(SEED + degree)
    samples_pred = rng_local.normal(means_per_post, sigma_samples[:, None])
    lower = np.percentile(samples_pred, 2.5, axis=0)
    upper = np.percentile(samples_pred, 97.5, axis=0)

    def _log_score_fn(y_eval):
        # log_p has shape (n_post, n_eval); logaddexp.reduce along axis 0
        # averages over posterior samples in log-space.
        log_p = stats.norm.logpdf(
            y_eval[None, :], loc=means_per_post, scale=sigma_samples[:, None]
        )
        return np.logaddexp.reduce(log_p, axis=0) - np.log(n_post)

    return {
        "mean": pred_mean,
        "lower_95": lower,
        "upper_95": upper,
        "log_score_fn": _log_score_fn,
        "inference_cost": int(n_post),
    }


def fit_student_mlp(x_distill, y_distill):
    """sklearn MLPRegressor with hidden_layer_sizes=(32, 32), max_iter=2000."""
    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 32), max_iter=2000, random_state=SEED,
    )
    mlp.fit(x_distill[:, None], y_distill)
    mean = mlp.predict(X_EVAL[:, None])
    resid = y_distill - mlp.predict(x_distill[:, None])
    sigma_hat = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.1
    lower = mean - 1.96 * sigma_hat
    upper = mean + 1.96 * sigma_hat

    def _log_score_fn(y_eval, _mean=mean, _sigma=sigma_hat):
        return stats.norm.logpdf(y_eval, loc=_mean, scale=_sigma)

    return {
        "mean": mean,
        "lower_95": lower,
        "upper_95": upper,
        "log_score_fn": _log_score_fn,
        "inference_cost": 1,
    }


def build_distillation_payload(extras: dict, rng: np.random.Generator) -> dict:
    print(f"\n{'='*70}\nDistillation grid: 5 students × 4 dataset sizes\n{'='*70}")
    teacher_cache = teacher_predictive(extras, rng, n_samples_per_x=60)
    y_eval = teacher_cache["y_eval"]
    x_mean = extras["x_mean"]
    x_std = extras["x_std"]

    distill_sizes = [50, 100, 200, 400]
    student_specs = [
        ("linear", "linear", None),
        ("poly-3", "poly", 3),
        ("poly-6", "poly", 6),
        ("poly-10", "poly", 10),
        ("mlp", "mlp", None),
    ]

    students_payload = {name: {} for name, _, _ in student_specs}

    # Best single candidate (by held-out log-score, not LOO ELPD — closer to deployment metric).
    LP_eval = teacher_cache["LP_eval_per_candidate"]
    finite_eval = np.isfinite(LP_eval).all(axis=1)
    per_candidate_score = np.array([
        float(np.mean(LP_eval[finite_eval, k])) for k in range(4)
    ])
    best_idx = int(np.argmax(per_candidate_score))
    best_single = {
        "name": CANDIDATE_NAMES[best_idx],
        "log_score": float(per_candidate_score[best_idx]),
    }

    for size in distill_sizes:
        # Distillation training inputs: uniformly-spaced, plus one teacher sample per input.
        x_distill = np.linspace(0.0, 1.0, size)
        teacher_samples = sample_teacher_at_inputs(
            x_distill, n_samples_each=1, extras=extras,
            teacher_cache=teacher_cache, rng=rng,
        )
        y_distill = teacher_samples[:, 0]
        size_key = str(size)
        for name, kind, degree in student_specs:
            print(f"  fitting {name} on {size}-sample distillation set ...")
            if kind == "linear":
                fit = fit_student_linear(x_distill, y_distill)
            elif kind == "poly":
                fit = fit_student_bayesian_poly(x_distill, y_distill, degree, x_mean, x_std)
            elif kind == "mlp":
                fit = fit_student_mlp(x_distill, y_distill)
            else:  # pragma: no cover
                raise ValueError(name)

            log_p_eval = fit["log_score_fn"](y_eval)
            log_score = float(np.mean(log_p_eval))
            students_payload[name][size_key] = {
                "mean": fit["mean"],
                "lower_95": fit["lower_95"],
                "upper_95": fit["upper_95"],
                "log_score": log_score,
                "inference_cost": int(fit["inference_cost"]),
            }

    payload = {
        "x_eval": X_EVAL,
        "teacher": {
            "mean": teacher_cache["mean"],
            "lower_95": teacher_cache["lower_95"],
            "upper_95": teacher_cache["upper_95"],
            "log_score": teacher_cache["log_score"],
            "inference_cost": teacher_cache["inference_cost"],
        },
        "best_single": best_single,
        "students": students_payload,
    }
    return payload


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print(f"PyMC version: {pm.__version__}")
    print(f"pymc-bart version: {pmb.__version__}")
    print(f"arviz version: {az.__version__}")
    print(f"Output directory: {OUT_DIR}")

    extras_n100 = None
    last_w_stack_n40 = None
    pipeline_summary = {}

    for n in [40, 60, 80, 100]:
        payload, extras = fit_full_pipeline(n, rng)
        path = OUT_DIR / f"pymc_pipeline_n{n}.json"
        size_bytes = write_json(path, payload)
        pipeline_summary[n] = {
            "path": str(path),
            "size_bytes": size_bytes,
            "w_stack": payload["w_stack"],
            "elpd_loo": payload["elpd_loo"],
            "holdout_log_score": payload["holdout_log_score"],
        }
        print(f"  -> wrote {path.name}  ({size_bytes:,} bytes)")
        print(f"     w_stack: {payload['w_stack']}")
        if n == 40:
            last_w_stack_n40 = payload["w_stack"]
        if n == 100:
            extras_n100 = extras

    # Pareto-k payload from n=100.
    if extras_n100 is not None:
        pk_payload = build_pareto_k_payload(100, extras_n100)
        pk_path = OUT_DIR / "pareto_k_n100.json"
        size_bytes = write_json(pk_path, pk_payload)
        print(f"\n-> wrote {pk_path.name}  ({size_bytes:,} bytes)")

    # Distillation grid (uses extras_n100 as the teacher).
    if extras_n100 is not None:
        distill_payload = build_distillation_payload(extras_n100, rng)
        distill_path = OUT_DIR / "distillation_grid.json"
        size_bytes = write_json(distill_path, distill_payload)
        print(f"-> wrote {distill_path.name}  ({size_bytes:,} bytes)")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"Total wall-time: {elapsed:.1f}s")

    # Sanity check table for n=100.
    if 100 in pipeline_summary:
        print("\nn=100 stacking weights (sanity check vs notebook printed [0.001, 0.018, 0.031, 0.950]):")
        for name, w in pipeline_summary[100]["w_stack"].items():
            print(f"  {name:>6s}: {w:.4f}")
        print("\nn=100 BB-pseudo-BMA weights:")
        # Reload to get the persisted values.
        with open(OUT_DIR / "pymc_pipeline_n100.json") as f:
            data = json.load(f)
        for name, w in data["w_bma"].items():
            print(f"  {name:>6s}: {w:.4f}")

    # Final integration-test grep tag.
    if last_w_stack_n40 is not None:
        w_compact = [round(float(last_w_stack_n40[name]), 4) for name in CANDIDATE_NAMES]
        print(f"\nSTACKING_PRECOMPUTE_DONE n40_w_stack={w_compact}")


if __name__ == "__main__":
    main()
