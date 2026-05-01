"""Precompute NUTS + ADVI + MAP fits for B5 InferenceDispatchExplorer.

Companion to ``01_probabilistic_programming.ipynb`` §4.5. Fits the Bayesian
logistic regression model from §4.1 — versicolor-vs-virginica Iris on
standardized petal length and petal width, with a Gaussian prior — three
times in PyMC: NUTS, ADVI mean-field Gaussian, and MAP via L-BFGS. Saves
the full set of intermediate quantities needed to render the §4.5 dispatch
comparison interactively:

  - NUTS draws of (alpha, beta_1, beta_2) — 2000 samples post-warmup.
  - ADVI variational mean and standard deviation per parameter
    (computed from posterior samples via approx.sample()).
  - ADVI ELBO loss trajectory across the optimization.
  - MAP point estimate from L-BFGS on the joint log-density.
  - Standardized Iris features (so the JS-side viz can plot the data too,
    if the component wants it).

Usage::

    cd notebooks/probabilistic-programming
    .venv/bin/python precompute_inference_dispatch.py

Runtime: ~30-60 seconds.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths and reproducibility
# --------------------------------------------------------------------------- #

SEED = 20260430
NOTEBOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NOTEBOOK_DIR.parents[1]

OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "probabilistic-programming",
    REPO_ROOT / "public" / "sample-data" / "probabilistic-programming",
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

OUT_FILENAME = "inference_dispatch.json"

# --------------------------------------------------------------------------- #
# JSON helpers (shared with the other precompute scripts)
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
# Iris versicolor-vs-virginica (binary classification on petal features)
# --------------------------------------------------------------------------- #


def load_iris_binary() -> tuple[np.ndarray, np.ndarray]:
    """Versicolor (class 1) vs virginica (class 2), petal length and width.

    Returns (X, y) with X standardized to zero mean and unit variance per
    column, and y in {0, 1}.
    """
    iris = load_iris()
    mask = iris.target != 0
    X_raw = iris.data[mask][:, [2, 3]]
    y = (iris.target[mask] - 1).astype(int)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    return X, y


# --------------------------------------------------------------------------- #
# Three fits
# --------------------------------------------------------------------------- #


def build_model(X: np.ndarray, y: np.ndarray) -> pm.Model:
    """§4.1's Bayesian logistic regression: alpha + N(0, 5²) priors on each
    coefficient, Bernoulli likelihood with logit link."""
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
        beta = pm.Normal("beta", mu=0.0, sigma=5.0, shape=X.shape[1])
        logits = alpha + pm.math.dot(X, beta)
        pm.Bernoulli("y_obs", logit_p=logits, observed=y)
    return model


def fit_nuts(model: pm.Model) -> az.InferenceData:
    rng = np.random.default_rng(SEED)
    with model:
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            cores=1,
            target_accept=0.9,
            random_seed=int(rng.integers(2**31)),
            progressbar=False,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": False},
        )
    return idata


def fit_advi(model: pm.Model) -> tuple[dict, list[float]]:
    """Mean-field ADVI via PyMC's pm.fit. Returns variational mean, sigma per
    variable (estimated from N=10000 posterior samples), and the ELBO loss
    trajectory (negated so it is non-decreasing when the optimizer is
    improving)."""
    rng = np.random.default_rng(SEED + 1)
    with model:
        approx = pm.fit(
            n=20000,
            method="advi",
            random_seed=int(rng.integers(2**31)),
            progressbar=False,
        )
        loss_full = np.array(approx.hist)
        thin = max(1, len(loss_full) // 400)
        loss_thinned = loss_full[::thin].tolist()
        # Sample-based estimate of variational mean and std-dev per parameter.
        # This avoids reaching into PyTensor primitives for direct parameter
        # extraction.
        idata_vi = approx.sample(10000, random_seed=int(rng.integers(2**31)))
        post_vi = idata_vi.posterior
        alpha_samples = post_vi["alpha"].values.reshape(-1)
        beta_samples = post_vi["beta"].values.reshape(-1, 2)
        alpha_mean = float(alpha_samples.mean())
        alpha_std = float(alpha_samples.std(ddof=1))
        beta_mean = beta_samples.mean(axis=0).tolist()
        beta_std = beta_samples.std(axis=0, ddof=1).tolist()
    return {
        "alpha_mean": alpha_mean,
        "beta_mean": beta_mean,
        "alpha_std": alpha_std,
        "beta_std": beta_std,
    }, loss_thinned


def fit_map(model: pm.Model) -> dict:
    rng = np.random.default_rng(SEED + 2)
    with model:
        out = pm.find_MAP(progressbar=False, seed=int(rng.integers(2**31)))
    return {
        "alpha": float(out["alpha"]),
        "beta": [float(out["beta"][0]), float(out["beta"][1])],
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    t0 = time.time()
    X, y = load_iris_binary()
    print(f"Loaded Iris versicolor-vs-virginica: N={X.shape[0]}, D={X.shape[1]}")

    model = build_model(X, y)

    print("Fitting NUTS …")
    idata = fit_nuts(model)
    posterior = idata.posterior
    n_chains = posterior.sizes["chain"]
    n_draws = posterior.sizes["draw"]
    n_total = n_chains * n_draws
    alpha_flat = posterior["alpha"].values.reshape(n_total)
    beta_flat = posterior["beta"].values.reshape(n_total, 2)
    summary = az.summary(idata, var_names=["alpha", "beta"], round_to=4)
    print(f"  Drew {n_total} NUTS samples")
    print(f"  R-hat: alpha={summary.loc['alpha', 'r_hat']:.4f}, "
          f"beta[0]={summary.loc['beta[0]', 'r_hat']:.4f}, "
          f"beta[1]={summary.loc['beta[1]', 'r_hat']:.4f}")

    print("Fitting ADVI …")
    advi_params, advi_loss = fit_advi(model)
    print(f"  ADVI mean: alpha={advi_params['alpha_mean']:.3f}, "
          f"beta=({advi_params['beta_mean'][0]:.3f}, {advi_params['beta_mean'][1]:.3f})")
    print(f"  ADVI std:  alpha={advi_params['alpha_std']:.3f}, "
          f"beta=({advi_params['beta_std'][0]:.3f}, {advi_params['beta_std'][1]:.3f})")
    print(f"  Final ELBO loss: {advi_loss[-1]:.4f}")

    print("Fitting MAP …")
    map_pt = fit_map(model)
    print(f"  MAP: alpha={map_pt['alpha']:.3f}, "
          f"beta=({map_pt['beta'][0]:.3f}, {map_pt['beta'][1]:.3f})")

    payload = {
        "metadata": {
            "model": "iris_logistic_regression",
            "engine": "PyMC",
            "pymc_version": pm.__version__,
            "n_obs": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "feature_names": ["petal_length_std", "petal_width_std"],
            "n_chains": n_chains,
            "n_draws_per_chain": n_draws,
            "n_total": n_total,
            "rhat": {
                "alpha": float(summary.loc["alpha", "r_hat"]),
                "beta_0": float(summary.loc["beta[0]", "r_hat"]),
                "beta_1": float(summary.loc["beta[1]", "r_hat"]),
            },
            "ess_bulk": {
                "alpha": int(summary.loc["alpha", "ess_bulk"]),
                "beta_0": int(summary.loc["beta[0]", "ess_bulk"]),
                "beta_1": int(summary.loc["beta[1]", "ess_bulk"]),
            },
            "advi_iterations": 20000,
            "seed": SEED,
        },
        "data": {
            "X_std": _round_floats(_to_jsonable(X)),
            "y": _to_jsonable(y),
        },
        "nuts": {
            "alpha": _round_floats(_to_jsonable(alpha_flat)),
            "beta": _round_floats(_to_jsonable(beta_flat)),
        },
        "advi": {
            "alpha_mean": round(advi_params["alpha_mean"], 4),
            "beta_mean": _round_floats(_to_jsonable(advi_params["beta_mean"])),
            "alpha_std": round(advi_params["alpha_std"], 4),
            "beta_std": _round_floats(_to_jsonable(advi_params["beta_std"])),
            "loss": _round_floats(_to_jsonable(advi_loss), 3),
        },
        "map": {
            "alpha": round(map_pt["alpha"], 4),
            "beta": _round_floats(_to_jsonable(map_pt["beta"])),
        },
    }

    body = json.dumps(payload, separators=(",", ":"))
    for d in OUT_DIRS:
        out_path = d / OUT_FILENAME
        out_path.write_text(body)
        print(f"  Wrote {out_path.relative_to(REPO_ROOT)} ({len(body) / 1024:.1f} KB)")

    print(f"Done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
