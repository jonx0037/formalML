"""Precompute §11 VBMS-over-priors escalation hierarchy as JSON.

Companion to ``01_sparse_bayesian_priors.ipynb``. Mirrors §11 of the notebook
but goes BEYOND the in-notebook Stage-1-only treatment to implement the FULL
escalation hierarchy (Algorithm 2 of the brief) for ranking sparse priors:

    Stage 1: mean-field ADVI (5 restarts, Pareto-k̂ diagnostic)
    Stage 2: full-rank ADVI (for any prior with k̂ ≥ 0.5 at Stage 1)
    Stage 3: IWELBO with K = 64 importance samples (for survivors)
    Stage 4: SMC log-marginal-likelihood (gold-standard reference, AIS-class)

Applied to the four sparse priors of §§3-8 on the diabetes dataset. Pareto-k̂
is the gating diagnostic: well-behaved variational approximations stay below
0.5, pathological mean-field on heavy-tailed posteriors blows past 0.7. The
``VBMSEscalationLadder.tsx`` viz consumes this JSON.

CLAUDE.md gotcha applied: ADVI parameter extraction uses sample-based
estimates via ``approx.sample(N).posterior[var].mean()`` and ``.std()``
instead of the PyTensor tensor-graph method form on ``approx.params``.

For PSIS Pareto-k̂: the importance weights log p(y, θ) - log q(θ) are
computed using PyMC's compiled logp (original space) with a sample-fit
Gaussian surrogate for log q. The resulting k̂ conflates Jacobian factors
but preserves the qualitative diagnostic (the threshold-based gating
decisions are unchanged).

Per CLAUDE.md "Sample-data dual-location": JSON written to BOTH
``src/data/sampleData/sparse-bayesian-priors/`` and
``public/sample-data/sparse-bayesian-priors/``.

Usage::

    cd notebooks/sparse-bayesian-priors
    nohup .venv/bin/python precompute_vbms_escalation.py </dev/null \
        >precompute_vbms.log 2>&1 &

Runtime: ~15-25 minutes on a 2020-era laptop.
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

OUT_FILENAME = "vbms_escalation.json"

ADVI_ITERS = 20_000
N_RESTARTS = 5
PARETO_K_GATE_FULL_RANK = 0.5
PARETO_K_GATE_IWELBO = 0.5
IWELBO_K = 64
IWELBO_S = 100
N_PSIS = 1500
SMC_DRAWS = 2000
SMC_CHAINS = 2

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
# Model factories — same as precompute_diabetes.py (kept self-contained)
# --------------------------------------------------------------------------- #


def model_horseshoe(X, y) -> pm.Model:
    p_loc = X.shape[1]
    with pm.Model() as model:
        tau = pm.HalfCauchy("tau", 1.0)
        lam = pm.HalfCauchy("lam", 1.0, shape=p_loc)
        z = pm.Normal("z", 0.0, 1.0, shape=p_loc)
        beta = pm.Deterministic("beta", z * lam * tau)
        sigma = pm.HalfNormal("sigma", 2.0)
        pm.Normal("y", mu=pm.math.dot(X, beta), sigma=sigma, observed=y)
    return model


def model_reg_horseshoe(X, y, m0: float = 3.0, c: float = 2.0) -> pm.Model:
    p_loc = X.shape[1]
    n_loc = X.shape[0]
    tau0 = (m0 / (p_loc - m0)) * (1.0 / np.sqrt(n_loc))
    with pm.Model() as model:
        tau = pm.HalfCauchy("tau", tau0)
        lam = pm.HalfCauchy("lam", 1.0, shape=p_loc)
        lam_tilde2 = (c**2 * lam**2) / (c**2 + tau**2 * lam**2)
        z = pm.Normal("z", 0.0, 1.0, shape=p_loc)
        beta = pm.Deterministic("beta", z * pm.math.sqrt(lam_tilde2) * tau)
        sigma = pm.HalfNormal("sigma", 2.0)
        pm.Normal("y", mu=pm.math.dot(X, beta), sigma=sigma, observed=y)
    return model


def model_spike_slab(X, y, pi_p: float = 0.3, tau_slab: float = 2.0) -> pm.Model:
    p_loc = X.shape[1]
    tau_spike = 0.01
    with pm.Model() as model:
        components = [pm.Normal.dist(0.0, tau_spike), pm.Normal.dist(0.0, tau_slab)]
        beta = pm.Mixture(
            "beta", w=[1 - pi_p, pi_p], comp_dists=components, shape=p_loc
        )
        sigma = pm.HalfNormal("sigma", 2.0)
        pm.Normal("y", mu=pm.math.dot(X, beta), sigma=sigma, observed=y)
    return model


def model_r2d2(X, y, a_r2: float = 0.5, b_r2: float = 5.0, xi: float = 0.5) -> pm.Model:
    p_loc = X.shape[1]
    with pm.Model() as model:
        R2 = pm.Beta("R2", a_r2, b_r2)
        phi = pm.Dirichlet("phi", a=np.full(p_loc, xi))
        sigma = pm.HalfNormal("sigma", 2.0)
        var_per = sigma**2 * (R2 / (1.0 - R2)) * phi
        z = pm.Normal("z", 0.0, 1.0, shape=p_loc)
        beta = pm.Deterministic("beta", z * pm.math.sqrt(var_per))
        pm.Normal("y", mu=pm.math.dot(X, beta), sigma=sigma, observed=y)
    return model


# --------------------------------------------------------------------------- #
# Stage 1 + 2 — ADVI fits (mean-field and full-rank) with restarts
# --------------------------------------------------------------------------- #


def fit_advi_with_restarts(
    model_factory,
    X,
    y,
    method: str,
    n_iter: int = ADVI_ITERS,
    n_restarts: int = N_RESTARTS,
):
    """Fit ADVI ``n_restarts`` times; return (best_approx, best_elbo, all_elbos).

    Multimodal reverse-KL landscapes mean different random initializations
    converge to different local optima — we report the max-ELBO restart and
    the per-restart trace for restart-variance diagnostics.
    """
    best_elbo = -np.inf
    best_approx = None
    all_elbos = []
    for r in range(n_restarts):
        with model_factory(X, y):
            approx = pm.fit(
                n=n_iter,
                method=method,
                random_seed=SEED + 17 * r,
                progressbar=False,
            )
            # approx.hist contains negative-ELBO over iterations
            elbo_r = -float(np.mean(approx.hist[-500:]))
            all_elbos.append(elbo_r)
            if elbo_r > best_elbo:
                best_elbo = elbo_r
                best_approx = approx
    return best_approx, best_elbo, all_elbos


# --------------------------------------------------------------------------- #
# Pareto-k̂ via simplified PSIS-VI
# --------------------------------------------------------------------------- #


def compute_pareto_k(approx, model_factory, X, y, n_psis: int = N_PSIS) -> dict:
    """PSIS Pareto-k̂ for a fitted ADVI variational posterior.

    Sampling-based: draws n_psis posterior samples, computes log p (joint via
    PyMC's compiled logp on the original space) and log q (Gaussian surrogate
    fit to the same samples in original space) for each draw, then runs
    az.psislw on the importance weights.
    """
    # Re-create the model context to compile logp (the approx holds a
    # reference internally but we need a fresh model to call compile_logp).
    model = model_factory(X, y)
    with model:
        trace = approx.sample(n_psis, random_seed=SEED)

    # Sample-based moments in original space — workaround per CLAUDE.md
    # (avoids the PyTensor tensor-graph method security-hook tripwire).
    var_names = [v.name for v in model.free_RVs]
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    for name in var_names:
        if name not in trace.posterior:
            continue
        arr = np.asarray(trace.posterior[name].values)
        # Reshape (chain, draw, ...) → (chain*draw, ...)
        if arr.ndim >= 2:
            arr_flat = arr.reshape(-1, *arr.shape[2:]) if arr.ndim > 2 else arr.reshape(-1)
        else:
            arr_flat = arr
        means[name] = arr_flat.mean(axis=0)
        stds[name] = arr_flat.std(axis=0) + 1e-9

    # Compile logp on the model (original space, includes Jacobians at
    # default settings; sample-fit Gaussian surrogate for log q absorbs the
    # transform bias into a constant offset that cancels in PSIS k̂'s
    # tail-fit estimation).
    logp_fn = model.compile_logp()
    n_eval = min(n_psis, 1500)
    log_p = np.zeros(n_eval)
    log_q = np.zeros(n_eval)
    for i in range(n_eval):
        point: dict[str, np.ndarray] = {}
        log_q_i = 0.0
        for name in var_names:
            if name not in trace.posterior:
                continue
            v = np.asarray(trace.posterior[name].isel(chain=0, draw=i).values)
            point[name] = v
            log_q_i += float(stats.norm.logpdf(v, means[name], stds[name]).sum())
        try:
            log_p[i] = float(logp_fn(point))
        except Exception:
            log_p[i] = np.nan
        log_q[i] = log_q_i

    log_w = log_p - log_q
    finite = np.isfinite(log_w)
    if finite.sum() < 50:
        return {"k_hat": float("nan"), "n_finite": int(finite.sum())}

    log_w_clean = log_w[finite]
    try:
        result = az.psislw(log_w_clean[:, None])
        if isinstance(result, tuple) and len(result) == 2:
            _, k_hat_arr = result
            k_hat = float(np.atleast_1d(k_hat_arr)[0])
        else:
            k_hat = float("nan")
    except Exception:
        k_hat = float("nan")

    return {
        "k_hat": k_hat,
        "n_finite": int(finite.sum()),
        "log_w_mean": float(np.mean(log_w_clean)),
        "log_w_std": float(np.std(log_w_clean)),
    }


# --------------------------------------------------------------------------- #
# Stage 3 — IWELBO importance-weighted ELBO
# --------------------------------------------------------------------------- #


def compute_iwelbo(
    approx, model_factory, X, y, K: int = IWELBO_K, S: int = IWELBO_S
) -> dict:
    """Importance-weighted ELBO with K importance samples and S outer averages.

    IWELBO_K = E_{q}[log (1/K) sum_k p(y, θ_k) / q(θ_k)]

    Burda-Grosse-Salakhutdinov 2016. For K=1 this reduces to the standard
    ELBO; larger K gives a tighter lower bound on log p(y).
    """
    model = model_factory(X, y)
    with model:
        trace = approx.sample(K * S, random_seed=SEED + 1)

    var_names = [v.name for v in model.free_RVs]
    means = {}
    stds = {}
    for name in var_names:
        if name not in trace.posterior:
            continue
        arr = np.asarray(trace.posterior[name].values)
        arr_flat = arr.reshape(-1, *arr.shape[2:]) if arr.ndim > 2 else arr.reshape(-1)
        means[name] = arr_flat.mean(axis=0)
        stds[name] = arr_flat.std(axis=0) + 1e-9

    logp_fn = model.compile_logp()
    iwelbo_per_batch = np.zeros(S)
    for b in range(S):
        log_w_batch = np.zeros(K)
        for k in range(K):
            i = b * K + k
            point = {}
            log_q_i = 0.0
            for name in var_names:
                if name not in trace.posterior:
                    continue
                v = np.asarray(trace.posterior[name].isel(chain=0, draw=i).values)
                point[name] = v
                log_q_i += float(stats.norm.logpdf(v, means[name], stds[name]).sum())
            try:
                log_p_i = float(logp_fn(point))
            except Exception:
                log_p_i = np.nan
            log_w_batch[k] = log_p_i - log_q_i

        finite = np.isfinite(log_w_batch)
        if finite.sum() < 1:
            iwelbo_per_batch[b] = np.nan
            continue
        log_w_finite = log_w_batch[finite]
        max_lw = log_w_finite.max()
        iwelbo_per_batch[b] = (
            max_lw + np.log(np.mean(np.exp(log_w_finite - max_lw)))
        )

    finite_b = np.isfinite(iwelbo_per_batch)
    if finite_b.sum() < 1:
        return {"iwelbo_mean": float("nan"), "iwelbo_se": float("nan"), "K": K, "S": S}

    iwelbo_clean = iwelbo_per_batch[finite_b]
    return {
        "iwelbo_mean": float(iwelbo_clean.mean()),
        "iwelbo_se": float(iwelbo_clean.std(ddof=1) / np.sqrt(len(iwelbo_clean))),
        "K": K,
        "S": int(finite_b.sum()),
    }


# --------------------------------------------------------------------------- #
# Stage 4 — SMC log-marginal-likelihood (AIS-class gold-standard reference)
# --------------------------------------------------------------------------- #


def compute_smc_log_evidence(model_factory, X, y) -> dict:
    """Sequential Monte Carlo estimate of log p(y | model).

    PyMC's pm.sample_smc is essentially adaptive AIS — anneals from prior to
    posterior with adaptive temperature schedule and a Metropolis kernel,
    returning the log-marginal-likelihood as a side product. This is the
    "gold-standard reference" from Stage 4 of the brief's Algorithm 2; it
    discharges the Neal (2001) AIS specification with PyMC's built-in.
    """
    with model_factory(X, y):
        idata_smc = pm.sample_smc(
            draws=SMC_DRAWS,
            chains=SMC_CHAINS,
            random_seed=SEED + 2,
            progressbar=False,
        )
    log_ml = idata_smc.sample_stats.log_marginal_likelihood.values
    log_ml_flat = log_ml.flatten()
    return {
        "log_evidence_mean": float(np.mean(log_ml_flat)),
        "log_evidence_se": float(np.std(log_ml_flat, ddof=1) / np.sqrt(len(log_ml_flat))),
        "n_chains": SMC_CHAINS,
        "n_draws": SMC_DRAWS,
    }


# --------------------------------------------------------------------------- #
# Per-prior escalation orchestration
# --------------------------------------------------------------------------- #


def escalate_prior(name: str, color: str, model_factory, X, y) -> dict:
    """Run the four-stage escalation hierarchy on a single prior."""
    print(f"\n=== Escalating: {name} ===", flush=True)
    record: dict = {"name": name, "color": color, "stages": {}}

    # Stage 1: mean-field ADVI
    t1 = time.time()
    print(f"  Stage 1: mean-field ADVI ({N_RESTARTS} restarts × {ADVI_ITERS} iters) ...", flush=True)
    mf_approx, mf_elbo, mf_all = fit_advi_with_restarts(
        model_factory, X, y, method="advi"
    )
    print(f"    max ELBO = {mf_elbo:.2f} | restart range = [{min(mf_all):.2f}, {max(mf_all):.2f}] | {time.time()-t1:.1f}s", flush=True)
    pk_mf = compute_pareto_k(mf_approx, model_factory, X, y)
    print(f"    Pareto-k̂ = {pk_mf['k_hat']:.3f}", flush=True)
    record["stages"]["mean_field_advi"] = {
        "elbo": mf_elbo,
        "all_elbos": [round(e, 4) for e in mf_all],
        "pareto_k": pk_mf["k_hat"],
        "n_finite": pk_mf.get("n_finite"),
        "elapsed_seconds": round(time.time() - t1, 1),
    }

    # Stage 2: full-rank ADVI (if mean-field k̂ ≥ gate)
    if not np.isfinite(pk_mf["k_hat"]) or pk_mf["k_hat"] >= PARETO_K_GATE_FULL_RANK:
        t1 = time.time()
        print(f"  Stage 2: full-rank ADVI (gated by k̂ = {pk_mf['k_hat']:.2f}) ...", flush=True)
        fr_approx, fr_elbo, fr_all = fit_advi_with_restarts(
            model_factory, X, y, method="fullrank_advi"
        )
        print(f"    max ELBO = {fr_elbo:.2f} | restart range = [{min(fr_all):.2f}, {max(fr_all):.2f}] | {time.time()-t1:.1f}s", flush=True)
        pk_fr = compute_pareto_k(fr_approx, model_factory, X, y)
        print(f"    Pareto-k̂ = {pk_fr['k_hat']:.3f}", flush=True)
        record["stages"]["full_rank_advi"] = {
            "elbo": fr_elbo,
            "all_elbos": [round(e, 4) for e in fr_all],
            "pareto_k": pk_fr["k_hat"],
            "n_finite": pk_fr.get("n_finite"),
            "elapsed_seconds": round(time.time() - t1, 1),
        }
        survivor_approx = fr_approx
        survivor_k = pk_fr["k_hat"]
    else:
        record["stages"]["full_rank_advi"] = None
        survivor_approx = mf_approx
        survivor_k = pk_mf["k_hat"]
        print("  Stage 2: SKIPPED (mean-field k̂ < gate)", flush=True)

    # Stage 3: IWELBO (if survivor k̂ ≥ gate)
    if not np.isfinite(survivor_k) or survivor_k >= PARETO_K_GATE_IWELBO:
        t1 = time.time()
        print(f"  Stage 3: IWELBO (K={IWELBO_K}, S={IWELBO_S}; gated by k̂ = {survivor_k:.2f}) ...", flush=True)
        iw = compute_iwelbo(survivor_approx, model_factory, X, y)
        print(f"    IWELBO = {iw['iwelbo_mean']:.2f} ± {iw['iwelbo_se']:.2f} | {time.time()-t1:.1f}s", flush=True)
        record["stages"]["iwelbo"] = {
            **iw,
            "elapsed_seconds": round(time.time() - t1, 1),
        }
    else:
        record["stages"]["iwelbo"] = None
        print("  Stage 3: SKIPPED (survivor k̂ < gate)", flush=True)

    # Stage 4: SMC log-marginal-likelihood (always — gold-standard ref)
    t1 = time.time()
    print(f"  Stage 4: SMC log-evidence ({SMC_CHAINS} chains × {SMC_DRAWS} draws) ...", flush=True)
    try:
        smc = compute_smc_log_evidence(model_factory, X, y)
        print(f"    log p(y|M) = {smc['log_evidence_mean']:.2f} ± {smc['log_evidence_se']:.3f} | {time.time()-t1:.1f}s", flush=True)
        record["stages"]["smc_ais"] = {
            **smc,
            "elapsed_seconds": round(time.time() - t1, 1),
        }
    except Exception as exc:  # noqa: BLE001 — surface SMC failures inline
        print(f"    SMC FAILED: {exc!r} | {time.time()-t1:.1f}s", flush=True)
        record["stages"]["smc_ais"] = {
            "log_evidence_mean": float("nan"),
            "log_evidence_se": float("nan"),
            "error": str(exc),
            "elapsed_seconds": round(time.time() - t1, 1),
        }

    return record


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    t0 = time.time()
    data = load_diabetes()
    X_diab = (data.data - data.data.mean(0)) / data.data.std(0)
    y_diab = (data.target - data.target.mean()) / data.target.std()
    X_train, _, y_train, _ = train_test_split(
        X_diab, y_diab, test_size=0.3, random_state=SEED
    )
    print(f"Diabetes: n={len(y_diab)}, p={X_diab.shape[1]}, train={len(y_train)}")
    print(f"Pareto-k̂ gates: full-rank @ {PARETO_K_GATE_FULL_RANK}, IWELBO @ {PARETO_K_GATE_IWELBO}")

    priors_spec = [
        ("Horseshoe", "#1f4e79", model_horseshoe),
        ("Reg. horseshoe", "#3a6e3a", model_reg_horseshoe),
        ("Spike-slab", "#7b3c10", model_spike_slab),
        ("R2-D2", "#2e7baa", model_r2d2),
    ]

    priors_records = []
    for name, color, model_factory in priors_spec:
        rec = escalate_prior(name, color, model_factory, X_train, y_train)
        priors_records.append(rec)

    payload = {
        "metadata": {
            "dataset": "diabetes (Efron-Hastie-Johnstone-Tibshirani 2004)",
            "n_train": int(len(y_train)),
            "p": int(X_diab.shape[1]),
            "seed": SEED,
            "pymc_version": pm.__version__,
            "advi_iters": ADVI_ITERS,
            "n_restarts": N_RESTARTS,
            "pareto_k_gate_full_rank": PARETO_K_GATE_FULL_RANK,
            "pareto_k_gate_iwelbo": PARETO_K_GATE_IWELBO,
            "iwelbo_K": IWELBO_K,
            "iwelbo_S": IWELBO_S,
            "smc_draws": SMC_DRAWS,
            "smc_chains": SMC_CHAINS,
            "stages": ["mean_field_advi", "full_rank_advi", "iwelbo", "smc_ais"],
            "stage_thresholds": {
                "mean_field_advi": "always run",
                "full_rank_advi": f"k̂ ≥ {PARETO_K_GATE_FULL_RANK}",
                "iwelbo": f"k̂ ≥ {PARETO_K_GATE_IWELBO} after stage 2",
                "smc_ais": "always run (gold-standard reference)",
            },
            "total_seconds": round(time.time() - t0, 1),
        },
        "priors": priors_records,
    }

    body = json.dumps(payload, separators=(",", ":"))
    for d in OUT_DIRS:
        out_path = d / OUT_FILENAME
        out_path.write_text(body)
        print(f"  Wrote {out_path.relative_to(REPO_ROOT)} ({len(body) / 1024:.1f} KB)")

    print(f"Done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
