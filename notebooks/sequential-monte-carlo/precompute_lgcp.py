"""precompute_lgcp.py — Log-Gaussian Cox process SMC inference at three grid
resolutions (m ∈ {8, 12, 16}), saved as a single JSON payload for the §11.3
LGCPInferenceExplorer viz.

Writes to BOTH:
  - src/data/sampleData/sequential-monte-carlo/lgcp.json
  - public/sample-data/sequential-monte-carlo/lgcp.json

The notebook's cell 23 runs only m = 16 (K = 256, ~20 s on CPU). For the
interactive viz we precompute m = 8 (K = 64), m = 12 (K = 144), and m = 16
(K = 256) so the reader can pick grid resolution without paying the SMC cost
in-browser. The K = 256 case is too heavy for browser-side SMC sampling.

Output schema:
{
  "seed": 20260518,
  "sigma_z_sq": 1.91,
  "ell": 0.10,
  "mu_0": ln(50),
  "tau_adapt": 0.9,
  "tau_resample": 0.5,
  "N": 300,
  "grids": {
    "8":  { ...payload_for_m_8  },
    "12": { ...payload_for_m_12 },
    "16": { ...payload_for_m_16 }
  }
}

Per-grid payload:
{
  "m": int,
  "K": int,
  "centers": [m] float,
  "y_obs": [K] int,
  "lambda_true": [K] float,
  "lambda_post": [K] float,
  "rmse_intensity": float,
  "beta_trace": [T+1] float,
  "ess_trace": [T+1] float,
  "log_z_trace": [T+1] float,
  "lambda_high_trace": [T+1] float,  # posterior mean intensity at the max-truth cell
  "lambda_low_trace":  [T+1] float,  # posterior mean intensity at the min-truth cell
  "high_cell_idx": int,
  "low_cell_idx":  int,
  "lambda_max": float,                # color-scale upper bound
  "T_steps": int,
  "wall_seconds": float
}

Runs in ~25-40 s end-to-end on a 2020-era laptop CPU.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

SEED = 20260518
SIGMA_Z = float(np.sqrt(1.91))
ELL = 0.10
MU_0 = float(np.log(50))
N_SMC = 300
TAU_ADAPT = 0.9
TAU_RESAMPLE = 0.5
M_VALUES = [8, 12, 16]

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "sequential-monte-carlo",
    REPO_ROOT / "public" / "sample-data" / "sequential-monte-carlo",
]


def kernel_matrix(C: np.ndarray, sigma: float, ell: float) -> np.ndarray:
    sqdist = (
        np.sum(C**2, axis=1)[:, None]
        + np.sum(C**2, axis=1)[None, :]
        - 2.0 * C @ C.T
    )
    return sigma**2 * np.exp(-0.5 * sqdist / ell**2)


def truth_field(cx: np.ndarray, cy: np.ndarray, mu0: float) -> np.ndarray:
    return (
        mu0
        + 1.5 * np.exp(-((cx - 0.3) ** 2 + (cy - 0.7) ** 2) / 0.05)
        + 1.2 * np.exp(-((cx - 0.7) ** 2 + (cy - 0.4) ** 2) / 0.08)
        - 0.5 * np.exp(-((cx - 0.5) ** 2 + (cy - 0.5) ** 2) / 0.15)
    ).ravel()


def run_one(m: int, rng_master: np.random.Generator) -> dict:
    K = m * m
    A = 1.0 / K
    edges = np.linspace(0.0, 1.0, m + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cy, cx = np.meshgrid(centers, centers, indexing="ij")
    centroids = np.column_stack([cx.ravel(), cy.ravel()])

    Sigma_K = kernel_matrix(centroids, SIGMA_Z, ELL) + 1e-6 * np.eye(K)
    L_K = np.linalg.cholesky(Sigma_K)

    # Match the notebook's seed for m=16 (cell 23 uses RNG_SEED + 11);
    # offset other grids deterministically so each m has its own stream.
    rng = np.random.default_rng(SEED + 11 + m * 53)
    z_true = truth_field(cx, cy, MU_0)
    lambda_true = np.exp(z_true)
    y_obs = rng.poisson(A * lambda_true)

    def log_prior(z_batch: np.ndarray) -> np.ndarray:
        diff = z_batch - MU_0
        u = np.linalg.solve(L_K, diff.T)
        quad = (u**2).sum(axis=0)
        log_det = 2.0 * np.log(np.diag(L_K)).sum()
        return -0.5 * quad - 0.5 * (K * np.log(2 * np.pi) + log_det)

    def log_lik(z_batch: np.ndarray) -> np.ndarray:
        return (y_obs[None, :] * z_batch).sum(axis=1) - A * np.exp(z_batch).sum(axis=1)

    def log_gamma(z_batch: np.ndarray, beta: float) -> np.ndarray:
        return log_prior(z_batch) + beta * log_lik(z_batch)

    t0 = time.perf_counter()

    zeta = rng.normal(0, 1, size=(N_SMC, K))
    theta = MU_0 + zeta @ L_K.T
    log_w = np.zeros(N_SMC)
    log_lik_curr = log_lik(theta)

    beta = 0.0
    log_z_running = 0.0
    beta_trace = [0.0]
    ess_trace = [float(N_SMC)]
    log_z_trace = [0.0]
    posterior_mean_history = [theta.mean(axis=0)]

    step = 0
    while beta < 1.0 - 1e-9 and step < 60:
        def projected_ess(b: float) -> float:
            log_alpha = (b - beta) * log_lik_curr
            log_w_new = log_w + log_alpha
            log_w_norm = log_w_new - logsumexp(log_w_new)
            return float(np.exp(-logsumexp(2.0 * log_w_norm)))

        if projected_ess(1.0) >= TAU_ADAPT * N_SMC:
            new_beta = 1.0
        else:
            lo, hi = beta, 1.0
            for _ in range(40):
                if hi - lo < 1e-6:
                    break
                mid = 0.5 * (lo + hi)
                if projected_ess(mid) >= TAU_ADAPT * N_SMC:
                    lo = mid
                else:
                    hi = mid
            new_beta = 0.5 * (lo + hi)

        delta_beta = new_beta - beta
        log_w = log_w + delta_beta * log_lik_curr
        log_z_running = float(logsumexp(log_w) - np.log(N_SMC))
        log_w_norm = log_w - logsumexp(log_w)
        ess = float(np.exp(-logsumexp(2.0 * log_w_norm)))

        if ess < TAU_RESAMPLE * N_SMC:
            cumw = np.cumsum(np.exp(log_w_norm))
            u = (rng.random() + np.arange(N_SMC)) / N_SMC
            idx = np.clip(np.searchsorted(cumw, u), 0, N_SMC - 1)
            theta = theta[idx]
            log_lik_curr = log_lik_curr[idx]
            log_w = np.full(N_SMC, log_z_running)
            log_w_norm = log_w - logsumexp(log_w)

        # IMH cloud-fit propagation
        w_norm = np.exp(log_w_norm)
        mu_fit = (w_norm[:, None] * theta).sum(axis=0)
        diff_fit = theta - mu_fit
        cov_fit = (w_norm[:, None] * diff_fit).T @ diff_fit + 1e-4 * np.eye(K)
        L_fit = np.linalg.cholesky(cov_fit)
        prop = mu_fit + rng.normal(0, 1, size=(N_SMC, K)) @ L_fit.T
        u_fit = np.linalg.solve(L_fit, (prop - mu_fit).T).T
        log_q_prop = (
            -0.5 * (u_fit**2).sum(axis=1)
            - np.log(np.diag(L_fit)).sum()
            - 0.5 * K * np.log(2 * np.pi)
        )
        u_curr = np.linalg.solve(L_fit, (theta - mu_fit).T).T
        log_q_curr = (
            -0.5 * (u_curr**2).sum(axis=1)
            - np.log(np.diag(L_fit)).sum()
            - 0.5 * K * np.log(2 * np.pi)
        )
        log_acc = (
            log_gamma(prop, new_beta)
            - log_gamma(theta, new_beta)
            + log_q_curr
            - log_q_prop
        )
        accept = np.log(rng.random(N_SMC)) < log_acc
        theta = np.where(accept[:, None], prop, theta)
        log_lik_curr = np.where(accept, log_lik(prop), log_lik_curr)

        beta = new_beta
        step += 1
        beta_trace.append(float(beta))
        ess_trace.append(ess)
        log_z_trace.append(log_z_running)
        posterior_mean_history.append(theta.mean(axis=0))

    elapsed = time.perf_counter() - t0

    log_w_norm = log_w - logsumexp(log_w)
    w_norm = np.exp(log_w_norm)
    z_post_mean = (w_norm[:, None] * theta).sum(axis=0)
    lambda_post = np.exp(z_post_mean)
    rmse_intensity = float(np.sqrt(np.mean((lambda_post - lambda_true) ** 2)))

    high_cell = int(np.argmax(lambda_true))
    low_cell = int(np.argmin(lambda_true))
    post_history_arr = np.array(posterior_mean_history)
    lambda_high_trace = np.exp(post_history_arr[:, high_cell]).astype(float).tolist()
    lambda_low_trace = np.exp(post_history_arr[:, low_cell]).astype(float).tolist()

    return {
        "m": m,
        "K": K,
        "centers": centers.astype(float).tolist(),
        "y_obs": y_obs.astype(int).tolist(),
        "lambda_true": lambda_true.astype(float).tolist(),
        "lambda_post": lambda_post.astype(float).tolist(),
        "rmse_intensity": rmse_intensity,
        "beta_trace": beta_trace,
        "ess_trace": ess_trace,
        "log_z_trace": log_z_trace,
        "lambda_high_trace": lambda_high_trace,
        "lambda_low_trace": lambda_low_trace,
        "high_cell_idx": high_cell,
        "low_cell_idx": low_cell,
        "lambda_high_truth": float(lambda_true[high_cell]),
        "lambda_low_truth": float(lambda_true[low_cell]),
        "lambda_max": float(lambda_true.max()),
        "T_steps": step,
        "wall_seconds": float(elapsed),
        "log_z_T": float(log_z_running),
    }


def main() -> None:
    print(f"precompute_lgcp.py: running LGCP SMC at m ∈ {M_VALUES}, N = {N_SMC}, "
          f"adaptive schedule (τ_adapt = {TAU_ADAPT})")
    rng_master = np.random.default_rng(SEED)
    payload = {
        "seed": SEED,
        "sigma_z_sq": SIGMA_Z**2,
        "ell": ELL,
        "mu_0": MU_0,
        "tau_adapt": TAU_ADAPT,
        "tau_resample": TAU_RESAMPLE,
        "N": N_SMC,
        "grids": {},
    }
    for m in M_VALUES:
        print(f"  running m = {m} (K = {m*m})...", flush=True)
        grid_payload = run_one(m, rng_master)
        payload["grids"][str(m)] = grid_payload
        print(
            f"    T = {grid_payload['T_steps']}, RMSE = {grid_payload['rmse_intensity']:.3f}, "
            f"log Ẑ = {grid_payload['log_z_T']:.3f}, wall = {grid_payload['wall_seconds']:.1f}s"
        )

    serialized = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        out_path = d / "lgcp.json"
        out_path.write_text(serialized)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
