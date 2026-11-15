"""precompute_lgcp.py — Log-Gaussian Cox process SMC inference at three grid
resolutions (m ∈ {8, 12, 16}), saved as a single JSON payload for the §11
LGCPInferenceExplorer viz.

Writes to BOTH:
  - src/data/sampleData/sequential-monte-carlo/lgcp.json
  - public/sample-data/sequential-monte-carlo/lgcp.json

The sampler is the notebook's cell 23 (Figure 11): conditional-ESS adaptive
temperatures (Zhou, Johansen & Aston 2016), systematic resampling at
tau_resample * N, and a preconditioned Crank–Nicolson (pCN) mutation kernel
(Cotter, Roberts, Stuart & White 2013) applied N_SWEEPS times per SMC step.
pCN proposes z' = mu_0 + sqrt(1 - rho^2) (z - mu_0) + rho L_K xi, which leaves
the GP prior invariant, so the acceptance ratio involves only the tempered
likelihood; rho is retuned between steps toward TARGET_ACC acceptance.

Each grid also gets a reference posterior from elliptical slice sampling
(Murray, Adams & MacKay 2010), N_CHAINS independent chains, so the viz can
show how close the SMC posterior mean is to the exact posterior — and how far
both are from the true field when only ~60–80 points are observed.

The m = 16 grid uses the notebook's seeds (RNG_SEED + 11 for the data and the
SMC run, RNG_SEED + 111 for the reference) and the same sequence of random
draws, so its numbers match Figure 11. The m = 8 and m = 12 grids use
deterministic offsets of those seeds.

Output schema:
{
  "seed": 20260518,
  "sigma_z_sq": 1.91,
  "ell": 0.10,
  "mu_0": ln(50),
  "tau_adapt": 0.9,
  "tau_resample": 0.5,
  "N": 300,
  "kernel": "pCN",
  "n_sweeps": 20,
  "target_accept": 0.3,
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
  "lambda_post": [K] float,          # SMC posterior mean intensity E[lambda_k | y]
  "lambda_ref": [K] float,           # reference (elliptical slice) E[lambda_k | y]
  "rmse_intensity": float,           # SMC posterior mean vs truth
  "rmse_ref": float,                 # reference posterior mean vs truth
  "dist_z_ref": float,               # RMS over cells, SMC E[z | y] vs reference E[z | y]
  "dist_z_prior": float,             # RMS over cells, prior mean mu_0 vs reference E[z | y]
  "dist_lambda_ref": float,          # RMS over cells, SMC E[lambda | y] vs reference
  "r_hat_max": float,                # max over cells of the reference chains' R-hat
  "beta_trace": [T+1] float,
  "ess_trace": [T+1] float,
  "log_z_trace": [T+1] float,
  "accept_trace": [T] float,         # mean pCN acceptance over the step's sweeps
  "rho_trace": [T] float,            # pCN step used at each SMC step
  "distinct_final": int,             # distinct particles at beta = 1
  "lambda_high_trace": [T+1] float,  # posterior mean intensity at the max-truth cell
  "lambda_low_trace":  [T+1] float,  # posterior mean intensity at the min-truth cell
  "high_cell_idx": int,
  "low_cell_idx":  int,
  "lambda_high_truth": float,
  "lambda_low_truth": float,
  "lambda_high_ref": float,
  "lambda_low_ref": float,
  "lambda_max": float,               # color-scale upper bound (truth)
  "T_steps": int,
  "wall_seconds": float,             # SMC only
  "log_z_T": float
}

Runs in about 20 s end-to-end on an Apple-silicon laptop CPU (SMC under 1 s
per grid; the reference chains take the rest).
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
MAX_STEPS = 200  # safety cap; the CESS schedule reaches beta = 1 well before it
N_SWEEPS = 20  # pCN sweeps per SMC step
TARGET_ACC = 0.30  # pCN step rho is retuned between steps toward this acceptance rate
RHO_GAIN = 2.0  # log rho moves by RHO_GAIN * (observed - target) acceptance
RHO_INIT = 1.0  # rho = 1 is an independent draw from the GP prior
N_CHAINS, N_ITER, N_BURN = 16, 20000, 4000  # elliptical-slice reference
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


def pcn_sweeps(theta, log_lik_curr, beta, rho, n_sweeps, rng, log_lik, L_K):
    """n_sweeps of pCN, invariant for gamma_beta(z) ∝ N(z; mu_0 1, Sigma_K) exp(beta * loglik(z)).

    The proposal is reversible for the GP prior, so the prior cancels and the
    acceptance ratio is exp(beta * (loglik(z') - loglik(z))).
    """
    accepts = []
    for _ in range(n_sweeps):
        prop = (
            MU_0
            + np.sqrt(1.0 - rho**2) * (theta - MU_0)
            + rho * (rng.normal(0, 1, size=theta.shape) @ L_K.T)
        )
        log_lik_prop = log_lik(prop)
        accept = np.log(rng.random(theta.shape[0])) < beta * (log_lik_prop - log_lik_curr)
        theta = np.where(accept[:, None], prop, theta)
        log_lik_curr = np.where(accept, log_lik_prop, log_lik_curr)
        accepts.append(accept.mean())
    return theta, log_lik_curr, float(np.mean(accepts))


def elliptical_slice_reference(log_lik, L_K, K, rng):
    """Elliptical slice sampling (Murray, Adams & MacKay 2010), N_CHAINS chains
    vectorized across chains. Returns per-cell E[z|y], E[lambda|y], max R-hat."""
    z_ch = MU_0 + rng.normal(0, 1, size=(N_CHAINS, K)) @ L_K.T
    ll_ch = log_lik(z_ch)
    sum_z = np.zeros((N_CHAINS, K))
    sum_z2 = np.zeros((N_CHAINS, K))
    sum_lam = np.zeros((N_CHAINS, K))
    for it in range(N_ITER):
        nu = rng.normal(0, 1, size=(N_CHAINS, K)) @ L_K.T
        log_thresh = ll_ch + np.log(rng.random(N_CHAINS))
        ang = rng.uniform(0, 2 * np.pi, N_CHAINS)
        ang_lo, ang_hi = ang - 2 * np.pi, ang.copy()
        pending = np.ones(N_CHAINS, dtype=bool)
        z_next, ll_next = z_ch.copy(), ll_ch.copy()
        while pending.any():
            prop = MU_0 + (z_ch - MU_0) * np.cos(ang)[:, None] + nu * np.sin(ang)[:, None]
            ll_prop = log_lik(prop)
            ok = pending & (ll_prop > log_thresh)
            z_next[ok], ll_next[ok] = prop[ok], ll_prop[ok]
            pending &= ~ok
            ang_lo = np.where(pending & (ang < 0), ang, ang_lo)
            ang_hi = np.where(pending & (ang >= 0), ang, ang_hi)
            ang = np.where(pending, rng.uniform(ang_lo, ang_hi), ang)
        z_ch, ll_ch = z_next, ll_next
        if it >= N_BURN:
            sum_z += z_ch
            sum_z2 += z_ch**2
            sum_lam += np.exp(z_ch)
    n_keep = N_ITER - N_BURN
    chain_mean_z = sum_z / n_keep
    W_within = (sum_z2 / n_keep - chain_mean_z**2).mean(axis=0)
    B_between = n_keep * chain_mean_z.var(axis=0, ddof=1)
    r_hat = np.sqrt(((n_keep - 1) / n_keep * W_within + B_between / n_keep) / W_within)
    return chain_mean_z.mean(axis=0), (sum_lam / n_keep).mean(axis=0), float(r_hat.max())


def run_one(m: int) -> dict:
    K = m * m
    A = 1.0 / K
    edges = np.linspace(0.0, 1.0, m + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cy, cx = np.meshgrid(centers, centers, indexing="ij")
    centroids = np.column_stack([cx.ravel(), cy.ravel()])

    Sigma_K = kernel_matrix(centroids, SIGMA_Z, ELL) + 1e-6 * np.eye(K)
    L_K = np.linalg.cholesky(Sigma_K)

    # m = 16 reproduces the notebook's cell 23 exactly (same seeds, same draw
    # order); the other grids get deterministic offsets of the same seeds.
    offset = 0 if m == 16 else m * 53
    rng = np.random.default_rng(SEED + 11 + offset)
    z_true = truth_field(cx, cy, MU_0)
    lambda_true = np.exp(z_true)
    y_obs = rng.poisson(A * lambda_true)

    def log_lik(z_batch: np.ndarray) -> np.ndarray:
        return (y_obs[None, :] * z_batch).sum(axis=1) - A * np.exp(z_batch).sum(axis=1)

    t0 = time.perf_counter()

    zeta = rng.normal(0, 1, size=(N_SMC, K))
    theta = MU_0 + zeta @ L_K.T
    log_w = np.zeros(N_SMC)
    log_lik_curr = log_lik(theta)

    beta = 0.0
    log_z_running = 0.0
    rho = RHO_INIT
    beta_trace = [0.0]
    ess_trace = [float(N_SMC)]
    log_z_trace = [0.0]
    accept_trace: list[float] = []
    rho_trace: list[float] = []
    posterior_mean_history = [np.exp(theta).mean(axis=0)]

    step = 0
    while beta < 1.0 and step < MAX_STEPS:
        # Conditional-ESS bisection (Zhou, Johansen & Aston 2016), as in the
        # notebook's cell 23: choose beta_{t+1} so that the incremental weights
        # keep CESS = tau_adapt * N under the current normalized weights.
        def cess_at(b: float) -> float:
            log_W = log_w - logsumexp(log_w)
            log_G = (b - beta) * log_lik_curr
            return float(
                N_SMC * np.exp(2.0 * logsumexp(log_W + log_G) - logsumexp(log_W + 2.0 * log_G))
            )

        if cess_at(1.0) >= TAU_ADAPT * N_SMC:
            new_beta = 1.0
        else:
            lo, hi = beta, 1.0
            for _ in range(60):
                if hi - lo < 1e-9:
                    break
                mid = 0.5 * (lo + hi)
                if cess_at(mid) >= TAU_ADAPT * N_SMC:
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

        # pCN mutation targeting gamma_{beta_{t+1}}: rho fixed within the step,
        # retuned from this step's acceptance before the next step.
        theta, log_lik_curr, acc = pcn_sweeps(
            theta, log_lik_curr, new_beta, rho, N_SWEEPS, rng, log_lik, L_K
        )
        accept_trace.append(acc)
        rho_trace.append(rho)
        rho = float(np.clip(rho * np.exp(RHO_GAIN * (acc - TARGET_ACC)), 0.01, 1.0))

        beta = new_beta
        step += 1
        beta_trace.append(float(beta))
        ess_trace.append(ess)
        log_z_trace.append(log_z_running)
        posterior_mean_history.append(np.exp(log_w - logsumexp(log_w)) @ np.exp(theta))

    elapsed = time.perf_counter() - t0

    w_norm = np.exp(log_w - logsumexp(log_w))
    z_post_mean = w_norm @ theta
    lambda_post = w_norm @ np.exp(theta)
    rmse_intensity = float(np.sqrt(np.mean((lambda_post - lambda_true) ** 2)))

    rng_ref = np.random.default_rng(SEED + 111 + offset)
    z_ref, lambda_ref, r_hat_max = elliptical_slice_reference(log_lik, L_K, K, rng_ref)
    rmse_ref = float(np.sqrt(np.mean((lambda_ref - lambda_true) ** 2)))

    high_cell = int(np.argmax(lambda_true))
    low_cell = int(np.argmin(lambda_true))
    post_history_arr = np.array(posterior_mean_history)

    return {
        "m": m,
        "K": K,
        "centers": centers.astype(float).tolist(),
        "y_obs": y_obs.astype(int).tolist(),
        "lambda_true": lambda_true.astype(float).tolist(),
        "lambda_post": lambda_post.astype(float).tolist(),
        "lambda_ref": lambda_ref.astype(float).tolist(),
        "rmse_intensity": rmse_intensity,
        "rmse_ref": rmse_ref,
        "dist_z_ref": float(np.sqrt(np.mean((z_post_mean - z_ref) ** 2))),
        "dist_z_prior": float(np.sqrt(np.mean((MU_0 - z_ref) ** 2))),
        "dist_lambda_ref": float(np.sqrt(np.mean((lambda_post - lambda_ref) ** 2))),
        "r_hat_max": r_hat_max,
        "beta_trace": beta_trace,
        "ess_trace": ess_trace,
        "log_z_trace": log_z_trace,
        "accept_trace": accept_trace,
        "rho_trace": rho_trace,
        "distinct_final": int(len(np.unique(theta[:, 0]))),
        "lambda_high_trace": post_history_arr[:, high_cell].astype(float).tolist(),
        "lambda_low_trace": post_history_arr[:, low_cell].astype(float).tolist(),
        "high_cell_idx": high_cell,
        "low_cell_idx": low_cell,
        "lambda_high_truth": float(lambda_true[high_cell]),
        "lambda_low_truth": float(lambda_true[low_cell]),
        "lambda_high_ref": float(lambda_ref[high_cell]),
        "lambda_low_ref": float(lambda_ref[low_cell]),
        "lambda_max": float(lambda_true.max()),
        "T_steps": step,
        "wall_seconds": float(elapsed),
        "log_z_T": float(log_z_running),
    }


def main() -> None:
    print(f"precompute_lgcp.py: running LGCP SMC at m ∈ {M_VALUES}, N = {N_SMC}, "
          f"adaptive schedule (τ_adapt = {TAU_ADAPT}), pCN × {N_SWEEPS} sweeps/step")
    payload = {
        "seed": SEED,
        "sigma_z_sq": SIGMA_Z**2,
        "ell": ELL,
        "mu_0": MU_0,
        "tau_adapt": TAU_ADAPT,
        "tau_resample": TAU_RESAMPLE,
        "N": N_SMC,
        "kernel": "pCN",
        "n_sweeps": N_SWEEPS,
        "target_accept": TARGET_ACC,
        "grids": {},
    }
    for m in M_VALUES:
        print(f"  running m = {m} (K = {m*m})...", flush=True)
        g = run_one(m)
        payload["grids"][str(m)] = g
        print(
            f"    points = {sum(g['y_obs'])}, T = {g['T_steps']}, "
            f"ESS {min(g['ess_trace'][1:]):.1f}–{max(g['ess_trace'][1:]):.1f}, "
            f"acc mean {np.mean(g['accept_trace']):.3f} "
            f"(min {min(g['accept_trace']):.3f}, max {max(g['accept_trace']):.3f}), "
            f"distinct = {g['distinct_final']}, log Ẑ = {g['log_z_T']:.3f}, wall = {g['wall_seconds']:.1f}s\n"
            f"    RMSE vs truth: SMC {g['rmse_intensity']:.2f}, reference {g['rmse_ref']:.2f}; "
            f"distance to reference: z {g['dist_z_ref']:.3f} (prior {g['dist_z_prior']:.3f}), "
            f"λ {g['dist_lambda_ref']:.2f}; max R-hat {g['r_hat_max']:.3f}"
        )

    serialized = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        out_path = d / "lgcp.json"
        out_path.write_text(serialized)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
