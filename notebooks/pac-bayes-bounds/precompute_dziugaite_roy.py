"""Precompute the §11.5 Dziugaite–Roy bound reproduction and serialize the result.

Standalone NumPy/SciPy/sklearn duplicate of the notebook's §11.5 cell.  Mirrors
the notebook's seeds, hyperparameters, and optimization loop so the final
certificate matches the verified value of 0.2369 at σ²_P = 0.030.  Writes
JSON to BOTH ``src/data/sampleData/pac-bayes-bounds/`` and
``public/sample-data/pac-bayes-bounds/`` per CLAUDE.md's "sample-data
dual-location" guidance.

Runtime budget: ~8–15 s on a 2020-era laptop (the slow step is the prior-grid
sweep, 5 priors × 500 Adam steps).  CPU-only, no PyTorch / JAX dependency.

Run::

    cd notebooks/pac-bayes-bounds
    .venv/bin/python precompute_dziugaite_roy.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

# ---------------------------------------------------------------------------
# Output locations (dual-write, per CLAUDE.md)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "pac-bayes-bounds",
    REPO_ROOT / "public" / "sample-data" / "pac-bayes-bounds",
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters — exact match to notebook §11.5
# ---------------------------------------------------------------------------

DELTA_DR = 0.05
SIGMA_GRID = np.array([0.01, 0.03, 0.1, 0.3, 1.0])
K_GRID = len(SIGMA_GRID)
N_ADAM_STEPS = 500
LR = 0.01
H_DIM = 16
N_TR = 1000
N_TE = 500
SGD_RANDOM_STATE = 42
MNIST_RNG_SEED = 20260512
ADAM_NOISE_SEED = 42


def main() -> None:
    t_total = time.time()

    print("Loading MNIST binary subset (digits 0 vs 1)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X_all = mnist.data / 255.0
    y_all = mnist.target.astype(int)
    mask = (y_all == 0) | (y_all == 1)
    X_01 = X_all[mask]
    y_01 = (y_all[mask] == 1).astype(np.float64)

    rng_mn = np.random.default_rng(MNIST_RNG_SEED)
    idx_perm = rng_mn.permutation(len(X_01))
    X_train = X_01[idx_perm[:N_TR]]
    y_train = y_01[idx_perm[:N_TR]]
    X_test = X_01[idx_perm[N_TR : N_TR + N_TE]]
    y_test = y_01[idx_perm[N_TR : N_TR + N_TE]]
    d_in = X_train.shape[1]
    print(f"  n_train = {N_TR}, n_test = {N_TE}, d_in = {d_in}")

    print("\nStep 1: SGD training with sklearn MLPClassifier...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(H_DIM,),
        activation="relu",
        solver="adam",
        learning_rate_init=0.01,
        max_iter=80,
        random_state=SGD_RANDOM_STATE,
        tol=1e-6,
    )
    mlp.fit(X_train, y_train.astype(int))
    test_err_sgd = float(np.mean(mlp.predict(X_test) != y_test))
    print(f"  trained in {time.time() - t_total:.1f}s, test 0/1 error = {test_err_sgd:.4f}")

    W1_0 = mlp.coefs_[0].T
    b1_0 = mlp.intercepts_[0]
    W2_0 = mlp.coefs_[1].flatten()
    b2_0 = float(mlp.intercepts_[1][0])
    d_dim = W1_0.size + b1_0.size + W2_0.size + 1
    print(f"  total parameters d = {d_dim}")

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        s = 0
        W1 = theta[s : s + H_DIM * d_in].reshape(H_DIM, d_in)
        s += H_DIM * d_in
        b1 = theta[s : s + H_DIM]
        s += H_DIM
        W2 = theta[s : s + H_DIM]
        s += H_DIM
        b2 = float(theta[s])
        return W1, b1, W2, b2

    theta_0 = np.concatenate([W1_0.flatten(), b1_0, W2_0, [b2_0]])

    def forward(theta: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        W1, b1, W2, b2 = unpack(theta)
        z1 = X @ W1.T + b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ W2 + b2
        return z1, h1, z2

    def bce_loss_and_grad(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        W1, b1, W2, b2 = unpack(theta)
        n_ = X.shape[0]
        z1, h1, z2 = forward(theta, X)
        p_pred = 1.0 / (1.0 + np.exp(-np.clip(z2, -50.0, 50.0)))
        loss = float(-np.mean(y * np.log(p_pred + 1e-12) + (1 - y) * np.log(1 - p_pred + 1e-12)))
        dz2 = (p_pred - y) / n_
        dW2 = h1.T @ dz2
        db2 = float(dz2.sum())
        dh1 = np.outer(dz2, W2)
        dz1 = dh1 * (z1 > 0)
        dW1 = dz1.T @ X
        db1 = dz1.sum(axis=0)
        grad = np.concatenate([dW1.flatten(), db1, dW2, [db2]])
        return loss, grad

    def zero_one_loss(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        _, _, z2 = forward(theta, X)
        preds = (z2 > 0).astype(np.float64)
        return float(np.mean(preds != y))

    def gaussian_kl_perparam(
        mu: np.ndarray, log_sigma2_Q: np.ndarray, sigma2_P: float
    ) -> tuple[float, float, float]:
        sigma2_Q = np.exp(log_sigma2_Q)
        mean_shift = float((mu**2).sum() / (2.0 * sigma2_P))
        rho = sigma2_Q / sigma2_P
        variance = float(0.5 * (rho - 1.0 - np.log(rho)).sum())
        return mean_shift + variance, mean_shift, variance

    def catoni_opt_certificate(emp_risk: float, kl: float, n_: int, delta_eff: float) -> float:
        return emp_risk + math.sqrt((kl + math.log(1.0 / delta_eff)) / (2.0 * n_))

    def optimize_bound_at_prior(sigma2_P: float, n_steps: int = N_ADAM_STEPS) -> tuple[
        np.ndarray, np.ndarray, list[dict]
    ]:
        mu = theta_0.copy()
        log_sigma2_Q = np.full(d_dim, np.log(1e-4))
        m_mu, v_mu = np.zeros(d_dim), np.zeros(d_dim)
        m_ls, v_ls = np.zeros(d_dim), np.zeros(d_dim)
        beta1, beta2, eps_a = 0.9, 0.999, 1e-8

        delta_eff = DELTA_DR / K_GRID
        trajectory: list[dict] = []
        rng_local = np.random.default_rng(ADAM_NOISE_SEED)

        for step in range(n_steps):
            sigma_Q = np.exp(0.5 * log_sigma2_Q)
            epsilon = rng_local.standard_normal(d_dim)
            w = mu + sigma_Q * epsilon

            _, grad_w = bce_loss_and_grad(w, X_train, y_train)
            grad_mu_data = grad_w
            grad_ls_data = 0.5 * sigma_Q * epsilon * grad_w

            kl_total, _, _ = gaussian_kl_perparam(mu, log_sigma2_Q, sigma2_P)
            c_num = kl_total + math.log(K_GRID / DELTA_DR)
            lambda_star = math.sqrt(8.0 * N_TR * max(c_num, 1.0))

            grad_mu_total = grad_mu_data + (mu / sigma2_P) / lambda_star
            grad_ls_total = grad_ls_data + 0.5 * (np.exp(log_sigma2_Q) / sigma2_P - 1.0) / lambda_star

            m_mu = beta1 * m_mu + (1 - beta1) * grad_mu_total
            v_mu = beta2 * v_mu + (1 - beta2) * grad_mu_total**2
            m_hat = m_mu / (1 - beta1 ** (step + 1))
            v_hat = v_mu / (1 - beta2 ** (step + 1))
            mu -= LR * m_hat / (np.sqrt(v_hat) + eps_a)

            m_ls = beta1 * m_ls + (1 - beta1) * grad_ls_total
            v_ls = beta2 * v_ls + (1 - beta2) * grad_ls_total**2
            m_hat = m_ls / (1 - beta1 ** (step + 1))
            v_hat = v_ls / (1 - beta2 ** (step + 1))
            log_sigma2_Q -= LR * m_hat / (np.sqrt(v_hat) + eps_a)
            log_sigma2_Q = np.clip(log_sigma2_Q, -20.0, math.log(sigma2_P * 4.0))

            if step % 25 == 0 or step == n_steps - 1:
                emp_risks_mc = []
                for _ in range(3):
                    eps_eval = rng_local.standard_normal(d_dim)
                    w_eval = mu + np.exp(0.5 * log_sigma2_Q) * eps_eval
                    emp_risks_mc.append(zero_one_loss(w_eval, X_train, y_train))
                emp_risk_est = float(np.mean(emp_risks_mc))
                kl_est, ms_est, var_est = gaussian_kl_perparam(mu, log_sigma2_Q, sigma2_P)
                bound = catoni_opt_certificate(emp_risk_est, kl_est, N_TR, delta_eff)
                trajectory.append(
                    {
                        "step": step,
                        "emp_risk": emp_risk_est,
                        "kl": kl_est,
                        "mean_shift": ms_est,
                        "variance": var_est,
                        "bound": bound,
                    }
                )

        return mu, log_sigma2_Q, trajectory

    print("\nSteps 2-4: bound optimization across prior grid...")
    best_bound = math.inf
    best_traj: list[dict] = []
    best_sigma2_P = float(SIGMA_GRID[0])

    for sigma2_P in SIGMA_GRID:
        t0 = time.time()
        _, _, traj = optimize_bound_at_prior(float(sigma2_P))
        final_bound = traj[-1]["bound"]
        print(
            f"  sigma_P^2 = {float(sigma2_P):>5.3f}:  final bound = {final_bound:.4f}  "
            f"({time.time() - t0:.1f}s)"
        )
        if final_bound < best_bound:
            best_bound = final_bound
            best_traj = traj
            best_sigma2_P = float(sigma2_P)

    print(f"\n*** Best PAC-Bayes certificate: {best_bound:.4f} at sigma_P^2 = {best_sigma2_P:.3f} ***")
    print(f"*** SGD network test 0/1 error:  {test_err_sgd:.4f} ***")
    print("*** Rademacher / classical bound: >= 1.0 (vacuous) ***")
    print(f"\nTotal runtime: {time.time() - t_total:.1f}s")

    # Build JSON payload.  The decomposition exposes the ADDITIVE pieces of the
    # numerator inside the Catoni slack — `mean_shift_kl_nats + variance_kl_nats
    # + log_K_delta_nats = numerator_nats` — and the resulting `slack` so the
    # viz can plot a mathematically honest sum-to-the-numerator bar chart.
    final = best_traj[-1]
    delta_eff = DELTA_DR / K_GRID
    log_K_delta_nats = math.log(1.0 / delta_eff)
    mean_shift_nats = float(final["mean_shift"])
    variance_nats = float(final["variance"])
    numerator_nats = mean_shift_nats + variance_nats + log_K_delta_nats
    slack = math.sqrt(numerator_nats / (2.0 * N_TR))

    payload = {
        "bound_trajectory": {
            "steps": [int(t["step"]) for t in best_traj],
            "values": [float(t["bound"]) for t in best_traj],
        },
        "final_decomposition": {
            "empirical_risk": float(final["emp_risk"]),
            "mean_shift_kl_nats": mean_shift_nats,
            "variance_kl_nats": variance_nats,
            "log_K_delta_nats": float(log_K_delta_nats),
            "numerator_nats": float(numerator_nats),
            "slack": float(slack),
            "total": float(best_bound),
        },
        "baseline_comparison": {
            "sgd_test_01": float(test_err_sgd),
            "rademacher_bound": 1.0,
            "pac_bayes_bound": float(best_bound),
        },
        "best_prior_sigma2": float(best_sigma2_P),
        "prior_grid": [float(s) for s in SIGMA_GRID],
        "config": {
            "n_train": int(N_TR),
            "n_test": int(N_TE),
            "hidden": int(H_DIM),
            "adam_steps": int(N_ADAM_STEPS),
            "K_final_mc": 3,
            "delta": float(DELTA_DR),
        },
    }

    # JSON-validity guard: allow_nan=False per CLAUDE.md
    text = json.dumps(payload, indent=2, allow_nan=False)
    for d in OUT_DIRS:
        out_path = d / "dziugaite_roy.json"
        out_path.write_text(text)
        print(f"  wrote {out_path.relative_to(REPO_ROOT)} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
