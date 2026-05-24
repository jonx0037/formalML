"""Precompute the §8.4 closed-form linear-Gaussian VIB trajectory and the §9 GD
trajectory at β = 5, dumping to JSON for the VIBvsIBSandbox viz component.

Mirrors notebook cells 33 (Figure 9) and 36 (Figure 10) exactly — same RNG seeds,
same L-BFGS-B settings, same 40 log-spaced β grid. The output JSON is consumed
by `src/components/viz/VIBvsIBSandbox.tsx` at runtime.

Per CLAUDE.md: writes to BOTH `src/data/sampleData/information-bottleneck/` and
`public/sample-data/information-bottleneck/`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "information-bottleneck",
    REPO_ROOT / "public" / "sample-data" / "information-bottleneck",
]

# §6.4 4-D toy setup — identical to notebook cell 26.
P_DIM = 4
RHO_DESIGN = np.array([0.95, 0.70, 0.30, 0.00])
SIGMA_X = np.eye(P_DIM)
SIGMA_Y = np.eye(P_DIM)
SIGMA_XY = np.diag(RHO_DESIGN)
SIGMA_YX = SIGMA_XY.T


def vib_objective(params, beta, Sigma_X, Sigma_Y, Sigma_XY, p):
    A = params[: p * p].reshape(p, p)
    log_sigma = params[p * p :]
    sigma2 = np.exp(2.0 * log_sigma)
    Sigma_xi = np.diag(sigma2)
    Sigma_T = A @ Sigma_X @ A.T + Sigma_xi
    rate = 0.5 * (np.trace(Sigma_T) - 2.0 * np.sum(log_sigma) - p)
    Sigma_T_inv = np.linalg.inv(Sigma_T)
    Sigma_Y_cond_T = Sigma_Y - Sigma_XY.T @ A.T @ Sigma_T_inv @ A @ Sigma_XY
    _, logdet = np.linalg.slogdet(Sigma_Y_cond_T)
    distortion = 0.5 * logdet
    return rate + beta * distortion


def evaluate_vib_point(params, Sigma_X, Sigma_Y, Sigma_XY, p):
    A = params[: p * p].reshape(p, p)
    log_sigma = params[p * p :]
    Sigma_xi = np.diag(np.exp(2.0 * log_sigma))
    Sigma_T = A @ Sigma_X @ A.T + Sigma_xi
    Sigma_Y_cond_T = Sigma_Y - Sigma_XY.T @ A.T @ np.linalg.solve(Sigma_T, A @ Sigma_XY)
    _, logdet_T = np.linalg.slogdet(Sigma_T)
    _, logdet_xi = np.linalg.slogdet(Sigma_xi)
    _, logdet_Y = np.linalg.slogdet(Sigma_Y)
    _, logdet_YT = np.linalg.slogdet(Sigma_Y_cond_T)
    I_XT = 0.5 * (logdet_T - logdet_xi)
    I_TY = 0.5 * (logdet_Y - logdet_YT)
    return float(I_XT), float(I_TY)


def closed_form_ib_curve(betas):
    """The §7 closed-form Gaussian IB curve at the same β grid."""
    lam = np.array([1 - r * r for r in RHO_DESIGN])  # eigenvalues of Σ_{X|Y} Σ_X^{-1}
    safe_lam = np.clip(lam, 1e-12, None)
    I_XT, I_TY = [], []
    for b in betas:
        active = b > 1.0 / np.clip(1 - lam, 1e-12, None)
        ixt = 0.0
        ity = 0.0
        for i in range(P_DIM):
            if active[i] and lam[i] < 1:
                ixt += 0.5 * np.log((b - 1) * (1 - lam[i]) / safe_lam[i])
                ity += 0.5 * np.log((b - 1) / (b * safe_lam[i]))
        I_XT.append(float(ixt))
        I_TY.append(float(ity))
    return np.array(I_XT), np.array(I_TY)


def run_vib_sweep():
    betas = np.logspace(0.05, 2.5, 40)

    rng_vib = np.random.default_rng(20260524)
    params = np.concatenate([
        0.05 * rng_vib.standard_normal(P_DIM * P_DIM),
        1e-3 * rng_vib.standard_normal(P_DIM),
    ])

    rows = []
    for beta in betas:
        res = minimize(
            vib_objective,
            params,
            args=(beta, SIGMA_X, SIGMA_Y, SIGMA_XY, P_DIM),
            method="L-BFGS-B",
            options={"maxiter": 500, "gtol": 1e-9},
        )
        params = res.x
        ixt, ity = evaluate_vib_point(res.x, SIGMA_X, SIGMA_Y, SIGMA_XY, P_DIM)
        rows.append({"beta": float(beta), "I_XT": ixt, "I_TY": ity, "L_VIB": float(res.fun)})
    return betas, rows


def run_gd_trajectory(beta_traj=5.0, n_steps=400, lr=0.04):
    """§9 / Figure 10 GD trajectory in the information plane on linear-Gaussian VIB."""

    def numerical_gradient(f, x, args=(), eps=1e-7):
        g = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            xm = x.copy()
            xm[i] -= eps
            g[i] = (f(xp, *args) - f(xm, *args)) / (2.0 * eps)
        return g

    rng = np.random.default_rng(20260524)
    params = np.concatenate([
        0.03 * rng.standard_normal(P_DIM * P_DIM),
        -0.5 * np.ones(P_DIM) + 0.01 * rng.standard_normal(P_DIM),
    ])

    traj_IXT, traj_ITY = [], []
    for _ in range(n_steps):
        ixt, ity = evaluate_vib_point(params, SIGMA_X, SIGMA_Y, SIGMA_XY, P_DIM)
        traj_IXT.append(ixt)
        traj_ITY.append(ity)
        g = numerical_gradient(
            vib_objective, params, args=(beta_traj, SIGMA_X, SIGMA_Y, SIGMA_XY, P_DIM)
        )
        params = params - lr * g
    return traj_IXT, traj_ITY


def main():
    print("Running L-BFGS-B VIB sweep (40 betas, warm-started)...")
    betas, vib_rows = run_vib_sweep()
    print(f"  done. {len(vib_rows)} points.")

    print("Computing closed-form Gaussian IB curve at the same betas...")
    I_XT_ib, I_TY_ib = closed_form_ib_curve(betas)
    ib_rows = [
        {"beta": float(b), "I_XT": float(ix), "I_TY": float(iy)}
        for b, ix, iy in zip(betas, I_XT_ib, I_TY_ib)
    ]

    # Also compute a denser IB curve (for plotting) — 200 betas from β_c,1 to 200.
    betas_dense = np.logspace(np.log10(1.108 * 1.001), np.log10(316.0), 200)
    I_XT_dense, I_TY_dense = closed_form_ib_curve(betas_dense)
    ib_dense = [
        {"beta": float(b), "I_XT": float(ix), "I_TY": float(iy)}
        for b, ix, iy in zip(betas_dense, I_XT_dense, I_TY_dense)
    ]

    # Variational gap I(T;Y)_IB - I(T;Y)_VIB, interpolated to the VIB beta grid
    gap = [
        max(0.0, ib["I_TY"] - vb["I_TY"]) for ib, vb in zip(ib_rows, vib_rows)
    ]

    print("Running §9 GD trajectory at beta=5...")
    traj_IXT, traj_ITY = run_gd_trajectory(beta_traj=5.0, n_steps=400, lr=0.04)
    print(f"  done. {len(traj_IXT)} steps.")

    peak_IXT = float(max(traj_IXT))
    final_IXT = float(traj_IXT[-1])
    compression_delta = peak_IXT - final_IXT

    rho = RHO_DESIGN.tolist()
    I_XY_total = float(-0.5 * sum(np.log(1 - r * r) for r in rho if r > 0))

    payload = {
        "_metadata": {
            "p_dim": P_DIM,
            "rho_design": rho,
            "I_XY_nats": I_XY_total,
            "beta_c": [
                float(1.0 / max(1 - r * r, 1e-12)) if r > 0 else None for r in rho
            ],
            "seed": 20260524,
        },
        "vib_trajectory": vib_rows,
        "ib_at_vib_betas": ib_rows,
        "variational_gap": gap,
        "ib_curve_dense": ib_dense,
        "gd_trajectory_beta5": {
            "beta": 5.0,
            "n_steps": 400,
            "lr": 0.04,
            "I_XT": [float(v) for v in traj_IXT],
            "I_TY": [float(v) for v in traj_ITY],
            "peak_I_XT": peak_IXT,
            "final_I_XT": final_IXT,
            "compression_delta": compression_delta,
        },
    }

    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        path = d / "vib_sandbox.json"
        with open(path, "w") as f:
            json.dump(payload, f, allow_nan=False)
        print(f"  wrote {path}")

    print("\nSummary:")
    print(f"  I(X;Y) = {I_XY_total:.4f} nats")
    print(f"  VIB at β = {vib_rows[0]['beta']:.2f}: (I(X;T), I(T;Y)) = ({vib_rows[0]['I_XT']:.4f}, {vib_rows[0]['I_TY']:.4f})")
    print(f"  VIB at β = {vib_rows[-1]['beta']:.2f}: (I(X;T), I(T;Y)) = ({vib_rows[-1]['I_XT']:.4f}, {vib_rows[-1]['I_TY']:.4f})")
    print(f"  GD trajectory at β=5: peak I(X;T) = {peak_IXT:.4f}, final = {final_IXT:.4f}, Δ = {compression_delta:.4f}")


if __name__ == "__main__":
    main()
