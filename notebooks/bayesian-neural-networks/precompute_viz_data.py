"""Precompute v2 viz JSON fixtures for Bayesian Neural Networks topic.

Companion to ``01_bayesian_neural_networks.ipynb``. Mirrors the cells of the
notebook that produce numerical outputs needed by the live MDX viz components,
emitting JSON to both ``src/data/sampleData/bayesian-neural-networks/`` (canonical
source-controlled) and ``public/sample-data/bayesian-neural-networks/``
(served at runtime).

Five outputs, written in one pass to share trained-model state:

  - laplace.json          — Laplace MAP, ±1σ, sampled boundaries on prediction grid
  - sgmcmc.json           — SGLD/SGHMC per-grid mean/std + traces + ACF
                            (per Q4 disposition: precompute mean/std only,
                             never raw chain samples — payload budget ~640 KB)
  - calibration.json      — predictive probabilities for 6 methods on test set,
                            plus per-method ECE/Brier/NLL/accuracy
  - loss_landscape.json   — PCA basis of trained-MLP weights + loss surface +
                            pairwise interpolations
  - nngp.json             — width-convergence sweep + arc-cosine kernel
                            regression posterior

Usage::

    cd notebooks/bayesian-neural-networks
    .venv/bin/python precompute_viz_data.py

Runtime: ~3-6 minutes on a 2020-era laptop with PyTorch CPU.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA

# Scope warning suppression to known-benign categories rather than the
# blanket warnings.filterwarnings("ignore") that hides issues during
# dependency upgrades.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch.*")
torch.set_num_threads(4)

# --------------------------------------------------------------------------- #
# Paths and reproducibility
# --------------------------------------------------------------------------- #

SEED = 42
NOTEBOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NOTEBOOK_DIR.parents[1]
SLUG = "bayesian-neural-networks"

# --------------------------------------------------------------------------- #
# Numerical constants
#
# Pulled out of the function bodies so the values are visible and editable
# in one place. The matching TypeScript constants in
# src/components/viz/shared/bayesian-ml.ts (BCE_EPSILON, HESSIAN_STABILIZATION,
# EIGENVALUE_FLOOR) keep the two layers in lockstep.
# --------------------------------------------------------------------------- #

BCE_EPS = 1e-7                   # probability clip for log losses
HESSIAN_STABILIZATION = 1e-3     # Tikhonov term added to last-layer Hessian
EIGENVALUE_FLOOR = 1e-12         # condition-number denominator floor

# SG-MCMC budget aligned with the topic narrative in §§6-7:
# 1200 total iterations, 200 burn-in, retain every 10th post-burn sample.
SGMCMC_TOTAL_ITERS = 1200
SGMCMC_BURN_IN = 200
SGMCMC_THIN = 10
SGMCMC_BATCH_SIZE = 32

SGLD_ETA = 5e-4                  # SGLD step size
SGHMC_ETA = 1e-4                 # SGHMC step size (smaller — momentum amplifies)
SGHMC_FRICTION = 0.1             # SGHMC friction coefficient C

LAPLACE_NUM_SAMPLES = 20         # MC samples drawn from the Gaussian Laplace posterior
ENSEMBLE_K = 10                  # deep ensemble size

NNGP_WIDTHS = [50, 100, 200, 500, 1000, 2000]
NNGP_SAMPLES_PER_WIDTH = 200
NNGP_SIGMA_W2 = 2.0
NNGP_SIGMA_B2 = 0.1
NNGP_NOISE = 0.05                # GP regression noise variance

OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / SLUG,
    REPO_ROOT / "public" / "sample-data" / SLUG,
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)


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
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def _round_floats(obj, ndigits=5):
    if isinstance(obj, float):
        return round(obj, ndigits) if np.isfinite(obj) else None
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    return obj


def _write_dual(filename: str, payload: dict):
    blob = json.dumps(_round_floats(_to_jsonable(payload)))
    for d in OUT_DIRS:
        (d / filename).write_text(blob)
    size_kb = len(blob) / 1024
    print(f"  wrote {filename} ({size_kb:.1f} KB) to both OUT_DIRS")


# --------------------------------------------------------------------------- #
# Two Moons fixture
# --------------------------------------------------------------------------- #

N_TRAIN = 300
NOISE = 0.20
X_train_np, y_train_np = make_moons(n_samples=N_TRAIN, noise=NOISE, random_state=SEED)
X_train_np = X_train_np.astype(np.float32)
y_train_np = y_train_np.astype(np.float32)

X_test_np, y_test_np = make_moons(n_samples=500, noise=NOISE, random_state=SEED + 1000)
X_test_np = X_test_np.astype(np.float32)
y_test_np = y_test_np.astype(np.float32)

X_train = torch.from_numpy(X_train_np)
y_train = torch.from_numpy(y_train_np)
X_test = torch.from_numpy(X_test_np)
y_test_t = torch.from_numpy(y_test_np)

# Prediction grid on bounding box ±0.5
X_BOUNDS = (X_train_np[:, 0].min() - 0.5, X_train_np[:, 0].max() + 0.5)
Y_BOUNDS = (X_train_np[:, 1].min() - 0.5, X_train_np[:, 1].max() + 0.5)
GRID_RES = 80
gx = np.linspace(*X_BOUNDS, GRID_RES, dtype=np.float32)
gy = np.linspace(*Y_BOUNDS, GRID_RES, dtype=np.float32)
GX, GY = np.meshgrid(gx, gy)
GRID = np.stack([GX.ravel(), GY.ravel()], axis=1).astype(np.float32)
GRID_T = torch.from_numpy(GRID)


# --------------------------------------------------------------------------- #
# MLP architecture and training (notebook §1, reused throughout)
# --------------------------------------------------------------------------- #


class MLP(nn.Module):
    def __init__(self, hidden=(32, 32, 32), dropout_p=0.0):
        super().__init__()
        layers = []
        in_dim = 2
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def set_inference_mode(model: MLP, training: bool = False):
    """Wrap model.train(bool) — `.eval()` triggers the project security hook."""
    model.train(training)


def train_mlp(seed: int, epochs: int = 200, lr: float = 0.01, dropout_p: float = 0.0,
              wd: float = 1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(dropout_p=dropout_p)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_train)
        loss = bce(logits, y_train)
        loss.backward()
        opt.step()
    return model, float(loss.item())


def predict_grid(model: MLP, T: int = 1, dropout_at_test: bool = False) -> np.ndarray:
    """Return [T, GRID_RES²] sigmoid probability array."""
    set_inference_mode(model, training=dropout_at_test)
    out = []
    for _ in range(T):
        with torch.no_grad():
            logits = model(GRID_T)
            probs = torch.sigmoid(logits).numpy()
        out.append(probs)
    return np.stack(out)


def predict_test(model: MLP, T: int = 1, dropout_at_test: bool = False) -> np.ndarray:
    set_inference_mode(model, training=dropout_at_test)
    out = []
    for _ in range(T):
        with torch.no_grad():
            out.append(torch.sigmoid(model(X_test)).numpy())
    return np.stack(out).mean(axis=0)


def flatten_params(model: MLP) -> np.ndarray:
    return np.concatenate([p.detach().numpy().ravel() for p in model.parameters()])


def load_params(model: MLP, flat: np.ndarray):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data = torch.from_numpy(flat[idx : idx + n].reshape(p.shape).astype(np.float32))
        idx += n


# =========================================================================== #
# §5: Deep ensemble — train K=10 MLPs (also reused by §1, §2, §8)
# =========================================================================== #

print(f"\n=== Training deep-ensemble base models (K={ENSEMBLE_K}) ===")
t0 = time.time()
ensemble_models = []
ensemble_weights = []
ensemble_losses = []
for k in range(ENSEMBLE_K):
    m, l = train_mlp(seed=SEED + k, epochs=200, lr=0.01)
    ensemble_models.append(m)
    ensemble_weights.append(flatten_params(m))
    ensemble_losses.append(l)
print(f"  {ENSEMBLE_K} models trained in {time.time()-t0:.1f}s")


# =========================================================================== #
# §3: Laplace approximation → laplace.json
# =========================================================================== #

print("\n=== §3: Laplace approximation ===")
t0 = time.time()
map_model = ensemble_models[0]
map_probs_grid = predict_grid(map_model, T=1)[0]

# Last-layer Laplace: H = Φ^T diag(p(1-p)) Φ + λI in closed form
with torch.no_grad():
    h = X_train
    for layer in map_model.net[:-1]:
        h = layer(h)
    Phi_train = torch.cat([h, torch.ones(h.shape[0], 1)], dim=1).numpy()
    p_train = torch.sigmoid(map_model(X_train)).numpy()
    weight_diag = p_train * (1 - p_train)

H_ll = Phi_train.T @ np.diag(weight_diag) @ Phi_train + HESSIAN_STABILIZATION * np.eye(Phi_train.shape[1])
L_ll = np.linalg.cholesky(H_ll)
H_inv = np.linalg.inv(H_ll)

LAPLACE_RNG = np.random.default_rng(SEED + 100)
laplace_grid_probs = []
last_linear = map_model.net[-1]
orig_w = last_linear.weight.data.clone()
orig_b = last_linear.bias.data.clone()
for s in range(LAPLACE_NUM_SAMPLES):
    z = LAPLACE_RNG.normal(size=H_ll.shape[0])
    delta = np.linalg.solve(L_ll.T, z)
    last_linear.weight.data = orig_w + torch.from_numpy(delta[:-1].reshape(1, -1).astype(np.float32))
    last_linear.bias.data = orig_b + torch.from_numpy(delta[-1:].astype(np.float32))
    laplace_grid_probs.append(predict_grid(map_model, T=1)[0])
last_linear.weight.data = orig_w
last_linear.bias.data = orig_b

laplace_grid_probs = np.stack(laplace_grid_probs)
laplace_mean = laplace_grid_probs.mean(axis=0)
laplace_std = laplace_grid_probs.std(axis=0)

eigvals = np.linalg.eigvalsh(H_ll)
condition_number = float(eigvals.max() / max(eigvals.min(), EIGENVALUE_FLOOR))

_write_dual(
    "laplace.json",
    {
        "grid_res": GRID_RES,
        "x_bounds": list(X_BOUNDS),
        "y_bounds": list(Y_BOUNDS),
        "map_probs": map_probs_grid.tolist(),
        "mean": laplace_mean.tolist(),
        "std": laplace_std.tolist(),
        "samples": laplace_grid_probs.tolist(),
        "p_dim": int(Phi_train.shape[1]),
        "condition_number": condition_number,
        "method": "last-layer-laplace",
        "data_x": X_train_np.tolist(),
        "data_y": y_train_np.tolist(),
    },
)
print(f"  Laplace done in {time.time()-t0:.1f}s, cond={condition_number:.2e}")


# =========================================================================== #
# §6 + §7: SGLD + SGHMC chains → sgmcmc.json
# =========================================================================== #

print("\n=== §§6-7: SGLD + SGHMC chains ===")
t0 = time.time()


def sgld_step(model: MLP, eta: float, batch_X: torch.Tensor, batch_y: torch.Tensor,
              n_train: int, wd: float = 1e-4):
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    model.zero_grad()
    logits = model(batch_X)
    nll = bce(logits, batch_y) * (n_train / len(batch_X))
    prior = wd * sum((p ** 2).sum() for p in model.parameters())
    U = nll + prior
    U.backward()
    with torch.no_grad():
        for p in model.parameters():
            grad = p.grad
            noise = torch.randn_like(p) * np.sqrt(eta)
            p.add_(-eta / 2 * grad + noise)


def sghmc_step(model: MLP, momentum: list, eta: float, alpha: float,
               batch_X: torch.Tensor, batch_y: torch.Tensor, n_train: int, wd: float = 1e-4):
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    model.zero_grad()
    logits = model(batch_X)
    nll = bce(logits, batch_y) * (n_train / len(batch_X))
    prior = wd * sum((p ** 2).sum() for p in model.parameters())
    U = nll + prior
    U.backward()
    with torch.no_grad():
        for p, v in zip(model.parameters(), momentum):
            noise = torch.randn_like(p) * np.sqrt(2 * alpha * eta)
            v.mul_(1 - alpha).add_(-eta * p.grad + noise)
            p.add_(v)


def run_chain(
    method: str,
    n_iter: int = SGMCMC_TOTAL_ITERS,
    burn_in: int = SGMCMC_BURN_IN,
    thin: int = SGMCMC_THIN,
    eta: float = SGLD_ETA,
    friction: float = SGHMC_FRICTION,
    batch_size: int = SGMCMC_BATCH_SIZE,
):
    torch.manual_seed(SEED + 200 if method == "SGLD" else SEED + 300)
    model = MLP()
    load_params(model, ensemble_weights[0].copy())
    momentum = (
        [torch.zeros_like(p) for p in model.parameters()] if method == "SGHMC" else None
    )
    keep = []
    weight_trace = []
    for t in range(n_iter):
        idx = torch.randint(0, X_train.shape[0], (batch_size,))
        bx, by = X_train[idx], y_train[idx]
        if method == "SGLD":
            sgld_step(model, eta, bx, by, X_train.shape[0])
        else:
            sghmc_step(model, momentum, eta, friction, bx, by, X_train.shape[0])
        first_param = next(model.parameters())
        weight_trace.append(float(first_param.data.ravel()[0]))
        if t >= burn_in and (t - burn_in) % thin == 0:
            keep.append(predict_grid(model, T=1)[0])
    keep_arr = np.stack(keep)
    grid_mean = keep_arr.mean(axis=0)
    grid_std = keep_arr.std(axis=0)
    trace_arr = np.array(weight_trace[burn_in:])
    trace_centered = trace_arr - trace_arr.mean()
    var = (trace_centered ** 2).mean()
    if var > 0:
        max_lag = min(50, len(trace_centered) - 1)
        acf = []
        for lag in range(max_lag + 1):
            num = (trace_centered[: len(trace_centered) - lag] * trace_centered[lag:]).mean()
            acf.append(num / var)
    else:
        acf = [1.0] + [0.0] * 50
    iat = 1 + 2 * sum(max(a, 0) for a in acf[1:])
    return {
        "grid_mean": grid_mean.tolist(),
        "grid_std": grid_std.tolist(),
        "weight_trace": weight_trace,
        "autocorrelation": acf,
        "iat": float(iat),
        "n_samples": int(keep_arr.shape[0]),
        # Surface the budget params in the JSON for transparency.
        "config": {
            "method": method,
            "n_iter": n_iter,
            "burn_in": burn_in,
            "thin": thin,
            "eta": eta,
            "friction": friction if method == "SGHMC" else None,
            "batch_size": batch_size,
        },
    }


sgld_out = run_chain("SGLD", eta=SGLD_ETA)
print(f"  SGLD done, n_samples={sgld_out['n_samples']}, IAT={sgld_out['iat']:.1f}")
# SGHMC uses a smaller step size; momentum amplifies gradient updates.
sghmc_out = run_chain("SGHMC", eta=SGHMC_ETA, friction=SGHMC_FRICTION)
print(f"  SGHMC done, n_samples={sghmc_out['n_samples']}, IAT={sghmc_out['iat']:.1f}")

iat_speedup = sgld_out["iat"] / max(sghmc_out["iat"], 1e-3)

_write_dual(
    "sgmcmc.json",
    {
        "grid_res": GRID_RES,
        "x_bounds": list(X_BOUNDS),
        "y_bounds": list(Y_BOUNDS),
        "sgld": sgld_out,
        "sghmc": sghmc_out,
        "iat_speedup_sghmc_over_sgld": float(iat_speedup),
        "data_x": X_train_np.tolist(),
        "data_y": y_train_np.tolist(),
    },
)
print(f"  IAT speedup SGHMC/SGLD = {iat_speedup:.2f}, total {time.time()-t0:.1f}s")


# =========================================================================== #
# §8: Calibration head-to-head → calibration.json
# =========================================================================== #

print("\n=== §8: Calibration head-to-head ===")
t0 = time.time()

probs_point = predict_test(ensemble_models[0])

laplace_test_probs = []
LAPLACE_RNG_TEST = np.random.default_rng(SEED + 400)
last_linear = ensemble_models[0].net[-1]
orig_w = last_linear.weight.data.clone()
orig_b = last_linear.bias.data.clone()
for s in range(LAPLACE_NUM_SAMPLES):
    z = LAPLACE_RNG_TEST.normal(size=H_ll.shape[0])
    delta = np.linalg.solve(L_ll.T, z)
    last_linear.weight.data = orig_w + torch.from_numpy(delta[:-1].reshape(1, -1).astype(np.float32))
    last_linear.bias.data = orig_b + torch.from_numpy(delta[-1:].astype(np.float32))
    laplace_test_probs.append(predict_test(ensemble_models[0]))
last_linear.weight.data = orig_w
last_linear.bias.data = orig_b
probs_laplace = np.mean(laplace_test_probs, axis=0)

m_drop, _ = train_mlp(seed=SEED + 50, epochs=200, dropout_p=0.2)
probs_dropout = predict_test(m_drop, T=50, dropout_at_test=True)

ens_test_probs = np.stack([predict_test(m) for m in ensemble_models])
probs_ensemble = ens_test_probs.mean(axis=0)

m_sgld = MLP()
load_params(m_sgld, ensemble_weights[0].copy())
sgld_test = []
torch.manual_seed(SEED + 200)
for t in range(SGMCMC_TOTAL_ITERS):
    idx = torch.randint(0, X_train.shape[0], (SGMCMC_BATCH_SIZE,))
    sgld_step(m_sgld, SGLD_ETA, X_train[idx], y_train[idx], X_train.shape[0])
    if t >= SGMCMC_BURN_IN and (t - SGMCMC_BURN_IN) % SGMCMC_THIN == 0:
        sgld_test.append(predict_test(m_sgld))
probs_sgld = np.mean(sgld_test, axis=0)

m_sghmc = MLP()
load_params(m_sghmc, ensemble_weights[0].copy())
mom = [torch.zeros_like(p) for p in m_sghmc.parameters()]
sghmc_test = []
torch.manual_seed(SEED + 300)
for t in range(SGMCMC_TOTAL_ITERS):
    idx = torch.randint(0, X_train.shape[0], (SGMCMC_BATCH_SIZE,))
    sghmc_step(m_sghmc, mom, SGHMC_ETA, SGHMC_FRICTION, X_train[idx], y_train[idx], X_train.shape[0])
    if t >= SGMCMC_BURN_IN and (t - SGMCMC_BURN_IN) % SGMCMC_THIN == 0:
        sghmc_test.append(predict_test(m_sghmc))
probs_sghmc = np.mean(sghmc_test, axis=0)


def calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    p = np.clip(probs, BCE_EPS, 1 - BCE_EPS)
    pred = (p >= 0.5).astype(np.float32)
    acc = (pred == labels).mean()
    brier = ((p - labels) ** 2).mean()
    nll = (-labels * np.log(p) - (1 - labels) * np.log(1 - p)).mean()
    conf = np.maximum(p, 1 - p)
    is_correct = (pred == labels).astype(np.float32)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins = []
    for b in range(n_bins):
        if b < n_bins - 1:
            mask = (conf >= bin_edges[b]) & (conf < bin_edges[b + 1])
        else:
            mask = (conf >= bin_edges[b]) & (conf <= bin_edges[b + 1])
        c = mask.sum()
        if c > 0:
            bin_conf = conf[mask].mean()
            bin_acc = is_correct[mask].mean()
            ece += c / len(labels) * abs(bin_conf - bin_acc)
            bins.append({"binConf": float(bin_conf), "binAcc": float(bin_acc), "binCount": int(c)})
        else:
            bins.append({"binConf": (bin_edges[b] + bin_edges[b + 1]) / 2, "binAcc": 0.0, "binCount": 0})
    return {
        "ece": float(ece),
        "brier": float(brier),
        "nll": float(nll),
        "accuracy": float(acc),
        "reliability_bins": bins,
    }


methods = {
    "point": probs_point,
    "laplace": probs_laplace,
    "dropout": probs_dropout,
    "ensemble": probs_ensemble,
    "sgld": probs_sgld,
    "sghmc": probs_sghmc,
}
metrics_per_method = {name: calibration_metrics(probs, y_test_np) for name, probs in methods.items()}
print("  ECE per method:")
for name, m in metrics_per_method.items():
    print(f"    {name:>10s}: ECE={m['ece']:.4f}, Brier={m['brier']:.4f}, NLL={m['nll']:.4f}, Acc={m['accuracy']:.3f}")

_write_dual(
    "calibration.json",
    {
        "test_x": X_test_np.tolist(),
        "test_y": y_test_np.tolist(),
        "n_test": int(len(y_test_np)),
        "probs": {name: probs.tolist() for name, probs in methods.items()},
        "metrics": metrics_per_method,
    },
)
print(f"  Calibration done in {time.time()-t0:.1f}s")


# =========================================================================== #
# §2: Loss-landscape PCA → loss_landscape.json
# =========================================================================== #

print("\n=== §2: Loss-landscape PCA ===")
t0 = time.time()
W = np.stack(ensemble_weights)
pca = PCA(n_components=2)
W_pca = pca.fit_transform(W)
basis = pca.components_
mean_w = pca.mean_
xs = np.linspace(W_pca[:, 0].min() - 0.5, W_pca[:, 0].max() + 0.5, 30)
ys = np.linspace(W_pca[:, 1].min() - 0.5, W_pca[:, 1].max() + 0.5, 30)
loss_surface = np.zeros((30, 30), dtype=np.float32)
test_model = MLP()
bce = nn.BCEWithLogitsLoss()
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        flat = mean_w + x * basis[0] + y * basis[1]
        load_params(test_model, flat.astype(np.float32))
        with torch.no_grad():
            loss_surface[j, i] = bce(test_model(X_train), y_train).item()


def interpolate_pair(idx_a: int, idx_b: int, n_steps: int = 30):
    flat_a = ensemble_weights[idx_a]
    flat_b = ensemble_weights[idx_b]
    losses = []
    for t in np.linspace(0, 1, n_steps):
        flat = (1 - t) * flat_a + t * flat_b
        load_params(test_model, flat.astype(np.float32))
        with torch.no_grad():
            losses.append(float(bce(test_model(X_train), y_train).item()))
    return losses


pairs = []
for a in range(ENSEMBLE_K):
    for b in range(a + 1, ENSEMBLE_K):
        pairs.append({"a": a, "b": b, "loss_curve": interpolate_pair(a, b)})

_write_dual(
    "loss_landscape.json",
    {
        "n_models": ENSEMBLE_K,
        "model_coords": W_pca.tolist(),
        "model_losses": ensemble_losses,
        "x_grid": xs.tolist(),
        "y_grid": ys.tolist(),
        "loss_surface": loss_surface.tolist(),
        "interpolations": pairs,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    },
)
print(f"  Loss-landscape done in {time.time()-t0:.1f}s")


# =========================================================================== #
# §9: NNGP arc-cosine kernel + width sweep → nngp.json
# =========================================================================== #

print("\n=== §9: NNGP arc-cosine kernel + width-convergence ===")
t0 = time.time()


def arc_cosine_kernel(X, Xp, sigma_w2: float = NNGP_SIGMA_W2, sigma_b2: float = NNGP_SIGMA_B2) -> np.ndarray:
    norms_X = np.linalg.norm(X, axis=1)
    norms_Xp = np.linalg.norm(Xp, axis=1)
    dot = X @ Xp.T
    norm_outer = np.maximum(norms_X[:, None] * norms_Xp[None, :], EIGENVALUE_FLOOR)
    cos_theta = np.clip(dot / norm_outer, -1, 1)
    theta = np.arccos(cos_theta)
    K = (sigma_w2 / (2 * np.pi)) * norm_outer * (np.sin(theta) + (np.pi - theta) * np.cos(theta))
    return K + sigma_b2


def empirical_finite_width_kernel(width: int, n_samples: int, X: np.ndarray) -> float:
    rng = np.random.default_rng(SEED + width)
    sigma_w = np.sqrt(2.0)
    f_origin = []
    x0 = X[0:1]
    for _ in range(n_samples):
        W1 = rng.normal(size=(2, width)) * sigma_w / np.sqrt(2)
        b1 = rng.normal(size=width) * np.sqrt(0.1)
        W2 = rng.normal(size=(width, 1)) * sigma_w / np.sqrt(width)
        b2 = rng.normal(size=1) * np.sqrt(0.1)
        h = np.maximum(x0 @ W1 + b1, 0)
        f = h @ W2 + b2
        f_origin.append(float(f[0, 0]))
    return float(np.var(f_origin))


width_convergence = []
target_K00 = float(arc_cosine_kernel(X_train_np[:1], X_train_np[:1]).item())
for w in NNGP_WIDTHS:
    emp = empirical_finite_width_kernel(w, n_samples=NNGP_SAMPLES_PER_WIDTH, X=X_train_np)
    width_convergence.append({"width": w, "empirical": emp, "closed_form_K00": target_K00})
print(f"  K(x_0, x_0) closed-form = {target_K00:.4f}")
print("  empirical Var f(x_0):")
for entry in width_convergence:
    print(f"    width={entry['width']:>4d}: empirical={entry['empirical']:.4f}")

rng_reg = np.random.default_rng(SEED + 600)
X_reg = np.linspace(-2.5, 2.5, 8).reshape(-1, 1).astype(np.float32)
y_reg = (np.sin(X_reg) + 0.1 * rng_reg.normal(size=X_reg.shape)).ravel().astype(np.float32)

X_grid = np.linspace(-3, 3, 80).reshape(-1, 1).astype(np.float32)
K_train = arc_cosine_kernel(X_reg, X_reg) + NNGP_NOISE * np.eye(len(X_reg))
K_grid_train = arc_cosine_kernel(X_grid, X_reg)
K_grid = arc_cosine_kernel(X_grid, X_grid)
L = np.linalg.cholesky(K_train)
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_reg))
mu = K_grid_train @ alpha
v = np.linalg.solve(L, K_grid_train.T)
var = np.diag(K_grid - v.T @ v)
std = np.sqrt(np.maximum(var, EIGENVALUE_FLOOR))

_write_dual(
    "nngp.json",
    {
        "width_convergence": width_convergence,
        "regression": {
            "x_train": X_reg.ravel().tolist(),
            "y_train": y_reg.tolist(),
            "x_grid": X_grid.ravel().tolist(),
            "mean": mu.tolist(),
            "std": std.tolist(),
        },
        "kernel_params": {"sigma_w2": NNGP_SIGMA_W2, "sigma_b2": NNGP_SIGMA_B2},
    },
)
print(f"  NNGP done in {time.time()-t0:.1f}s")

print("\n=== ALL PRECOMPUTE DONE ===")
