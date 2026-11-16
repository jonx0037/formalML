"""
precompute_np_curves.py — Export the CNP and Latent NP training curves +
held-out posterior predictive snapshot from cells 34-37 of the verified
notebook (01_meta_learning.ipynb, which must sit next to this script).

The notebook draws its §4.5 training tasks from the module-level `rng`
seeded in cell 1, which cells 3-33 have already advanced. A fresh
`default_rng(SEED)` therefore trains on a different task stream and
produces different curves (the pre-2026-11 JSON did exactly that: CNP NLL
35.24 -> 58.72 instead of the notebook's 31.92 -> 42.91). To guarantee the
JSON reproduces the notebook's own run, this script executes the notebook's
code cells 1-37 in order (Agg backend, figure saving and display stubbed
out) and serializes the resulting arrays.

Writes to BOTH:
  - src/data/sampleData/meta-learning/np_curves.json
  - public/sample-data/meta-learning/np_curves.json

Output schema:
{
  "epochs": 1000,
  "cnpLoss":  [...1000 floats],
  "lnpLoss":  [...1000 floats],
  "smoothedCnpLoss": [...976 floats],   // 25-pt moving average (matches notebook)
  "smoothedLnpLoss": [...976 floats],
  "smoothedXStart": 12,                  // start index of smoothed series in original grid
  "heldOut": {
    "ell": 1.0,
    "xFull":  [...100 floats],
    "yFull":  [...100 floats],
    "xContext": [...8 floats],
    "yContext": [...8 floats],
    "xDense":   [...200 floats],
    "muGp":     [...200 floats],
    "sdGp":     [...200 floats],
    "muCnp":    [...200 floats],
    "sdCnp":    [...200 floats],
    "muLnp":    [...200 floats],
    "sdLnp":    [...200 floats],
    "lnpSamples": [[...200], [...200], ...20 traces]
  }
}

Runs in ~5-10 s on CPU (cells 1-37 include the §2-3 MAML/FOMAML/Reptile runs).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
NOTEBOOK = THIS_DIR / "01_meta_learning.ipynb"
LAST_CELL = 37  # held-out comparison cell (Figure 09)
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "meta-learning",
    REPO_ROOT / "public" / "sample-data" / "meta-learning",
]


def smooth(a, w=25):
    a = np.asarray(a)
    kernel = np.ones(w) / w
    return np.convolve(a, kernel, mode="valid")


def run_notebook_through(last_cell):
    """Execute code cells 0..last_cell of the notebook; return the namespace."""
    nb = json.loads(NOTEBOOK.read_text())
    plt.savefig = lambda *a, **k: None  # never overwrite the notebook's PNGs
    plt.show = lambda *a, **k: None
    ns = {"__name__": "__notebook__"}
    for i, cell in enumerate(nb["cells"][: last_cell + 1]):
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), f"<cell {i}>", "exec"), ns)
        plt.close("all")
    return ns


def main():
    if not NOTEBOOK.exists():
        sys.exit(f"notebook not found: {NOTEBOOK}")
    print(f"Executing {NOTEBOOK.name} cells 0-{LAST_CELL}…", file=sys.stderr)
    ns = run_notebook_through(LAST_CELL)

    cnp_losses = ns["cnp_losses"]
    lnp_losses = ns["lnp_losses"]
    lnp_samples = np.asarray(ns["lnp_samples"])
    print(f"CNP NLL {cnp_losses[0]:.2f} -> {cnp_losses[-1]:.2f}; "
          f"Latent NP -ELBO {lnp_losses[0]:.2f} -> {lnp_losses[-1]:.2f}",
          file=sys.stderr)

    def floats(a):
        return [float(x) for x in np.asarray(a).ravel().tolist()]

    payload = {
        "epochs": int(ns["N_EPOCHS_NP"]),
        "cnpLoss": floats(cnp_losses),
        "lnpLoss": floats(lnp_losses),
        "smoothedCnpLoss": floats(smooth(cnp_losses)),
        "smoothedLnpLoss": floats(smooth(lnp_losses)),
        "smoothedXStart": 12,
        "heldOut": {
            "ell": float(ns["ell_true"]),
            "xFull": floats(ns["x_full"]),
            "yFull": floats(ns["y_full"]),
            "xContext": floats(ns["xc"]),
            "yContext": floats(ns["yc"]),
            "xDense": floats(ns["x_dense"]),
            "muGp": floats(ns["mu_gp"]),
            "sdGp": floats(ns["sd_gp"]),
            "muCnp": floats(ns["mu_cnp"]),
            "sdCnp": floats(ns["sd_cnp"]),
            "muLnp": floats(ns["mu_lnp"]),
            "sdLnp": floats(ns["sd_lnp"]),
            "lnpSamples": [floats(row) for row in lnp_samples[:5]],
        },
    }

    for outdir in OUT_DIRS:
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "np_curves.json", "w") as f:
            json.dump(payload, f, allow_nan=False)
        print(f"wrote {outdir / 'np_curves.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
