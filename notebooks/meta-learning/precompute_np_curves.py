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
code cells 0-37 in order via _notebook_exec.py and serializes the resulting
arrays.

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


import sys

import numpy as np

from _notebook_exec import floats, run_notebook_through, write_json

LAST_CELL = 37  # held-out comparison cell (Figure 09)


def smooth(a, w=25):
    a = np.asarray(a)
    kernel = np.ones(w) / w
    return np.convolve(a, kernel, mode="valid")


def main():
    print(f"Executing notebook cells 0-{LAST_CELL}…", file=sys.stderr)
    ns = run_notebook_through(LAST_CELL)

    cnp_losses = ns["cnp_losses"]
    lnp_losses = ns["lnp_losses"]
    lnp_samples = np.asarray(ns["lnp_samples"])
    print(f"CNP NLL {cnp_losses[0]:.2f} -> {cnp_losses[-1]:.2f}; "
          f"Latent NP -ELBO {lnp_losses[0]:.2f} -> {lnp_losses[-1]:.2f}",
          file=sys.stderr)

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
    write_json("np_curves.json", payload)


if __name__ == "__main__":
    main()
