"""
precompute_sample_efficiency.py — Sample-efficiency sweep for CNP and Latent NP
on held-out GP tasks (cell 69 of the notebook, Figure 15).

Executes the notebook's code cells 0-69 in order via _notebook_exec.py, so the
CNP and Latent NP are the ones trained in cell 35 (on the module-level `rng`
after cells 3-33 have advanced it) and the Latent NP's sampled latents in
cell 69 draw from torch's global RNG in the same state as in the notebook. An
earlier version retrained both models from a fresh `default_rng(SEED)` and
did not match the notebook's run.

Writes to BOTH:
  - src/data/sampleData/meta-learning/sample_efficiency.json
  - public/sample-data/meta-learning/sample_efficiency.json

Output schema:
{
  "K_ctx": [2, 4, 8, 12, 16, 20],
  "cnpNllMean": [...6 floats],
  "cnpNllSE":   [...6 floats],
  "lnpNllMean": [...6 floats],
  "lnpNllSE":   [...6 floats]
}

Runs in ~10 s on CPU.
"""

import sys

from _notebook_exec import floats, run_notebook_through, write_json

LAST_CELL = 69


def main():
    print(f"Executing notebook cells 0-{LAST_CELL}…", file=sys.stderr)
    ns = run_notebook_through(LAST_CELL)
    payload = {
        "K_ctx": [int(k) for k in ns["K_ctx_values"]],
        "cnpNllMean": floats(ns["cnp_nll_means"]),
        "cnpNllSE": floats(ns["cnp_nll_stds"]),  # cell 69 stores std / sqrt(20)
        "lnpNllMean": floats(ns["lnp_nll_means"]),
        "lnpNllSE": floats(ns["lnp_nll_stds"]),
    }
    for k in ("cnpNllMean", "lnpNllMean"):
        print(f"{k}: " + ", ".join(f"{v:.3f}" for v in payload[k]), file=sys.stderr)
    write_json("sample_efficiency.json", payload)


if __name__ == "__main__":
    main()
