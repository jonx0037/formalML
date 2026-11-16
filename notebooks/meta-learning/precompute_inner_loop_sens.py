"""
precompute_inner_loop_sens.py — Inner-loop step-count sensitivity sweep for
MAML/FOMAML/Reptile on held-out sinusoidal tasks (cell 71 of the notebook,
Figure 16).

Executes the notebook's code cells 0-71 in order via _notebook_exec.py, so the
three meta-trained θ₀s come from the notebook's own §2-3 runs (which draw from
the module-level `rng` in sequence) and the 20 held-out tasks are cell 71's
single `sens_rng = default_rng(SEED + 700)` draw shared by all three methods.
An earlier version retrained each method from a fresh `default_rng(SEED)` and
gave each method its own held-out tasks, so its curves (FOMAML and Reptile
starting near 4.2) disagreed with Figure 16.

Writes to BOTH:
  - src/data/sampleData/meta-learning/inner_loop_sens.json
  - public/sample-data/meta-learning/inner_loop_sens.json

Output schema:
{
  "N_test": [0, 1, 2, 3, 5, 8, 12],
  "MAML":    [...7 floats],   // mean test MSE
  "FOMAML":  [...7 floats],
  "Reptile": [...7 floats]
}

Runs in ~10 s on CPU.
"""

import sys

from _notebook_exec import floats, run_notebook_through, write_json

LAST_CELL = 71


def main():
    print(f"Executing notebook cells 0-{LAST_CELL}…", file=sys.stderr)
    ns = run_notebook_through(LAST_CELL)
    results = ns["results"]
    payload = {"N_test": [int(n) for n in ns["N_test_values"]]}
    for method in ("MAML", "FOMAML", "Reptile"):
        payload[method] = floats(results[method])
        print(f"{method}: " + ", ".join(f"{v:.3f}" for v in payload[method]),
              file=sys.stderr)
    write_json("inner_loop_sens.json", payload)


if __name__ == "__main__":
    main()
