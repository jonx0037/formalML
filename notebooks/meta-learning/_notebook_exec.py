"""
_notebook_exec.py — Shared helper for the meta-learning precompute scripts.

The notebook (01_meta_learning.ipynb, which must sit next to this file) seeds
one module-level `rng = np.random.default_rng(SEED)` in cell 1 and lets every
later cell draw from it in order; torch's global RNG is likewise consumed
cell by cell. A precompute script that starts its own `default_rng(SEED)`
therefore trains on a different task stream than the notebook and produces
different numbers from the figures and prose. To reproduce the notebook's run
exactly, the precompute scripts execute the notebook's own code cells in order
(Agg backend, figure saving and display stubbed out) and read the arrays they
need from the resulting namespace.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
NOTEBOOK = THIS_DIR / "01_meta_learning.ipynb"
OUT_DIRS = [
    REPO_ROOT / "src" / "data" / "sampleData" / "meta-learning",
    REPO_ROOT / "public" / "sample-data" / "meta-learning",
]


def code_cells():
    """Return {cell_index: source} for every code cell of the notebook."""
    if not NOTEBOOK.exists():
        sys.exit(f"notebook not found: {NOTEBOOK}")
    nb = json.loads(NOTEBOOK.read_text())
    return {i: "".join(c["source"]) for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code"}


def run_cell(ns, index, source):
    """Execute one cell's source in namespace `ns` without writing figures."""
    plt.savefig = lambda *a, **k: None  # never overwrite the notebook's PNGs
    plt.show = lambda *a, **k: None
    exec(compile(source, f"<cell {index}>", "exec"), ns)
    plt.close("all")


def run_notebook_through(last_cell, ns=None, first_cell=0):
    """Execute code cells first_cell..last_cell in order; return the namespace."""
    ns = {"__name__": "__notebook__"} if ns is None else ns
    for i, src in sorted(code_cells().items()):
        if first_cell <= i <= last_cell:
            run_cell(ns, i, src)
    return ns


def floats(a):
    import numpy as np
    return [float(x) for x in np.asarray(a, dtype=float).ravel().tolist()]


def write_json(name, payload):
    for outdir in OUT_DIRS:
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / name, "w") as f:
            json.dump(payload, f, allow_nan=False)
        print(f"wrote {outdir / name}", file=sys.stderr)
