#!/usr/bin/env python3
"""
Extract figure PNGs from the verified notebooks/double-descent/01_double_descent.ipynb
into public/images/topics/double-descent/.

The notebook is read-only (verified, immutable). This script reads the embedded
image/png base64 in each cell's outputs and writes the PNG bytes to disk. No
notebook re-execution.

Mapping (brief Figure N → notebook cell index → output filename):

  Figure  1  →  cell [005]  →  01_empirical_double_descent.png        (§1.2)
  Figure  2  →  cell [012]  →  02_classical_u_curve.png               (§2.1)
  Figure  3  →  cell [021]  →  03_interpolation_threshold_synced.png  (§3.4)
  Figure  4  →  cell [027]  →  04_ridgeless_vs_ridge.png              (§4.4)
  Figure  5  →  cell [033]  →  05_marchenko_pastur_densities.png      (§5.4)
  Figure  6  →  cell [039]  →  06_hastie_analytic_vs_mc.png           (§6.4)
  Figure  7  →  cell [040]  →  07_hastie_decomposition.png            (§6.4)
  Figure  8  →  cell [044]  →  08_modelwise_vs_samplewise.png         (§7.3)
  Figure  9  →  cell [048]  →  09_random_features_three_activations.png (§8.1)
  Figure 10  →  cell [057]  →  10_gd_trajectory_4panel.png            (§9.3)
  Figure 11  →  cell [062]  →  11_deep_double_descent_width.png       (§10.1)
  Figure 12  →  cell [068]  →  12_eigenvalue_spectra_two_panel.png    (§11.1)
"""

import base64
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks/double-descent/01_double_descent.ipynb"
OUT_DIR = REPO_ROOT / "public/images/topics/double-descent"

MAP = [
    (5, "01_empirical_double_descent.png"),
    (12, "02_classical_u_curve.png"),
    (21, "03_interpolation_threshold_synced.png"),
    (27, "04_ridgeless_vs_ridge.png"),
    (33, "05_marchenko_pastur_densities.png"),
    (39, "06_hastie_analytic_vs_mc.png"),
    (40, "07_hastie_decomposition.png"),
    (44, "08_modelwise_vs_samplewise.png"),
    (48, "09_random_features_three_activations.png"),
    (57, "10_gd_trajectory_4panel.png"),
    (62, "11_deep_double_descent_width.png"),
    (68, "12_eigenvalue_spectra_two_panel.png"),
]


def extract_png(cell):
    """Return PNG bytes from the first image/png output of `cell`, or None."""
    for out in cell.get("outputs", []):
        data = out.get("data", {})
        if "image/png" in data:
            payload = data["image/png"]
            if isinstance(payload, list):
                payload = "".join(payload)
            return base64.b64decode(payload)
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK.open() as f:
        nb = json.load(f)
    cells = nb["cells"]

    written = []
    missing = []
    for cell_idx, filename in MAP:
        cell = cells[cell_idx]
        png_bytes = extract_png(cell)
        if png_bytes is None:
            missing.append((cell_idx, filename))
            continue
        out_path = OUT_DIR / filename
        out_path.write_bytes(png_bytes)
        written.append((cell_idx, filename, len(png_bytes)))

    print(f"Wrote {len(written)} PNGs to {OUT_DIR}")
    for cell_idx, filename, n in written:
        print(f"  cell[{cell_idx:03d}] → {filename}  ({n:,} bytes)")
    if missing:
        print(f"\nMISSING ({len(missing)}):")
        for cell_idx, filename in missing:
            print(f"  cell[{cell_idx:03d}] → {filename} (no image/png in outputs)")


if __name__ == "__main__":
    main()
