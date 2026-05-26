"""Extract all PNG figures from the GMM notebook into public/images/topics/generalized-method-of-moments/.

One-pass cell.outputs[].data['image/png'] loop per CLAUDE.md
("Notebook figure extraction is a one-pass, no-execution operation").

Run from repo root: python3 notebooks/generalized-method-of-moments/_extract_figures.py
"""
import base64
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks/generalized-method-of-moments/01_generalized_method_of_moments.ipynb"
OUT_DIR = REPO_ROOT / "public/images/topics/generalized-method-of-moments"
FIG_DIR = REPO_ROOT / "notebooks/generalized-method-of-moments/figures"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    # Cell-index → output filename map. Derived from the Phase-1 cell survey.
    # Each entry pins (cell_idx, image_idx_within_cell) → descriptive name.
    name_for_index: dict[tuple[int, int], str] = {
        (5, 0): "02_gamma_mom_vs_mle.png",
        (7, 0): "02_overidentification.png",
        (11, 0): "03_criterion_surface.png",
        (12, 0): "03_criterion_surface_weightings.png",
        (16, 0): "04_consistency.png",
        (19, 0): "05_loewner_ellipses.png",
        (20, 0): "05_qq_normality.png",
        (23, 0): "06_hansen_bound.png",
        (27, 0): "07_two_step_surfaces.png",
        (29, 0): "07_finite_sample_bias.png",
        (32, 0): "08_J_under_H0.png",
        (33, 0): "08_J_power_curve.png",
        (36, 0): "09_linear_iv_sampling.png",
        (38, 0): "09_weak_instruments.png",
        (41, 0): "10_bias_hierarchy.png",
        (45, 0): "11_quasi_mle_coverage.png",
        (48, 0): "12_dml_partial_linear.png",
        (52, 0): "13_lte_quasi_posterior.png",
    }

    written: list[tuple[int, int, str]] = []
    for cell_idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        img_idx = 0
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if "image/png" not in data:
                continue
            key = (cell_idx, img_idx)
            fname = name_for_index.get(key)
            if fname is None:
                fname = f"_unmapped_cell{cell_idx}_img{img_idx}.png"
            (OUT_DIR / fname).write_bytes(base64.b64decode(data["image/png"]))
            written.append((cell_idx, img_idx, fname))
            img_idx += 1

    # Backup: any figure that lives on disk in figures/ but didn't end up in
    # the inline outputs gets copied over. The notebook savefig'd to figures/
    # with the same descriptive names used in the brief.
    disk_fallbacks = {
        "02_gamma_mom_vs_mle.png": "02_gamma_mom_vs_mle.png",
        "02_overidentification.png": "02_overidentification.png",
        "03_criterion_surface.png": "03_criterion_surface.png",
        "03_criterion_surface_weightings.png": "03_criterion_surface_weightings.png",
        "04_consistency.png": "04_consistency.png",
        "05_loewner_ellipses.png": "05_loewner_ellipses.png",
        "05_qq_normality.png": "05_qq_normality.png",
        "06_hansen_bound.png": "06_hansen_bound.png",
        "07_two_step_surfaces.png": "07_two_step_surfaces.png",
        "07_finite_sample_bias.png": "07_finite_sample_bias.png",
        "08_J_under_H0.png": "08_J_under_H0.png",
        "08_J_power_curve.png": "08_J_power_curve.png",
        "09_linear_iv_sampling.png": "09_linear_iv_sampling.png",
        "09_weak_instruments.png": "09_weak_instruments.png",
        "10_bias_hierarchy.png": "10_bias_hierarchy.png",
        "11_quasi_mle_coverage.png": "11_quasi_mle_coverage.png",
        "12_dml_partial_linear.png": "12_dml_partial_linear.png",
        "13_lte_quasi_posterior.png": "13_lte_quasi_posterior.png",
    }
    if FIG_DIR.exists():
        for src_name, dst_name in disk_fallbacks.items():
            src = FIG_DIR / src_name
            dst = OUT_DIR / dst_name
            if src.exists() and not dst.exists():
                dst.write_bytes(src.read_bytes())
                written.append((-1, -1, dst_name + " (from figures/)"))

    for cell_idx, img_idx, fname in sorted(written, key=lambda x: x[2]):
        print(f"  cell={cell_idx:>2} img={img_idx:>2}  {fname}")
    print(f"\nWrote {len(written)} figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
