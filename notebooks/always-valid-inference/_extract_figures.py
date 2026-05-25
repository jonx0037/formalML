"""Extract all PNG figures from the AVI notebook into public/images/topics/always-valid-inference/.

One-pass cell.outputs[].data['image/png'] loop per CLAUDE.md
("Notebook figure extraction is a one-pass, no-execution operation").

Run from repo root: python3 notebooks/always-valid-inference/_extract_figures.py
"""
import base64
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks/always-valid-inference/01_always_valid_inference.ipynb"
OUT_DIR = REPO_ROOT / "public/images/topics/always-valid-inference"
FIG_DIR = REPO_ROOT / "notebooks/always-valid-inference/figures"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb = json.loads(NOTEBOOK_PATH.read_text())

    # Walk the notebook cells, dump every PNG output to OUT_DIR with a
    # numbered, descriptive filename. The order matches the cell map from
    # planning. Some cells emit multiple images; index within cell tracks that.
    written: list[tuple[int, int, str]] = []
    name_for_index: dict[tuple[int, int], str] = {
        (3, 0): "01_peeking_failure.png",
        (4, 0): "02_lil_preview.png",
        (6, 0): "03_supermartingale_paths.png",
        (7, 0): "04_villes_typeI_power.png",
        (10, 0): "05_sprt_trajectories.png",
        (11, 0): "06_sprt_efficiency.png",
        (13, 0): "07_mixture_envelope.png",
        (14, 0): "08_mixture_fixes_peeking.png",
        (16, 0): "09_boundary_atlas.png",
        (17, 0): "10_empirical_coverage.png",
        (19, 0): "11_betting_wealth.png",
        (20, 0): "12_betting_cs_comparison.png",
        (23, 0): "13_eprocess_catalogue.png",
        (23, 1): "14_vovk_wang_calibrators.png",
        (25, 0): "15_gro_growth_race.png",
        (25, 1): "16_universal_portfolio_regret.png",
        (27, 0): "17_merging_strength.png",
        (29, 0): "18_ab_dashboard.png",
        (29, 1): "19_offpolicy_cs.png",
        (31, 0): "20_runtime_benchmark.png",
    }

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
                # Unmapped image — give it a placeholder name we can rename if useful
                fname = f"_unmapped_cell{cell_idx}_img{img_idx}.png"
            (OUT_DIR / fname).write_bytes(base64.b64decode(data["image/png"]))
            written.append((cell_idx, img_idx, fname))
            img_idx += 1

    # Backup: any figure that lives on disk in figures/ but didn't end up in
    # the inline outputs gets copied over with a sensible name.
    disk_fallbacks = {
        "avi_villes_typeI_power.png": "04_villes_typeI_power.png",
        "avi_vovk_wang_calibrators.png": "14_vovk_wang_calibrators.png",
        "avi_universal_portfolio_regret.png": "16_universal_portfolio_regret.png",
        "avi_offpolicy_cs.png": "19_offpolicy_cs.png",
        "avi_universal_coverage.png": "10_empirical_coverage.png",
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
