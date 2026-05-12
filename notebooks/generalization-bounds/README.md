# Generalization Bounds — companion notebook

Source of truth for the math, code, and numerical findings of the
[generalization-bounds](https://formalml.com/topics/generalization-bounds) topic
on formalml.com. CPU-only; end-to-end runtime under 60 s on a 2020-era laptop.

## Setup

```bash
cd notebooks/generalization-bounds
uv venv
uv pip install -e .
.venv/bin/python -m jupyter lab
```

## Files

- `01_generalization_bounds.ipynb` — the long-form derivation walk-through.
- `precompute_margin_bound.py` — emits `margin_bound.json` (SVM on two-moons).
- `precompute_vacuousness.py` — emits `vacuousness.json` (binary-MNIST MLP).

Each `precompute_*.py` writes to both `src/data/sampleData/generalization-bounds/`
and `public/sample-data/generalization-bounds/` so the dev server and
production build serve identical JSON.
