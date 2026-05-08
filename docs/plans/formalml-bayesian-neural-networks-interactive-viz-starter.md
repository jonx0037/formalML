# Bayesian Neural Networks — Interactive Viz Layer (v2 Starter Prompt)

This is the v2 follow-up to the Bayesian Neural Networks topic. The v1+v2-substrate PR shipped the topic content, the shared module (`bayesian-ml.ts` extended with seven BNN primitives + verification harness), the curriculum graph, the cross-references, the figures, the precompute script (`precompute_viz_data.py` emitting five JSON fixtures), the interactive substrate (`BNNInteractiveFigure.tsx`), and **2 of 9 viz components promoted to fully interactive React+D3** — `BNNCalibrationComparisonViz` (§8) and `NNGPSidebarViz` (§9). The remaining 7 still render via `BNNStaticFigure`. **This v2 task is to promote those 7 wrappers** to fully interactive React+D3 components matching the per-section interactive-control specs in the handoff brief's "Viz design intent §X" subsections.

> **Why this is its own PR.** The remaining-viz scope is multi-hours of focused work; splitting it from the v1 content PR kept each diff reviewable and let the topic ship to readers immediately with content + 2 showcase interactive components while this v2 lands the rest of the interactive layer.

---

## Pre-flight (do these first)

```bash
cd /Users/jonathanrocha/Developer/Sites/formalML

# 1. Verify v1 state is committed and pushed.
git log --oneline feat/bayesian-neural-networks ^main | head -10
#    If empty / uncommitted, the v1 PR hasn't landed yet — pause and confirm with Jonathan
#    which branch this v2 work should branch off (probably feat/bayesian-neural-networks
#    if v1 is still in review, or main if v1 has merged).

# 2. Create the v2 feature branch.
git checkout feat/bayesian-neural-networks   # or `main` if v1 has merged
git pull --ff-only
git checkout -b feat/bayesian-neural-networks-interactive-viz

# 3. Sanity-check the v1 deliverables you'll be extending.
ls src/content/topics/bayesian-neural-networks.mdx                       # the topic, do not rewrite
ls src/components/viz/BNN*.tsx src/components/viz/{LossLandscape,Laplace,MCDropout,DeepEnsemble,SGLD,SGHMC,NNGP}*.tsx
ls src/components/viz/shared/bayesian-ml.ts                          # shared module, do extend
ls notebooks/bayesian-neural-networks/.venv/bin/python                   # if absent, run `cd notebooks/bayesian-neural-networks && uv sync`
ls notebooks/bayesian-neural-networks/figures/*.png | wc -l              # expect 9
ls public/images/topics/bayesian-neural-networks/*.png | wc -l           # expect 9
```

If the `.venv` is absent, create it via `uv sync` from the notebook directory before running any precompute script.

---

## Required reading (in this order)

1. `CLAUDE.md` (repo root) — project conventions, especially the **"Default to in-browser TS for viz computation; reserve Python precompute for genuinely non-conjugate cases"** rule and the **sample-data dual-location** convention (precompute scripts must dual-write to `src/data/sampleData/<slug>/` AND `public/sample-data/<slug>/` from inside `main()`).
2. `docs/plans/formalml-bayesian-neural-networks-handoff-brief.md` — the implementation spec. Specifically:
   - §4 — re-read the **"Viz design intent §X"** subsection at the end of each of §§1–9. The "Interactive controls (v1 deliverable)" bullet block in each is the per-component spec the v2 components must implement.
   - §5 — the shared-module signatures the components call into.
   - §10 — open questions (Q3, Q4 are about JSON precompute payloads for §3/§6/§7).
3. `notebooks/bayesian-neural-networks/01_bayesian_neural_networks.ipynb` — the source of truth for math and parameter defaults. Don't modify; mirror its training recipes in the precompute scripts.
4. `docs/plans/formalml-handoff-reference.md` — cross-cutting conventions (§4 documents the `useD3` + `useResizeObserver` + CSS-custom-property + mobile-breakpoint pattern every viz component must follow).
5. **Pattern references** — read these before writing your first component:
   - `src/components/viz/shared/bayesian-ml-stacking.ts` (32 KB) — peer-topic precedent for substantial in-browser numerics in a Bayesian-track shared module.
   - `src/components/viz/shared/gaussian-processes.ts` (43 KB) — the Cholesky / `mulberry32` / `gaussianSampler` / `matVec` primitives you'll import.
   - `src/components/viz/StackingSimplexExplorer.tsx` (~10 KB), `src/components/viz/CAVITrajectoryExplorer.tsx` (~16 KB) — recent T5 viz components, similar in scope and complexity to what you're building.
   - `notebooks/probabilistic-programming/precompute_neals_funnel.py` — canonical precompute-script structure (paths/seed → `_to_jsonable`/`_round_floats` helpers → fit functions → `main()` writing to BOTH `OUT_DIRS`).

If any required file is missing, stop and report which one.

---

## What's already in place (do not rebuild)

- **Topic content** at `src/content/topics/bayesian-neural-networks.mdx` — 11,882 words, 11 H2 sections, 46 TheoremBlocks, 24 references. Math, prose, and frontmatter are locked. Don't rewrite.
- **Shared module** at `src/components/viz/shared/bayesian-ml.ts`:
  - Color palette `paletteBNN` (6 method colors).
  - Types: `MLPArchSpec`, `TrainingSpec`, `TrainingData`, `TrainedMLP`, `DeepEnsembleResult`, `LaplaceResult`, `MCDropoutResult`, `SGMCMCSpec`, `SGMCMCResult`, `CalibrationMetrics`, `ReliabilityBin`.
  - Implemented primitives: `bnnCalibrationDiagnostic` (ECE/Brier/NLL + reliability bins), `nngpArcCosineKernel` (Cho-Saul recursion).
  - Stubs that throw `BNNInBrowserNotImplementedError`: `deepEnsembleTraining`, `laplaceApproxBNN`, `mcDropoutInference`, `sgMCMCBNNTraining` — these are where you fill in the in-browser implementations OR replace the throw with a JSON-loader call.
  - Loaders: `loadDeepEnsemblePayload`, `loadLaplacePayload`, `loadMCDropoutPayload`, `loadSGMCMCPayload`, `loadCalibrationComparison` — already wired to fetch + parse the precomputed payload schema you'll emit from the Python precompute scripts.
  - Verification harness: `src/components/viz/shared/__tests__/verify-bayesian-neural-networks.ts` — when you add new primitives, extend `verifyBNNPrimitives()` with new closed-form tests and run `pnpm exec tsx src/components/viz/shared/__tests__/verify-bayesian-neural-networks.ts`.
- **Static-figure wrappers** at `src/components/viz/{BNNPredictiveMotivation,LossLandscapeModes,LaplaceBNN,MCDropoutBNN,DeepEnsemble,SGLDBNN,SGHMCBNN,BNNCalibrationComparison,NNGPSidebar}Viz.tsx` (each ~12 lines). These wrap `BNNStaticFigure`. **You will replace these wrappers, file by file, with full interactive components**, keeping the same default-export name so the MDX imports don't change.
- **`BNNStaticFigure.tsx`** — the v1 layout component. Keep it; useful as the no-JS / pre-hydration fallback inside each interactive component (see "Static-fallback pattern" below).
- **9 figures** at `public/images/topics/bayesian-neural-networks/0[1-9]_*.png` — used as the static fallback inside each interactive component.
- **MDX** uses each viz with `<BNNXxxViz client:visible />`. The MDX doesn't need to change for v2 — the interactive components hydrate inside the same import sites.

---

## The build sequence (priority-ordered)

Build in this order — each later component leans on infrastructure (helpers in the shared module, precompute payload conventions) added by the earlier ones.

### Stage 1: closed-form interactive (no precompute, no MLP training)

**§9 — `NNGPSidebarViz.tsx`.** Brief §9 viz design intent. Use `nngpArcCosineKernel` from the shared module.
- Two panels (a) NNGP-prior convergence and (b) NNGP regression posterior.
- Controls: width slider $h \in [2, 1024]$ (log scale), activation dropdown (ReLU only in v2 — defer erf/tanh per shared-module decision below), training-set-size slider $n \in [2, 20]$.
- For finite-width samples: sample MLP weights $w_1 \sim \mathcal{N}(0, \sigma_w^2/d)$, $b_1 \sim \mathcal{N}(0, \sigma_b^2)$, etc. and forward-pass on the 1D grid. K=5 samples per width.
- For closed-form NNGP samples: build kernel matrix via `nngpArcCosineKernel`, Cholesky-factor (`choleskyFactor` from `gaussian-processes.ts`), draw $z \sim \mathcal{N}(0, I)$, output $L z$.
- For panel (b) GP regression: standard conjugate Gaussian conditional on $(X_{\text{train}}, y_{\text{train}})$ — reuse `gpPredict` from `gaussian-processes.ts`.
- Estimated: ~400 lines TSX. **Pause and report after this is wired in and rendering correctly.**

**Decision needed:** the brief lists `erf` and `tanh` activations in the dropdown. The closed-form recursion for those (Williams 1998, Cho-Saul 2009) is implementable but adds ~100 lines to `bayesian-ml.ts`. Recommendation: **ship ReLU-only in v2 and document erf/tanh as v3** — the ReLU case alone demonstrates the convergence phenomenon and matches the most-cited NNGP literature. Confirm with Jonathan before deciding.

### Stage 2: in-browser MLP training (no Python precompute)

These three viz components share an in-browser small-MLP training loop. **First add helpers to `bayesian-ml.ts`**:

```typescript
// Pure-TS Adam + cross-entropy on a 3-hidden-layer × 32-unit ReLU MLP.
// p ≈ 2241 parameters; n=300 Two Moons; 200 epochs. ~1–2 s per network on a
// 2020-era laptop, so K=5 ensembles complete in 5–10 s on first hydration —
// acceptable per `client:visible`.
export function makeMoonsData(n: number, noise: number, seed: number): TrainingData;
export function trainMLP(arch: MLPArchSpec, training: TrainingSpec, data: TrainingData): TrainedMLP;
export function forwardPassMLP(model: TrainedMLP, X: Float32Array, n: number): Float32Array;
```

Add closed-form-verifiable tests for these in `verifyBNNPrimitives()` (e.g., training loss decreases monotonically across an early-stopping window; trained accuracy on Two Moons is ≥ 95% at noise=0.20).

Then build:

**§1 — `BNNPredictiveMotivationViz.tsx`.** Brief §1 viz design intent. Pre-train pool of ~10 MLPs at default noise on first hydration (~5 s blocking). Slider interactions (K, noise) recompute predictions on the precomputed model pool; noise-slider re-trains and uses the loading-state JSX swap pattern from CLAUDE.md.

**§4 — `MCDropoutBNNViz.tsx`.** Brief §4 viz design intent. Train one dropout MLP at default rate on first hydration. Dropout-rate slider triggers a re-train (debounced ~500 ms). T-slider and dropout-on-test toggle reuse the cached model.

**§5 — `DeepEnsembleViz.tsx`.** Brief §5 viz design intent. Pre-train pool of 20 MLPs on first hydration (~10 s — share with §2's pool if §2's component has rendered first). K-slider subsets the pool; member-highlight click re-renders panel (a)'s heatmap as the highlighted member's predictive.

Estimated: ~1,500 lines TSX across the three components, plus ~300 lines added to `bayesian-ml.ts` for the training helpers.

### Stage 3: in-browser MLP training, heavier compute

**§2 — `LossLandscapeModesViz.tsx`.** Brief §2 viz design intent. Pre-train 30 MLPs on first hydration (~15 s — the heaviest in-browser hydration cost in the topic; cache aggressively). PCA via `numeric.svd`-style closed-form. Click-to-pick-modes recomputes the loss-along-interpolation profile in <100 ms via cached forward passes.

Estimated: ~500 lines TSX.

### Stage 4: Python precompute + JSON loaders

**§3 — `LaplaceBNNViz.tsx`.** Brief §3 viz design intent. The full Hessian for $p = 2241$ is too slow in-browser (~1 s blocking, per brief §10 Q3). Write `notebooks/bayesian-neural-networks/precompute_laplace.py` mirroring the notebook's §3 cell:

- Train MAP, compute `H_data` via `torch.autograd.functional.hessian`, save to JSON: `{wMap: [...], hessianCholesky: [...], conditionNumber: ..., gridProbs: [[]] for S=100}` per the `LaplaceResult` schema in `bayesian-ml.ts`.
- Run for the default $\tau^2 = 10^4$, plus precomputed payloads for the prior-scale-slider tick values (e.g., $\tau^2 \in \{10^2, 10^3, 10^4, 10^5, 10^6\}$). Five payloads × ~5 MB each = ~25 MB total. If this is too large, reduce the tick count.
- Dual-write to `src/data/sampleData/bayesian-neural-networks/laplace_tau2_{value}.json` AND `public/sample-data/bayesian-neural-networks/laplace_tau2_{value}.json`.
- Curvature-reduction dropdown variants: emit separate payloads for `last-layer` and `diagonal-fisher` reductions. The reductions are cheap in Python; emit all three for the default $\tau^2$.

The viz component then loads the right payload based on slider state. The shared module's `loadLaplacePayload` already exists; you may need to extend it to accept a parameter (τ²) or mint per-tick paths. ~500 lines TSX + ~150 lines Python.

**§6 — `SGLDBNNViz.tsx`.** Brief §6 viz design intent. Per Q4: pre-run SGLD chains for a discrete grid of $(\eta, b, \text{schedule})$ values (e.g., 5 × 4 × 2 = 40 chains). Each chain emits per-grid-point mean+std (~640 KB) + the trace-plot data + autocorrelation. Total payload: ~25 MB across 40 chains. Acceptable.

Write `notebooks/bayesian-neural-networks/precompute_sgld.py`. Dual-write per CLAUDE.md. The viz component switches between cached chains in <50 ms.

~500 lines TSX + ~200 lines Python.

**§7 — `SGHMCBNNViz.tsx`.** Brief §7 viz design intent. Same pattern as §6. Write `notebooks/bayesian-neural-networks/precompute_sghmc.py`. Grid is $(\eta, c)$ with method-overlay being just a UI toggle that compares to the §6 SGLD payload. ~500 lines TSX + ~200 lines Python.

**§8 — `BNNCalibrationComparisonViz.tsx`.** Brief §8 viz design intent. Per Q5: lock the test-set seed (`SEED=42`, `random_state=SEED+1000`, $N_{\text{test}} = 500$). Write `notebooks/bayesian-neural-networks/precompute_calibration.py` that emits a single JSON: `{testProbs: {point: [...], laplace: [...], dropout: [...], ensemble: [...], sgld: [...], sghmc: [...]}, testLabels: [...]}`. ~12 KB total. The shared module's `loadCalibrationComparison` already exists.

In-browser: method-toggle checkboxes filter the cached metrics map; temperature-scaling toggle fits a per-method scalar T via 1D L-BFGS on a held-out 200-of-500 validation slice and recomputes the metrics; cold-posterior slider applies temperature to the cached posterior samples for Laplace/SG-MCMC only (per Q6 — the rec is to ship at least Laplace's interactive cold-posterior since the recompute is a single $\tau^2 \to \tau^2/T$ rescale).

~500 lines TSX + ~100 lines Python.

---

## Static-fallback pattern (every interactive component)

Every interactive component should have this structure:

```tsx
// XViz.tsx
import { useEffect, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import BNNStaticFigure from './BNNStaticFigure';
// import primitives, types, loaders from './shared/bayesian-ml'

export default function XViz() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [hydrated, setHydrated] = useState(false);
  const [data, setData] = useState(/* lazy-precomputed state */);

  useEffect(() => {
    // First-hydration cost: train models, fetch JSON, etc.
    // Set `hydrated = true` once ready.
  }, []);

  // Pre-hydration: show the static PNG fallback (server-rendered, no flash).
  if (!hydrated) {
    return <BNNStaticFigure figurePath="/images/topics/bayesian-neural-networks/0X_*.png"
                            alt="..." caption="..." ariaLabel="..." />;
  }

  // Hydrated: render the D3-driven interactive viz.
  return (
    <div ref={ref}>
      {/* sliders, dropdowns, toggles */}
      <svg ref={svgRef /* from useD3 */} width={width} height={...} />
    </div>
  );
}
```

This keeps the no-JS / pre-hydration UX identical to v1 (the static figure renders immediately) and only swaps to interactive after the component has hydrated and any first-hydration compute has completed.

---

## CSS / styling rules (all viz)

- Use CSS custom properties for colors: `var(--color-text)`, `var(--color-surface)`, `var(--color-border)`, `var(--color-accent)`. **Never hardcode hex colors that break dark mode.** Reference `paletteBNN` from the shared module for method-specific colors (those tokens are designed to work in both light and dark mode).
- Mobile breakpoint: 640 px. Below that, sliders should stack vertically; multi-panel layouts should reflow to single-column.
- Labels and tick text use `font-family: var(--font-mono)` for axis ticks, `var(--font-sans)` for slider labels and captions.
- Don't use `.attr("style", ...)` on D3 selections — use `.style()` (per CLAUDE.md).
- All viz must respond to viewport resize via `useResizeObserver`. Don't hardcode SVG widths.

---

## Workflow rules

Follow the brief's build-order step-by-step. Pause and report progress before proceeding to the next stage at:

- After Stage 1 (NNGPSidebarViz). Report: in-browser sample timings, kernel-matrix Cholesky timing, screenshot of the rendered component.
- After helpers added to `bayesian-ml.ts` for in-browser MLP training. Report: extended verification table.
- After each Stage 2 component (§1, §4, §5). Report: training-pool hydration time, slider re-render time.
- After Stage 3 (§2). Report: 30-MLP pool hydration time, click-to-pick re-render time.
- Before each Python precompute script (Stage 4). Report: planned grid coverage, expected payload size.
- After each Stage 4 component is wired. Report: payload size, loader timing, slider re-render time.
- After full build verification (`pnpm build`).

---

## Do NOT

- Rewrite topic content in `src/content/topics/bayesian-neural-networks.mdx`. The math, prose, and frontmatter are locked. The MDX imports the same component names as v1 — only the component implementations change.
- Re-run, modify, or regenerate the notebook. It is read-only. Mirror its training recipes in the precompute scripts.
- Change the `bayesian-ml.ts` API surface that v1 ships (interfaces, function signatures, color palette names). You can ADD primitives and ADD optional fields to interfaces, but don't break what v1 exports.
- Add tests, lint rules, ESLint configuration, or CI workflow changes that aren't strictly needed for this work.
- Build interactive D3 viz that hardcode colors that break dark mode. Use CSS custom properties.
- Skip the `useD3` + `useResizeObserver` pattern. Every viz must use both.
- Use npm or generate `package-lock.json` (per CLAUDE.md — pnpm only).
- Skip the static-fallback pattern. Pre-hydration must render the PNG; D3 swaps in after hydration.
- Stash untracked files (`git stash -u` is forbidden per memory). The brief and notebook directory are user-owned untracked content; don't move them.

---

## Stop and ask before continuing if

- Anything in the brief conflicts with `CLAUDE.md` or with what you find in the notebook.
- A required file from the "Required reading" list is missing.
- A precompute payload exceeds the brief's budget (e.g., §6 SGLD payload > 30 MB).
- A first-hydration cost exceeds 30 s (the brief budgets 5–15 s for individual viz; longer than 30 s degrades UX even with `client:visible`).
- An "Open Questions for Implementation" item from brief §10 applies to your current step and Q3/Q4/Q5/Q6 haven't been resolved.
- You find yourself wanting to add a feature, helper, or refactor not specified in the brief.
- You're about to commit precomputed JSON > 30 MB total. Flag for review — Git LFS may be needed.

---

## Verify as you go

- **After each component is wired:** Take a screenshot via the `preview_*` MCP tools. Walk through every slider/toggle/click target on the live preview. Check console (`preview_console_logs level=error`) for errors. Confirm: no console errors, all interactions render in <100 ms (except first-hydration computes), static-fallback PNG visible pre-hydration, dark-mode rendering correct.
- **After each precompute script is run:** Verify the JSON validates against the matching loader's schema in `bayesian-ml.ts`. Verify the dual-write happened (file exists in both `src/data/sampleData/...` and `public/sample-data/...`).
- **After full build:** `pnpm build` exits 0. No `katex-error` spans in built HTML (`grep -c katex-error dist/topics/bayesian-neural-networks/index.html` should return 0). Pagefind index includes BNN page. `pnpm preview` re-walks identically to dev.
- **After all 9 viz are wired:** Walk brief §14 testing checklist top to bottom — the "Visualization components" subsection in particular. Every checkbox should now check (the v1 PR left some unchecked with `(deferred to v2)` annotations; v2 closes them).

---

## Final deliverable

A draft pull-request description containing:

- Short summary paragraph: "Rebuilds the 9 BNN viz components from v1's static-figure wrappers into fully interactive React + D3 components per brief §4's per-section 'Interactive controls (v1 deliverable)' specs."
- Files touched: `components/viz/*.tsx` (9 components rewritten), `components/viz/shared/bayesian-ml.ts` (extended with in-browser MLP training helpers + verification tests), `notebooks/bayesian-neural-networks/precompute_*.py` (4 scripts), `src/data/sampleData/bayesian-neural-networks/*.json` + `public/sample-data/bayesian-neural-networks/*.json` (precomputed payloads, dual-written).
- Per-component verification: a row per component listing first-hydration cost, slider re-render time, payload size (if precomputed), and screenshot.
- Brief §14 "Visualization components" checklist — fully checked, no `(deferred to v2)` annotations remaining.
- A "Deviations from the brief" list. Likely items: erf/tanh activation deferral in §9, any precompute-grid coarsening adopted to keep payload sizes reasonable.
- Pointer to the dev-preview rendered topic page.

---

## How to start

1. Read the four required files (CLAUDE.md, the brief — focus on §4 viz design intents and §10 open questions, the notebook, the handoff-reference doc).
2. Read the pattern references (`bayesian-ml-stacking.ts`, `gaussian-processes.ts`, two recent T5 viz components, the precompute precedent).
3. Summarize the brief's per-component interactive-control specs in your own words (one bullet per component, 3–6 lines total) so I can confirm we're aligned.
4. Decide on the erf/tanh activation question for §9 (recommended: defer to v3).
5. Start with **Stage 1: NNGPSidebarViz** — closed-form, no precompute, no MLP training. Pause and report when wired.
6. Continue stage-by-stage with check-ins per the workflow rules.

---

*v1 PR shipped: topic content + curriculum + cross-references + 9 static-figure wrappers + figures + shared module (types + ECE/NLL/Brier + NNGP kernel + JSON loaders + heavy-training stubs). v2 PR's job is to replace the 9 wrappers with fully interactive components matching brief §4 specs.*
