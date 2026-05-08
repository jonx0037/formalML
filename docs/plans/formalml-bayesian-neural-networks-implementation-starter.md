# Bayesian Neural Networks — filled-in Claude Code starter

## Placeholder values

| Placeholder | Value |
|---|---|
| `{{TOPIC_SLUG}}` | `bayesian-neural-networks` |
| `{{TOPIC_TITLE}}` | `Bayesian Neural Networks` |
| `{{BRIEF_PATH}}` | `docs/plans/formalml-bayesian-neural-networks-handoff-brief.md` |
| `{{NOTEBOOK_PATH}}` | `notebooks/bayesian-neural-networks/01_bayesian_neural_networks.ipynb` |
| `{{NOTEBOOK_RUNTIME_S}}` | `TBD — measure locally before pasting; brief §12 estimates ~58 s end-to-end on a 2020-era CPU laptop. Update this value once the notebook has been run end-to-end via notebooks/bayesian-neural-networks/.venv/.` |

**Pre-paste verification** (run before pasting the body into Claude Code):
1. `cd /Users/jonathanrocha/Developer/Sites/formalML`
2. `python3 -m venv notebooks/bayesian-neural-networks/.venv && notebooks/bayesian-neural-networks/.venv/bin/pip install numpy scipy scikit-learn matplotlib torch`
3. Run the notebook end-to-end (`jupyter nbconvert --to notebook --execute notebooks/bayesian-neural-networks/01_bayesian_neural_networks.ipynb --output 01_bayesian_neural_networks.executed.ipynb` or open in Jupyter), confirm wall-clock under ~81 s, all 9 verification triples in expected ranges per brief §14 "Notebook" section.
4. Update `{{NOTEBOOK_RUNTIME_S}}` in the closing footer of the body below with the measured value.

---

## Template Body

> Copy from the line below to the end of this document, into a fresh Claude Code session in the formalML repo root. Claude Code will automatically load `CLAUDE.md`.

---

# Implement formalML topic: Bayesian Neural Networks

You are implementing the formalML topic **Bayesian Neural Networks** in this repo. The notebook for this topic is verified and immutable; the handoff brief is the implementation spec. Your job is to produce a single self-contained pull request that ships the topic to the live site.

## Required reading

Read these files in order before doing anything else:

1. `CLAUDE.md` (repo root) — project conventions and the do-NOT list.
2. `docs/plans/formalml-bayesian-neural-networks-handoff-brief.md` — the implementation spec for this topic.
3. `notebooks/bayesian-neural-networks/01_bayesian_neural_networks.ipynb` — the verified source-of-truth notebook for math, code, and figures.
4. `docs/plans/formalml-handoff-reference.md` — cross-cutting conventions across all formalML topics.

If any of these files is missing, stop and report which one. Do not proceed.

## Workflow rules

**The brief is the spec. The notebook is the source of truth.** Both are immutable. Pull math, proofs, code snippets, and figure content from the notebook; pull MDX structure, component specifications, curriculum-graph deltas, and cross-reference details from the brief. The brief explicitly tells you which notebook cell each MDX section maps to (brief §3 section outline) — use that mapping rather than inventing your own.

Follow the brief **§13 (Build Order)** step-by-step. Treat each numbered step as a discrete unit of work. Pause and report progress to me before proceeding to the next step at any of the following:

- After step 1 (notebook end-to-end verification). Report the actual wall-clock and the 9 verification triples.
- After step 3 (extending the shared module). Report the §5 verification-test results.
- Before building each new viz component (steps 5–7 in build order). Confirm the static-figure fallback is in place first.
- After each viz component is wired in. Report numerical agreement with the corresponding notebook output.
- Before touching the curriculum graph (step 8).
- Before the inbound MDX edits (step 8).
- Before running `pnpm audit:cross-site` (step 9).

## Do not

- Modify, re-run, or regenerate the notebook. It is read-only.
- Re-derive math, rewrite proofs, or paraphrase the notebook's mathematical exposition. The MDX should pull definitions, theorems, proofs, and remarks directly from the brief / notebook (per brief §3 mapping). Adapt formatting for `TheoremBlock` rendering, but do not rewrite content.
- Add tests, lint rules, ESLint configuration, or CI workflow changes that are not specified in the brief.
- Make architectural changes to shared UI components (`TheoremBlock`, `ExternalLink`, `NamedSection`, etc.) without explicit authorization.
- Create new tracks, new domain keys, or new frontmatter schema fields without explicit authorization. (`bayesian-ml` is the existing T5 domain key; do not mint a new one.)
- Use npm or generate `package-lock.json` (per CLAUDE.md — pnpm only).
- Skip the verification steps in brief §5 or the testing checklist in brief §14.
- Refactor the notebook's PyTorch code to NumPy/SciPy. PyTorch is whitelisted for this topic per strategic doc §8.2; the language policy is intentional.

## Verify as you go

- **After step 1 (notebook verification):** Report the §14 "Notebook" checklist results as a table — three columns (test name, expected range from brief, observed value).
- **After step 3 (shared module):** Run the §5 verification tests against the notebook's printed numerical outputs. Same three-column-table reporting format.
- **After each viz component is wired:** Report numerical agreement with the corresponding notebook cell's printed outputs.
- **After full build:** Walk brief §14 testing checklist top to bottom. Report as a fully checked list. Any item that cannot be verified locally (e.g., production-deploy first-paint times) gets a `(deferred to deploy)` annotation rather than a check.

## Stop and ask before continuing if

- Anything in the brief conflicts with `CLAUDE.md` or with what you find in the notebook.
- A required file from the "Required reading" list is missing.
- A verification test fails by more than the brief's stated tolerance.
- An "Open Questions for Implementation" item from brief §10 applies to your current step and has not been resolved (the topic-specific notes below flag the ones that need pre-decisions).
- Anything in the brief reads ambiguously to you after a careful read.
- You find yourself wanting to add a feature, helper, or refactor not specified in the brief.

## Content metrics spreadsheet

After the topic MDX is complete and all references have `url` fields, update `docs/formalml-content-metrics.xlsx`:

1. **Topic Detail sheet** — add a row for Bayesian Neural Networks with all 18 columns (domain = `bayesian-ml`, title = `Bayesian Neural Networks`, difficulty = `advanced`, word count, section count = 9, overview/formal/code/viz/connections flags, python blocks, theorem blocks, display equations, inline math, refs total, refs with URL, refs missing URL, section names from brief §3).
2. **Domain Summary sheet** — recompute the row for `bayesian-ml` (T5 now has 6 published topics: VI, GPs, PP, mixed-effects, stacking, BNN).
3. **Gap Analysis sheet** — BNN should clear all thresholds (≥7 sections, ≥3k words, has Code section). No new gap entries expected.

Every reference in the BNN `references` frontmatter array must have a `url` field. The brief §11 supplies all 24 references with URLs already vetted (DOI / proceedings / arXiv) — copy through verbatim.

## Final deliverable

A draft pull-request description containing:

- A short summary paragraph stating what shipped (topic name, viz component count = 9, figure count = 9, prereq additions = 5 internal + 5 formalstatistics + 1 formalcalculus, T5 published topics now = 6).
- Files touched, grouped by category: `content/`, `components/viz/`, `components/viz/shared/`, `components/ui/` (if any), `data/`, `public/images/`, `pages/`, and inbound `.mdx` updates.
- The §5 shared-module verification table.
- The §14 testing checklist (fully checked or with deferral annotations).
- Content metrics spreadsheet update confirmation (refs total / refs with URL / refs missing URL = 24 / 24 / 0).
- A short "Deviations from the brief" list — items where you departed from the spec, with rationale. If empty, write `None.`
- A pointer to the dev-preview rendered topic page (URL or path).

## How to start

1. Read the four required files.
2. Summarize the brief's §13 build order in your own words (3–6 lines) so I can confirm we're aligned.
3. Identify any open questions from the topic-specific notes below that block immediate progress.
4. Propose the first batch of steps (typically build-order steps 1–3) and wait for my approval before implementing.

---

## Topic-specific notes

This is the only section that varies by topic. Pointers to brief sections plus genuine gotchas not obvious from a fresh read.

**Open Questions to resolve before specific build-order steps** (brief §10, six items):

1. **Q1 — formalcalculus SDE prereq.** Brief §2's frontmatter lists `formalcalculusPrereqs: stochastic-differential-equations`. Verify formalcalculus has this topic before step 2 (frontmatter writing); if not, drop the entry and leave §§6–7 self-contained on the SDE math.
2. **Q2 — `<ExternalLink>` component.** Verify whether `src/components/ui/ExternalLink.astro` already exists. If not, port it from formalstatistics during step 7 of the build order per CLAUDE.md / handoff-reference §10. (Port is in scope for this PR if no prior topic shipped a body link to formalstatistics; otherwise it should already exist.)
3. **Q3 — In-browser vs. precomputed Laplace samples.** §3's React component should consume precomputed JSON output (Laplace MC samples + per-grid predictive probabilities) from the notebook's figures directory rather than recomputing the Hessian in-browser. Confirm this approach during step 5; the in-browser alternative blocks first paint by ~1 s.
4. **Q4 — SG-MCMC chain payload size.** §§6–7 chain output is too large to ship raw (16 MB per method). Pre-compute per-grid mean and standard deviation only (~640 KB per method). Confirm during step 6.
5. **Q5 — Calibration test set.** Notebook uses `make_moons(n=500, random_state=SEED+1000)`. Decide whether the live MDX exposes a seed slider or locks the test set during step 7.
6. **Q6 — Cold-posterior interactive demo.** §8.5 introduces cold posteriors as a remark; an interactive viz over $T$ would make it concrete. Decide v1 vs. v2 during step 7. Defaulting to v2 is fine.

**Genuine gotchas not obvious from a fresh read:**

- **All 9 viz components are interactive React + D3, not static-figure renderers.** Per CLAUDE.md ("Every topic gets three pillars: rigorous math, interactive visualization, and working code") and brief §4's per-section "Interactive controls (v1 deliverable)" subsections. The notebook PNGs in `public/images/topics/bayesian-neural-networks/` are *static fallbacks* — used pre-hydration and when JS is disabled — *not* the on-page rendering. Each component exposes 2–4 controls (sliders, dropdowns, click targets, toggles) per the brief's spec. Build them as full React + D3 interactive components per `docs/plans/formalml-handoff-reference.md` §4 (use `useD3` hook, `useResizeObserver`, CSS-custom-property color tokens via `var(--color-*)`, `client:visible` hydration). Do not ship "JSON-renderer" components that just paint the static figure — that misses the point of the topic.
- **PyTorch is intentional, not legacy.** This topic is on the seven-topic PyTorch-allowed exception list per strategic doc §8.2 and CLAUDE.md "Code-example language policy." Do not refactor to NumPy/SciPy. The architecture (3 hidden × 32 ReLU, $p = 2241$) is small enough that PyTorch CPU runs in seconds.
- **The §3 full Hessian via `torch.autograd.functional.hessian` is feasible only because $p$ is small.** For any architecture change pushing $p > 5000$, switch to last-layer or KFAC Laplace per brief §3.4. Do not change the Two Moons architecture without recomputing the brief's per-cell runtime estimates.
- **The notebook reuses trained models across sections.** §1 trains 5 MLPs that §§2 and 5 reuse; §3's Laplace MAP is the §1 first-seed model; §4's dropout MLP is its own; §5 trains 10 (5 reused from §1 + 5 new); §6's SGLD warm-starts from MAP; §7's SGHMC warm-starts from MAP; §8 reuses *all* trained models from §§1, 3, 4, 5, 6, 7 for the calibration head-to-head. Replicate this caching in the React layer (load precomputed JSON from `public/sample-data/bayesian-neural-networks/` per the dual-location convention in CLAUDE.md "Sample-data dual-location") to stay within client-side payload budget.
- **Per-section figure → React viz mapping** is in brief §13 build-order steps 5–7 and §9's images table. Each viz component name is suggested in step 5–7; the brief's §4 "Viz design intent" subsections specify the panel layouts and color palettes (using setup-cell `COLOR_*` tokens — `COLOR_LAPLACE` for §3, `COLOR_DROPOUT` for §4, etc.).
- **The pre-commit hook (`.claude/hooks/preview-before-commit.sh`) blocks Claude-triggered `git commit` on staged topic MDX unless `astro dev` is running.** Per CLAUDE.md "Preview before publish" section. Start `pnpm dev` before any commit attempt during build-order step 11.
- **`weight-decay` is a named subsection inside §2 (Prop 2.2), not a separate topic.** Per CLAUDE.md "Do NOT" list and strategic doc §7.3 — the slug `weight-decay` is reserved-but-never-minted. The brief makes this explicit; preserve it as a named subsection under §2, not a separate page.
- **`bayesian-neural-networks` is the T5 flagship.** Brief difficulty rating is `advanced`. The §14 testing checklist's expected calibration ordering (deep ensemble + SGHMC tied for best, MC-dropout / point-estimate at the bottom on Two Moons) is empirically what the notebook produces — if implementation produces a different ordering, debug rather than altering the brief's expected ordering.
- **Sister-site reciprocity is a follow-up PR, not in scope for the formalML PR.** Brief §8 documents the formalstatistics PR work that follows shipment. The formalML PR ships independently; the audit is expected to flag five deferred reciprocals on shipment, all listed in `docs/plans/deferred-reciprocals.md` under "When `formalml/bayesian-neural-networks` ships."

---

*Notebook verified runtime: TBD — measure locally before pasting; brief §12 estimates ~58 s end-to-end on a 2020-era CPU laptop. Update this value once the notebook has been run end-to-end via `notebooks/bayesian-neural-networks/.venv/`. If you re-run the notebook for figure export and the runtime exceeds 1.4× this value, something is wrong; stop and report.*
