# Drafting formalML topic: Riemann Manifold HMC

## Project context

formalML is a static site of long-form mathematical explainers for ML practitioners, grad students, and researchers, live at https://formalml.com. Every topic gets three pillars: rigorous math, interactive visualization, and working code. formalML is the third site in the triad **formalcalculus → formalstatistics → formalML**.

## Deliverables (two of them)

This conversation produces **two coupled artifacts**. Both feed downstream into a separate Claude Code session that ships the topic to the live site.

1. **Markdown handoff brief** — saved at `docs/plans/formalml-riemann-manifold-hmc-handoff-brief.md`. The implementation spec: section outline, theorem statements, full proofs, concrete examples, viz design intent (component-level, not React/D3 code), code-experiment sketches, cross-site prereq list, references. Structured prose Claude Code reads to know what to build.

2. **Native Jupyter notebook** — saved at `notebooks/riemann-manifold-hmc/01_riemann_manifold_hmc.ipynb` (directory uses hyphens, filename uses underscores). The source of truth for math, code, and figures: markdown cells for the math exposition (definitions, theorems, proofs, examples) and code cells that validate the math numerically and generate the figures. Constraints: CPU-only, end-to-end runtime under 60 seconds on a 2020-era laptop, libraries per the code-language policy below. Output as a native `.ipynb` artifact (raw JSON), not as inline code blocks scattered through chat.

**The brief is the spec. The notebook is the source of truth.** Both should be drafted *in parallel* — math derivation in the brief informs the code experiment in the notebook, and numerical results from the notebook feed back into the brief as remarks and figure references. We are NOT writing MDX, modifying repo files, running audits, or creating PRs in this conversation; that's the next session's job.

## Topic

- **Title:** Riemann Manifold HMC
- **Slug:** riemann-manifold-hmc
- **Track:** T5 Bayesian & Probabilistic ML
- **Difficulty:** advanced
- **Positioning:** Specialized MCMC variant in T5's "Advanced Bayesian Computation" cluster — second member of the specialized-sampler set after `stochastic-gradient-mcmc`, with `sequential-monte-carlo` and `reversible-jump-mcmc` to follow as the remaining two T5 topics that close out the ML Methodology layer. The defining move, per Girolami & Calderhead 2011 (*J. R. Statist. Soc. B* 73(2)): replace standard HMC's flat-Euclidean mass matrix $M$ with a position-dependent Riemannian metric tensor $G(\theta)$ — canonically the Fisher information $G(\theta) = -\mathbb{E}_{y \mid \theta}[\nabla^2_\theta \log p(y \mid \theta)]$ — so Hamiltonian flow follows geodesics on a manifold whose curvature reflects the local geometry of the target. The pedagogical payoff: pathological posteriors that defeat standard HMC (Neal's funnel, banana distributions, correlated near-singular targets) get flattened in Riemannian coordinates, where the same fixed step size now traverses the support at a rate matched to local information density. The cost: the leapfrog integrator becomes *implicit* (fixed-point iteration per step) and each step requires a Cholesky of $G(\theta)$ — wall-clock dominated by metric-tensor evaluation rather than per-step proposal cost. RMHMC is the first T5 topic where the shipped Differential Geometry foundation pays off as constitutive rather than incidental: `riemannian-geometry` supplies the metric tensor and geodesic equation, `geodesics-curvature` supplies the explicit Christoffel symbols and parallel-transport machinery, and `information-geometry` supplies the Fisher metric as the *unique* (Čencov) Riemannian structure on the manifold of probability distributions — the same Fisher metric that becomes the RMHMC mass matrix. The topic also delivers on a thread already opened in `stochastic-gradient-mcmc`'s §10, which introduced the Riemann-manifold metric tensor as a prelude. Discharges the `formalstatistics/bayesian-computation-and-mcmc` §26.10 Rem 29 forward-pointer (per-topic discharge count is 1, but the cross-track integration depth makes it disproportionately load-bearing for the curriculum graph).
- **Code language:** Python + NumPy / SciPy / scikit-learn / pandas / matplotlib (the project default). RMHMC is **not** on the seven-topic PyTorch / JAX exception list, so the notebook stays in the NumPy stack. The Girolami–Calderhead 2011 demo models all have closed-form Fisher metrics that can be hand-derived and evaluated as NumPy expressions: (i) multivariate Gaussian with unknown mean and covariance — $G(\theta)$ block-diagonal in $(\mu, \Sigma)$; (ii) the banana distribution (warped bivariate normal, the canonical "RMHMC beats HMC" demo) — Fisher metric is a $2\times2$ matrix of polynomial entries; (iii) log-Gaussian Cox process (the topic's recurring spatial-statistics application) — Fisher metric requires one matrix-vector solve per likelihood evaluation but no autodiff. Useful primitives: `scipy.linalg.cho_factor` and `cho_solve` for the per-step Cholesky and metric-inverse solves (prefer over `numpy.linalg.solve` — Cholesky is ~2× faster and numerically more stable for SPD systems); `scipy.optimize.fsolve` for the implicit-leapfrog fixed-point iteration, or hand-rolled Newton with a tight tolerance (the inner solver is the dominant runtime cost and warrants direct control); `arviz` for trace plots, $\hat R$, IAT, and ESS diagnostics (matches the SG-MCMC notebook's diagnostic vocabulary). CPU-only, end-to-end under 60 s on a 2020-era laptop; the budget assumes ~500 samples on the banana benchmark and ~200 samples on the log-Cox process — both feasible without GPU acceleration because the implicit-leapfrog inner-solve cost dominates over the per-likelihood evaluation cost on these low-dimensional demos.

## Reference notebooks

I've attached the following notebooks to this chat as `.ipynb` files. They are exemplars of the formalML house style — math exposition, code idioms, viz design intent, proof depth, cell organization, library imports, figure styling. When in doubt about voice or structural convention, defer to these. They are also templates for the notebook deliverable you'll produce for this topic: same cell layout, same import pattern, same figure aesthetics. They are NOT source material for **Riemann Manifold HMC** itself; that's what we're drafting in this conversation.

- **`notebooks/stochastic-gradient-mcmc/01_stochastic_gradient_mcmc.ipynb`** — closest substantive sibling (T5 Bayesian ML, advanced difficulty, same specialized-MCMC cluster as RMHMC). Its §10 introduces the Riemann-manifold metric tensor $G(\theta)$ as a prelude, deriving the position-dependent drift term that bridges SGLD's flat-Euclidean noise to RMHMC's Riemannian leapfrog — RMHMC is the topic that delivers on that thread. Use this notebook as the *house-style template for sampler topics*: the diagnostic vocabulary ($\hat R$, IAT, ESS via `arviz`), the trace-plot aesthetics, the head-to-head benchmarking framework against NUTS, and the cell-organization pattern (closed-form derivation → numerical verification → figure save). The SG-MCMC notebook's bias-variance trade-off comparison panels (§8 Vollmer–Zygalakis–Teh) provide the template for RMHMC's "step-size vs metric-evaluation-cost" comparison panels.
- **`notebooks/density-ratio-estimation/01_density_ratio_estimation.ipynb`** — freshest shipped exemplar (T3 Unsupervised & Generative, advanced difficulty, merged via PR #93 on 2026-05-17). Latest house conventions for cell layout, figure aesthetics (matplotlib styling, palette choices, axis annotation), and the dual-output brief+notebook coordination pattern. Demonstrates the current preferred structure for math-cell ↔ code-cell interleaving and the figure-numbering convention that downstream MDX `<Figure>` components will reference. Not a substantive cousin (DRE and RMHMC share no math), but the *most up-to-date stylistic exemplar* — defer to it for anything that has drifted since the SG-MCMC notebook shipped.
- **`notebooks/information-geometry/04_information_geometry.ipynb`** — prereq-edge notebook from the Differential Geometry foundation (advanced difficulty). The Fisher information metric, Čencov's uniqueness theorem, the dual α-connections, the KL divergence as the infinitesimal limit of the Fisher metric, and the natural-gradient story are all already developed there. RMHMC reuses this material as its metric tensor — this notebook *is* the mathematical prerequisite the topic depends on, and the brief should explicitly cite §-references from the published `information-geometry` MDX (and from this notebook's matching cells) rather than re-deriving Čencov or the Fisher-as-Hessian-of-KL identity. The natural-gradient connection in particular is the conceptual bridge: natural gradient is the *first-order* analog of RMHMC, and the topic's §1 motivation should make this explicit.

## Editorial voice (non-negotiable)

- **Tone:** Informed peer, not lecturer. Think "a sharp colleague explaining something at a whiteboard" — conversational enough to use contractions and the occasional aside, but precise enough that no claim is hand-wavy. The prose should read well *as prose*, not just as a vehicle for equations.
- **Pronouns:** Default to "we" as the collaborative mathematical "we" (we define, we observe, we can now see that…). Use "you" sparingly and only for direct reader instructions — "you can verify this by…" or "try dragging the slider to see…". Avoid passive voice for derivations; if someone is doing the math, say who.
- **Assumed reader knowledge:** The reader has taken linear algebra and multivariable calculus and has seen some probability. They may or may not have a CS degree. Don't explain what a matrix is; do explain why a specific decomposition matters here. For *this* advanced topic, assume comfort with standard HMC (Neal 2011 review), Bayesian inference with conjugate priors, basic Riemannian-geometry vocabulary (metric tensor, geodesic, Christoffel symbols — all developed on formalML's foundation track), the Fisher information matrix and its score-covariance interpretation, and intermediate-level MCMC diagnostics ($\hat R$, ESS, IAT). Do not assume familiarity with symplectic integrators beyond the basic leapfrog, with semi-explicit ODE solvers, or with the Bayesian-statistics applications of the log-Gaussian Cox process.
- **Jargon and notation:** Introduce notation explicitly on first use in every topic — even standard stuff like $\|\mathbf{x}\|$. Never let a symbol appear without a plain-English gloss nearby. Jargon is fine once defined, but prefer the concrete name over the abstract one when both exist (say "the matrix of expected second derivatives of the log-likelihood" before saying "Fisher information").
- **Attitude toward the reader:** Respect without flattery. Don't say "simply," "obviously," or "it's easy to see." If something is genuinely straightforward, the exposition will make that self-evident. If something is hard, say so — "this step is where the real work happens" is more useful than pretending it's trivial.

## Mathematical exposition style

- **Geometric-first:** Introduce concepts visually and concretely before algebraic machinery. RMHMC is constitutively geometric — Hamiltonian flow on a Riemannian manifold — so this is the natural register rather than a stylistic constraint.
- **Foundational topics:** Zero algebra — stop at geometric intuition.
- **Intermediate topics:** Algebra only after geometric setup is established.
- **Proofs:** Expand fully with combinatorial detail — never "it can be shown."
- **Examples:** Concrete, motivating examples before every definition.

## Code-language policy

Default for all topics: **Python + NumPy / SciPy / scikit-learn / pandas / matplotlib.** Notebook cells must run CPU-only in under 60 seconds on a 2020-era laptop — no GPU requirements.

PyTorch / JAX is allowed only for these seven planned topics: `normalizing-flows`, `bayesian-neural-networks`, `meta-learning`, `stochastic-gradient-mcmc`, `variational-inference`, `density-ratio-estimation` (neural DRE section only), `probabilistic-programming` (Stan / PyMC / NumPyro is the subject matter). All other topics stay in the NumPy/SciPy default.

**Riemann Manifold HMC is not on the exception list — use the default stack.**

## Cross-site references

formalML topics declare prerequisites on the two sister sites via four frontmatter fields, each entry an object with `topic` (slug, no extension), `site` (`formalcalculus` | `formalstatistics`), and `relationship` (≥40 chars of explanatory prose):

- `formalcalculusPrereqs` — calculus topics this ML topic requires (backward link).
- `formalstatisticsPrereqs` — statistics topics this ML topic requires (backward link).
- `formalcalculusConnections` — calculus topics this ML topic informs (forward link, rare).
- `formalstatisticsConnections` — statistics topics this ML topic informs (forward link, rare).

Do **not** use `formalmlPrereqs` or `formalmlConnections` — those are reserved for sister sites pointing inward at formalML, not for formalML pointing at itself.

For RMHMC, the most likely sister-site prereqs to surface during the section drafting:
- `formalstatistics/bayesian-computation-and-mcmc` — the deep MCMC prerequisite. RMHMC defines itself against the §26 catalog: Gibbs, Metropolis–Hastings, HMC, NUTS. The §26.10 Rem 29 forward-pointer is the one this topic discharges.
- `formalstatistics/modes-of-convergence` — for the geometric-ergodicity statement and the relationship between trajectory length and effective sample size.
- `formalcalculus/jacobian` — for the change-of-variables under reparameterization, which RMHMC's volume-preservation argument depends on.
- `formalcalculus/implicit-function-theorem` — the implicit leapfrog's well-posedness rests on this; cite rather than re-prove.

## Do not

- Add algebra to foundational-level topics.
- Write one-line proof sketches — expand fully or omit the proof. (RMHMC has at least three load-bearing proofs that need full treatment: volume preservation of the implicit leapfrog, reversibility of the generalized leapfrog, detailed-balance of the Metropolis correction.)
- Propose a topic that duplicates engineering coverage of an existing formalstatistics topic — and specifically do **not** re-derive standard HMC from scratch (cite `formalstatistics/bayesian-computation-and-mcmc` and the Neal 2011 *Handbook of MCMC* chapter for the prereq treatment; this topic *extends* HMC, not re-introduces it).
- Propose standalone sections for `cross-validation`, `ab-testing`, or `weight-decay` — those are named sections within host topics, never their own pages.
- Place RMHMC in the T2 Supervised Learning track — it's T5 Bayesian Computation.
- Use the `formalmlPrereqs` or `formalmlConnections` fields on cross-site reference lists you propose — those are flagged as `self-site` by the audit and are reserved for sister sites.

## What I want from this conversation

The artifacts below populate **both** deliverables. Definitions, theorem statements, proofs, and examples appear in the brief as structured prose AND in the notebook as markdown cells (same content, two surfaces). Code-experiment design becomes runnable cells in the notebook AND a section in the brief describing what runs and why. Viz intent and references go to the brief only — the notebook produces matplotlib figures, not React/D3 components.

In rough order:

1. **Section outline** — 10–14 H2 sections, each with 3–5 H3 subsections and a one-line summary per H2 of what it covers (and a one-clause summary per H3). The outline structures both the brief and the notebook (same headings on both sides). This depth matches recently shipped topics: variational-bayes-for-model-selection (13 H2), sparse-bayesian-priors (13 H2), stochastic-gradient-mcmc (the direct sibling — match its depth). H3 subsections are first-class structural elements — populate them, don't fold them into prose.
2. **Definitions and theorem statements** — formal, with display math, every symbol glossed on first use.
3. **Full proofs** — expanded with combinatorial detail. Never "it can be shown" or "the rest follows."
4. **Concrete motivating examples** — at least one before every definition; reusable across the topic if possible. The banana distribution and the log-Gaussian Cox process should each appear in §1 (motivation) and recur in §-by-§ numerical validation.
5. **Viz design intent** — as many interactive visualizations as the §-by-§ outline naturally produces; **don't artificially cap**. Recent T5 shipments are 8–12 per topic (VBMS shipped 10, sparse-bayesian-priors shipped 11, SG-MCMC shipped 10). For each viz: what it shows, what parameters the reader can manipulate, what the reader should learn. Component-level *intent*, not React/D3 implementation. Roughly one viz per H2 that has interactive content; some sections (computational notes, connections-and-limits) are text-only and intentionally lack viz.
6. **Code-experiment design** — what numerical experiments validate the math, what figures they produce, what the runtime budget allows. Specify libraries and key API calls; do not write full notebook code. The two recurring demos (banana + log-Cox) should each get a dedicated §-by-§ "what's running here" treatment.
7. **Cross-site prereq list** — formalcalculus and formalstatistics topics this topic requires, with `relationship` prose ≥40 chars per entry. Forward `Connections` only if a sister-site topic genuinely benefits from this one as a follow-up.
8. **References** — Chicago 17th edition (Notes and Bibliography). Group as foundational / advanced / applied. Prefer canonical primary sources over secondary surveys. **Every reference must include a `url` field** — a DOI link (`https://doi.org/...`) for journal articles and books, a proceedings URL for conference papers, or an arXiv link as a last resort. References without URLs render as unlinked plain text on the site and are tracked as gaps in the content metrics spreadsheet (`docs/formalml-content-metrics.xlsx`). The Girolami & Calderhead (2011) RSS paper, the Betancourt (2017) *A Conceptual Introduction to HMC* review, and the Livingstone & Girolami (2014) "Information-geometric Markov chain Monte Carlo methods" survey are the foundational triple; the Hairer–Lubich–Wanner (2006) *Geometric Numerical Integration* book is the canonical reference for the implicit-leapfrog numerical analysis.

## How to start

1. Acknowledge that you've read the constraints above and skimmed the attached reference notebooks (math style, code idioms, viz parameter choices, proof depth, cell organization). Flag anything in either that's unclear or contradictory before drafting.
2. Propose a section outline of **10–14 H2 sections, each with 3–5 H3 subsections**. One-line summary per H2, one-clause summary per H3. The same outline structures both the brief and the notebook. Wait for my confirmation before drafting any section.
3. After I confirm the outline, ask which section to draft first.
4. Draft one section per turn, producing **both** the brief prose (Markdown) and the matching notebook cells (markdown cells for exposition + code cells for experiments). Show the code cells inline for me to review before committing them to the notebook artifact. Do not bundle multiple sections into one response unless I explicitly ask.
5. After each section, propose what to draft next (or what to revise) and wait for my go-ahead. Do not autonomously continue to the next section.
6. When all sections are drafted and I confirm the topic is complete, assemble two final artifacts: the full brief as a single Markdown document, and the full notebook as a native `.ipynb` (raw JSON). I'll save them to disk and then start the Claude Code implementation session.

## Stop and ask if

- The proposed outline conflicts with the track positioning or with the do-not list.
- A proof needs a tool, lemma, or prerequisite I haven't mentioned and that isn't on a sister-site topic the reader can be assumed to have read.
- A viz design needs a dataset, parameter regime, or numerical experiment I haven't specified.
- A reference can't be verified or you're unsure of bibliographic details — ask before fabricating.
- You find yourself wanting to write MDX, edit `curriculum-graph.json`, run `pnpm audit:cross-site`, or do anything that touches "the repo." This conversation is pre-brief drafting only — those steps belong in a later Claude Code session.

---

*Two deliverables, both saved before the next session: the brief at `docs/plans/formalml-riemann-manifold-hmc-handoff-brief.md`, and the notebook at `notebooks/riemann-manifold-hmc/01_riemann_manifold_hmc.ipynb`. Then hand off to Claude Code via the companion `Claude Code Starter Prompt — formalML Topic Implementation Template.md`. The Chat conversation produces brief + notebook; the Code session reads both and ships the topic to the live site.*
