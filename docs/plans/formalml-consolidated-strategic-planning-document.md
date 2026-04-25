# formalML.com Strategic Planning Document — Consolidated Forward-Topic Buildout

**Status:** strategic planning document (Claude Chat architectural artifact; upstream of per-topic handoff briefs)
**Scope:** 32 forward-identified formalML.com topics originating from formalstatistics.com forward-pointers across Topics 22–32, consolidated into an extended track scaffold and first-wave sequencing plan
**Audience:** Jonathan as decision-holder; secondary audience is Claude Code implementation sessions that will reference this document when composing per-topic handoff briefs
**Version:** v1.1 (April 24, 2026 — drafted after formalstatistics.com Topic 32 shipped and the curriculum closed; revised April 25, 2026 against the cross-site audit infrastructure shipped in formalML PR #52, see §1.3 / §5.2 / §5.5 / §5.6 / §12.4 / §18)
**Relationship to prior briefs:** consolidates and extends `formalML-five-forward-topics-planning-brief.md` (the Topic 29 brief, covering 5 topics) and `formalML-satellite-topics-planning-brief.md` (the Topic 30 brief, covering 5 topics). Both remain valid at the content level; this document layers architectural decisions and adds the 22 newly-identified topics from Topics 22/23/24/25/26/27/28/31/32.

---

## § 1 Purpose & Scope

### §1.1 Why this document exists

formalstatistics.com is now feature-complete: 32 topics across 8 tracks, every forward-promise either discharged internally or aimed at a named formalML target slug. The resulting forward-pointer surface is substantial — 32 distinct slugs scattered across 11 formalstatistics topics — and materially larger than the two prior planning briefs anticipated. Those briefs were written before Topics 25–28 and 31–32 shipped, and therefore, they couldn't see the full handoff surface.

This document does three things that briefs couldn't:

1. **Consolidates** the full 32-topic discharge surface into a single inventory, deduplicated against already-published formalML topics and engineering-duplicate scope decisions.
2. **Extends** formalML's existing 8-track math-foundation scaffold with 5 new application-focused tracks (Supervised Learning; Unsupervised & Generative; Nonparametric & Distribution-Free; Bayesian & Probabilistic ML; Learning Theory & Methodology), placing every new topic into its track with explicit rationale.
3. **Commits** to the architectural decisions that are one-time rather than per-topic: slug conventions, cross-site reference schema, code-example language policy, shared-module infrastructure plan, and amendments to `CLAUDE.md` and `formalML-handoff-reference.md`.

The per-topic handoff briefs that follow this document inherit these architectural decisions as givens and focus on content (section narratives, theorem selection, figure sketches, component specs). That separation of concerns is exactly what went wrong when handoff briefs tried to litigate track-placement decisions one topic at a time.

### §1.2 What this document is NOT

This is not a handoff brief for Claude Code. Handoff briefs are per-topic, MDX-structure-specific, and contain YAML frontmatter, section outlines, theorem statements, component specs, and implementation notes. Each of the 32 topics will get its own handoff brief at its turn in the sequencing queue. This document is the upstream architectural substrate that those briefs reference for track placement, cross-site schemas, naming, and shared-module coordination.

This document also does not include per-topic content plans for the 22 newly identified topics (beyond one-sentence scope framings). Content-level planning for those 22 will happen in planning briefs drafted during the approach to each topic's handoff-brief stage. The model is: strategic → content planning → handoff brief → Claude Code implementation.

### §1.3 Assumptions this document locks in

- **Tech stack inherited:** Astro 6 + React 19 + MDX + KaTeX + Tailwind CSS 4 + D3.js 7 + pnpm + Vercel (per existing `CLAUDE.md` and `formalML-handoff-reference.md`).
- **Editorial voice inherited:** Geometric-first exposition, full-proof tolerance, collaborative "we," the tone patterns in current `CLAUDE.md` §Editorial Voice.
- **Slug discipline:** short, filename-style slugs matching `formalML-handoff-reference.md` §8 precedent (`svd` not `singular-value-decomposition`; `pac-learning` not `pac-learning-framework`).
- **Cross-site reference infrastructure (revised, 2026-04-25):** This document was composed in Claude Chat at the same time as — but architecturally upstream of — the cross-site reciprocity audit infrastructure that shipped via formalML PR #52 (`scripts/audit-cross-site-links.mjs`, `pnpm audit:cross-site`). At publication time the formalstatistics and formalcalculus repos already had this infrastructure 100% adopted (32/32 topics each carry the new frontmatter schema), while formalML still had it 0% adopted (0/35 topics). §5 below has been revised to reference the actual six-field schema in production, the audit-script workflow, and the [deferred-reciprocals.md](deferred-reciprocals.md) ledger as the canonical "what to add when X ships" log. The original Claude Chat draft used a single-array `crossSitePrereqs` proposal and a manual `<ExternalLink>`-flipping PR-sweep protocol; both are obsolete.

---

## § 2 Executive Summary

### §2.1 Headline numbers

- **32 total new topics** across 5 new application-focused tracks (T2–T6)
- **Existing formalML state:** 35 topics across 8 complete math-foundation tracks
- **Final buildout target:** 67 topics across 13 tracks
- **Perfect discharge match:** 32 topics absorb all 32 distinct formalstatistics forward-pointer slugs (by construction of this inventory)
- **Estimated total authoring time:** 18–24 months at the Topic-30 cadence, assuming interleaving with other commitments continues

### §2.2 At-a-glance topic inventory

"Density" is the number of distinct formalstatistics topics whose forward-pointers a given slug discharges. Higher density = higher payoff per topic shipped (more link-debt retired per release).

| # | Slug | Track | Difficulty | Density | Primary source |
|---|---|---|---|---|---|
| 1 | `conformal-prediction` | T4 | intermediate | 2 | Topics 29/32 |
| 2 | `quantile-regression` | T4 | intermediate | 1 | Topic 29 |
| 3 | `rank-tests` | T4 | intermediate | 1 | Topic 29 |
| 4 | `extreme-value-theory` | T4 | advanced | 2 | Topics 29/32 |
| 5 | `statistical-depth` | T4 | advanced | 1 | Topic 29 |
| 6 | `prediction-intervals` | T4 | intermediate | 1 | Topic 31 |
| 7 | `kernel-regression` | T2 | intermediate | 1 | Topic 30 |
| 8 | `local-regression` | T2 | intermediate | 1 | Topic 30 |
| 9 | `high-dimensional-regression` | T2 | advanced | 3 | Topics 22/23/32 |
| 10 | `clustering` | T3 | intermediate | 1 | Topic 30 |
| 11 | `density-ratio-estimation` | T3 | advanced | 1 | Topic 30 |
| 12 | `normalizing-flows` | T3 | advanced | 1 | Topic 30 |
| 13 | `variational-inference` | T5 | intermediate | 3 | Topics 25/26/28 |
| 14 | `variational-bayes-for-model-selection` | T5 | advanced | 1 | Topic 27 |
| 15 | `bayesian-neural-networks` | T5 | advanced | 2 | Topics 25/28 |
| 16 | `gaussian-processes` | T5 | intermediate | 1 | Topic 25 |
| 17 | `probabilistic-programming` | T5 | intermediate | 2 | Topics 25/26 |
| 18 | `sparse-bayesian-priors` | T5 | advanced | 1 (heavy) | Topic 28 |
| 19 | `meta-learning` | T5 | advanced | 1 | Topic 28 |
| 20 | `mixed-effects` | T5 | intermediate | 2 | Topics 22/28 |
| 21 | `stacking-and-predictive-ensembles` | T5 | intermediate | 1 | Topic 27 |
| 22 | `stochastic-gradient-mcmc` | T5 | advanced | 1 | Topic 26 |
| 23 | `sequential-monte-carlo` | T5 | advanced | 1 | Topic 26 |
| 24 | `reversible-jump-mcmc` | T5 | advanced | 2 | Topics 26/27 |
| 25 | `riemann-manifold-hmc` | T5 | advanced | 1 | Topic 26 |
| 26 | `generalization-bounds` | T6 | intermediate | 2 | Topics 24/32 |
| 27 | `pac-bayes-bounds` | T6 | advanced | 1 | Topic 32 |
| 28 | `vc-dimension` | T6 | intermediate | 1 | Topic 24 |
| 29 | `semiparametric-inference` | T6 | advanced | 1 | Topic 32 |
| 30 | `causal-inference-methods` | T6 | advanced | 2 | Topics 22/32 |
| 31 | `double-descent` | T6 | advanced | 1 | Topic 23 |
| 32 | `uncertainty-quantification` | T6 | intermediate | 1 | Topic 31 |

`sparse-bayesian-priors` carries a "1 (heavy)" density marker because although it is named only in Topic 28 as the discharge venue, Topic 28 itself chains together forward-promises from Topics 25, 26, and 27 — so shipping this one topic retires four separate formalstatistics remarks (Topic 25 Rem 23, Topic 26 Rem 25, Topic 27's forward-map, Topic 28 Rem 23).

### §2.3 Track distribution

- T2 Supervised Learning: 3 topics
- T3 Unsupervised & Generative: 3 topics
- T4 Nonparametric & Distribution-Free: 6 topics
- T5 Bayesian & Probabilistic ML: 13 topics (the biggest track — split possibility noted in §3.6)
- T6 Learning Theory & Methodology: 7 topics

Plus T1 "Foundations" as a meta-track pointing to the existing 8 math tracks — no new topics introduced there.

### §2.4 Decisions this document locks in

| Decision | Resolution | §reference |
|---|---|---|
| Merge aggressiveness | Less aggressive — distinct math = distinct topic | §7.2 |
| cross-validation, ab-testing, weight-decay | Named sections inside larger topics, not standalone | §7.3 |
| formalML logistic-regression, generalized-linear-models (engineering flavors) | Redirect out — logistic → `high-dimensional-regression`; standalone GLM → external reference | §5.4 |
| Specialized MCMC variants (RJ-MCMC, Riemann HMC) | Standalone topics in first wave | §2.2, §4 |
| Track scaffold | Extend to 13 tracks (existing 8 + new 5), not replace | §3 |

---

## § 3 The Extended Track Scaffold

The scaffold extends formalML's existing 8-track structure with 5 new application-focused tracks. The existing 8 math tracks (Topology & TDA; Linear Algebra; Probability & Statistics; Optimization; Differential Geometry; Information Theory; Graph Theory; Category Theory) become the "foundations layer"; the 5 new tracks build on that foundation toward specific ML-practitioner deliverables.

### §3.1 T1 — Foundations (meta-track, no new topics)

The 8 existing math-foundation tracks collectively constitute formalML's "Foundations" layer. They get a single navigation-level grouping on `/paths` but no new curriculum work. The role of T1 in this scaffold is architectural: it makes explicit that the new 5 tracks sit *above* the existing 8, and it gives the `/paths` page a clean two-layer taxonomy ("Foundations" vs "ML Methodology") that the current flat-list presentation doesn't have.

When the `/paths` page is revised to accommodate the new tracks, the 8 existing math tracks should appear as a grouped "Foundations" section and the 5 new tracks should appear as a grouped "ML Methodology" section. This is a presentation-layer change for the existing `/paths` page; no backend data restructuring is required beyond the track additions documented here.

### §3.2 T2 — Supervised Learning (3 topics)

**Scope.** Nonparametric and high-dimensional supervised-learning methodology, centered on covariate-conditional estimation and the modern theory of regression in $p \gtrsim n$ regimes.

| Slug | Difficulty | One-line scope |
|---|---|---|
| `kernel-regression` | intermediate | Nadaraya–Watson as the covariate-conditional extension of KDE; leading-order bias/variance with the design-density correction; curse of dimensionality. |
| `local-regression` | intermediate | Fan–Gijbels local-polynomial extension; boundary-bias elimination at odd degree; equivalent-kernel formulation. |
| `high-dimensional-regression` | advanced | $p \gg n$ regression asymptotics, restricted-eigenvalue conditions, lasso $\ell_2$-consistency, debiased lasso, minimax bounds. Absorbs the engineering `logistic-regression` scope. |

**Placement rationale.** These three share the "covariate-conditional estimation" narrative and the kernel-methods / penalized-estimation mathematical substrate. Gaussian processes could be cross-listed here, but they're placed in T5 Bayesian because the Bayesian framing is essential to their identity (see §3.5).

**Track size caveat.** Three topics are thin relative to the existing 4-per-track norm. That's acceptable here because most supervised-learning methods are implemented as specific techniques across other tracks (convex analysis → gradient descent in Optimization; spectral methods in Linear Algebra; GNNs in Graph Theory). T2 is the "what's left over" track for nonparametric and high-dim supervised methodology specifically.

### §3.3 T3 — Unsupervised & Generative (3 topics)

**Scope.** Density estimation, dimensionality reduction, clustering, and generative modeling from a statistical rather than a purely neural perspective.

| Slug | Difficulty | One-line scope |
|---|---|---|
| `clustering` | intermediate | Mean-shift clustering as gradient ascent on $\hat{f}_h$; fixed-point convergence; spectral-clustering connections (light). |
| `density-ratio-estimation` | advanced | KLIEP / LSIF / uLSIF for $r(x) = p(x)/q(x)$ direct estimation; covariate-shift correction; MMD as kernel ratio; GANs as implicit DRE. |
| `normalizing-flows` | advanced | Invertible neural density estimators; change-of-variables; coupling / autoregressive / continuous-time flow architectures; KDE ↔ flow comparison. |

**Placement rationale.** All three extend KDE (Topic 30) in "what you can do with a density once you have one" directions that diverge from the regression / conditional-mean story in T2. Inherited directly from the satellite brief.

### §3.4 T4 — Nonparametric & Distribution-Free (6 topics)

**Scope.** Distribution-free inference, permutation-based testing, extreme-value asymptotics, and multivariate nonparametric geometry. Reuses the "five-forward-topics brief" scope wholesale and adds `prediction-intervals` as a standalone topic (not a section within `conformal-prediction`) per §7.2.

| Slug | Difficulty | One-line scope |
|---|---|---|
| `conformal-prediction` | intermediate | Split / full / jackknife+ conformal prediction; marginal coverage guarantee; CQR; conditional-coverage impossibility. |
| `quantile-regression` | intermediate | Koenker–Bassett 1978 check-loss regression; Bahadur asymptotics at covariate values; quantile crossing. |
| `rank-tests` | intermediate | Wilcoxon / Mann–Whitney / Kruskal–Wallis; permutation distributions; Pitman ARE; Hodges–Lehmann estimator. |
| `extreme-value-theory` | advanced | Fisher–Tippett–Gnedenko trichotomy; GEV / GPD families; Hill / Pickands estimators; return levels; tail-risk quantification. |
| `statistical-depth` | advanced | Tukey / Mahalanobis / halfspace / simplicial / spatial depth; Zuo–Serfling axioms; DD-plots; depth regions as multivariate quantile contours. |
| `prediction-intervals` | intermediate | Frequentist / Bayesian / conformal / quantile-regression prediction-interval construction; bootstrap PIs; miscoverage analysis. Dedicated topic per §7.2. |

**Placement rationale.** Six topics are right at the top of the comfort range but still coherent. All six share the "inference without parametric assumptions" narrative that distinguishes them from T2 (which is about *estimation*, typically with explicit parametric structure). The original five-forward-topics brief's internal cross-links (CQR ↔ conformal, depth ↔ conformal for multi-output) remain valid; `prediction-intervals` is the natural sixth member.

### §3.5 T5 — Bayesian & Probabilistic ML (13 topics)

**Scope.** The Bayesian toolkit scaled up to modern ML regimes: variational methods, neural-network posteriors, Gaussian processes, probabilistic programming, specialized MCMC variants, ensemble methods.

| Slug | Difficulty | One-line scope |
|---|---|---|
| `variational-inference` | intermediate | ELBO, mean-field VI, structured VI, stochastic VI, normalizing-flow variational posteriors. |
| `variational-bayes-for-model-selection` | advanced | ELBO as log-marginal-likelihood approximation; variational-Bayes model comparison; KL projection biases. |
| `bayesian-neural-networks` | advanced | Weight-space posteriors via MC-dropout, deep ensembles, SG-MCMC, Laplace approximation; BNN calibration. |
| `gaussian-processes` | intermediate | GP prior over functions; conditional-MVN posterior; kernel design; GP classification via Laplace / EP. |
| `probabilistic-programming` | intermediate | Stan / PyMC / NumPyro compiler stack; automatic differentiation; constrained-to-unconstrained transforms; NUTS integration. |
| `sparse-bayesian-priors` | advanced | Horseshoe, regularized horseshoe, spike-and-slab, R2-D2; adaptive sparsity at prior level; HMC-friendly reparameterizations. |
| `meta-learning` | advanced | MAML, Neural Processes, hierarchical Bayes over task distributions; few-shot adaptation. |
| `mixed-effects` | intermediate | REML, frequentist GLMMs (`lme4`), Bayesian GLMMs (Stan), random-effects structures in recommendation systems. |
| `stacking-and-predictive-ensembles` | intermediate | Wolpert 1992 / Yao et al. 2018 stacking as M-open alternative to BMA; predictive-weight optimization. |
| `stochastic-gradient-mcmc` | advanced | SGLD, SGHMC, CSGLD; minibatch gradient + injected noise; bias-variance trade-off; Bayesian deep learning application. |
| `sequential-monte-carlo` | advanced | Particle filters; importance sampling + resampling; state-space models; annealed-SMC for marginal-likelihood estimation. |
| `reversible-jump-mcmc` | advanced | Green 1995 trans-dimensional MCMC; Jacobian corrections for dimension-changing moves; variable-selection and mixture-model applications. |
| `riemann-manifold-hmc` | advanced | Girolami–Calderhead 2011; position-dependent mass matrix via Fisher information; pathological-posterior remediation. |

**Placement rationale.** All 13 extend the Bayesian machinery of Topics 25/26/27/28 in the direction of "Bayes at scale" or "Bayes in modern ML." Gaussian processes live here rather than in T2 because the Bayesian framing (prior over functions, conditional posterior, hyperparameter inference) is constitutive rather than incidental.

### §3.6 T5 size caveat and optional split

Thirteen topics are larger than any existing formalML track (current maximum: 7 in Topology & TDA). Two options:

- **Keep as one track** — all 13 share the Bayesian foundation and the track functions as a coherent "modern Bayesian ML" unit. Potential confusion: the track is less navigable on `/paths` than smaller tracks. This is the recommended default.
- **Split into two sub-tracks** — "Bayesian ML Core" (VI, BNNs, GPs, PPL, sparse priors, meta-learning, mixed-effects) and "Advanced Bayesian Computation" (VBMS, stacking, SG-MCMC, SMC, RJ-MCMC, Riemann HMC). This gets the size per track down to 7 and 6, respectively, matching existing norms. The downside is a less-clean narrative split — these two sub-tracks genuinely interleave rather than cleanly separate.

**Recommendation:** Defer this decision until T5 reaches ~6 published topics, and the right split becomes observable from usage patterns. For the strategic-planning stage, treat T5 as one 13-topic track.

### §3.7 T6 — Learning Theory & Methodology (7 topics)

**Scope.** Generalization theory, semiparametric efficiency, causal inference, and uncertainty quantification as unifying methodological frameworks.

| Slug | Difficulty | One-line scope |
|---|---|---|
| `generalization-bounds` | intermediate | PAC framework beyond Topic 32 §32.4; Rademacher complexity; uniform convergence of risk; empirical Rademacher bounds. |
| `pac-bayes-bounds` | advanced | McAllester 1999, Catoni; posterior-over-hypotheses framework; KL-between-posterior-and-prior penalty; empirical-PAC-Bayes tightness. |
| `vc-dimension` | intermediate | Vapnik–Chervonenkis theory; shattering; Sauer–Shelah lemma; structural risk minimization. |
| `semiparametric-inference` | advanced | Functional delta method at scale; tangent spaces; efficient-influence functions; one-step estimators; TMLE. |
| `causal-inference-methods` | advanced | IPW, AIPW, doubly-robust estimators; targeted maximum likelihood; front-door / back-door / instrumental-variables identification. |
| `double-descent` | advanced | Overparameterized regimes violating Topic 23 assumptions; implicit regularization; minimum-norm interpolation; NTK perspective. |
| `uncertainty-quantification` | intermediate | Epistemic vs aleatoric uncertainty; calibration metrics; predictive-distribution diagnostics; UQ across models (conformal, Bayesian, ensembles). |

**Placement rationale.** The seven topics all sit at the "methodology that cuts across model families" layer — none of them is specific to a particular estimator or architecture; all of them are ways to reason about estimators or architectures. This gives T6 a clear identity distinct from T2/T3/T5.

---

## § 4 First-Wave Sequencing Recommendation

### §4.1 Sequencing principles

Three principles drive the recommended order:

1. **Discharge density first.** Topics with higher density (e.g., `variational-inference` = 3, `high-dimensional-regression` = 3) retire more formalstatistics link-debt per release. Early shipment flips more `(forthcoming)` markers to live `<ExternalLink>` components.
2. **Prerequisite independence.** Topics with no formalML-internal prerequisites can ship in any order; topics with strong internal dependencies (e.g., `local-regression` after `kernel-regression`) must respect those.
3. **ML-practitioner traction.** Given limited authoring bandwidth, prioritize topics that maximize inbound traffic and practitioner utility (conformal prediction, VI, BNNs, GPs, high-dim regression all score high here).

### §4.2 Recommended first-wave order (8 topics, ~6 months at current cadence)

1. **`conformal-prediction`** (T4) — Density 2, highest practitioner traction, smallest prerequisite load. Retires Topic 29 Rem 21 and Topic 32 pointer. Opens T4 and demonstrates the cross-site architecture for subsequent topics.
2. **`kernel-regression`** (T2) — Density 1, but direct extension of Topic 30 material. Opens T2 and establishes kernel-methods shared-module foundations for `local-regression` and `density-ratio-estimation` to build on.
3. **`variational-inference`** (T5) — Density 3, the highest-payoff topic in the inventory. Opens T5 and retires three distinct formalstatistics Track 7 pointers in one shipment.
4. **`gaussian-processes`** (T5) — Density 1 but foundational for Bayesian ML; establishes GP shared modules used by meta-learning and Bayesian-optimization applications later.
5. **`high-dimensional-regression`** (T2) — Density 3, retires Topic 22 Rem 28, Topic 23 Rem 29, and (via absorption) the engineering `logistic-regression` redirect pointer. Opens T2's advanced content.
6. **`quantile-regression`** (T4) — Density 1, natural successor to `conformal-prediction` via CQR cross-reference. Retires Topic 29 Rem 19.
7. **`bayesian-neural-networks`** (T5) — Density 2, the T5 flagship. Depends on `variational-inference` (approximate posteriors) and `gaussian-processes` (infinite-width limit). Retires Topic 25 and Topic 28 BNN pointers.
8. **`generalization-bounds`** (T6) — Density 2, opens T6. Retires Topic 24 Rem 30 and Topic 32 pointer. Natural companion to `pac-bayes-bounds` as the T6 opener pair.

After this wave, formalML has 8 of 32 new topics live, 5 of 6 new tracks opened (only T3 Unsupervised & Generative remains untouched), and 11 of 32 distinct formalstatistics forward-pointer slugs flipped to live cross-site links.

### §4.3 Second-wave candidates (topics 9–16)

`local-regression` (T2), `probabilistic-programming` (T5), `rank-tests` (T4), `causal-inference-methods` (T6), `sparse-bayesian-priors` (T5), `normalizing-flows` (T3 opener), `vc-dimension` (T6), `mixed-effects` (T5). Rough ordering by a blend of density and prerequisite-chain considerations.

### §4.4 Tail-wave notes

Several advanced topics (`riemann-manifold-hmc`, `reversible-jump-mcmc`, `double-descent`, `meta-learning`, `statistical-depth`, `semiparametric-inference`) are intentionally not included in the first wave, despite each having its own claim to priority. They tend to require more published-topic infrastructure before they can land coherently — `meta-learning` needs `bayesian-neural-networks`, `riemann-manifold-hmc` needs `variational-inference` and `bayesian-neural-networks` as context, etc.

---

## § 5 Cross-Site Architecture

### §5.1 The three-site mesh

formalML sits at the top of a three-site mesh:

```
formalcalculus → formalstatistics → formalML
  (foundational)   (inferential)     (methodological)
```

Cross-site pointers in each of the 32 new formalML topics will flow in three directions:

- **Backward to formalstatistics** — every topic has multiple formalstatistics prerequisites. These are full-URL hyperlinks: `[Topic 30 (Kernel Density Estimation)](https://formalstatistics.com/topics/kernel-density-estimation)`.
- **Backward to formalcalculus** — Jacobians, gradients, convex optimization, variational calculus, ODEs. Full-URL hyperlinks: `[Jacobians](https://formalcalculus.com/topics/jacobians)`.
- **Internal to formalML** — cross-references among the 32 new topics and the existing 35. Relative paths: `[PAC Learning](/topics/pac-learning)`.

### §5.2 Cross-site reference schema (revised against PR #52 production state)

The original Claude Chat draft of this section proposed adding a single `crossSitePrereqs` array to nodes in `curriculum-graph.json`. **That proposal is superseded.** The actual schema, shipped via formalML PR #52 and 100% adopted on formalstatistics + formalcalculus, lives in MDX frontmatter (not in the curriculum graph) and uses **six separate fields** parameterized by target-site × direction:

| Field | Direction | Use on formalML topics |
|---|---|---|
| `formalcalculusPrereqs` | backward | Calculus topics this ML topic requires |
| `formalstatisticsPrereqs` | backward | Statistics topics this ML topic requires |
| `formalcalculusConnections` | forward | Calculus topics this ML topic informs (rare) |
| `formalstatisticsConnections` | forward | Statistics topics this ML topic informs (rare) |
| `formalmlPrereqs` / `formalmlConnections` | self | Do **not** use on formalML topics — flagged as `self-site` by the audit |

Per-entry shape (mirrors the formalstatistics convention):

```yaml
formalstatisticsPrereqs:
  - topic: "bayesian-foundations-and-prior-selection"
    site: "formalstatistics"
    relationship: "Topic 25's conjugate-prior framework is the substrate for VI's
                   variational family choice; see §25.4 Thm 3 for the exponential-
                   family conjugacy that this topic specializes in §3."
  - topic: "bayesian-computation-and-mcmc"
    site: "formalstatistics"
    relationship: "Topic 26 §26.5 establishes the NUTS sampler used as the
                   reference-implementation comparison for amortized VI in §6."
formalcalculusPrereqs:
  - topic: "lebesgue-integral"
    site: "formalcalculus"
    relationship: "The ELBO integral over q(z) requires Lebesgue integration over
                   the latent space; see §4 derivation."
```

**Validation.** `pnpm audit:cross-site` (script: `scripts/audit-cross-site-links.mjs`) walks every `.mdx` in all three sibling repos, extracts these fields, and checks that every `A.x → B.y` edge has a reciprocal `B.y → A.x` with the opposite direction (`Prereqs ↔ Connections`). Outputs:

- [docs/plans/cross-site-audit-report.md](cross-site-audit-report.md) — consolidated reciprocity report (slug drift, missing reciprocals, deferred targets, frontmatter quality flags)
- [docs/plans/deferred-reciprocals.md](deferred-reciprocals.md) — log of edges pointing at slugs that don't yet exist on the target repo, organized by "When `<repo>/<slug>` ships" with source-side prose stubs
- `docs/plans/audit-output/<site>-references.json` — per-site edge dumps

**Curriculum-graph note.** `src/data/curriculum-graph.json` continues to encode internal-only edges; cross-site relationships do **not** belong there. The `/paths` page can choose to surface cross-site prereqs by reading the MDX frontmatter at build time — that's a future UI decision (see §18). Shipping cross-site rendering on `/paths` is **not** a prerequisite for the first new topic, since cross-site infrastructure is already in place at the data layer.

### §5.3 Forward-pointer convention (internal forthcoming)

Within formalML, when a new topic references another planned-but-not-yet-published new topic, use the existing `(coming soon)` plain-text convention per `formalML-handoff-reference.md` §3:

```mdx
**Tensor Decompositions** *(coming soon)* generalize PCA to multi-way arrays.
```

No link, no 404 risk. When the referenced topic ships, subsequent authors convert plain-text references to Markdown links.

### §5.4 Handling the two redirect decisions

Per §2.4, two formalstatistics forward-pointers to formalML do NOT get their own formalML topic: `logistic-regression` (engineering flavor) and `generalized-linear-models` (engineering flavor).

- **`logistic-regression` redirects to `high-dimensional-regression`.** When Topic 22 is eventually revised, its `formalmlConnections` entry for `logistic-regression` should be replaced with an entry pointing to `high-dimensional-regression` (with slightly softened relationship prose). The cross-site link becomes live when `high-dimensional-regression` ships. Until then, the pointer can stay as `(forthcoming)` plain text with no change.
- **`generalized-linear-models` redirects to an external reference.** When Topic 22 is eventually revised, its `formalmlConnections` entry for `generalized-linear-models` should be removed, and the relevant Topic 22 §22.10 remark should be rewritten to cite Hastie–Tibshirani–Friedman 2009 Ch. 4 (or Agresti 2015 Ch. 6–7 for multinomial / ordinal / Tweedie extensions) as the external reference.

**Action item for Jonathan:** Note these two redirects as pending formalstatistics PR work. Not urgent — the plain-text `(forthcoming)` markers remain valid indefinitely. Worth a short PR when `high-dimensional-regression` ships and the full redirect story becomes resolvable in one pass.

### §5.5 Reciprocity workflow at ship time (revised against PR #52 production state)

The original Claude Chat draft of this section described a manual protocol of flipping formalstatistics plain-text `(forthcoming)` markers into live `<ExternalLink>` components and adding `url` fields to `formalMLConnections` entries. **That protocol is superseded** by the audit-script workflow that shipped in PR #52.

Reality at publication time:

- formalstatistics already declares **`formalmlConnections`** entries (with `topic`, `site`, `relationship`) for every forward-pointer. There are no `formalMLConnections` (capital ML) fields and never were — the canonical name is lowercase `formalml`.
- formalstatistics also embeds inline `<ExternalLink site="formalml" ...>` components in MDX prose for in-flow mentions; these do **not** require flipping at ship time and may render as 404s today (see §5.6 for the slug-drift cleanup).
- The `pnpm audit:cross-site` script catches every formalML slug that any sister site points at and lacks a reciprocal — that list lives in [docs/plans/deferred-reciprocals.md](deferred-reciprocals.md), regenerated on every audit run.

**Workflow when a new formalML topic ships:**

1. **Frontmatter, formalML side.** The new topic's MDX includes `formalcalculusPrereqs` and/or `formalstatisticsPrereqs` arrays for every cross-site prereq, each entry with `topic`, `site`, `relationship` (≥40 chars).
2. **Pull the deferred-reciprocal entries.** Search [deferred-reciprocals.md](deferred-reciprocals.md) for the `When formalml/<slug> ships` heading. Every bullet under that heading is a sister-site topic that needs a reciprocal entry.
3. **Sister-site PR.** Open a PR on formalstatistics (and/or formalcalculus) that adds a `formalstatisticsConnections` (or `formalcalculusConnections`) entry to each source topic listed in step 2, copying the source-side prose stub from `deferred-reciprocals.md` and rewriting from the new topic's vantage. Confirm `relationship` is ≥40 chars.
4. **Re-audit.** Run `pnpm audit:cross-site` (from any of the three repos with sibling paths configured). The deferred-reciprocal entry should disappear; the audit's `Reciprocated` count should increment by the number of source topics; no new `Missing reciprocals` should appear.
5. **Inline body links (optional).** If the new formalML topic mentions a sister-site topic in prose, port the formalstatistics `<ExternalLink>` component to formalML (`src/components/ui/ExternalLink.astro`) and use it. Build is one-time; reuse for all subsequent topics.

A single formalML topic may discharge anywhere from 1 to 9+ deferred-reciprocal entries on the sister sites, depending on how many sister-site topics named it as a forward pointer. The numbers are visible in [deferred-reciprocals.md](deferred-reciprocals.md) at the moment of ship.

Appendix A retains the original Claude Chat per-pointer map for historical reference and as a sanity check against the audit's deferred-reciprocal list, but `pnpm audit:cross-site` is the canonical source of truth.

### §5.6 Inventory reconciliation against deferred-reciprocals.md (new — 2026-04-25)

When `pnpm audit:cross-site` was run after PR #52, [deferred-reciprocals.md](deferred-reciprocals.md) listed **~90 unique `formalml/<slug>` targets** that the sister sites already point at. The 32-topic inventory in §2.2 of this document is a strict subset of that surface. The remaining ~58 deferred-reciprocal slugs fall into one of four buckets:

| Bucket | Count | What to do | Examples |
|---|---:|---|---|
| **A. In strategic doc inventory (will discharge when topic ships)** | 18 | None — covered by §4 sequencing. Audit will retire these as topics ship. | `bayesian-neural-networks`, `causal-inference-methods`, `gaussian-processes`, `generalization-bounds`, `high-dimensional-regression`, `meta-learning`, `mixed-effects`, `normalizing-flows`, `probabilistic-programming`, `quantile-regression`, `reversible-jump-mcmc`, `riemann-manifold-hmc`, `sequential-monte-carlo`, `sparse-bayesian-priors`, `stochastic-gradient-mcmc`, `variational-inference`, `conformal-prediction`, `prediction-intervals` |
| **B. Already-published formalML topics (sister-site reciprocal needed now)** | ~6 | Open a sister-site PR adding the reciprocal entry now — no formalML work needed. | `bayesian-nonparametrics`, `information-geometry`, `pac-learning`, `measure-theoretic-probability`, `shannon-entropy`, `minimum-description-length` |
| **C. Slug drift / wrong-name pointers from sister sites** | ~6 | Sister-site PR to fix the slug. Formalstatistics points at `principal-component-analysis` (formalML has `pca-low-rank`); `variational-methods` (formalML inventory: `variational-inference`); `stochastic-gradient-descent` (formalML has `gradient-descent`). The audit's "Slug drift" section names each. | `principal-component-analysis` → `pca-low-rank`; `variational-methods` → `variational-inference`; `stochastic-gradient-descent` → `gradient-descent` |
| **D. Out-of-inventory slugs pointed at but unplanned** | ~28 | Decision per slug: (a) fold into an inventory topic as a named section; (b) redirect the sister-site pointer per §5.4 / §7.3; (c) add to a future inventory expansion. None of these block first-wave authoring. | `ab-testing`, `ab-testing-platforms`, `always-valid-inference`, `bayesian-inference`, `bayesian-model-averaging`, `bias-variance-tradeoff`, `cross-entropy-loss`, `cross-fitting`, `cross-validation`, `differential-privacy`, `diffusion-models`, `dimensionality-reduction`, `embedding-spaces`, `empirical-risk-minimization`, `ensemble-methods`, `expectation-maximization`, `feature-engineering`, `feature-selection-and-multiplicity`, `fourier-neural-operators`, `generalized-method-of-moments`, `generative-modeling`, `graphical-models`, `high-dimensional-testing-knockoffs`, `importance-sampling`, `information-bottleneck`, `kernel-methods`, `metric-learning`, `model-comparison`, `monte-carlo-methods`, `naive-bayes`, `online-fdr`, `optimization-theory`, `post-selection-inference`, `probability-spaces`, `regression`, `regularization`, `reinforcement-learning`, `representation-learning`, `robust-statistics`, `score-matching`, `spectral-methods`, `statistical-learning-theory`, `structural-risk-minimization`, `wasserstein-distances`, `weight-decay` |

**Bucket-specific notes:**

- **Bucket B — already-published reciprocals.** These are zero-cost edits and should ship in the next sister-site PR sweep, regardless of formalML topic cadence. Confirm the formalML topics are live, then add the reciprocal `formalstatisticsConnections` / `formalcalculusConnections` entry to each source topic listed in `deferred-reciprocals.md` for that slug.
- **Bucket C — slug drift.** The audit's "Slug drift" section in [cross-site-audit-report.md](cross-site-audit-report.md) suggests the right target slug for each. These should be fixed on the sister-site repos in a single sweep PR.
- **Bucket D — unplanned slugs.** Several of these correspond to §7.3 named-section redirects (`cross-validation`, `ab-testing`, `weight-decay`). Several others (`monte-carlo-methods`, `bayesian-inference`, `regularization`, `optimization-theory`) are concepts that probably live within an inventory topic as a named section. A handful (`reinforcement-learning`, `diffusion-models`, `fourier-neural-operators`, `generative-modeling`, `wasserstein-distances`) are large enough to justify their own future topics — candidates for an inventory expansion in a v2 of this document. None are urgent.

**Action item.** A separate session should triage Bucket D — for each slug, pick (a) named-section redirect, (b) inventory addition, or (c) sister-site pointer removal. That triage is out of scope for v1 of this document but should precede the second-wave sequencing in §4.3.

---

## § 6 Shared-Module Architecture

### §6.1 Module organization principle

formalML's `src/components/viz/shared/` currently holds cross-cutting utilities (`useD3.ts`, `useResizeObserver.ts`, `colorScales.ts`, `types.ts`). For the 32 new topics, each track introduces track-specific computational primitives that benefit from a shared module pattern matching formalstatistics precedent (Topic 29 introduced `nonparametric.ts`, Topic 30 extended it, etc.).

**Rule:** each track's first-shipped topic introduces its shared module; subsequent topics in the same track extend that module rather than creating siblings.

### §6.2 Proposed shared modules by track

| Track | Module | Introduced by | Later extended by |
|---|---|---|---|
| T2 Supervised | `supervised-learning.ts` | `kernel-regression` | `local-regression`, `high-dimensional-regression` |
| T3 Unsupervised & Generative | `unsupervised.ts` | `clustering` | `density-ratio-estimation`, `normalizing-flows` |
| T4 Nonparametric & Distribution-Free | `nonparametric-ml.ts` | `conformal-prediction` | `quantile-regression`, `rank-tests`, `extreme-value-theory`, `statistical-depth`, `prediction-intervals` |
| T5 Bayesian & Probabilistic ML | `bayesian-ml.ts` | `variational-inference` | 12 others in T5 |
| T6 Learning Theory & Methodology | `learning-theory.ts` | `generalization-bounds` | `pac-bayes-bounds`, `vc-dimension`, `semiparametric-inference`, `causal-inference-methods`, `double-descent`, `uncertainty-quantification` |

### §6.3 Cross-track shared primitives

Some primitives are genuinely cross-track and should live in `shared/` proper rather than in a track module:

- **`multivariate.ts`** — multivariate Normal density, MVN sampling, covariance matrix parameterizations. Used by T3 (normalizing-flows), T4 (statistical-depth), T5 (gaussian-processes, bayesian-neural-networks).
- **`information-theory.ts`** — KL divergence, entropy, mutual information, f-divergences. Used by T3 (density-ratio-estimation), T5 (variational-inference, variational-bayes-for-model-selection), T6 (uncertainty-quantification).
- **`optimization.ts`** — projected gradient, ADMM, coordinate descent helpers. Used by T2 (high-dimensional-regression), T3 (density-ratio-estimation), T5 (variational-inference), T6 (causal-inference-methods).

Introduce each cross-track module when the first topic that needs it ships. Don't pre-introduce.

### §6.4 TypeScript conventions for shared modules

Follow formalstatistics precedent and the current formalML patterns per `formalML-handoff-reference.md` §4:

- Explicit types on every export; JSDoc on every public function.
- Matching `.test.ts` in the same directory with expected values printed by a verification notebook cell.
- Lazy initialization for expensive computations (`getX()` pattern, not module-level `const x = compute()`).
- No global state; all module exports are pure functions or factories.

### §6.5 Appendix C exports

The full list of proposed shared-module exports per topic is in Appendix C. For each topic's handoff brief, the exports list gets refined based on per-topic content planning; Appendix C is the strawman for planning-brief authors to work against, not the final spec.

---

## § 7 Slug & Naming Conventions

### §7.1 Slug discipline (carried from handoff reference)

Every new slug follows the short-filename pattern per `formalML-handoff-reference.md` §2. Specifically avoided antipatterns:

| Would-be slug | Final slug | Reason |
|---|---|---|
| `extreme-value-theory-and-tail-asymptotics` | `extreme-value-theory` | Length |
| `density-ratio-estimation-methods` | `density-ratio-estimation` | "methods" is noise |
| `generalized-linear-models-for-machine-learning` | N/A — redirected out | Per §5.4 |
| `bayesian-neural-networks-and-posterior-approximation` | `bayesian-neural-networks` | Trailing qualifier |
| `stacking-and-predictive-ensembles-vs-bma` | `stacking-and-predictive-ensembles` | No vs-clauses |

All 32 slugs in §2.2 are locked by this document.

### §7.2 Merge policy — less aggressive (per Jonathan's Q1 answer)

Where two forward-pointer targets are mathematically distinct, they get distinct topics:

- `variational-inference` ≠ `variational-bayes-for-model-selection` — VI is a computational framework; VBMS is a model-comparison criterion built on top. Distinct mathematical content (ELBO bound tightness vs. KL-projection bias for log-marginal-likelihood estimation).
- `generalization-bounds` ≠ `pac-bayes-bounds` ≠ `vc-dimension` — three distinct theoretical frameworks. VC theory handles concept-class complexity; Rademacher/uniform-convergence bounds generalize to real-valued functions; PAC-Bayes handles posterior-over-hypotheses with KL penalties. Three topics.
- `conformal-prediction` ≠ `prediction-intervals` ≠ `uncertainty-quantification` — conformal is a specific finite-sample-coverage procedure; prediction-intervals is the umbrella topic covering frequentist + Bayesian + conformal + quantile-regression PI constructions; UQ is the broader methodology for reasoning about estimator confidence across model families. Three topics.

### §7.3 Named-section policy (per Jonathan's Q2 answer)

Three ML-fundamentals that formalstatistics forward-pointed but are better positioned as named sections rather than standalone topics:

- **`cross-validation`** — appears as a named section within at least two topics (`conformal-prediction` §3.8 for CV+ variants; `high-dimensional-regression` §9 for $\lambda$-selection). The topic-slug `cross-validation` never gets minted; formalstatistics pointers to it get redirected to the appropriate host topic.
- **`ab-testing`** — a named section within `rank-tests` (permutation-based AB testing) and within `prediction-intervals` (sample-size planning for confidence-interval coverage on conversion-rate diffs). Topic-slug never minted.
- **`weight-decay`** — a named section within `bayesian-neural-networks` (Gaussian prior = weight decay) and within `high-dimensional-regression` (ridge-regression connection, MAP framing). Topic-slug never minted.

When Topic 31 or Topic 23 pointers to these slugs are eventually updated in formalstatistics, they should redirect to the host topic using a `#section-N` anchor rather than to the missing slug.

### §7.4 Domain keys for curriculum-graph.json

The current `curriculum-graph.json` uses 8 domain keys per `formalML-handoff-reference.md` §6: `topology`, `linear-algebra`, `probability`, `optimization`, `geometry`, `information-theory`, `graph-theory`, `category-theory`.

The 5 new tracks need 5 new domain keys. Recommended:

- T2 Supervised Learning → `supervised-learning`
- T3 Unsupervised & Generative → `unsupervised`
- T4 Nonparametric & Distribution-Free → `nonparametric-ml`
- T5 Bayesian & Probabilistic ML → `bayesian-ml`
- T6 Learning Theory & Methodology → `learning-theory`

These match the shared-module names from §6.2 for discoverability.

---

## § 8 Code-Example Language Policy

### §8.1 Default: Python + NumPy / SciPy

Consistent with formalstatistics precedent and existing formalML notebooks. Every topic's accompanying notebook is Python; every inline code snippet in the MDX is Python. Libraries in the default stack:

- `numpy`, `scipy`, `pandas` for numerical and data handling
- `matplotlib` for figure generation that feeds into `/public/images/topics/`
- `statsmodels` for classical statistical fits where available
- `scikit-learn` for standard ML primitives when the topic's content doesn't hinge on hand-coding the method

### §8.2 PyTorch / JAX allowance

Deep-learning-adjacent topics may introduce PyTorch or JAX in notebook code and in bridge-to-production inline code snippets. The allowed set:

| Topic | Library | Use case |
|---|---|---|
| `normalizing-flows` | PyTorch | Coupling / autoregressive / Neural-ODE flow implementations |
| `bayesian-neural-networks` | PyTorch | BNN training loops, MC-dropout, deep ensemble training |
| `meta-learning` | PyTorch | MAML / Reptile / Neural Process implementations |
| `stochastic-gradient-mcmc` | PyTorch | SGLD / SGHMC training loops with PyTorch optimizers |
| `variational-inference` | PyTorch or JAX (author's choice) | Amortized VI, normalizing-flow posteriors |
| `density-ratio-estimation` (neural DRE section only) | PyTorch | Logistic-regression DRE, neural-DRE baseline |
| `probabilistic-programming` | Stan / PyMC / NumPyro examples | These are the subject matter, not the implementation |

All other topics use Python + NumPy / SciPy / scikit-learn. If a topic's handoff brief proposes PyTorch / JAX for a case not in the above table, the deviation must be justified in the brief.

### §8.3 Notebook structure pattern

Notebooks live at `notebooks/<slug>/<NN>_<slug_with_underscores>.ipynb` per `formalML-handoff-reference.md` §2. Each notebook has four standard cells:

1. **Setup** — imports, `np.random.seed(42)`, shared plotting helpers.
2. **Theory verification** — numerical verification of the topic's theorems (e.g., coverage guarantees, bias-variance formulas) to reproducible 3-digit precision.
3. **Component data generation** — produces the JSON / CSV files that get embedded into the React viz components. Outputs saved to `src/data/<slug>-data.json` or similar.
4. **Figure generation** — produces `/public/images/topics/<slug>/*.png` via matplotlib.

For PyTorch/JAX-allowed topics, an optional cell 5 provides a demonstration of the deep learning implementation. Training must be CPU-runnable in under 60 seconds on a 2020-era laptop; no GPU requirements.

### §8.4 Import convention

At the top of every MDX body (after frontmatter `---`), after the standard component imports (`TheoremBlock`, viz components), notebook code is referenced but not included inline. Inline code snippets in MDX prose are short (≤ 20 lines) and always Python unless §8.2 allows otherwise.

---

## § 9 Amendments to CLAUDE.md

This section lists changes to apply to the current `/mnt/project/CLAUDE.md`. Apply as a single commit once this strategic document is accepted.

### §9.1 Track-structure amendment

Add a "Curriculum Architecture" section after "Project Overview" specifying the 13-track scaffold:

```markdown
## Curriculum Architecture

formalML has two layers:

**Foundations layer (8 tracks, 35 topics — feature-complete):**
- Topology & TDA, Linear Algebra, Probability & Statistics, Optimization,
  Differential Geometry, Information Theory, Graph Theory, Category Theory

**ML Methodology layer (5 tracks, 32 topics planned):**
- T2 Supervised Learning (3 topics)
- T3 Unsupervised & Generative (3 topics)
- T4 Nonparametric & Distribution-Free (6 topics)
- T5 Bayesian & Probabilistic ML (13 topics)
- T6 Learning Theory & Methodology (7 topics)

See `formalML-consolidated-strategic-planning-document.md` for the full
inventory and rationale.
```

### §9.2 Cross-site reference conventions

Add to the "Content Conventions" section:

```markdown
### Cross-site references

formalML is the third site in the triad (formalCalculus → formalStatistics → formalML).
Cross-site pointers appear in three directions:

- **Back to formalStatistics** — full-URL hyperlinks: `[Topic 30 (KDE)](https://formalstatistics.com/topics/kernel-density-estimation)`.
- **Back to formalCalculus** — full-URL hyperlinks: `[Jacobians](https://formalcalculus.com/topics/jacobians)`.
- **Forward within formalML** — relative paths: `[PAC Learning](/topics/pac-learning)`.

For planned-but-not-yet-published formalML topics, use plain text:
`**Variational Inference** *(coming soon)*`.

Cross-site prerequisites live in MDX frontmatter using the six-field schema
(`formalcalculusPrereqs`, `formalstatisticsPrereqs`, etc.), per the schema
in the strategic planning document §5.2. Validated by `pnpm audit:cross-site`.
```

### §9.3 Code-example language policy

Add to "Code Style":

```markdown
### Code-example language policy

Default: Python + NumPy / SciPy / scikit-learn / pandas / matplotlib.

PyTorch / JAX allowed for seven named topics (see strategic planning document §8.2):
normalizing-flows, bayesian-neural-networks, meta-learning, stochastic-gradient-mcmc,
variational-inference, density-ratio-estimation (neural DRE section), probabilistic-programming.

All other topics stay in the default stack.
```

### §9.4 "Do NOT" additions

Add to the existing Do-NOT list:

```markdown
- Create formalML topics for engineering duplicates of formalstatistics coverage
  (logistic-regression, generalized-linear-models engineering flavors) — redirect
  per strategic planning document §5.4
- Create standalone topics for cross-validation, ab-testing, weight-decay — these
  are named sections within host topics per strategic planning document §7.3
- Place Gaussian processes in T2 Supervised Learning — they belong in T5 Bayesian ML
```

---

## § 10 Amendments to formalML-handoff-reference.md

### §10.1 Published topics table

Section 5 of `formalML-handoff-reference.md` lists current published topics. When each new topic ships, add its row to the appropriate track table (creating new table sections for T2–T6 as the tracks open).

Specifically, add five new section headers that match the domain keys from §7.4:

```markdown
### Supervised Learning (3 topics planned)
| Slug | Title | Difficulty |
|------|-------|------------|
| `kernel-regression` | Kernel Regression (Nadaraya–Watson) | intermediate |
...

### Unsupervised & Generative (3 topics planned)
...

### Nonparametric & Distribution-Free (6 topics planned)
...

### Bayesian & Probabilistic ML (13 topics planned)
...

### Learning Theory & Methodology (7 topics planned)
...
```

### §10.2 Domain keys section

Section 6 ("Curriculum Graph Updates") currently lists 8 domain keys. Update to list all 13:

```markdown
### Domain keys (exhaustive list)

Foundations layer: `topology`, `linear-algebra`, `probability`, `optimization`,
`geometry`, `information-theory`, `graph-theory`, `category-theory`.

ML Methodology layer: `supervised-learning`, `unsupervised`, `nonparametric-ml`,
`bayesian-ml`, `learning-theory`.
```

### §10.3 Cross-site references section (new)

Add a new Section 10 (after the current "Editorial Voice" section):

```markdown
## 10. Cross-Site References

Every formalML topic has backward prerequisites on formalStatistics and/or
formalCalculus. Specify these in handoff briefs under a new "Cross-Site
Prerequisites" subsection. Use full URLs.

### Prerequisite schema in MDX frontmatter

```yaml
formalstatisticsPrereqs:
  - topic: "bayesian-foundations-and-prior-selection"
    site: "formalstatistics"
    relationship: "Topic 25's conjugate-prior framework is the substrate for VI's
                   variational family choice."
formalcalculusPrereqs:
  - topic: "functional-derivatives"
    site: "formalcalculus"
    relationship: "The ELBO gradient in §4 is a functional derivative in q."
```

### Reciprocity workflow at ship time

When a formalML topic ships, search [`docs/plans/deferred-reciprocals.md`](deferred-reciprocals.md)
for the new topic's slug under "When `formalml/<slug>` ships" headings. Each entry
is a sister-site topic that needs a reciprocal `formalstatisticsConnections` /
`formalcalculusConnections` entry added (with prose stub provided in the deferred
log). Re-run `pnpm audit:cross-site` to confirm the deferred entry retires.
See strategic planning document §5.5 for the full workflow.
```

### §10.4 Error-log additions

Add to Section 8 ("Lessons from Past Briefs"):

```markdown
| Brief Section | What the brief said | What the codebase needed | Impact |
|--------------|-------------------|------------------------|--------|
| Cross-site prereq | Missing from frontmatter | `formalcalculusPrereqs` / `formalstatisticsPrereqs` (six-field schema) per strategic plan §5.2 | Missing backward link; audit flags as missing reciprocal |
| Track placement | `gaussian-processes` in T2 Supervised | T5 Bayesian per strategic plan §3.5 | Wrong navigation grouping |
| Standalone slug | `cross-validation` as own topic | Named section per strategic plan §7.3 | Duplicate content |
```

---

## § 11 Handoff-Brief Template Adjustments

Each per-topic handoff brief drafted after this strategic document inherits the following section additions.

### §11.1 Add "Cross-Site Prerequisites" section

After the existing "Prerequisites" section (which lists internal formalML prereqs), add:

```markdown
## Cross-Site Prerequisites

### formalstatistics
- [Topic NN — Topic Title](https://formalstatistics.com/topics/slug) — Why needed; which section primarily.

### formalcalculus
- [Topic — Title](https://formalcalculus.com/topics/slug) — Which derivative / integral / Jacobian.
```

This section must be filled in for every topic. Empty = explicit acknowledgment that no cross-site prereqs exist (rare but possible for T6's `uncertainty-quantification`, which could be self-contained on formalML).

### §11.2 Add "Reciprocity Sweep at Ship Time" section

Toward the end of the handoff brief, before "Open Questions":

```markdown
## Reciprocity Sweep at Ship Time

When this topic ships, the following sister-site MDX files need a reciprocal
`formalstatisticsConnections` / `formalcalculusConnections` entry added (see
[`docs/plans/deferred-reciprocals.md`](deferred-reciprocals.md) under the
"When `formalml/<slug>` ships" heading for the canonical list and source-side
prose stubs):

- `formalstatistics/src/content/topics/<source-slug>.mdx` — add reciprocal entry
  to its `formalmlConnections` field's predecessor counterpart on the source
  side: i.e., the source topic gains a `formalstatisticsConnections` reciprocal
  on the new formalML topic side, and the new topic's `formalstatisticsPrereqs`
  closes the reciprocal pair.
- ... (enumerate all source topics per `deferred-reciprocals.md` for this slug)

After edits: re-run `pnpm audit:cross-site` from any of the three repos with the
sibling paths configured (`FORMAL_CALCULUS_PATH`, `FORMAL_STATISTICS_PATH`).
The deferred entries for this slug should disappear, and `Reciprocated` count
should increment by the number of source topics. Typical sweep: 1–9 source
topics per formalML topic ship.
```

### §11.3 Decision-reference callbacks

Handoff briefs should not re-litigate decisions made in the strategic document. Instead, they reference:

- "Slug per strategic plan §2.2 — no discussion."
- "Track placement: T5 Bayesian & Probabilistic ML per strategic plan §3.5."
- "Code-example language: Python + PyTorch per strategic plan §8.2."
- "Shared module extension: extends `bayesian-ml.ts` per strategic plan §6.2."

This keeps per-topic briefs focused on content and prevents architectural drift.

---

## § 12 Pacing & Timeline Estimate

### §12.1 Per-topic authoring time

Based on formalstatistics Topic 30 and Topic 32 cadence (roughly 3–4 weeks per intermediate-difficulty topic, 5–8 weeks per advanced-difficulty topic in the narrow-scope case):

| Difficulty | Rough authoring time |
|---|---|
| intermediate | 3–5 weeks |
| advanced (narrow scope) | 5–7 weeks |
| advanced (broad scope) | 7–10 weeks |

Broad-scope advanced topics in this inventory: `bayesian-neural-networks`, `normalizing-flows`, `causal-inference-methods`, `high-dimensional-regression`, `meta-learning`, `extreme-value-theory`. These are the multi-month commitments.

### §12.2 First-wave timeline (8 topics)

Applying §12.1 to §4.2's first wave:

- `conformal-prediction` (intermediate, 4 weeks)
- `kernel-regression` (intermediate, 4 weeks)
- `variational-inference` (intermediate, 5 weeks)
- `gaussian-processes` (intermediate, 5 weeks)
- `high-dimensional-regression` (advanced broad, 8 weeks)
- `quantile-regression` (intermediate, 4 weeks)
- `bayesian-neural-networks` (advanced broad, 9 weeks)
- `generalization-bounds` (intermediate, 5 weeks)

Summing: ~44 weeks of authoring time for the first wave, realistically 9–12 months with other commitments.

### §12.3 Full buildout timeline

32 topics at a weighted average of ~5 weeks each = ~160 weeks of authoring. At 40–50% of working time going to formalML (the rest to other commitments), that's ~6–8 calendar years. Not a rush.

Realistic near-term target: **first 8 topics by mid-2027; first 16 by mid-2028; full buildout by 2030.**

### §12.4 Pacing does not need commitment now

The 32 topics are mostly independent; order can shift based on emerging ML trends, reader interest, or external milestones. The first-wave recommendation in §4.2 is a default; overrides at any point are fine. What this document locks in is not the *schedule* but the *architecture* — the slugs, the track placement, the cross-site schema, the shared-module plan.

**Timeline note (revised, 2026-04-25).** The §12.3 estimate ("first 8 by mid-2027; full buildout by 2030") was framed assuming formalstatistics and formalcalculus authoring continued to compete for working time. Both sister sites are now feature-complete (32 topics each). With that competing demand removed, the formalML cadence may compress meaningfully — possibly halving the elapsed-time estimate, depending on how much of the freed bandwidth flows back into formalML versus other commitments. The architectural decisions in this document are unaffected; the timeline is.

---

## § 13 Open Questions Deferred to Per-Topic Briefs

Not everything can be decided up front. The following per-topic decisions are explicitly deferred to each topic's handoff-brief stage:

- **Theorem-depth calls.** How many full proofs vs. stated-with-sketch? Depends on the topic's difficulty, the budget, and what feels rhetorically essential.
- **Component-count budget per topic.** Rough range 2–4 interactive components; exact count per topic negotiated in the handoff brief.
- **Figure-count budget per topic.** Rough range 6–10 figures; exact count per topic.
- **Notebook length.** Rough range 500–1500 lines per notebook.
- **Cross-reference density to other formalML topics.** Depends on how much of the strategic plan's internal-edge graph is live at ship time.

The one strategic-level question still pending is the T5 split (§3.6). Defer to T5's 6th-topic mark and revisit then.

---

## § 14 Recommendations for Jonathan

### §14.1 Immediate next step

Before the first new topic enters per-topic-handoff-brief drafting, apply the amendments from §§9–10 to `CLAUDE.md` and `formalML-handoff-reference.md`. One commit, ~15 minutes.

### §14.2 Before starting the handoff-brief for the first new topic

Decide which topic opens the first wave: `conformal-prediction` is recommended per §4.2, but not mandatory. If ML-practitioner traction is the priority, `conformal-prediction` or `variational-inference` (two near-tied winners). If architecture-first (open as many tracks as fast as possible), rotate through one-topic-per-new-track in the first 5 ships.

### §14.3 /paths page update

When any new topic ships, the `/paths` page needs to surface the new track. Recommended: update `/paths` once at the top of the first-wave (after `conformal-prediction` ships) to add the 5 new tracks as planned-content placeholders with 0 published / N planned badges. Then update counters as each topic ships.

The cross-site prereq rendering (§5.2) should also ship in this same /paths revision, because `conformal-prediction` will be the first topic with cross-site prereqs visible.

### §14.4 Don't over-plan the tail

Topics 9–32 (everything past the first wave) can be planned shallowly now and deeply when each approaches its handoff-brief stage. The strategic document's role is to lock architecture; content evolves.

### §14.5 Pacing is negotiable

If other commitments slow formalML work, the 32-topic target doesn't change — the timeline does. The `(forthcoming)` markers on formalstatistics are patient. No reader lands on Topic 29 §29.10 Rem 21 and urgently needs the conformal-prediction page to exist this quarter.

---

## § 15 Appendix A — formalstatistics → formalML Forward-Pointer Map

Every formalstatistics pointer to a formalML slug, with PR-sweep target location.

| formalstatistics topic | Location | Target slug | Edit at ship time |
|---|---|---|---|
| Topic 22 (GLMs) | §22.10 Rem 25 | `mixed-effects` | Plain text → `<ExternalLink>` |
| Topic 22 | §22.10 Rem 27 | `causal-inference-methods` | Plain text → `<ExternalLink>` |
| Topic 22 | §22.10 Rem 28 | `high-dimensional-regression` | Plain text → `<ExternalLink>` |
| Topic 22 | §22.4 Rem 9 + §22.10 Rem 26 | `logistic-regression` | REDIRECT: rewrite to point to `high-dimensional-regression` per §5.4 |
| Topic 22 | §22.10 Rem 26 | `generalized-linear-models` (engineering) | REDIRECT: external reference per §5.4 |
| Topic 23 (Regularization) | §23.10 Rem 28 | `weight-decay` | REDIRECT: section within `bayesian-neural-networks` or `high-dimensional-regression` per §7.3 |
| Topic 23 | §23.10 Rem 29 | `high-dimensional-regression` | Plain text → `<ExternalLink>` (same as Topic 22) |
| Topic 23 | §23.10 Rem 30 | `double-descent` | Plain text → `<ExternalLink>` |
| Topic 24 (Model Selection) | §24.10 Rem 30 | `vc-dimension` | Plain text → `<ExternalLink>` |
| Topic 24 | §24.10 Rem 30 | `generalization-bounds` | Plain text → `<ExternalLink>` |
| Topic 25 (Bayes Foundations) | §25.10 formalmlConnections | `bayesian-neural-networks` | Plain text → `<ExternalLink>`; url field confirmation |
| Topic 25 | §25.10 formalmlConnections | `variational-inference` | Plain text → `<ExternalLink>`; url field confirmation |
| Topic 25 | §25.10 formalmlConnections | `gaussian-processes` | Plain text → `<ExternalLink>`; url field confirmation |
| Topic 25 | §25.10 formalmlConnections | `probabilistic-programming` | Plain text → `<ExternalLink>`; url field confirmation |
| Topic 26 (MCMC) | §26.10 Rem 26 | `variational-inference` | Plain text → `<ExternalLink>` (same as Topic 25) |
| Topic 26 | §26.10 Rem 28 | `reversible-jump-mcmc` | Plain text → `<ExternalLink>` |
| Topic 26 | §26.10 Rem 29 | `riemann-manifold-hmc` | Plain text → `<ExternalLink>` |
| Topic 26 | §26.10 Rem 30 | `stochastic-gradient-mcmc` | Plain text → `<ExternalLink>` |
| Topic 26 | §26.10 Rem 31 | `sequential-monte-carlo` | Plain text → `<ExternalLink>` |
| Topic 26 | §26.10 Rem 33 | `probabilistic-programming` | Plain text → `<ExternalLink>` (same as Topic 25) |
| Topic 27 (BMA) | §27.10 Rem 27 | `stacking-and-predictive-ensembles` | Plain text → `<ExternalLink>` |
| Topic 27 | §27.10 Rem 28 | `variational-bayes-for-model-selection` | Plain text → `<ExternalLink>` |
| Topic 27 | §27.10 Rem 29 | `reversible-jump-mcmc` | Plain text → `<ExternalLink>` (same as Topic 26) |
| Topic 28 (Hierarchical Bayes) | §28.10 Rem 23 | `sparse-bayesian-priors` | Plain text → `<ExternalLink>` (heavy discharge — also retires Topic 25/26/27 chained pointers) |
| Topic 28 | §28.10 Rem 24 | `variational-inference` | Plain text → `<ExternalLink>` (same as Topic 25, 26) |
| Topic 28 | §28.10 Rem 24 | `bayesian-neural-networks` | Plain text → `<ExternalLink>` (same as Topic 25) |
| Topic 28 | §28.10 Ex 14 | `meta-learning` | Plain text → `<ExternalLink>` |
| Topic 28 | §28.8 Rem 20 | `mixed-effects` | Plain text → `<ExternalLink>` (same as Topic 22) |
| Topic 29 (Order Stats) | §29.10 Rem 18 | `extreme-value-theory` | Plain text → `<ExternalLink>` |
| Topic 29 | §29.10 Rem 19 | `quantile-regression` | Plain text → `<ExternalLink>` |
| Topic 29 | §29.10 Rem 20 | `rank-tests` | Plain text → `<ExternalLink>` |
| Topic 29 | §29.10 Rem 21 | `conformal-prediction` | Plain text → `<ExternalLink>` |
| Topic 29 | §29.10 Rem 22 | `statistical-depth` | Plain text → `<ExternalLink>` |
| Topic 30 (KDE) | §30.10 Rem 17 (kernel-regression pointer) | `kernel-regression` | Plain text → `<ExternalLink>` |
| Topic 30 | §30.10 Rem 18 (mean-shift pointer) | `clustering` | Plain text → `<ExternalLink>` |
| Topic 30 | §30.10 Rem 20 (flows pointer) | `normalizing-flows` | Plain text → `<ExternalLink>` |
| Topic 30 | §30.10 local-regression pointer | `local-regression` | Plain text → `<ExternalLink>` |
| Topic 30 | §30.10 DRE pointer | `density-ratio-estimation` | Plain text → `<ExternalLink>` |
| Topic 31 (Bootstrap) | §31.10 formalmlConnections | `prediction-intervals` | Plain text → `<ExternalLink>` |
| Topic 31 | §31.10 formalmlConnections | `uncertainty-quantification` | Plain text → `<ExternalLink>` |
| Topic 31 | §31.10 formalmlConnections | `cross-validation` | REDIRECT: section within `conformal-prediction` or `high-dimensional-regression` per §7.3 |
| Topic 31 | §31.10 formalmlConnections | `ab-testing` | REDIRECT: section within `rank-tests` per §7.3 |
| Topic 32 (Empirical Processes) | §32.10 formalmlConnections | `generalization-bounds` | Plain text → `<ExternalLink>` |
| Topic 32 | §32.10 formalmlConnections | `pac-bayes-bounds` | Plain text → `<ExternalLink>` |
| Topic 32 | §32.10 formalmlConnections | `conformal-prediction` | Plain text → `<ExternalLink>` (same as Topic 29) |
| Topic 32 | §32.10 formalmlConnections | `semiparametric-inference` | Plain text → `<ExternalLink>` |
| Topic 32 | §32.10 formalmlConnections | `causal-inference-methods` | Plain text → `<ExternalLink>` (same as Topic 22) |
| Topic 32 | §32.10 formalmlConnections | `extreme-value-theory` | Plain text → `<ExternalLink>` (same as Topic 29) |

**Total edits per formalML topic ship:** 1–4 formalstatistics MDX files, ~1–6 remark edits, plus 1 frontmatter `formalmlConnections` confirmation per pointing file.

---

## § 16 Appendix B — Already-Published formalML Topics Referenced by formalstatistics

Three already-published formalML topics sit in formalstatistics's pointer graph and should already have live `<ExternalLink>` components (confirm during the first formalstatistics PR sweep).

| formalML slug | formalstatistics referencing topic(s) |
|---|---|
| `bayesian-nonparametrics` | Topic 28 §28.10 Rem 25 |
| `information-geometry` | Topic 22 §22.10 Rem 26 |
| `pac-learning` | Topic 32 cross-reference in §32.4 |

If any of these formalstatistics references currently use plain-text `(forthcoming)` markers, flip them during the next formalstatistics PR. These are zero-cost edits and should not wait for any formalML ship.

---

## § 17 Appendix C — Proposed Shared-Module Exports (Strawman)

Abbreviated list of primary exports per topic per §6. Full lists refined during per-topic handoff-brief authoring.

### T2 Supervised (`supervised-learning.ts`)
`kernel-regression` introduces: `nadarayaWatson`, `nwBandwidthLOOCV`, `nwKernelOptions`, `boundaryReflection`.
`local-regression` extends with: `localPolynomialFit`, `equivalentKernel`, `fanGijbelsBandwidthSelector`.
`high-dimensional-regression` extends with: `lassoCoordinateDescent`, `debiasedLasso`, `restrictedEigenvalueCheck`, `sparseCovEstimator`.

### T3 Unsupervised & Generative (`unsupervised.ts`)
`clustering` introduces: `meanShift`, `modeFinder`, `bandwidthSelectorForMeanShift`.
`density-ratio-estimation` extends with: `plugInRatio`, `kliep`, `uLSIF`, `relativeDensityRatio`, `mmdStatistic`.
`normalizing-flows` extends with: `couplingLayer`, `maskedAutoregressiveFlow`, `logDetJacobian`, `flowTrainingStep`.

### T4 Nonparametric & Distribution-Free (`nonparametric-ml.ts`)
`conformal-prediction` introduces: `splitConformalInterval`, `nonconformityScore`, `jackknifePlusInterval`, `cqrInterval`, `apsClassificationSet`, `empiricalCoverage`.
`quantile-regression` extends with: `checkLoss`, `quantileRegressionFit`, `conditionalQuantile`, `rearrangementMonotonize`, `pinballLoss`.
`rank-tests` extends with: `rankTransform`, `wilcoxonSignedRank`, `mannWhitneyU`, `kruskalWallisH`, `permutationDistribution`, `hodgesLehmann`, `pitmanARE`.
`extreme-value-theory` extends with: `gevDensity`, `gpdDensity`, `gevFitMLE`, `hillEstimator`, `returnLevel`, `valueAtRisk`, `expectedShortfall`.
`statistical-depth` extends with: `halfspaceDepth2D`, `halfspaceMedian2D`, `mahalanobisDepth`, `simplicialDepth2D`, `spatialDepth`, `depthRegion2D`, `ddPlot`.
`prediction-intervals` extends with: `bootstrapPercentilePI`, `bcaPredictionInterval`, `frequentistPredictiveInterval`, `bayesianCredibleInterval`, `coverageSimulator`.

### T5 Bayesian & Probabilistic ML (`bayesian-ml.ts`)
`variational-inference` introduces: `meanFieldELBO`, `elboGradient`, `amortizedVI`, `flowBasedVariationalPosterior`, `structuredVIFactorization`.
`gaussian-processes` extends with: `gpPosterior`, `kernelMatrix`, `hyperparameterOptMarginalLikelihood`, `rbfKernel`, `maternKernel`.
`bayesian-neural-networks` extends with: `mcDropoutInference`, `deepEnsembleTraining`, `laplaceApproxBNN`, `sgMCMCBNNTraining`, `bnnCalibrationDiagnostic`.
`sparse-bayesian-priors` extends with: `horseshoePrior`, `regularizedHorseshoe`, `spikeAndSlabGibbs`, `horseshoeHMCFriendlyReparam`.
Further, each of the 9 T5 topics contributes its own primitives; see future handoff briefs.

### T6 Learning Theory & Methodology (`learning-theory.ts`)
`generalization-bounds` introduces: `rademacherComplexity`, `empiricalRademacher`, `uniformConvergenceBound`, `generalizationGapEstimate`.
`vc-dimension` extends with: `shatteringCoefficient`, `sauerShelah`, `vcDimEstimator`.
`pac-bayes-bounds` extends with: `pacBayesBound`, `klPosteriorPrior`, `catoniBound`.
`semiparametric-inference` extends with: `influenceFunction`, `oneStepEstimator`, `tmleTargeting`, `tangentSpaceProjection`.
`causal-inference-methods` extends with: `ipwEstimator`, `aipwEstimator`, `doublyRobust`, `frontDoorAdjustment`, `instrumentalVariables`.
Further, 2 T6 topics extend with their own primitives.

### Cross-track shared modules
`multivariate.ts`: `mvnDensity`, `mvnSample`, `covarianceParameterizations`.
`information-theory.ts`: `klDivergence`, `entropyEstimator`, `mutualInformationEstimator`, `fDivergenceFamily`.
`optimization.ts`: `projectedGradient`, `admmSolver`, `coordinateDescentGeneric`.

---

## § 18 Open Architectural Questions for Future Review

Two questions this document intentionally does not settle. Revisit as conditions warrant.

1. **T5 split (§3.6).** Defer to T5's 6th published topic; revisit then based on usage-pattern data.
2. **Cross-site /paths graph rendering details.** UI decision — separate node styling for cross-site prereqs? Clickable cross-site navigation? Breadcrumb-style three-site nav? Best resolved during the /paths revision in §14.3. Note that cross-site relationships now live in MDX frontmatter (per §5.2), not in `curriculum-graph.json`, so a `/paths` graph that wants to surface them must read frontmatter at build time.

**Resolved (2026-04-25):** The original §18 question of whether formalML wants its own `<ExternalLink>` component is settled — yes, port the formalstatistics implementation (interface `{ href, site, topic }`) to `src/components/ui/ExternalLink.astro` and extend the `site` enum to `'formalstatistics' | 'formalcalculus'`. Build during the first ship that introduces an in-prose cross-site link (likely `conformal-prediction` or `variational-inference`). The component is one small Astro file; not blocking.

None of the open questions blocks the first wave.

---

*End of formalML.com consolidated strategic planning document v1. Thirty-two topics planned across 5 new tracks; architecture locked; sequencing recommended; per-topic handoff briefs unblocked. Claude Code sessions referencing this document should treat §§2.2, 3, 5, 7, 8 as authoritative and flag any apparent contradictions for a revision cycle rather than local resolution.*
