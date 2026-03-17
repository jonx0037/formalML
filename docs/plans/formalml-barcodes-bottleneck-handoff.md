# Claude Code Handoff Brief: Barcodes & Bottleneck Distance
**Project:** formalML (formalml.com)  
**Repo:** github.com/jonx0037/formalML  
**Stack:** Astro + MDX, TypeScript, Vercel  
**Author:** Jonathan Rocha  
**Date:** March 2026  
**Status:** Ready for implementation

---

## 1. Overview

Add a new topic article to the Topology & TDA learning path: **Barcodes & Bottleneck Distance**. This is the 5th article in the path sequence:

```
Simplicial Complexes → Čech Complexes & Nerve Theorem → Persistent Homology → Mapper Algorithm → [NEW] Barcodes & Bottleneck Distance
```

The article covers the mathematical foundations of persistence barcodes as a representation of persistent homology, then formalizes the bottleneck and Wasserstein distances used to compare persistence diagrams. It connects directly to the Stability Theorem introduced (but not fully developed) in the Persistent Homology article, and provides working Python code for computing and comparing barcodes in practice.

**Difficulty level:** Intermediate  
**Estimated read time:** 30 min  
**Prerequisites:** Persistent Homology  
**Track:** Topology & TDA

---

## 2. Deliverables

### Files to Create

1. `src/content/topics/barcodes-bottleneck.mdx` — The full article content (MDX format)
2. `drafts/notebooks/barcodes-bottleneck.ipynb` — Companion Jupyter notebook with all working code
3. Update `src/data/paths.ts` (or equivalent data file) — Change "Barcodes & Bottleneck Distance" from `planned` to `published` and add the route link

### Files to Modify

4. Update the curriculum roadmap / paths data so the article appears as published with a link
5. Update any topic connection/dependency metadata to register `barcodes-bottleneck` as depending on `persistent-homology`

---

## 3. Article Structure (MDX Content)

Follow the **exact structural pattern** established by the Persistent Homology article (`src/content/topics/persistent-homology.mdx`). Match its:
- Frontmatter schema (level, tags, readTime, prerequisites, track, connections)
- Section ordering (Overview → Formal Framework → Visual Intuition → Working Code → Connections → References)
- Mathematical notation style (LaTeX via KaTeX/MathJax, Definition/Theorem/Proof blocks)
- Code block style (Python with ripser, persim, numpy)
- Tone (rigorous but motivated, practitioner-aware)

### Section-by-Section Content Spec

#### Frontmatter
```yaml
---
title: "Barcodes & Bottleneck Distance"
slug: "barcodes-bottleneck"
level: "intermediate"
tags: ["topology", "tda"]
readTime: 30
prerequisites: ["persistent-homology"]
track: "topology"
description: "Comparing persistence diagrams — the metrics that make TDA a rigorous statistical tool"
connections:
  buildsOn: ["persistent-homology"]
  usesConceptsFrom: ["metric-spaces"]
---
```

#### Overview & Motivation (~400 words)

Opening hook: The Persistent Homology article showed how to *compute* a persistence diagram — a multiset of birth-death pairs that summarizes the topological features of a dataset across scales. But a single diagram is just a picture. To do science, you need to **compare** diagrams: Is dataset A topologically similar to dataset B? Is this diagram stable under noise? Can I average over a population of diagrams?

Key motivating questions:
- How do we define a "distance" between two persistence diagrams?
- Why can't we just use Hausdorff distance on the point sets?
- What makes bottleneck distance the natural choice for stability, and when do you want Wasserstein instead?

Frame the answer: Barcodes and persistence diagrams are two equivalent representations. The bottleneck distance measures the worst-case cost of matching features between two diagrams. The Wasserstein distance measures the total cost. Both are true metrics on the space of persistence diagrams, and both satisfy stability theorems — but with different tradeoffs.

Connect to practice: These distances are what make TDA a *statistical* method rather than just a visualization trick. Without them, you can't do hypothesis testing, confidence sets, or machine learning on topological features.

#### Formal Framework

**Section 3.1: Barcodes as Interval Decompositions**

- Definition: persistence barcode as a multiset of intervals $\{[b_i, d_i)\}$
- Theorem: Structure Theorem for persistence modules (state precisely, cite Zomorodian & Carlsson 2005)
  - A persistence module $\mathbb{V}$ over a field decomposes uniquely (up to reordering) into interval modules: $\mathbb{V} \cong \bigoplus_{i} \mathbb{I}[b_i, d_i)$
  - Each interval corresponds to a bar in the barcode
- Remark: equivalence between barcodes and persistence diagrams (bijection between intervals $[b,d)$ and off-diagonal points $(b,d)$)
- Example: compute the barcode of the Vietoris-Rips filtration on 4 points forming a square, showing each bar explicitly

**Section 3.2: The Space of Persistence Diagrams**

- Definition: persistence diagram as a multiset $D \subset \{(b,d) \in \mathbb{R}^2 \mid b < d\} \cup \Delta$, where $\Delta = \{(x,x) \mid x \in \mathbb{R}\}$ is the diagonal (with infinite multiplicity)
- Why the diagonal: diagrams can have different numbers of off-diagonal points; the diagonal serves as a "graveyard" for unmatched features
- Definition: partial matching between diagrams

**Section 3.3: Bottleneck Distance**

- Definition (formal):
  $$d_B(D, D') = \inf_{\gamma: D \to D'} \sup_{p \in D} \|p - \gamma(p)\|_\infty$$
  where $\gamma$ ranges over all bijections (possible because of diagonal points)
- Geometric interpretation: the bottleneck distance is the minimum cost of a perfect matching where cost = worst single match
- Theorem: $d_B$ is a metric on the space of persistence diagrams
- Worked example: compute $d_B$ between two small diagrams by hand (3-4 points each), showing the optimal matching

**Section 3.4: Wasserstein Distance**

- Definition:
  $$W_p(D, D') = \left(\inf_{\gamma: D \to D'} \sum_{x \in D} \|x - \gamma(x)\|_\infty^p \right)^{1/p}$$
- Contrast with bottleneck: Wasserstein penalizes *all* mismatches, not just the worst one
- When to use which:
  - Bottleneck: stability guarantees, worst-case analysis, theoretical proofs
  - Wasserstein (especially $W_2$): statistical applications, persistence images, ML pipelines — because it's sensitive to *all* features, not just the most persistent one
- Remark: $d_B = \lim_{p \to \infty} W_p$

**Section 3.5: The Stability Theorem (Full Treatment)**

- Theorem (Cohen-Steiner, Edelsbrunner, Harer 2007): For tame functions $f, g: X \to \mathbb{R}$:
  $$d_B(\text{Dgm}(f), \text{Dgm}(g)) \leq \|f - g\|_\infty$$
- Proof sketch: outline the interleaving argument (reference the persistent homology article's treatment, then extend)
- Corollary (point cloud stability): Hausdorff perturbation bound
- Theorem (Wasserstein stability, Cohen-Steiner et al. 2010): For $W_p$, stability holds with different constants depending on the total persistence
- Key takeaway: stability is what makes TDA safe to use on real data

**Section 3.6: Isometry Theorem**

- Theorem (Bauer & Lesnick 2015 / Lesnick 2015): The bottleneck distance between persistence diagrams equals the interleaving distance between the corresponding persistence modules
  $$d_B(\text{Dgm}(\mathbb{V}), \text{Dgm}(\mathbb{W})) = d_I(\mathbb{V}, \mathbb{W})$$
- Significance: this is the deep reason *why* the bottleneck distance is the right metric — it captures algebraic proximity, not just geometric proximity of point sets
- Brief statement only; full proof is beyond scope

#### Visual Intuition

Interactive visualization (React component or embedded widget):

1. **Barcode viewer**: Two point clouds side by side (user can toggle between circle, cluster, figure-eight, torus cross-section). Each shows the Vietoris-Rips barcode below it, with bars colored by dimension (H₀ teal, H₁ purple, H₂ if applicable).

2. **Matching visualizer**: Show two persistence diagrams. Draw the optimal bottleneck matching as lines between matched points (and lines to diagonal for unmatched). Display $d_B$ value. Let user drag points to see how the matching and distance change.

3. **Stability demo**: A point cloud with a noise slider. As noise increases, show the persistence diagram shifting and display both $d_B$ and the Hausdorff perturbation, confirming $d_B \leq 2 \cdot d_H$ empirically.

**Implementation note:** Match the interactive style from the Persistent Homology article (the ε-slider Vietoris-Rips visualization). Use the same React component patterns and styling.

#### Working Code

All code uses `numpy`, `ripser`, `persim`, `scipy`. Provide 4 code blocks:

**Block 1: Computing and Plotting Barcodes**
```python
# Generate point clouds from different shapes
# Compute persistence with ripser
# Plot barcodes using persim.plot_diagrams (barcode mode)
# Compare barcode structure across shapes
```

**Block 2: Bottleneck Distance Computation**
```python
# Use persim.bottleneck to compute d_B between diagram pairs
# Show the optimal matching
# Verify triangle inequality empirically
```

**Block 3: Wasserstein Distance and Comparison**
```python
# Use persim.wasserstein to compute W_p for p=1,2
# Compare W_1, W_2, d_B on the same diagram pairs
# Show when they disagree and why
```

**Block 4: Stability Verification Pipeline**
```python
# Systematic noise experiment:
# - Base point cloud (circle, 100 points)
# - Add Gaussian noise at σ = 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5
# - Compute d_B(Dgm(base), Dgm(noisy)) and d_H(base, noisy)
# - Plot d_B vs d_H, overlay the 2x bound line
# - Repeat for Wasserstein
```

#### Connections & Applications (~200 words)

- **Statistical TDA**: Bottleneck and Wasserstein distances enable hypothesis testing on topological features (Fasy et al. 2014, bootstrap confidence sets for persistence diagrams)
- **Persistence landscapes** (Bubenik 2015): an alternative representation that lives in a Banach space, enabling standard statistical tools (means, variances, t-tests)
- **Optimal transport**: Wasserstein distance on persistence diagrams is a special case of optimal transport — connecting TDA to the rich theory of Monge-Kantorovich problems
- **Sheaf theory** (planned): sheaf-theoretic generalizations of persistence provide multi-parameter extensions where barcode decomposition no longer holds — the bottleneck distance generalizes to interleaving distance in this setting

#### References & Further Reading

1. Cohen-Steiner, Edelsbrunner & Harer (2007) — "Stability of Persistence Diagrams" — *Discrete & Computational Geometry*
2. Cohen-Steiner, Edelsbrunner, Harer & Morozov (2010) — "Lipschitz Functions Have $L_p$-Stable Persistence" — *Foundations of Computational Mathematics*
3. Zomorodian & Carlsson (2005) — "Computing Persistent Homology" — *Discrete & Computational Geometry*
4. Bauer & Lesnick (2015) — "Induced Matchings and the Algebraic Stability of Persistence Barcodes" — *Journal of Computational Geometry*
5. Fasy, Lecci, Rinaldo, Wasserman, Balakrishnan & Singh (2014) — "Confidence Sets for Persistence Diagrams" — *Annals of Statistics*
6. Bubenik (2015) — "Statistical Topological Data Analysis using Persistence Landscapes" — *JMLR*
7. Edelsbrunner & Harer (2010) — *Computational Topology: An Introduction* — Chapters 8-9 on stability and distances
8. Kerber, Morozov & Nigmetov (2017) — "Geometry Helps to Compare Persistence Diagrams" — *Journal of Experimental Algorithmics* (efficient algorithms for Wasserstein computation)

---

## 4. Companion Jupyter Notebook

The file `drafts/notebooks/barcodes-bottleneck.ipynb` is provided alongside this brief. It contains all four code blocks from the Working Code section, fully executable, with:

- Markdown cells matching the article narrative
- All imports and seeds specified
- Inline comments explaining each step
- Output cells cleared (Claude Code should not pre-run)

**Dependencies** (add to any requirements/environment file if not already present):
```
numpy
ripser
persim
scipy
matplotlib
scikit-learn
```

---

## 5. Paths Data Update

In whatever data structure drives `formalml.com/paths`, update the Topology & TDA track:

**Before:**
```
○  Barcodes & Bottleneck Distance  Planned
```

**After:**
```
●  Barcodes & Bottleneck Distance  Intermediate  [link: /topics/barcodes-bottleneck]
```

---

## 6. Placeholder: Statistical TDA (Next Planned Article)

**Do not implement — add as a planned entry only.**

After "Barcodes & Bottleneck Distance", add a new planned article to the Topology & TDA track:

```
○  Statistical TDA  Planned
```

This article will cover:
- Bootstrap confidence sets for persistence diagrams (Fasy et al. 2014)
- Permutation tests for comparing populations of diagrams
- Persistence landscapes as Banach space elements (Bubenik 2015)
- Integration with scikit-tda and GUDHI
- Practical pipelines: point cloud → persistence → feature vector → statistical test

**Rationale:** This rounds out the applied side of the TDA track before pivoting to the more algebraic direction (Sheaf Theory). It gives readers the statistical tools to actually *use* the distances formalized in the Barcodes & Bottleneck Distance article.

Updated Topology & TDA sequence after both additions:

```
● Simplicial Complexes          Foundational   [Start here]
● Čech Complexes & Nerve Theorem  Intermediate
● Persistent Homology           Intermediate
● The Mapper Algorithm          Advanced
● Barcodes & Bottleneck Distance  Intermediate  ← THIS BRIEF
○ Statistical TDA               Planned         ← PLACEHOLDER ONLY
○ Sheaf Theory                  Planned
```

---

## 7. Implementation Notes for Claude Code

1. **Match the existing article pattern exactly.** Read `src/content/topics/persistent-homology.mdx` as the reference implementation before writing anything. Match frontmatter schema, component imports, definition/theorem block syntax, code block formatting, and section structure.

2. **Interactive visualizations** should use the same React component patterns as the persistent homology article's ε-slider widget. Check `src/components/` for reusable TDA visualization components.

3. **LaTeX rendering**: Use whatever math rendering the site already has configured (KaTeX or remark-math). Do not introduce new math dependencies.

4. **The notebook is a draft asset**, not a deployed page. It goes in `drafts/notebooks/`, not in the Astro content directory.

5. **Test locally** with `pnpm dev` before committing. Verify:
   - Article renders at `/topics/barcodes-bottleneck`
   - Math renders correctly (especially the $\inf$, $\sup$, and $\|\cdot\|_\infty$ notation)
   - Prerequisites link resolves to persistent homology article
   - Paths page shows the article as published
   - Statistical TDA appears as planned (no link)

---

*Brief prepared by Jonathan Rocha with Claude (Anthropic), March 2026.*
