# Handoff Brief — Bayesian Neural Networks (Topic 6, T5)

**Status:** 🚧 Ready for implementation (pre-brief stage complete)
**Track:** T5 Bayesian & Probabilistic ML (6th topic in track, T5 flagship)
**Slug:** `bayesian-neural-networks`
**Difficulty:** advanced
**Author:** Jonathan Rocha — Last updated 2026-05-07

This brief layers MDX-structure, viz-component, and integration spec on top of the verified notebook at `notebooks/bayesian-neural-networks/01_bayesian_neural_networks.ipynb`. The notebook is the source of truth for math, code, and figures. Where this brief and the notebook disagree, the notebook wins.

A point-estimate neural network produces a single function and confidently extrapolates that function into regions where no training data lives. Bayesian neural networks replace the single weight vector $w^*$ with a distribution $p(w \mid \mathcal{D})$ over weights and integrate predictions against it, so predictive variance grows where the data has nothing to say. The catch is that $p(w \mid \mathcal{D})$ is intractable in four distinct ways for any non-trivial network — no closed-form marginal likelihood, deeply non-convex log-posterior, high dimension, and full-data gradients. Each of the methods in this topic is an answer to one of these obstacles. As the T5 flagship, this topic develops Laplace approximation (asymptotic local Gaussian), MC-dropout (Bernoulli variational family), deep ensembles (function-space samples via independent training), and stochastic-gradient MCMC (SGLD and SGHMC, the asymptotically-exact methods that scale to deep-learning regimes), then evaluates all four head-to-head on calibration metrics, then closes with the function-space view from NNGP and NTK that ties weight-space methods to Gaussian processes.

---

## 1. Decision Log (Locked During Pre-Brief Drafting)

These decisions emerged during the conversational pre-brief drafting session. Do not relitigate during implementation.

| Decision | Choice | Rationale |
|---|---|---|
| Track placement | T5 Bayesian & Probabilistic ML (6th topic in track, flagship) | Strategic plan §3.5; first-wave ordering §4.2 |
| Slug | `bayesian-neural-networks` | Strategic plan §2 |
| Difficulty | advanced | Strategic plan §3.5; broad-scope advanced per §12.2 |
| Section count | 9 H2 sections | Topic-driven: 1 motivation + 1 foundational object + 5 methods + 1 calibration + 1 function-space closer |
| Code language | Python + NumPy / SciPy / scikit-learn / matplotlib **+** PyTorch (default for BNN training loops in §§3–7) | Strategic plan §8.2 — BNN is on the PyTorch exception list |
| §§3–7 running example | Two Moons binary classification (`sklearn.datasets.make_moons`, $n=300$, noise $0.20$) | 2D so predictive uncertainty visualizes naturally; CPU-only training in <5s; distinct from VI's banana, PP's eight schools, ME's six-classroom |
| Base architecture | 3-hidden-layer × 32-unit ReLU MLP, BCE-with-logits loss, Adam + $L_2$ weight decay $\lambda=10^{-4}$ | Small enough that all five methods (§§3–7) train inside the 60s budget; deep enough that the loss landscape is non-convex |
| §§3–7 method coverage | Each of Laplace / MC-dropout / deep ensembles / SGLD / SGHMC gets its own H2 section with full mathematical treatment | Strategic doc one-liner names them as co-equal pillars; `bnnCalibrationDiagnostic` is its own shared-module primitive |
| §6 / §7 split | SGLD and SGHMC in separate sections | Welling & Teh 2011 first-order Langevin SDE vs. Chen, Fox & Guestrin 2014 second-order Langevin SDE are mathematically distinct stationary-distribution arguments; treating them as one section either compresses both or makes one section disproportionately heavy |
| §8 calibration scope | Full section: ECE, Brier with reliability/resolution/uncertainty decomposition, NLL as proper scoring rule, epistemic/aleatoric decomposition, cold-posterior effect, temperature scaling | `bnnCalibrationDiagnostic` shared-module primitive plus the strategic-doc one-liner naming calibration co-equal with the four methods |
| §9 function-space view | Sidebar in closing section: NNGP (Neal 1996, Lee et al. 2017) + NTK (Jacot et al. 2018) as forward-pointers, no full kernel-machinery derivation | Per pre-brief drafting decision; full NTK treatment would push topic past the 9-section budget and duplicate GP machinery |
| `weight-decay` host | Named subsection inside §2 (Prop 2.2) | CLAUDE.md and strategic doc §7.3 — `weight-decay` slug never minted; lives as a section in BNN |
| Notebook structure | Title + setup + 9 (markdown, code) pairs = 20 cells | Mirrors VI/PP/ME notebook layout, scaled to 9 sections |

---

## 2. Frontmatter (Proposed for `src/content/topics/bayesian-neural-networks.mdx`)

```yaml
---
title: "Bayesian Neural Networks"
description: "Weight-space posteriors over neural networks: Laplace approximation, MC-dropout as approximate VI, deep ensembles, stochastic-gradient MCMC (SGLD and SGHMC), calibration diagnostics, and the function-space view via NNGP and NTK."
slug: "bayesian-neural-networks"
domain: "bayesian-ml"
difficulty: "advanced"
status: "published"
prerequisites:
  - variational-inference
  - gaussian-processes
  - kl-divergence
  - gradient-descent
  - convex-analysis
formalstatisticsPrereqs:
  - topic: "bayesian-foundations-and-prior-selection"
    site: "formalstatistics"
    relationship: "BNN posteriors are MVN near the MAP estimate by Bernstein–von Mises (§25.8 Thm 5); the Laplace approximation (§25.8 Rem 16) is the canonical construction lifted from formalstatistics into deep-learning scale in §3 of this topic."
  - topic: "central-limit-theorem"
    site: "formalstatistics"
    relationship: "Bernstein–von Mises is the Bayesian CLT — the posterior of √n(θ − θ_MLE) converges in total variation to N(0, I(θ_0)^{-1}). §3.2 cites this theorem as the asymptotic justification for the Laplace approximation in well-specified BNNs."
  - topic: "modes-of-convergence"
    site: "formalstatistics"
    relationship: "Bernstein–von Mises convergence is in total variation distance, hence in distribution. §2.3 invokes the total-variation framing to argue that in the large-data limit, the BNN posterior is approximated arbitrarily well by a single Gaussian centered at the MLE."
  - topic: "multivariate-distributions"
    site: "formalstatistics"
    relationship: "Weight priors w ~ N_p(0, σ²I) are multivariate Normal. The Laplace approximation in §3 produces a Gaussian posterior whose inverse covariance is the Hessian of the negative log-posterior at the MAP. §4 and §6 each carry MVN-sampling subroutines."
  - topic: "hierarchical-bayes-and-partial-pooling"
    site: "formalstatistics"
    relationship: "Hierarchical priors over groups of weights (e.g., one variance per layer) and the partial-pooling-via-shrinkage view inform §3.3's discussion of structured prior choices and §8.5's cold-posterior remark on prior misspecification."
formalcalculusPrereqs:
  - topic: "jacobians"
    site: "formalcalculus"
    relationship: "The Hessian of the negative log-posterior in §3.1 is a Jacobian computation against the gradient; KFAC and diagonal-Fisher reductions in §3.3 are structured approximations to this Jacobian."
  - topic: "stochastic-differential-equations"
    site: "formalcalculus"
    relationship: "SGLD §6 is a discretization of the first-order Langevin SDE; SGHMC §7 is a discretization of the second-order Langevin SDE with friction. The stationary-distribution arguments invoke the Fokker–Planck framework formalcalculus develops."
  # NOTE: if formalcalculus does not yet have an SDE topic, log this as a deferred-reciprocal
  # and use a hierarchical-bayes / measure-theoretic-probability surrogate prereq during shipment.
references:
  # See §11 of this brief for the full Chicago bibliography.
---
```

> **NOTE on the SDE prereq:** the brief currently assumes formalcalculus has a topic on stochastic differential equations or Fokker–Planck. Verify at ship time; if no such topic exists, drop this prereq, leave §§6–7 self-contained on the SDE math, and route the formalstatistics SDE-related forward-pointer (if any) through `bayesian-computation-and-mcmc` instead.

---

## 3. Section Outline

| § | Title | Theorems / Definitions | Figure |
|---|-------|------------------------|--------|
| §1 | Why Bayesian over the weights | none (geometric motivation) | `01_point_vs_bayesian_predictive.png` |
| §2 | The weight-space posterior | Def 2.1 (Gaussian-prior MLP); Prop 2.2 (Negative-log-posterior = penalized risk; MAP = weight-decay solution); Thm 2.3 (Bernstein–von Mises for parametric models, stated; cross-link to formalstatistics CLT); Rem 2.4 (Why naïve sampling fails: multimodality, high-d, identifiability) | `02_loss_landscape_modes.png` |
| §3 | The Laplace approximation | Def 3.1 (Laplace approximation); Thm 3.2 (Asymptotic exactness via BvM); Prop 3.3 (Predictive distribution under linearization — last-layer, KFAC, diagonal Fisher); Alg 3.4 (Laplace-fit BNN) | `03_laplace_two_moons.png` |
| §4 | MC-dropout as approximate variational inference | Def 4.1 (Bernoulli variational family over weights); Thm 4.2 (Gal–Ghahramani 2016 — dropout-MLP training = mean-field VI under that family); Prop 4.3 (Predictive variance from $T$ stochastic forward passes); Rem 4.4 (Concrete-dropout and structured-dropout extensions) | `04_mcdropout_two_moons.png` |
| §5 | Deep ensembles as a function-space posterior proxy | Def 5.1 (Deep ensemble); Thm 5.2 (Loss-landscape mode coverage and the function-space ensemble interpretation); Prop 5.3 (Mixture-of-Gaussians predictive form); Rem 5.4 (Connection to stacking — pointer to that topic) | `05_deep_ensemble_two_moons.png` |
| §6 | Stochastic-gradient Langevin dynamics (SGLD) | Def 6.1 (SGLD update); Thm 6.2 (Welling & Teh 2011 — stationary distribution under decaying stepsize via first-order Langevin SDE); Prop 6.3 (Mini-batch gradient noise as part of the injected Gaussian noise); Prop 6.4 (Bias-variance tradeoff in constant-stepsize SGLD); Alg 6.5 (SGLD posterior-sampling loop) | `06_sgld_two_moons.png` |
| §7 | Stochastic-gradient HMC (SGHMC) | Def 7.1 (SGHMC update with momentum and friction); Thm 7.2 (Chen, Fox & Guestrin 2014 — second-order Langevin SDE stationary distribution); Prop 7.3 (Friction term compensates stochastic-gradient variance); Rem 7.4 (Practical preconditioning — RMSProp-SGHMC); Alg 7.5 (SGHMC posterior-sampling loop) | `07_sghmc_two_moons.png` |
| §8 | Calibration and uncertainty quantification | Def 8.1 (Expected calibration error); Def 8.2 (Brier score and its reliability/resolution/uncertainty decomposition); Def 8.3 (Negative log-likelihood as proper scoring rule); Prop 8.4 (Epistemic vs. aleatoric decomposition under the BNN predictive); Rem 8.5 (Cold-posterior effect — Wenzel et al. 2020); Rem 8.6 (Temperature scaling as post-hoc calibration) | `08_calibration_comparison.png` |
| §9 | Function-space view: NNGP, NTK, and open problems | Rem 9.1 (NNGP — Neal 1996; Lee et al. 2017 — infinite-width MLP prior is a GP); Rem 9.2 (NTK — Jacot et al. 2018 — gradient-descent training of infinite-width nets reduces to kernel regression); Rem 9.3 (What this means for the §§3–7 weight-space methods); Rem 9.4 (Open: prior misspecification, finite-width gap, scalable HMC, function-space VI) | `09_nngp_sidebar.png` |

Total: 6 numbered theorems + 9 propositions + 11 numbered definitions + 4 algorithms + 9 figures.

The same nine headings structure both this brief and the notebook (with the standard `# === §N: ... ===` code-cell convention).

---

## 4. Section-by-Section Content

> **A note on visualization scope.** Every section below has both a *static figure* (the PNG generated by the notebook, used as the fallback rendering) **and** an *interactive React + D3 component* (the primary on-page rendering, hydrated via `client:visible` per CLAUDE.md). The "Viz design intent" subsections specify the figure's static panel layout *and* the interactive controls the React component must expose — sliders, dropdowns, click targets, toggles. Interactive viz is one of the three pillars per CLAUDE.md ("rigorous math, interactive visualization, and working code") and a v1 deliverable for every topic on formalml.com, not a v2 stretch goal. Component conventions (`useD3` hook, `useResizeObserver`, CSS-custom-property color tokens, mobile breakpoint at 640 px, default export, named function) are documented in `docs/plans/formalml-handoff-reference.md` §4.

---

### §1. Why Bayesian over the weights

A trained neural network gives us a function, but it does not tell us how confident to be in that function. A point-estimate classifier on the *Two Moons* distribution — two interleaving crescents in $\mathbb{R}^2$, separable by a smooth nonlinear boundary, contaminated with isotropic noise — produces a single decision rule and confidently extrapolates that rule into regions where no training data lives. The rule is wrong somewhere, but the model has no way to say where. For an ML practitioner, this is a structural problem: medical-imaging classifiers, autonomous-vehicle perception models, and clinical-decision-support systems all need to know when they don't know. A model that returns "99% confident" everywhere — including on inputs unlike anything it has ever seen — is a liability.

Bayesian neural networks are the response. Instead of a single weight vector $w^* \in \mathbb{R}^p$ that minimizes a training loss, we work with a *distribution* $p(w \mid \mathcal{D})$ over weights, conditioned on the training data $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$. Predictions integrate over that distribution. This topic is the catalogue of practical recipes for building, fitting, and reading that distribution when the model has thousands or millions of weights and the posterior is intractable in every standard sense. §§3–7 develop five recipes — Laplace approximation, MC-dropout, deep ensembles, SGLD, SGHMC. §8 evaluates them head-to-head on calibration. §9 closes with the function-space view. This section frames *why* the recipes are needed and what they share.

#### §1.1 What a point-estimate model can't tell us

Fix notation. A *neural network* with weights $w \in \mathbb{R}^p$ is a parametric function $f_w : \mathcal{X} \to \mathcal{Y}$ assembled from compositions of affine maps and elementwise nonlinearities. For binary classification, $\mathcal{Y} = \{0, 1\}$ and the model produces a real-valued *logit* $f_w(x) \in \mathbb{R}$ which we squash through the logistic sigmoid $\sigma(z) = (1 + e^{-z})^{-1}$ to get a class-1 probability $\sigma(f_w(x))$. The likelihood under the Bernoulli observation model is

$$p(y \mid x, w) \;=\; \sigma(f_w(x))^y \bigl(1 - \sigma(f_w(x))\bigr)^{1 - y}.$$

A *point-estimate* network minimizes a regularized cross-entropy loss

$$w^* \;=\; \arg\min_{w \in \mathbb{R}^p}\,\Bigl\{ -\sum_{i=1}^n \log p(y_i \mid x_i, w) + \tfrac{\lambda}{2}\|w\|^2 \Bigr\}$$

using stochastic gradient descent or one of its momentum variants, and returns the single learned weight vector $w^*$. At test time the model produces a single predicted probability $\sigma(f_{w^*}(x^*))$. Two things are missing.

The first is *uncertainty about the prediction*. The model returns a number near $0$ or near $1$, with no companion estimate of how much that number would change under a different reasonable choice of $w$. The second is *uncertainty about $w$ itself*. The training loss is non-convex in $w$; the SGD trajectory ends in some local minimum determined by initialization, batch order, and learning-rate schedule; a different run with a different seed lands in a different minimum, with potentially different predictions far from the training data. The point estimate $w^*$ is one of many plausible weight vectors, and the model has no language for that fact.

#### §1.2 The predictive distribution and the four obstacles

The Bayesian fix is to replace $w^*$ with the full posterior. Place a prior $p(w)$ on the weights — typically the isotropic Gaussian $\mathcal{N}(0, \tau^2 I_p)$ that pairs with the $\ell_2$ regularizer above (§2.2 makes this correspondence explicit). Bayes' rule gives

$$p(w \mid \mathcal{D}) \;=\; \frac{p(\mathcal{D} \mid w)\,p(w)}{p(\mathcal{D})}, \qquad p(\mathcal{D}) \;=\; \int p(\mathcal{D} \mid w)\,p(w)\,dw,$$

where $p(\mathcal{D} \mid w) = \prod_{i=1}^n p(y_i \mid x_i, w)$ is the likelihood of the training data under weights $w$. The *predictive distribution* at a new input $x^*$ marginalizes the weights out:

$$p(y^* \mid x^*, \mathcal{D}) \;=\; \int p(y^* \mid x^*, w)\,p(w \mid \mathcal{D})\,dw.$$

Where the data is dense, plausible weight settings agree on what $y^*$ should be, and the integral concentrates — predictive variance is small. Where the data is sparse, plausible weight settings disagree, the integral spreads, and predictive variance grows. *The model knows when it doesn't know, because "I don't know" is encoded as breadth in the weight posterior.*

The catch is that $p(w \mid \mathcal{D})$ is a distribution on $\mathbb{R}^p$ for $p$ on the order of $10^4$ to $10^9$, and the posterior is intractable in four distinct ways. The marginal likelihood $p(\mathcal{D})$ has no closed form for any non-trivial network, so we cannot evaluate $p(w \mid \mathcal{D})$ pointwise without an approximation. The negative log-likelihood is *deeply non-convex* in $w$: by symmetry, every loss-landscape mode has copies under permutations of the hidden units, and the modes are separated by sharp ridges where Gaussian approximations break down. The dimension $p$ is high enough that vanilla MCMC mixes too slowly to be useful at deep-learning scale. And the likelihood gradient $\nabla_w \log p(\mathcal{D} \mid w) = \sum_{i=1}^n \nabla_w \log p(y_i \mid x_i, w)$ requires a full pass over $\mathcal{D}$ — feasible for $n$ in the hundreds, infeasible for the millions that motivate using a neural network in the first place.

Every method in this topic is an answer to one of these obstacles. **Laplace approximation** (§3) gives up on the multimodal structure and fits a single local Gaussian centered at the MAP estimate — cheap and asymptotically justified by the [Bernstein–von Mises theorem](https://www.formalstatistics.com/topics/central-limit-theorem) that §2 imports from formalstatistics. **MC-dropout** (§4) reinterprets the dropout regularizer used at training time as a Bernoulli variational posterior and turns the deterministic predictor's stochastic forward passes into Monte Carlo posterior samples — almost free, but the variational family is rigid. **Deep ensembles** (§5) abandon weight-space inference altogether and treat $K$ independently-trained networks as approximate samples in *function space* — empirically the strongest of the cheap methods, but the theoretical justification is delicate. **Stochastic-gradient Langevin dynamics** (§6) and **stochastic-gradient HMC** (§7) bring the asymptotic exactness of MCMC into the deep-learning regime by injecting calibrated Gaussian noise into mini-batch gradient updates — the noise compensates for the bias mini-batching introduces, and the resulting stochastic process has the posterior as its stationary distribution.

#### §1.3 The geometric picture and the road ahead

For the rest of this section we work on a 2D toy classifier so the picture is visible. Data are drawn from Two Moons with noise level $0.20$ at $n = 300$ points. We fit a small MLP — three hidden layers, 32 ReLU units each — by minimizing the regularized binary cross-entropy loss with $\lambda = 10^{-4}$ for 200 Adam epochs. The point-estimate model gives the decision surface in panel (a) of Figure 1: a clean boundary separating the two classes, with predicted probabilities near $0$ or near $1$ across most of the input space — high confidence even at distances from the data where any honest model should hesitate.

To see what an honest predictive distribution looks like we fit five copies of the same architecture from independent random initializations. Panel (b) overlays the five $0.5$-probability decision boundaries. Where the data is dense, the boundaries agree and the overlay is sharp. Far from the data — in the corners of the input frame — the boundaries fan out: five plausible models trained on the same data give five different answers in the regions where the data has nothing to say. Panel (c) renders the disagreement as a heatmap of *predictive variance* over the input space, the variance of the predicted class-1 probabilities across the five models. Bright regions are where the model knows it doesn't know. The five-model ensemble previewed here is the embryo of every method in this topic, and §5 formalizes it as a *deep ensemble*; §§3, 4, 6, 7 will produce comparable heatmaps from very different mechanisms.

A second axis of uncertainty matters too. If two data points sit in the same input neighborhood but carry different class labels — which Two Moons can produce in the strip between the crescents at high noise — no model on any architecture can confidently predict that neighborhood. That is *aleatoric* uncertainty: irreducible noise inherent in the data-generating process. The variance from disagreement among trained models is *epistemic* uncertainty: it reflects what the model would learn from more data, not what no model can ever learn. §8 disentangles the two formally; for now we note that BNNs primarily address epistemic uncertainty, and that a calibrated BNN should report large epistemic variance off-distribution and large aleatoric variance in noisy regions of the support.

The §2 derivation lifts this geometric picture into a formal weight-space posterior — the object the rest of the topic approximates. With that machinery in hand, §3 fits its first Bayesian neural network.

#### Code experiment §1

Five independently-trained Two Moons MLPs, each a 3-hidden-layer 32-unit ReLU network optimized by Adam against the binary cross-entropy loss with $L_2$ weight decay $\lambda = 10^{-4}$ for 200 epochs. PyTorch end-to-end so the same training loop carries through §§3–7. Total cell runtime ~5s on CPU. Library calls: `torch.nn.Sequential` for the MLP, `torch.optim.Adam` with `weight_decay=1e-4` for the optimizer, `sklearn.datasets.make_moons(n_samples=300, noise=0.20, random_state=42)` for the data, `numpy.linspace + meshgrid` for the prediction grid. Cell prints a verification triple: mean predictive variance on data support, mean predictive variance off data support, and their ratio (expected to be roughly $5\times$ to $30\times$, depending on seed — the MDX implementation should check the printed ratio is at least $3\times$).

#### Viz design intent §1

**Figure 1 (`01_point_vs_bayesian_predictive.png`).** Three panels on a common $200 \times 200$ prediction grid covering $[-2.5, 2.5] \times [-2.0, 2.0]$.

- **Panel (a)** — Two Moons data overlaid with one deterministic MLP's predicted-probability heatmap (single seed). Colormap `RdBu_r`, $v_{\min} = 0$, $v_{\max} = 1$, 20 contour levels. The reader should see a sharp, confident decision surface that is correct near the data and arbitrary far from it. No uncertainty information shown.
- **Panel (b)** — Same data and grid; overlay the $0.5$-probability contours of all five independently-trained MLPs as semi-transparent red lines (alpha $0.5$, linewidth $1.2$). Reader should see tight agreement where the data is dense and visible fanning-out in the corners.
- **Panel (c)** — Heatmap of predictive variance: for each grid point, compute the variance of the five predicted class-1 probabilities, render with a `viridis` colormap, attach a colorbar labeled `Var[p(y=1|x,w_k)]`. Overlay the data points in light gray for reference. Reader should see dark (low-variance) regions hugging the data and bright (high-variance) regions far from any data — the desideratum the rest of the topic delivers via different mechanisms.

The PNG (matplotlib, `bbox_inches='tight'`, dpi $150$) is the static fallback. The on-page rendering is the React + D3 component `BNNPredictiveMotivationViz.tsx`.

**Interactive controls (v1 deliverable):**

- *Two Moons noise slider* (`0.05` to `0.40`, default `0.20`). Re-runs `make_moons` data generation and re-trains all 5 MLPs on the new data, then re-renders all three panels. Caches per noise value to avoid re-training on every interaction.
- *Number-of-models slider* ($K = 2$ to $K = 10$, default $K = 5$). Subsets the precomputed pool of trained models and re-renders panels (b) and (c). Animates smoothly without re-training.
- *Resample-seed button.* Re-rolls the random initialization seeds for the $K$ models and re-trains. Lets the reader confirm the variance pattern is robust across seed sets, not an artifact of the default seed.

The component pre-trains a pool of $\sim 10$ MLPs at the default noise level on first hydration (~5 s blocking on first load — acceptable per `client:visible` since it only runs when the component scrolls into view), caches the trained weights in component state, and re-renders the three panels in <50 ms on subsequent slider interactions. The noise-slider interaction triggers a fresh training pass (~5 s) shown via a loading-state JSX swap (per CLAUDE.md "Loading-state JSX rule for fetch-based viz").

**What the reader should learn from Figure 1 / the §1 viz:** (i) point-estimate predictions don't communicate "I don't know"; (ii) disagreement among independently-trained models is itself a kind of uncertainty quantification — it's the embryo of every method in this topic; (iii) variance over the input space matches the geometric intuition that a model should be uncertain where it lacks data; (iv) — accessible *only* via the interactive controls — the noise level controls how much aleatoric variance contaminates the predictive, foreshadowing the §8 epistemic-aleatoric decomposition.

---

### §2. The weight-space posterior

The §1 picture — a model that should report breadth of plausible weight settings as predictive variance — is geometric. §2 lifts it into formal mathematics. The deliverable of this section is four facts: (i) the posterior under a Gaussian weight prior has a closed-form negative-log expression; (ii) maximizing that posterior is equivalent to L2-penalized empirical-risk minimization, so the well-known *weight decay* regularizer is a maximum-a-posteriori estimator under a specific prior; (iii) the *Bayesian central-limit theorem* (Bernstein–von Mises) guarantees that, in well-specified parametric models, the posterior asymptotically concentrates as a Gaussian centered at the MLE; and (iv) the regularity conditions for that theorem fail for typical neural networks, in three specific ways that motivate the four approximation strategies of §§3–7.

#### §2.1 The Gaussian-prior MLP

Fix a parametric MLP $f_w: \mathcal{X} \to \mathbb{R}$ with weights $w \in \mathbb{R}^p$. For binary classification the conditional distribution of the label given the input is Bernoulli with logit $f_w(x)$:
$$p(y \mid x, w) \;=\; \sigma(f_w(x))^y \bigl(1 - \sigma(f_w(x))\bigr)^{1 - y},$$
where $\sigma(z) = (1 + e^{-z})^{-1}$ is the logistic sigmoid. We complete the model with a Gaussian prior on the weights:

> **Definition 2.1 (Gaussian-prior MLP).** Let $f_w: \mathcal{X} \to \mathbb{R}$ be an MLP with weights $w \in \mathbb{R}^p$ and Bernoulli observation likelihood as above. Place the *isotropic Gaussian prior* $w \sim \mathcal{N}(0, \tau^2 I_p)$ for some scale $\tau > 0$. The *weight-space posterior* given training data $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ is the conditional distribution
> $$p(w \mid \mathcal{D}) \;=\; \frac{p(\mathcal{D} \mid w)\,p(w)}{p(\mathcal{D})}, \qquad p(\mathcal{D} \mid w) \;=\; \prod_{i=1}^n p(y_i \mid x_i, w).$$

The choice of $\tau^2 I_p$ as the prior covariance is the default in the BNN literature and the one we'll work with throughout. It assumes weights are *a priori* independent and identically distributed — a strong assumption that ignores all structure in the weight tensor (layer, channel, position) but keeps the math tractable. Heavy-tailed and hierarchical priors are the subject of `sparse-bayesian-priors` *(coming soon)*; per-layer prior scales are revisited in §8.5's cold-posterior remark. For now, $\tau$ is a single scalar hyperparameter — large $\tau$ is a weak prior (close to MLE), small $\tau$ is a strong prior (heavy regularization).

#### §2.2 The negative log-posterior, and weight decay as a Bayesian prior

Take the negative log of the posterior:
$$-\log p(w \mid \mathcal{D}) \;=\; -\log p(\mathcal{D} \mid w) \;-\; \log p(w) \;+\; \log p(\mathcal{D}).$$
The first term is the negative log-likelihood — the cross-entropy training loss (modulo signs). The second is a quadratic in $w$, since the prior is Gaussian: $-\log p(w) = \frac{1}{2\tau^2}\|w\|_2^2 + \mathrm{const}_{\tau, p}$, where the constant absorbs $\log\bigl((2\pi\tau^2)^{p/2}\bigr)$. The third does not depend on $w$. Stripping $w$-independent constants we get
$$-\log p(w \mid \mathcal{D}) \;=\; -\sum_{i=1}^n \log p(y_i \mid x_i, w) \;+\; \frac{1}{2\tau^2}\|w\|_2^2 \;+\; C,$$
where $C$ is a constant in $w$. This is structurally the L2-penalized cross-entropy training loss with regularization strength $\lambda = 1/\tau^2$.

> **Proposition 2.2 (MAP equals weight decay).** Let
> $$\mathcal{L}_{\mathrm{WD}}(w) \;:=\; -\sum_{i=1}^n \log p(y_i \mid x_i, w) \;+\; \frac{\lambda}{2}\|w\|_2^2$$
> be the L2-penalized cross-entropy loss with weight-decay strength $\lambda > 0$, and let $\hat{w}_{\mathrm{WD}} = \arg\min_w \mathcal{L}_{\mathrm{WD}}(w)$. Under Definition 2.1's Gaussian prior with $\tau^2 = 1/\lambda$, the *maximum-a-posteriori estimator*
> $$\hat{w}_{\mathrm{MAP}} \;:=\; \arg\max_w p(w \mid \mathcal{D})$$
> coincides with $\hat{w}_{\mathrm{WD}}$.

*Proof.* Maximizing a function and minimizing its negative are equivalent, so $\hat{w}_{\mathrm{MAP}} = \arg\min_w \bigl[-\log p(w \mid \mathcal{D})\bigr]$. From the calculation above, with $\tau^2 = 1/\lambda$,
$$-\log p(w \mid \mathcal{D}) \;=\; -\sum_{i=1}^n \log p(y_i \mid x_i, w) \;+\; \frac{\lambda}{2}\|w\|_2^2 \;+\; C.$$
The first two terms are exactly $\mathcal{L}_{\mathrm{WD}}(w)$, and $C$ does not depend on $w$. So $\arg\min_w \bigl[-\log p(w \mid \mathcal{D})\bigr] = \arg\min_w \mathcal{L}_{\mathrm{WD}}(w) = \hat{w}_{\mathrm{WD}}$. $\square$

Three corollaries are worth pulling out, each with downstream consequences for §§3–8.

*Weight decay is a Bayesian regularizer.* The standard "weight decay" hyperparameter $\lambda$ in deep-learning libraries has a Bayesian interpretation: $\lambda$ is the inverse variance of the implicit Gaussian prior on the weights, $\lambda = 1/\tau^2$. A weight-decay-trained model with $\lambda = 10^{-4}$ is the MAP estimator under a $\mathcal{N}(0, 10^4\, I_p)$ prior on each weight. The choice of $\lambda$ is therefore a choice of prior, and "tuning $\lambda$ on a validation set" is empirical-Bayes hyperparameter selection. This is the *named-section host* for the cross-cutting concept `weight-decay`, which per CLAUDE.md and the strategic-planning doc never gets its own topic page; the concept lives here.

*The MAP is the starting point for §3.* The Laplace approximation builds a Gaussian posterior approximation centered at $\hat{w}_{\mathrm{MAP}}$. Because $\hat{w}_{\mathrm{MAP}} = \hat{w}_{\mathrm{WD}}$, any standard PyTorch model trained with weight decay is *already* the center of a Laplace approximation; we just haven't computed the surrounding curvature yet. §3 fills in that missing piece.

*The cold-posterior caveat.* Many BNN practitioners empirically find that *tempered* posteriors $p(w \mid \mathcal{D})^{1/T}$ with $T < 1$ — equivalently, larger $\lambda$ than the prior naturally specifies — give better-calibrated predictions than the strict Bayesian $T = 1$. This is the "cold-posterior effect" (Wenzel et al. 2020) and indicates that the Gaussian prior is misspecified for typical training datasets: the data effectively want a stronger regularizer than the principled $\lambda = 1/\tau^2$ delivers. The phenomenon is one of the central open problems in BNNs and is treated formally in §8.5.

#### §2.3 The Bernstein–von Mises asymptotic

What does the weight-space posterior *look like* for large $n$? The classical result is the *Bayesian central-limit theorem*, also known as Bernstein–von Mises:

> **Theorem 2.3 (Bernstein–von Mises).** Let $\{p(\cdot \mid w) : w \in \mathbb{R}^p\}$ be a regular parametric family with $p$ fixed, and suppose the data $\mathcal{D}_n = \{(x_i, y_i)\}_{i=1}^n$ are iid from $p(\cdot \mid w_0)$ for some interior $w_0$. Let $w^*_n$ denote the MLE based on $\mathcal{D}_n$, and let $I(w_0)$ denote the Fisher information matrix at the true parameter. Under standard regularity conditions (see formalstatistics's [`central-limit-theorem`](https://www.formalstatistics.com/topics/central-limit-theorem) topic), as $n \to \infty$:
> $$d_{\mathrm{TV}}\!\Bigl(\,p\bigl(\sqrt{n}(w - w^*_n) \,\big|\, \mathcal{D}_n\bigr) \,,\; \mathcal{N}\!\bigl(0,\, I(w_0)^{-1}\bigr)\,\Bigr) \;\to\; 0$$
> in posterior probability, where $d_{\mathrm{TV}}$ denotes total-variation distance.

The proof is the subject of formalstatistics's `central-limit-theorem` topic. In words: the posterior, recentered at the MLE and rescaled by $\sqrt{n}$, converges in total variation to a Normal distribution with covariance equal to the inverse Fisher information. Equivalently, for large $n$ the un-rescaled posterior is approximately
$$p(w \mid \mathcal{D}_n) \;\approx\; \mathcal{N}\!\Bigl(w^*_n,\; \tfrac{1}{n}\, I(w_0)^{-1}\Bigr),$$
a Gaussian centered at the MLE with covariance the inverse total Fisher information $H_n^{-1} \approx (n I(w_0))^{-1}$.

This is the asymptotic license for §3's Laplace approximation. Under BvM, fitting a Gaussian centered at the MAP with the right curvature gives a posterior approximation that is exact in the $n \to \infty$ limit. The Laplace construction in §3 will use the *observed* Fisher information $H_n = -\nabla^2 \log p(w \mid \mathcal{D}_n) \big|_{w = \hat{w}_{\mathrm{MAP}}}$, which differs from $n I(w_0)$ by a fluctuation that is itself $O(\sqrt{n})$. The point is: there is a principled reason to expect the posterior to be approximately Gaussian centered at the MAP for large $n$, *provided* the regularity conditions hold.

#### §2.4 Why Bernstein–von Mises fails for neural networks

The regularity conditions for BvM are: $p$ fixed, model well-specified, MLE consistent and asymptotically Normal, Fisher information matrix positive-definite at $w_0$, posterior absolutely continuous with respect to the prior. These conditions fail for typical neural networks in three structural ways.

**Hidden-unit permutation symmetry.** Consider an MLP layer with $h$ hidden units, $W \in \mathbb{R}^{d \times h}$ input weights and $b \in \mathbb{R}^h$ biases. Permuting the columns of $W$ (and the corresponding entries of $b$ and the next layer's row weights) gives an *exactly equivalent* function. So the function $f_w$ is invariant under $S_h$, the symmetric group on $h$ elements. The likelihood $p(\mathcal{D} \mid w)$ inherits this symmetry, and so does the posterior. There are at least $h!$ identical modes per hidden layer. For a 32-unit layer, $32! \approx 10^{35}$. The posterior has more than $10^{35}$ identical modes, separated by ridges of low likelihood — interpolating linearly between two modes via $w_t = (1 - t) w_a + t w_b$ does not generally produce another mode, since the loss along the interpolation rises through a barrier. Vanilla MCMC can mix over the connected component of the posterior support that it starts in, but cannot in finite wall-clock time mix between disconnected modes. BvM's "single Gaussian centered at *the* MLE" is wrong globally because there is no single MLE.

**ReLU positive scaling.** For ReLU activations, $\mathrm{ReLU}(c\, z) = c \cdot \mathrm{ReLU}(z)$ for any $c > 0$. So if we multiply the input weights to a hidden unit by $c > 0$ and divide the outgoing weights by $c$, the function is unchanged: $f_w$ is invariant under the action of the multiplicative group $(\mathbb{R}_{>0})^h$ on weights. This is a continuous symmetry, not a discrete one, and it means that even within a single permutation class the MLE is not isolated — it sits on a $h$-dimensional submanifold of equally-good weights. The Fisher information matrix is *singular* along this submanifold (the gradient of the log-likelihood is zero in those directions because the function does not change), so BvM's invocation of $I(w_0)^{-1}$ as a positive-definite covariance fails. Weight decay breaks the symmetry by penalizing rescalings — $\|c\, w_{\mathrm{in}}\|^2 + \|w_{\mathrm{out}} / c\|^2$ is minimized at a specific $c$ — so the *MAP* is locally unique even when the MLE is not. But the underlying likelihood is not strongly identified.

**Over-parametrization.** Modern deep networks have $p \gg n$. ResNet-50 has $\sim\!25 \times 10^6$ parameters; ImageNet has $\sim\!10^6$ training images. BvM operates in the regime $p$ fixed, $n \to \infty$. The opposite regime — $p \to \infty$ with $n$ fixed or growing slowly — is the regime of high-dimensional statistics, and the asymptotics there are different. The infinite-width limit of §9 makes one specific version of this regime tractable (function-space rather than weight-space), but for finite-width over-parametrized networks, BvM gives no asymptotic guidance.

These three failure modes organize the rest of the topic. **Laplace** (§3) gives up on the global picture and approximates the posterior locally at the MAP — accepting the failure of BvM globally and pursuing a local Gaussian that captures local curvature even if it misses the $h!$ permutation copies and the ReLU rescaling submanifold. **MC-dropout** (§4) uses a tractable but rigid variational family that sidesteps the exact-posterior question entirely. **Deep ensembles** (§5) sample $K$ approximate samples from $K$ different initializations, getting some coverage of distinct modes. **SG-MCMC** (§§6–7) accepts the multimodality and tries to mix between modes via Langevin dynamics on minibatches.

> **Remark 2.4 (Why naïve sampling fails).** Three obstacles in compact form, as a checklist for §§3–7. (i) *Multimodality:* $h!$ exact replicas per layer; vanilla MCMC cannot mix in any practical wall-clock time. (ii) *Identifiability:* ReLU positive-scaling submanifold; Fisher information singular within each mode under the un-regularized likelihood. (iii) *Computational cost:* full-data gradients infeasible at deep-learning scale; vanilla HMC needs the gradient at every leapfrog step, an $O(n)$ pass. Each method in §§3–7 sidesteps a subset of these obstacles via a specific technical trick.

#### Code experiment §2

Train 20 MLPs (same architecture as §1, different random seeds), record their final weight vectors, project to 2D via PCA. Compute the loss along a linear interpolation between two of them. PyTorch end-to-end. Total cell runtime ~12 s.

Library calls: `train_mlp` from setup, `torch.nn.utils.parameters_to_vector` to flatten weight tensors into a single $\mathbb{R}^p$ vector for PCA, `numpy.linalg.svd` for the PCA projection, `np.linspace` for the interpolation grid. Verification: print $(t^*, \mathcal{L}(t^*))$ where $t^*$ is the argmax of the loss along the interpolation, plus the trained-model loss floor — expected barrier ratio $\mathcal{L}(t^*)/\mathcal{L}(0)$ at least $3\times$.

#### Viz design intent §2

**Figure 2 (`02_loss_landscape_modes.png`).** Two panels.

**Panel (a)** — *PCA-projected weight vectors.* Scatter of 20 trained MLP weight vectors $\hat{w}^{(s)} \in \mathbb{R}^p$ projected to their first two principal components. Color points by final training loss (uniform — all near zero — but use a slight `viridis` shading to emphasize they all converged). Annotate two visibly distinct points as "mode $a$" and "mode $b$" for use in panel (b). Reader should see discrete clusters of trained-MLP weights — each cluster a permutation/scaling class of the others, with most $h!$-symmetry copies projecting close together so only a few distinct clusters are visible.

**Panel (b)** — *Loss-along-interpolation profile.* For two trained models $\hat{w}^{(a)}, \hat{w}^{(b)}$ from different clusters, evaluate the training loss $\mathcal{L}_{\mathrm{WD}}\bigl((1 - t)\hat{w}^{(a)} + t\hat{w}^{(b)}\bigr)$ at $t \in [0, 1]$ on a 100-point grid. Plot the resulting curve. The curve should rise from the trained-model loss floor up to a barrier and back down — a non-convex ridge separating two modes. Annotate the trained-model floor (dashed horizontal line) and the peak barrier height.

The PNG is the static fallback. The on-page rendering is the React + D3 component `LossLandscapeModesViz.tsx`.

**Interactive controls (v1 deliverable):**

- *Click-to-pick-modes interaction.* Reader clicks any two points in the PCA scatter (panel a) to set the interpolation endpoints. Panel (b) re-renders the loss profile along the new line. Selected points are highlighted in red and blue; default selection is the two points with the largest pairwise PC-distance (matching the notebook's automatic choice).
- *Number-of-models slider* ($N = 5$ to $N = 30$, default $N = 20$). Re-trains and re-projects with the new ensemble size. Re-rendering on slider drag is debounced (~500 ms) and uses cached pre-trained models when possible.
- *Interpolation grid resolution slider* ($M = 20$ to $M = 200$ points, default $M = 100$). Trades smoothness of the loss profile in panel (b) for re-render speed.

The component pre-trains 30 MLPs on first hydration (~15 s, the heaviest pre-compute in the topic), caches their flattened weight vectors and a pre-computed pairwise-distance matrix, and re-renders both panels in <100 ms on click or slider interaction (the loss profile recomputation along a 100-point line is ~50 ms via cached forward passes).

**What the reader should learn from Figure 2 / the §2 viz:** (i) the loss landscape of even a small MLP is genuinely multi-modal; (ii) a straight line in weight space between two modes passes through a region of strictly higher loss — and the click-to-pick interaction lets the reader confirm this for *every* pair of trained models, not just the default pair; (iii) a single Gaussian posterior approximation (Laplace, mean-field VI) is by construction unable to capture more than one mode, and the methods of §5 (deep ensembles) and §§6–7 (SG-MCMC) exist exactly to recover the multi-mode structure that single-Gaussian methods discard.

---

### §3. The Laplace approximation

§2 ends with a problem and a permission. The problem is that the true weight-space posterior $p(w \mid \mathcal{D})$ is intractable: no closed-form normalizer, more than $h!$ identical modes, and continuous symmetries that make the Fisher information singular within each mode. The permission is Bernstein–von Mises: under the regularity conditions the posterior is asymptotically a Gaussian centered at the MLE. The Laplace approximation accepts that the BvM regularity conditions don't strictly hold globally but observes that they hold *locally* at any sufficiently regular minimum of the negative log-posterior — and builds a Gaussian approximation by Taylor-expanding around such a minimum. The construction goes back to Laplace's 1774 calculation of the asymptotic value of $\int e^{-n h(\theta)}\,d\theta$; MacKay's 1992 PhD thesis brought it into Bayesian neural networks; the modern revival is Daxberger et al. 2021 (`laplace-torch`), which scales the construction to ImageNet-class networks via Kronecker-factored curvature.

#### §3.1 The construction

Let $U(w) := -\log p(w \mid \mathcal{D})$ be the negative log-posterior, modulo the unknown constant $\log p(\mathcal{D})$. From §2.2 we have
$$U(w) \;=\; -\sum_{i=1}^n \log p(y_i \mid x_i, w) \;+\; \frac{1}{2\tau^2}\|w\|_2^2 \;+\; \mathrm{const}.$$
Suppose $\hat{w} := \hat{w}_{\mathrm{MAP}} = \arg\min_w U(w)$ is a *local* minimum at which $U$ is twice continuously differentiable and the Hessian $H := \nabla^2 U(w)|_{w = \hat{w}}$ is positive-definite. Taylor-expand $U$ to second order around $\hat{w}$:
$$U(w) \;=\; U(\hat{w}) \;+\; \nabla U(\hat{w})^\top (w - \hat{w}) \;+\; \tfrac{1}{2}(w - \hat{w})^\top H (w - \hat{w}) \;+\; O(\|w - \hat{w}\|^3).$$
At a local minimum $\nabla U(\hat{w}) = 0$, so the linear term vanishes. Dropping the higher-order terms — the *Laplace approximation* itself — we get
$$U(w) \;\approx\; U(\hat{w}) \;+\; \tfrac{1}{2}(w - \hat{w})^\top H (w - \hat{w}).$$
Exponentiating and renormalizing,
$$p(w \mid \mathcal{D}) \;\approx\; q(w) \;:=\; \mathcal{N}\bigl(w \,\big|\, \hat{w},\, H^{-1}\bigr).$$
The construction takes a Gaussian whose mean is the MAP and whose precision is the local curvature of the negative log-posterior. Two further facts come for free.

The Hessian decomposes additively. From the form of $U$,
$$H \;=\; -\nabla^2_w \log p(\mathcal{D} \mid w)\big|_{w = \hat{w}} \;+\; \frac{1}{\tau^2} I_p \;=:\; H_{\mathrm{data}} + H_{\mathrm{prior}},$$
the data Hessian plus the prior precision. The prior contribution is constant in $w$, isotropic, and explicitly positive-definite; it stabilizes the data Hessian when the latter has near-zero or negative eigenvalues from saddle directions or the §2.4 ReLU rescaling submanifold. In practice we either compute $H$ directly or compute $H_{\mathrm{data}}$ and add $\tau^{-2} I_p$ explicitly.

The marginal likelihood comes for free as a side effect. The Laplace approximation also gives a closed-form estimate of $p(\mathcal{D})$:
$$\log p(\mathcal{D}) \;\approx\; -U(\hat{w}) \;+\; \frac{p}{2}\log(2\pi) \;-\; \frac{1}{2}\log\det H,$$
known as the *Laplace marginal-likelihood approximation*. This is the basis of variational Bayesian model selection (planned `variational-bayes-for-model-selection`) and the BIC's asymptotic form (cross-link to formalstatistics's `model-selection-and-information-criteria`).

> **Definition 3.1 (Laplace approximation).** Given a twice-differentiable negative log-posterior $U(w)$ with positive-definite Hessian $H = \nabla^2 U(\hat{w})$ at a local minimum $\hat{w}$, the *Laplace approximation* to the posterior is the Gaussian
> $$q_{\mathrm{Lap}}(w) \;:=\; \mathcal{N}\bigl(w \,\big|\, \hat{w},\, H^{-1}\bigr).$$

#### §3.2 Asymptotic exactness — and where it stops

> **Theorem 3.2 (Asymptotic exactness of Laplace under BvM).** Suppose the conditions of Theorem 2.3 hold. Let $\hat{w}_n$ be the MAP based on $n$ iid observations and $H_n$ the negative-log-posterior Hessian at $\hat{w}_n$. Let $q_{\mathrm{Lap}}^{(n)}(w) = \mathcal{N}(w \,|\, \hat{w}_n, H_n^{-1})$ be the Laplace approximation. Then
> $$d_{\mathrm{TV}}\!\bigl(p(w \mid \mathcal{D}_n),\; q_{\mathrm{Lap}}^{(n)}(w)\bigr) \;\to\; 0 \quad \text{as } n \to \infty$$
> in posterior probability.

*Proof.* Two ingredients combine. First, $\hat{w}_n = w^*_n + O(1/n)$, where $w^*_n$ is the MLE: differentiating $U_n(w) = -\log p(\mathcal{D}_n \mid w) - \log p(w)$ and setting to zero gives $\nabla[-\log p(\mathcal{D}_n \mid w)]\big|_{\hat{w}_n} = -\nabla \log p(w)\big|_{\hat{w}_n}$. The right-hand side is $\hat{w}_n / \tau^2$ for the Gaussian prior, so $\nabla[\text{NLL}]\big|_{\hat{w}_n} = \hat{w}_n/\tau^2$. The MLE satisfies $\nabla[\text{NLL}]\big|_{w^*_n} = 0$, so $\hat{w}_n - w^*_n$ solves the first-order condition $\nabla^2[\text{NLL}]\big|_{w^*_n}(\hat{w}_n - w^*_n) = -w^*_n/\tau^2 + O(\|\hat{w}_n - w^*_n\|^2)$, which gives $\hat{w}_n - w^*_n = O(\|H_n\|^{-1}) = O(1/n)$ since the data Hessian scales as $n$.

Second, $H_n / n \to I(w_0)$ in probability by the law of large numbers applied to the per-observation Fisher information contributions. So $H_n^{-1} \to (n I(w_0))^{-1}$ at first order.

Putting these together, $\mathcal{N}(\hat{w}_n, H_n^{-1}) \to \mathcal{N}(w^*_n, (n I(w_0))^{-1})$ in total variation. Theorem 2.3 says the true posterior also converges to this limit. By the triangle inequality for $d_{\mathrm{TV}}$, the Laplace approximation converges to the true posterior. $\square$

The §2.4 caveats apply with full force here. The proof above assumes (i) $p$ fixed, (ii) $H_n$ positive-definite at $\hat{w}_n$, and (iii) the BvM regularity conditions. For neural networks: $p$ is not fixed in any meaningful sense (over-parametrization), $H_{\mathrm{data}}$ has near-zero eigenvalues from the ReLU rescaling submanifold (we rescue this with $H_{\mathrm{prior}} = \tau^{-2} I_p$, but the result is a Laplace approximation around an artificially regularized minimum), and the global posterior has many modes that the Laplace approximation around any one of them cannot represent. So Theorem 3.2 should be read as a *local* guarantee that the Gaussian fit is the best second-order approximation around $\hat{w}$, plus an *asymptotic* guarantee that this best approximation converges to the truth in well-specified parametric settings. For BNNs we get the local part and lose most of the asymptotic part.

This is enough for a useful method. The Laplace BNN's predictive mean recovers the point estimate, its predictive variance grows away from the data exactly because the local quadratic in $U$ flattens away from $\hat{w}$, and the construction needs no sampling at all once $\hat{w}$ and $H$ are in hand. What it loses — multi-modality, ReLU-rescaling spread — is exactly what §§5–7 will recover.

#### §3.3 The predictive distribution under linearization

A Gaussian posterior over weights does not directly give a Gaussian posterior over outputs, because $f_w(x)$ is a nonlinear function of $w$. The integral
$$p(y^* \mid x^*, \mathcal{D}) \;=\; \int p(y^* \mid x^*, w)\,q_{\mathrm{Lap}}(w)\,dw$$
has no closed form. The standard reduction is *first-order linearization* of the network around the MAP:
$$f_w(x) \;\approx\; f_{\hat{w}}(x) \;+\; J(x)^\top (w - \hat{w}), \qquad J(x) := \nabla_w f_w(x)\big|_{w = \hat{w}}.$$
Under this linearization, $f_w(x)$ is an *affine* function of $w$. If $w \sim \mathcal{N}(\hat{w}, H^{-1})$, then the logit $f_w(x)$ is Gaussian with mean $f_{\hat{w}}(x)$ and variance $J(x)^\top H^{-1} J(x)$.

> **Proposition 3.3 (Linearized Laplace predictive).** Define $\tilde{m}(x) := f_{\hat{w}}(x)$ and $\tilde{v}(x) := J(x)^\top H^{-1} J(x)$. Under the linearized Laplace approximation, the logit at $x$ is approximately Gaussian:
> $$f_w(x) \mid \mathcal{D} \;\stackrel{\mathrm{lin}}{\sim}\; \mathcal{N}\bigl(\tilde{m}(x),\, \tilde{v}(x)\bigr).$$
> The class-1 probability under the moment-matched probit approximation is
> $$\hat{p}(y = 1 \mid x, \mathcal{D}) \;\approx\; \sigma\!\left(\frac{\tilde{m}(x)}{\sqrt{1 + \pi \tilde{v}(x) / 8}}\right).$$

The factor $\pi/8$ comes from MacKay's 1992 Gaussian-CDF approximation to the sigmoid: $\int \sigma(z) \mathcal{N}(z; m, v)\,dz \approx \sigma(m / \sqrt{1 + \pi v / 8})$. The reader sees the predictive variance $\tilde{v}(x)$ enter explicitly: where $\tilde{v}(x)$ is small (data-dense regions, where the network is well-determined), the predictive collapses to $\sigma(\tilde{m}(x))$ — the point estimate from §1. Where $\tilde{v}(x)$ is large (off-distribution regions, where small changes in $w$ produce large changes in $f_w(x)$), the predictive is squashed toward $0.5$ — the model's confession of ignorance.

In practice we have two implementation choices for the predictive: (a) closed-form via Prop 3.3, computing $J(x)$ at each test point and the matrix product $J(x)^\top H^{-1} J(x)$; (b) Monte Carlo, sampling $w^{(s)} \sim \mathcal{N}(\hat{w}, H^{-1})$ and averaging $\sigma(f_{w^{(s)}}(x))$ over $s = 1, \ldots, S$ samples. For the §3.4 Two Moons figure we use (b) because it lets us also visualize sample decision boundaries; in production BNN libraries the closed-form (a) is faster.

#### §3.4 Practical curvature

Computing and storing $H \in \mathbb{R}^{p \times p}$ for $p$ in the millions is impossible — $p^2$ memory, $O(p^3)$ inversion. Three standard reductions trade fidelity for tractability.

*Last-layer Laplace* (Daxberger et al. 2021). Fix all but the final linear layer's weights at their MAP values; do Laplace only over the last layer. The last layer typically has $p_L \ll p$ parameters — for a 32-unit penultimate layer × $C$ output classes, $p_L \approx 32 C$. $H_{\mathrm{last\text{-}layer}} \in \mathbb{R}^{p_L \times p_L}$ is cheap. Empirically competitive with full Laplace on a wide range of tasks; the underlying intuition is that in over-parametrized networks, most of the weight-space uncertainty that affects predictions is concentrated in the final layer.

*Kronecker-factored approximate curvature* (KFAC; Martens & Grosse 2015). For each layer, approximate the Fisher information block as a Kronecker product $H_\ell \approx A_\ell \otimes G_\ell$, where $A_\ell$ is the input-activation second moment and $G_\ell$ is the output-gradient second moment. Storage cost reduces from $p_\ell^2$ to $\dim(A_\ell)^2 + \dim(G_\ell)^2$ per layer. The Kronecker assumption is structural (it ignores cross-layer covariance entirely), but matches the block-diagonal-plus-Kronecker structure of the Fisher information of natural-gradient methods.

*Diagonal Fisher.* Take only the diagonal of $H$. Cheapest possible — $O(p)$ memory — but discards all weight correlations. Often too aggressive: the diagonal Hessian eigenvalues are unreliable estimates of the full spectrum, and diagonal-Fisher Laplace BNNs frequently underestimate predictive variance in correlated directions.

For the §3.5 Two Moons example with $p = 2241$ parameters, full Laplace is feasible. For real-world BNNs, last-layer or KFAC is the practical choice; the `laplace-torch` library (Daxberger et al. 2021) implements all three reductions and is the default tool.

#### §3.5 Algorithm and worked example

> **Algorithm 3.4 (Laplace-fit BNN).**
> *Input:* training data $\mathcal{D}$, network $f_w$, prior scale $\tau^2$, sample count $S$.
> *Output:* predictive distribution $\hat{p}(y \mid x)$ at test inputs.
> 1. Compute $\hat{w} := \arg\min_w U(w)$ via Adam or SGD with weight decay $\lambda = 1/\tau^2$.
> 2. Compute the data Hessian $H_{\mathrm{data}} := -\nabla^2 \log p(\mathcal{D} \mid w)|_{w = \hat{w}}$ via autodiff (or one of the §3.4 reductions).
> 3. Form $H := H_{\mathrm{data}} + \tau^{-2} I_p$ and stabilize: $H \leftarrow H + \delta I_p$ for small $\delta > 0$ if any eigenvalue is non-positive.
> 4. Cholesky-factor $H = L L^\top$.
> 5. Sample $z^{(s)} \sim \mathcal{N}(0, I_p)$ for $s = 1, \ldots, S$; set $w^{(s)} := \hat{w} + L^{-\top} z^{(s)}$.
> 6. Predict via $\hat{p}(y \mid x) := S^{-1} \sum_s p(y \mid x, w^{(s)})$, or via Prop 3.3's closed-form.

The §1 first-seed model serves as $\hat{w}$. Step 2 uses `torch.autograd.functional.hessian` on the negative-log-posterior of the full training set; for $p = 2241$ this is $\sim\!2$ s on CPU. The $\tau^{-2} I_p$ contribution is automatic (we trained with weight decay $\lambda = 10^{-4}$, so $\tau^2 = 10^4$ and the prior precision contribution is $10^{-4} I_p$). Stabilization $\delta = 10^{-3}$ handles the residual ReLU-rescaling near-singularity. The Cholesky factorization on a $2241 \times 2241$ matrix is $\sim\!1$ s; sampling $S = 100$ posteriors and predicting on the $40{,}000$-point grid is $\sim\!3$ s.

#### Code experiment §3

PyTorch end-to-end. Total cell runtime ~10 s. Key library calls: `torch.autograd.functional.hessian` for the Hessian, `torch.linalg.cholesky` and `torch.linalg.solve_triangular` for the factorization and sampling. Verification: print Hessian condition number (expected $10^4$–$10^7$), predictive-std ratio off-support / on-support (expected at least $2\times$), and the mean accuracy of the Laplace predictive on the training set (should match the point-estimate accuracy to within $1\%$).

#### Viz design intent §3

**Figure 3 (`03_laplace_two_moons.png`).** Three panels on the standard $200 \times 200$ Two Moons grid.

**Panel (a)** — *Laplace predictive mean.* Heatmap of $\mathbb{E}_{q_{\mathrm{Lap}}}[\sigma(f_w(x))]$ via $S = 100$ Monte Carlo samples. Colormap `RdBu_r`, $v_{\min}=0$, $v_{\max}=1$. Reader should see that the mean is visually indistinguishable from the §1 point-estimate predictive — the Laplace approximation does not change the *mean* prediction, only its uncertainty.

**Panel (b)** — *Laplace predictive standard deviation.* Heatmap of $\mathrm{SD}_{q_{\mathrm{Lap}}}[\sigma(f_w(x))]$. Colormap `viridis`. Reader should see the desideratum from §1's panel (c) recovered through a different mechanism: dark (low-std) regions hugging the data, bright (high-std) regions far from any data.

**Panel (c)** — *Sampled decision boundaries.* Overlay $20$ of the $S = 100$ Laplace-sampled $0.5$-contours on the data, each as a translucent blue line. Reader should see narrow agreement near the data, fanning-out far from the data — analogous to §1's panel (b) but with all 20 boundaries from a *single* trained model's local Gaussian rather than 20 independently-trained models.

The PNG is the static fallback. The on-page rendering is the React + D3 component `LaplaceBNNViz.tsx`.

**Interactive controls (v1 deliverable):**

- *Prior scale slider* ($\tau^2$ on a log scale from $10^2$ to $10^6$, default $10^4$ matching the §1 weight-decay $\lambda = 10^{-4}$). Recomputes $H = H_{\mathrm{data}} + \tau^{-2} I_p$ and re-samples from $\mathcal{N}(\hat w, H^{-1})$. Re-renders all three panels. Reader sees: small $\tau^2$ tightens the posterior and shrinks predictive std everywhere; large $\tau^2$ inflates predictive std especially off-distribution.
- *Curvature reduction dropdown* (`Full Hessian` / `Last-layer` / `Diagonal Fisher`, default `Full Hessian`). Switches between the §3.4 reductions. Re-computes $H$ and re-samples. Reader sees: last-layer Laplace gives qualitatively similar predictive std at much lower compute cost; diagonal Fisher visibly under-estimates off-distribution variance — concrete demonstration of the §3.4 caveat.
- *Sample-count slider* ($S = 10$ to $S = 200$, default $S = 100$). Re-renders panel (a)'s mean and panel (b)'s std with fewer or more MC samples. Reader sees the Monte Carlo noise floor at small $S$.

The component precomputes the data Hessian $H_{\mathrm{data}}$ once (~2 s on first hydration), caches its eigendecomposition, and recomputes $H = H_{\mathrm{data}} + \tau^{-2} I_p$ + Cholesky factor on every $\tau^2$ change in <100 ms. Sample regeneration is <50 ms per slider tick.

**What the reader should learn from Figure 3:** (i) Laplace builds a Gaussian over weights from the MAP and its local curvature, with no extra training; (ii) the predictive *mean* is the point estimate, so Laplace doesn't change accuracy; (iii) the predictive *standard deviation* recovers the §1 desideratum — uncertainty grows where the data is sparse — through a single-trained-model mechanism that is much cheaper than the §5 deep ensemble; (iv) but the local-Gaussian story is exactly that, *local*: panel (c)'s sampled boundaries fan out within one mode of the loss landscape, and miss the multi-mode structure §2's Figure 2 showed and §5 will recover.

---

### §4. MC-dropout as approximate variational inference

The Laplace approximation of §3 builds a Gaussian posterior over weights from a single point estimate plus its local curvature. The construction is principled but expensive once the network is large — full Laplace needs $p^2$ Hessian storage, and even the §3.4 reductions (last-layer, KFAC) require non-trivial implementation effort. *MC-dropout* (Gal & Ghahramani 2016) is the cheap end of the BNN spectrum: it observes that a dropout-regularized network, trained the standard way, is *already* doing variational inference under a specific variational family, and that the only change needed at test time to extract a predictive distribution is to leave dropout on and average over $T$ stochastic forward passes. No new training, no Hessian, no extra hyperparameters beyond what dropout already requires. The trade is that the variational family is rigid in a way that systematically underestimates epistemic uncertainty in some regimes; §8's calibration analysis quantifies the trade.

#### §4.1 Recap: dropout as a regularizer

*Dropout* (Srivastava et al. 2014) trains a network by, at each minibatch, sampling a Bernoulli mask $b_\ell \in \{0, 1\}^{h_{\ell-1}}$ for each layer's input activations with $b_{\ell, j} \sim \mathrm{Bernoulli}(\rho_\ell)$, multiplying activations elementwise by $b_\ell$ and rescaling by $1/\rho_\ell$ so the post-mask expected activation is unchanged. The forward pass uses the masked activations; the backward pass differentiates only through the unmasked units. At test time, dropout is conventionally *turned off* — the masks are replaced by the mean activation $\rho_\ell$ — and the resulting deterministic prediction is interpreted as an approximate average over the implicit ensemble of "thinned" subnetworks.

The standard story stops here: dropout is a regularizer, the deterministic test-time prediction is the answer, full stop. Gal & Ghahramani 2016 disrupt this by proving that the standard dropout training loss is, up to constants and a fixed prior choice, the negative ELBO of a specific variational posterior — and that to extract that posterior's predictive distribution at test time, you don't turn dropout off, you leave it on and Monte-Carlo over $T$ stochastic forward passes.

#### §4.2 The Bernoulli variational family

Let the network have $L$ weight matrices $W_\ell \in \mathbb{R}^{h_{\ell-1} \times h_\ell}$, $\ell = 1, \ldots, L$. For each layer, define a *Bernoulli mask matrix* $Z_\ell$ as a diagonal matrix with $\mathrm{Bern}(\rho_\ell)$ entries on the diagonal:
$$Z_\ell = \mathrm{diag}(b_\ell), \qquad b_\ell \in \{0, 1\}^{h_{\ell-1}}, \qquad b_{\ell, j} \stackrel{\mathrm{iid}}{\sim} \mathrm{Bernoulli}(\rho_\ell).$$
The *variational weight matrix* at layer $\ell$ is the random matrix $W_\ell^{\mathrm{var}} := M_\ell Z_\ell$, where $M_\ell \in \mathbb{R}^{h_{\ell-1} \times h_\ell}$ is a *learnable mean weight matrix*. Equivalently, the $j$-th column of $W_\ell^{\mathrm{var}}$ is either the $j$-th column of $M_\ell$ (with probability $\rho_\ell$) or the zero column (with probability $1 - \rho_\ell$).

> **Definition 4.1 (Bernoulli variational family).** Let $\boldsymbol{\rho} = (\rho_1, \ldots, \rho_L) \in (0, 1]^L$ be a vector of dropout retain-probabilities, fixed in advance. The *Bernoulli variational family* over network weights is
> $$\mathcal{Q}_{\mathrm{Bern}}(\boldsymbol{\rho}) \;=\; \Bigl\{\, q_{\boldsymbol{M}}(W_1, \ldots, W_L) \;:\; W_\ell = M_\ell \,\mathrm{diag}(b_\ell),\;\; b_{\ell, j} \stackrel{\mathrm{iid}}{\sim} \mathrm{Bern}(\rho_\ell) \,\Bigr\}.$$
> The variational parameters are the *mean weight matrices* $\boldsymbol{M} = (M_1, \ldots, M_L)$.

This is a *discrete* variational family. At each evaluation, $W_\ell$ is one of $2^{h_{\ell-1}}$ possible matrices, parametrized by which subset of $M_\ell$'s columns are zeroed. The family is rigid: $\rho_\ell$ is a fixed hyperparameter, not learned (Concrete dropout in §4.5 lifts this), and the structure of the variational posterior is determined entirely by the network's architecture and the choice of which activations to drop.

#### §4.3 The Gal–Ghahramani equivalence

> **Theorem 4.2 (Dropout training is variational inference).** Let $\mathcal{L}_{\mathrm{drop}}(\boldsymbol{M})$ be the standard dropout training objective: minibatch-averaged cross-entropy of the network with Bernoulli-masked weights $W_\ell = M_\ell \,\mathrm{diag}(b_\ell)$, with $L_2$ weight decay $\lambda$ on the mean weights $\boldsymbol{M}$. Suppose the prior over weights is the layer-wise isotropic Gaussian $p(M_\ell) = \mathcal{N}(0, \tau^2 I)$ with $\tau^2 = (1 - \rho_\ell) / (2 N \lambda)$, where $N$ is the size of the training set. Then
> $$\mathcal{L}_{\mathrm{drop}}(\boldsymbol{M}) \;=\; -\mathrm{ELBO}\bigl(q_{\boldsymbol{M}} \,\big\|\, p(\cdot \mid \mathcal{D})\bigr) \;+\; \mathrm{const},$$
> where the ELBO is the variational evidence lower bound of $q_{\boldsymbol{M}} \in \mathcal{Q}_{\mathrm{Bern}}(\boldsymbol{\rho})$ against the posterior $p(\boldsymbol{M} \mid \mathcal{D})$ of Definition 2.1's BNN with the prior above. Minimizing $\mathcal{L}_{\mathrm{drop}}(\boldsymbol{M})$ over $\boldsymbol{M}$ is equivalent to maximizing the ELBO, hence to reverse-KL minimization within $\mathcal{Q}_{\mathrm{Bern}}(\boldsymbol{\rho})$.

*Proof.* The ELBO decomposes (cf. [Variational Inference](/topics/variational-inference) Definition 2.1) as
$$\mathrm{ELBO}(q) \;=\; \mathbb{E}_q[\log p(\mathcal{D} \mid w)] \;-\; \mathrm{KL}\bigl(q \,\big\|\, p\bigr).$$
For $q = q_{\boldsymbol{M}} \in \mathcal{Q}_{\mathrm{Bern}}(\boldsymbol{\rho})$, the expected log-likelihood is the expectation over Bernoulli masks of the network's log-likelihood at the masked weights:
$$\mathbb{E}_{q_{\boldsymbol{M}}}[\log p(\mathcal{D} \mid w)] \;=\; \mathbb{E}_{b_1, \ldots, b_L}\!\left[\sum_{i=1}^N \log p(y_i \mid x_i, M_1 \mathrm{diag}(b_1), \ldots, M_L \mathrm{diag}(b_L))\right].$$
Standard dropout training Monte-Carlo-estimates this expectation by drawing one mask per minibatch, which is the unbiased single-sample estimator of the inner expectation.

For the KL term, $q_{\boldsymbol{M}}$ has all of its mass on points of the form $W_\ell = M_\ell \mathrm{diag}(b_\ell)$. The KL of $q_{\boldsymbol{M}}$ to a continuous Gaussian prior $p(W_\ell) = \mathcal{N}(0, \tau^2 I)$ is technically infinite (a discrete distribution against a continuous one), so Gal & Ghahramani treat the variational family as a *limit* of Gaussian approximations whose covariance shrinks to zero around the discrete support points. After the algebra (Gal & Ghahramani 2016 Appendix), the KL term reduces, up to additive constants in $\boldsymbol{M}$, to
$$\mathrm{KL}\bigl(q_{\boldsymbol{M}} \,\big\|\, p\bigr) \;=\; \sum_{\ell = 1}^L \frac{1 - \rho_\ell}{2\tau^2}\,\|M_\ell\|_F^2 \;+\; \mathrm{const}.$$
With $\tau^2 = (1 - \rho_\ell)/(2 N \lambda)$ this becomes $N \lambda \sum_\ell \|M_\ell\|_F^2$, which is the $L_2$-weight-decay penalty in the standard dropout loss. Putting the two pieces together, $-\mathrm{ELBO}(q_{\boldsymbol{M}})$ equals $\mathcal{L}_{\mathrm{drop}}(\boldsymbol{M})$ up to a constant, as claimed. $\square$

The theorem's content is that the same algorithm — minibatch SGD on cross-entropy plus $L_2$, with Bernoulli masks at each forward pass — can be read in two ways. The frequentist reads it as "regularized empirical-risk minimization with dropout noise"; the Bayesian reads it as "variational inference under the Bernoulli family." Both readings yield the same trained network. The Bayesian reading does, however, instruct us to do something different at *test time*.

#### §4.4 The MC-dropout predictive

The standard test-time recipe — turn dropout off and use the deterministic mean network — corresponds, in the variational reading, to evaluating $f_{\boldsymbol{M}}(x)$ at the mean weights, which is the *MAP-style point estimate* under $q_{\boldsymbol{M}}$. To get a *Bayesian* predictive distribution we leave dropout on and Monte-Carlo:

> **Proposition 4.3 (MC-dropout predictive).** Let $\boldsymbol{M}$ be the mean weights of a trained dropout network and let $\hat{f}_t(x)$ denote the output at $x$ produced by a stochastic forward pass with a *fresh* set of Bernoulli masks drawn at every layer, $t = 1, \ldots, T$. Then under the Bernoulli variational posterior $q_{\boldsymbol{M}} \in \mathcal{Q}_{\mathrm{Bern}}(\boldsymbol{\rho})$:
> $$\hat{p}(y = 1 \mid x, \mathcal{D}) \;\approx\; \frac{1}{T} \sum_{t=1}^T \sigma\bigl(\hat{f}_t(x)\bigr),$$
> $$\widehat{\mathrm{Var}}_{q_{\boldsymbol{M}}}[\sigma(f_w(x))] \;\approx\; \frac{1}{T}\sum_{t=1}^T \sigma\bigl(\hat{f}_t(x)\bigr)^2 - \left(\frac{1}{T}\sum_{t=1}^T \sigma\bigl(\hat{f}_t(x)\bigr)\right)^2.$$
> Both estimators converge in $T$ at the standard $O(T^{-1/2})$ Monte Carlo rate.

*Proof.* Direct application of the law of large numbers to iid samples $\{\hat{f}_t(x)\}_{t=1}^T$ from the variational predictive. The first identity is the sample mean of the bounded random variable $\sigma(f_w(x)) \in [0, 1]$ under $q_{\boldsymbol{M}}$, which has variance at most $1/4$, so by the CLT the error is $O(T^{-1/2})$. The second identity is the corresponding sample variance, which converges at the same rate. $\square$

In code, this is two lines: keep the model in `train()` mode (which leaves dropout active) at test time and run $T$ forward passes. The single hyperparameter is $T$. Practical choice: $T \in [10, 100]$ for visual-quality predictive heatmaps, $T \geq 50$ for stable calibration-metric estimates (§8).

#### §4.5 Limits and extensions

The Bernoulli family is rigid in a specific way: each weight column is either kept exactly (with probability $\rho$) or zeroed exactly (with probability $1 - \rho$). The family cannot represent continuous deviations from the mean weights — it has zero probability mass on weight matrices that are slightly perturbed from $M_\ell$ but not exactly some Bernoulli-masked version. Compared to the §3 Laplace family (a full Gaussian with covariance $H^{-1}$), MC-dropout has dramatically less expressive power. This shows up in three places.

> **Remark 4.4 (Limits of MC-dropout).**
>
> *Underestimated epistemic variance in over-parametrized regimes.* When the network has many redundant units, dropout's mask-induced variance is largely cancelled by redundancy — the predictive variance saturates at a value determined by the dropout rate, not by how far the test point is from the data. Empirical studies (Foong et al. 2019, "On the Expressiveness of Approximate Inference in Bayesian Neural Networks") show MC-dropout predictive variance converges to a constant in the network width even in regions where the true posterior predictive variance grows.
>
> *No multi-modality.* The Bernoulli family, like the §3 Laplace family, parametrizes a single mode of the loss landscape. The §2.4 hidden-unit permutation symmetry is invisible to MC-dropout — the variational posterior captures noise around the trained mean weights, not separated modes. §5 deep ensembles will recover multi-modality; MC-dropout cannot.
>
> *Concrete dropout* (Gal et al. 2017) lifts the rigid-$\rho$ limitation by making the dropout rates $\rho_\ell$ learnable parameters optimized jointly with $\boldsymbol{M}$ via a continuous relaxation of the Bernoulli (the Concrete distribution, Maddison et al. 2017). *Variational dropout* (Kingma et al. 2015) replaces Bernoulli masks with continuous Gaussian multiplicative noise, enabling weight-correlation modeling at the cost of more careful KL accounting. *Structured dropout* (DropConnect, DropBlock for CNNs, attention-head dropout for transformers) adapts the Bernoulli family to architecture-specific symmetries.

#### Code experiment §4

Train a 3-hidden-layer × 32-unit MLP with `Dropout(p=0.1)` after each ReLU (so $\rho = 0.9$). Standard `Adam(weight_decay=1e-4)` for 200 epochs, same recipe as §1's point estimate. Test-time predictive: keep model in `train()` mode (dropout active), run $T = 100$ stochastic forward passes on the prediction grid. PyTorch end-to-end. Total cell runtime ~5 s.

Verification: print the off/on predictive-std ratio (expected at least $2\times$, but likely lower than §3's Laplace ratio per Rem 4.4's "underestimated epistemic variance" caveat), and the train accuracy of the MC-dropout predictive (should match the point-estimate to within $1\%$).

#### Viz design intent §4

**Figure 4 (`04_mcdropout_two_moons.png`).** Three panels on the standard $200 \times 200$ Two Moons grid, mirroring §3's panel layout for visual comparability across method sections.

**Panel (a)** — *MC-dropout predictive mean.* Heatmap of $T^{-1} \sum_t \sigma(\hat{f}_t(x))$ via $T = 100$ Monte Carlo samples. Colormap `RdBu_r`, $v_{\min}=0$, $v_{\max}=1$. Reader should see a decision surface visually similar to §3 panel (a) and §1 panel (a) — the *mean* prediction is essentially the point estimate.

**Panel (b)** — *MC-dropout predictive standard deviation.* Heatmap of the sample standard deviation of $\sigma(\hat{f}_t(x))$ over $T = 100$ samples. Colormap `viridis`. Reader should see the same qualitative pattern as §3 panel (b) — bright off-distribution, dark near data — but with *flatter* off-distribution variance, illustrating the Foong et al. saturation phenomenon (Rem 4.4).

**Panel (c)** — *MC-dropout sampled decision boundaries.* Overlay 20 of the $T = 100$ MC-dropout sampled $0.5$-contours on the data, each as a translucent green line (`COLOR_DROPOUT`). Reader should see narrower fanning-out off-distribution than §3 panel (c)'s Laplace samples — concrete visual confirmation of the lower epistemic-variance regime MC-dropout captures.

The PNG is the static fallback. The on-page rendering is the React + D3 component `MCDropoutBNNViz.tsx`.

**Interactive controls (v1 deliverable):**

- *Dropout rate slider* ($1 - \rho$ from $0.0$ to $0.5$, default $0.10$). At $1 - \rho = 0$ the network reduces to the §1 point estimate. As $1 - \rho$ grows the predictive variance grows but the *mean* prediction also degrades — over-aggressive dropout breaks the model. Re-trains the dropout MLP at the new rate (~5 s blocking) and re-samples; re-renders all three panels. Use the loading-state JSX swap pattern.
- *Forward-pass count slider* ($T = 10$ to $T = 200$, default $T = 100$). Recomputes the predictive mean and std from $T$ stochastic forward passes through the *cached* trained model. <50 ms per slider tick.
- *Toggle: dropout on at test* (boolean, default on). When off, the network reverts to deterministic test-time prediction (`model.eval()`), revealing that the *mean* prediction is the same as the standard deterministic predictor — only the *variance* requires the test-time stochasticity. Pedagogically important: many readers conflate dropout-as-regularizer with dropout-as-VI.

The component pre-trains a default dropout MLP on first hydration (~5 s), caches its weights, and re-renders panels (a)–(c) in <50 ms on $T$-slider or toggle interactions. The dropout-rate slider re-trains and is debounced (~500 ms).

**What the reader should learn from Figure 4:** (i) MC-dropout produces a Bayesian predictive at almost zero extra cost — train a dropout network the standard way, just leave dropout on at test time and average $T$ forward passes; (ii) the predictive captures epistemic uncertainty qualitatively in the right regions; (iii) it underestimates the magnitude of epistemic uncertainty far off-distribution compared to Laplace, and §8's calibration metrics quantify this gap directly.

---

### §5. Deep ensembles as a function-space posterior proxy

The §3 Laplace approximation and §4 MC-dropout share a structural limitation: each parametrizes a single mode of the loss landscape. Laplace's Gaussian is centered at one MAP; MC-dropout's Bernoulli posterior puts its mass around one mean weight matrix $\boldsymbol{M}$. Neither captures the §2.4 hidden-unit permutation symmetry, the §2.4 ReLU rescaling submanifold, or any of the genuinely *separated* modes that §2's Figure 2 made visible. *Deep ensembles* (Lakshminarayanan, Pritzel & Blundell 2017) take the opposite tack: don't fit a parametric posterior at all. Instead, train $K$ networks from independent random initializations, treat the $K$ trained weight vectors as approximate samples from $K$ different modes of the posterior, and report the ensemble's predictive as an equal-weighted mixture. The construction is brutally simple — the entire method is "train more times" — but it consistently outperforms more sophisticated single-mode methods on calibration and out-of-distribution detection benchmarks. Wilson & Izmailov 2020 articulate the why: deep ensembles are a coarse but legitimate function-space approximation to the Bayesian posterior predictive, in a regime where the function-space view dominates the weight-space view.

#### §5.1 The construction

> **Definition 5.1 (Deep ensemble).** Let $K \in \mathbb{N}$ be an ensemble size. A *deep ensemble* of size $K$ is a collection $\{\hat{w}^{(k)}\}_{k=1}^K$ of MAP estimates obtained by training the same model architecture from $K$ independent random initializations on the same training data $\mathcal{D}$. The *ensemble predictive distribution* is the equal-weighted mixture
> $$\hat{p}_{\mathrm{ens}}(y \mid x, \mathcal{D}) \;:=\; \frac{1}{K}\sum_{k=1}^K p(y \mid x, \hat{w}^{(k)}).$$

Two technical points are worth pulling out before the theory.

First, *random initialization is essential* — without it, the $K$ networks are identical. The standard recipe in PyTorch is to draw $\hat{w}^{(k)}_{0} \sim$ Kaiming-normal (or Xavier-normal, for older networks), with the random seed varied across $k$. The data and the optimizer are otherwise identical: same minibatches, same learning rate, same weight decay, same number of epochs.

Second, *the ensemble is heterogeneous in function space even when each member is well-trained*. Two networks trained from different initializations on the same data converge to different points in weight space (§2.4's argument that there is no unique MLE), and although they typically agree on the training data, they extrapolate differently — exactly the §1 observation that motivated the topic.

The construction has no hyperparameters beyond $K$. No Hessians, no variational parameters, no Bernoulli rates, no learning-rate schedules for sampling. It is the simplest possible BNN method and, as §8 will show, often the most accurate.

#### §5.2 The function-space-posterior interpretation

Why does training $K$ models with different initializations approximate the Bayesian posterior predictive? The argument has two layers.

> **Theorem 5.2 (Mode-collapse limit of the deep ensemble).** Suppose the weight-space posterior $p(w \mid \mathcal{D})$ is supported on $K$ disjoint regions $\mathcal{R}_1, \ldots, \mathcal{R}_K$ of equal posterior mass $1/K$, and that within each region $\mathcal{R}_k$ the posterior collapses to a Dirac at a point $w^{(k)}_*$ — i.e., $p(w \mid \mathcal{D}) = K^{-1} \sum_{k=1}^K \delta(w - w^{(k)}_*)$. Suppose further that each independent training run produces $\hat{w}^{(k)} = w^{(k)}_*$ exactly, with each region hit by exactly one run. Then the ensemble predictive equals the posterior predictive:
> $$\hat{p}_{\mathrm{ens}}(y \mid x, \mathcal{D}) \;=\; p(y \mid x, \mathcal{D}) \;:=\; \int p(y \mid x, w)\,p(w \mid \mathcal{D})\,dw.$$

*Proof.* By assumption, $p(w \mid \mathcal{D}) = K^{-1} \sum_k \delta(w - w^{(k)}_*)$. So
$$p(y \mid x, \mathcal{D}) \;=\; \int p(y \mid x, w)\,K^{-1}\sum_{k=1}^K \delta(w - w^{(k)}_*)\,dw \;=\; \frac{1}{K}\sum_{k=1}^K p(y \mid x, w^{(k)}_*).$$
By the second assumption $\hat{w}^{(k)} = w^{(k)}_*$ for each $k$, so the right-hand side equals $\hat{p}_{\mathrm{ens}}(y \mid x, \mathcal{D})$. $\square$

The theorem's preconditions are gross idealizations: real posteriors are not collapsed Diracs, real ensembles do not perfectly cover all $K$ modes, and the assumption of equal posterior mass across modes is a strong identifiability condition. But the spirit of the result is the right intuition. A deep ensemble is the right approximation to the Bayesian predictive in a *function-space* regime where (i) the posterior has multiple modes, (ii) the within-mode predictive variance is small relative to the between-mode predictive variance, and (iii) the modes have comparable posterior mass. For BNNs trained with weight decay, conditions (i) and (iii) hold approximately by §2.4's symmetry arguments — every mode has $h!$ permutation copies, each with the same posterior mass — and (ii) holds whenever the network has sufficient capacity to fit the training data (the within-mode predictive variance shrinks as the network overfits).

The function-space view of Wilson & Izmailov 2020 makes this explicit. The Bayesian posterior over functions is
$$p(f \mid \mathcal{D}) \;=\; \int p(f \mid w)\,p(w \mid \mathcal{D})\,dw,$$
where $p(f \mid w) = \delta(f - f_w)$ is the deterministic mapping from weights to functions. Distinct weight modes that produce the same function on the training data are *redundant* in function space — they all map to the same point $f$. Distinct weight modes that produce *different* functions on the training data correspond to different points in function space, and a deep ensemble's diversity in function space is what matters for predictive uncertainty.

#### §5.3 The mixture predictive form

For regression with Gaussian observation noise, each ensemble member's predictive is Gaussian, and the ensemble predictive is a mixture of Gaussians with closed-form moments. This is the cleanest setting in which to read the *epistemic-vs-aleatoric* decomposition that §8 formalizes.

> **Proposition 5.3 (Mixture-of-Gaussians ensemble predictive).** Suppose the observation model is $y \mid x, w \sim \mathcal{N}(f_w(x), \sigma^2_{\mathrm{noise}})$ with known noise variance $\sigma^2_{\mathrm{noise}}$. Let $\bar{f}(x) := K^{-1}\sum_{k=1}^K f_{\hat{w}^{(k)}}(x)$. The deep-ensemble predictive distribution is the Gaussian mixture
> $$\hat{p}_{\mathrm{ens}}(y \mid x) \;=\; \frac{1}{K}\sum_{k=1}^K \mathcal{N}\bigl(y;\, f_{\hat{w}^{(k)}}(x),\, \sigma^2_{\mathrm{noise}}\bigr),$$
> with predictive mean $\bar{f}(x)$ and predictive variance
> $$\mathrm{Var}_{\hat{p}_{\mathrm{ens}}}[y \mid x] \;=\; \underbrace{\sigma^2_{\mathrm{noise}}}_{\text{aleatoric}} \;+\; \underbrace{\frac{1}{K}\sum_{k=1}^K \bigl(f_{\hat{w}^{(k)}}(x) - \bar{f}(x)\bigr)^2}_{\text{epistemic}}.$$

*Proof.* The mixture form is immediate from Definition 5.1 with $p(y \mid x, w) = \mathcal{N}(y; f_w(x), \sigma^2_{\mathrm{noise}})$. For the variance, apply the law of total variance to the random variable $y$ under the mixture:
$$\mathrm{Var}[y \mid x] \;=\; \mathbb{E}_k\bigl[\mathrm{Var}[y \mid x, k]\bigr] + \mathrm{Var}_k\bigl[\mathbb{E}[y \mid x, k]\bigr],$$
where $k$ is the ensemble-member index drawn uniformly from $\{1, \ldots, K\}$. The conditional variance $\mathrm{Var}[y \mid x, k] = \sigma^2_{\mathrm{noise}}$ is constant in $k$, so its expectation equals itself. The conditional mean is $f_{\hat{w}^{(k)}}(x)$, whose variance over uniform $k$ is the sample variance $K^{-1}\sum_k (f_{\hat{w}^{(k)}}(x) - \bar{f}(x))^2$. Adding gives the claim. $\square$

The decomposition is structural: aleatoric uncertainty is the noise level the model assumes (irreducible by adding more data), and epistemic uncertainty is the variance across ensemble members (reducible by adding more data, which would shrink each mode toward zero width and pull the modes themselves toward the truth). For binary classification with Bernoulli observation likelihood the analogous identity uses $\mathrm{Var}[y \mid x, k] = \sigma(f_{\hat{w}^{(k)}}(x))(1 - \sigma(f_{\hat{w}^{(k)}}(x)))$, and the epistemic term is the sample variance of the $\sigma(f_{\hat{w}^{(k)}}(x))$'s — exactly the predictive-variance heatmaps we have been plotting.

#### §5.4 Connection to stacking

Deep ensembles use uniform weights $\pi_k = 1/K$ on their members. The [Stacking & Predictive Ensembles](/topics/stacking-and-predictive-ensembles) topic develops a more general framework: given $K$ candidate predictive distributions, learn weights $\boldsymbol{\pi} \in \Delta^{K-1}$ that maximize the leave-one-out posterior-predictive log-density (Yao, Vehtari, Simpson & Gelman 2018). Stacking dominates uniform weighting when the candidate predictives are *heterogeneous* — different model classes, different priors, different architectures — because uniform weighting can be far from the optimum on the simplex. For the homogeneous deep-ensemble case (same architecture, different random seeds), the candidates are *exchangeable* by construction and uniform weighting is approximately optimal. The two methods are complementary: stacking is the right tool when you have a heterogeneous catalog of models; deep ensembles are the right tool when you have one architecture and want quick, well-calibrated Bayesian uncertainty.

> **Remark 5.4 (Stacking weights vs. uniform weights).** A reader who has shipped the [Stacking & Predictive Ensembles](/topics/stacking-and-predictive-ensembles) topic should think of deep ensembles as the special case "stacking on $K$ same-architecture, same-prior, same-data candidates with $\pi_k$ fixed at $1/K$." Lifting the uniform-weight constraint and learning $\boldsymbol{\pi}$ from PSIS-LOO is a strict improvement when the candidates have any genuine heterogeneity. For a homogeneous deep ensemble, the PSIS-LOO-optimal weights are within Monte Carlo noise of uniform, so the stacking machinery typically returns to the uniform answer.

#### §5.5 Algorithm and worked example

> **Algorithm 5.5 (Deep ensemble).**
> *Input:* training data $\mathcal{D}$, network architecture $f_w$, ensemble size $K$, training recipe (optimizer, learning rate, epochs, weight decay).
> *Output:* ensemble predictive $\hat{p}_{\mathrm{ens}}(y \mid x)$ at test inputs.
> 1. For $k = 1, \ldots, K$:
>    a. Sample initialization $\hat{w}^{(k)}_0 \sim$ Kaiming-normal with seed $k$.
>    b. Run the standard training loop to convergence: $\hat{w}^{(k)} \leftarrow \arg\min_w \mathcal{L}_{\mathrm{WD}}(w; \mathcal{D})$ via Adam.
> 2. Predict at test input $x$: $\hat{p}_{\mathrm{ens}}(y \mid x) = K^{-1}\sum_k p(y \mid x, \hat{w}^{(k)})$.

For the Two Moons running example we use $K = 10$ — large enough to make the function-space mode coverage visible, small enough to fit comfortably in the runtime budget. Each member trains in ~0.5 s on CPU, so the §5 cell wall-clock is dominated by training: ~5 s. (This is also why deep ensembles are often called "expensive" in production settings — for ImageNet-scale models, training $K=10$ networks costs $K \times$ a single training run. The Two Moons cost is negligible because each network is tiny.)

#### Code experiment §5

Train $K = 10$ MLPs with `make_mlp(seed)` for $\mathrm{seed} \in \{0, \ldots, 9\}$, each with the standard training recipe. PyTorch end-to-end. Total cell runtime ~5 s. Use the `models_5` from §1's first 5 plus 5 new ones for $k = 5, \ldots, 9$. Compute predictive mean and standard deviation on the prediction grid; produce the same three-panel layout as §§3–4 for visual comparability.

Verification: print the off/on predictive-std ratio (expected at least $4\times$, typically larger than §3 Laplace and §4 MC-dropout because the ensemble captures multi-mode function-space variance), and the train accuracy of the ensemble predictive (should match the per-member accuracy to within $1\%$).

#### Viz design intent §5

**Figure 5 (`05_deep_ensemble_two_moons.png`).** Three panels on the standard $200 \times 200$ Two Moons grid.

**Panel (a)** — *Deep-ensemble predictive mean.* Heatmap of $K^{-1}\sum_k \sigma(f_{\hat{w}^{(k)}}(x))$ for $K = 10$. Colormap `RdBu_r`, $v_{\min}=0$, $v_{\max}=1$. Reader should see a decision surface visually similar to §§1, 3, 4 panel (a) — the *mean* prediction is robust across methods.

**Panel (b)** — *Deep-ensemble predictive standard deviation.* Heatmap of the sample standard deviation of $\sigma(f_{\hat{w}^{(k)}}(x))$ over $k$. Colormap `viridis`. Reader should see noticeably brighter (larger) std off-distribution than §3 and §4's panels (b), reflecting the function-space mode coverage that single-mode methods miss.

**Panel (c)** — *Ensemble decision-boundary overlay.* Overlay all $K = 10$ ensemble members' $0.5$-contours on the data, each as a translucent orange line (`COLOR_ENSEMBLE`). Reader should see fanning-out off-distribution that is *wider* than §3 panel (c)'s Laplace samples and §4 panel (c)'s MC-dropout samples — visual confirmation of multi-mode coverage.

The PNG is the static fallback. The on-page rendering is the React + D3 component `DeepEnsembleViz.tsx`.

**Interactive controls (v1 deliverable):**

- *Ensemble-size slider* ($K = 2$ to $K = 20$, default $K = 10$). Subsets the precomputed pool of trained models and re-renders all three panels. Reader sees: $K = 2$ gives noisy, sparse uncertainty estimates; $K = 5$ stabilizes; $K = 10$ is the sweet spot; $K = 20$ adds little (diminishing returns curve).
- *Member-highlight click target.* Clicking any of the $K$ decision-boundary contours in panel (c) highlights that specific ensemble member's full predictive heatmap (overlaid in panel a as a translucent layer) — lets the reader see one member's idiosyncrasies relative to the ensemble mean.
- *Resample-seed button.* Re-rolls the random seeds for the $K$ ensemble members and re-trains. Confirms that ensemble diversity is robust to the specific seed set.

The component pre-trains a pool of 20 MLPs on first hydration (~10 s), caches their weights, and re-renders the three panels in <50 ms on $K$-slider interactions.

**What the reader should learn from Figure 5:** (i) deep ensembles are conceptually trivial — train more times — but capture multi-modality that single-mode methods (Laplace, MC-dropout) cannot; (ii) the predictive variance off-distribution is larger than for the single-mode methods, in line with the function-space-posterior interpretation; (iii) the cost is $K \times$ a single training run, which is the engineering reason single-mode methods exist at all — for ImageNet-scale models, $K \times$ training is the expensive option. The §8 calibration analysis quantifies whether the extra cost buys better calibration; spoiler: it usually does.

---

### §6. Stochastic-gradient Langevin dynamics (SGLD)

§§3–5 each give up something to gain tractability. Laplace gives up multi-modality; MC-dropout gives up the Gaussian variational family in favor of a more rigid Bernoulli; deep ensembles give up the posterior interpretation in favor of a function-space mixture. *Stochastic-gradient MCMC* (SG-MCMC) takes a different bargain: keep the asymptotic exactness of MCMC and pay for it in wall-clock per posterior sample, but make the per-iteration cost cheap enough that the trade is favorable in practice. The first method in the family — *stochastic-gradient Langevin dynamics* (SGLD; Welling & Teh 2011) — is a discretization of the Langevin SDE on the negative log-posterior, with mini-batch gradients standing in for full-data gradients and calibrated Gaussian noise injected at every step. Under a square-summable step-size schedule, SGLD samples converge in distribution to the true posterior. §7 will lift this to second-order Langevin (SGHMC) for faster mixing; §8 will compare both to the §§3–5 methods on calibration.

#### §6.1 The Langevin SDE and its stationary distribution

The starting point is the *(overdamped) Langevin SDE* on the negative log-posterior $U(w) = -\log p(w \mid \mathcal{D})$:
$$dw_t \;=\; -\nabla U(w_t)\,dt \;+\; \sqrt{2}\,dB_t,$$
where $B_t$ is standard Brownian motion in $\mathbb{R}^p$. The fundamental fact about this SDE — proved via the Fokker–Planck equation — is that under mild regularity conditions on $U$ (smoothness, growth at infinity), its unique stationary distribution is
$$\pi(w) \;\propto\; \exp(-U(w)) \;=\; p(w \mid \mathcal{D}).$$
That is, simulating the SDE forward in time and sampling $w_t$ at large $t$ produces samples from the posterior. The proof is a verification: under the Fokker–Planck equation $\partial_t \rho = \nabla \cdot (\rho \nabla U) + \nabla^2 \rho$, the candidate $\rho_\infty(w) \propto e^{-U(w)}$ satisfies $\nabla \rho_\infty = -\rho_\infty \nabla U$, so $\rho_\infty \nabla U + \nabla \rho_\infty = 0$ and $\partial_t \rho_\infty = 0$ — the proposed stationary distribution is genuinely stationary. Uniqueness follows from standard ergodicity arguments on the Langevin diffusion.

So the Langevin SDE solves the posterior-sampling problem in continuous time. To use it on a computer we need to discretize, and to scale it to deep learning we need to replace the full-data gradient with a mini-batch estimate. SGLD is exactly that.

#### §6.2 The SGLD update

> **Definition 6.1 (SGLD update).** Let $U(w) = -\log p(\mathcal{D} \mid w) - \log p(w)$ be the negative log-posterior under Definition 2.1's Gaussian-prior BNN. Let $\{\eta_t\}_{t \geq 0}$ be a positive step-size schedule, and let $b$ be the mini-batch size. At each iteration $t$, draw a mini-batch $\mathcal{B}_t \subset \mathcal{D}$ uniformly at random with replacement, compute the *stochastic gradient*
> $$\hat g_t \;:=\; -\frac{n}{b}\sum_{i \in \mathcal{B}_t} \nabla_w \log p(y_i \mid x_i, w_t) \;+\; \frac{1}{\tau^2}\,w_t,$$
> draw a fresh isotropic Gaussian $\xi_t \sim \mathcal{N}(0, I_p)$, and update
> $$w_{t+1} \;=\; w_t \;-\; \frac{\eta_t}{2}\,\hat g_t \;+\; \sqrt{\eta_t}\,\xi_t.$$

Three structural notes. First, $\hat g_t$ is an unbiased estimator of $\nabla U(w_t)$: $\mathbb{E}[\hat g_t \mid w_t] = \nabla U(w_t)$, because the mini-batch term averaged over the uniform random index has expectation equal to the full-data gradient and the prior term is exact. Second, the noise scale $\sqrt{\eta_t}$ matches the Euler–Maruyama discretization of $\sqrt{2}\,dB_t$ when the convention $\eta_t / 2$ is used on the gradient — these factors of 2 are pure choice of parametrization, but they have to be consistent. Third, the step-size schedule $\{\eta_t\}$ is the central design choice. Welling & Teh prove that, under the schedule $\eta_t = a (b + t)^{-\gamma}$ with $\gamma \in (1/2, 1]$, the chain converges to the posterior; this schedule satisfies the *Robbins–Monro conditions* $\sum_t \eta_t = \infty$ (the chain explores all of weight space) and $\sum_t \eta_t^2 < \infty$ (the discretization error vanishes asymptotically).

#### §6.3 Asymptotic exactness

> **Theorem 6.2 (Welling & Teh 2011).** Suppose (i) $U$ is twice continuously differentiable with bounded Hessian; (ii) the per-example gradient variance $\mathrm{Var}[\nabla \log p(y_i \mid x_i, w)]$ is bounded uniformly in $w$ on compacts; (iii) the step-size schedule satisfies $\sum_t \eta_t = \infty$ and $\sum_t \eta_t^2 < \infty$. Then the SGLD chain $\{w_t\}_{t \geq 0}$ in Definition 6.1 has $p(w \mid \mathcal{D})$ as its asymptotic distribution: for any bounded measurable test function $\varphi$,
> $$\frac{\sum_{t=0}^{T} \eta_t\,\varphi(w_t)}{\sum_{t=0}^{T} \eta_t} \;\xrightarrow[T \to \infty]{\text{a.s.}}\; \int \varphi(w)\,p(w \mid \mathcal{D})\,dw.$$

*Proof sketch.* Decompose the SGLD step into three contributions:
$$w_{t+1} - w_t \;=\; -\frac{\eta_t}{2} \nabla U(w_t) \;+\; \sqrt{\eta_t}\,\xi_t \;+\; \underbrace{-\frac{\eta_t}{2}\bigl(\hat g_t - \nabla U(w_t)\bigr)}_{=: \zeta_t}.$$
The first two terms are exactly an Euler–Maruyama step of the Langevin SDE with step $\eta_t / 2$ (and the corresponding $\sqrt{\eta_t}$ noise). The third term $\zeta_t$ is mini-batch gradient noise: zero-mean, variance $(\eta_t^2 / 4) \mathrm{Var}[\hat g_t]$. As $\eta_t \to 0$ on the schedule, the variance of $\zeta_t$ scales as $\eta_t^2$, while the variance of the Brownian noise scales as $\eta_t$. So the ratio $\mathrm{Var}[\zeta_t] / \mathrm{Var}[\sqrt{\eta_t}\,\xi_t] = O(\eta_t) \to 0$. The mini-batch noise is asymptotically dominated by the Brownian noise, and in the limit the chain mimics the exact Langevin SDE — whose stationary distribution is the posterior. The square-summability $\sum \eta_t^2 < \infty$ controls the cumulative discretization error; the divergence $\sum \eta_t = \infty$ ensures the chain has time to explore. The full proof (Welling & Teh 2011, Sato & Nakagawa 2014, Vollmer, Zygalakis & Teh 2016) makes these ratio-and-cumulative arguments rigorous via martingale convergence and ergodic-theorem machinery. $\square$

This is the asymptotic-exactness guarantee that motivated SG-MCMC. Note what it does *not* say: nothing about a constant-stepsize regime, nothing about the §2.4 BvM-failure caveats for over-parametrized neural networks, nothing about practical mixing time. The theorem says SGLD's invariant measure is the posterior; it does not say SGLD mixes to that measure quickly.

#### §6.4 Mini-batch noise budget

For practical step sizes the mini-batch noise is *not* negligible relative to the Brownian noise, and understanding how the two interact is the difference between a working SGLD chain and a chain that biases the wrong way.

> **Proposition 6.3 (Mini-batch noise as a fraction of the noise budget).** The conditional variance of the SGLD step at iteration $t$ decomposes as
> $$\mathrm{Var}[w_{t+1} - w_t \mid w_t] \;=\; \underbrace{\eta_t\,I_p}_{\text{Brownian noise}} \;+\; \underbrace{\frac{\eta_t^2}{4}\,\Sigma_{\hat g}(w_t)}_{\text{minibatch noise}},$$
> where $\Sigma_{\hat g}(w_t) = \mathrm{Cov}[\hat g_t \mid w_t]$. The two contributions are independent (the mini-batch index is drawn independently of the noise injection). The mini-batch noise contribution scales as $\eta_t^2$ and the Brownian as $\eta_t$, so their ratio is $\eta_t \cdot \mathrm{tr}\,\Sigma_{\hat g} / 4$.

*Proof.* By definition $w_{t+1} - w_t = -\tfrac{\eta_t}{2}\,\hat g_t + \sqrt{\eta_t}\,\xi_t$. The two random variables are independent (mini-batch and noise injection draws are independent), so the variance is the sum of variances:
$$\mathrm{Var}[w_{t+1} - w_t \mid w_t] \;=\; \mathrm{Var}\Bigl[-\tfrac{\eta_t}{2}\,\hat g_t \,\Big|\, w_t\Bigr] \;+\; \mathrm{Var}\bigl[\sqrt{\eta_t}\,\xi_t\bigr] \;=\; \frac{\eta_t^2}{4}\,\Sigma_{\hat g}(w_t) \;+\; \eta_t\,I_p,$$
which is the claim. $\square$

The practical takeaway: when $\eta_t$ is small, the mini-batch noise is negligible relative to the Brownian noise, and the chain is well-approximated by the exact Langevin SDE. When $\eta_t$ is large (constant-stepsize or aggressive schedule), the mini-batch noise becomes a significant — and *anisotropic* — perturbation, and the chain's stationary distribution is no longer exactly the posterior.

#### §6.5 The constant-stepsize bias-variance tradeoff

In production we often run SGLD with a *constant* step-size $\eta$ for a fixed wall-clock budget rather than the asymptotically-correct decaying schedule. This trades asymptotic exactness for faster mixing and a fixed iteration cost, and the bias structure has been quantified.

> **Proposition 6.4 (Bias-variance tradeoff in constant-stepsize SGLD).** With constant step-size $\eta > 0$, the SGLD chain $\{w_t\}$ has stationary distribution $\pi_\eta(w) \neq p(w \mid \mathcal{D})$ in general. Under regularity conditions (Vollmer, Zygalakis & Teh 2016), for any test function $\varphi$ in a suitable function space, the asymptotic bias of the time-averaged Monte Carlo estimator decomposes as
> $$\Bigl|\,\mathbb{E}_{\pi_\eta}[\varphi] - \mathbb{E}_{p(\cdot \mid \mathcal{D})}[\varphi]\,\Bigr| \;=\; O(\eta) \;+\; O\!\left(\frac{\eta\,\mathrm{tr}\,\Sigma_{\hat g}}{n}\right),$$
> the first term from Euler–Maruyama discretization error and the second from minibatch gradient noise. The Monte Carlo variance of the time-averaged estimator over $T$ post-burn-in iterations is $\mathrm{Var}[\hat\varphi] = O(\tau_{\mathrm{auto}} / T)$, where $\tau_{\mathrm{auto}}$ is the chain's autocorrelation time.

*Sketch.* Both bias terms come from the stationarity equation of the SGLD chain. Setting $\partial_t \pi_\eta = 0$ in the modified Fokker–Planck equation that includes mini-batch noise, expanding $\pi_\eta = e^{-U} (1 + O(\eta) + O(\eta\,\mathrm{tr}\,\Sigma_{\hat g}/n))$, and integrating against $\varphi$ produces the stated rates. The variance term is the standard MCMC variance scaling. $\square$

The reader should leave §6.5 with a concrete heuristic. *Larger $\eta$ mixes faster but biases more*; *more samples reduces variance but not bias*. The bias-variance tradeoff in constant-stepsize SGLD is qualitatively different from the bias-variance tradeoff in standard MCMC (which is bias-free at any stepsize, so only variance trades against burn-in). For practical BNN inference, the tradeoff is usually navigated by picking $\eta$ small enough that the bias is below the noise floor of the downstream calibration metrics — a few times $10^{-4}$ for typical small-to-medium networks, smaller for ImageNet-scale.

#### §6.6 Algorithm and worked example

> **Algorithm 6.5 (SGLD posterior sampling).**
> *Input:* training data $\mathcal{D}$, network $f_w$, prior scale $\tau^2$, step-size schedule $\{\eta_t\}$ or constant $\eta$, mini-batch size $b$, burn-in $T_{\mathrm{burn}}$, sample count $T$, thinning interval $\Delta$.
> *Output:* posterior samples $w^{(1)}, \ldots, w^{(T)}$.
> 1. Initialize $w_0$ — random init or warm-start from a MAP $\hat{w}$.
> 2. For $t = 0, 1, \ldots, T_{\mathrm{burn}} + T \cdot \Delta$:
>    a. Sample mini-batch $\mathcal{B}_t \subset \mathcal{D}$.
>    b. Compute stochastic gradient $\hat g_t$ per Definition 6.1.
>    c. Draw $\xi_t \sim \mathcal{N}(0, I_p)$.
>    d. Update $w_{t+1} \leftarrow w_t - \tfrac{\eta_t}{2}\,\hat g_t + \sqrt{\eta_t}\,\xi_t$.
> 3. Discard the first $T_{\mathrm{burn}}$ iterations as burn-in.
> 4. Return $w^{(s)} := w_{T_{\mathrm{burn}} + s\Delta}$ for $s = 1, \ldots, T$ (thinning to reduce autocorrelation).

For the §6.7 Two Moons example we use constant $\eta = 10^{-3}$, mini-batch size $b = 32$, burn-in $T_{\mathrm{burn}} = 200$, sample count $T = 100$, thinning $\Delta = 10$ — total $200 + 100 \cdot 10 = 1200$ iterations, each iteration a single forward+backward pass on a $b = 32$ mini-batch. Total cell runtime ~7 s.

#### Code experiment §6

PyTorch end-to-end. Standard SGLD loop: at each iteration sample a mini-batch, compute the negative log-posterior gradient via autograd, take an Euler–Maruyama step with calibrated noise. Initialize from the §1 first-seed MAP for warm-start (skips most of the burn-in compared to random init). Total cell runtime ~7 s. Verification: print $(\eta, b, T)$, the off/on predictive-std ratio (expected at least $3\times$, and of similar magnitude to deep ensembles since SG-MCMC also captures multi-mode coverage), and the train accuracy of the SGLD predictive (should match within $1\%$).

#### Viz design intent §6

**Figure 6 (`06_sgld_two_moons.png`).** Three panels mirroring §§3–5 for visual comparability.

**Panel (a)** — *SGLD predictive mean.* Heatmap of $T^{-1} \sum_s \sigma(f_{w^{(s)}}(x))$ over $T = 100$ thinned post-burn-in samples. Reader should see a decision surface visually similar to §§3–5 panels (a) — the mean prediction is the topic-invariant.

**Panel (b)** — *SGLD predictive standard deviation.* Heatmap of the sample standard deviation across SGLD samples. Reader should see std comparable to or larger than §5 deep ensembles — SGLD's mode-mixing brings in more weight-space variance than the §3 Laplace local Gaussian.

**Panel (c)** — *Trace plot for one weight component.* On a separate axes layout (or as the third panel in landscape), plot the trajectory of one particular weight $w_{t, j}$ across the $1200$ iterations, with the burn-in shaded. Reader should see the chain mixing — large fluctuations early, smaller-amplitude oscillations after burn-in. This panel is qualitatively different from §§3–5's panel (c) — it's a *diagnostic*, not a predictive — but it makes the Langevin-dynamics nature of the chain visible in a way that the heatmaps do not.

The PNG is the static fallback. The on-page rendering is the React + D3 component `SGLDBNNViz.tsx`.

**Interactive controls (v1 deliverable):**

- *Step-size slider* ($\eta$ on a log scale from $10^{-5}$ to $10^{-2}$, default $10^{-3}$). Re-runs SGLD with the new step size and re-renders all three panels. Reader sees: too-small $\eta$ gives a chain that barely moves (panel c trace nearly flat), bias is small but variance is high; too-large $\eta$ gives a chain that explores aggressively but biases the wrong direction (panel b std heatmap is *too* uncertain even on the data); the default $\eta = 10^{-3}$ is the practical sweet spot.
- *Mini-batch size slider* ($b = 8$ to $b = 128$, default $b = 32$). Re-runs SGLD with the new batch size. Reader sees: small $b$ means more stochastic-gradient noise (per Prop 6.3) and slightly worse mixing; larger $b$ approaches full-batch Langevin dynamics.
- *Step-size schedule toggle* (`Constant` / `Decaying $\eta_t = \eta_0 t^{-1/2}$`, default `Constant`). When `Decaying` is selected, the chain transitions from exploration to refinement over the burn-in period — visible in panel (c) as an early high-amplitude trace settling to a tight oscillation.

Re-running 1200 SGLD iterations on every interaction is too slow (~7 s) for smooth UI. The component pre-runs SGLD chains for a discrete grid of $(\eta, b, \text{schedule})$ values on first hydration (~30 s — the heaviest pre-compute in the topic), caches the chain trajectories, and switches between them in <50 ms on slider interaction.

**What the reader should learn from Figure 6:** (i) SGLD samples the posterior asymptotically exactly (Theorem 6.2), at the cost of running a chain for many iterations; (ii) constant-stepsize SGLD has a bias-variance tradeoff in $\eta$; (iii) the trace plot in panel (c) is the practical mixing diagnostic — slow oscillations indicate poor mixing, fast oscillations after burn-in indicate the chain is exploring the posterior; (iv) §7 will introduce SGHMC, which adds momentum to the Langevin dynamics for faster mixing without extra wall-clock cost.

---

### §7. Stochastic-gradient HMC (SGHMC)

SGLD's mixing is rate-limited by the first-order Langevin diffusion: each step is a small drift along the gradient plus an isotropic Brownian kick, so traversing a long, low-curvature ridge in the loss landscape requires many small steps. *Stochastic-gradient Hamiltonian Monte Carlo* (SGHMC; Chen, Fox & Guestrin 2014) lifts the dynamics from first-order to second-order Langevin: introduce a momentum variable, let the chain accumulate velocity along the gradient, and damp the velocity with a *friction* term that simultaneously enforces stationarity and absorbs the variance contributed by stochastic gradients. Empirically the result mixes considerably faster than SGLD at the same wall-clock cost.

#### §7.1 The second-order Langevin SDE

Augment the state with a momentum variable $v \in \mathbb{R}^p$ and let $M \in \mathbb{R}^{p \times p}$ be a positive-definite *mass matrix* (typically $M = I_p$). The *underdamped Langevin SDE* is
$$dw_t = M^{-1} v_t\,dt, \qquad dv_t = -\nabla U(w_t)\,dt - C\, M^{-1} v_t\,dt + \sqrt{2 C}\,dB_t,$$
where $C \in \mathbb{R}^{p \times p}$ is a positive-semidefinite *friction matrix*. The Fokker–Planck calculation (analogous to §6.1) gives the stationary distribution
$$\pi(w, v) \;\propto\; \exp\!\Bigl(-U(w) - \tfrac{1}{2}\, v^\top M^{-1} v\Bigr),$$
the joint posterior over $(w, v)$ in which the marginal $\pi(w) \propto e^{-U(w)} = p(w \mid \mathcal{D})$ is exactly the BNN posterior we want and the velocity is independent Gaussian noise we can discard. So the second-order Langevin SDE samples the posterior at *higher mixing rate* than SGLD because momentum carries the chain through low-curvature regions in many fewer steps.

#### §7.2 The complication: stochastic gradients add variance to velocity

If we discretize the SDE via Euler–Maruyama with full-data gradients, we get a $w$-and-$v$ analogue of SGLD that converges to $\pi$ as the step size decays. But replacing $\nabla U$ with a stochastic mini-batch estimate $\hat g$ introduces an extra noise term in the velocity dynamics:
$$dv_{\mathrm{stoch}} = -\hat g\,dt - C M^{-1} v\,dt + \sqrt{2 C}\,dB \;=\; -\nabla U\,dt - \zeta\,dt - C M^{-1} v\,dt + \sqrt{2 C}\,dB,$$
where $\zeta := \hat g - \nabla U$ is zero-mean stochastic-gradient noise with covariance $B(w) := \mathrm{Cov}[\hat g \mid w]$. The extra $\zeta$ term changes the *effective* noise covariance from $2C$ to $2C + B$, and the stationary distribution shifts away from $\pi$. Without correction, stochastic-gradient HMC samples a perturbed posterior. The Chen–Fox–Guestrin fix is the *friction-compensation*: choose the friction matrix $C$ large enough that the *injected* Brownian noise can absorb the gradient noise.

#### §7.3 The SGHMC update

> **Definition 7.1 (SGHMC update).** Let $\eta > 0$ be the step size, $\alpha := \eta C$ the *friction parameter* (with $C$ a positive-semidefinite matrix; typically $C = c I_p$ for scalar $c > 0$), $M$ the mass matrix (typically $I_p$), and $\hat B(w)$ a non-negative-definite estimate of the stochastic-gradient noise covariance $B(w)$ satisfying $C \succeq \hat B$. At each iteration, draw mini-batch $\mathcal{B}_t$, compute $\hat g_t$ as in Definition 6.1, draw $\xi_t \sim \mathcal{N}(0, I_p)$, and update:
> $$v_{t+1} \;=\; (1 - \alpha) v_t \;-\; \eta\, \hat g_t \;+\; \sqrt{2\eta\,(C - \hat B(w_t))}\,\xi_t,$$
> $$w_{t+1} \;=\; w_t \;+\; \eta\, M^{-1} v_{t+1}.$$

The structural reading: $(1 - \alpha)$ scales the previous velocity (friction damping), $-\eta \hat g_t$ accelerates along the negative gradient (Hamiltonian drift), and the Brownian-noise injection has scale $\sqrt{2\eta(C - \hat B)}$ — *less* than the $\sqrt{2\eta C}$ that the un-compensated SDE would inject, by exactly the right amount to absorb the additional variance the stochastic gradient contributes. When $\hat B = 0$ (the simplest practical choice — no covariance estimation), we use $\sqrt{2\eta C}$ noise injection and accept an $O(\eta)$ bias of the same order as constant-stepsize SGLD; when $\hat B$ is exact and $C \succeq B(w)$, the chain has the *correct* stationary distribution at any step size.

#### §7.4 Stationary distribution

> **Theorem 7.2 (Chen, Fox & Guestrin 2014).** Suppose the friction $C \succeq B(w)$ uniformly in $w$ and $\hat B(w) = B(w)$ exactly. Then the continuous-time analogue of the SGHMC update has stationary distribution
> $$\pi(w, v) \;\propto\; \exp\!\Bigl(-U(w) - \tfrac{1}{2}\, v^\top M^{-1} v\Bigr),$$
> whose $w$-marginal is the BNN posterior. Under appropriate step-size schedules, the discretized SGHMC chain converges in distribution to $\pi$.

*Proof sketch.* The Fokker–Planck equation for the joint $(w, v)$ density under the continuous-time SGHMC dynamics — including the stochastic-gradient noise term $\zeta$ — has the form
$$\partial_t \rho \;=\; -\nabla_w \cdot (\rho\, M^{-1} v) \;+\; \nabla_v \cdot \bigl(\rho\, [\nabla U + C M^{-1} v]\bigr) \;+\; \nabla_v^2 \cdot \bigl[(C - \hat B + B(w))\,\rho\bigr].$$
Setting $\partial_t \rho = 0$ and inserting $\rho_\infty = \pi(w, v)$ above: the gradient terms vanish (as for the noiseless second-order Langevin), and the diffusion-Laplacian terms vanish *if and only if* the effective noise covariance equals $C$ — i.e., $C - \hat B + B = C$, which holds when $\hat B = B$. So with exact noise compensation, $\pi$ is stationary. The discrete-time convergence argument follows the standard SG-MCMC machinery (Sato & Nakagawa 2014, applied to the joint $(w, v)$ Markov chain): the per-iteration discretization error vanishes under square-summable step sizes, and the time-averaged Monte Carlo estimator converges almost surely to the posterior expectation. $\square$

In practice $\hat B(w) = 0$ is the standard simplifying choice — the stochastic-gradient covariance is hard to estimate cheaply and depends on $w$. The resulting chain has $O(\eta)$ bias from the un-compensated gradient noise (Prop 7.3 below), of the same order as constant-stepsize SGLD's bias, and the friction $C$ is tuned to be large enough that the bias is below the noise floor of downstream metrics.

#### §7.5 Friction-vs-noise tradeoff

> **Proposition 7.3 (Friction compensates stochastic-gradient variance).** With $\hat B = 0$, the SGHMC chain's stationary distribution $\pi_{\eta, C}(w, v)$ satisfies, for any test function $\varphi$ in a suitable class,
> $$\Bigl|\,\mathbb{E}_{\pi_{\eta, C}}[\varphi(w)] - \mathbb{E}_{p(\cdot \mid \mathcal{D})}[\varphi]\,\Bigr| \;=\; O\!\left(\frac{\eta\, \|B(w)\|}{C}\right) \;+\; O(\eta),$$
> where the first term is the noise-compensation bias and the second is the discretization bias. *Increasing $C$* shrinks the first term but slows mixing (the friction damps velocity, eliminating the momentum advantage); *decreasing $\eta$* shrinks both terms but slows wall-clock progress. Practical tuning: pick $C$ so that the stationary velocity variance $C^{-1}$ is comparable to the desired exploration scale, then pick $\eta$ small enough that the bias is below the calibration noise floor.

*Proof sketch.* The first bias term comes from solving the Fokker–Planck equation for $\pi_{\eta, C}$ with the un-compensated stochastic-gradient covariance: setting $\partial_t \rho = 0$ in the equation from §7.4 with $\hat B = 0$ gives $\rho_\infty \propto \exp(-U^*) \exp(-\tfrac{1}{2} v^\top M^{*-1} v)$ for shifted potential $U^*(w) = U(w) + O(B(w)/C)$ and shifted mass matrix; the integrated bias on $w$-marginal expectations is the integrated shift in $U$, which scales as $B(w)/C$. The second bias term is standard Euler-Maruyama-on-second-order-Langevin discretization error, $O(\eta)$. $\square$

The Chen–Fox–Guestrin contribution, in summary: SGHMC at fixed $\eta$ has the same $O(\eta)$ bias as SGLD, but with the momentum-variable advantage that mixing is much faster — the chain explores the posterior in fewer iterations. For the same wall-clock budget, SGHMC typically produces a more accurate estimate of posterior expectations than SGLD, even at the cost of carrying the velocity variable through every iteration.

#### §7.6 Practical preconditioning

> **Remark 7.4 (Adaptive preconditioning).** Vanilla SGHMC uses a constant friction $C = c I_p$ across all dimensions, but neural-network loss landscapes have wildly varying curvature across weights — sharp directions in the early layers, flat directions in the later layers, almost-degenerate directions along the §2.4 ReLU rescaling submanifold. *Preconditioned SGHMC* (Li, Chen, Carlson & Carin 2016) makes $C$ a per-parameter or per-layer quantity, often using running estimates of squared gradients in the spirit of *RMSProp* or *Adam*. The result is a faster-mixing chain on ill-conditioned posteriors at the cost of some additional bookkeeping per iteration. The `pyro` and `numpyro` libraries implement preconditioned SGHMC; `laplace-torch` implements vanilla SGHMC. For the §7.7 Two Moons example we use vanilla SGHMC with a single scalar friction $c = 0.1$ — the network is small enough that preconditioning gains are small.

#### §7.7 Algorithm and worked example

> **Algorithm 7.5 (SGHMC posterior sampling).**
> *Input:* training data $\mathcal{D}$, network $f_w$, prior scale $\tau^2$, step size $\eta$, friction $C = c I_p$, mass $M = I_p$, mini-batch size $b$, burn-in $T_{\mathrm{burn}}$, sample count $T$, thinning interval $\Delta$.
> *Output:* posterior samples $w^{(1)}, \ldots, w^{(T)}$.
> 1. Initialize $w_0$ (random init or MAP warm-start), $v_0 = 0$.
> 2. For $t = 0, 1, \ldots, T_{\mathrm{burn}} + T \cdot \Delta$:
>    a. Sample mini-batch $\mathcal{B}_t \subset \mathcal{D}$.
>    b. Compute stochastic gradient $\hat g_t$ per Definition 6.1.
>    c. Draw $\xi_t \sim \mathcal{N}(0, I_p)$.
>    d. Update: $v_{t+1} \leftarrow (1 - \eta c) v_t - \eta\, \hat g_t + \sqrt{2 \eta c}\, \xi_t$; $\;w_{t+1} \leftarrow w_t + \eta\, v_{t+1}$.
> 3. Discard the first $T_{\mathrm{burn}}$ iterations as burn-in.
> 4. Return $w^{(s)} := w_{T_{\mathrm{burn}} + s\Delta}$ for $s = 1, \ldots, T$.

For the Two Moons example we use $\eta = 10^{-3}$, $c = 0.1$, $b = 32$, $T_{\mathrm{burn}} = 200$, $T = 100$, $\Delta = 10$ — same total iteration count as §6, exposing the head-to-head mixing comparison. Total cell runtime ~7 s.

#### Code experiment §7

PyTorch end-to-end. Standard SGHMC loop: at each iteration sample a mini-batch, compute the negative-log-posterior gradient via autograd, update velocity (with friction and Brownian noise), and update position from velocity. Initialize from §1's first-seed MAP for warm-start, $v_0 = 0$. Total cell runtime ~7 s. Verification: print $(\eta, c, b, T)$, the off/on predictive-std ratio (expected at least $3\times$, similar to or slightly larger than SGLD due to better mode coverage), and the *autocorrelation time* of the SGHMC chain on a single weight component (expected to be smaller than SGLD's autocorrelation time at the same iteration budget — the momentum advantage).

#### Viz design intent §7

**Figure 7 (`07_sghmc_two_moons.png`).** Three panels mirroring §§3–6 for visual comparability.

**Panel (a)** — *SGHMC predictive mean.* Heatmap of $T^{-1} \sum_s \sigma(f_{w^{(s)}}(x))$ over $T = 100$ thinned post-burn-in samples. Reader should see a decision surface visually similar to §§3–6 panels (a).

**Panel (b)** — *SGHMC predictive standard deviation.* Heatmap of the sample standard deviation across SGHMC samples. Reader should see std comparable to or slightly larger than §6 SGLD.

**Panel (c)** — *Autocorrelation comparison.* Plot the autocorrelation function of one specific weight component for SGLD (§6) and SGHMC (§7) side-by-side, with x-axis = lag in iterations and y-axis = $\hat\rho(\mathrm{lag})$. Reader should see the SGHMC autocorrelation decay faster than the SGLD — the visual signature of the momentum-induced mixing speedup.

The PNG is the static fallback. The on-page rendering is the React + D3 component `SGHMCBNNViz.tsx`.

**Interactive controls (v1 deliverable):**

- *Friction slider* ($c$ on a log scale from $10^{-3}$ to $10^{1}$, default $10^{-1}$). Re-runs SGHMC with the new friction and re-renders all three panels. Reader sees: very small $c$ → momentum dominates, chain is unstable, biases significantly; very large $c$ → friction dominates, chain reduces to SGLD (autocorrelation in panel c approaches the SGLD curve); the sweet spot is $c \in [0.05, 0.2]$.
- *Step-size slider* ($\eta$ on a log scale from $10^{-5}$ to $10^{-2}$, default $10^{-3}$). Same role as in §6 SGLD; but the bias-variance tradeoff (Prop 7.3) involves $c$ as well, and the slider lets the reader explore the joint $\eta$-$c$ surface.
- *Method overlay toggle* (`SGHMC only` / `SGLD vs. SGHMC`, default the comparison). When the comparison is on, panel (c) overlays both autocorrelation functions; when off, only SGHMC is shown. Lets the reader confirm the §7 mixing-speedup claim across the slider grid.

Same pre-compute strategy as §6: pre-run SGHMC chains over a discrete $(\eta, c, \text{toggle})$ grid on first hydration (~30 s, runs in parallel with §6's pre-compute via separate chains), cache, and switch in <50 ms on slider interaction.

**What the reader should learn from Figure 7:** (i) SGHMC's predictive distribution is similar to SGLD's at the same iteration budget; (ii) but SGHMC mixes faster — fewer effective iterations to traverse the same posterior region; (iii) the friction coefficient $c$ is the new hyperparameter and trades momentum advantage against bias from un-compensated stochastic-gradient noise; (iv) for production BNN inference at deep-learning scale, SGHMC (or its preconditioned variants) is typically the preferred SG-MCMC method because the wall-clock-per-effective-sample is lower than SGLD.

---

### §8. Calibration and uncertainty quantification

§§3–7 each produce a predictive distribution. §1's panel (c) and the per-section heatmaps make the qualitative claim that BNN methods produce sharper uncertainty in the right places, but qualitative is not enough — the strategic-doc one-liner names *calibration* as a co-equal pillar with the four method families because reading a predictive distribution responsibly requires knowing whether the reported probabilities match empirical frequencies. A model that confidently predicts "90% chance class 1" should be right on roughly $90\%$ of inputs that get that prediction; if it is right $70\%$ of the time, it is *over-confident* and downstream decisions made under those probabilities will be miscalibrated. This section develops three calibration metrics — *expected calibration error*, the *Brier score*, and *negative log-likelihood* — that quantify how well predictive probabilities match empirical reality, decomposes BNN predictive variance into *epistemic* and *aleatoric* components, and runs the four §§3–7 methods plus the §1 point estimate head-to-head on a held-out Two Moons test set. The cold-posterior effect (Wenzel et al. 2020) and post-hoc temperature scaling (Guo et al. 2017) get their own remarks at the end.

#### §8.1 Expected calibration error

The simplest calibration metric. Bin the test predictions by predicted probability into $B$ bins of equal width on $[0, 1]$, and compare in-bin accuracy to in-bin average confidence.

> **Definition 8.1 (Expected calibration error, ECE).** Let $\{(x_i, y_i)\}_{i=1}^N$ be a held-out test set, $\hat{p}_i \in [0, 1]$ the model's predicted probability of the *predicted class* at $x_i$, and $\hat{y}_i \in \{0, 1, \ldots, K-1\}$ the predicted class. Partition $[0, 1]$ into $B$ equal-width bins $\mathcal{B}_1, \ldots, \mathcal{B}_B$ and let $I_b := \{i : \hat{p}_i \in \mathcal{B}_b\}$. Define the *bin accuracy* and *bin confidence*
> $$\mathrm{acc}(b) := \frac{1}{|I_b|}\sum_{i \in I_b} \mathbb{1}[\hat{y}_i = y_i], \qquad \mathrm{conf}(b) := \frac{1}{|I_b|}\sum_{i \in I_b} \hat{p}_i.$$
> The *expected calibration error* is
> $$\mathrm{ECE} \;:=\; \sum_{b=1}^B \frac{|I_b|}{N}\,\bigl|\mathrm{acc}(b) - \mathrm{conf}(b)\bigr|.$$

A perfectly-calibrated model has $\mathrm{ECE} = 0$: in every bin, the empirical accuracy matches the average confidence. ECE is always non-negative. Standard practice uses $B = 10$ or $B = 15$ bins; results are not very sensitive to $B$ for $N \geq 10^3$. ECE has a known weakness — it does not reward sharp predictions when accuracy is bin-averaged — but its interpretability (it has units of "probability points off") makes it the most-reported calibration metric in the BNN literature.

#### §8.2 The Brier score

A *strictly proper* scoring rule for binary classification. Where ECE is bin-based, Brier is point-wise: it averages the squared error between predicted probability and binary outcome.

> **Definition 8.2 (Brier score and its decomposition).** Let $\{(x_i, y_i)\}_{i=1}^N$ be a held-out test set with $y_i \in \{0, 1\}$ and $\hat{p}_i \in [0, 1]$ the model's predicted probability of class 1. The *Brier score* is
> $$\mathrm{BS} \;:=\; \frac{1}{N}\sum_{i=1}^N (\hat{p}_i - y_i)^2.$$
> Murphy 1973 decomposes this score as
> $$\mathrm{BS} \;=\; \underbrace{\sum_b \frac{|I_b|}{N}\,(\mathrm{conf}(b) - \mathrm{acc}(b))^2}_{\text{Reliability}} \;-\; \underbrace{\sum_b \frac{|I_b|}{N}\,(\mathrm{acc}(b) - \bar{y})^2}_{\text{Resolution}} \;+\; \underbrace{\bar{y}(1 - \bar{y})}_{\text{Uncertainty}},$$
> where $\bar{y} = N^{-1}\sum_i y_i$ is the marginal class rate and the bins $\mathcal{B}_b$ are as in Def 8.1.

*Proof.* Direct expansion. Writing $\hat p_i$ as $\mathrm{conf}(b(i))$ within its bin and $y_i$ as $\mathrm{acc}(b(i)) + r_i$ for the within-bin residual (so $\sum_{i \in I_b} r_i = 0$), the term $(\hat p_i - y_i)^2$ becomes $(\mathrm{conf}(b) - \mathrm{acc}(b))^2 - 2(\mathrm{conf}(b) - \mathrm{acc}(b)) r_i + r_i^2$. Sum within each bin: the cross-term vanishes because $\sum r_i = 0$, the first squared-term gives the Reliability piece, and the residual variance term equals $\mathrm{acc}(b)(1 - \mathrm{acc}(b))$. Sum across bins and rearrange using the identity $\sum_b \frac{|I_b|}{N} \mathrm{acc}(b)(1 - \mathrm{acc}(b)) = \bar y(1 - \bar y) - \sum_b \frac{|I_b|}{N}(\mathrm{acc}(b) - \bar y)^2$, which gives the claimed decomposition. $\square$

The decomposition reads: BS = (how miscalibrated within bins) − (how varied accuracy is across bins) + (irreducible class-base-rate variance). Uncertainty is the "no-skill" baseline; Resolution rewards a model whose accuracy varies meaningfully across bins; Reliability is the calibration-error analog of ECE. A perfectly calibrated model has Reliability = 0; a fully-discriminating model has Resolution = Uncertainty.

#### §8.3 Negative log-likelihood

> **Definition 8.3 (Negative log-likelihood as a proper scoring rule).** The *negative log-likelihood* of the model on the test set is
> $$\mathrm{NLL} \;:=\; -\frac{1}{N}\sum_{i=1}^N \log p(y_i \mid x_i, \mathcal{D}),$$
> where $p(y_i \mid x_i, \mathcal{D})$ is the model's predictive probability of the true label. NLL is a *strictly proper* scoring rule (Gneiting & Raftery 2007): it is uniquely minimized in expectation by the true conditional distribution.

NLL has two practical properties that ECE and Brier do not. It penalizes *over-confident wrong predictions* most aggressively: $-\log p$ blows up as $p \to 0$ for the true class. And it directly compares to held-out log-likelihoods used elsewhere in Bayesian model selection — the BIC, the marginal likelihood, the WAIC, and the LOO predictive log-density all share NLL's units (nats per observation). For BNN inference, NLL is usually the headline metric, with ECE and Brier as complementary diagnostics.

#### §8.4 The epistemic-aleatoric decomposition

A BNN's predictive variance decomposes into two pieces — what the model would learn from more data versus what no model can ever learn — and the decomposition is exactly the law of total variance applied to the BNN predictive.

> **Proposition 8.4 (Epistemic-aleatoric decomposition).** For a BNN with weight posterior $p(w \mid \mathcal{D})$ and observation likelihood $p(y \mid x, w)$, the predictive variance at test point $x^*$ decomposes as
> $$\mathrm{Var}\bigl[y \mid x^*, \mathcal{D}\bigr] \;=\; \underbrace{\mathbb{E}_{w \mid \mathcal{D}}\bigl[\mathrm{Var}[y \mid x^*, w]\bigr]}_{\mathrm{aleatoric}} \;+\; \underbrace{\mathrm{Var}_{w \mid \mathcal{D}}\bigl[\mathbb{E}[y \mid x^*, w]\bigr]}_{\mathrm{epistemic}}.$$
> The *aleatoric* term is the average of within-model conditional variance (irreducible label noise); the *epistemic* term is the variance of the conditional mean across weight samples (model uncertainty, which would shrink to zero with infinite data and a correctly specified family).

*Proof.* Direct application of the law of total variance to the random variable $y$ given $x^*$, treating $w$ as an auxiliary random variable conditional on $\mathcal{D}$:
$$\mathrm{Var}[y \mid x^*, \mathcal{D}] \;=\; \mathbb{E}_{w}\bigl[\mathrm{Var}[y \mid x^*, w]\bigr] \;+\; \mathrm{Var}_{w}\bigl[\mathbb{E}[y \mid x^*, w]\bigr],$$
where $w$ is averaged over $p(w \mid \mathcal{D})$. The first term is the average within-weight conditional variance — for a Bernoulli observation, $\mathrm{Var}[y \mid x^*, w] = \sigma(f_w(x^*))(1 - \sigma(f_w(x^*)))$, which is large in the *aleatoric* regions where the conditional probability is near $0.5$. The second term is the variance across weight samples of the predictive mean — large where weight uncertainty produces different mean predictions, exactly the §1 desideratum. $\square$

Practically, the epistemic term is what BNN methods compute via Monte Carlo: the §3, §4, §5, §6, §7 predictive standard-deviation heatmaps are $\sqrt{\text{epistemic variance}}$. The aleatoric term is the average over weight samples of the per-sample conditional variance — for a Bernoulli observation, this is the sigmoid-of-logit variance and is large near the decision boundary regardless of how confident the model is in its weights. The §8.5 head-to-head figure renders both components separately for one method (deep ensemble) so the reader can see the decomposition concretely.

#### §8.5 The cold-posterior effect

> **Remark 8.5 (Cold posteriors).** Wenzel et al. 2020 ("How Good is the Bayes Posterior in Deep Neural Networks Really?") observed empirically that BNN predictive accuracy and calibration consistently improve when the posterior is *tempered* by raising it to a power $1/T$ for $T < 1$:
> $$p_T(w \mid \mathcal{D}) \;\propto\; p(w \mid \mathcal{D})^{1/T} \;=\; p(\mathcal{D} \mid w)^{1/T}\,p(w)^{1/T}.$$
> Equivalently, with the Gaussian prior of Definition 2.1, tempering by $1/T$ scales the negative log-likelihood by $1/T$ and the prior precision by $1/T$, so the *effective* weight-decay strength is $\lambda_{\mathrm{eff}} = \lambda / T > \lambda$ when $T < 1$ — i.e., a *stronger* regularizer than the principled prior calls for. Across image classification, regression, and language tasks, the optimal $T$ tends to be in the range $0.01$–$0.1$, an order of magnitude or more away from the strict-Bayesian $T = 1$.
>
> The phenomenon is one of the central open problems in BNNs. The two leading hypotheses are: (i) the *Gaussian prior is misspecified* — real-world weight distributions are heavier-tailed than $\mathcal{N}(0, \tau^2 I)$, so the principled prior over-regularizes and tempering compensates; (ii) the *likelihood is misspecified* in some other way (data augmentation, label noise) that interacts with the prior. Aitchison 2021 ("A statistical theory of cold posteriors"), Adlam, Snoek & Smith 2020, and Izmailov et al. 2021 contribute partial resolutions, but the question is not closed.

#### §8.6 Temperature scaling

> **Remark 8.6 (Post-hoc temperature scaling).** A pragmatic alternative to choosing $T$ in the prior is *post-hoc temperature scaling* (Guo et al. 2017): after training a network the standard way, learn a single scalar temperature $\hat T$ that minimizes NLL on a held-out validation set, and rescale all test-time logits by $1/\hat T$. The construction does not change the model's accuracy (the argmax of the rescaled logits matches the argmax of the original logits), but it can dramatically reduce ECE by softening over-confident predictions. Temperature scaling is now the default post-hoc calibration step in production deep-learning pipelines and is included in the `laplace-torch` library as an automatic post-processing step. The §8.7 head-to-head comparison reports each method's NLL both before and after temperature scaling.

#### §8.7 Head-to-head comparison

> **Algorithm 8.7 (Head-to-head calibration evaluation).**
> *Input:* training data $\mathcal{D}$, held-out test set $\mathcal{D}_{\mathrm{test}} = \{(x^*_i, y^*_i)\}_{i=1}^{N_{\mathrm{test}}}$, the five method outputs from §§1, 3, 4, 5, 6, 7 (point estimate, Laplace, MC-dropout, deep ensemble, SGLD, SGHMC).
> *Output:* table of $\{\mathrm{ECE}, \mathrm{BS}, \mathrm{NLL}\}$ for each method, with reliability diagrams.
> 1. For each method $m$, compute predicted class-1 probabilities $\hat{p}^{(m)}_i$ on the test set via Monte Carlo over the method's posterior samples.
> 2. Compute ECE with $B = 10$ bins, BS via Def 8.2, NLL via Def 8.3.
> 3. Plot reliability diagrams: predicted-confidence bin centers on $x$-axis, $\mathrm{acc}(b)$ on $y$-axis, with the diagonal $y = x$ marked as the perfectly-calibrated reference.

For Two Moons we use $N_{\mathrm{test}} = 500$ held-out points generated with `make_moons` at the same noise level as training but a different `random_state`, so the test set is iid from the same distribution as training and ECE is well-defined.

#### Code experiment §8

PyTorch end-to-end. Reuse the trained models from §§1, 3, 4, 5, 6, 7 (`models_5[0]` for the point estimate; `W_lap` from §3; `dropout_model` from §4; `ensemble_models` from §5; `sgld_W` from §6; `sghmc_W` from §7). Generate $N_{\mathrm{test}} = 500$ held-out points with `make_moons(noise=0.20, random_state=SEED+1000)`. For each method, compute the test-set predictive probabilities via the method's Monte Carlo or closed-form recipe. Compute ECE / BS / NLL for each method; plot reliability diagrams. Total cell runtime ~3 s (no new training, just inference). Verification: print the metrics table; expected ordering on Two Moons is roughly *deep ensemble ≈ SGHMC < SGLD < Laplace ≈ MC-dropout < point estimate* on NLL and ECE.

#### Viz design intent §8

**Figure 8 (`08_calibration_comparison.png`).** Two panels.

**Panel (a)** — *Reliability diagram overlay.* X-axis: predicted-confidence bin center ($0.05, 0.15, \ldots, 0.95$). Y-axis: empirical accuracy in that bin. Plot the dashed reference line $y = x$ (perfect calibration), then overlay one connected line per method using the per-method colors from setup (`COLOR_POINT`, `COLOR_LAPLACE`, `COLOR_DROPOUT`, `COLOR_ENSEMBLE`, `COLOR_SGLD`, `COLOR_SGHMC`). Reader should see the point estimate's reliability curve consistently below the diagonal in mid-confidence bins (over-confidence), the BNN methods' curves closer to the diagonal, and the deep-ensemble + SG-MCMC curves typically the closest.

**Panel (b)** — *Metric bar chart.* Three metric groups (ECE, Brier × 10, NLL × 10 — the scaling makes the bars visually comparable) on the x-axis; six methods as grouped bars within each group, colored by `COLOR_*`. Reader should see the BNN methods reduce ECE and NLL relative to the point estimate, with deep ensemble + SG-MCMC typically the strongest.

The PNG is the static fallback. The on-page rendering is the React + D3 component `BNNCalibrationComparisonViz.tsx`.

**Interactive controls (v1 deliverable):**

- *Method-toggle checkboxes* (one per method: Point estimate, Laplace, MC-dropout, Deep ensemble, SGLD, SGHMC; all default on). Toggling a method off removes its line from panel (a) and its bars from panel (b). Lets the reader focus on subsets — e.g., "BNN methods only" by toggling the point estimate off, or "single-mode methods only" by toggling deep ensemble + SG-MCMC off.
- *Bin-count slider* ($B = 5$ to $B = 25$, default $B = 10$). Re-bins the test set and recomputes ECE and the reliability-diagram bin centers. Smaller $B$ gives more stable per-bin estimates with fewer bins; larger $B$ gives finer-grained calibration assessment with noisier bin counts. Reader sees the ECE values shift slightly with $B$ — calibration metrics are not perfectly invariant to binning choice.
- *Temperature scaling toggle* (`Off` / `On`, default `Off`). When on, fits a single scalar temperature $\hat T$ per method on a held-out validation set (200 of the 500 test points), rescales test-set logits by $1/\hat T$, and recomputes the metrics. Reader sees: every method's NLL drops when temperature scaling is applied; ECE generally improves; accuracy is unchanged. Concrete demonstration of Rem 8.6's "free calibration improvement."
- *Cold-posterior temperature slider* ($T \in [0.01, 1.0]$, default $1.0$). Applies temperature to the *prior* by raising the posterior to $1/T$ — for the methods that admit it (Laplace, SG-MCMC; not MC-dropout or deep ensembles, which require re-training). Reader sees: at $T < 1$, the posterior tightens and predictive variance shrinks; ECE can improve dramatically (the Wenzel et al. 2020 cold-posterior effect made interactive). This is the Q6 v1-vs-v2 decision in §10 Open Questions — recommended v1 for at least Laplace, since the recompute is a single $\tau^2 \to \tau^2 / T$ rescale of the cached Hessian decomposition.

The component consumes precomputed per-method test-set predictive probabilities (six arrays of length 500, ~12 KB total — trivial payload). All metric and reliability-bin computations run in-browser in <50 ms per interaction. Temperature-scaling fit is one-dimensional optimization, also <50 ms.

**What the reader should learn from Figure 8:** (i) BNN methods reduce calibration error relative to the point estimate, often dramatically; (ii) the four methods have qualitatively different calibration profiles — Laplace is locally accurate but misses multi-mode regions, MC-dropout systematically underestimates epistemic variance, deep ensembles and SG-MCMC capture the multi-mode posterior and tend to be best-calibrated; (iii) the metric ordering may invert for different domains, especially under distribution shift, so calibration evaluation is a domain-by-domain exercise; (iv) post-hoc temperature scaling (Rem 8.6) typically improves all methods' NLL by a factor of $1.5$–$3\times$ for free.

---

### §9. Function-space view: NNGP, NTK, and open problems

§§3–7 work in *weight space*. Each method approximates the posterior $p(w \mid \mathcal{D})$ over the network's parameters and reads predictions off the resulting weight distribution. The §2.4 obstacles — multimodality, ReLU rescaling, over-parametrization — are all weight-space pathologies, and the methods of §§3–7 are weight-space workarounds. *Function space* offers a different vantage. What we ultimately care about is the predictive distribution over outputs, which lives in function space; weight space is the awkward intermediate representation. Two classical results — Neal's 1996 *neural network Gaussian process* (NNGP) and Jacot, Gabriel & Hongler's 2018 *neural tangent kernel* (NTK) — show that in the infinite-width limit, both Bayesian inference and gradient-descent training reduce to operations on a fixed kernel function. The function-space view connects BNNs to [Gaussian Processes](/topics/gaussian-processes), explains why deep ensembles work, and provides the asymptotic reference against which the §§3–7 methods can be evaluated. This section gives the four key facts, with proofs deferred to references; the running example is one panel of NNGP-prior samples that visualizes the infinite-width convergence.

#### §9.1 The neural network Gaussian process

> **Remark 9.1 (NNGP — Neal 1996; Lee et al. 2017).** Consider an MLP with $L$ hidden layers of widths $h_1, \ldots, h_L$, weights $W^{(\ell)}_{ij} \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(0, \sigma_w^2 / h_{\ell - 1})$, biases $b^{(\ell)}_j \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(0, \sigma_b^2)$, and elementwise nonlinearity $\phi$. As $h_\ell \to \infty$ for all $\ell$, the prior over the function $f_w(x)$ converges, in the sense of finite-dimensional distributions, to a Gaussian process $\mathcal{GP}(0, k_{\mathrm{NNGP}})$ with covariance kernel computable by the recursion
> $$k^{(0)}(x, x') = \sigma_b^2 + \sigma_w^2\,\frac{x \cdot x'}{d}, \qquad k^{(\ell)}(x, x') = \sigma_b^2 + \sigma_w^2\,\mathbb{E}_{(u, u') \sim \mathcal{N}(0, K^{(\ell-1)}(x, x'))}[\phi(u)\,\phi(u')],$$
> with $K^{(\ell-1)}(x, x') = \begin{pmatrix} k^{(\ell-1)}(x, x) & k^{(\ell-1)}(x, x') \\ k^{(\ell-1)}(x', x) & k^{(\ell-1)}(x', x') \end{pmatrix}$. The output kernel is $k_{\mathrm{NNGP}} = k^{(L)}$. For ReLU, the per-layer expectation has the closed-form *arc-cosine kernel* (Cho & Saul 2009).

Neal's 1996 PhD thesis proved the one-hidden-layer case via a direct CLT argument: each output $f_w(x) = \sum_j w^{(2)}_j \phi(W^{(1)}_j x + b^{(1)}_j)$ is a sum of $h_1$ iid bounded random variables (in the sense that suitable moments are bounded), so the CLT applies and $f_w(x)$ converges to a Gaussian. Joint distributions across multiple inputs $x_1, \ldots, x_n$ likewise converge to a multivariate Gaussian — i.e., a Gaussian process. Lee et al. 2017 extended the argument to deep networks via induction over layers, giving the recursive kernel formula above.

The implication is structural: at infinite width, BNN posterior inference reduces to GP regression (or GP classification, via Laplace or EP, per [Gaussian Processes](/topics/gaussian-processes) §5). The NNGP kernel encodes the entire architectural choice — depth, activation, prior scales — and once the kernel is in hand the $O(n^3)$ GP-inference machinery applies directly. There is no weight-space optimization, no MAP, no Hessian, no SGLD. The infinite-width BNN is a Gaussian process.

#### §9.2 The neural tangent kernel

> **Remark 9.2 (NTK — Jacot, Gabriel & Hongler 2018).** Consider an MLP trained by gradient descent on the squared-error loss starting from random initialization $w_0$. The *neural tangent kernel* at $w_0$ is
> $$\Theta(x, x') \;:=\; \bigl\langle \nabla_w f_w(x),\, \nabla_w f_w(x') \bigr\rangle\bigm|_{w = w_0}.$$
> As widths $\to \infty$, $\Theta$ becomes deterministic (concentrated at its expectation $\Theta_*$) and *constant during training* — the gradient is dominated by the linearization at $w_0$, and the network evolves as if it were the linearized model $f_{w_0}(x) + \nabla_w f_{w_0}(x)^\top (w - w_0)$. The training dynamics converge to a deterministic ODE in function space:
> $$\frac{df_t(x)}{dt} \;=\; -\sum_{i=1}^n \Theta_*(x, x_i)\,(f_t(x_i) - y_i),$$
> whose solution at $t \to \infty$ is exactly the kernel-regression predictor under $\Theta_*$.

Jacot et al.'s argument has two parts. First, at initialization, the gradient $\nabla_w f_{w_0}(x)$ is itself a random function whose pairwise inner products converge in the infinite-width limit to a deterministic kernel $\Theta_*$ — a CLT analogous to NNGP's. Second, during training, the change in weights $w_t - w_0$ stays small in a width-dependent norm (the *lazy regime* of Chizat & Bach 2019), so the linearization $f_w(x) \approx f_{w_0}(x) + \nabla_w f_{w_0}(x)^\top (w - w_0)$ remains valid and the network's outputs evolve linearly. The combination gives kernel-regression dynamics in function space.

The NTK is *not* the same as the NNGP kernel. NNGP describes the *prior* over functions before any training; NTK describes the *trained* function under gradient descent. In general $\Theta_{\mathrm{NTK}}(x, x') \neq k_{\mathrm{NNGP}}(x, x')$, and at infinite width they describe two different inference regimes (Bayesian posterior vs. gradient-descent training). Lee et al. 2019 ("Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent") gave the precise statement of NTK convergence and the relationship between the two kernels.

#### §9.3 What this means for the §§3–7 methods

> **Remark 9.3 (Function-space asymptotic ordering).** At infinite width, the §§3–7 methods can be ordered by how well they recover the NNGP posterior.
>
> *SG-MCMC (§§6–7).* Asymptotically exact: in the joint limit of infinite width and infinitesimal step size with infinite chain length, SGLD and SGHMC sample exactly the NNGP posterior. This is the strongest function-space guarantee in the catalog.
>
> *Deep ensembles (§5).* Approximately correct: the modes of an infinitely-wide network with random initialization are approximately samples from the NNGP prior, so $K$-ensemble averaging at large $K$ recovers the NNGP posterior up to mode-coverage error (Wilson & Izmailov 2020). The argument is delicate but the spirit is right.
>
> *Laplace (§3).* Coarse approximation: a single local Gaussian centered at one MAP estimate captures the local curvature of the NNGP posterior at one point, but misses the full function-space distribution. At infinite width the NNGP posterior is itself a Gaussian, and Laplace's approximation converges to a sub-Gaussian within it — the bias is $O(1)$ in the function-space sense.
>
> *MC-dropout (§4).* Structurally biased: the Bernoulli variational family does *not* converge to the NNGP posterior at any width, because no Bernoulli posterior over weights induces a Gaussian process over functions. Foong et al. 2020 ("On the Expressiveness of Approximate Inference in Bayesian Neural Networks") formalize this gap.

The §8 head-to-head ordering on Two Moons is roughly consistent with this asymptotic ordering: deep ensembles and SG-MCMC are the most-calibrated, Laplace is in the middle, and MC-dropout is at the bottom. For finite-width networks the differences are smaller than the infinite-width view suggests, and for some narrow regimes the ordering inverts — but the function-space asymptotic gives the right intuition for which method to reach for first.

#### §9.4 Open problems

> **Remark 9.4 (Active research directions).** Four open problems organize ongoing work in BNNs.
>
> *The cold-posterior effect.* Wenzel et al. 2020's empirical observation (§8.5) that BNN performance improves under tempering with $T < 1$ remains poorly understood. Aitchison 2021, Adlam, Snoek & Smith 2020, and Izmailov et al. 2021 offer partial explanations — prior misspecification, label noise, data augmentation interacting with the implicit posterior — but no consensus has emerged.
>
> *Finite-width corrections to NNGP and NTK.* The infinite-width theory is clean but practical networks have finite width. The leading-order correction in $1 / h$ is the subject of active work — Yang & Hu 2021 ("Feature Learning in Infinite-Width Neural Networks") propose a width-rescaled regime where features are learned even at infinite width; Bordelon, Canatar & Pehlevan 2020 give explicit finite-width corrections to NTK regression. The gap between the infinite-width theory and practical BNN inference at finite width remains one of the most active questions in deep-learning theory.
>
> *Scalable asymptotically-exact MCMC.* Beyond SGHMC, methods like *symmetric splitting integrators* (Leimkuhler & Matthews 2013), *cyclic SG-MCMC* (Zhang, Sun, Duvenaud & Grosse 2020), and *full-batch HMC at scale* (Izmailov et al. 2021) push the asymptotically-exact frontier toward ImageNet-class production. The planned `stochastic-gradient-mcmc` *(coming soon)* topic will develop these methods in detail.
>
> *Function-space variational inference.* Inferring directly in function space rather than weight space — Wang, Shi & Cheng 2019 ("Function-space VI through Stein discrepancy"), Sun, Zhang, Shi & Grosse 2019, Burt, Ober, Garriga-Alonso & van der Wilk 2020 — sidesteps the weight-space identifiability problems §2.4 catalogues. The conceptual appeal: do inference where the inference target lives. The technical difficulty: function space is infinite-dimensional, so variational families and divergences have to be chosen carefully.

#### Code experiment §9

A single sidebar figure illustrating the NNGP convergence on a 1D toy regression problem — *not* the Two Moons running example. PyTorch + NumPy. Total cell runtime ~3 s. Key library calls: `numpy.random.normal` for sampling random-init MLP weights at three widths ($h = 4, 32, 256$), forward-pass each width's $K = 5$ MLPs on a 1D grid, plot the resulting prior samples; compute the closed-form NNGP-arc-cosine kernel via the Cho-Saul recursion at the same grid and overlay one prior sample from $\mathcal{GP}(0, k_{\mathrm{NNGP}})$ as the infinite-width reference. Verification: print the empirical variance of the function values at the grid for each width and compare to the closed-form NNGP kernel diagonal; expected ratio $\approx 1$ as width grows.

#### Viz design intent §9

**Figure 9 (`09_nngp_sidebar.png`).** Two panels.

**Panel (a)** — *NNGP-prior convergence.* On a 1D input grid $x \in [-3, 3]$, plot $K = 5$ random-init MLP function samples $f_w(x)$ for each of three widths $h \in \{4, 32, 256\}$ — color-coded by width using a perceptually-ordered colormap. Overlay one closed-form NNGP-prior sample (drawn from $\mathcal{N}(0, K_{\mathrm{NNGP}})$ via Cholesky) in black. Reader should see the finite-width samples becoming progressively smoother and statistically indistinguishable from the closed-form NNGP sample as width grows — visual confirmation of Remark 9.1.

**Panel (b)** — *NNGP regression posterior on a toy dataset.* On the same 1D input grid, fix a small synthetic regression dataset (e.g., $\sin(x) + \mathrm{noise}$ at 8 training points). Plot the closed-form NNGP-kernel GP regression posterior mean $\pm 2 \sigma$ band, plus 5 posterior function samples. Reader should see a coherent regression posterior that grows uncertain between training points — exactly the GP regression picture from [Gaussian Processes](/topics/gaussian-processes) topic, now derived from an infinite-width MLP.

The PNG is the static fallback. The on-page rendering is the React + D3 component `NNGPSidebarViz.tsx`. Even though §9 is a sidebar, the interactive controls earn their place — the convergence-with-width phenomenon is much more compelling when the reader drives the slider themselves.

**Interactive controls (v1 deliverable):**

- *Width slider* ($h$ on a log scale from $h = 2$ to $h = 1024$). The default panel (a) shows $K = 5$ random-init MLP samples at the slider value plus the closed-form NNGP overlay. Reader sees the convergence in real time: at $h = 2$ the samples are visibly piecewise-linear (only two ReLU pieces possible), at $h = 32$ they look almost like the NNGP, at $h = 1024$ they are statistically indistinguishable from the closed-form sample. Concrete demonstration of Remark 9.1.
- *Activation function dropdown* (`ReLU` / `erf` / `tanh`, default `ReLU`). Switches the NNGP kernel via the corresponding closed-form recursion (arc-cosine for ReLU, Cho-Saul-style integrals for erf and tanh). Reader sees: different activations give qualitatively different prior behaviors — ReLU samples are piecewise-linear-looking, tanh samples are smoother and bounded. Pedagogically connects to the §9.1 recursion's dependence on $\phi$.
- *Training-set-size slider* ($n = 2$ to $n = 20$, default $n = 8$) for panel (b). Re-fits the NNGP regression posterior on the new training set. Reader sees the posterior tightening between training points as $n$ grows — the standard GP-regression intuition, now arising from an infinite-width MLP.

The component computes NNGP kernels in closed form (no training, no MC), so all interactions render in <100 ms with no pre-compute beyond the kernel-matrix Cholesky factorization at the default width.

**What the reader should learn from Figure 9:** (i) the infinite-width MLP prior really does converge to a Gaussian process, and the convergence is observable visually at moderate widths ($h = 32$ is already close to the infinite-width sample); (ii) NNGP regression is a closed-form, no-training BNN inference recipe at the infinite-width limit; (iii) the §§3–7 weight-space methods are *finite-width approximations* to the function-space picture this figure makes concrete; (iv) for problems where the infinite-width approximation is good enough, doing GP inference directly is often cheaper and more accurate than any of the §§3–7 weight-space methods; (v) accessible via the slider — different activation functions give different priors, which the §3–7 methods inherit.

#### Connections and further reading

The function-space view connects this topic to neighboring formalML topics. [Gaussian Processes](/topics/gaussian-processes) develops the GP machinery the NNGP relies on — Cholesky factorization of the kernel matrix, conditional-MVN posteriors, hyperparameter learning by marginal likelihood. [Variational Inference](/topics/variational-inference) is the substrate for §4's MC-dropout derivation and for the planned function-space VI of Rem 9.4. [Stacking & Predictive Ensembles](/topics/stacking-and-predictive-ensembles) generalizes §5's uniform-weighted deep ensemble to learned weights. The planned topics `stochastic-gradient-mcmc` and `meta-learning` build directly on §§6–7's SG-MCMC machinery; `sparse-bayesian-priors` will revisit §2.1's prior choice with heavy-tailed alternatives that resolve some of Rem 9.4's cold-posterior questions. Cross-site, formalstatistics's [`bayesian-foundations-and-prior-selection`](https://www.formalstatistics.com/topics/bayesian-foundations-and-prior-selection) provides the prior-and-likelihood machinery this topic takes as given, and [`central-limit-theorem`](https://www.formalstatistics.com/topics/central-limit-theorem) is the rigorous source of the Bernstein–von Mises invocation in §2.3.

---

## 5. Shared Module Additions (`src/components/viz/shared/bayesian-ml.ts`)

The strategic-planning doc §13 names five primitives this topic should contribute to the T5 shared module: `mcDropoutInference`, `deepEnsembleTraining`, `laplaceApproxBNN`, `sgMCMCBNNTraining`, `bnnCalibrationDiagnostic`. **These primitives back the interactive React + D3 components in §§1–9** — the components call into the shared module on every slider / toggle interaction, so the primitives must be efficient and have stable APIs. The brief's notebook is the source of truth for the math; the TypeScript signatures below are the *intent* of each primitive — the React/D3 viz layer at MDX implementation time will fill in concrete types.

```typescript
// src/components/viz/shared/bayesian-ml.ts (extended)

export interface MLPArchSpec {
  inputDim: number;
  hiddenDims: number[];   // e.g., [32, 32, 32] for the §1 architecture
  outputDim: number;
  activation: 'relu';
  dropoutP?: number;      // 0 for §§1-3, 5-8; >0 for §4
}

export interface TrainingSpec {
  lr: number;
  weightDecay: number;
  epochs: number;
  optimizer: 'adam';
  seed: number;
}

/** §1, §5, §8: train K independent MLPs from different seeds. */
export interface DeepEnsembleResult {
  weights: Float32Array[];          // K weight vectors
  finalLosses: number[];            // K final training losses
  predictOnGrid: (grid: Float32Array) => Float32Array[];  // K probability arrays
}
export function deepEnsembleTraining(
  arch: MLPArchSpec, training: TrainingSpec, K: number, data: TrainingData
): DeepEnsembleResult;

/** §3, §8: full-Hessian Laplace approximation around the MAP (small networks only).
 *  For larger networks, use lastLayerLaplace() or kfacLaplace() variants. */
export interface LaplaceResult {
  wMap: Float32Array;
  hessianCholesky: Float32Array;    // L such that H = L L^T
  pDim: number;
  conditionNumber: number;
  sampleWeights: (S: number, seed: number) => Float32Array[];  // S Laplace samples
}
export function laplaceApproxBNN(
  arch: MLPArchSpec, training: TrainingSpec, data: TrainingData, deltaStabilization?: number
): LaplaceResult;
export function lastLayerLaplace(/* ... */): LaplaceResult;
export function kfacLaplace(/* ... */): LaplaceResult;

/** §4, §8: train a dropout MLP, run T stochastic forward passes at test time. */
export interface MCDropoutResult {
  trainedModel: TrainedMLP;
  predict: (grid: Float32Array, T: number, seed: number) => Float32Array[];  // T prob arrays
}
export function mcDropoutInference(
  arch: MLPArchSpec, training: TrainingSpec, data: TrainingData
): MCDropoutResult;

/** §§6-7, §8: SGLD or SGHMC chain. */
export interface SGMCMCSpec {
  method: 'SGLD' | 'SGHMC';
  eta: number;
  batchSize: number;
  burnIn: number;
  samples: number;
  thin: number;
  friction?: number;       // SGHMC only
  warmStart?: Float32Array; // optional MAP warm-start
  seed: number;
}
export interface SGMCMCResult {
  weights: Float32Array[];           // post-burn-in, thinned weight samples
  weightTrace: number[];             // single-component trace for diagnostics
  autocorrelation: number[];         // ACF of the trace, lag 0..maxLag
}
export function sgMCMCBNNTraining(
  arch: MLPArchSpec, training: TrainingSpec, data: TrainingData, spec: SGMCMCSpec
): SGMCMCResult;

/** §8: calibration diagnostics. */
export interface CalibrationMetrics {
  ece: number;
  brier: number;
  nll: number;
  accuracy: number;
  reliabilityBins: { binConf: number; binAcc: number; binCount: number }[];
}
export function bnnCalibrationDiagnostic(
  testProbs: Float32Array, testLabels: Float32Array, nBins?: number
): CalibrationMetrics;

/** §9: closed-form NNGP arc-cosine kernel for ReLU networks (single hidden layer). */
export function nngpArcCosineKernel(
  X: Float32Array[], Xprime: Float32Array[], sigmaW2: number, sigmaB2: number
): number[][];
```

The `TrainedMLP` type is the existing model wrapper used elsewhere in `bayesian-ml.ts` (see VI's `meanFieldELBO` and GPs's `gpPredict` for precedent). The `TrainingData` type holds `(X, y)` arrays.

**Verification tests** (per VI brief §5 pattern). Each primitive must reproduce the corresponding notebook cell's printed numerical output to a tolerance of at most 1% on the variance-ratio metrics, 5% on autocorrelation times, and absolute differences below `1e-3` for ECE/Brier/NLL on the §8 reference test set.

---

## 6. Curriculum Graph Updates

**Node to add to `src/data/curriculum-graph.json`:**
```json
{ "id": "bayesian-neural-networks", "label": "Bayesian Neural Networks", "domain": "bayesian-ml", "status": "published", "url": "/topics/bayesian-neural-networks" }
```

**Edges to add** (prerequisite → BNN):
```json
{ "source": "variational-inference", "target": "bayesian-neural-networks" }
{ "source": "gaussian-processes", "target": "bayesian-neural-networks" }
{ "source": "kl-divergence", "target": "bayesian-neural-networks" }
{ "source": "gradient-descent", "target": "bayesian-neural-networks" }
{ "source": "convex-analysis", "target": "bayesian-neural-networks" }
```

**Changes to `src/data/curriculum.ts`:** remove `"Bayesian Neural Networks"` from the `bayesian-ml` track's `planned` array; add to the `published` array (or equivalent state-transition the existing track schema uses — verify against the most recently shipped T5 topic, `stacking-and-predictive-ensembles`).

**Domain key:** `bayesian-ml` (already minted by VI; no new domain key needed).

---

## 7. Cross-References (Internal formalML)

**Outbound** (BNN MDX links to existing topics — already in §1.2 / §4.3 / §5.4 / §9 prose):

| Section | Link target | Anchor text | Purpose |
|---------|-------------|-------------|---------|
| §1.2 | `/topics/variational-inference` | "[Variational Inference](/topics/variational-inference)" | Forward pointer for the four-obstacles narrative |
| §2.3 | `https://www.formalstatistics.com/topics/central-limit-theorem` | "[`central-limit-theorem`](https://www.formalstatistics.com/topics/central-limit-theorem)" | BvM source — sister site (will become `<ExternalLink>` once that component is ported) |
| §4.3 | `/topics/variational-inference` | "[Variational Inference](/topics/variational-inference)" | ELBO definition referenced in MC-dropout proof |
| §5.4 | `/topics/stacking-and-predictive-ensembles` | "[Stacking & Predictive Ensembles](/topics/stacking-and-predictive-ensembles)" | Forward link from deep ensembles to learned-weight stacking |
| §9 | `/topics/gaussian-processes` | "[Gaussian Processes](/topics/gaussian-processes)" | Function-space view connection (NNGP → GP regression) |
| §9 | `https://www.formalstatistics.com/topics/bayesian-foundations-and-prior-selection` | "[`bayesian-foundations-and-prior-selection`](https://...)" | Sister-site prereq |

**Plain-text forward pointers** (for unwritten formalML topics — must remain plain text per CLAUDE.md):

| Section | Topic | Anchor text |
|---------|-------|-------------|
| §2.1 | `sparse-bayesian-priors` | "`sparse-bayesian-priors` *(coming soon)*" |
| §9.4 | `stochastic-gradient-mcmc` | "`stochastic-gradient-mcmc` *(coming soon)*" |

If `<ExternalLink>` (the cross-site MDX component documented in CLAUDE.md / handoff-reference §10) has already been ported by the time this topic ships, convert the two formalstatistics URLs above to `<ExternalLink site="formalstatistics" topic="…">…</ExternalLink>`. Otherwise leave as plain Markdown URLs and capture the conversion as a follow-up item in `docs/plans/deferred-reciprocals.md`.

---

## 8. Inbound MDX Edits (existing topics that should link to BNN)

After BNN ships, the following published-topic MDX files should be updated to add forward pointers and `connections` frontmatter entries.

**`src/content/topics/variational-inference.mdx`:**
- Add to `connections` frontmatter array:
  ```yaml
  - topic: "bayesian-neural-networks"
    relationship: "MC-dropout (BNN §4) is a Bernoulli variational family on weights; the BNN topic uses VI's ELBO machinery as substrate."
  ```
- In §6 (Connections), upgrade the existing plain-text "Bayesian neural networks" mention to a Markdown link `[Bayesian Neural Networks](/topics/bayesian-neural-networks)`.

**`src/content/topics/gaussian-processes.mdx`:**
- Add to `connections` frontmatter array:
  ```yaml
  - topic: "bayesian-neural-networks"
    relationship: "The infinite-width MLP prior converges to a Gaussian process (NNGP); the BNN topic's §9 derives this connection and uses GP machinery for the function-space view."
  ```
- In the connections / further-reading section, add a forward link to BNN with anchor text "Bayesian Neural Networks" pointing at `/topics/bayesian-neural-networks`.

**`src/content/topics/stacking-and-predictive-ensembles.mdx`:**
- Add to `connections` frontmatter array:
  ```yaml
  - topic: "bayesian-neural-networks"
    relationship: "Deep ensembles (BNN §5) are the special case of stacking with K same-architecture, same-prior candidates and uniform weights; the topics are complementary."
  ```
- In the §5 Rem-on-deep-ensembles or the connections section, link forward to BNN.

**Cross-site reciprocity** (formalstatistics PRs; tracked in `deferred-reciprocals.md` "When `formalml/bayesian-neural-networks` ships" section): on shipment, open a sister-site PR adding reciprocal `formalmlConnections` entries to the five formalstatistics topics declared as `formalstatisticsPrereqs` in BNN's frontmatter (`bayesian-foundations-and-prior-selection`, `central-limit-theorem`, `modes-of-convergence`, `multivariate-distributions`, `hierarchical-bayes-and-partial-pooling`). Each gets a `formalmlConnections` entry pointing back to `bayesian-neural-networks` with `relationship` prose ≥40 chars. After the sister-site PR merges, run `pnpm audit:cross-site` from formalML to confirm the deferred entry disappears.

---

## 9. Images

All §§1–9 figures live at `public/images/topics/bayesian-neural-networks/0N_*.png`, generated by the notebook and copied flat (no subdirectories). **These PNGs are *static fallbacks* used when JavaScript is disabled or before the React component hydrates.** The primary on-page rendering for every section is the corresponding interactive React + D3 component (per §4's "Interactive controls (v1 deliverable)" subsection in each section's viz design intent). The MDX imports both: the static `<Figure>` rendering of the PNG with a caption, *and* the interactive component with `client:visible`. The interactive component visually replaces the static figure when it hydrates.

| Component | File | Source cell | Description |
|-----------|------|-------------|-------------|
| `BNNPredictiveMotivationViz.tsx` | `01_point_vs_bayesian_predictive.png` | §1 | Three-panel motivation: point-estimate predictive, 5-MLP overlapping boundaries, predictive-variance heatmap. **Interactive:** noise slider, $K$ slider, resample button. |
| `LossLandscapeModesViz.tsx` | `02_loss_landscape_modes.png` | §2 | Two-panel: PCA-projected weights of 20 trained MLPs, loss along linear interpolation between two modes. **Interactive:** click-to-pick endpoints, $N$-models slider, interp-grid slider. |
| `LaplaceBNNViz.tsx` | `03_laplace_two_moons.png` | §3 | Three-panel Laplace BNN: predictive mean, predictive std, 20 sampled decision boundaries. **Interactive:** prior-scale slider, curvature-reduction dropdown, sample-count slider. |
| `MCDropoutBNNViz.tsx` | `04_mcdropout_two_moons.png` | §4 | Three-panel MC-dropout: predictive mean, std, 20 sampled boundaries. **Interactive:** dropout-rate slider, $T$ slider, dropout-on-at-test toggle. |
| `DeepEnsembleViz.tsx` | `05_deep_ensemble_two_moons.png` | §5 | Three-panel $K=10$ deep ensemble: mean, std, $K$ ensemble-member boundaries. **Interactive:** $K$ slider, member-highlight click, resample button. |
| `SGLDBNNViz.tsx` | `06_sgld_two_moons.png` | §6 | Three-panel SGLD: mean, std, weight-component trace. **Interactive:** step-size slider, batch-size slider, schedule toggle. |
| `SGHMCBNNViz.tsx` | `07_sghmc_two_moons.png` | §7 | Three-panel SGHMC: mean, std, autocorrelation comparison. **Interactive:** friction slider, step-size slider, method-overlay toggle. |
| `BNNCalibrationComparisonViz.tsx` | `08_calibration_comparison.png` | §8 | Two-panel head-to-head: reliability diagrams overlay (6 methods), grouped bar chart of ECE/Brier/NLL. **Interactive:** method-toggle checkboxes, bin-count slider, temperature-scaling toggle, cold-posterior temperature slider. |
| `NNGPSidebarViz.tsx` | `09_nngp_sidebar.png` | §9 | Two-panel NNGP sidebar: width-convergence panel + GP regression panel. **Interactive:** width slider, activation dropdown, training-set-size slider. |

**Sample-data dual-location.** Per CLAUDE.md, viz components that fetch precomputed JSON at runtime require the JSON to live in *both* `src/data/sampleData/bayesian-neural-networks/` (tracked, target of the precompute script) and `public/sample-data/bayesian-neural-networks/` (Astro serves only `public/`). The `src/components/viz/SGLDBNNViz.tsx` and `SGHMCBNNViz.tsx` components in particular consume precomputed chain trajectories per Q4 in §10; the precompute Python script must write to both directories from inside `main()` per the convention documented in CLAUDE.md "Sample-data dual-location."

---

## 10. Open Questions for Implementation

The pre-brief drafting session left the following decisions for the Claude Code implementation phase. None block §13's build-order step 1, but each should be resolved before its referenced step.

1. **Formalcalculus SDE prereq.** The proposed frontmatter (§2) lists `topic: "stochastic-differential-equations"` under `formalcalculusPrereqs` for §§6–7. If formalcalculus does not yet have this topic, drop the entry, leave §§6–7 self-contained on the SDE math, and route the cross-site forward-pointer (if any) through `bayesian-computation-and-mcmc` on formalstatistics instead. *Resolve before:* §13 step 2 (frontmatter writing).

2. **`<ExternalLink>` component availability.** The §7 cross-references table assumes `src/components/ui/ExternalLink.astro` is implemented. If not — likely if no prior topic has shipped a body link to formalstatistics — port it during this topic per CLAUDE.md / handoff-reference §10, or leave the formalstatistics URLs as plain Markdown URLs and flag in `deferred-reciprocals.md`. *Resolve before:* §13 step 7 (MDX cross-references).

3. **Full Hessian vs. last-layer Laplace in §3 viz.** The notebook uses full Hessian via `torch.autograd.functional.hessian` because $p = 2241$ is feasible. The React/D3 viz component for §3 should mirror the notebook (precomputed JSON output of the Laplace samples, no in-browser Hessian computation). Confirm the JSON-payload approach is acceptable, or if Claude Code prefers in-browser computation via the §5 `laplaceApproxBNN` shared primitive; the latter is more elegant but costs a ~1 s in-browser blocking computation on first load. *Resolve before:* §13 step 5 (§3 viz component).

4. **SGLD / SGHMC chain length in viz.** The notebook runs 1200 SGLD iterations + 1200 SGHMC iterations. Re-running these in-browser would be too slow; the React component should consume precomputed JSON output (sampled weight vectors + per-grid-point predictive probabilities). Confirm payload size budget — ~100 samples × 40,000 grid points × 4 bytes = 16 MB per method, which is too large; the payload should pre-compute the per-grid-point mean and standard deviation only (~640 KB per method, total ~5 MB across §§3-7 viz). *Resolve before:* §13 step 6–7 (§§6-7 viz components).

5. **Calibration test set choice.** The notebook uses Two Moons held-out at $N_{\mathrm{test}} = 500$. For the live MDX, this calibration evaluation could be re-run on a different test set or seed range. Confirm the locked test set or expose a seed slider. *Resolve before:* §13 step 8 (§8 viz component).

6. **Cold-posterior interactive demo.** §8.5 introduces the cold-posterior effect as a remark; an interactive viz that lets the reader vary the temperature $T$ on a log scale and watch ECE/NLL/accuracy redraw would make the effect concrete. Decide whether to ship this in v1 or defer to v2. *Resolve before:* §13 step 8.

---

## 11. References

Chicago 17th edition (Notes and Bibliography). Every entry includes a `url` field. Grouped as **foundational** (the canonical primary sources for each method), **advanced** (theoretical refinements and finite-width corrections), and **applied** (practical-implementation references and the modern revival of the methods).

The frontmatter `references` array (per the topic schema in `src/content/topics/`) takes a list of objects with `type: "book" | "paper"`, `title`, `authors`, `year`, `url`, `note`. Below is the bibliography in roughly that schema, ready to translate.

### Foundational

```yaml
- type: "paper"
  title: "Bayesian Learning for Neural Networks"
  authors: "Neal, Radford M."
  year: 1996
  url: "https://doi.org/10.1007/978-1-4612-0745-0"
  note: "Originating reference for both Bayesian neural networks and the NNGP. PhD thesis published as Springer Lecture Notes in Statistics 118; the infinite-width-prior-equals-GP result is in Chapter 2."
- type: "paper"
  title: "A Practical Bayesian Framework for Backpropagation Networks"
  authors: "MacKay, David J. C."
  year: 1992
  url: "https://doi.org/10.1162/neco.1992.4.3.448"
  note: "Originating reference for the Laplace approximation to BNN posteriors and the moment-matched probit predictive (BNN §3.3 Prop 3.3 attribution)."
- type: "paper"
  title: "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"
  authors: "Gal, Yarin and Zoubin Ghahramani"
  year: 2016
  url: "https://proceedings.mlr.press/v48/gal16.html"
  note: "MC-dropout. Theorem 4.2 (dropout training is variational inference) is this paper's main result; the predictive recipe of §4.4 is its implementation contribution."
- type: "paper"
  title: "Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles"
  authors: "Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell"
  year: 2017
  url: "https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html"
  note: "Originating reference for deep ensembles. The §5 construction is theirs; the §5.3 mixture-of-Gaussians predictive form follows their §3.2."
- type: "paper"
  title: "Bayesian Learning via Stochastic Gradient Langevin Dynamics"
  authors: "Welling, Max and Yee Whye Teh"
  year: 2011
  url: "https://www.icml-2011.org/papers/398_icmlpaper.pdf"
  note: "SGLD. The asymptotic-exactness Theorem 6.2 is their main result; the step-size-schedule conditions are their Robbins-Monro requirements."
- type: "paper"
  title: "Stochastic Gradient Hamiltonian Monte Carlo"
  authors: "Chen, Tianqi, Emily B. Fox, and Carlos Guestrin"
  year: 2014
  url: "https://proceedings.mlr.press/v32/cheni14.html"
  note: "SGHMC. The friction-noise compensation construction (Definition 7.1) and Theorem 7.2 (stationary distribution under exact noise compensation) are theirs."
- type: "paper"
  title: "Neural Tangent Kernel: Convergence and Generalization in Neural Networks"
  authors: "Jacot, Arthur, Franck Gabriel, and Clément Hongler"
  year: 2018
  url: "https://proceedings.neurips.cc/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html"
  note: "NTK. The infinite-width gradient-descent-as-kernel-regression result of Remark 9.2."
```

### Advanced

```yaml
- type: "paper"
  title: "Deep Neural Networks as Gaussian Processes"
  authors: "Lee, Jaehoon, Yasaman Bahri, Roman Novak, Samuel S. Schoenholz, Jeffrey Pennington, and Jascha Sohl-Dickstein"
  year: 2017
  url: "https://arxiv.org/abs/1711.00165"
  note: "Generalizes Neal 1996 to deep networks. The recursive NNGP kernel formula in Remark 9.1 is theirs."
- type: "paper"
  title: "How Good is the Bayes Posterior in Deep Neural Networks Really?"
  authors: "Wenzel, Florian, Kevin Roth, Bastiaan S. Veeling, Jakub Świątkowski, Linh Tran, Stephan Mandt, Jasper Snoek, Tim Salimans, Rodolphe Jenatton, and Sebastian Nowozin"
  year: 2020
  url: "https://proceedings.mlr.press/v119/wenzel20a.html"
  note: "Cold-posterior effect. The §8.5 / §9.4 open problem statement and the empirical evidence motivating it are theirs."
- type: "paper"
  title: "Bayesian Deep Learning and a Probabilistic Perspective of Generalization"
  authors: "Wilson, Andrew Gordon and Pavel Izmailov"
  year: 2020
  url: "https://proceedings.neurips.cc/paper/2020/hash/322f62469c5e3c7dc3e58f5a4d1ea399-Abstract.html"
  note: "Function-space-posterior interpretation of deep ensembles (Remark 9.3 attribution); argues that ensemble diversity in function space is what matters."
- type: "paper"
  title: "Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent"
  authors: "Lee, Jaehoon, Lechao Xiao, Samuel S. Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington"
  year: 2019
  url: "https://proceedings.neurips.cc/paper/2019/hash/0d1a9651497a38d8b1c3871c84528bd4-Abstract.html"
  note: "Makes precise the NTK convergence and the relationship between the NNGP and NTK kernels."
- type: "paper"
  title: "On the Expressiveness of Approximate Inference in Bayesian Neural Networks"
  authors: "Foong, Andrew Y. K., David R. Burt, Yingzhen Li, and Richard E. Turner"
  year: 2020
  url: "https://proceedings.neurips.cc/paper/2020/hash/b6dfd41875bc090bd31d0b1740eb5b1b-Abstract.html"
  note: "Quantifies the gap between Bernoulli-family MC-dropout and the true posterior. Source for Remark 4.4's saturation observation."
- type: "paper"
  title: "What Are Bayesian Neural Network Posteriors Really Like?"
  authors: "Izmailov, Pavel, Sharad Vikram, Matthew D. Hoffman, and Andrew Gordon Wilson"
  year: 2021
  url: "https://proceedings.mlr.press/v139/izmailov21a.html"
  note: "Full-batch HMC at deep-learning scale; reference posteriors against which §§3-7 approximations can be compared. Cited in Remark 9.4."
- type: "paper"
  title: "Stochastic Gradient Descent as Approximate Bayesian Inference"
  authors: "Mandt, Stephan, Matthew D. Hoffman, and David M. Blei"
  year: 2017
  url: "https://www.jmlr.org/papers/v18/17-214.html"
  note: "Connection between SGD-with-decay and SGLD; useful background for §6's bias-variance tradeoff (Proposition 6.4)."
- type: "paper"
  title: "Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning"
  authors: "Zhang, Ruqi, Chunyuan Li, Jianyi Zhang, Changyou Chen, and Andrew Gordon Wilson"
  year: 2020
  url: "https://openreview.net/forum?id=rkeS1RVtPS"
  note: "Cyclic SG-MCMC; one of the open-problem references in Remark 9.4."
```

### Applied

```yaml
- type: "paper"
  title: "Laplace Redux — Effortless Bayesian Deep Learning"
  authors: "Daxberger, Erik, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, and Philipp Hennig"
  year: 2021
  url: "https://proceedings.neurips.cc/paper/2021/hash/a7c9585703d275249f30a088cebba0ad-Abstract.html"
  note: "The `laplace-torch` library and the modern revival of the Laplace approximation for BNNs. Reference for §3.4 (last-layer / KFAC reductions) and §8.6 (post-hoc temperature scaling integration)."
- type: "paper"
  title: "On Calibration of Modern Neural Networks"
  authors: "Guo, Chuan, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger"
  year: 2017
  url: "https://proceedings.mlr.press/v70/guo17a.html"
  note: "Originating reference for post-hoc temperature scaling (Remark 8.6) and the empirical observation that modern deep networks are systematically over-confident (motivating §8 entirely)."
- type: "paper"
  title: "Strictly Proper Scoring Rules, Prediction, and Estimation"
  authors: "Gneiting, Tilmann and Adrian E. Raftery"
  year: 2007
  url: "https://doi.org/10.1198/016214506000001437"
  note: "Reference for Definition 8.3 (NLL as a strictly proper scoring rule) and the broader scoring-rule theory underlying §8."
- type: "paper"
  title: "Concrete Dropout"
  authors: "Gal, Yarin, Jiri Hron, and Alex Kendall"
  year: 2017
  url: "https://proceedings.neurips.cc/paper/2017/hash/84ddfb34126fc3a48ee38d7044e87276-Abstract.html"
  note: "Concrete-dropout extension of MC-dropout (Remark 4.4); makes the dropout rates learnable parameters."
- type: "paper"
  title: "Optimizing Neural Networks with Kronecker-factored Approximate Curvature"
  authors: "Martens, James and Roger Grosse"
  year: 2015
  url: "https://proceedings.mlr.press/v37/martens15.html"
  note: "KFAC curvature approximation, used in §3.4 as a scalable alternative to full Hessian Laplace."
- type: "paper"
  title: "Kernel Methods for Deep Learning"
  authors: "Cho, Youngmin and Lawrence K. Saul"
  year: 2009
  url: "https://proceedings.neurips.cc/paper/2009/hash/5751ec3e9a4feab575962e78e006250d-Abstract.html"
  note: "Arc-cosine kernel for ReLU networks (used in the §9.1 NNGP recursive kernel formula and §9 code experiment)."
- type: "paper"
  title: "Bayesian Inference with Stochastic Volatility Models Using Continuous Superpositions of Non-Gaussian Ornstein-Uhlenbeck Processes"
  authors: "Vollmer, Sebastian J., Konstantinos C. Zygalakis, and Yee Whye Teh"
  year: 2016
  url: "https://www.jmlr.org/papers/v17/15-494.html"
  note: "Constant-stepsize SGLD bias analysis (Proposition 6.4 attribution)."
- type: "paper"
  title: "A Statistical Theory of Cold Posteriors"
  authors: "Aitchison, Laurence"
  year: 2021
  url: "https://openreview.net/forum?id=Rd138pWXMvG"
  note: "One of several proposed explanations for the cold-posterior effect; cited in Remark 8.5 / 9.4."
- type: "book"
  title: "Pattern Recognition and Machine Learning"
  authors: "Bishop, Christopher M."
  year: 2006
  url: "https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/"
  note: "Standard reference for the Laplace approximation (Chapter 4.4) and the moment-matched probit approximation (Chapter 4.5.2)."
```

---

## 12. Notebook Verified Runtime

**Target:** under 60 s end-to-end on a 2020-era CPU laptop, per the strategic-planning doc §8 budget.

**Per-cell budget (estimated, to be confirmed):**

| Cell | Estimated runtime | Dominant cost |
|------|-------------------|---------------|
| Setup (cell 2) | <1 s | Imports, `make_moons`, MLP factory definitions |
| §1 (cell 4) | ~5 s | 5 MLP fits at 200 epochs each |
| §2 (cell 6) | ~12 s | 20 MLP fits + PCA + interpolation loss profile |
| §3 (cell 8) | ~10 s | Full Hessian via autograd ($p = 2241$), Cholesky, 100-sample MC predictive |
| §4 (cell 10) | ~5 s | 1 dropout MLP fit + 100 stochastic forward passes |
| §5 (cell 12) | ~5 s | 10 MLP fits (5 already trained in §1, 5 new) |
| §6 (cell 14) | ~7 s | 1200 SGLD iterations on $b = 32$ minibatches |
| §7 (cell 16) | ~7 s | 1200 SGHMC iterations on $b = 32$ minibatches |
| §8 (cell 18) | ~3 s | Inference only (reuses trained models from §§1, 3, 4, 5, 6, 7) |
| §9 (cell 20) | ~3 s | Random-init MLP samples + closed-form NNGP + small GP regression |
| **Total** | **~58 s** | Within the 60 s budget |

**Verification at implementation time** (Claude Code session, before MDX shipment): re-run the full notebook end-to-end on the local development machine and confirm the actual total runtime is within $1.4\times$ the estimated total — i.e., $\leq 81$ s. If the runtime exceeds this, the dominant offending cells are §§3, 6, 7 (the Hessian-and-MCMC computations); reduce $S$ in §3 or shorten the SGLD/SGHMC chains in §§6–7 before changing other parameters.

**Reproducibility.** All notebook cells use `SEED = 42` (set in the setup cell). Each section's cell prints a verification triple (off/on variance ratio, autocorrelation, etc.) that the MDX implementation cross-checks against the brief's expected ranges.

---

## 13. Build Order (for Claude Code Implementation Session)

The following is the recommended step-by-step build order for the Claude Code session that ships this topic. Each step is a discrete unit of work; pause and report progress before moving to the next.

**Step 1: Verify the notebook end-to-end.** Set up `notebooks/bayesian-neural-networks/.venv/` per CLAUDE.md, install dependencies (`numpy scipy scikit-learn matplotlib torch`), and run all 22 cells end-to-end. Confirm: (a) total wall-clock under 81 s; (b) all printed verification triples land in the expected ranges (off/on variance ratios per §1, §3, §4, §5, §6, §7; autocorrelation comparison per §7; calibration metric ordering per §8); (c) all 9 figures saved to `notebooks/bayesian-neural-networks/figures/`. If any cell fails or any verification is out of range, stop and report.

**Step 2: Write the MDX file.** Create `src/content/topics/bayesian-neural-networks.mdx` with the §2 frontmatter and the §§1–9 section structure. Pull definitions, theorems, propositions, and proof bodies from §4 of this brief verbatim (adapt to `<TheoremBlock>` syntax per CLAUDE.md). The MDX should be self-contained for prose; viz components are imported inline at this step but not yet wired up. Pause at this step's completion for a content-only review.

**Step 3: Extend the shared module.** Add the §5 primitives to `src/components/viz/shared/bayesian-ml.ts`. Implement `deepEnsembleTraining`, `mcDropoutInference`, `laplaceApproxBNN` (full + last-layer + KFAC variants), `sgMCMCBNNTraining` (SGLD + SGHMC), `bnnCalibrationDiagnostic`, and `nngpArcCosineKernel`. Each function should mirror the corresponding notebook cell's logic but in TypeScript. Run the §5-prescribed verification tests against the notebook's printed numerical outputs. Report a three-column table (test name, expected range from brief, observed value).

**Step 4: Copy notebook figures to `public/`.** All 9 PNG figures from the notebook's `figures/` directory go to `public/images/topics/bayesian-neural-networks/`. Match filenames exactly per §9 of this brief.

**Step 5: Build the §1, §3, §5 interactive viz components.** `BNNPredictiveMotivationViz.tsx` (§1 three-panel + noise / $K$ / resample controls), `LaplaceBNNViz.tsx` (§3 three-panel + prior-scale / curvature-reduction / sample-count controls), `DeepEnsembleViz.tsx` (§5 three-panel + $K$ / member-highlight / resample controls). Each is a real React + D3 component per the §4 viz design intent in this brief and the conventions in `docs/plans/formalml-handoff-reference.md` §4 (use `useD3` hook, `useResizeObserver`, CSS-custom-property color tokens via `var(--color-*)`, default export, `client:visible` hydration directive in the MDX). For each: (a) confirm the static-figure fallback (the PNG with `<Figure>`) is in place first; (b) implement the interactive controls per the brief's "Interactive controls (v1 deliverable)" subsections; (c) verify the static and interactive renderings agree on the default-controls values. Pause and report after each component is wired.

**Step 6: Build the §4, §6, §7 interactive viz components.** `MCDropoutBNNViz.tsx` (§4 three-panel + dropout-rate / $T$ / on-at-test controls), `SGLDBNNViz.tsx` (§6 three-panel + step-size / batch-size / schedule controls), `SGHMCBNNViz.tsx` (§7 three-panel + friction / step-size / overlay controls). The §§6–7 components require precomputed JSON payloads for the chain trajectories (see §10 Q4) — write the precompute Python script as `notebooks/bayesian-neural-networks/precompute_sgmcmc_grid.py`, output JSON to both `src/data/sampleData/bayesian-neural-networks/` and `public/sample-data/bayesian-neural-networks/` per the dual-location convention in CLAUDE.md. Pause and report after each component.

**Step 7: Build the §2, §8, §9 interactive viz components.** `LossLandscapeModesViz.tsx` (§2 PCA + interpolation + click-to-pick / $N$-models / interp-grid controls), `BNNCalibrationComparisonViz.tsx` (§8 head-to-head + method-toggles / bin-count / temperature-scaling / cold-posterior controls), `NNGPSidebarViz.tsx` (§9 width-convergence + GP regression + width / activation / training-set-size controls). The §8 component is the most complex of the topic — it consumes precomputed test-set predictive probabilities for all 6 methods (~12 KB total payload) and computes ECE / Brier / NLL in-browser on every interaction; in particular the cold-posterior temperature slider is the §10 Q6 v1 decision point and is recommended ON for at least Laplace and SG-MCMC.

**Step 8: Curriculum graph + cross-references.** Apply the §6 graph deltas to `src/data/curriculum-graph.json` and `src/data/curriculum.ts`. Apply the §7 cross-reference outbound links to the BNN MDX. Apply the §8 inbound MDX edits to `variational-inference.mdx`, `gaussian-processes.mdx`, `stacking-and-predictive-ensembles.mdx`. Pause for review before this step.

**Step 9: Cross-site audit.** Run `pnpm audit:cross-site` from the formalML repo root. The expected output is: BNN's five `formalstatisticsPrereqs` entries log as deferred reciprocals on formalstatistics's side (sister-site PRs to follow). No `self-site` warnings, no missing-target errors. If any entry is missing or malformed, return to step 2 to fix the frontmatter.

**Step 10: Content metrics spreadsheet.** Update `docs/formalml-content-metrics.xlsx` per CLAUDE.md / handoff-reference §13. Add the BNN row to *Topic Detail* (18 columns), recompute *Domain Summary* for `bayesian-ml`, add any *Gap Analysis* entries (BNN should be well above the gap-analysis thresholds — $\geq 7$ sections, $\geq 3$k words, has Code section, has Viz section, all references with URLs).

**Step 11: Local dev preview, walkthrough.** Start `pnpm dev`, navigate to `/topics/bayesian-neural-networks`, walk every section top-to-bottom. Confirm: math renders without `katex-error` spans; figures load (check Network tab for any 404s); each viz component loads via `client:visible` when scrolled into view; all cross-links resolve; all sister-site URLs open in new tabs. Capture screenshots of each section for the PR description.

**Step 12: Build verification.** Run `pnpm build`. Confirm exit 0, no warnings about KaTeX errors, all references rendered as links. Run `pnpm preview` and re-walk the published version.

**Step 13: PR description.** Draft the PR body per the §14 testing-checklist completion summary, with the screenshots from step 11 attached.

---

## 14. Testing Checklist

The following checklist must be fully checked (with deferral annotations where appropriate) before opening the BNN PR.

### Notebook

- [ ] `notebooks/bayesian-neural-networks/.venv/` exists and is set up per CLAUDE.md.
- [ ] All 22 notebook cells execute end-to-end without error.
- [ ] Total notebook runtime is $\leq 81$ s (1.4× the §12 estimated budget).
- [ ] All 9 PNG figures are saved to `notebooks/bayesian-neural-networks/figures/`.
- [ ] §1's variance-ratio off/on $\geq 3\times$.
- [ ] §2's barrier-peak / mode-floor ratio $\geq 3\times$.
- [ ] §3's predictive-std off/on ratio $\geq 2\times$ and Hessian condition number is finite.
- [ ] §4's predictive-std off/on ratio $\geq 2\times$.
- [ ] §5's predictive-std off/on ratio $\geq 4\times$.
- [ ] §6's predictive-std off/on ratio $\geq 3\times$.
- [ ] §7's predictive-std off/on ratio $\geq 3\times$ and SGHMC autocorrelation time is shorter than SGLD's.
- [ ] §8's calibration table prints in roughly the expected ordering: deep ensemble + SGHMC tied for best NLL/ECE, MC-dropout / point-estimate at the bottom.
- [ ] §9's empirical-variance / NNGP-kernel ratios converge to ~1 as width grows.

### Frontmatter & schema

- [ ] All 5 internal `prerequisites` resolve to existing topic slugs in `src/content/topics/`.
- [ ] All 5 `formalstatisticsPrereqs` entries have `relationship` prose $\geq$ 40 chars.
- [ ] All 1–2 `formalcalculusPrereqs` entries have `relationship` prose $\geq$ 40 chars.
- [ ] No `formalmlPrereqs` or `formalmlConnections` entries (would trip `self-site` audit).
- [ ] Every reference in `references` has a `url` field.
- [ ] `type` field on every reference is one of `"book" | "paper" | "course" | "blog" | "video"`.
- [ ] `pnpm audit:cross-site` runs from the repo root with no errors and no `self-site` warnings.

### MDX content

- [ ] All `<TheoremBlock>` usages have a valid `type` (`definition | theorem | lemma | proposition | corollary | proof | remark | example`).
- [ ] All numbered `<TheoremBlock>` definitions / theorems / propositions are sequential within their type.
- [ ] No proofs end without a `$\square$` (auto-rendered by `<TheoremBlock type="proof">`).
- [ ] All `$$ ... $$` display math blocks are on their own line (CLAUDE.md `\begin{aligned}` rule).
- [ ] All inline math `$ ... $` renders correctly under KaTeX.
- [ ] All cross-links to `/topics/...` resolve to existing topic pages.
- [ ] All sister-site URLs (`https://www.formalstatistics.com/topics/...`) point to live pages.
- [ ] Plain-text "(coming soon)" forward pointers (§2.1, §9.4) match exactly two planned-topic mentions; no stale `<a href>` links to unwritten topics.

### Visualization components — structural

- [ ] All 9 viz components live in `src/components/viz/` and use the standard `useResizeObserver` + `useD3` pattern.
- [ ] All viz components use `client:visible` (not `client:load`) in the MDX.
- [ ] All viz components use CSS custom properties (`var(--color-text)` etc.) — no hardcoded colors that break dark mode.
- [ ] All viz components have a static-figure fallback (the notebook PNGs from `public/images/topics/bayesian-neural-networks/`).
- [ ] Static-figure fallbacks render correctly when JavaScript is disabled or before client hydration.
- [ ] Each viz component's numerical output at default-controls values matches the corresponding notebook cell's printed values to within the §5 verification tolerances.
- [ ] No viz component executes a $> 5$ s computation on first paint without a loading-state JSX swap (per CLAUDE.md "Loading-state JSX rule for fetch-based viz").
- [ ] Pre-computed JSON payloads (§§6–7 chain trajectories per §10 Q4) live in *both* `src/data/sampleData/bayesian-neural-networks/` and `public/sample-data/bayesian-neural-networks/` per CLAUDE.md "Sample-data dual-location."

### Visualization components — interactivity (per-section, all v1 deliverables)

- [ ] **§1 `BNNPredictiveMotivationViz`** — Two Moons noise slider works; $K$ slider works; resample-seed button works; all three panels re-render correctly on each control change.
- [ ] **§2 `LossLandscapeModesViz`** — Click-to-pick interaction picks two modes from the PCA scatter and re-renders the loss profile; $N$-models slider works; interpolation-grid slider works; default automatic mode-pair matches the notebook's choice.
- [ ] **§3 `LaplaceBNNViz`** — Prior-scale slider works on log scale; curvature-reduction dropdown switches between full / last-layer / diagonal Fisher and the visible difference matches §3.4's caveat (diagonal Fisher under-estimates off-distribution variance); sample-count slider works.
- [ ] **§4 `MCDropoutBNNViz`** — Dropout-rate slider triggers re-training with loading state; $T$ slider works on the cached trained model; dropout-on-at-test toggle correctly switches between deterministic prediction (mean only) and MC-dropout prediction (mean + variance).
- [ ] **§5 `DeepEnsembleViz`** — $K$ slider works on the cached pool of trained models; member-highlight click works; resample-seed button re-trains all $K$ members with new seeds.
- [ ] **§6 `SGLDBNNViz`** — Step-size slider switches between pre-computed chains for the discrete $\eta$ grid; batch-size slider similarly; schedule toggle correctly displays decaying-stepsize behavior in panel (c)'s trace.
- [ ] **§7 `SGHMCBNNViz`** — Friction slider switches between pre-computed chains; step-size slider works; method-overlay toggle correctly shows / hides the SGLD vs. SGHMC autocorrelation comparison in panel (c).
- [ ] **§8 `BNNCalibrationComparisonViz`** — Method-toggle checkboxes correctly add / remove method lines from panel (a) and bars from panel (b); bin-count slider re-bins the test set and recomputes ECE; temperature-scaling toggle reduces NLL for every method; cold-posterior temperature slider correctly tightens the predictive at $T < 1$ for at least Laplace.
- [ ] **§9 `NNGPSidebarViz`** — Width slider correctly shows convergence: $h = 2$ samples are visibly piecewise-linear, $h = 1024$ matches the closed-form NNGP overlay; activation dropdown switches between ReLU / erf / tanh kernels; training-set-size slider re-fits panel (b)'s NNGP regression.

### Visualization components — responsive layout

- [ ] All 9 components render correctly at desktop widths (≥640 px).
- [ ] All 9 components render correctly at mobile widths (<640 px) — multi-panel layouts stack vertically per the handoff-reference §4 mobile-breakpoint convention.
- [ ] No component overflows its parent container at any width.
- [ ] All sliders / dropdowns / checkboxes are keyboard-accessible.

### Curriculum & cross-references

- [ ] `src/data/curriculum-graph.json` has the BNN node with `domain: "bayesian-ml"`, `status: "published"`.
- [ ] `src/data/curriculum-graph.json` has all 5 prereq edges.
- [ ] `src/data/curriculum.ts` removes "Bayesian Neural Networks" from `bayesian-ml.planned` and adds to `bayesian-ml.published`.
- [ ] `variational-inference.mdx` has new `connections` entry pointing to BNN.
- [ ] `gaussian-processes.mdx` has new `connections` entry pointing to BNN.
- [ ] `stacking-and-predictive-ensembles.mdx` has new `connections` entry pointing to BNN.

### Build & deploy

- [ ] `pnpm dev` serves the topic at `/topics/bayesian-neural-networks` without warnings.
- [ ] `pnpm build` exits 0.
- [ ] No `katex-error` spans in the rendered HTML output (grep the build output).
- [ ] `pnpm preview` (production build) renders the topic identically to dev.
- [ ] First-paint of the topic page on a 4G-throttled connection is $\leq 3$ s.  *(deferred to deploy)*
- [ ] All viz components hydrate when scrolled into view; no in-browser errors during walkthrough.

### Content metrics

- [ ] BNN row added to `docs/formalml-content-metrics.xlsx` *Topic Detail* sheet (18 columns).
- [ ] BNN domain row updated in *Domain Summary* sheet.
- [ ] No *Gap Analysis* entries for BNN (topic should clear all thresholds).
- [ ] Refs Total = Refs w/ URL; Refs Missing URL = 0.

### Sister-site reciprocity (post-shipment follow-up)

- [ ] formalstatistics PR opened adding `formalmlConnections` reciprocals to `bayesian-foundations-and-prior-selection`, `central-limit-theorem`, `modes-of-convergence`, `multivariate-distributions`, `hierarchical-bayes-and-partial-pooling`.
- [ ] After sister-site PR merges, `pnpm audit:cross-site` from formalML reports no deferred-reciprocal entries for BNN's slugs.

---

*End of handoff brief.*
