# formalML — Density-Ratio Estimation Topic — Handoff Brief

> **Topic:** Density-Ratio Estimation
> **Slug:** `density-ratio-estimation`
> **Track:** T3 — Unsupervised & Generative
> **Difficulty:** advanced
> **Code language:** Python + NumPy / SciPy / scikit-learn (default). PyTorch is allowed for §8 only (neural DRE and GAN-as-DRE); all other sections stay in the NumPy / SciPy default.
> **Runtime budget:** end-to-end notebook under 60 s on a 2020-era laptop, CPU-only
> **Mandatory cross-site prereq:** `formalstatistics/kernel-density-estimation` (Topic 30)
> **Companion notebook:** `notebooks/density-ratio-estimation/01_density_ratio_estimation.ipynb`

This brief is the implementation spec for formalML's `density-ratio-estimation` topic. It is paired with the source-of-truth Jupyter notebook at `notebooks/density-ratio-estimation/01_density_ratio_estimation.ipynb`. The brief contains structured prose for every section (math definitions, theorem statements, full proofs, motivating examples), viz design intent at the component level (what each interactive figure shows, what the reader manipulates, what they should learn), code-experiment design (what runs and why), cross-site prerequisite metadata, and the Chicago-17 reference list with verified URLs. The notebook contains the same math content as markdown cells plus the matplotlib-figure code that produces the static analogues of the interactive vizes.

The topic's mathematical center of gravity is **direct density-ratio estimation as a single Bregman-divergence loss family whose population minimizer is $r(x) = p(x)/q(x)$**. Four classical estimator families — KMM (Huang, Smola, Gretton, Borgwardt, Schölkopf 2007), KLIEP (Sugiyama, Suzuki, Nakajima, Kashima, von Bünau, Kawanabe 2008), LSIF/uLSIF (Kanamori, Hido, Sugiyama 2009), and probabilistic classification (Menon and Ong 2016) — fall out of choosing the convex generator $\phi$ in that one framework. The Nguyen–Wainwright–Jordan (2010) variational lower bound on $f$-divergences subsumes the four and lifts to neural witness functions and the $f$-GAN family (Nowozin, Cseke, Tomioka 2016). Downstream applications close the loop: covariate-shift correction (Shimodaira 2000), weighted conformal prediction (Tibshirani, Barber, Candès, Ramdas 2019), and MMD-based two-sample testing (Gretton, Borgwardt, Rasch, Schölkopf, Smola 2012).

This topic closes T3 Unsupervised & Generative as the direct-density-ratio counterpart to `formalstatistics/kernel-density-estimation` (Topic 30): rather than estimating $p$ and $q$ separately and dividing, the §4–§7 estimators target the ratio $r$ as a first-class object. The natural downstream bridge is to `conformal-prediction` (T4), via the §10 weighted-exchangeability extension; the §11 MMD section connects back to the kernel-methods machinery from `kernel-regression` and `gaussian-processes`. The §8 PyTorch section is the only place in the topic that uses GPU-style ML tooling; the §8 import block is local to that section so the rest of the notebook stays NumPy-clean.

---

## Section outline

The 13 H2 sections, each with its single-line summary:

1. **Motivation: why estimate $p/q$ directly** — the plug-in KDE-divided-by-KDE pathology; three downstream uses in one breath (covariate shift, two-sample testing, mutual information); what "direct" buys us; roadmap.
2. **The density-ratio object and the importance-weighting identity** — fix $r$ as a first-class object; prove the IW identity $\mathbb{E}_p[f] = \mathbb{E}_q[r\,f]$ with a Radon–Nikodym detour; absolute continuity, the support condition, chi-squared variance bound; numerical MC verification.
3. **The Bregman-divergence loss family for DRE** — Sugiyama–Suzuki–Kanamori (2012) master identity that turns any convex $\phi$ into a sample-computable DRE objective; specializations recover LSIF (squared loss) and KLIEP (KL loss); logistic-loss preview for §7. Text-only with a sanity-check cell.
4. **KMM — kernel mean matching** — the reweighted moment-matching objective in an RKHS, the canonical QP, KKT conditions; normalization and capping constraints; numerical demonstration on the shift-Gaussian toy.
5. **KLIEP — Kullback–Leibler importance estimation procedure** — derive the constrained KL objective; convex-program structure with non-negativity; projected-gradient ascent (Sugiyama et al. 2008, Algorithm 1) and KL cross-validation for bandwidth selection.
6. **LSIF and uLSIF — least-squares importance fitting** — squared-loss derivation with the analytic closed form; uLSIF drops non-negativity for the killer feature, **analytic** leave-one-out CV via Sherman-Morrison (Kanamori, Hido, Sugiyama 2009, §3.5); head-to-head with KLIEP on accuracy and runtime.
7. **Probabilistic classification as DRE** — pool labelled data, fit any well-calibrated classifier, exponentiate the logit; Bayes-rule identity recovers the ratio; calibration matters (proper scoring rules from Gneiting and Raftery 2007); head-to-head logistic-regression-DRE vs uLSIF on the same Gaussian basis.
8. **The $f$-divergence variational view and neural DRE** — Nguyen–Wainwright–Jordan (2010) variational lower bound on $f$-divergences with full proof; LSIF and KLIEP recovered as $\phi$-choices; neural-MLP witness trained by SGD on the NWJ-KL objective (PyTorch); GAN-as-implicit-DRE observation, with an inner-D discriminator demo verifying $D = \sigma(\log r)$.
9. **Covariate-shift correction** — Shimodaira (2000) IW-ERM identity and the consistency theorem under misspecification; variance inflation via the chi-squared divergence; effective sample size; misspecified linear-regression demo on the running toy.
10. **Conformal prediction under shift** — Tibshirani–Barber–Candès–Ramdas (2019) weighted exchangeability and the algorithm; coverage validity as a function of DRE accuracy; empirical coverage on the §9 shifted toy with vs without weighting.
11. **MMD as a special-case kernel ratio** — MMD as $\|\mu_p - \mu_q\|_\mathcal{H}$ in the kernel-mean-embedding form (Gretton et al. 2012); permutation-based two-sample test with finite-sample validity; discriminative-power vs DRE-smoothness trade-off (Sutherland et al. 2017); permutation test demo with empirical power curve.
12. **Practical considerations, diagnostics, and pitfalls** — bandwidth and basis selection across the kernel-basis family; $n_{\text{eff}}$ as the runtime diagnostic; curse-of-dimensionality demo for uLSIF on isotropic shift Gaussians; when to prefer classification-DRE over kernel-DRE.
13. **Connections, limits, and forward pointers** — cross-site prereq frontmatter (`kernel-density-estimation` mandatory; `maximum-likelihood-estimation`, `convex-optimization`/`lagrangian-duality`, and `fenchel-conjugate` as verify-candidates); within-formalML siblings (`kernel-regression`, `gaussian-processes`, `normalizing-flows`, `conformal-prediction`, `clustering`); honest limits; open problems and forward research pointers.

The notebook uses the same 13-section spine with the same H3 subsections, so a reader who has the brief open alongside the notebook can navigate both in lockstep.

---

## §1. Motivation: why estimate $p/q$ directly

The density-ratio function $r(x) = p(x) / q(x)$ pulls more weight than any single object in modern unsupervised and generative ML. It tells us how to reweight samples from one distribution to look like samples from another (covariate-shift correction); it is the optimal witness for distinguishing two distributions (two-sample testing and GAN discriminators); it is the integrand of the Kullback–Leibler divergence (mutual-information estimation and variational inference). Estimating $r$ should be easy — we have samples from both $p$ and $q$, and the ratio is one scalar function. Yet the most obvious recipe — fit a density estimator to each sample, then divide — is so badly behaved that an entire family of *direct* density-ratio estimators grew up to avoid it. This section explains what goes wrong with divide-then-estimate, where the direct approach pays off, and why we built the next twelve sections around it.

### §1.1 The plug-in pathology

The plug-in recipe writes itself: estimate $\hat p$ from the $p$-samples, estimate $\hat q$ from the $q$-samples, declare $\hat r(x) = \hat p(x) / \hat q(x)$, and call it a day. The trouble is concentrated in the denominator. Wherever $q$ has little mass — but $p$ might have a lot — $\hat q(x)$ is a small noisy positive number, and dividing by it explodes the noise in $\hat r$. The picture is starkest in the *tails of $q$ that overlap the bulk of $p$*: the true ratio is large there (that's exactly the region where reweighting matters most), but the plug-in's variance is largest there too, so the plug-in tells us nothing reliable about the very region we care about.

We can see this numerically on a setup with a closed-form ratio. Take $p = \mathcal{N}(0, 1)$ and $q = \mathcal{N}(1, 1)$. The ratio admits a one-line algebra:
$$
r(x) \;=\; \frac{\varphi(x)}{\varphi(x - 1)} \;=\; \exp\!\left(\frac{1}{2} - x\right),
$$
where $\varphi$ is the standard normal density. The ratio grows exponentially as $x \to -\infty$ — at $x = -3$ it is $e^{3.5} \approx 33$ — exactly where $\hat q$ from a finite sample is most fragile. The §1.1 figure shows the two KDEs and the underlying samples alongside the ratio comparison: the plug-in tracks $r$ inside the bulk and falls apart in the left tail, with log-MSE several orders of magnitude worse there than in the centre.

**Viz: plug-in KDE-ratio blowup (1-D).** *On the site, this lands as **two separate interactive figures** — splitting the notebook's single 2-panel matplotlib output into one "densities" widget and one "ratio" widget for clearer focus.*

*Figure 1 — densities.* (Shows.) KDE estimates of $p$ and $q$ overlaid on the true densities with sample tick-marks at the bottom of the axis. (Reader controls.) Sample size $n_p = n_q \in [50, 1000]$ (slider), shift $\mu_q \in [0.5, 3]$, KDE bandwidth multiplier $\in [0.3, 3]$. (Reader takeaway.) The KDEs match the true densities in the bulk and have visible noise in the tails — the source of the ratio blowup that the next widget visualizes.

*Figure 2 — ratio.* (Shows.) The plug-in ratio $\hat p / \hat q$ vs the closed-form $e^{1/2 - x}$ on a log $y$-axis over $x \in [-4, 5]$. (Reader controls.) Same sliders as Figure 1, kept in sync. (Reader takeaway.) The plug-in's failure is structural, not a bandwidth-tuning issue. Throwing more data at it shrinks the blowup region but doesn't remove the asymmetric over- and undershoot in the tails. This is the canonical motivation for the direct-DRE methods of §4–§7.

**Code experiment.**
- Sample $n_p = n_q = 200$ from $p$ and $q$ using `sample_p`/`sample_q` (defined in the preamble).
- Fit `sklearn.neighbors.KernelDensity` with the Silverman bandwidth on each sample.
- Evaluate $\hat r = \exp(\log \hat p - \log \hat q)$ on a 600-point grid; compare to `true_ratio(x)`.
- Validate: log-MSE in the bulk region $x \in (-1, 3)$ is on the order of $10^{-2}$, while log-MSE in the left tail $x < -2$ is one to two orders of magnitude larger.

### §1.2 Three applications in one breath

The reason a community of estimators evolved around $r$ rather than $(p, q)$ is that several questions different ML papers care about turn out to be the same question dressed differently — namely, "what is $p(x)/q(x)$?"

*Covariate shift.* Train and test draws have the same conditional $p(y \mid x)$ but different marginals: $p_{\text{train}}(x) \ne p_{\text{test}}(x)$. Shimodaira (2000) showed that the risk-minimization estimator on training data is asymptotically optimal for the test distribution if we reweight each training loss by $r(x) = p_{\text{test}}(x) / p_{\text{train}}(x)$. The covariate-shift problem is a density-ratio-estimation problem in disguise. We come back to it in §9.

*Two-sample testing.* Asking "are these two samples from the same distribution?" is asking whether $r \equiv 1$. The maximum mean discrepancy of Gretton, Borgwardt, Rasch, Schölkopf, and Smola (2012) and the classifier-as-statistic constructions of Lopez-Paz and Oquab (2017) both sit on this observation. We come back to it in §11.

*Mutual information and KL.* The KL divergence is $\mathrm{KL}(p \,\|\, q) = \mathbb{E}_p[\log r]$, and mutual information is the KL between the joint and the product of marginals. Estimating either reduces to estimating $\log r$ from samples — exactly what the Nguyen–Wainwright–Jordan (2010) variational lower bound and MINE (Belghazi et al. 2018) do. We come back to this view in §8.

### §1.3 What "direct" buys

There is a single recurring intuition behind direct DRE: estimating two unknown functions and dividing them solves two hard problems to answer one easy question. The two densities $p$ and $q$ are infinite-dimensional, and density estimation suffers a curse of dimensionality with rate $n^{-2/(4+d)}$ for kernel methods at second-order smoothness (Stone 1982). The ratio $r$ is also a function on $\mathbb{R}^d$, but in many practical settings $r$ is much smoother than $p$ or $q$ individually — they share the same fine structure, and the ratio cancels it out. Direct DRE attacks $r$ in a single optimization that is, in a precise sense the Bregman-divergence framework of §3 will make clear, the *only* loss whose population minimizer is $r$. The classic estimators all fall out of choosing different convex $\phi$'s in that one framework.

The other thing "direct" buys is selection by cross-validation against a real loss. We cannot cross-validate a plug-in ratio because we have no loss whose minimum is at the truth; the KDE bandwidth that minimizes a density loss is not the bandwidth that minimizes the ratio's MSE. KLIEP (§5) lets us cross-validate against its own KL objective; uLSIF (§6) gives us *analytic* leave-one-out — a closed form, no refitting. This is the practical reason uLSIF became the default in industry.

### §1.4 Roadmap

§2 fixes the object and proves the importance-weighting identity that every downstream application depends on. §3 wraps the four estimator families into a single Bregman-divergence loss family, so subsequent sections feel like specializations rather than four unrelated papers. §4–§7 are those four specializations: KMM (kernel mean matching), KLIEP, LSIF / uLSIF, and probabilistic classification. §8 takes the variational $f$-divergence view, which subsumes the previous four and lifts to neural witness functions and the GAN family. §9 turns the estimators loose on the canonical downstream application, covariate-shift correction; §10 extends that to conformal prediction under shift via weighted exchangeability. §11 closes the loop with kernel methods by showing MMD as a special-case ratio statistic. §12 collects the practical questions — bandwidth, regularization, effective sample size, where DRE breaks. §13 maps connections to the sister sites and the rest of formalML.


## §2. The density-ratio object and the importance-weighting identity

§2 fixes the object we'll spend the rest of the topic estimating, and proves the one identity every downstream section depends on. The identity is a single line of measure-theoretic algebra, but it's load-bearing: covariate-shift correction (§9), two-sample testing (§11), and the variational $f$-divergence view (§8) all dissolve into special cases of it. We work it carefully here so that later sections can call on it without ceremony.

### §2.1 Setup

We work with two probability distributions $p$ and $q$ on a measurable space $\mathcal{X} \subseteq \mathbb{R}^d$, both with densities (with respect to Lebesgue measure unless we flag otherwise). We observe two iid samples:
$$
X_1^p, \ldots, X_{n_p}^p \;\stackrel{\text{iid}}{\sim}\; p, \qquad X_1^q, \ldots, X_{n_q}^q \;\stackrel{\text{iid}}{\sim}\; q,
$$
and our goal is to recover the *density ratio*
$$
r(x) \;:=\; \frac{p(x)}{q(x)}, \qquad x \in \mathcal{X},
$$
as a function $r: \mathcal{X} \to \mathbb{R}_{\geq 0}$. The asymmetry of the role labels is deliberate: $p$ is the *target* distribution we'd like to take expectations under, and $q$ is the *source* distribution we have abundant samples from. In covariate shift (§9) $p$ is the test marginal and $q$ is the train marginal; in two-sample testing (§11) the role labels collapse and we only care whether $r \equiv 1$.

Two notational alternates worth flagging because they appear in the literature. (i) Some authors estimate $w(x) = q(x)/p(x)$, the reciprocal, when their downstream use is reweighting $p$-samples to look like $q$-samples; the two are interchangeable up to a sign flip in the log. (ii) When the discussion is purely measure-theoretic, $r$ is written as the Radon–Nikodym derivative $\mathrm{d}p / \mathrm{d}q$ and the integrals below become $\int \cdot \, \mathrm{d}q$. We stick with the density-ratio notation throughout because every estimator in §3–§8 works with densities.

### §2.2 The importance-weighting identity

This is the workhorse.

**Detour: what $p \ll q$ actually asserts.** Before stating the identity, we unpack the absolute-continuity hypothesis, because the entire proof leans on it. Consider the probability measures $p$ and $q$ as set functions on a $\sigma$-algebra $\mathcal{F}$ of subsets of $\mathcal{X}$ (the Borel $\sigma$-algebra in everything that follows). We say $p$ is *absolutely continuous with respect to* $q$, written $p \ll q$, if every $\mathcal{F}$-measurable set $A$ with $q(A) = 0$ also has $p(A) = 0$ — in words, $q$ vanishing on a region forces $p$ to vanish there too. The **Radon–Nikodym theorem** (Royden 1988, Theorem 11.23) then promises that under $p \ll q$ there exists a $q$-almost-everywhere unique non-negative $\mathcal{F}$-measurable function $r: \mathcal{X} \to \mathbb{R}_{\geq 0}$ — the *Radon–Nikodym derivative*, written $r = \mathrm{d}p / \mathrm{d}q$ — such that
$$
p(A) \;=\; \int_A r(x) \, q(\mathrm{d}x) \qquad \text{for every } A \in \mathcal{F}.
$$
For two densities with respect to a common dominating measure (Lebesgue, throughout this topic), this abstract derivative is just the pointwise ratio: $r(x) = p(x)/q(x)$ wherever $q(x) > 0$, extended by zero wherever $q(x) = 0$. So the Radon–Nikodym derivative and the density ratio agree as functions on $\mathcal{X}$. We need the abstract version because the proof of the identity passes through the Radon–Nikodym integral characterization, not through pointwise quotients — without it, the algebra in Move 3 would be merely formal manipulation of "$p/q$" on the set $\{q = 0\}$ where the pointwise expression is undefined.

**Theorem (Importance-Weighting Identity).** *Let $p$ and $q$ be probability densities on $\mathcal{X}$ with $p \ll q$, and let $r = \mathrm{d}p / \mathrm{d}q$. Then for any measurable $f: \mathcal{X} \to \mathbb{R}$ with $\mathbb{E}_p|f(X)| < \infty$,*
$$
\boxed{\;\mathbb{E}_p[f(X)] \;=\; \mathbb{E}_q[r(X)\, f(X)].\;}
$$

*Proof.* Four moves.

**Move 1 — write the $p$-expectation as an integral against $p$.** By definition,
$$
\mathbb{E}_p[f(X)] \;=\; \int_{\mathcal{X}} f(x)\, p(\mathrm{d}x).
$$

**Move 2 — invoke Radon–Nikodym to rewrite the measure $p$ as an $r$-weighted version of $q$.** The detour above tells us that for every $\mathcal{F}$-measurable set $A$, $p(A) = \int_A r\, \mathrm{d}q$. The standard approximation argument (first indicators, then simple functions, then non-negative measurable functions by monotone convergence, then arbitrary measurable $f$ by splitting $f = f^+ - f^-$ and applying dominated convergence under $\mathbb{E}_p|f| < \infty$) lifts this from sets to integrals:
$$
\int_{\mathcal{X}} f(x)\, p(\mathrm{d}x) \;=\; \int_{\mathcal{X}} f(x)\, r(x)\, q(\mathrm{d}x).
$$

**Move 3 — handle the null set $\{q = 0\}$ explicitly.** The integrand $f(x)\, r(x)$ on the right-hand side is unambiguously defined on $\{q > 0\}$ and is irrelevant on $\{q = 0\}$ since $q(\{q = 0\}) = 0$ kills its contribution. The convention $r := 0$ on $\{q = 0\}$ (from the detour) is the standard choice; any other convention agrees $q$-almost-everywhere, and integrals don't care.

**Move 4 — recognize the $q$-expectation.** The right-hand side of Move 2 is, by definition, $\mathbb{E}_q[r(X)\, f(X)]$. Combining Moves 1–4 gives the identity. Finiteness of the right-hand side follows from $\mathbb{E}_q|r\, f| = \mathbb{E}_p|f| < \infty$, which is the same chain in reverse. $\blacksquare$

The identity has two readings. *Statistical:* if we have $q$-samples, we can estimate $p$-expectations by reweighting. *Geometric:* the ratio $r$ is the unique deformation that pulls $q$ onto $p$. Both readings will reappear.

**Corollary (Importance-Sampling Estimator).** *With $X_1^q, \ldots, X_{n_q}^q \stackrel{\text{iid}}{\sim} q$ and $r$ the true ratio, the estimator*
$$
\hat \mu^{\mathrm{IS}} \;:=\; \frac{1}{n_q} \sum_{i=1}^{n_q} r(X_i^q)\, f(X_i^q)
$$
*is unbiased for $\mathbb{E}_p[f(X)]$ and has variance $n_q^{-1} \mathrm{Var}_q(r(X) f(X))$.*

*Proof.* Unbiasedness is the identity above applied inside the expectation of each summand. The variance formula is the iid variance of a sample mean. $\blacksquare$

### §2.3 Absolute continuity and the support condition

The identity requires $p \ll q$. In density terms this is the *support condition*: $\operatorname{supp}(p) \subseteq \operatorname{supp}(q)$ — every region where $p$ puts mass must also be a region where $q$ puts mass. If the support condition fails, $r$ is undefined (or infinite) on a set of positive $p$-measure, and no finite-sample estimator can hope to recover it there: the $q$-samples never visit that region. Direct DRE methods (§4–§7) silently inherit this requirement; classification-based DRE (§7) makes it operationally visible because the classifier fails to separate the classes only where both supports overlap.

Absolute continuity is necessary but not sufficient for $\hat \mu^{\mathrm{IS}}$ to be *useful*. Even when $r$ is everywhere finite, the IS estimator's variance can be infinite. Setting $f \equiv 1$ in the corollary's variance formula gives
$$
n_q \cdot \mathrm{Var}(\hat \mu^{\mathrm{IS}}) \;=\; \mathrm{Var}_q(r(X)) \;=\; \mathbb{E}_q[r(X)^2] - 1.
$$
The quantity $\mathbb{E}_q[r(X)^2] - 1$ is exactly the **chi-squared divergence** $\chi^2(p \,\|\, q)$. So the IS estimator has finite variance for the simplest possible $f$ (the constant one) if and only if $\chi^2(p \,\|\, q) < \infty$. For our running shift-Gaussian toy with $p = \mathcal{N}(0, 1)$ and $q = \mathcal{N}(\mu_q, 1)$, a one-page calculation (sketched in the §2.4 cell) gives
$$
\chi^2(p \,\|\, q) \;=\; e^{\mu_q^2} - 1,
$$
which is gentle for $\mu_q = 1$ ($\chi^2 \approx 1.7$) but ferocious for $\mu_q = 3$ ($\chi^2 \approx 8102$). The IS estimator's variance at $\mu_q = 3$ is roughly $5000\times$ that at $\mu_q = 1$ — even though the true ratio exists and is finite everywhere.

The runtime diagnostic that picks up this pathology in practice is the **effective sample size**
$$
n_{\text{eff}} \;:=\; \frac{\left( \sum_{i=1}^{n_q} w_i \right)^2}{\sum_{i=1}^{n_q} w_i^2}, \qquad w_i = r(X_i^q),
$$
which ranges from $1$ (one weight dominates) to $n_q$ (all weights equal). It estimates how many "effectively-independent" $p$-samples our weighted $q$-sample is worth. When $n_{\text{eff}} \ll n_q$, the IS estimator is unreliable regardless of what $n_q$ says.

### §2.4 Numerical hook — verifying the identity and watching it strain

The §2.4 cell verifies the identity numerically with the closed-form ratio from §1, then re-runs the experiment under a larger shift to watch the variance and $n_{\text{eff}}$ degrade. Target: $\mathbb{E}_p[X^2]$, which equals $1$ in closed form for $p = \mathcal{N}(0, 1)$. We compare two estimators: the direct sample mean from $p$-samples, and the importance-weighted sample mean from $q$-samples using the true ratio. We also derive the closed-form $\chi^2(p \,\|\, q)$ for two equal-variance Gaussians inline in the cell.

**Viz: importance-weighting identity at two shifts.**
- *Shows:* two-panel figure. (Left) Monte Carlo estimates of $\mathbb{E}_p[X^2]$ vs $n$, on a log-$n$ axis, comparing the direct estimator from $p$-samples against the IS estimator from $q$-samples weighted by the true ratio; both converge to 1, but the IS estimator's $95\%$ band is much wider. (Right) the same comparison at $\mu_q = 3$ instead of $\mu_q = 1$; the IS band is now enormous, and the IS curve takes orders of magnitude more samples to stabilize.
- *Reader controls (interactive site version):* $\mu_q \in [0.5, 4]$ (slider) drives the shift; the panels recompute and the $\chi^2(p\,\|\,q)$ and $n_{\text{eff}}$ readouts update live.
- *What the reader should learn:* the identity is mathematically clean, but its finite-sample usefulness is governed by $\chi^2(p\,\|\,q)$, not by sample size. This is the conceptual handhold for §6's regularization choices and §9's effective-sample-size diagnostics.

**Code experiment.**
- Use the preamble's `sample_p`, `sample_q`, `true_ratio` with $\mu_q \in \{1, 3\}$.
- Set $f(x) = x^2$ so the target is $\mathbb{E}_p[X^2] = 1$ in closed form.
- For each $n \in \{10^2, 10^{2.5}, \ldots, 10^4\}$ and each of $200$ Monte Carlo replicates, compute the direct estimator and the IS estimator; plot mean ± $2\times$ standard error.
- Print the closed-form $\chi^2$ at both shifts (formula derived in the cell) and the empirical $n_{\text{eff}}$ at $n = 10^3$.
- Validate: at $\mu_q = 1$, IS std-error is within $3\times$ the direct std-error at $n = 10^3$; at $\mu_q = 3$, IS std-error is at least $30\times$ the direct std-error at $n = 10^3$.

---

## §3. The Bregman-divergence loss family for DRE

§3 is the section that turns the next four sections from "four unrelated 2008–2012 papers" into "four specializations of one framework." Sugiyama, Suzuki, and Kanamori (2012, Chapter 5) showed that every classical direct-DRE estimator — KMM, KLIEP, LSIF, uLSIF, and probabilistic classification — corresponds to a choice of strictly convex generator $\phi$ in a single Bregman-divergence loss. The framework promises three things at once: a unified derivation, a unified convergence theory (each loss is convex, with the true ratio as its unique population minimizer), and an unambiguous cross-validation criterion (each loss can be estimated unbiasedly from samples without knowing $r$). This section proves the master identity and shows how the LSIF and KLIEP objectives drop out of it; §3.4 previews the logistic-loss specialization that §7 will unpack in full.

This section has no figure by design. The numerical hook is a small sanity-check cell at the end that verifies, on the running toy, that both the LSIF and KLIEP empirical objectives bottom out at the true ratio.

### §3.1 Bregman divergences and the master identity

A **Bregman divergence** generated by a strictly convex differentiable function $\phi: \mathbb{R}_{>0} \to \mathbb{R}$ is the function $\mathrm{BR}_\phi : \mathbb{R}_{>0} \times \mathbb{R}_{>0} \to \mathbb{R}_{\geq 0}$ defined by
$$
\mathrm{BR}_\phi(u, v) \;:=\; \phi(u) \;-\; \phi(v) \;-\; \phi'(v)\,(u - v).
$$
Geometrically, $\phi(v) + \phi'(v)(u - v)$ is the tangent line to $\phi$ at $v$ evaluated at $u$, so $\mathrm{BR}_\phi(u, v)$ measures how much $\phi(u)$ exceeds that tangent at $u$. Convexity gives $\mathrm{BR}_\phi(u, v) \geq 0$, with equality iff $u = v$ (strict convexity is what rules out flat segments). Three examples set the pattern:

- $\phi(t) = \tfrac{1}{2} t^2$: $\mathrm{BR}_\phi(u, v) = \tfrac{1}{2}(u - v)^2$ — squared loss.
- $\phi(t) = t \log t - t$: $\mathrm{BR}_\phi(u, v) = u \log(u/v) - u + v$ — KL between two positive scalars.
- $\phi(t) = -\log t$: $\mathrm{BR}_\phi(u, v) = u/v - \log(u/v) - 1$ — Itakura–Saito divergence.

To lift the pointwise divergence to a discrepancy between *functions* $r$ and $g$, integrate against $q$:
$$
D_\phi(r, g) \;:=\; \mathbb{E}_q\bigl[\mathrm{BR}_\phi\bigl(r(X), g(X)\bigr)\bigr] \;=\; \mathbb{E}_q\bigl[\phi(r) - \phi(g) - \phi'(g)\,(r - g)\bigr].
$$
This $D_\phi$ is the *target* functional. Its minimum over $g$ is $0$, attained at $g = r$. But we can't compute it directly because $r$ appears inside $\phi(r)$ and $\phi'(g) r$, and $r$ is unknown. The master identity converts $D_\phi$ into a computable objective.

**Theorem (Master DRE Identity; Sugiyama–Suzuki–Kanamori 2012, Theorem 5.1).** *Let $p \ll q$ and $r = p/q$. For any measurable $g: \mathcal{X} \to \mathbb{R}_{>0}$,*
$$
D_\phi(r, g) \;=\; \underbrace{\mathbb{E}_q\bigl[\phi(r(X))\bigr]}_{\text{constant in } g} \;+\; J_\phi(g),
$$
*where*
$$
\boxed{\;J_\phi(g) \;:=\; \mathbb{E}_q\bigl[\phi'(g(X))\, g(X) - \phi(g(X))\bigr] \;-\; \mathbb{E}_p\bigl[\phi'(g(X))\bigr].\;}
$$

*Proof.* Expand the definition of $D_\phi$:
$$
D_\phi(r, g) \;=\; \mathbb{E}_q[\phi(r)] \;-\; \mathbb{E}_q[\phi(g)] \;-\; \mathbb{E}_q[\phi'(g)\, (r - g)].
$$
Distribute the third term: $\mathbb{E}_q[\phi'(g)\, (r - g)] = \mathbb{E}_q[\phi'(g)\, r] - \mathbb{E}_q[\phi'(g)\, g]$. Substitute back:
$$
D_\phi(r, g) \;=\; \mathbb{E}_q[\phi(r)] \;-\; \mathbb{E}_q[\phi(g)] \;-\; \mathbb{E}_q[\phi'(g)\, r] \;+\; \mathbb{E}_q[\phi'(g)\, g].
$$
The middle term $\mathbb{E}_q[\phi'(g)\, r]$ is exactly an importance-weighted expectation: by §2.2's identity with $f = \phi' \circ g$,
$$
\mathbb{E}_q[\phi'(g(X))\, r(X)] \;=\; \mathbb{E}_p[\phi'(g(X))].
$$
Substitute and collect terms in $g$:
$$
D_\phi(r, g) \;=\; \mathbb{E}_q[\phi(r)] \;+\; \bigl\{ \mathbb{E}_q[\phi'(g)\, g - \phi(g)] \;-\; \mathbb{E}_p[\phi'(g)] \bigr\}.
$$
The first term is constant in $g$; the brace is $J_\phi(g)$ by definition. $\blacksquare$

Two consequences. First, $J_\phi$ is computable from samples — we can replace the two expectations with their empirical analogues $\hat J_\phi(g) = (1/n_q) \sum_i [\phi'(g(X_i^q)) g(X_i^q) - \phi(g(X_i^q))] - (1/n_p) \sum_j \phi'(g(X_j^p))$, and the estimator is unbiased for $J_\phi(g)$. Second, $J_\phi$ inherits convexity from $\phi$ (for a fixed $g$-parametrization that's linear in $g$, the empirical $\hat J_\phi$ is convex in the parameters), so first-order methods find the unique minimizer. The two consequences together are what makes the Bregman framework operational: pick $\phi$ for the downstream use, optimize $\hat J_\phi$, done.

### §3.2 Squared loss recovers LSIF

Choose $\phi(t) = \tfrac{1}{2} t^2$. Then $\phi'(t) = t$, and
$$
\phi'(t)\, t - \phi(t) \;=\; t^2 - \tfrac{1}{2} t^2 \;=\; \tfrac{1}{2} t^2.
$$
The master identity collapses to
$$
J_{\mathrm{LSIF}}(g) \;=\; \tfrac{1}{2}\, \mathbb{E}_q[g(X)^2] \;-\; \mathbb{E}_p[g(X)],
$$
with sample version $\hat J_{\mathrm{LSIF}}(g) = (1/(2 n_q)) \sum_i g(X_i^q)^2 - (1/n_p) \sum_j g(X_j^p)$. Population stationarity in $g$ gives $\delta J_{\mathrm{LSIF}}/\delta g = q g - p = 0$, so $g^* = p/q = r$, as the framework requires.

The LSIF specialization gets a closed form for *linear-in-parameters* models. Parametrize $g(x) = \boldsymbol{\alpha}^\top \boldsymbol{\psi}(x)$ with $\boldsymbol{\psi}: \mathcal{X} \to \mathbb{R}^b$ a fixed basis (we'll typically take $\boldsymbol{\psi}$ to be Gaussian kernels centred at a subsample). Plugging into the sample objective,
$$
\hat J_{\mathrm{LSIF}}(\boldsymbol{\alpha}) \;=\; \tfrac{1}{2}\, \boldsymbol{\alpha}^\top \hat{\mathbf{H}}\, \boldsymbol{\alpha} \;-\; \hat{\mathbf{h}}^\top \boldsymbol{\alpha},
$$
where
$$
\hat{\mathbf{H}} \;=\; \frac{1}{n_q} \sum_{i=1}^{n_q} \boldsymbol{\psi}(X_i^q) \boldsymbol{\psi}(X_i^q)^\top \in \mathbb{R}^{b \times b}, \qquad \hat{\mathbf{h}} \;=\; \frac{1}{n_p} \sum_{j=1}^{n_p} \boldsymbol{\psi}(X_j^p) \in \mathbb{R}^b.
$$
$\hat{\mathbf{H}}$ is positive semi-definite, so this is a convex quadratic. The minimizer is the linear system $\hat{\mathbf{H}}\, \hat{\boldsymbol{\alpha}} = \hat{\mathbf{h}}$, with the regularized version $(\hat{\mathbf{H}} + \lambda \mathbf{I}) \hat{\boldsymbol{\alpha}} = \hat{\mathbf{h}}$ that yields $\hat{\boldsymbol{\alpha}} = (\hat{\mathbf{H}} + \lambda \mathbf{I})^{-1} \hat{\mathbf{h}}$. §6 picks up here.

### §3.3 KL loss recovers KLIEP

Choose $\phi(t) = t \log t - t$. Then $\phi'(t) = \log t$, and
$$
\phi'(t)\, t - \phi(t) \;=\; t \log t - (t \log t - t) \;=\; t.
$$
The master identity gives
$$
J_{\mathrm{KLIEP}}^{\mathrm{unc}}(g) \;=\; \mathbb{E}_q[g(X)] \;-\; \mathbb{E}_p[\log g(X)],
$$
the *unconstrained* KLIEP objective, with sample version $\hat J^{\mathrm{unc}}_{\mathrm{KLIEP}}(g) = (1/n_q) \sum_i g(X_i^q) - (1/n_p) \sum_j \log g(X_j^p)$. Population stationarity: $\delta J / \delta g = q - p/g = 0$, so $g^* = p/q = r$.

The original KLIEP (Sugiyama, Suzuki, Nakajima, Kashima, von Bünau, Kawanabe 2008) presents the *constrained* form,
$$
\max_g \; \mathbb{E}_p[\log g(X)] \quad \text{subject to} \quad \mathbb{E}_q[g(X)] = 1,
$$
which is mathematically equivalent (the equality constraint is exactly the first-order condition that the multiplier in front of $\mathbb{E}_q[g]$ equal $1$). The constrained form has a tighter feasible set and is what Sugiyama et al.'s projected-gradient algorithm operates on; the unconstrained form is what gradient methods see directly. §5 covers both.

### §3.4 Logistic loss recovers probabilistic classification (preview)

The probabilistic-classification approach to DRE proceeds by pooling $p$-samples (label $y = 1$) and $q$-samples (label $y = 0$) and fitting a classifier $\hat \eta(x) = \log \mathbb{P}(y = 1 \mid x) / \mathbb{P}(y = 0 \mid x)$. The Bayes-rule reformulation in §7.2 will give
$$
r(x) \;=\; \frac{n_q}{n_p}\, \exp\!\bigl(\hat \eta(x)\bigr),
$$
and the classifier's training loss — logistic cross-entropy on the pooled labels — falls into the Bregman family with generator
$$
\phi_{\mathrm{LR}}(t) \;=\; t \log t - (1 + t) \log(1 + t) + (1 + t)\log 2,
$$
modulo the normalization conventions catalogued in Menon and Ong (2016). We don't unpack this here because §7 owns the full derivation, the calibration discussion, and the numerical comparison with uLSIF. The point for §3 is that *the framework is complete*: KMM (§4, via a kernelized squared-loss view), LSIF/uLSIF, KLIEP, and probabilistic classification all live inside one Bregman generator $\phi$, and the choice of $\phi$ controls the bias–variance trade-off rather than the algorithm class.

### §3 numerical hook

The small validation cell at the end of §3 picks the closed-form $r(x) = \exp(\tfrac{1}{2} - x)$ from the running toy and a one-parameter perturbation family $g_\alpha(x) = \exp(\alpha (\tfrac{1}{2} - x))$ — at $\alpha = 1$ we have $g = r$ exactly. We compute $\hat J_{\mathrm{LSIF}}(g_\alpha)$ and $\hat J^{\mathrm{unc}}_{\mathrm{KLIEP}}(g_\alpha)$ on a grid of $\alpha$ values from a single MC sample and confirm that both empirical objectives bottom out at $\hat \alpha \approx 1$. No figure: §3 is the section where the framework lands, not where we tune anything.

---

## §4. KMM — kernel mean matching

§4 is the first section in which we estimate something. Kernel mean matching (Huang, Smola, Gretton, Borgwardt, and Schölkopf 2007) was the first widely-used direct DRE method, and it takes a sample-reweighting view of the problem: rather than estimate a function $r(\cdot)$ on all of $\mathcal{X}$, KMM estimates only the $n_q$ numbers $w_1, \ldots, w_{n_q}$ that approximate $\{r(X_i^q)\}_{i=1}^{n_q}$. These weights are the only quantities a downstream importance-weighted estimator (covariate-shift correction in §9, weighted conformal in §10) ever asks for, so producing a function $r(\cdot)$ on the whole space is unnecessary effort. KMM trades that out-of-sample generality for a tighter sample-only optimization with a clean QP structure and a population-level guarantee that the kernel-mean discrepancy is zero exactly at the true ratio.

This section derives KMM from the kernel-mean-embedding viewpoint, writes down the QP explicitly with its KKT conditions, explains the two practical constraints (normalization and capping) and what they buy, and runs the algorithm on the §1 shift-Gaussian toy to verify that the KMM weights reproduce the closed-form ratio at the $q$-samples.

### §4.1 The reweighted moment-matching objective in an RKHS

Let $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ be a positive-definite kernel and let $\mathcal{H}$ be its reproducing kernel Hilbert space. For any probability measure $P$ on $\mathcal{X}$ satisfying $\mathbb{E}_{X \sim P}[\sqrt{k(X, X)}] < \infty$, the **kernel mean embedding** of $P$ is
$$
\mu_P \;:=\; \mathbb{E}_{X \sim P}[k(X, \cdot)] \;\in\; \mathcal{H}.
$$
$\mu_P$ is an element of $\mathcal{H}$; the reproducing property gives $\langle \mu_P, f \rangle_\mathcal{H} = \mathbb{E}_P[f(X)]$ for every $f \in \mathcal{H}$, so $\mu_P$ encodes all moments of $P$ that $\mathcal{H}$ can witness. When $k$ is **characteristic** — which the Gaussian RBF $k(x, y) = \exp(-\|x - y\|^2 / (2 \sigma_k^2))$ is, on $\mathbb{R}^d$ for any $\sigma_k > 0$ (Sriperumbudur et al. 2010) — the map $P \mapsto \mu_P$ is injective: two probability measures agree if and only if their kernel mean embeddings agree.

The KMM construction is to find a non-negative weight function $g: \mathcal{X} \to \mathbb{R}_{\geq 0}$ such that the kernel mean of the $g$-reweighted source distribution matches the kernel mean of the target:
$$
\mu_q^g \;:=\; \mathbb{E}_q[g(X)\, k(X, \cdot)] \;\stackrel{?}{=}\; \mu_p.
$$
By the importance-weighting identity, $\mu_q^g = \mu_p$ holds at $g = r$, since
$$
\mathbb{E}_q[r(X)\, k(X, \cdot)] \;=\; \mathbb{E}_p[k(X, \cdot)] \;=\; \mu_p.
$$
The population KMM objective is the squared $\mathcal{H}$-norm of the gap:
$$
J_{\mathrm{KMM}}(g) \;:=\; \tfrac{1}{2}\, \big\| \mu_q^g - \mu_p \big\|^2_\mathcal{H}.
$$
$J_{\mathrm{KMM}}$ is convex and non-negative, and for a characteristic kernel the population minimizer is $g^\star = r$ (uniquely $q$-a.e.). The injectivity of the kernel-mean embedding does the work: $J_{\mathrm{KMM}}(g) = 0$ implies $\mathbb{E}_q[g(X) k(X, \cdot)] = \mathbb{E}_p[k(X, \cdot)]$, which says $g \cdot q = p$ as measures, hence $g = p/q$ $q$-a.e.

### §4.2 The quadratic program and its KKT conditions

KMM works on the *empirical* version, in which the population kernel means are replaced by sample averages and we estimate a single weight $w_i \geq 0$ for each $q$-sample rather than a function. Let $\tilde w_i := w_i / n_q$ be the implicit reweighting of $q$'s empirical measure; the empirical reweighted kernel mean is $\sum_i \tilde w_i\, k(X_i^q, \cdot)$, and we match it to the empirical $p$ kernel mean $(1/n_p) \sum_j k(X_j^p, \cdot)$. Expanding the squared $\mathcal{H}$-norm via the reproducing property $\langle k(x, \cdot), k(y, \cdot) \rangle_\mathcal{H} = k(x, y)$,
$$
\Big\| \sum_i \tilde w_i\, k(X_i^q, \cdot) - \frac{1}{n_p} \sum_j k(X_j^p, \cdot) \Big\|^2_\mathcal{H}
\;=\; \tilde{\mathbf{w}}^\top \mathbf{K}\, \tilde{\mathbf{w}} \;-\; 2\, \tilde{\mathbf{w}}^\top \tilde{\boldsymbol{\kappa}} \;+\; C,
$$
where $\mathbf{K}_{ii'} = k(X_i^q, X_{i'}^q) \in \mathbb{R}^{n_q \times n_q}$, $\tilde{\boldsymbol{\kappa}}_i = (1/n_p) \sum_j k(X_i^q, X_j^p) \in \mathbb{R}^{n_q}$, and $C = \|\frac{1}{n_p}\sum_j k(X_j^p, \cdot)\|^2_\mathcal{H}$ is constant in $\tilde{\mathbf{w}}$. Rescaling to $\mathbf{w} = n_q\, \tilde{\mathbf{w}}$ and defining $\boldsymbol{\kappa}_i = (n_q / n_p) \sum_j k(X_i^q, X_j^p)$ gives the **KMM quadratic program** in its canonical form:
$$
\boxed{
\begin{aligned}
\min_{\mathbf{w} \in \mathbb{R}^{n_q}} \quad & \tfrac{1}{2}\, \mathbf{w}^\top \mathbf{K}\, \mathbf{w} - \boldsymbol{\kappa}^\top \mathbf{w} \\
\text{subject to} \quad & 0 \le w_i \le B, \quad i = 1, \ldots, n_q, \\
& \Big| \tfrac{1}{n_q} \textstyle\sum_i w_i - 1 \Big| \le \epsilon.
\end{aligned}
}
$$
$\mathbf{K}$ is positive semi-definite (it's a Gram matrix), so the objective is convex; the feasible set is a polytope; the QP has a unique optimum (modulo the kernel of $\mathbf{K}$, which collapses if the kernel is strictly positive definite, as the Gaussian RBF is).

**KKT conditions.** Form the Lagrangian
$$
\mathcal{L}(\mathbf{w}, \mu_+, \mu_-, \boldsymbol{\lambda}_-, \boldsymbol{\lambda}_+) = \tfrac{1}{2}\, \mathbf{w}^\top \mathbf{K} \mathbf{w} - \boldsymbol{\kappa}^\top \mathbf{w} + \mu_+ \Bigl(\tfrac{1}{n_q}\textstyle\sum_i w_i - 1 - \epsilon\Bigr) + \mu_- \Bigl(1 - \epsilon - \tfrac{1}{n_q}\textstyle\sum_i w_i\Bigr) - \boldsymbol{\lambda}_-^\top \mathbf{w} + \boldsymbol{\lambda}_+^\top (\mathbf{w} - B \mathbf{1}),
$$
with $\mu_\pm, \boldsymbol{\lambda}_\pm \geq 0$ and complementary slackness on each inequality. Stationarity gives
$$
\mathbf{K} \mathbf{w} - \boldsymbol{\kappa} + (\mu_+ - \mu_-)\, \mathbf{1}/n_q - \boldsymbol{\lambda}_- + \boldsymbol{\lambda}_+ \;=\; \mathbf{0}.
$$
For an interior optimum (no active bounds, normalization constraint slack), this reduces to $\mathbf{w}^\star = \mathbf{K}^{-1} \boldsymbol{\kappa}$. Pure-interior optima do happen in practice, but the bound constraints typically activate on a handful of indices — those are the $q$-samples where the unconstrained QP wants a negative or pathologically large weight, and KMM clips them to the polytope boundary instead. The active set is what distinguishes KMM from naive least-squares moment matching.

### §4.3 Constraints — why $B$ and $\epsilon$ exist, and what they cost

The two practical constraints play different roles, and confusing them leads to badly tuned KMM runs.

**The capping constraint $w_i \le B$.** Without an upper bound, individual weights can grow arbitrarily large when a $q$-sample sits in a region where $p$ has substantial mass but few other $q$-samples cover. The unbounded QP responds by inflating a single $w_i$ to absorb that mismatch, which produces an estimator whose effective sample size $n_{\text{eff}} = (\sum w_i)^2 / \sum w_i^2$ collapses to a small number. The capping bound $B$ is a hard prior on the maximum allowable weight; Huang et al. recommend $B \in [10, 10^3]$ in practice, calibrated against the maximum the true ratio could plausibly reach in the support overlap. Setting $B$ too small biases the weights downward in the high-$r$ region; setting $B$ too large gives the variance back.

**The normalization constraint $|n_q^{-1} \sum_i w_i - 1| \le \epsilon$.** The true ratio satisfies $\mathbb{E}_q[r(X)] = \mathbb{E}_p[1] = 1$, so a faithful weight vector should average to roughly $1$. The relaxed inequality version with slack $\epsilon$ (rather than a hard equality) accommodates finite-sample fluctuations and avoids over-constraining the QP. Huang et al. recommend $\epsilon = (\sqrt{n_q} - 1)/\sqrt{n_q}$, which goes to zero as $n_q \to \infty$. The constraint isn't strictly necessary for the QP to be well-posed — the squared $\mathcal{H}$-norm objective alone determines $\mathbf{w}$ up to the kernel of $\mathbf{K}$ — but it is necessary for the reweighted empirical measure to be a proper probability distribution, which downstream importance-weighted estimators silently rely on.

**Bandwidth selection.** The Gaussian kernel needs a bandwidth $\sigma_k$. The median heuristic — set $\sigma_k = \mathrm{median}\{ \|X_i - X_j\| : 1 \le i < j \le n_q + n_p \}$ over the pooled sample — is the standard choice and works well across orders of magnitude in dimension. More principled bandwidth selection requires a held-out scoring rule, which KMM doesn't naturally provide (its objective is unobserved at the population level); §6's uLSIF gives us analytic LOO-CV, which is a major reason it overtook KMM in practice.

### §4.4 Numerical demonstration

The §4.4 cell runs KMM on the shift-Gaussian toy ($p = \mathcal{N}(0, 1)$, $q = \mathcal{N}(1, 1)$) and compares the recovered weights $\hat w_i$ against $r(X_i^q) = \exp(\tfrac{1}{2} - X_i^q)$ from the closed form. The kernel bandwidth is set by the median heuristic on the pooled $(p, q)$ sample; the QP is solved by SciPy SLSQP with $B = 1000$ and $\epsilon = (\sqrt{n_q} - 1)/\sqrt{n_q}$.

**Viz: KMM weights vs the true ratio.**
- *Shows:* two-panel figure. (Left) the $q$-samples on a 1-D axis with marker size proportional to the recovered KMM weight, overlaid on the closed-form $r(x)$ curve; the visual story is that high-weight samples cluster in the left tail where $p$ has mass and $q$ doesn't. (Right) scatter of $(r(X_i^q), \hat w_i)$ with the $y = x$ diagonal; points falling on the diagonal mean KMM has recovered the true ratio.
- *Reader controls (interactive site version):* $\mu_q \in [0.5, 2.5]$ (slider), $n_q \in [100, 500]$, bandwidth multiplier $\in [0.3, 3]$ on top of the median heuristic, $B \in [10, 1000]$ (log-scale slider).
- *What the reader should learn:* (i) KMM weights track the true ratio with high correlation when the supports overlap well; (ii) raising $\mu_q$ stretches the support gap, the median-heuristic bandwidth degrades, and the diagonal scatter loosens; (iii) lowering $B$ visibly truncates the upper tail of the recovered weights, biasing the high-$r$ region.

**Code experiment.**
- Sample $n_p = n_q = 300$ from the running toy.
- Compute pairwise distances and set $\sigma_k$ by the median heuristic on the pooled sample.
- Build $\mathbf{K}$ and $\boldsymbol{\kappa}$ via vectorised `cdist`-based Gaussian-kernel evaluation.
- Solve the QP with `scipy.optimize.minimize(method="SLSQP")` using analytic gradient.
- Plot the two panels described above; print Pearson correlation between $\{\hat w_i\}$ and $\{r(X_i^q)\}$, the recovered $\bar w$, the empirical $n_{\text{eff}}$, and how many $w_i$ are clipped at the bounds.
- Validate: Pearson correlation $> 0.9$, $\bar w$ within the normalization slack, fewer than 10% of samples clipped at $B$.

---

## §5. KLIEP — Kullback–Leibler importance estimation procedure

§5 picks up the KL specialization of the Bregman framework from §3.3 and runs it through to a working estimator. KLIEP (Sugiyama, Suzuki, Nakajima, Kashima, von Bünau, and Kawanabe 2008) was historically the first direct DRE method to deliver a principled bandwidth-selection scheme — the KLIEP objective is itself a valid held-out scoring rule, so cross-validation comes for free. That is a meaningful operational advantage over KMM (§4), which has no built-in CV criterion; it's the reason KLIEP and its descendants (uLSIF in §6) eclipsed KMM as the practitioner default.

This section derives the constrained KL objective from the §3.3 unconstrained form, lays out the convex-program structure, presents Sugiyama et al.'s projected-gradient-ascent algorithm explicitly, and verifies bandwidth selection by KL-CV on the running toy. Two figures: a convergence trace at a fixed bandwidth, and a KL-CV curve that picks the bandwidth automatically.

### §5.1 The KL objective and the linear-in-parameters model

Take the §3.3 KL choice $\phi(t) = t \log t - t$. The cleanest derivation of KLIEP doesn't start from $\phi$ — it starts from the question "if I treat $\hat r(x) \cdot q(x)$ as my estimate of $p(x)$, what loss should I minimize?" The natural answer is the KL divergence from $p$ to the estimated density:
$$
\mathrm{KL}\bigl(p \,\big\|\, \hat r \cdot q\bigr) \;=\; \int p(x) \log \frac{p(x)}{\hat r(x)\, q(x)}\, \mathrm{d}x \;=\; \mathbb{E}_p\bigl[\log r(X)\bigr] \;-\; \mathbb{E}_p\bigl[\log \hat r(X)\bigr].
$$
The first term is constant in $\hat r$ — it's just $\mathrm{KL}(p \,\|\, q)$ as far as the optimizer cares. Minimizing the KL is therefore *equivalent to* maximizing $\mathbb{E}_p[\log \hat r(X)]$. The KL framing makes one constraint visible that the unconstrained Bregman form (§3.3) leaves implicit: for $\hat r \cdot q$ to be a proper probability density, we need $\int \hat r(x)\, q(x)\, \mathrm{d}x = \mathbb{E}_q[\hat r(X)] = 1$. The non-negativity $\hat r \geq 0$ is the second standing requirement.

The constrained KLIEP problem is therefore
$$
\boxed{\;\max_{\hat r} \;\; \mathbb{E}_p\bigl[\log \hat r(X)\bigr] \quad \text{subject to} \quad \mathbb{E}_q[\hat r(X)] = 1, \quad \hat r \geq 0. \;}
$$
The constrained and unconstrained KL forms are Lagrangian-dual to each other: the equality constraint's multiplier turns out to equal $1$ at the optimum, and the resulting Lagrangian is exactly the unconstrained $J^{\mathrm{unc}}_{\mathrm{KLIEP}}(g) = \mathbb{E}_q[g] - \mathbb{E}_p[\log g]$ from §3.3.

**Linear-in-parameters model.** Sugiyama et al. (2008) parametrize $\hat r$ as a non-negative linear combination of a fixed kernel basis:
$$
\hat r(x) \;=\; \sum_{\ell = 1}^{b} \alpha_\ell\, \psi_\ell(x), \qquad \boldsymbol{\psi}(x) = \bigl(\psi_1(x), \ldots, \psi_b(x)\bigr)^\top,
$$
with $\boldsymbol{\alpha} \in \mathbb{R}_{\geq 0}^b$ ensuring $\hat r \geq 0$ provided each $\psi_\ell \geq 0$. The standard basis is Gaussian-RBF kernels centred at a random subsample of the $p$-samples:
$$
\psi_\ell(x) \;=\; \exp\!\left( -\frac{\|x - c_\ell\|^2}{2 \sigma^2} \right), \qquad \{c_\ell\}_{\ell = 1}^b \subset \{X_j^p\}_{j = 1}^{n_p}.
$$
The centres $c_\ell$ are *not* parameters — they're fixed once at initialization (Sugiyama et al. recommend $b = 100$ randomly chosen from the $p$-samples, which works robustly across problem sizes). The only tunable hyperparameter is the bandwidth $\sigma$; §5.4 selects it by KL-CV.

### §5.2 Convex-program structure

Substituting the linear model into the constrained problem and replacing expectations with sample averages,
$$
\max_{\boldsymbol{\alpha} \in \mathbb{R}^b}\; \hat{\mathcal{L}}(\boldsymbol{\alpha}) \;:=\; \frac{1}{n_p} \sum_{j=1}^{n_p} \log\!\bigl(\boldsymbol{\alpha}^\top \boldsymbol{\psi}(X_j^p)\bigr) \quad \text{subject to} \quad \boldsymbol{\alpha}^\top \bar{\boldsymbol{\psi}}_q = 1, \quad \boldsymbol{\alpha} \geq \mathbf{0},
$$
where $\bar{\boldsymbol{\psi}}_q := (1/n_q) \sum_i \boldsymbol{\psi}(X_i^q) \in \mathbb{R}^b$. The objective $\hat{\mathcal{L}}$ is the average of $\log$ of a linear function of $\boldsymbol{\alpha}$ — a concave function of $\boldsymbol{\alpha}$ on the feasible region where $\boldsymbol{\alpha}^\top \boldsymbol{\psi}(X_j^p) > 0$ for every $j$. The feasible set $\{\boldsymbol{\alpha} \geq 0,\; \bar{\boldsymbol{\psi}}_q^\top \boldsymbol{\alpha} = 1\}$ is a convex polytope (a probability simplex re-scaled by $\bar{\boldsymbol{\psi}}_q$). The constrained maximization is therefore a convex program with a unique global optimum, modulo degeneracies on the boundary of the simplex.

The equality and non-negativity constraints are not redundant. Without normalization, the unconstrained objective $J^{\mathrm{unc}}_{\mathrm{KLIEP}}$ can drift to infinite ratios on a finite sample because there's nothing pinning the scale; the equality constraint fixes the only free degree of freedom and makes $\hat r \cdot q$ a probability density. Without non-negativity, the gradient direction may push individual $\alpha_\ell < 0$, which would let $\hat r(x)$ take negative values somewhere — a meaningless density ratio. Sugiyama et al.'s projected-gradient algorithm handles both constraints by alternating projection.

### §5.3 Projected-gradient ascent — Sugiyama et al. (2008), Algorithm 1

The gradient of $\hat{\mathcal{L}}$ at $\boldsymbol{\alpha}$ has a clean form. Let $\boldsymbol{\Psi}_p \in \mathbb{R}^{n_p \times b}$ be the design matrix with $(\boldsymbol{\Psi}_p)_{j\ell} = \psi_\ell(X_j^p)$, and write $\boldsymbol{r} = \boldsymbol{\Psi}_p \boldsymbol{\alpha} \in \mathbb{R}^{n_p}$ for the current estimate evaluated at the $p$-samples. Then
$$
\nabla_{\boldsymbol{\alpha}} \hat{\mathcal{L}}(\boldsymbol{\alpha}) \;=\; \frac{1}{n_p} \boldsymbol{\Psi}_p^\top \bigl(1 / \boldsymbol{r}\bigr) \;\in\; \mathbb{R}^b,
$$
where the division is component-wise. Each iteration takes a gradient-ascent step and then projects back onto the feasible set:

**Algorithm 1 (KLIEP; Sugiyama et al. 2008).**
1. Initialize $\boldsymbol{\alpha} \gets \mathbf{1} / (\bar{\boldsymbol{\psi}}_q^\top \mathbf{1})$ so the equality constraint holds at start.
2. Repeat:
    1. **Gradient ascent:** $\boldsymbol{\alpha} \gets \boldsymbol{\alpha} + \eta\, \boldsymbol{\Psi}_p^\top (1 / (\boldsymbol{\Psi}_p \boldsymbol{\alpha}))$.
    2. **Project onto $\bar{\boldsymbol{\psi}}_q^\top \boldsymbol{\alpha} = 1$:** $\boldsymbol{\alpha} \gets \boldsymbol{\alpha} + (1 - \bar{\boldsymbol{\psi}}_q^\top \boldsymbol{\alpha})\, \bar{\boldsymbol{\psi}}_q / \|\bar{\boldsymbol{\psi}}_q\|^2$.
    3. **Project onto $\boldsymbol{\alpha} \geq \mathbf{0}$:** $\boldsymbol{\alpha} \gets \max(\boldsymbol{\alpha}, \mathbf{0})$ element-wise.
    4. **Re-normalize:** $\boldsymbol{\alpha} \gets \boldsymbol{\alpha} / (\bar{\boldsymbol{\psi}}_q^\top \boldsymbol{\alpha})$.
3. Until $|\hat{\mathcal{L}}(\boldsymbol{\alpha}_{t}) - \hat{\mathcal{L}}(\boldsymbol{\alpha}_{t-1})| < \mathrm{tol}$ or max iterations reached.

The alternating projection in 2b–2d is the part that requires care: 2b enforces the equality constraint exactly, then 2c may push some components negative back to zero (breaking 2b), then 2d re-normalizes (restoring equality but possibly slightly perturbing the gradient direction). Convergence of alternating projection onto convex sets is classical (von Neumann 1933; Dykstra 1983); KLIEP's variant doesn't add the Dykstra correction, but in practice it converges to a feasible point that is within numerical tolerance of the constrained optimum, especially when the active set on the non-negativity constraint stabilizes after a few hundred iterations.

**Step size.** Sugiyama et al. recommend $\eta = 0.5$ with backtracking when the log-likelihood decreases; for the well-conditioned 1-D toys we work with, a fixed $\eta = 0.5$ converges in a few hundred iterations. For higher dimensions or pathological $r$, Armijo backtracking on $\eta$ is the standard upgrade.

### §5.4 Bandwidth selection by KL cross-validation

The bandwidth $\sigma$ in the Gaussian-kernel basis is the only hyperparameter left after fixing $b = 100$ centres. The KLIEP objective doubles as a held-out scoring rule, because the empirical $\mathbb{E}_p[\log \hat r(X)]$ on data the estimator has not seen estimates the population $\mathbb{E}_p[\log \hat r(X)] = -\mathrm{KL}(p \,\|\, \hat r \cdot q) + \text{const}$ unbiasedly. So $K$-fold KL-CV with $K = 5$ proceeds:

1. Partition the $p$-samples into $K$ folds $\{P_k\}_{k=1}^K$.
2. For each candidate $\sigma$ and each fold $k$: fit KLIEP on $\bigcup_{k' \neq k} P_{k'}$ paired with the full $q$-sample, then score $\hat{\mathcal{L}}_k(\sigma) := |P_k|^{-1} \sum_{j \in P_k} \log \hat r_\sigma(X_j^p)$.
3. The CV score is $\hat{\mathcal{L}}^{\mathrm{CV}}(\sigma) := K^{-1} \sum_k \hat{\mathcal{L}}_k(\sigma)$.
4. Choose $\sigma^\star := \arg\max_\sigma \hat{\mathcal{L}}^{\mathrm{CV}}(\sigma)$.

This is the principled bandwidth-selection scheme KMM (§4) lacked. The CV log-likelihood is concave-ish in $\log \sigma$ (typically unimodal) with the peak near the true scale of the support overlap — for our running toy with $\sigma_{\text{data}} = 1$, KL-CV picks $\sigma^\star \approx 0.5$–$1.0$ depending on the random fold split.

**Viz: KLIEP convergence trace.**
- *Shows:* two-panel figure. (Left) the log-likelihood $\hat{\mathcal{L}}(\boldsymbol{\alpha}_t)$ vs iteration $t$, with a thin gray reference line at the truth $\hat{\mathcal{L}}(\boldsymbol{\alpha} = r)$ computed using the closed-form ratio. (Right) the recovered $\hat r$ at convergence, overlaid on the true $r(x) = e^{1/2 - x}$ on a log-$y$ axis over $x \in [-3.5, 4.5]$.
- *Reader controls (interactive site version):* bandwidth $\sigma$ (slider), number of basis centres $b \in [20, 200]$, step size $\eta \in [0.1, 1.0]$ (so the reader can watch convergence accelerate or oscillate).
- *What the reader should learn:* projected gradient converges within a few hundred iterations even at coarse step sizes; the limiting $\hat r$ tracks $r$ well in the bulk and slightly mis-fits the deep left tail (where $q$-samples are scarce, expected from §2.3).

**Viz: KL-CV bandwidth selection.**
- *Shows:* two-panel figure. (Left) the $K = 5$-fold CV log-likelihood $\hat{\mathcal{L}}^{\mathrm{CV}}(\sigma)$ vs $\sigma$ on a log-$\sigma$ axis, with $\sigma^\star$ marked. (Right) the $\hat r$ estimate at $\sigma^\star$ vs $\hat r$ at $\sigma = \sigma^\star / 3$ (too small, jagged) and $\sigma = 3 \sigma^\star$ (too large, over-smoothed) overlaid on the true $r$.
- *Reader controls (interactive site version):* shift $\mu_q$ (changes the optimal bandwidth), sample sizes $n_p = n_q$, number of folds $K \in \{2, 5, 10\}$.
- *What the reader should learn:* the CV criterion is unimodal in $\log \sigma$ and its peak reliably picks a bandwidth that produces a visibly good fit; the alternative bandwidths visibly under- or over-smooth.

**Code experiment.**
- Sample $n_p = n_q = 300$ from the running toy.
- Implement `fit_kliep(x_p, x_q, sigma, ...)` per Algorithm 1, returning $\hat{\boldsymbol{\alpha}}$, the kernel centres, and the iteration-by-iteration log-likelihood history.
- Cell 1 (after §5.3): fit at $\sigma$ = median-heuristic, plot convergence + final $\hat r$.
- Cell 2 (after §5.4): sweep $\sigma$ over a log-spaced grid of $\sim 10$ values; run 5-fold KL-CV; identify $\sigma^\star$; plot the CV curve and the recovered $\hat r$ at three bandwidths.
- Validate: convergence log-likelihood reaches within $0.05$ of the truth-evaluated baseline; $\sigma^\star \in [0.3, 1.5]$; the $\hat r$ at $\sigma^\star$ has Pearson correlation $> 0.95$ with the true $r$ evaluated on a fine grid.


## §6. LSIF and uLSIF — least-squares importance fitting

§6 closes the loop on the kernel-based estimator family. LSIF (Kanamori, Hido, and Sugiyama 2009) is the squared-loss specialization of the Bregman framework from §3.2 — replacing the KL objective of §5 with $\tfrac{1}{2} \mathbb{E}_q[g(X)^2] - \mathbb{E}_p[g(X)]$, which turns the optimization into a *linearly-constrained convex quadratic program* with a non-negativity constraint $\boldsymbol{\alpha} \geq 0$. uLSIF (Kanamori, Hido, and Sugiyama 2009, §3) is the simple but consequential observation that dropping the non-negativity constraint turns the QP into a single linear solve with a closed-form solution, and — more importantly — admits an *analytic* leave-one-out cross-validation formula that costs $O(b^3 + b^2 (n_p + n_q))$ to compute, no refitting required. That single feature is why uLSIF overtook both LSIF and KLIEP as the practitioner default for kernel-based DRE. Kanamori, Suzuki, and Sugiyama (2012) provide the formal statistical analysis (consistency, convergence rates, the KuLSIF kernelized variant), but the algorithm and the analytic-LOO formula are from the 2009 JMLR paper.

This section derives both estimators, presents the Sherman-Morrison machinery behind the LOO formula, and runs uLSIF on the §1 toy with bandwidth and regularization selected by analytic LOO-CV — then compares against the KLIEP fit from §5 on both accuracy and runtime.

### §6.1 The squared-loss closed form

Recall from §3.2 that the squared-loss Bregman objective is
$$
J_{\mathrm{LSIF}}(g) \;=\; \tfrac{1}{2}\, \mathbb{E}_q[g(X)^2] \;-\; \mathbb{E}_p[g(X)],
$$
with sample version $\hat J_{\mathrm{LSIF}}(g) = (1/(2 n_q)) \sum_i g(X_i^q)^2 - (1/n_p) \sum_j g(X_j^p)$, and population minimizer $g^\star = r$. We use the same linear-in-parameters model as KLIEP, $\hat r(x) = \boldsymbol{\alpha}^\top \boldsymbol{\psi}(x)$, with $\boldsymbol{\psi}: \mathcal{X} \to \mathbb{R}_{\geq 0}^b$ a fixed Gaussian-RBF basis centred at $b = 100$ random $p$-samples.

Defining the empirical design matrices $\boldsymbol{\Psi}_p \in \mathbb{R}^{n_p \times b}$, $\boldsymbol{\Psi}_q \in \mathbb{R}^{n_q \times b}$ with rows $\boldsymbol{\psi}(X_j^p)^\top, \boldsymbol{\psi}(X_i^q)^\top$, and the sample-mean quantities
$$
\hat{\mathbf{H}} \;:=\; \tfrac{1}{n_q}\, \boldsymbol{\Psi}_q^\top \boldsymbol{\Psi}_q \;\in\; \mathbb{R}^{b \times b}, \qquad \hat{\mathbf{h}} \;:=\; \tfrac{1}{n_p}\, \boldsymbol{\Psi}_p^\top \mathbf{1}_{n_p} \;\in\; \mathbb{R}^b,
$$
the empirical objective is the convex quadratic
$$
\hat J_{\mathrm{LSIF}}(\boldsymbol{\alpha}) \;=\; \tfrac{1}{2}\, \boldsymbol{\alpha}^\top \hat{\mathbf{H}}\, \boldsymbol{\alpha} \;-\; \hat{\mathbf{h}}^\top \boldsymbol{\alpha}.
$$
Adding a Tikhonov regularizer $\tfrac{1}{2} \lambda \|\boldsymbol{\alpha}\|^2$ to keep the linear system well-conditioned, the LSIF problem with non-negativity is
$$
\min_{\boldsymbol{\alpha} \in \mathbb{R}^b} \;\; \tfrac{1}{2}\, \boldsymbol{\alpha}^\top (\hat{\mathbf{H}} + \lambda \mathbf{I})\, \boldsymbol{\alpha} \;-\; \hat{\mathbf{h}}^\top \boldsymbol{\alpha} \quad \text{subject to} \quad \boldsymbol{\alpha} \geq \mathbf{0}.
$$
Without the non-negativity constraint, first-order optimality is the linear system $(\hat{\mathbf{H}} + \lambda \mathbf{I})\, \hat{\boldsymbol{\alpha}} = \hat{\mathbf{h}}$, with the closed-form solution
$$
\boxed{\;\hat{\boldsymbol{\alpha}}(\lambda) \;=\; (\hat{\mathbf{H}} + \lambda \mathbf{I})^{-1}\, \hat{\mathbf{h}}.\;}
$$
The matrix $\hat{\mathbf{H}}$ is positive semi-definite (it's $\boldsymbol{\Psi}_q^\top \boldsymbol{\Psi}_q / n_q$, a Gram-matrix average), so $\hat{\mathbf{H}} + \lambda \mathbf{I}$ is strictly positive definite for any $\lambda > 0$ and the inverse exists. *With* the non-negativity constraint, this becomes a non-negativity-constrained QP that Kanamori, Hido, and Sugiyama (2009) solve via a customized active-set algorithm (NNQP). The constrained version is **LSIF**.

### §6.2 uLSIF — dropping non-negativity for the closed form

The motivation for dropping the non-negativity constraint is computational: the NNQP solver needs an iterative active-set sweep with $O(b)$ outer iterations and $O(b^3)$ inner work, while the unconstrained closed form is a single $O(b^3)$ linear solve. The motivation for *trusting* the relaxation is empirical: when the Gaussian basis is rich enough and the bandwidth $\sigma$ is sensible, the unconstrained $\hat{\boldsymbol{\alpha}}$ rarely has many negative components, and even when it does, the predicted ratio $\hat r(x) = \boldsymbol{\alpha}^\top \boldsymbol{\psi}(x)$ is positive almost everywhere in the support overlap (negative outputs occur only far from any kernel centre, where extrapolation is unreliable regardless of the constraint).

The standard uLSIF post-processing is to clip predictions at zero: $\hat r_+(x) := \max(\hat r(x), 0)$. Kanamori, Suzuki, and Sugiyama (2012, §4.3) show that the convergence rate of $\hat r_+$ to the true $r$ matches that of the constrained LSIF estimator up to a constant — the relaxation pays no statistical price asymptotically and a modest one in finite samples, but the operational simplification (closed form + analytic LOO) more than compensates.

**uLSIF in three lines.**
1. Sample $b$ kernel centres from the $p$-samples; build $\boldsymbol{\Psi}_p, \boldsymbol{\Psi}_q$ at bandwidth $\sigma$.
2. Form $\hat{\mathbf{H}} = \boldsymbol{\Psi}_q^\top \boldsymbol{\Psi}_q / n_q$ and $\hat{\mathbf{h}} = \boldsymbol{\Psi}_p^\top \mathbf{1}_{n_p} / n_p$.
3. Solve $(\hat{\mathbf{H}} + \lambda \mathbf{I})\, \hat{\boldsymbol{\alpha}} = \hat{\mathbf{h}}$; predict $\hat r(x) = \max(\hat{\boldsymbol{\alpha}}^\top \boldsymbol{\psi}(x), 0)$.

The only thing missing is a principled way to pick $(\sigma, \lambda)$. That's what §6.3 gives us.

### §6.3 Analytic leave-one-out cross-validation

LOO-CV evaluates the LSIF objective on left-out samples: for each $q$-sample $i$, refit on the other $n_q - 1$ samples and evaluate the squared-loss contribution; for each $p$-sample $j$, refit and evaluate the linear-loss contribution. The total LOO score is
$$
\mathrm{SC}_{\mathrm{LOO}}(\sigma, \lambda) \;:=\; \frac{1}{2 n_q} \sum_{i=1}^{n_q} \bigl(\hat{\boldsymbol{\alpha}}^{(-i,q)\,\top} \boldsymbol{\psi}(X_i^q)\bigr)^2 \;-\; \frac{1}{n_p} \sum_{j=1}^{n_p} \hat{\boldsymbol{\alpha}}^{(-j,p)\,\top} \boldsymbol{\psi}(X_j^p).
$$
Naively, this needs $n_p + n_q$ refits per $(\sigma, \lambda)$, each costing $O(b^3)$. For uLSIF, the closed form lets us compute *all* refits in two precomputed $b \times b$ matrix inverses plus $O(b^2)$ per left-out sample, by Sherman-Morrison-style rank-one updates. We derive the formula carefully, treating the $(n_q - 1)$ and $(n_p - 1)$ effective-sample-size bookkeeping exactly, then read off the asymptotic form that appears in Kanamori, Hido, and Sugiyama (2009).

**Lemma (Leave-out empirical matrices, exact).** *The leave-$i$-out empirical second-moment matrix and leave-$j$-out empirical mean satisfy*
$$
\hat{\mathbf{H}}^{(-i,q)} \;=\; \frac{n_q}{n_q - 1}\, \hat{\mathbf{H}} \;-\; \frac{1}{n_q - 1}\, \boldsymbol{\psi}_q^{(i)} \boldsymbol{\psi}_q^{(i)\top}, \qquad \hat{\mathbf{h}}^{(-j,p)} \;=\; \frac{n_p}{n_p - 1}\, \hat{\mathbf{h}} \;-\; \frac{1}{n_p - 1}\, \boldsymbol{\psi}_p^{(j)},
$$
*where $\boldsymbol{\psi}_q^{(i)} := \boldsymbol{\psi}(X_i^q)$ and $\boldsymbol{\psi}_p^{(j)} := \boldsymbol{\psi}(X_j^p)$.*

*Proof.* $\hat{\mathbf{H}} = (1/n_q) \sum_{i'} \boldsymbol{\psi}_q^{(i')} \boldsymbol{\psi}_q^{(i')\top}$, so $\sum_{i' \ne i} \boldsymbol{\psi}_q^{(i')} \boldsymbol{\psi}_q^{(i')\top} = n_q \hat{\mathbf{H}} - \boldsymbol{\psi}_q^{(i)} \boldsymbol{\psi}_q^{(i)\top}$, and dividing by $(n_q - 1)$ gives the first display. The argument for $\hat{\mathbf{h}}^{(-j,p)}$ is identical. $\blacksquare$

**Lemma (Sherman-Morrison; standard).** *For invertible $\mathbf{A} \in \mathbb{R}^{b \times b}$ and $\mathbf{u} \in \mathbb{R}^b$ with $1 - \mathbf{u}^\top \mathbf{A}^{-1} \mathbf{u} \neq 0$,*
$$
(\mathbf{A} - \mathbf{u}\, \mathbf{u}^\top)^{-1} \;=\; \mathbf{A}^{-1} \;+\; \frac{\mathbf{A}^{-1} \mathbf{u}\, \mathbf{u}^\top \mathbf{A}^{-1}}{1 - \mathbf{u}^\top \mathbf{A}^{-1} \mathbf{u}}.
$$

**Theorem (Exact analytic LOO for uLSIF).** *Let $\mathbf{B} := \hat{\mathbf{H}} + \lambda \mathbf{I}$ be the regularized matrix from the full fit, and let $\mathbf{A}_q := (n_q / (n_q - 1))\, \hat{\mathbf{H}} + \lambda \mathbf{I}$ be the "deflated" base matrix that absorbs the leave-out rescaling. Define, for each $i, j$:*
$$
\tilde r_i := \boldsymbol{\psi}_q^{(i)\top} \mathbf{A}_q^{-1} \hat{\mathbf{h}}, \qquad \eta_i := \boldsymbol{\psi}_q^{(i)\top} \mathbf{A}_q^{-1} \boldsymbol{\psi}_q^{(i)}, \qquad s_j := \boldsymbol{\psi}_p^{(j)\top} \hat{\boldsymbol{\alpha}}, \qquad \delta_j := \boldsymbol{\psi}_p^{(j)\top} \mathbf{B}^{-1} \boldsymbol{\psi}_p^{(j)}.
$$
*Then the exact leave-one-out predicted ratios are*
$$
\hat r^{\mathrm{LOO},q}_i \;=\; \frac{\tilde r_i}{1 - \eta_i / (n_q - 1)}, \qquad \hat r^{\mathrm{LOO},p}_j \;=\; \frac{n_p\, s_j - \delta_j}{n_p - 1}.
$$

*Proof.* For the $q$-removal, the first lemma gives the leave-$i$-out regularized matrix as
$$
\mathbf{B}^{(-i,q)} \;=\; \hat{\mathbf{H}}^{(-i,q)} + \lambda \mathbf{I} \;=\; \underbrace{\frac{n_q}{n_q - 1}\, \hat{\mathbf{H}} + \lambda \mathbf{I}}_{=\, \mathbf{A}_q} \;-\; \frac{1}{n_q - 1}\, \boldsymbol{\psi}_q^{(i)} \boldsymbol{\psi}_q^{(i)\top}.
$$
$\mathbf{A}_q$ is independent of $i$, so we precompute $\mathbf{A}_q^{-1}$ once per $(\sigma, \lambda)$. Apply Sherman-Morrison with $\mathbf{A} = \mathbf{A}_q$ and $\mathbf{u} = \boldsymbol{\psi}_q^{(i)} / \sqrt{n_q - 1}$:
$$
(\mathbf{B}^{(-i,q)})^{-1} \;=\; \mathbf{A}_q^{-1} \;+\; \frac{1}{n_q - 1} \cdot \frac{\mathbf{A}_q^{-1} \boldsymbol{\psi}_q^{(i)}\, \boldsymbol{\psi}_q^{(i)\top} \mathbf{A}_q^{-1}}{1 - \eta_i / (n_q - 1)}.
$$
$\hat{\mathbf{h}}$ is unchanged when we remove a $q$-sample, so $\hat{\boldsymbol{\alpha}}^{(-i,q)} = (\mathbf{B}^{(-i,q)})^{-1}\, \hat{\mathbf{h}}$ and the prediction at $X_i^q$ is
$$
\hat r^{\mathrm{LOO},q}_i \;=\; \boldsymbol{\psi}_q^{(i)\top} (\mathbf{B}^{(-i,q)})^{-1}\, \hat{\mathbf{h}} \;=\; \tilde r_i \;+\; \frac{1}{n_q - 1} \cdot \frac{\eta_i\, \tilde r_i}{1 - \eta_i / (n_q - 1)} \;=\; \frac{\tilde r_i}{1 - \eta_i / (n_q - 1)},
$$
using $\boldsymbol{\psi}_q^{(i)\top} \mathbf{A}_q^{-1} \hat{\mathbf{h}} = \tilde r_i$ and $\boldsymbol{\psi}_q^{(i)\top} \mathbf{A}_q^{-1} \boldsymbol{\psi}_q^{(i)} = \eta_i$.

For the $p$-removal, the regularized matrix $\mathbf{B}$ is unchanged (it depends only on $q$-samples), but $\hat{\mathbf{h}}$ is replaced by $\hat{\mathbf{h}}^{(-j,p)} = (n_p / (n_p - 1)) \hat{\mathbf{h}} - (1/(n_p - 1)) \boldsymbol{\psi}_p^{(j)}$, so
$$
\hat{\boldsymbol{\alpha}}^{(-j,p)} \;=\; \mathbf{B}^{-1}\, \hat{\mathbf{h}}^{(-j,p)} \;=\; \frac{n_p}{n_p - 1}\, \hat{\boldsymbol{\alpha}} \;-\; \frac{1}{n_p - 1}\, \mathbf{B}^{-1} \boldsymbol{\psi}_p^{(j)},
$$
and the prediction at $X_j^p$ is
$$
\hat r^{\mathrm{LOO},p}_j \;=\; \boldsymbol{\psi}_p^{(j)\top} \hat{\boldsymbol{\alpha}}^{(-j,p)} \;=\; \frac{n_p\, s_j - \delta_j}{n_p - 1}. \;\blacksquare
$$

**Corollary (Asymptotic form; Kanamori–Hido–Sugiyama 2009, §3.5).** *In the limit $n_q, n_p \to \infty$ with $\lambda$ fixed,* $\mathbf{A}_q \to \mathbf{B}$, *so* $\eta_i \to h_i := \boldsymbol{\psi}_q^{(i)\top} \mathbf{B}^{-1} \boldsymbol{\psi}_q^{(i)}$ *and* $\tilde r_i \to r_i := \boldsymbol{\psi}_q^{(i)\top} \hat{\boldsymbol{\alpha}}$, *and the exact formulas collapse to*
$$
\hat r^{\mathrm{LOO},q}_i \;\longrightarrow\; \frac{r_i}{1 - h_i / n_q}, \qquad \hat r^{\mathrm{LOO},p}_j \;\longrightarrow\; s_j - \delta_j / n_p.
$$

The asymptotic form differs from the exact form by an $O(1/n)$ correction in the denominators. For $n_q \ge 100$ the two forms agree to better than $1\%$ on every LOO quantity — well below the finite-sample noise in the scoring rule. We implement the asymptotic form below because it reuses the single matrix inverse $\mathbf{B}^{-1}$ that we already had to compute for the fit, instead of requiring a second inverse $\mathbf{A}_q^{-1}$ per $(\sigma, \lambda)$ pair. Numerical verification on the running toy (printed at the end of §6.4) confirms agreement to four decimal places between the exact-form and asymptotic-form LOO scores.

**Computational cost.** Asymptotic implementation: $\mathbf{B}^{-1}$ costs $O(b^3)$ once. Each $h_i, \delta_j, r_i, s_j$ costs $O(b^2)$, so one $(\sigma, \lambda)$ LOO score is $O(b^3 + b^2 (n_p + n_q))$. Exact implementation adds a second $O(b^3)$ matrix inverse for $\mathbf{A}_q^{-1}$ — a factor-of-two penalty on the inverse part, negligible against the dominant $b^2 (n_p + n_q)$ Sherman-Morrison work for moderate $n_p + n_q$. Either way, this is dramatically cheaper than KLIEP's $K$-fold CV in §5, which costs $K \cdot |\sigma\text{-grid}| \cdot (\text{KLIEP iterations}) \cdot O(b^2)$ — measurably worse, even before counting projected-gradient overhead per iteration.

### §6.4 Numerical demonstration

The §6.4 cell sweeps uLSIF over a grid of $(\sigma, \lambda)$ pairs, picks the analytic-LOO-CV optimum, and compares the fit and runtime to the KLIEP estimate from §5 on the same toy.

**Viz: uLSIF LOO-CV landscape and three-way fit comparison.**
- *Shows:* two-panel figure. (Left) heatmap of $\mathrm{SC}_{\mathrm{LOO}}(\sigma, \lambda)$ over a log-spaced grid in $\sigma \in [0.25, 4]$ and $\lambda \in [10^{-4}, 10^{-1}]$, with the argmin $(\sigma^\star, \lambda^\star)$ marked. (Right) the recovered $\hat r$ at the uLSIF optimum overlaid on the KLIEP estimate from §5 and the closed-form true ratio.
- *Reader controls (interactive site version):* shift $\mu_q$, sample sizes, $\sigma$ and $\lambda$ grid resolution, and a toggle for the "with vs without non-negativity clipping" post-processing step.
- *What the reader should learn:* (i) the LOO-CV landscape is smooth and unimodal in $\log \sigma$ for any fixed $\lambda$, so the bandwidth selection is robust; (ii) the uLSIF and KLIEP fits agree to within their respective fit errors on the bulk of the support and diverge slightly in the tails (uLSIF more conservatively in the high-$r$ region); (iii) uLSIF runtime is one to two orders of magnitude lower than KLIEP-with-$K$-fold-CV at equivalent grid resolution.

**Code experiment.**
- Sample $n_p = n_q = 300$ from the running toy (reuse §5's samples for direct fit comparison).
- Implement `ulsif_fit(x_p, x_q, sigma, lam)` returning $\hat{\boldsymbol{\alpha}}$ and centres (one linear solve).
- Implement `ulsif_loocv(...)` per the §6.3 theorem.
- Sweep $\sigma$ over $9$ values, $\lambda$ over $7$ values; record LOO score for each.
- Identify $(\sigma^\star, \lambda^\star)$, fit uLSIF there, predict on the grid.
- Plot the heatmap and the three-way comparison; print the (Pearson, runtime) table.
- Validate: $\sigma^\star \in [0.3, 1.5]$; Pearson(uLSIF $\hat r$, true $r$) on the grid > 0.95; uLSIF total runtime (grid sweep + fit) is at least $5\times$ smaller than the KLIEP CV sweep timing from §5.

---

## §7. Probabilistic classification as DRE

§7 is the section where direct DRE quietly stops being a separate algorithm family and becomes "fit any well-calibrated classifier on a pooled labelled dataset." The reformulation is short — half a page of Bayes' rule — and the consequences are large: every gradient-boosted tree, every neural-network classifier, every penalized logistic regression that exists in a practitioner's pipeline can be repurposed as a density-ratio estimator without rebuilding the algorithm. The catch is calibration: the classifier's predicted probabilities must approximate the true posterior on the pooled-label problem, not just rank-order it. AUC alone is not enough.

This section derives the pooled-classification reformulation, recovers the logistic loss as a Bregman-DRE specialization (closing the loop on §3.4's preview), explains calibration carefully, and runs a head-to-head comparison between L2-penalized logistic regression on the same Gaussian basis we used for uLSIF (§6) and uLSIF itself, on the same toy.

### §7.1 The pooled-dataset construction

Pool the $p$-samples and $q$-samples into one dataset of size $N = n_p + n_q$, and attach a binary label that records which sample each came from:
$$
\mathcal{D}_{\mathrm{pool}} \;:=\; \bigl\{ (X_i^p, Y = 1) \bigr\}_{i=1}^{n_p} \,\cup\, \bigl\{ (X_j^q, Y = 0) \bigr\}_{j=1}^{n_q}.
$$
This dataset has the structure of a labelled binary classification problem with class-conditional densities $f(x \mid Y = 1) = p(x)$ and $f(x \mid Y = 0) = q(x)$, and class-prior probabilities $\pi_1 = n_p / N$ and $\pi_0 = n_q / N$. The pooled marginal density is the mixture $f(x) = \pi_1 p(x) + \pi_0 q(x)$.

Fit a *probabilistic classifier* — anything that produces an estimate $\hat\pi(x) := \hat{\mathbb{P}}(Y = 1 \mid X = x)$ rather than just a hard label. The next subsection shows that this estimate, divided by $1 - \hat\pi(x)$ and multiplied by the prior-odds ratio, *is* an estimate of $r(x)$.

### §7.2 The Bayes-rule identity

**Theorem (Classification-DRE identity).** *Under the pooled-dataset construction with class-conditionals $f(x \mid Y = 1) = p(x)$, $f(x \mid Y = 0) = q(x)$ and priors $\pi_1, \pi_0$,*
$$
r(x) \;=\; \frac{p(x)}{q(x)} \;=\; \frac{\pi_0}{\pi_1} \cdot \frac{\mathbb{P}(Y = 1 \mid X = x)}{\mathbb{P}(Y = 0 \mid X = x)} \;=\; \frac{n_q}{n_p} \cdot \frac{\mathbb{P}(Y = 1 \mid X = x)}{\mathbb{P}(Y = 0 \mid X = x)}.
$$

*Proof.* By Bayes' rule applied to the pooled distribution,
$$
\mathbb{P}(Y = 1 \mid X = x) \;=\; \frac{\pi_1\, f(x \mid Y = 1)}{f(x)} \;=\; \frac{\pi_1\, p(x)}{\pi_1\, p(x) + \pi_0\, q(x)},
$$
$$
\mathbb{P}(Y = 0 \mid X = x) \;=\; \frac{\pi_0\, q(x)}{\pi_1\, p(x) + \pi_0\, q(x)}.
$$
Dividing — the mixture denominator $f(x)$ cancels — gives
$$
\frac{\mathbb{P}(Y = 1 \mid X = x)}{\mathbb{P}(Y = 0 \mid X = x)} \;=\; \frac{\pi_1\, p(x)}{\pi_0\, q(x)} \;=\; \frac{\pi_1}{\pi_0}\, r(x),
$$
and the claim follows by solving for $r(x)$ with $\pi_1 / \pi_0 = n_p / n_q$. $\blacksquare$

In the balanced case $n_p = n_q$ the prior-odds ratio is $1$ and $r(x) = \mathbb{P}(Y = 1 \mid x) / \mathbb{P}(Y = 0 \mid x) = \exp(\eta(x))$, where $\eta(x) := \log \mathbb{P}(Y = 1 \mid x) / \mathbb{P}(Y = 0 \mid x)$ is the **logit**. The DRE estimator is then $\hat r(x) = \exp(\hat\eta(x))$ — the exponentiated logit of any probabilistic classifier we choose to fit.

**The logistic loss as Bregman-DRE (closing §3.4).** The logistic-loss training objective for a binary classifier with logit $\eta$ is the negative cross-entropy
$$
\mathcal{L}_{\mathrm{LR}}(\eta) \;=\; -\frac{1}{N} \sum_{i=1}^{N} \bigl[ Y_i\, \log \sigma(\eta(X_i)) + (1 - Y_i)\, \log(1 - \sigma(\eta(X_i))) \bigr],
$$
with $\sigma(t) = 1 / (1 + e^{-t})$ the sigmoid. Substituting $Y_i$'s label structure (1 for $p$-samples, 0 for $q$-samples) and using the population-level decomposition $\frac{1}{N} \sum Y_i \,\cdot\, = \pi_1 \mathbb{E}_p[\cdot]$, $\frac{1}{N} \sum (1 - Y_i) \,\cdot\, = \pi_0 \mathbb{E}_q[\cdot]$, the population objective is
$$
\mathcal{L}_{\mathrm{LR}}^{\mathrm{pop}}(\eta) \;=\; -\pi_1\, \mathbb{E}_p[\log \sigma(\eta(X))] - \pi_0\, \mathbb{E}_q[\log(1 - \sigma(\eta(X)))].
$$
Reparametrizing with $g = e^\eta$ (so $\sigma(\eta) = g / (1 + g)$ and $1 - \sigma(\eta) = 1 / (1 + g)$) and dropping the prior weights for clarity, the loss in the $g$-parametrization is
$$
\mathcal{L}_{\mathrm{LR}}^{\mathrm{pop}}(g) \;\propto\; -\mathbb{E}_p\bigl[\log \tfrac{g(X)}{1 + g(X)}\bigr] + \mathbb{E}_q\bigl[\log(1 + g(X))\bigr],
$$
which is precisely $J_\phi(g)$ from §3.1 with the Bregman generator $\phi_{\mathrm{LR}}(t) = t \log t - (1 + t) \log(1 + t) + (1 + t) \log 2$ (Menon and Ong 2016, Table 1, row 3). The population minimizer is $g^\star = r$, so when the priors are balanced the Bregman framework recovers exactly the result the Bayes-rule identity gives directly: train logistic regression on the pooled labels, exponentiate the logit, that's a DRE estimator.

### §7.3 Calibration

The classification-DRE identity assumes the classifier's predicted probabilities $\hat\pi(x)$ are approximately equal to the true posterior $\mathbb{P}(Y = 1 \mid X = x)$ — that is, the classifier is **calibrated**:
$$
\mathbb{P}(Y = 1 \mid \hat\pi(X) = p) \;\approx\; p, \quad \text{for all } p \in [0, 1].
$$
This is a strictly stronger requirement than discriminative performance. A classifier with perfect AUC = 1 can be arbitrarily badly calibrated — for example, if it squashes all probabilities to $\{\epsilon, 1 - \epsilon\}$ for some small $\epsilon$, the rank-order is right but the absolute values are wrong, and the DRE estimate $\hat r(x) = (n_q/n_p) \hat\pi(x) / (1 - \hat\pi(x))$ takes only two values regardless of $x$.

Which off-the-shelf classifiers are calibrated?

- **L2-penalized logistic regression** with proper-loss (cross-entropy) training is well-calibrated by construction, because the loss is a *strictly proper scoring rule* (Gneiting and Raftery 2007) — the population minimizer of cross-entropy is the true posterior, and a well-specified parametric family converges to it. This is why logistic regression is the natural default for classification-DRE.
- **SVMs, random forests, gradient-boosted trees, AdaBoost** are typically *not* well-calibrated. SVMs hinge-loss optimize for margin, not probabilities; tree ensembles tend to push predictions toward extremes (Niculescu-Mizil and Caruana 2005). These need post-hoc recalibration — **Platt scaling** (sigmoid fit on a held-out calibration set) for SVMs, **isotonic regression** (monotone non-parametric fit) for tree ensembles.
- **Deep neural network classifiers** are usually over-confident on modern architectures (Guo et al. 2017); calibration is an active research area, with temperature scaling the most common post-hoc fix.

For this section's numerical demo we use logistic regression on the same Gaussian basis we used for uLSIF (§6), which gives the cleanest head-to-head comparison: identical features, identical $b$, identical $\lambda$, only the loss differs (squared vs logistic). The L2 penalty in `sklearn.linear_model.LogisticRegression` is exactly the analogue of uLSIF's $\lambda$ Tikhonov ridge.

### §7.4 Numerical demonstration

The §7.4 cell pools the $n_p = n_q = 300$ shift-Gaussian samples from §1 with labels $\{1, 0\}$, fits an L2-penalized logistic regression on the Gaussian basis at the same centres and bandwidth uLSIF picked in §6, exponentiates the logit to get $\hat r_{\mathrm{LR}}$, and compares it head-to-head with $\hat r_{\mathrm{uLSIF}}$ and the closed-form truth. We also produce a reliability diagram to verify that the logistic-regression classifier is well-calibrated on this problem.

**Viz: classification-DRE vs uLSIF comparison.**
- *Shows:* two-panel figure. (Left) the three-way 1-D plot — true $r(x)$, $\hat r_{\mathrm{uLSIF}}$ from §6, $\hat r_{\mathrm{LR}}$ from logistic regression — overlaid on the running $x$-grid with log-$y$ axis. (Right) reliability diagram for the logistic-regression classifier: predicted $\hat\pi(X)$ binned into deciles, empirical fraction of $Y = 1$ in each bin, with the $y = x$ calibration diagonal.
- *Reader controls (interactive site version):* a dropdown to swap the classifier between `LogisticRegression`, `RandomForest`, `GradientBoosting`, and `MLPClassifier`; a toggle for "apply Platt scaling" post-processing; the L2 penalty $C$ on a log slider.
- *What the reader should learn:* (i) logistic regression on the same Gaussian basis produces a DRE estimate effectively indistinguishable from uLSIF — the loss choice doesn't dominate when both estimators are well-specified; (ii) the calibration curve sits on the diagonal, confirming that the Bayes-rule identity's hypothesis is met; (iii) swapping in a tree ensemble (in the interactive version) visibly biases the DRE estimate even though the AUC barely moves, illustrating §7.3's caution.

**Code experiment.**
- Pool $X_1^p, \ldots, X_{n_p}^p$ (label 1) with $X_1^q, \ldots, X_{n_q}^q$ (label 0) on the shift-Gaussian toy from §1; reuse the §6 samples for direct comparison.
- Build the Gaussian-basis design matrix on the pooled $x$'s using the same kernel centres and bandwidth $\sigma^\star$ that uLSIF picked in §6.4.
- Fit `sklearn.linear_model.LogisticRegression(penalty="l2", C=1/lambda_star, solver="lbfgs")`.
- Predict $\hat\pi(x)$ on the $x$-grid; form $\hat r_{\mathrm{LR}}(x) = (n_q / n_p) \cdot \hat\pi / (1 - \hat\pi)$.
- Plot the three-way fit and the reliability diagram; print Pearson correlations between each estimator and the truth, and the calibration-curve maximum vertical deviation from the diagonal.
- Validate: Pearson(LR, true) > 0.95; Pearson(LR, uLSIF) > 0.97 (they agree more closely with each other than either does with truth, since both have the same finite-sample basis-limitations); reliability-diagram max deviation < 0.10.

---

## §8. The $f$-divergence variational view and neural DRE

§8 is the section that opens the door from kernel-basis methods to arbitrary function classes — and ultimately to deep generative models. Nguyen, Wainwright, and Jordan (2010) showed that every $f$-divergence between $p$ and $q$ has a *variational representation* as the supremum, over a class of "witness" functions $T$, of a sample-computable functional whose optimizer is $T^\star = f'(r)$. The construction recovers LSIF (from the chi-squared generator) and KLIEP (from the KL generator) as special cases of one framework, but the framework's real power is that $T$ can be parametrized by anything we like — including a small neural network trained by SGD. §8.3 implements that idea on the running toy with PyTorch, and §8.4 closes the section with the observation that vanilla GANs are doing exactly this estimation implicitly: the discriminator's sigmoid output *is* the ratio $r / (1 + r)$, and minimizing the GAN objective over the generator is minimizing an estimated $f$-divergence between data and generator distributions.

This is the section where the PyTorch policy kicks in: §8.3 and §8.4 each have a short PyTorch example; all other sections in the topic stay in NumPy / SciPy / scikit-learn. The PyTorch import block lives only inside the §8 code cells.

### §8.1 The Nguyen–Wainwright–Jordan variational lower bound

We begin by recalling the $f$-divergence family and the Fenchel conjugate machinery the NWJ theorem rests on.

**Definition ($f$-divergence).** For a convex function $f: (0, \infty) \to \mathbb{R}$ with $f(1) = 0$, the **$f$-divergence** between probability densities $p, q$ with $p \ll q$ is
$$
D_f(p \,\|\, q) \;:=\; \int_\mathcal{X} f\!\left(\frac{p(x)}{q(x)}\right) q(x)\, \mathrm{d}x \;=\; \mathbb{E}_q[f(r(X))].
$$
Familiar choices: $f(u) = u \log u$ gives $\mathrm{KL}(p \,\|\, q) = \mathbb{E}_p[\log r]$; $f(u) = (u - 1)^2 / 2$ gives $\tfrac{1}{2} \chi^2(p \,\|\, q)$; $f(u) = u \log u - (1 + u) \log\!\bigl(\tfrac{1 + u}{2}\bigr)$ gives twice the Jensen–Shannon divergence. $D_f \geq 0$ with equality iff $p = q$ (Jensen's inequality on convex $f$ and $\mathbb{E}_q[r] = 1$).

**Definition (Fenchel conjugate).** The **Fenchel conjugate** of a convex function $f: \mathbb{R} \to \mathbb{R} \cup \{+\infty\}$ is
$$
f^\star(t) \;:=\; \sup_{u \in \mathbb{R}}\, \bigl\{ u\, t - f(u) \bigr\}.
$$
$f^\star$ is convex and lower semi-continuous. The Fenchel–Young inequality $f(u) + f^\star(t) \geq u\, t$ holds for all $u, t$, with equality iff $t = f'(u)$ (assuming $f$ differentiable on the interior of its domain). For our purposes, $f$ is differentiable on $(0, \infty)$ and the relevant conjugates compute cleanly:

| $f(u)$                  | $f'(u)$        | $f^\star(t)$         | corresponding DRE estimator |
|-------------------------|----------------|----------------------|------------------------------|
| $u \log u$              | $\log u + 1$   | $e^{t - 1}$          | KLIEP (§5)                   |
| $(u - 1)^2 / 2$         | $u - 1$        | $t^2/2 + t$          | LSIF (§6)                    |
| $-\log u$               | $-1/u$         | $-1 - \log(-t)$, $t < 0$ | reverse-KL DRE          |

**Theorem (NWJ Variational Representation; Nguyen–Wainwright–Jordan 2010, Lemma 1).** *Let $f$ be a convex function with Fenchel conjugate $f^\star$. Then*
$$
D_f(p \,\|\, q) \;=\; \sup_{T: \mathcal{X} \to \operatorname{dom}(f^\star)}\, \Bigl\{\, \mathbb{E}_p[T(X)] \;-\; \mathbb{E}_q[f^\star(T(X))] \,\Bigr\},
$$
*where the supremum runs over all measurable $T$ for which the integrals exist, and is achieved at*
$$
T^\star(x) \;=\; f'(r(x)).
$$

*Proof.* Two directions.

**Lower bound ($\geq$).** Apply the Fenchel–Young inequality pointwise with $u = r(x)$ and $t = T(x)$:
$$
f(r(x)) \;\geq\; T(x)\, r(x) - f^\star(T(x)).
$$
Taking $\mathbb{E}_q$ on both sides,
$$
\mathbb{E}_q[f(r(X))] \;\geq\; \mathbb{E}_q[T(X)\, r(X)] \;-\; \mathbb{E}_q[f^\star(T(X))].
$$
By the §2.2 importance-weighting identity, $\mathbb{E}_q[T(X)\, r(X)] = \mathbb{E}_p[T(X)]$, so
$$
D_f(p \,\|\, q) \;\geq\; \mathbb{E}_p[T(X)] \;-\; \mathbb{E}_q[f^\star(T(X))]
$$
for every $T$ in the supremum's class.

**Achievement of the bound ($=$ at $T^\star$).** Set $T^\star(x) := f'(r(x))$. Fenchel–Young holds with equality at $t = f'(u)$, so for every $x$,
$$
f(r(x)) \;=\; r(x)\, T^\star(x) - f^\star(T^\star(x)).
$$
Taking $\mathbb{E}_q$ and using importance weighting once more,
$$
D_f(p \,\|\, q) \;=\; \mathbb{E}_q[r(X)\, T^\star(X)] - \mathbb{E}_q[f^\star(T^\star(X))] \;=\; \mathbb{E}_p[T^\star(X)] - \mathbb{E}_q[f^\star(T^\star(X))]. \;\blacksquare
$$

**Corollary (Variational DRE estimator).** *A sample-based estimator $\hat T$ of $T^\star = f'(r)$ is obtained by maximizing the empirical Lagrangian*
$$
\hat T \;=\; \arg\max_{T \in \mathcal{T}}\, \Biggl\{\, \frac{1}{n_p} \sum_{j=1}^{n_p} T(X_j^p) \;-\; \frac{1}{n_q} \sum_{i=1}^{n_q} f^\star(T(X_i^q)) \,\Biggr\},
$$
*and the density-ratio estimator is $\hat r(x) = (f')^{-1}(\hat T(x))$, where $\mathcal{T}$ is any function class — linear-in-basis, kernel-RKHS, neural network, anything we can optimize over.*

The framework's flexibility is what makes §8.3 and §8.4 possible: $\mathcal{T}$ can be a small MLP, and the same empirical objective trains by SGD without any change to the underlying mathematics.

### §8.2 Recovering LSIF and KLIEP

Two short verifications that the NWJ framework subsumes the Bregman-DRE machinery of §3.

**Chi-squared $\Rightarrow$ LSIF.** Take $f(u) = (u - 1)^2 / 2$ (so $D_f = \tfrac{1}{2} \chi^2(p \,\|\, q)$). From the table, $f'(u) = u - 1$ and $f^\star(t) = t^2 / 2 + t$. The NWJ objective is
$$
\sup_T \, \bigl\{ \mathbb{E}_p[T(X)] - \mathbb{E}_q[T(X)^2 / 2 + T(X)] \bigr\}.
$$
Reparametrize $g(x) = T(x) + 1$ (so $T = g - 1$, with $T^\star = f'(r) = r - 1$ corresponding to $g^\star = r$ — the natural variable). Substituting and expanding,
$$
\mathbb{E}_p[g - 1] - \mathbb{E}_q\!\bigl[\tfrac{1}{2}(g - 1)^2 + (g - 1)\bigr] \;=\; \mathbb{E}_p[g] - \tfrac{1}{2}\mathbb{E}_q[g^2] \;-\; \underbrace{(1/2 + \mathbb{E}_p[1] - \mathbb{E}_q[1])}_{= 1/2},
$$
so $\sup_g \bigl\{ \mathbb{E}_p[g] - \tfrac{1}{2} \mathbb{E}_q[g^2] \bigr\} = D_f(p \,\|\, q) + \tfrac{1}{2}$ — which up to the additive $1/2$ is exactly $-J_{\mathrm{LSIF}}(g)$ from §3.2. Maximizing the NWJ lower bound is equivalent to minimizing the LSIF objective.

**KL $\Rightarrow$ KLIEP.** Take $f(u) = u \log u$ (so $D_f = \mathrm{KL}(p \,\|\, q)$). From the table, $f'(u) = \log u + 1$ and $f^\star(t) = e^{t - 1}$. The NWJ objective is
$$
\sup_T \, \bigl\{ \mathbb{E}_p[T(X)] - \mathbb{E}_q[e^{T(X) - 1}] \bigr\}.
$$
Reparametrize $g(x) = e^{T(x) - 1}$ (so $T = \log g + 1$, with $g^\star = r$ corresponding to $T^\star = \log r + 1 = f'(r)$). Then
$$
\mathbb{E}_p[\log g + 1] - \mathbb{E}_q[g] \;=\; 1 + \mathbb{E}_p[\log g] - \mathbb{E}_q[g],
$$
so $\sup_g \bigl\{ \mathbb{E}_p[\log g] - \mathbb{E}_q[g] \bigr\} = D_f(p \,\|\, q) - 1$ — and the inner expression is $-J^{\mathrm{unc}}_{\mathrm{KLIEP}}(g)$ from §3.3. NWJ with $f = $ KL recovers the unconstrained KLIEP objective.

The two recoveries make explicit what the Bregman framework of §3 promised: the same loss family that produced the kernel-basis estimators of §5 and §6 is the NWJ family in disguise, with the witness function $T$ playing the role of $f'(g)$. The Bregman framing emphasizes the relationship between $g$ and the underlying convex generator; the NWJ framing emphasizes the variational structure that makes the same objective compatible with arbitrary function classes for $T$ (or $g$). Both views are useful — the rest of the topic uses whichever framing is more direct for the question at hand.

### §8.3 Neural DRE — parametrize the witness with an MLP

The NWJ corollary's "any function class $\mathcal{T}$" is the operational handle for neural DRE. Pick a small multilayer perceptron $T_\theta: \mathbb{R}^d \to \mathbb{R}$, train by SGD on the empirical NWJ lower bound for some $f$, recover $\hat r(x) = (f')^{-1}(T_\theta(x))$. For our running 1-D toy with $n_p = n_q = 300$, an MLP with two hidden layers of $32$ ReLU units (≈$1{,}100$ parameters) and $\sim 1{,}500$ Adam steps converges to a fit competitive with uLSIF, in a couple of seconds on CPU.

We use the KL generator $f(u) = u \log u$ for the demonstration, because the KL ratio is the most common neural-DRE target (mutual-information estimation in MINE — Belghazi et al. 2018 — uses essentially the same objective). The empirical NWJ objective to maximize is
$$
\hat L_{\mathrm{NWJ-KL}}(\theta) \;=\; \frac{1}{n_p} \sum_{j=1}^{n_p} T_\theta(X_j^p) \;-\; \frac{1}{n_q} \sum_{i=1}^{n_q} \exp\!\bigl(T_\theta(X_i^q) - 1\bigr),
$$
and the predicted ratio is $\hat r(x) = \exp(T_\theta(x) - 1)$. As the optimization converges, $\hat L_{\mathrm{NWJ-KL}}(\theta)$ approaches $\mathrm{KL}(p \,\|\, q) = (\mu_p - \mu_q)^2 / 2 = 0.5$ — the closed-form benchmark for our shift-Gaussian toy.

**Practical caveats.** (i) The exponential in $f^\star$ can overflow on rare $X_i^q$ values where $T_\theta(X_i^q)$ becomes large; we clip the argument at a safe upper bound. (ii) Full-batch gradients are fine at $n = 300$; for larger $n$, mini-batches with batch size $\geq 128$ are needed for stable estimates of both expectations. (iii) Adam with $\mathrm{lr} = 10^{-3}$ and no scheduling is robust; SGD requires more careful tuning. (iv) The MLP has no inductive bias toward the actual closed-form $r(x) = e^{1/2 - x}$ — it could learn arbitrary 1-D functions — yet the NWJ loss reliably steers it toward the truth.

**Viz: neural DRE training and fit.**
- *Shows:* two-panel figure. (Left) NWJ lower-bound objective vs SGD iteration, with horizontal reference line at the closed-form $\mathrm{KL}(p \,\|\, q) = 0.5$. (Right) the neural-DRE $\hat r_{\mathrm{NN}}(x) = \exp(T_\theta(x) - 1)$ overlaid on $\hat r_{\mathrm{uLSIF}}$ from §6 and the closed-form $r(x) = e^{1/2 - x}$ on a log-$y$ axis.
- *Reader controls (interactive site version):* MLP width (16 / 32 / 64) and depth (1 / 2 / 3 layers), learning rate, $f$-divergence choice (KL / chi-squared / Jensen-Shannon), batch size, $\mu_q$ shift slider.
- *What the reader should learn:* (i) the NWJ objective converges within $\sim 1{,}000$ iterations to its closed-form ceiling; (ii) the resulting $\hat r_{\mathrm{NN}}$ is functionally indistinguishable from $\hat r_{\mathrm{uLSIF}}$ on this 1-D toy because both estimators are well-specified — the neural function class is *more* expressive than the kernel basis, but the estimator's small-$n$ variance is comparable; (iii) the neural approach scales to high-$d$ situations where the kernel-basis methods of §4–§7 degrade, which is the practical reason to bring SGD machinery to bear.

**Code experiment.**
- Reuse the §6 samples ($n_p = n_q = 300$) for direct fit comparison.
- Define `WitnessMLP(d_in=1, hidden=32, depth=2)`; instantiate, Adam optimizer at $\mathrm{lr} = 10^{-3}$.
- Run $1{,}500$ iterations of full-batch gradient ascent on the empirical NWJ-KL objective; record the objective every iteration.
- After training, evaluate $T_\theta$ on the $x$-grid, compute $\hat r_{\mathrm{NN}}(x) = \exp(T_\theta(x) - 1)$, plot the two-panel figure.
- Validate: NWJ objective converges to within $\sim 0.05$ of $\mathrm{KL}(p \,\|\, q) = 0.5$; Pearson($\hat r_{\mathrm{NN}}, \mathrm{true}\ r$) $> 0.95$ on the grid; Pearson($\hat r_{\mathrm{NN}}, \hat r_{\mathrm{uLSIF}}$) $> 0.95$.

### §8.4 GAN as implicit density-ratio estimation

§8.4 closes the section with a conceptual observation that's surprisingly recent given the GAN literature's age: vanilla generative adversarial networks (Goodfellow et al. 2014) are *implicitly* doing density-ratio estimation. The discriminator is the ratio estimator; the generator is updated to push the ratio toward $1$.

The vanilla GAN objective is the minimax
$$
\min_G\; \max_D\; V(G, D) \;:=\; \mathbb{E}_{x \sim p_{\mathrm{data}}}[\log D(x)] \;+\; \mathbb{E}_{x \sim p_G}[\log(1 - D(x))],
$$
where $p_{\mathrm{data}}$ is the true data distribution and $p_G$ is the distribution induced by pushing latent samples $z \sim p_z$ through the generator $G$. The discriminator $D: \mathcal{X} \to (0, 1)$ outputs a probability that $x$ came from the data rather than the generator.

For fixed $G$, the inner maximum over $D$ is achieved at
$$
D^\star(x) \;=\; \frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x) + p_G(x)}.
$$
This is the Bayes-optimal probabilistic classifier on the pooled labelled dataset $\{(x \sim p_{\mathrm{data}}, Y = 1), (x \sim p_G, Y = 0)\}$ with balanced priors. By the §7.2 classification-DRE identity (with $n_p = n_q$ so the prior-odds = 1),
$$
\frac{p_{\mathrm{data}}(x)}{p_G(x)} \;=\; \frac{D^\star(x)}{1 - D^\star(x)}, \qquad D^\star(x) \;=\; \sigma\!\bigl(\log r(x)\bigr).
$$
So the discriminator implements the sigmoid of the log-ratio: $D^\star = \sigma \circ \log r$, equivalently $r = D^\star / (1 - D^\star)$. The GAN apparatus *is* a classification-DRE machine. Mohamed and Lakshminarayanan (2016) made this explicit and built the "implicit generative models" framework around it.

**$f$-GAN.** Nowozin, Cseke, and Tomioka (2016) extended the construction to arbitrary $f$-divergences by replacing the vanilla GAN's discriminator loss with the NWJ lower bound from §8.1:
$$
\min_G\; \max_T\; \bigl\{ \mathbb{E}_{x \sim p_{\mathrm{data}}}[T(x)] \;-\; \mathbb{E}_{x \sim p_G}[f^\star(T(x))] \bigr\}.
$$
The inner maximum recovers $T^\star = f'(r)$ — that is, the $f$-GAN discriminator is an estimator of the ratio's $f$-derivative, and the generator is updated to minimize the corresponding $f$-divergence between $p_{\mathrm{data}}$ and $p_G$. Vanilla GAN (Jensen–Shannon proxy), Wasserstein GAN (a related but distinct construction), and least-squares GAN (chi-squared) are members of this family.

**Why this matters for §10–§12.** Many modern generative models — GANs, normalizing flows trained adversarially, score-based diffusion models with auxiliary discriminators — have discriminator components that double as ratio estimators. For downstream uses that need $r$ explicitly (§9 covariate-shift correction, §10 conformal prediction under shift), these pretrained discriminators are a usable starting point. The §12 practical considerations section returns to this: when the available compute budget exceeds what kernel methods need, neural classifier-DRE and adversarially-trained discriminator-DRE become the practitioner defaults.

**Viz: GAN discriminator as ratio estimator.**
- *Shows:* two-panel figure. (Left) the trained discriminator $D(x)$ overlaid on the closed-form Bayes-optimal $D^\star(x) = p(x) / (p(x) + q(x))$ on the running $x$-grid. (Right) the recovered ratio $\hat r(x) = D(x) / (1 - D(x))$ overlaid on the closed-form $r(x) = e^{1/2 - x}$ on a log-$y$ axis.
- *Reader controls (interactive site version):* the shift $\mu_q$, the discriminator MLP width, the number of BCE training iterations, a toggle for "show generator update step" that runs an additional one-parameter $G$ update loop to drive $\hat r \to 1$.
- *What the reader should learn:* (i) trained on fixed $(p, q)$ data with BCE, the discriminator's sigmoid output is indistinguishable from $D^\star$ within a few hundred iterations; (ii) inverting via $\hat r = D / (1 - D)$ recovers the true ratio to high precision; (iii) in a full GAN, $G$ would alternate to push $\hat r$ toward $1$ — this demo isolates the inner-$D$ DRE step.

**Code experiment.**
- Pool $n = 500$ samples from $p = \mathcal{N}(0, 1)$ (label $1$) and $q = \mathcal{N}(1, 1)$ (label $0$) into a BCE binary-classification problem.
- Train the `WitnessMLP` from §8.3 (reused as the discriminator) for $1{,}000$ Adam iterations on the pooled BCE loss.
- Evaluate $D(x)$ on the $x$-grid; recover $\hat r(x) = D(x) / (1 - D(x))$; compare to the closed-form $D^\star$ and $r$.
- Plot the two panels described above; print the Pearson correlation between $\mathrm{logit}(D)$ and the closed-form $\log r$.
- Validate: $\mathrm{Pearson}(\mathrm{logit}(D), \log r) > 0.98$.


## §9. Covariate-shift correction

§9 turns the estimator machinery of §4–§8 onto the canonical downstream application: **covariate-shift correction**, in which training and test data come from distributions with the same conditional $p(y \mid x)$ but different marginals $p_{\text{train}}(x) \ne p_{\text{test}}(x)$. Shimodaira (2000) showed that under that assumption, the test-data risk minimizer is recovered asymptotically by *reweighting* the training loss by $r(x) = p_{\text{test}}(x) / p_{\text{train}}(x)$. The DRE estimators of §4–§8 produce $\hat r$ from a labelled train set and an unlabelled test set — exactly the data we typically have — closing a loop that's been open since §1.

This section states the assumption precisely, proves the IW-ERM consistency theorem (Shimodaira's central claim), characterizes the variance inflation that limits how aggressively we can reweight, and runs a clean misspecified-linear-regression demo on the running shift toy: train data drawn from $\mathcal{N}(1, 1)$, test data drawn from $\mathcal{N}(0, 1)$, true conditional $y = x^2 + \varepsilon$, fit by a linear model. The train-optimal and test-optimal linear fits differ dramatically (slope 2 vs slope 0) and IW correction with $\hat r$ from uLSIF recovers near-oracle test MSE.

### §9.1 The covariate-shift assumption

Train and test data are drawn from joint distributions $p_{\text{train}}(x, y)$ and $p_{\text{test}}(x, y)$ on $\mathcal{X} \times \mathcal{Y}$. The **covariate-shift assumption** is that the conditional distribution of the label given the features is the same on both sides:
$$
\boxed{\;p_{\text{train}}(y \mid x) \;=\; p_{\text{test}}(y \mid x) \;=:\; p(y \mid x), \quad \text{but} \quad p_{\text{train}}(x) \;\ne\; p_{\text{test}}(x).\;}
$$
Equivalently, the joint factorizes as $p_{\text{train}}(x, y) = p_{\text{train}}(x)\, p(y \mid x)$ and $p_{\text{test}}(x, y) = p_{\text{test}}(x)\, p(y \mid x)$, sharing the conditional and differing only in the input marginal.

This is the load-bearing assumption: it says the *mechanism* by which $y$ is generated from $x$ is invariant across train and test, and only the input distribution shifts. Real-world settings that approximately satisfy it include selection bias in clinical trials (the patient mechanism producing outcomes given symptoms is invariant, but the trial-eligibility filter shifts the marginal of symptoms), domain adaptation in vision (the underlying object/label relationship is fixed, but the imaging conditions shift the pixel marginal), and active learning where a query strategy shifts the input distribution but not the labelling oracle. Settings where the assumption fails — label shift ($p_{\text{train}}(y) \ne p_{\text{test}}(y)$ with class-conditional features shared), concept drift ($p(y \mid x)$ itself changes over time), or domain generalization (no train-test correspondence in $x$ either) — need different machinery; we don't address them here, but the §13 forward pointers list the canonical references.

### §9.2 Importance-weighted ERM

Under covariate shift, the test risk of a predictor $f$ admits a reweighting in terms of train-distribution expectations.

**Theorem (Importance-Weighted Risk Identity).** *Suppose covariate shift holds and $p_{\text{test}}(x) \ll p_{\text{train}}(x)$, with ratio $r(x) = p_{\text{test}}(x) / p_{\text{train}}(x)$. For any loss $L: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}$ and predictor $f: \mathcal{X} \to \mathcal{Y}$ with $\mathbb{E}_{\text{train}}|r(X)\, L(f(X), Y)| < \infty$,*
$$
R_{\text{test}}(f) \;:=\; \mathbb{E}_{(X, Y) \sim p_{\text{test}}}[L(f(X), Y)] \;=\; \mathbb{E}_{(X, Y) \sim p_{\text{train}}}\bigl[r(X)\, L(f(X), Y)\bigr].
$$

*Proof.* By the covariate-shift assumption and the §2.2 importance-weighting identity,
$$
R_{\text{test}}(f) \;=\; \int L(f(x), y)\, p(y \mid x)\, p_{\text{test}}(x)\, \mathrm{d}x\, \mathrm{d}y \;=\; \int L(f(x), y)\, p(y \mid x)\, r(x)\, p_{\text{train}}(x)\, \mathrm{d}x\, \mathrm{d}y.
$$
The integrand factorizes back into $r(x)\, L(f(x), y)\, p_{\text{train}}(x, y)$, and integrating gives the right-hand side. $\blacksquare$

The IW identity converts a test-distribution risk into a train-distribution expectation — exactly the kind of quantity we can estimate from labelled training data, provided we know (or can estimate) $r$. The natural finite-sample estimator is the **importance-weighted empirical risk**
$$
\hat R_{\text{IW}}(f) \;:=\; \frac{1}{n_{\text{train}}} \sum_{i=1}^{n_{\text{train}}} r(X_i^{\text{train}})\, L(f(X_i^{\text{train}}), Y_i^{\text{train}}),
$$
and the corresponding **IW-ERM** estimator $\hat f_{\text{IW}} := \arg\min_f \hat R_{\text{IW}}(f)$, taken over a hypothesis class $\mathcal{F}$.

**Theorem (Shimodaira 2000, IW-ERM Consistency under Misspecification).** *Let $\{f_\theta : \theta \in \Theta\}$ be a parametric hypothesis class with $\Theta$ compact, and suppose the model is misspecified — that is, $f^\star := \arg\min_f R_{\text{test}}(f)$ over all measurable $f$ is not in the parametric family. Under standard regularity conditions on $L$ and the family $\{f_\theta\}$ (continuous differentiability, identifiability of the population argmin, dominated-convergence-applicable envelope), the IW-ERM estimator*
$$
\hat\theta_{\text{IW}}^{(n)} \;:=\; \arg\min_{\theta \in \Theta}\, \frac{1}{n} \sum_{i=1}^n r(X_i^{\text{train}})\, L(f_\theta(X_i^{\text{train}}), Y_i^{\text{train}})
$$
*converges in probability to $\theta_{\text{test}}^\star := \arg\min_{\theta \in \Theta} R_{\text{test}}(f_\theta)$ as $n \to \infty$, the parameter that minimizes test risk within the family. The unweighted ERM $\hat\theta_n$ converges instead to $\theta_{\text{train}}^\star := \arg\min_{\theta} R_{\text{train}}(f_\theta)$, which differs from $\theta_{\text{test}}^\star$ in general when the model is misspecified.*

*Proof sketch.* By the IW identity, $\mathbb{E}_{\text{train}}[r(X)\, L(f_\theta(X), Y)] = R_{\text{test}}(f_\theta)$ pointwise in $\theta$. A uniform law of large numbers on the compact $\Theta$ (under the regularity conditions) gives $\sup_\theta |\hat R_{\text{IW}}(f_\theta) - R_{\text{test}}(f_\theta)| \to 0$ in probability. Continuity of $R_{\text{test}}(f_\theta)$ in $\theta$ and identifiability of the argmin then deliver $\hat\theta_{\text{IW}}^{(n)} \to \theta_{\text{test}}^\star$ in probability via the standard M-estimator consistency argument (van der Vaart 1998, Theorem 5.7). The unweighted statement is the same theorem applied without the $r$ weighting. $\blacksquare$

The misspecification clause matters: when the model is *well-specified* — when $\{f_\theta\}$ contains $f^\star$ — the unweighted and IW-ERM estimators have the same population minimizer (namely $f^\star$), and reweighting only inflates variance without changing the limit. The IW correction is doing work only in the misspecified case, which is the case that matters in practice.

### §9.3 Variance inflation and effective sample size

The IW-ERM estimator's asymptotic correctness comes at a variance cost. For a fixed predictor $f$, the variance of the empirical IW estimator of $R_{\text{test}}(f)$ is
$$
\mathrm{Var}\bigl(\hat R_{\text{IW}}(f)\bigr) \;=\; \frac{1}{n_{\text{train}}}\, \mathrm{Var}_{\text{train}}\bigl(r(X)\, L(f(X), Y)\bigr).
$$
Compare to the unweighted-but-target-misaligned estimator $\hat R_{\text{train}}(f) = (1/n_{\text{train}}) \sum L(f(X_i^{\text{train}}), Y_i^{\text{train}})$, which has variance $(1/n_{\text{train}})\, \mathrm{Var}_{\text{train}}(L(f(X), Y))$. The variance ratio
$$
\frac{\mathrm{Var}_{\text{train}}(r(X)\, L)}{\mathrm{Var}_{\text{train}}(L)} \;\approx\; \mathbb{E}_{\text{train}}[r(X)^2] \;=\; \chi^2(p_{\text{test}} \,\|\, p_{\text{train}}) + 1
$$
when $L$ is approximately uncorrelated with $r^2$, so the IW estimator's variance inflation factor is governed by the chi-squared divergence between the two marginals — the same quantity we met in §2.3 controlling importance-sampling variance.

The operational diagnostic is the **effective sample size** of the empirical weights:
$$
n_{\text{eff}} \;:=\; \frac{\bigl( \sum_i w_i \bigr)^2}{\sum_i w_i^2}, \qquad w_i = \hat r(X_i^{\text{train}}),
$$
introduced in §2.3 and revisited here as a runtime check. $n_{\text{eff}}$ ranges from $1$ (one weight dominates) to $n_{\text{train}}$ (all weights equal). When $n_{\text{eff}} \ll n_{\text{train}}$, the IW estimator's variance is dominated by a few high-weight samples; despite asymptotic correctness, the finite-sample bias-variance trade-off may favour the unweighted estimator at small $n$. Two common practical adjustments:

- **Truncated importance weights:** replace $w_i$ with $\min(w_i, B)$ for some upper cap $B$. This trades bias (the truncated weights underestimate the test risk in the high-$r$ region) for variance. Sugiyama, Krauledat, and Müller (2007) use this routinely.
- **Self-normalized importance weights:** replace the estimator with $\hat R_{\text{IW}}^{\text{SN}}(f) := (\sum_i w_i\, L_i) / (\sum_i w_i)$. Self-normalization introduces $O(1/n_{\text{train}})$ bias (Owen 2013, §9.2) but typically reduces variance by a constant factor, often substantially.

In §9.4's demo we use plain IW weights without truncation or self-normalization to keep the comparison clean, and report $n_{\text{eff}}$ as a diagnostic.

### §9.4 Numerical demonstration

The §9.4 cell sets up a misspecified linear-regression problem under covariate shift, fits four estimators, and compares their test MSE.

**The DGP.** True conditional $y = x^2 + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, 0.3^2)$. Input marginals: $p_{\text{train}}(x) = \mathcal{N}(1, 1)$, $p_{\text{test}}(x) = \mathcal{N}(0, 1)$ — the same shift-Gaussian pair from §1, with the role labels flipped so that "test" plays the role of "p" in our DRE convention. The closed-form ratio is $r(x) = e^{1/2 - x}$ exactly as before.

**The misspecification.** We fit a linear model $\hat f(x) = a + b x$. The closed-form best-linear-approximation under each marginal can be computed analytically: under $\mathcal{N}(\mu, 1)$ with target $y = x^2$, the optimal coefficients are $b^\star = 2\mu$, $a^\star = 1 - \mu^2$. So $(a^\star_{\text{train}}, b^\star_{\text{train}}) = (0, 2)$ and $(a^\star_{\text{test}}, b^\star_{\text{test}}) = (1, 0)$. The unweighted train fit converges to slope $2$, intercept $0$; the test-oracle fit converges to slope $0$, intercept $1$. IW correction is supposed to move us from the former to the latter.

**Viz: covariate-shift fits and test MSE.**
- *Shows:* two-panel figure. (Left) the $(x, y)$ training scatter overlaid with four linear fits — unweighted, IW with closed-form $r$, IW with uLSIF $\hat r$, and a test-oracle fit — plus the true $y = x^2$ curve. Shaded bands at the bottom indicate the train and test $x$-distributions for context. (Right) bar chart of test MSE for the four estimators with numerical annotations.
- *Reader controls (interactive site version):* train sample size $n_{\text{train}} \in [100, 2000]$ (slider), shift severity $\mu_{\text{train}}$, noise level $\sigma_\varepsilon$, a toggle for self-normalized vs plain IW.
- *What the reader should learn:* (i) the unweighted slope-$2$ fit visibly mis-tracks the test region of the $x$-axis; (ii) the IW fit (true $r$) lies essentially on the test-oracle line; (iii) the IW fit with the uLSIF-estimated ratio is slightly noisier than the true-$r$ fit but still substantially better than unweighted; (iv) the bar chart makes the gap quantitative — unweighted is $\sim 3\times$ worse than IW or the test oracle on this DGP.

**Code experiment.**
- Generate $n_{\text{train}} = 500$ pairs $(x, y)$ from $p_{\text{train}} \times p(y \mid x)$ and $n_{\text{test}} = 5{,}000$ from $p_{\text{test}} \times p(y \mid x)$.
- Estimate $\hat r$ via `ulsif_fit_at` (from §6.4) using $x_{\text{test}}$ as the "$p$" target and $x_{\text{train}}$ as the "$q$" source, at the §6.4 hyperparameters $(\sigma^\star_u, \lambda^\star_u)$.
- Fit four weighted-least-squares linear models via closed-form normal equations: unweighted, weights $r(X_i^{\text{train}})$ (true), weights $\hat r(X_i^{\text{train}})$ (uLSIF), and unweighted on the test set (oracle).
- Evaluate each fit's MSE on the held-out test set.
- Plot the two panels described above; print the table of $(a, b, \mathrm{MSE})$ for each fit and the empirical $n_{\text{eff}}$ for both IW estimators.
- Validate: unweighted test MSE is at least $2\times$ the IW-with-true-$r$ test MSE; IW-with-uLSIF test MSE is within $10\%$ of IW-with-true-$r$; both IW estimators have slopes within $0.3$ of the test-oracle slope of $0$.

---

## §10. Conformal prediction under covariate shift

§10 carries the §9 covariate-shift application one step further: instead of producing a *point* predictor with calibrated risk, we produce a *prediction interval* with calibrated coverage. Vanilla split conformal prediction (Vovk, Gammerman, and Shafer 2005; Lei et al. 2018) gives a finite-sample, distribution-free coverage guarantee — but only under exchangeability of train, calibration, and test data, which covariate shift breaks. Tibshirani, Barber, Candès, and Ramdas (2019) showed that *weighted* exchangeability survives under covariate shift, with weights proportional to $r(x) = p_{\text{test}}(x) / p_{\text{train}}(x)$, and that the resulting **weighted split conformal** algorithm recovers nominal coverage as long as $r$ is known or well-estimated. The DRE estimators of §4–§8 provide $\hat r$; plugging it into the TBCR algorithm closes the loop from "raw shifted data" to "valid prediction intervals under shift" with no additional machinery.

This section defines weighted exchangeability rigorously, states and sketches the TBCR coverage theorem, characterizes the coverage gap that arises when we substitute $\hat r$ for the true $r$, and runs the algorithm on the §9 shift toy: training a misspecified linear predictor, calibrating it by holding out a fold, then constructing intervals on a fresh test set under three weighting schemes — none (vanilla conformal), true $r$, and uLSIF $\hat r$.

### §10.1 Weighted exchangeability and the conformal quantile

We work in the split-conformal setting. Given a labelled training fold and a labelled **calibration fold** of size $m$ from the same distribution, plus a fresh test point $(X_{m+1}, Y_{m+1})$ — possibly from a different distribution — split conformal asks: under what conditions on the joint distribution can we build a finite-sample prediction interval with provable coverage?

The classical answer is **exchangeability**. A sequence $Z_1, \ldots, Z_n$ is exchangeable if its joint distribution is invariant under permutations, i.e., $(Z_1, \ldots, Z_n) \stackrel{d}{=} (Z_{\sigma(1)}, \ldots, Z_{\sigma(n)})$ for every permutation $\sigma$. The iid case is a special case (with the joint factorizing as a product); under exchangeability of $(Z_1, \ldots, Z_{m+1})$, the rank of any equivariant statistic $V_i$ at the test position $m+1$ is uniform on $\{1, \ldots, m+1\}$. That uniformity is what produces the vanilla split-conformal coverage guarantee.

Under covariate shift, $(X_i, Y_i)_{i=1}^m$ are iid from $P_{\text{train}}$ but $(X_{m+1}, Y_{m+1})$ is from $P_{\text{test}}$, so the joint is *not* exchangeable. It is, however, **weighted exchangeable** in the precise sense of TBCR's Definition 1.

**Definition (Weighted Exchangeability; TBCR 2019, Definition 1).** *Random variables $Z_1, \ldots, Z_n$ with joint density $f$ are weighted exchangeable with weight functions $w_1, \ldots, w_n: \mathcal{Z} \to \mathbb{R}_{\geq 0}$ if there exists a permutation-invariant function $g: \mathcal{Z}^n \to \mathbb{R}_{\geq 0}$ such that*
$$
f(z_1, \ldots, z_n) \;=\; \biggl(\prod_{i=1}^{n} w_i(z_i)\biggr) \cdot g(z_1, \ldots, z_n).
$$

Exchangeability is the special case $w_i \equiv 1$ (with $g$ itself the joint density). In the covariate-shift setting, $(Z_1, \ldots, Z_m, Z_{m+1})$ with $Z_i = (X_i, Y_i)$ has joint density
$$
f(z_1, \ldots, z_{m+1}) \;=\; \prod_{i=1}^m p_{\text{train}}(z_i) \cdot p_{\text{test}}(z_{m+1}) \;=\; r(x_{m+1}) \cdot \prod_{i=1}^{m+1} p_{\text{train}}(z_i),
$$
the second equality by $p_{\text{test}}(z) = r(x)\, p_{\text{train}}(z)$ under the covariate-shift conditional-invariance. Setting $g(z_1, \ldots, z_{m+1}) = \prod_{i=1}^{m+1} p_{\text{train}}(z_i)$ (which is permutation-symmetric in its arguments) and $w_i(z) = 1$ for $i \leq m$, $w_{m+1}(z) = r(x)$ exhibits the factorization. So the calibration plus test data are weighted exchangeable with weights "1 for calibration points, $r$ for the test point."

**Lemma (Weighted permutation; TBCR 2019, Lemma 3).** *Let $Z_1, \ldots, Z_n$ be weighted exchangeable with weights $w_1, \ldots, w_n$, and let $V: \mathcal{Z}^n \to \mathbb{R}$ be a permutation-invariant function. For any $i \in \{1, \ldots, n\}$ and any measurable set $A$,*
$$
\mathbb{P}(V(Z_{(i)}, Z_{-i}) \in A) \;=\; \mathbb{E}\!\left[\, \frac{w_i(Z_{(i)})}{\sum_{j=1}^n w_j(Z_{(j)})} \cdot \mathbb{1}\{V(Z) \in A\} \,\right],
$$
*where $Z_{(i)}$ is the value taking the $i$-th position after some permutation.*

This lemma says that when we apply a permutation-invariant statistic to a weighted-exchangeable sequence, the marginal distribution at position $i$ is a *weighted average* of the joint statistic, with weights proportional to $w_i$ at each candidate position. For the conformal application, $V$ is the empirical CDF of conformity scores and the weights pick out the contribution of each calibration point's residual.

### §10.2 The weighted split-conformal algorithm

The TBCR algorithm modifies the vanilla split-conformal procedure (Lei et al. 2018) at exactly one place — the quantile computation — to incorporate the weights.

**Algorithm (Weighted Split Conformal; TBCR 2019, Algorithm 1).**
1. Split labelled $P_{\text{train}}$ data into training and calibration folds of sizes $n_{\text{tr}}$ and $m$.
2. Train predictor $\hat\mu$ on the training fold (using IW-ERM per §9, or any method).
3. Compute calibration residuals $R_i := |Y_i - \hat\mu(X_i)|$ for $i = 1, \ldots, m$.
4. For each new test point $X_{m+1} = x^\star$:
    1. Compute weights $w_i := \hat r(X_i)$ for $i = 1, \ldots, m$ and $w_{m+1} := \hat r(x^\star)$.
    2. Normalize: $\tilde p_i := w_i / \sum_{j=1}^{m+1} w_j$ for $i = 1, \ldots, m+1$.
    3. Define the weighted empirical CDF $\tilde F(t) := \sum_{i=1}^m \tilde p_i \cdot \mathbb{1}\{R_i \leq t\} + \tilde p_{m+1} \cdot \mathbb{1}\{+\infty \leq t\}$, treating the unobserved test-point residual as $+\infty$.
    4. The weighted conformal quantile is $\hat q(x^\star) := \inf\{ t : \tilde F(t) \geq 1 - \alpha \}$.
    5. The prediction set is $\hat C(x^\star) := \bigl\{ y : |y - \hat\mu(x^\star)| \leq \hat q(x^\star) \bigr\}$.

The vanilla split-conformal special case is recovered when all $w_i \equiv 1$, in which case $\tilde p_i = 1/(m+1)$ and $\hat q$ is just the $\lceil (m+1)(1-\alpha) \rceil$-th order statistic of $\{R_1, \ldots, R_m\}$ (the $+\infty$ entry from the test point gets included so the index lands at an attainable rank). Under covariate shift, the unequal weights reweight which calibration residuals "count" — points near the test region get more weight, points far from it get less — and the resulting quantile adapts to the test distribution.

**Theorem (Weighted Conformal Coverage; TBCR 2019, Theorem 2).** *Suppose $(X_i, Y_i)_{i=1}^m$ are iid from $P_{\text{train}}$, $(X_{m+1}, Y_{m+1})$ is independently from $P_{\text{test}}$, the covariate-shift assumption holds, and $P_{\text{test}} \ll P_{\text{train}}$ with ratio $r$. If we run the Algorithm 1 above using the **true** $r$ for the weights, the resulting prediction set $C(X_{m+1})$ satisfies*
$$
\mathbb{P}\bigl(Y_{m+1} \in C(X_{m+1})\bigr) \;\geq\; 1 - \alpha.
$$
*The guarantee is finite-sample, distribution-free over the marginal $P_{\text{train}}$, and depends only on the iid-ness within the calibration fold and the covariate-shift assumption linking test to train.*

*Proof sketch.* The conformity score $V_i = R_i = |Y_i - \hat\mu(X_i)|$ is permutation-invariant in its construction. The calibration-plus-test sequence $(Z_1, \ldots, Z_{m+1})$ is weighted exchangeable with weights $w_i = 1$ (calibration) and $w_{m+1} = r(x_{m+1})$ (test). The weighted-permutation lemma above implies that the rank of the test residual $R_{m+1}$ among $\{R_1, \ldots, R_m, R_{m+1}\}$ has the *weighted* discrete distribution with probabilities $\tilde p_i$ — exactly the distribution that $\tilde F$ approximates. The quantile $\hat q$ is then constructed so that, with probability at least $1 - \alpha$, $R_{m+1} \leq \hat q$, which is the event $Y_{m+1} \in C(X_{m+1})$. The full proof (TBCR 2019, Appendix A) handles the edge cases — ties in the residuals, the convention that the test point's residual is treated as $+\infty$ — and tightens the inequality to an equality up to a discrete adjustment of order $1/(m+1)$. $\blacksquare$

### §10.3 Coverage validity as a function of DRE accuracy

The TBCR coverage guarantee assumes we use the **true** ratio $r$, which in practice we don't have. The natural substitution is $\hat r$ from any DRE estimator (§4–§8). The coverage gap that this incurs is bounded by a divergence between $P_{\text{test}}$ and the "implicit test distribution" $\hat P_{\text{test}}(z) := \hat r(x) P_{\text{train}}(z)$ that the estimator pretends is the test distribution.

**Theorem (Approximate Coverage with Estimated Ratio; TBCR 2019, Theorem 3, paraphrased).** *Under the §10.2 setup with weights computed from an estimate $\hat r$ instead of the true $r$, the resulting prediction set $\hat C(X_{m+1})$ satisfies*
$$
\bigl|\, \mathbb{P}(Y_{m+1} \in \hat C(X_{m+1})) \;-\; (1 - \alpha) \,\bigr| \;\leq\; d_{\mathrm{TV}}\bigl(P_{\text{test}}, \hat r \cdot P_{\text{train}}\bigr) \;+\; O(1/m),
$$
*where $d_{\mathrm{TV}}$ is the total variation distance.*

The bound says: the coverage error is governed by how badly $\hat r$ misrepresents the test distribution, plus a finite-sample $O(1/m)$ correction. In particular, if $\hat r \to r$ in TV (or any stronger sense) as the DRE training sample grows, the coverage approaches the nominal $1 - \alpha$ at the same rate.

The practical implication is that *the DRE accuracy is the bottleneck*: any of the §4–§8 estimators that fits $r$ accurately will give approximately valid coverage when plugged in. In §10.4 we observe that uLSIF — the practitioner default — produces coverage within a couple of percentage points of nominal at $m = 500$ calibration points and $n_{\text{train}} = 500$ for the DRE estimator, while vanilla unweighted conformal under-covers by roughly $10$ percentage points at the same $\alpha$.

### §10.4 Numerical demonstration

The §10.4 cell reuses the §9 covariate-shift DGP (true conditional $y = x^2 + \varepsilon$; $p_{\text{train}}(x) = \mathcal{N}(1, 1)$; $p_{\text{test}}(x) = \mathcal{N}(0, 1)$), trains a linear predictor by IW-ERM on the training fold, computes residuals on a held-out calibration fold, then evaluates three conformal procedures on a fresh test set:

1. **Vanilla split conformal** (all weights $= 1$): the baseline that ignores the shift.
2. **Weighted conformal with true $r$**: the oracle that the TBCR theorem guarantees achieves nominal coverage.
3. **Weighted conformal with uLSIF $\hat r$**: the practitioner reality, where $\hat r$ comes from §6's uLSIF on $(X_{\text{train}}, X_{\text{test, unlabelled}})$.

We compute the empirical coverage rate on the test set for each procedure at nominal $1 - \alpha \in \{0.80, 0.90, 0.95\}$, and visualize the resulting interval widths along the test $x$-range.

**Viz: weighted conformal coverage under shift.**
- *Shows:* two-panel figure. (Left) the test scatter with the three prediction intervals overlaid as bands $\hat\mu(x) \pm \hat q(x)$ — vanilla, weighted-true, weighted-uLSIF — colour-coded; the closed-form true conditional $y = x^2$ included for reference. (Right) empirical coverage rate vs nominal $1 - \alpha$ for the three procedures, with a $y = x$ diagonal showing perfect calibration.
- *Reader controls (interactive site version):* shift severity $\mu_{\text{train}}$, calibration-fold size $m$, target $\alpha$, DRE method dropdown (uLSIF / KLIEP / classifier-DRE).
- *What the reader should learn:* (i) the vanilla unweighted procedure under-covers by a visible margin (e.g., $\sim 80\%$ when targeting $90\%$); (ii) the weighted-true procedure achieves nominal coverage as the theorem guarantees; (iii) the weighted-uLSIF procedure is within $1$–$3$ percentage points of nominal — the residual gap is the §10.3 DRE-accuracy term; (iv) the interval widths in the left panel show how weighting adapts: the weighted bands are slightly wider in the test region (where unweighted intervals are systematically too short) and slightly narrower elsewhere.

**Code experiment.**
- Generate $n_{\text{train}} = 500$, $n_{\text{cal}} = 500$, $n_{\text{test}} = 2{,}000$ samples per the §9 DGP.
- Train a linear predictor $\hat\mu(x) = \hat a + \hat b x$ by IW-ERM with uLSIF weights on the training fold.
- Compute calibration residuals $R_i = |Y_i - \hat\mu(X_i)|$.
- Fit uLSIF for DRE using $(X_{\text{test, unlabelled}}, X_{\text{train+cal}})$ at the §6.4 hyperparameters.
- Implement `weighted_conformal_quantile(R, w_cal, w_test, alpha)` returning the conformal quantile for each test point.
- Evaluate empirical coverage for each of the three weighting schemes at $\alpha \in \{0.20, 0.10, 0.05\}$; plot the two panels described above.
- Validate: unweighted coverage at $\alpha = 0.10$ is below $0.85$; weighted-true coverage is within $\pm 0.02$ of $0.90$; weighted-uLSIF coverage is within $\pm 0.03$ of $0.90$.

---

## §11. MMD as a special-case kernel ratio

§11 closes the loop on the kernel-methods machinery that §4 (KMM) opened, by examining the **maximum mean discrepancy** — a kernel-embedding distance between distributions that doubles as a two-sample test statistic. The two-sample-testing problem "is $p = q$?" is the special case of the density-ratio problem "what is $r = p/q$?" where we only care whether $r \equiv 1$. Gretton, Borgwardt, Rasch, Schölkopf, and Smola (2012) showed that MMD admits a clean kernel-mean-embedding form, an unbiased sample estimator, and a permutation-based null calibration that yields a finite-sample-valid test. The connection to DRE is precise: KMM (§4) minimizes the squared MMD between the reweighted source and the target to *find* the ratio; the MMD test of Gretton et al. *uses* the same squared MMD as a statistic to *decide* whether the ratio is trivial. The shared kernel-mean-embedding machinery — including the median-heuristic bandwidth and the characteristic-kernel injectivity from §4.1 — carries over verbatim.

This section recasts MMD in the kernel-mean-embedding form, derives the unbiased sample estimator, presents the permutation test with its finite-sample validity, discusses the discriminative-power versus DRE-smoothness trade-off in kernel choice, and runs a permutation test on the running shift toy under both a null (no shift) and a shifted alternative.

### §11.1 MMD definition and kernel-mean-embedding form

**Definition (Maximum Mean Discrepancy; Gretton et al. 2012, Eq. 2).** *For a function class $\mathcal{F}$ and probability distributions $P, Q$ on $\mathcal{X}$, the maximum mean discrepancy is*
$$
\mathrm{MMD}(P, Q; \mathcal{F}) \;:=\; \sup_{f \in \mathcal{F}}\, \bigl|\, \mathbb{E}_P[f(X)] - \mathbb{E}_Q[f(X)] \,\bigr|.
$$
*When $\mathcal{F}$ is the unit ball of an RKHS $\mathcal{H}$ with kernel $k$, the MMD admits a closed kernel-mean-embedding form:*
$$
\mathrm{MMD}(P, Q; \mathcal{H}) \;=\; \bigl\|\, \mu_P - \mu_Q \,\bigr\|_\mathcal{H},
$$
*where $\mu_P = \mathbb{E}_{X \sim P}[k(X, \cdot)]$ is the kernel mean embedding from §4.1.*

*Proof.* By the reproducing property, for $f \in \mathcal{H}$,
$$
\mathbb{E}_P[f(X)] - \mathbb{E}_Q[f(X)] \;=\; \langle f, \mu_P \rangle_\mathcal{H} - \langle f, \mu_Q \rangle_\mathcal{H} \;=\; \langle f, \mu_P - \mu_Q \rangle_\mathcal{H}.
$$
Cauchy-Schwarz gives $|\langle f, \mu_P - \mu_Q \rangle_\mathcal{H}| \leq \|f\|_\mathcal{H} \cdot \|\mu_P - \mu_Q\|_\mathcal{H}$, with equality at $f = (\mu_P - \mu_Q) / \|\mu_P - \mu_Q\|_\mathcal{H}$ (when the denominator is nonzero). The supremum over $\|f\|_\mathcal{H} \leq 1$ is therefore $\|\mu_P - \mu_Q\|_\mathcal{H}$ exactly. $\blacksquare$

The squared MMD admits an expansion in pairwise kernel evaluations that is the basis for both the sample estimator and the connection to KMM:
$$
\mathrm{MMD}^2(P, Q; \mathcal{H}) \;=\; \mathbb{E}_{X, X' \sim P}[k(X, X')] \;-\; 2\, \mathbb{E}_{X \sim P, Y \sim Q}[k(X, Y)] \;+\; \mathbb{E}_{Y, Y' \sim Q}[k(Y, Y')].
$$
This follows from $\|\mu_P - \mu_Q\|^2_\mathcal{H} = \|\mu_P\|^2 - 2 \langle \mu_P, \mu_Q \rangle + \|\mu_Q\|^2$ and the identity $\langle \mu_P, \mu_Q \rangle = \mathbb{E}_{X \sim P, Y \sim Q}[k(X, Y)]$.

**Characteristic kernels and the injectivity that MMD inherits.** Recall from §4.1 that a kernel is **characteristic** if the embedding map $P \mapsto \mu_P$ is injective on the space of probability measures. The Gaussian RBF, Laplacian, and Matérn kernels on $\mathbb{R}^d$ are characteristic (Sriperumbudur et al. 2010). For a characteristic kernel,
$$
\mathrm{MMD}(P, Q; \mathcal{H}) \;=\; 0 \;\iff\; P = Q,
$$
so MMD genuinely separates distinct distributions. This is the property that makes MMD a valid test statistic for $H_0: P = Q$, and it's the same property that makes KMM's population minimum unique at $g = r$ (§4.1).

**Unbiased sample estimator (Gretton et al. 2012, Eq. 5).** Given iid samples $X_1, \ldots, X_n \sim P$ and $Y_1, \ldots, Y_m \sim Q$, the **U-statistic** estimator
$$
\widehat{\mathrm{MMD}}^2_u \;:=\; \frac{1}{n(n - 1)} \sum_{i \ne i'} k(X_i, X_{i'}) \;-\; \frac{2}{n\, m} \sum_{i, j} k(X_i, Y_j) \;+\; \frac{1}{m(m - 1)} \sum_{j \ne j'} k(Y_j, Y_{j'})
$$
is unbiased for $\mathrm{MMD}^2(P, Q)$. Removing the diagonals avoids the $O(1/n)$ bias of the plug-in V-statistic estimator. Computation is $O((n + m)^2)$ kernel evaluations — affordable up to a few thousand samples per side on a laptop.

### §11.2 Two-sample testing as ratio-distinguishability

The hypothesis-testing problem is
$$
H_0: P = Q \quad \text{vs} \quad H_1: P \ne Q,
$$
which under any characteristic kernel is equivalent to $H_0: \mathrm{MMD}^2(P, Q) = 0$ vs $H_1: \mathrm{MMD}^2(P, Q) > 0$. From the DRE perspective, this is the question "is $r \equiv 1$?" — the special case of density-ratio estimation where we don't need to know $r$, just whether it's the trivial ratio.

**The permutation null.** Under $H_0$, the pooled sample $(X_1, \ldots, X_n, Y_1, \ldots, Y_m)$ is iid from the common distribution. The labels assigning each point to "the $X$ sample" vs "the $Y$ sample" are therefore exchangeable. A permutation test exploits this directly:

**Algorithm (Permutation MMD Two-Sample Test).**
1. Pool the data: $Z = (X_1, \ldots, X_n, Y_1, \ldots, Y_m)$ of size $N = n + m$.
2. Compute the observed statistic $T_{\text{obs}} := \widehat{\mathrm{MMD}}^2_u(X, Y)$.
3. For $b = 1, \ldots, B$: draw a uniform random permutation $\pi^{(b)}$ of $\{1, \ldots, N\}$, split as $X^{(b)} = (Z_{\pi^{(b)}(1)}, \ldots, Z_{\pi^{(b)}(n)})$ and $Y^{(b)} = (Z_{\pi^{(b)}(n+1)}, \ldots, Z_{\pi^{(b)}(N)})$, compute $T^{(b)} := \widehat{\mathrm{MMD}}^2_u(X^{(b)}, Y^{(b)})$.
4. The exchangeability-adjusted $p$-value is $\hat p := (1 + \#\{b : T^{(b)} \geq T_{\text{obs}}\}) / (B + 1)$.
5. Reject $H_0$ if $\hat p \leq \alpha$.

**Theorem (Finite-Sample Validity of Permutation Test).** *Under $H_0: P = Q$, the permutation $p$-value $\hat p$ satisfies $\mathbb{P}(\hat p \leq \alpha) \leq \alpha$ for every $\alpha \in (0, 1)$ and every $B \geq 1$. The test controls Type-I error exactly at level $\alpha$ for $\alpha \in \{1/(B+1), 2/(B+1), \ldots\}$.*

*Proof sketch.* Under $H_0$, the labels are exchangeable, so $(T_{\text{obs}}, T^{(1)}, \ldots, T^{(B)})$ are exchangeable; the rank of $T_{\text{obs}}$ among the $B + 1$ statistics is uniform on $\{1, \ldots, B + 1\}$. The event $\hat p \leq \alpha$ corresponds to $T_{\text{obs}}$ being in the top $\lfloor \alpha (B+1) \rfloor$ statistics, which has probability at most $\lfloor \alpha (B+1) \rfloor / (B+1) \leq \alpha$. $\blacksquare$

The permutation test inherits its validity from exchangeability — exactly the same property that vanilla split conformal (§10.1) inherited and that broke under covariate shift. Here we're in the iid-pooled-sample regime, so exchangeability is back, and the test has finite-sample validity without distributional assumptions.

**Why "ratio-distinguishability."** Under $H_1$, $\mathrm{MMD}^2(P, Q) > 0$ measures how far $r = p/q$ is from the constant $1$ — equivalently, how strongly the kernel-mean embedding distinguishes the reweighted vs unweighted source. The same KMM machinery from §4 that *recovers* this ratio when it's non-trivial is what the MMD test *uses* to decide whether estimation is even worth doing. In some pipelines, an MMD pre-test for $r \equiv 1$ is run before any DRE — if the test fails to reject, no reweighting is applied and the unweighted estimator is used directly.

### §11.3 Kernel choice and the discriminative-power-vs-smoothness trade-off

The kernel bandwidth $\sigma_k$ governs both the MMD test's power and the KMM estimator's smoothness, but it does so in opposite directions.

**For two-sample testing.** Small $\sigma_k$ (narrow kernel) gives a witness function class with high resolution but high variance — the test statistic's null distribution has heavy tails and individual sample-pair distances dominate, so power is low for moderate-shift alternatives. Large $\sigma_k$ (wide kernel) gives a smooth witness class but compresses fine-scale differences between $P$ and $Q$, so power is low for high-frequency alternatives. Gretton et al. (2012, §8) recommend choosing $\sigma_k$ to maximize test power on a held-out fold, which typically lands at the **median pairwise distance heuristic** for Gaussian-vs-Gaussian-like alternatives.

**For KMM-style DRE.** Small $\sigma_k$ produces highly localized reweighting that over-fits to individual sample pairings — the recovered $w_i$ are noisy at finite $n$. Large $\sigma_k$ produces smoother reweighting but compresses the high-$r$ region's signal. The KMM bandwidth choice favors slightly **larger $\sigma_k$ than the test-power-optimal value**, because the goal is to estimate a smooth function on $\mathcal{X}$ rather than to maximize a discrimination statistic.

The two-sample-testing optimum and the DRE optimum coincide only when the alternative is "Gaussian-shaped" at the scale of the median heuristic, which is the case for our running shift-Gaussian toy. For pathological alternatives (heavy tails, multimodal $q$, anisotropic shifts) the two optima diverge, and kernel selection must be done with the downstream task in mind. §12 (practical considerations) revisits this.

The full empirical quantification of how the two bandwidth optima diverge is in Sutherland, Tung, Strathmann, De, Ramdas, Smola, and Gretton (2017), "Generative models and model criticism via optimized MMD," which derives the data-dependent kernel-selection objective that maximizes test power and shows it is provably distinct from the median-heuristic optimum on multimodal targets. For the typical DRE practitioner working on tabular shift problems with smooth marginals, the median heuristic remains a sensible default; for two-sample testing on structured data (images, text) the Sutherland et al. selector is the canonical upgrade.

### §11.4 Numerical demonstration

The §11.4 cell runs the MMD permutation test under two scenarios on the running 1-D toy: (i) **null**, where both samples are iid from $\mathcal{N}(0, 1)$, and (ii) **alternative**, where the second sample comes from the shifted $\mathcal{N}(0.5, 1)$. Sample sizes $n = m = 200$, kernel bandwidth from the median heuristic on the pooled sample, $B = 500$ permutations. A third panel computes a small empirical power curve to confirm Type-I control at $\delta = 0$ and rising power as the shift grows.

**Viz: MMD permutation distributions under null, alternative, and a small power curve.**
- *Shows:* three-panel figure. (Left) histogram of permutation-distribution MMD$^2$ values under the null scenario, with the observed statistic marked — the observed value falls well inside the bulk, $p \approx 0.5$. (Middle) the same histogram under the shifted alternative, with the observed value far to the right of the bulk, $p \ll 0.05$. (Right) empirical power curve — rejection rate at $\alpha = 0.05$ vs shift size $\delta \in \{0.0, 0.1, 0.2, 0.5, 1.0\}$, computed by repeating the permutation test $B_{\text{outer}} = 10$ times per shift with $B = 150$ permutations each. The curve rises from near $\alpha = 0.05$ at $\delta = 0$ (Type-I level) toward $1.0$ as $\delta$ grows.
- *Reader controls (interactive site version):* sample size $n = m \in [50, 1000]$, shift $\mu_Q \in [0, 1.5]$, bandwidth multiplier on the median heuristic, number of permutations $B$, $B_{\text{outer}}$ for the power curve.
- *What the reader should learn:* (i) under the null, the observed statistic is one of many similar permutation values; (ii) under even a modest shift, the observed statistic separates from the permutation null cleanly; (iii) the empirical power increases monotonically with shift size, from near $\alpha$ at $\delta = 0$ to near $1$ at $\delta = 1.0$, demonstrating the test's discrimination ability across the shift range.

**Code experiment.**
- Sample $n = m = 200$ from $\mathcal{N}(0, 1)$ for the null scenario, and from $\mathcal{N}(0, 1)$ and $\mathcal{N}(0.5, 1)$ for the alternative.
- Set the kernel bandwidth by the median heuristic on the pooled sample for each scenario.
- Compute the unbiased $\widehat{\mathrm{MMD}}^2_u$ using `cdist`-based Gaussian-kernel evaluation with explicit diagonal-zeroing.
- Run $B = 500$ permutations per scenario, recording each permuted statistic.
- For the power curve: at five shift sizes, repeat $B_{\text{outer}} = 10$ times with $B = 150$, count rejections at $\alpha = 0.05$.
- Plot the three panels with observed statistics; print observed value, permutation-quantile, and $p$-value for each scenario.
- Validate: null $p$-value is $> 0.1$; alternative $p$-value is $< 0.01$; rejection rate at $\delta = 0$ is within $\pm 0.15$ of $\alpha = 0.05$; rejection rate at $\delta = 1.0$ is $\geq 0.9$.

---

## §12. Practical considerations, diagnostics, and pitfalls

§12 brings the four estimator families (§4 KMM, §5 KLIEP, §6 LSIF/uLSIF, §7 classification-DRE) back to the practitioner's bench and asks: what do we tune, what do we monitor, and where does DRE quietly fail in production? The earlier sections derived each estimator carefully and ran a controlled-toy demo; this section consolidates the operational guidance and surfaces a single failure mode that all four estimators share — the curse of dimensionality. The curse-of-dimensionality demo at the end of §12.3 is the topic's most important practical caution; without it, a reader might leave §6 and §8 thinking that "uLSIF" or "neural DRE" solves DRE in general, when in fact both degrade rapidly past $d \approx 20$ in the absence of structural assumptions.

This section is mostly synthesis prose, with one diagnostic figure: the curse-of-dimensionality demo in §12.3 showing uLSIF MSE and effective sample size as functions of input dimension on a controlled isotropic-Gaussian shift.

### §12.1 Bandwidth and basis selection across KLIEP, LSIF, uLSIF

The three kernel-basis estimators (§5 KLIEP, §6 LSIF, §6 uLSIF) all share a Gaussian basis $\boldsymbol{\psi}_\ell(x) = \exp(-\|x - c_\ell\|^2 / (2 \sigma^2))$ with $b$ centres drawn as a random subsample of the $p$-samples. They differ in how they select the bandwidth $\sigma$ and the number of centres $b$, with consequences for both accuracy and runtime.

**Number of centres $b$.** Sugiyama et al. (2008, §4.2) recommend $b = 100$ as a default — small enough that the linear system is fast to solve, large enough that the basis can approximate smooth ratios on $\mathbb{R}^d$ for moderate $d$. Increasing $b$ past $100$ buys very little in practice on 1-D and 2-D problems, but matters for $d \geq 5$: as the data spreads in higher dimensions, more centres are needed for the basis to span the support overlap. A rule of thumb is $b = \min(200, n_p)$ for $d \leq 10$ and $b = \min(500, n_p)$ for higher $d$; the runtime cost is $O(b^3 + b^2 (n_p + n_q))$ for uLSIF and dominated by the basis-evaluation step for KLIEP. We use $b = 100$ throughout this topic because the running 1-D toy is well-served by it.

**Bandwidth $\sigma$.** The three estimators have qualitatively different bandwidth-selection options:

- **KMM (§4)** has no built-in CV criterion. The standard default is the median-pairwise-distance heuristic on the pooled $(p, q)$ sample. Cross-validation against test-set performance on the downstream IW task (covariate-shift correction) is possible but requires labelled test data — usually unavailable.
- **KLIEP (§5)** uses $K$-fold KL cross-validation: split $p$-samples into $K$ folds, train KLIEP on $K-1$ folds, score held-out log-likelihood on the remaining fold, average across folds and bandwidth candidates. This is the principled choice but costs $K \cdot |\sigma\text{-grid}|$ KLIEP refits, which adds up.
- **uLSIF (§6)** uses analytic LOO-CV from the Sherman-Morrison machinery of §6.3 — no refits, single $O(b^3 + b^2 (n_p + n_q))$ score per $(\sigma, \lambda)$ pair. This is the practical default and explains why uLSIF dominates kernel-DRE in industry.

**Tikhonov ridge $\lambda$ for uLSIF.** The ridge serves two purposes: numerical stability of the linear solve, and bias-variance shrinkage. A log-spaced grid of $\lambda \in [10^{-4}, 10^{-1}]$ with 7–9 candidates, run alongside the bandwidth sweep via analytic LOO, is the standard recipe (Kanamori, Hido, Sugiyama 2009, §3). Larger $\lambda$ shrinks $\hat r$ toward $1$ — useful when the support overlap is poor and the un-shrunk estimate is unstable.

### §12.2 Effective sample size as a runtime diagnostic

The effective-sample-size diagnostic $n_{\text{eff}} = (\sum_i w_i)^2 / \sum_i w_i^2$, introduced in §2.3 and revisited in §9.3, is the most important runtime check on any DRE-driven pipeline. After fitting any DRE estimator on $(p, q)$ samples, compute $n_{\text{eff}}$ on the $q$-samples using $w_i = \hat r(X_i^q)$. The number ranges from $1$ to $n_q$:

| $n_{\text{eff}} / n_q$ | Interpretation                                    | Action                                      |
|-------------------------|---------------------------------------------------|---------------------------------------------|
| $> 0.5$                 | Reweighting is well-conditioned                   | Proceed with downstream IW estimator        |
| $0.1 \le \cdot \le 0.5$ | Moderate effective shrinkage; usable but variable | Use IW with awareness; consider truncation  |
| $0.01 \le \cdot < 0.1$  | Severe shrinkage; IW estimator is high-variance   | Strongly consider self-normalized IW or truncation; sanity-check $\hat r$ |
| $< 0.01$                | $\hat r$ concentrates almost all mass on a few samples | Almost certainly the DRE estimator is failing; check the support overlap |

The $n_{\text{eff}}$ value gives the practitioner a fast read on whether the IW estimator in §9 or the weighted-conformal procedure in §10 is operating in a regime where the asymptotic guarantees apply. Reporting $n_{\text{eff}}$ alongside any IW-based estimate is a strong defensive habit.

When $n_{\text{eff}}$ is small relative to $n_q$, the diagnostic doesn't tell us *why* — it could be the true $\chi^2(p \,\|\, q)$ is genuinely large (severe shift, partial support overlap), or the DRE estimator is over-fitting and producing a spuriously peaked $\hat r$. Distinguishing these requires a side check, typically by comparing $\hat r$ to the truncated estimator $\min(\hat r, B)$ and seeing whether the IW point estimate is stable to truncation. If truncation barely moves the estimate, the high-weight region is small and IW is fragile but not wrong; if truncation moves the estimate a lot, the high-weight region is doing real work and we have a more delicate problem.

### §12.3 The curse of dimensionality for DRE

All four estimator families derived in this topic — KMM, KLIEP, LSIF/uLSIF, and probabilistic classification — fit a function on $\mathbb{R}^d$ from finite samples. Nonparametric estimation in $d$ dimensions suffers a curse-of-dimensionality rate $n^{-2/(4+d)}$ for second-order-smooth targets (Stone 1982), and DRE is no different: the convergence rate of $\hat r$ to the true $r$ degrades polynomially with $d$.

The convergence rates for kernel-basis DRE are derived in Kanamori, Suzuki, and Sugiyama (2012, §6) for uLSIF: under second-order smoothness of $r$, the MISE converges at rate $n^{-4/(4+d)}$ in dimension $d$. For $d = 1$ this is $n^{-4/5} \approx n^{-0.8}$; for $d = 5$ it is $n^{-4/9} \approx n^{-0.44}$; for $d = 20$ it is $n^{-4/24} \approx n^{-0.17}$ — essentially flat. By $d = 50$, the convergence rate is so slow that no reasonable sample size produces a useful estimator without structural assumptions.

**The §12.3 demo** quantifies this on a controlled DGP: $p = \mathcal{N}(\mathbf{0}_d, \mathbf{I}_d)$, $q = \mathcal{N}(\mathbf{1}_d, \mathbf{I}_d)$, with closed-form ratio $r(\mathbf{x}) = \exp(d/2 - \mathbf{1}_d^\top \mathbf{x})$. Fit uLSIF in dimensions $d \in \{1, 2, 5, 10, 20\}$ at fixed $n_p = n_q = 500$ with the median-heuristic bandwidth and a small $\lambda$ grid. Report log-MSE of $\hat r$ against the closed-form $r$ on a held-out test sample, and the empirical $n_{\text{eff}}$.

**What the practitioner should take away.** Past $d \approx 20$, kernel-basis DRE on isotropic Gaussian shifts is no longer a small-data exercise — it requires either (i) explicit structural assumptions (low-dimensional manifold structure, factorization of $r$), (ii) neural classification-DRE that can exploit feature-learning to compress the effective dimension (§7, §8), or (iii) acceptance that the IW estimator will be high-variance and partial corrections like truncation or self-normalization are necessary. The kernel-basis family covered in §4–§6 is most useful for $d \leq 10$, which spans most tabular shift problems in practice; for higher-dimensional data (images, text, sequences), classification-DRE with a deep classifier is the practitioner default.

**Viz: curse of dimensionality.**
- *Shows:* two-panel figure. (Left) log-MSE of $\hat r_{\mathrm{uLSIF}}$ vs the closed-form $r$ on a held-out test sample, as a function of input dimension $d \in \{1, 2, 5, 10, 20\}$, on a log-$y$ axis. (Right) effective sample size $n_{\text{eff}} / n_q$ at the $q$-samples as a function of $d$, on a linear axis with a horizontal reference at the "$> 0.5$ is well-conditioned" guideline from §12.2.
- *Reader controls (interactive site version):* sample sizes $n_p = n_q \in [200, 2000]$, shift severity $\|\boldsymbol{\mu}_q\|$, bandwidth multiplier on the median heuristic.
- *What the reader should learn:* (i) MSE grows by roughly an order of magnitude with each five-dimensional increment, consistent with the Stone (1982) rate; (ii) $n_{\text{eff}}/n_q$ collapses from $> 0.5$ at $d = 1$ to $< 0.05$ at $d = 20$ — the IW estimator becomes unusable in this regime; (iii) increasing $n$ shifts both curves but does not change the slope.

**Code experiment.**
- For each $d \in \{1, 2, 5, 10, 20\}$, sample $n_p = n_q = 500$ from isotropic shift Gaussians.
- Fit uLSIF with $b = 100$ Gaussian-basis centres at the median-heuristic bandwidth on the pooled sample, with a fixed $\lambda = 10^{-2}$ for runtime budget.
- Sample $n_{\text{test}} = 2{,}000$ fresh points from $q$, evaluate the closed-form $r$ and the uLSIF $\hat r$, compute log-MSE and $n_{\text{eff}}$ at these test points.
- Plot the two panels; print the diagnostic table $(d, \mathrm{MSE}, n_{\text{eff}}/n_q)$.
- Validate: log-MSE at $d = 20$ exceeds log-MSE at $d = 1$ by at least one order of magnitude; $n_{\text{eff}}/n_q$ at $d = 20$ is below $0.20$.

### §12.4 When to prefer classification-DRE over kernel-DRE

The four estimator families have different strengths, and the right choice depends on dimensionality, sample size, and what kind of downstream object you need. The decision rules below are practitioner heuristics, not theorems, and they assume the typical regime of tabular-to-moderately-structured data.

**Use kernel-basis DRE (uLSIF in §6 as the default) when:**
- Input dimension is moderate ($d \leq 20$).
- The downstream use only needs sample-level weights or function evaluations on the train/test feature distributions (covariate-shift correction in §9, weighted conformal in §10).
- The smoothness of the underlying $r$ is plausible (kernel-basis estimators assume $r$ is at least Hölder-continuous with order $\geq 1$).
- The runtime budget is small and the practitioner wants principled hyperparameter selection without setting up an SGD pipeline.

**Use classification-DRE (logistic regression in §7 as the default, neural classifier in §8 for high $d$) when:**
- Input dimension is high ($d > 50$), or the input space is structured (images, text, time series).
- A pretrained classifier or feature extractor is already available — exponentiating its logit gives a DRE estimator for free.
- The classifier loss is naturally proper (cross-entropy on a well-specified parametric family); for tree ensembles or SVMs, apply post-hoc Platt scaling or isotonic regression first (§7.3).
- The end use is a *function* on the input space rather than sample-level weights — neural classifiers extrapolate to out-of-sample inputs naturally, while in-sample-only KMM does not.

**Use neural variational DRE (NWJ-trained MLP from §8.3) when:**
- The estimation target is a divergence (KL or chi-squared) rather than the ratio itself — mutual-information estimation, distributional distances for representation learning.
- High dimension and structured inputs make classification-DRE the natural choice, but the downstream use is a divergence integral rather than per-sample weights.
- Compute budget allows for SGD training; the kernel-basis family is faster on small-data low-$d$ tabular problems.

**Avoid DRE entirely when:**
- $n_{\text{eff}} / n_q$ from any of the above estimators falls below $0.01$. At that point, the support overlap is so poor that no IW estimator can correct the shift; the right answer is to acknowledge the limit and either collect more diverse training data or accept that the test-distribution risk is not estimable from the train data alone.
- Concept drift or label shift is suspected — covariate shift's load-bearing assumption (§9.1) has failed and IW correction will introduce bias rather than removing it.

---

## §13. Connections, limits, and forward pointers

§13 is the closing section, text-only by design. It does three jobs: maps the cross-site prerequisite structure that the formalML topic graph will encode in the MDX frontmatter, lists the within-formalML sibling topics that share machinery with DRE, and surfaces the topic's honest limits and open research directions. The cross-site prereq entries below are drafted in the schema the Claude Code session will paste into the MDX frontmatter; the slugs will be verified against `curriculum-graph.json` and the sister-site `pages.json` files during the implementation pass.

### §13.1 Sister-site upstream prereqs

The DRE topic draws on the following sister-site foundations. Each entry below is in the schema $\{\text{topic}, \text{site}, \text{relationship}\}$ with `relationship` prose at $\geq 40$ characters.

**`formalstatisticsPrereqs`** (statistics topics this ML topic requires):

- **topic:** `kernel-density-estimation`
  - **site:** `formalstatistics`
  - **relationship:** "DRE generalizes the plug-in KDE-divided-by-KDE estimator from Topic 30, replacing it with direct estimation that avoids the §1.1 tail-blowup pathology; KDE bandwidth selection, kernel choice, and the AMISE rate machinery transfer to KMM (§4), KLIEP (§5), and uLSIF (§6) verbatim."

- **topic:** `maximum-likelihood-estimation`
  - **site:** `formalstatistics`
  - **relationship:** "KLIEP's KL-divergence framing in §5.1 is built on the MLE machinery from this topic — the constrained-MLE Lagrangian, the consistency theorem under misspecification, and the projected-gradient algorithm all reuse standard MLE technology applied to a ratio model rather than a density model."

- **topic:** `monte-carlo-integration` *(slug may differ; verify against the sister-site graph)*
  - **site:** `formalstatistics`
  - **relationship:** "The §2.2 importance-weighting identity is the central tool from this topic, generalizing single-distribution Monte Carlo to ratio-weighted cross-distribution estimation; the chi-squared-divergence variance bound in §2.3 and the effective-sample-size diagnostic in §12.2 are standard importance-sampling results applied to DRE."

**`formalcalculusPrereqs`** (calculus / optimization / measure-theory topics this ML topic requires):

- **topic:** `convex-optimization` *(or `lagrangian-duality` — verify against the sister-site graph)*
  - **site:** `formalcalculus`
  - **relationship:** "KMM's QP in §4.2, KLIEP's constrained convex program in §5.2, and uLSIF's closed-form normal equation in §6.1 all rely on the convex-optimization machinery from this topic — including Lagrangian duality, KKT conditions, projected-gradient methods, and Sherman-Morrison rank-one matrix updates."

- **topic:** `fenchel-conjugate` *(or `convex-conjugate-functions` — verify against the sister-site graph; may be folded into `convex-optimization`)*
  - **site:** `formalcalculus`
  - **relationship:** "The Nguyen-Wainwright-Jordan variational lower bound on $f$-divergences derived in §8.1 rests on the Fenchel-Young inequality and the explicit convex-conjugate computations from this topic; the LSIF and KLIEP recoveries in §8.2 require evaluating $f^\\star$ for $\\phi(u) = (u-1)^2/2$ and $\\phi(u) = u \\log u$ respectively."

- **topic:** `radon-nikodym-theorem` *(if a standalone topic; otherwise rolled into measure-theory or `lebesgue-integration`)*
  - **site:** `formalcalculus`
  - **relationship:** "The §2.2 detour on absolute continuity uses the Radon-Nikodym theorem to ground the density-ratio function $r = \\mathrm{d}p/\\mathrm{d}q$ as a measurable object before the importance-weighting identity is applied; without it, the proof in §2.2's Move 2 would be merely formal."

**`formalcalculusConnections`** and **`formalstatisticsConnections`**: empty. DRE does not naturally lead a reader into a calculus or statistics topic as a follow-up; the forward direction is to other ML topics (within `formalmlConnections` — handled in §13.2).

*Implementation note for the Code session:* the three slugs flagged with "verify" above should be checked against `curriculum-graph.json` and the sister-site `pages.json` files before merging. If the exact slug doesn't exist, the relationship prose should be either rewritten to point at the closest existing slug, or removed from the frontmatter list with a `TODO` comment recorded in the audit log.

### §13.2 Within-formalML sibling links

Within formalML, DRE shares machinery or framing with the following sibling topics. None of these are formal prereqs (they live on the same site and the audit rules forbid `formalmlPrereqs` self-references); they are exposition-level cross-links for the §13.2 narrative paragraph and the related-topics sidebar.

- **`kernel-regression`** (T2 Supervised Learning). The kernel-weight matrix construction in §4–§6 (Gaussian RBF basis, median-heuristic bandwidth, Tikhonov-regularized linear systems, leave-one-out cross-validation) is the same machinery this topic uses for the Nadaraya–Watson estimator. KMM, KLIEP, and uLSIF can be read as "Nadaraya–Watson for the density ratio" — same kernel idioms, different target functional.

- **`gaussian-processes`** (T5 Bayesian ML). The kernel-mean-embedding machinery introduced in §4.1 and reused in §11.1 lives in the same RKHS framework Gaussian processes use for posterior computation. The "characteristic kernel" notion that drives MMD's injectivity and KMM's identifiability is the same condition that gives GPs their universality.

- **`normalizing-flows`** (T3 Unsupervised & Generative, same track). Both topics target generative modeling — normalizing flows estimate $p(x)$ directly via change-of-variables; DRE estimates the ratio $p(x)/q(x)$. The §8.4 GAN-as-implicit-DRE observation connects the two: a normalizing flow trained adversarially has a discriminator that is implicitly doing DRE between the flow's pushforward and the data distribution.

- **`conformal-prediction`** (T4 Statistical ML Theory). The §10 weighted-exchangeability extension of split conformal is the immediate downstream use of DRE in calibrated prediction-interval construction. Tibshirani, Barber, Candès, and Ramdas (2019) built that bridge; DRE supplies the $\hat r$ that closes it.

- **`clustering`** (T3 Unsupervised & Generative, same track). The §1.1 plug-in-pathology demo — KDE divided by KDE — is the same KDE machinery the clustering topic uses for mean-shift's density gradient. Both topics extend KDE in different directions: clustering follows the gradient uphill; DRE replaces the quotient with direct estimation.

### §13.3 Honest limits of DRE

The four estimator families derived in this topic are mathematically clean and operationally effective in the regime they were built for — tabular data, moderate dimension, smooth ratios — but they have hard limits that the practitioner should hold close.

**Curse of dimensionality (revisited from §12.3).** All four estimators degrade polynomially with input dimension at the Stone (1982) rate. By $d \approx 50$, none of them produce a usable estimate at any reasonable sample size without structural assumptions. The neural classification-DRE of §7 and §8 partially escape this via feature-learning, but only when the feature representation actually compresses the effective dimension — which requires a domain-specific architecture and training regime.

**Support overlap is often violated in practice.** The absolute-continuity hypothesis $p \ll q$ from §2.3 is a strong assumption. In real covariate-shift settings — clinical trials with new patient demographics, vision models deployed in new geographic conditions, language models facing new content domains — the test distribution routinely has support outside the training distribution. When this happens, the true $r$ is infinite on a set of positive $p$-measure, no DRE estimator can recover it there, and the IW estimator's bias is not just a finite-sample artefact but a fundamental obstruction.

**Calibration requirements for classification-DRE.** §7's classification-DRE identity assumes the classifier's predicted probabilities approximate the true posterior. Most modern classifiers — gradient-boosted trees, random forests, deep neural networks — are miscalibrated by default (Niculescu-Mizil and Caruana 2005; Guo et al. 2017). Plugging an uncalibrated classifier into the DRE identity gives a systematically biased ratio estimate, and the bias propagates into every downstream IW estimator (§9, §10) without warning. Post-hoc calibration (Platt scaling, isotonic regression, temperature scaling) is necessary in most pipelines and is itself a finite-sample procedure with its own variance.

**IW estimator variance is fundamental, not removable.** Even with a perfectly known $r$, the IW estimator's variance is inflated by $\chi^2(p \,\|\, q) + 1$ relative to the unweighted estimator (§2.3, §9.3). When the chi-squared divergence is large — which is the regime where reweighting matters most — IW estimators are unavoidably high-variance. Truncation and self-normalization (§9.3) help but introduce bias. There is no free lunch: large $\chi^2$ implies high IW variance.

**DRE in modern deep-learning practice.** The classical DRE pipeline — fit $\hat r$, then plug into a downstream IW estimator — is not always how the deep-learning community engages with density ratios. Modern generative models (GANs, score-based diffusion, VAE-GAN hybrids), contrastive representation learning (InfoNCE, SimCLR), and energy-based models all *implicitly* estimate density ratios as a side effect of their primary training objective. These implicit estimators are typically more accurate on high-dimensional structured data than explicit kernel-DRE, but they don't expose $\hat r$ as a first-class object; recovering it requires careful logit extraction or auxiliary discriminator probes. This is an active area of research and the integration between classical DRE theory (this topic) and modern implicit-DRE practice is incomplete.

### §13.4 Open problems and forward pointers

DRE is an active research area with several open directions that exceed the scope of this topic. Pointers below are deliberately concrete, naming the line of work and a representative recent reference.

**Bregman-divergence DRE with neural witnesses.** The §3 Bregman framework with the §8 neural parametrization is a marriage of convenience that hasn't been fully systematized — for arbitrary $\phi$, when does the neural-NWJ training actually converge to $f'(r)$ at the right rate? Recent work (Rhodes, Xu, Gutmann 2020 on telescoping density-ratio estimation; Choi, Liao, Ermon 2021 on featurized density-ratio estimation) makes progress but doesn't deliver a unified theory.

**DRE under sample-selection bias.** Sample-selection bias is a strict generalization of covariate shift in which the training-sample inclusion probability depends on $(x, y)$ rather than $x$ alone. Standard DRE cannot identify $r$ in this setting without additional assumptions (instrumental variables, missing-at-random conditions, or auxiliary unlabelled data). The Heckman (1979) selection-correction framework provides one route; modern semi-parametric efficiency theory (Kennedy, Ma, McHugh, Small 2017) provides a richer one.

**Off-policy evaluation in reinforcement learning.** Off-policy evaluation — estimating the value of a target policy from data collected by a different behaviour policy — is DRE applied to action-conditioned distributions, often with horizon effects that produce exponential blow-up in the ratio. Doubly-robust estimators (Jiang and Li 2016) and marginalized importance sampling (Liu et al. 2018) are active areas; the §9 framework here is the building block, but the RL setting adds sequential dependence and horizon control.

**Score-based DRE.** A recent line of work uses score matching to estimate $\nabla \log r$ directly, then integrates to recover $r$ up to a normalizing constant. Hyvärinen (2005) introduced the score-matching framework; Choi, Liao, and Ermon (2021) adapted it to DRE specifically. Score-based DRE sidesteps some of the variance issues of classical DRE in high dimensions but introduces its own integration-constant ambiguity.

**Conditional DRE.** Estimating $p(x \mid z) / q(x \mid z)$ for some conditioning variable $z$ — rather than the marginal ratio $p(x)/q(x)$ — comes up in conditional-distribution shift, fairness-constrained DRE, and treatment-effect estimation. The kernel-basis methods of §4–§6 extend in principle by working in the joint $(x, z)$ space, but the dimensionality penalty becomes severe; classification-DRE with a $z$-conditioned classifier is the practitioner default.

**DRE for sequential and non-stationary data.** When the training and test distributions are not just shifted but evolving over time, the iid foundation of §2 breaks down. Online DRE, drift detection, and concept-drift correction are active areas (Gama et al. 2014 for a survey).

---

## References

Chicago 17th edition (Notes and Bibliography), grouped foundational / advanced / applied. Every entry includes a verified URL.

### Foundational

Gretton, Arthur, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, and Alexander Smola. 2012. "A Kernel Two-Sample Test." *Journal of Machine Learning Research* 13: 723–773. https://jmlr.csail.mit.edu/papers/v13/gretton12a.html.

Huang, Jiayuan, Alexander J. Smola, Arthur Gretton, Karsten M. Borgwardt, and Bernhard Schölkopf. 2007. "Correcting Sample Selection Bias by Unlabeled Data." In *Advances in Neural Information Processing Systems 19*, edited by B. Schölkopf, J. Platt, and T. Hofmann, 601–608. Cambridge, MA: MIT Press. https://papers.nips.cc/paper/3075-correcting-sample-selection-bias-by-unlabeled-data.

Kanamori, Takafumi, Shohei Hido, and Masashi Sugiyama. 2009. "A Least-Squares Approach to Direct Importance Estimation." *Journal of Machine Learning Research* 10: 1391–1445. https://jmlr.org/papers/volume10/kanamori09a/kanamori09a.pdf.

Owen, Art B. 2013. *Monte Carlo Theory, Methods and Examples*. Online textbook (Stanford). https://artowen.su.domains/mc/.

Royden, Halsey L. 1988. *Real Analysis*. 3rd ed. New York: Macmillan. https://www.worldcat.org/oclc/17738963.

Shimodaira, Hidetoshi. 2000. "Improving Predictive Inference under Covariate Shift by Weighting the Log-Likelihood Function." *Journal of Statistical Planning and Inference* 90 (2): 227–244. https://doi.org/10.1016/S0378-3758(00)00115-4.

Sriperumbudur, Bharath K., Arthur Gretton, Kenji Fukumizu, Bernhard Schölkopf, and Gert R. G. Lanckriet. 2010. "Hilbert Space Embeddings and Metrics on Probability Measures." *Journal of Machine Learning Research* 11: 1517–1561. https://www.jmlr.org/papers/v11/sriperumbudur10a.html.

Stone, Charles J. 1982. "Optimal Global Rates of Convergence for Nonparametric Regression." *The Annals of Statistics* 10 (4): 1040–1053. https://doi.org/10.1214/aos/1176345969.

Sugiyama, Masashi, Matthias Krauledat, and Klaus-Robert Müller. 2007. "Covariate Shift Adaptation by Importance Weighted Cross Validation." *Journal of Machine Learning Research* 8: 985–1005. https://jmlr.org/papers/v8/sugiyama07a.html.

Sugiyama, Masashi, Taiji Suzuki, Shinichi Nakajima, Hisashi Kashima, Paul von Bünau, and Motoaki Kawanabe. 2008. "Direct Importance Estimation for Covariate Shift Adaptation." *Annals of the Institute of Statistical Mathematics* 60 (4): 699–746. https://doi.org/10.1007/s10463-008-0197-x.

Sugiyama, Masashi, Taiji Suzuki, and Takafumi Kanamori. 2012. *Density Ratio Estimation in Machine Learning*. New York: Cambridge University Press. https://doi.org/10.1017/CBO9781139035613.

van der Vaart, Aad W. 1998. *Asymptotic Statistics*. Cambridge: Cambridge University Press. https://doi.org/10.1017/CBO9780511802256.

Vovk, Vladimir, Alex Gammerman, and Glenn Shafer. 2005. *Algorithmic Learning in a Random World*. New York: Springer. https://doi.org/10.1007/b106715.

### Advanced

Belghazi, Mohamed Ishmael, Aristide Baratin, Sai Rajeshwar, Sherjil Ozair, Yoshua Bengio, Aaron Courville, and Devon Hjelm. 2018. "Mutual Information Neural Estimation." In *Proceedings of the 35th International Conference on Machine Learning*, edited by Jennifer Dy and Andreas Krause, PMLR 80: 531–540. https://proceedings.mlr.press/v80/belghazi18a.html.

Gneiting, Tilmann, and Adrian E. Raftery. 2007. "Strictly Proper Scoring Rules, Prediction, and Estimation." *Journal of the American Statistical Association* 102 (477): 359–378. https://doi.org/10.1198/016214506000001437.

Goodfellow, Ian J., Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. "Generative Adversarial Nets." In *Advances in Neural Information Processing Systems 27*, 2672–2680. https://papers.nips.cc/paper/5423-generative-adversarial-nets.

Guo, Chuan, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. "On Calibration of Modern Neural Networks." In *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70: 1321–1330. https://proceedings.mlr.press/v70/guo17a.html.

Hyvärinen, Aapo. 2005. "Estimation of Non-Normalized Statistical Models by Score Matching." *Journal of Machine Learning Research* 6: 695–709. https://jmlr.org/papers/v6/hyvarinen05a.html.

Kanamori, Takafumi, Taiji Suzuki, and Masashi Sugiyama. 2012. "Statistical Analysis of Kernel-Based Least-Squares Density-Ratio Estimation." *Machine Learning* 86 (3): 335–367. https://doi.org/10.1007/s10994-011-5266-3.

Lei, Jing, Max G'Sell, Alessandro Rinaldo, Ryan J. Tibshirani, and Larry Wasserman. 2018. "Distribution-Free Predictive Inference for Regression." *Journal of the American Statistical Association* 113 (523): 1094–1111. https://doi.org/10.1080/01621459.2017.1307116.

Menon, Aditya Krishna, and Cheng Soon Ong. 2016. "Linking Losses for Density Ratio and Class-Probability Estimation." In *Proceedings of the 33rd International Conference on Machine Learning*, PMLR 48: 304–313. https://proceedings.mlr.press/v48/menon16.html.

Mohamed, Shakir, and Balaji Lakshminarayanan. 2016. "Learning in Implicit Generative Models." arXiv:1610.03483. https://arxiv.org/abs/1610.03483.

Nguyen, XuanLong, Martin J. Wainwright, and Michael I. Jordan. 2010. "Estimating Divergence Functionals and the Likelihood Ratio by Convex Risk Minimization." *IEEE Transactions on Information Theory* 56 (11): 5847–5861. https://doi.org/10.1109/TIT.2010.2068870.

Niculescu-Mizil, Alexandru, and Rich Caruana. 2005. "Predicting Good Probabilities with Supervised Learning." In *Proceedings of the 22nd International Conference on Machine Learning*, 625–632. https://doi.org/10.1145/1102351.1102430.

Nowozin, Sebastian, Botond Cseke, and Ryota Tomioka. 2016. "f-GAN: Training Generative Neural Samplers Using Variational Divergence Minimization." In *Advances in Neural Information Processing Systems 29*, 271–279. https://proceedings.neurips.cc/paper/2016/hash/cedebb6e872f539bef8c3f919874e9d7-Abstract.html.

Sutherland, Danica J., Hsiao-Yu Tung, Heiko Strathmann, Soumyajit De, Aaditya Ramdas, Alexander J. Smola, and Arthur Gretton. 2017. "Generative Models and Model Criticism via Optimized Maximum Mean Discrepancy." In *International Conference on Learning Representations*. https://arxiv.org/abs/1611.04488.

Tibshirani, Ryan J., Rina Foygel Barber, Emmanuel J. Candès, and Aaditya Ramdas. 2019. "Conformal Prediction Under Covariate Shift." In *Advances in Neural Information Processing Systems 32*, 2526–2536. https://proceedings.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html.

### Applied

Choi, Kristy, Madeline Liao, and Stefano Ermon. 2021. "Featurized Density Ratio Estimation." In *Proceedings of the 37th Conference on Uncertainty in Artificial Intelligence*, PMLR 161: 1924–1933. https://proceedings.mlr.press/v161/choi21a.html.

Gama, João, Indrė Žliobaitė, Albert Bifet, Mykola Pechenizkiy, and Abdelhamid Bouchachia. 2014. "A Survey on Concept Drift Adaptation." *ACM Computing Surveys* 46 (4): Article 44. https://doi.org/10.1145/2523813.

Heckman, James J. 1979. "Sample Selection Bias as a Specification Error." *Econometrica* 47 (1): 153–161. https://doi.org/10.2307/1912352.

Jiang, Nan, and Lihong Li. 2016. "Doubly Robust Off-Policy Value Evaluation for Reinforcement Learning." In *Proceedings of the 33rd International Conference on Machine Learning*, PMLR 48: 652–661. https://proceedings.mlr.press/v48/jiang16.html.

Kennedy, Edward H., Zongming Ma, Matthew D. McHugh, and Dylan S. Small. 2017. "Non-parametric Methods for Doubly Robust Estimation of Continuous Treatment Effects." *Journal of the Royal Statistical Society: Series B* 79 (4): 1229–1245. https://doi.org/10.1111/rssb.12212.

Liu, Qiang, Lihong Li, Ziyang Tang, and Dengyong Zhou. 2018. "Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation." In *Advances in Neural Information Processing Systems 31*. https://proceedings.neurips.cc/paper/2018/hash/dda04f9d634145a9c68d5dfe53b21272-Abstract.html.

Lopez-Paz, David, and Maxime Oquab. 2017. "Revisiting Classifier Two-Sample Tests." In *International Conference on Learning Representations*. https://arxiv.org/abs/1610.06545.

Rhodes, Benjamin, Kai Xu, and Michael U. Gutmann. 2020. "Telescoping Density-Ratio Estimation." In *Advances in Neural Information Processing Systems 33*, 4905–4916. https://proceedings.neurips.cc/paper/2020/hash/33d3b157ddc0896addfb22fa2a519097-Abstract.html.

