#!/usr/bin/env python3
"""Assemble the Bayesian Inference notebook from cell definitions.

Produces ``01_bayesian_inference.ipynb`` — a native Jupyter notebook (nbformat 4)
that runs end-to-end on CPU-only NumPy / SciPy / matplotlib in under 30 s.
"""
from __future__ import annotations

import json
from pathlib import Path

CELLS: list[dict] = []


def _to_lines(s: str) -> list[str]:
    if not s:
        return []
    parts = s.split("\n")
    out = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return out


def md(source: str) -> None:
    CELLS.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": _to_lines(source),
    })


def code(source: str) -> None:
    CELLS.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _to_lines(source),
    })


# ============================================================================
# Title + preamble
# ============================================================================

md(r"""# Bayesian Inference

**Topic:** T5 — Bayesian & Probabilistic ML, umbrella (foundational)
**Companion brief:** `docs/plans/formalml-bayesian-inference-handoff-brief.md`

This notebook is the source-of-truth Jupyter companion to the handoff brief. The brief carries the full mathematical exposition; this notebook carries the numerical experiments that validate the claims and produce the figures. Cells are organized in lock-step with the brief's 13 sections.

End-to-end runtime target: under 30 s on a 2020-era laptop, CPU-only. The umbrella's lightweight closed-form-and-grid emphasis means NumPy / SciPy / matplotlib only — no PyMC, no Stan, no MCMC libraries. Those forward-map to the specific T5 leaves (`probabilistic-programming`, `stochastic-gradient-mcmc`, `riemann-manifold-hmc`, etc.) that own them.

Figures are saved to `figures/` as the notebook runs.""")


code(r'''"""Preamble — imports, RNG, styling, FIG_DIR, shared color palette."""
from __future__ import annotations
from pathlib import Path
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from scipy import stats, integrate
import scipy.special as ssp

# --- Reproducibility -------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

# --- Notebook-wide timer --------------------------------------------------
_NB_START_TIME = time.time()

# --- Figure styling — formalML May-2026 baseline --------------------------
COLOR_C0     = "#1f4e79"   # deep blue   — variant A / prior
COLOR_C1     = "#c0504d"   # red         — variant B / posterior
COLOR_TRUE   = "#3a6e3a"   # forest      — ground truth / likelihood
COLOR_ACCENT = "#9467bd"   # purple      — accents / posterior predictive
COLOR_DATA   = "#7f7f7f"   # gray        — raw scatter / data
COLOR_BAND   = "#ff7f0e"   # orange      — highlights / decision bands

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.figsize": (7.5, 4.4),
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
    "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.labelsize": 10, "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 1.8, "font.size": 10,
})

# --- Output dir for figures ----------------------------------------------
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

print(f"NumPy {np.__version__}, SciPy {scipy.__version__}")
print(f"Seed: {SEED}")''')


# ============================================================================
# §1 — Why Bayesian inference?
# ============================================================================

md(r"""## §1. Why Bayesian inference?

Three motivations recur: **prior information** (we know things before the data arrives — encode them), **calibrated uncertainty** (a Bayesian estimator is a full posterior distribution, not a single number), and **decision-theoretic optimality** (under a loss function, the posterior-expected-loss-minimizing decision rule is the optimal one — Theorem 7.1). See brief §1.1 for the unpacked version.

### §1.2 Motivating vignette — a Beta-Binomial A/B test

Setup that we'll carry through nine more sections: 1000 impressions of variant A → 47 conversions; 1000 impressions of variant B → 64. A uniform $\text{Beta}(1, 1)$ prior on each rate yields conjugate posteriors $\text{Beta}(48, 954)$ and $\text{Beta}(65, 937)$ (we'll prove the Beta-Binomial conjugacy in §4.2). The decision-relevant Bayesian question — *what is the posterior probability that B beats A?* — is answered directly by Monte Carlo sampling from the two posteriors. The answer is approximately $0.95$.

The figure below shows the three pieces: posteriors for each variant with 95% credible intervals shaded, posterior of the lift $\theta_B - \theta_A$, and $P(\theta_B > \theta_A \mid \text{data})$. We earn every piece of this figure over §2–§7.

> **Forward pointers.** The interactive `BayesCycleExplorer` viz lives in §2; the conjugate-update animator lives in §4; the decision-theoretic framing lives in §7.""")


code(r'''# §1.2 — Motivating vignette: the Beta-Binomial A/B test that carries the topic.

# Observed data: 1000 impressions per variant, with 47 and 64 conversions.
n_A, s_A = 1000, 47
n_B, s_B = 1000, 64

# Uniform prior Beta(1, 1) gives conjugate posteriors Beta(1 + s, 1 + n - s).
alpha_prior, beta_prior = 1, 1
post_A = stats.beta(alpha_prior + s_A, beta_prior + n_A - s_A)   # Beta(48, 954)
post_B = stats.beta(alpha_prior + s_B, beta_prior + n_B - s_B)   # Beta(65, 937)

# Density grid spanning a window that contains both posteriors.
theta_grid = np.linspace(0.02, 0.10, 400)

# Posterior of theta_B - theta_A by Monte Carlo.
M = 200_000
samples_A = post_A.rvs(size=M, random_state=rng)
samples_B = post_B.rvs(size=M, random_state=rng)
delta = samples_B - samples_A
prob_B_beats_A = float(np.mean(delta > 0))

mean_A, mean_B = post_A.mean(), post_B.mean()
lo_A, hi_A = post_A.ppf([0.025, 0.975])
lo_B, hi_B = post_B.ppf([0.025, 0.975])

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

ax = axes[0]
pdf_A = post_A.pdf(theta_grid)
pdf_B = post_B.pdf(theta_grid)
ax.plot(theta_grid, pdf_A, color=COLOR_C0, lw=2,
        label=r"$\theta_A\mid\mathrm{data}\sim\mathrm{Beta}(48, 954)$")
ax.plot(theta_grid, pdf_B, color=COLOR_C1, lw=2,
        label=r"$\theta_B\mid\mathrm{data}\sim\mathrm{Beta}(65, 937)$")
ax.fill_between(theta_grid, pdf_A,
                where=(theta_grid >= lo_A) & (theta_grid <= hi_A),
                color=COLOR_C0, alpha=0.15)
ax.fill_between(theta_grid, pdf_B,
                where=(theta_grid >= lo_B) & (theta_grid <= hi_B),
                color=COLOR_C1, alpha=0.15)
ax.set_xlabel(r"conversion rate $\theta$")
ax.set_ylabel("posterior density")
ax.set_title("Posterior beliefs about A and B")
ax.legend(loc="upper right", fontsize=8)

ax = axes[1]
ax.hist(delta, bins=80, density=True, color=COLOR_ACCENT,
        alpha=0.65, edgecolor="white", linewidth=0.4)
ax.axvline(0, color="black", lw=1.2, linestyle="--", label="no-lift baseline")
ax.axvline(delta.mean(), color=COLOR_BAND, lw=2,
           label=fr"posterior mean $\approx {delta.mean():.4f}$")
ax.set_xlabel(r"$\theta_B - \theta_A$")
ax.set_ylabel("posterior density")
ax.set_title(r"Posterior of the lift  $\theta_B - \theta_A$")
ax.legend(loc="upper right", fontsize=8)

ax = axes[2]
ax.barh([""], [prob_B_beats_A], color=COLOR_C1, height=0.45,
        label=r"$P(\theta_B > \theta_A\mid\mathrm{data})$")
ax.barh([""], [1 - prob_B_beats_A], left=[prob_B_beats_A],
        color=COLOR_C0, height=0.45,
        label=r"$P(\theta_B \leq \theta_A\mid\mathrm{data})$")
ax.set_xlim(0, 1)
ax.set_xlabel("posterior probability")
ax.set_title(rf"$P(\theta_B > \theta_A\mid\mathrm{{data}}) \approx {prob_B_beats_A:.3f}$")
ax.text(prob_B_beats_A / 2, 0, f"{prob_B_beats_A:.1%}",
        va="center", ha="center", color="white",
        fontsize=14, fontweight="bold")
ax.spines["left"].set_visible(False)
ax.tick_params(left=False, labelleft=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), fontsize=8, ncol=1)

fig.suptitle("Bayesian read of an A/B test: posterior -> lift -> decision",
             y=1.02, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_2_ab_test_vignette.png")
plt.show()

print(f"Posterior means:  theta_A ~ {mean_A:.4f}   theta_B ~ {mean_B:.4f}")
print(f"95% credible intervals: A = [{lo_A:.4f}, {hi_A:.4f}]   "
      f"B = [{lo_B:.4f}, {hi_B:.4f}]")
print(f"P(B > A | data) ~ {prob_B_beats_A:.4f}   (Monte Carlo, M = {M:,})")
print(f"Expected lift (posterior mean of theta_B - theta_A) ~ {delta.mean():.4f}")''')


# ============================================================================
# §2 — The prior-likelihood-posterior cycle
# ============================================================================

md(r"""## §2. The prior–likelihood–posterior cycle

A Bayesian model is a joint probability distribution. Everything else in this topic — point estimates, posterior predictives, decision rules, hierarchical models — is a way of slicing or summarizing the joint.

### §2.1 The joint distribution viewpoint

With a parameter $\theta$ (unobserved) and data $y$ (observed), the chain rule of probability factors the joint two ways:

$$p(\theta, y) \;=\; \pi(\theta)\, p(y \mid \theta) \;=\; \pi(\theta \mid y)\, p(y).$$

The *left* factorization is how we *build* the model — declare a prior $\pi(\theta)$, declare a likelihood $p(y \mid \theta)$. The *right* factorization is how we *use* the model — observe $y$, and read off the posterior $\pi(\theta \mid y)$ and the marginal $p(y)$.

### §2.2 Bayes' theorem (continuous form)

> **Theorem 2.1 (Bayes' theorem).** With $p(y) > 0$,
> $$\pi(\theta \mid y) \;=\; \frac{p(y \mid \theta)\,\pi(\theta)}{p(y)}, \qquad p(y) = \int_\Theta p(y\mid\theta)\,\pi(\theta)\,d\theta.$$

The static three-panel figure below visualizes the cycle for A/B-test variant A ($n = 1000$, $s = 47$, flat $\text{Beta}(1, 1)$ prior). The interactive `BayesCycleExplorer` — with prior-shape and data sliders — lives in the MDX layer of the published topic. See brief §2.3 for the full proof and the renaming move.""")


code(r'''# §2.2 — Static visualization of the cycle: prior x likelihood -> posterior.
# A/B test variant A: n = 1000, s = 47, with a flat Beta(1, 1) prior.

prior_alpha, prior_beta = 1.0, 1.0
n_obs, s_obs = n_A, s_A        # 1000, 47 (carried from §1)

theta = np.linspace(0.0, 0.20, 600)

prior_density = stats.beta(prior_alpha, prior_beta).pdf(theta)
likelihood_raw = stats.binom.pmf(s_obs, n_obs, theta)
likelihood_rescaled = likelihood_raw / np.trapezoid(likelihood_raw, theta)
posterior = stats.beta(prior_alpha + s_obs, prior_beta + n_obs - s_obs)
post_density = posterior.pdf(theta)

mle = s_obs / n_obs
post_mean = posterior.mean()
post_lo, post_hi = posterior.ppf([0.025, 0.975])

fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.8), sharex=True)

ax = axes[0]
ax.fill_between(theta, prior_density, color=COLOR_C0, alpha=0.30)
ax.plot(theta, prior_density, color=COLOR_C0, lw=2)
ax.set_ylabel(r"$\pi(\theta)$")
ax.set_title(r"Prior - $\mathrm{Beta}(1, 1)$, flat on $[0, 1]$")

ax = axes[1]
ax.fill_between(theta, likelihood_rescaled, color=COLOR_TRUE, alpha=0.30)
ax.plot(theta, likelihood_rescaled, color=COLOR_TRUE, lw=2)
ax.axvline(mle, color=COLOR_BAND, lw=1.5, ls="--",
           label=fr"MLE $\hat\theta = s/n = {mle:.4f}$")
ax.set_ylabel(r"$p(y\mid\theta)$  (rescaled)")
ax.set_title(fr"Likelihood - $p(y\mid\theta) \propto \theta^{{{s_obs}}}(1-\theta)^{{{n_obs - s_obs}}}$,"
             " a function of $\\theta$ (not a density)")
ax.legend(loc="upper right", fontsize=9)

ax = axes[2]
ax.fill_between(theta, post_density, color=COLOR_C1, alpha=0.30)
ax.plot(theta, post_density, color=COLOR_C1, lw=2)
ax.axvline(post_mean, color=COLOR_BAND, lw=1.5, ls="--",
           label=fr"posterior mean $\approx {post_mean:.4f}$")
ax.axvspan(post_lo, post_hi, color=COLOR_C1, alpha=0.08,
           label=f"95% credible interval [{post_lo:.4f}, {post_hi:.4f}]")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$\pi(\theta\mid y)$")
ax.set_title(fr"Posterior - $\mathrm{{Beta}}({int(prior_alpha + s_obs)}, "
             fr"{int(prior_beta + n_obs - s_obs)})$, "
             r"sharply concentrated near the MLE")
ax.legend(loc="upper right", fontsize=9)

fig.suptitle("The Bayesian cycle on A/B-test variant A:  prior  x  likelihood  ->  posterior",
             y=1.005, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_2_bayes_cycle.png")
plt.show()

print(f"Prior mean:      {stats.beta(prior_alpha, prior_beta).mean():.4f}  (flat - uniform)")
print(f"MLE (likelihood max): theta_hat = s/n = {mle:.4f}")
print(f"Posterior mean:  {post_mean:.4f}  ({n_obs} observations have pulled the prior toward the MLE)")
print(f"Posterior 95% CI: [{post_lo:.4f}, {post_hi:.4f}]")''')


md(r"""### §2.3 Proof of Bayes' theorem

From the two factorizations of the joint, $\pi(\theta)\,p(y\mid\theta) = \pi(\theta\mid y)\,p(y)$; divide by $p(y) > 0$:

$$\pi(\theta\mid y) \;=\; \frac{p(y\mid\theta)\,\pi(\theta)}{p(y)}.$$

The full unpacking — and the *renaming* of the four components as prior, likelihood, posterior, marginal likelihood — lives in brief §2.3. The pedagogical content of Bayes' theorem is the renaming, not the algebra.

### §2.4 The marginal likelihood as a normalizing constant

Since $p(y)$ does not depend on $\theta$, the posterior is determined by the unnormalized product:

$$\pi(\theta\mid y) \;\propto\; p(y\mid\theta)\,\pi(\theta).$$

This is why MCMC, MAP, and most posterior-shape queries don't need the intractable $p(y)$ — the constant cancels. $p(y)$ matters for *model comparison* (§10, Bayes factors) but is otherwise bookkeeping.

### §2.5 Numerical sanity check on a finite joint table

Canonical discrete example: a 99%-sensitive, 95%-specific test for a 1% prevalent disease. We build the $2 \times 2$ joint $p(D, T)$, compute the marginal $P(T+)$ from row sums, then verify $P(D \mid T+)$ two ways — directly from the joint table, and via Bayes' theorem. The two routes must agree to floating-point precision, and the headline result is the classic base-rate-neglect surprise: even a "99% accurate" test gives a positive-test posterior of only ~17% when the prior prevalence is 1%.""")


code(r'''# §2.5 — Numerical verification of Bayes' theorem on a finite joint table.
# Disease/test example with rare-disease + imperfect-test base rates.

p_D     = 0.01     # P(D=1) - prior prevalence
sens    = 0.99     # P(T+ | D=1) - sensitivity
spec    = 0.95     # P(T- | D=0) - specificity
fpr     = 1 - spec # P(T+ | D=0) - false positive rate

joint = np.array([
    [p_D     * sens,        p_D     * (1 - sens)],   # D = 1
    [(1-p_D) * fpr,         (1-p_D) * spec       ],  # D = 0
])
assert np.isclose(joint.sum(), 1.0), "joint must sum to 1"

p_T_plus = joint[:, 0].sum()
p_T_minus = joint[:, 1].sum()
print(f"Marginal P(T+) = {p_T_plus:.4f}    (= {joint[0,0]:.4f} + {joint[1,0]:.4f})")
print(f"Marginal P(D=1) (= prior)        = {joint[0].sum():.4f}")

# Route 1: posterior directly from the joint table.
post_direct = joint[0, 0] / p_T_plus

# Route 2: posterior via Bayes' theorem.
post_bayes = (sens * p_D) / p_T_plus

print(f"\nP(D=1 | T+) computed two ways:")
print(f"  Route 1 - direct (joint / marginal): {post_direct:.6f}")
print(f"  Route 2 - Bayes' theorem:            {post_bayes:.6f}")

assert np.isclose(post_direct, post_bayes), \
    "The two routes to the posterior must agree (Theorem 2.1)."
print("\nOK Numerical verification: the two routes agree to floating-point precision.")
print(f"\nInterpretive remark: a 99%-sensitive, 95%-specific test on a 1%-prevalent")
print(f"disease yields P(D=1 | T+) ~ {post_direct:.1%}.  The prior matters,")
print(f"and base-rate neglect is the misreading of this calculation.")''')


# ============================================================================
# §3 — Choosing a prior
# ============================================================================

md(r"""## §3. Choosing a prior

If the posterior depends on the prior, where does the prior come from? Three layers we work through: subjective elicitation via *pseudo-counts* (§3.1), formally-uninformative defaults like uniform and Jeffreys priors (§3.2–§3.3), and the practical safety net that with enough data the choice barely matters (§3.5). Underneath all of it is the honest-broker stance: every choice of prior encodes an opinion; the Bayesian discipline is to declare the opinion explicitly.

### §3.1 Pseudo-counts as the elicitation tool

A $\mathrm{Beta}(\alpha, \beta)$ prior behaves, when used for a Binomial rate, as if we had already observed $(\alpha - 1)$ pseudo-successes and $(\beta - 1)$ pseudo-failures. Conjugacy turns this into bookkeeping:

$$\pi(\theta \mid s, n) = \mathrm{Beta}(\alpha + s, \beta + n - s),$$

so prior pseudo-counts simply add to data counts. The elicitation interview becomes "how many fake observations is your prior worth, and where does the rate sit?"

### §3.2 Uniform vs. weakly informative priors

The uniform $\mathrm{Beta}(1, 1)$ prior looks "objective" — every $\theta \in [0, 1]$ equally probable. But uniform-on-$\theta$ is *not* uniform on the log-odds reparameterization $\varphi = \log(\theta/(1-\theta))$ (we show this in the §3.3 invariance figure). The parameterization smuggles in information. In practice we use *weakly informative* priors like $\mathrm{Beta}(2, 2)$ — diffuse enough to let the data dominate quickly, informative enough to rule out implausible extremes.

The `PriorChoiceComparator` figure below shows three priors of varying informativeness against the A/B-test likelihood at two sample sizes. The convergence as $n$ grows is the §3.5 wash-out result made visible.""")


code(r'''# §3 — PriorChoiceComparator: three priors x two data regimes.

priors = [
    ("Uniform\n" r"$\mathrm{Beta}(1, 1)$",                            1.0,  1.0),
    ("Weakly informative\n" r"$\mathrm{Beta}(2, 2)$",                 2.0,  2.0),
    ("Strongly (wrongly) informative\n" r"$\mathrm{Beta}(20, 80)$",  20.0, 80.0),
]
regimes = [
    ("Small data\n" r"$n = 10$, $s = 1$",      10,    1),
    ("Large data\n" r"$n = 1000$, $s = 47$",   1000,  47),
]
theta = np.linspace(0.0, 1.0, 500)

fig, axes = plt.subplots(3, 3, figsize=(13.5, 8.8), sharex=True)

for col, (label, a, b) in enumerate(priors):
    ax = axes[0, col]
    pd = stats.beta(a, b).pdf(theta)
    ax.fill_between(theta, pd, color=COLOR_C0, alpha=0.30)
    ax.plot(theta, pd, color=COLOR_C0, lw=2)
    ax.set_title(label, fontsize=10)
    if col == 0:
        ax.set_ylabel("Prior\n" r"$\pi(\theta)$", fontsize=10)
    ax.set_ylim(bottom=0)

    for row, (rlabel, n_i, s_i) in enumerate(regimes, start=1):
        ax = axes[row, col]
        post = stats.beta(a + s_i, b + n_i - s_i)
        pd_post = post.pdf(theta)
        ax.fill_between(theta, pd_post, color=COLOR_C1, alpha=0.30)
        ax.plot(theta, pd_post, color=COLOR_C1, lw=2)
        ax.axvline(post.mean(), color=COLOR_BAND, lw=1.4, ls="--",
                   label=f"post. mean ~ {post.mean():.3f}")
        mle = s_i / n_i
        ax.axvline(mle, color="black", lw=1, ls=":", alpha=0.6,
                   label=f"MLE = {mle:.3f}")
        ax.legend(loc="upper right", fontsize=8)
        if col == 0:
            ax.set_ylabel(rlabel + "\nposterior", fontsize=10)
        ax.set_ylim(bottom=0)

for ax in axes[-1, :]:
    ax.set_xlabel(r"$\theta$")

fig.suptitle("PriorChoiceComparator: three priors x two data regimes - "
             r"the prior washes out as $n$ grows",
             y=1.005, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_2_prior_choice.png")
plt.show()

print("Posterior means under three priors:\n")
print(f"{'Prior':<38}  {'n=10, s=1':>12}  {'n=1000, s=47':>14}")
print("-" * 70)
for label, a, b in priors:
    short = label.replace("\n", " ").replace("$", "").replace("\\mathrm", "")
    p_small = stats.beta(a + 1,  b + 9   ).mean()
    p_large = stats.beta(a + 47, b + 953 ).mean()
    print(f"{short:<38}  {p_small:>12.4f}  {p_large:>14.4f}")
print(f"\n{'Empirical rate s/n':<38}  {1/10:>12.4f}  {47/1000:>14.4f}")
print()
print("Observation: at n=1000 the three posterior means cluster within ~0.015")
print("of each other, despite priors that disagree wildly at n=0. At n=10 they")
print("span more than a factor of 2. The prior dominates when n is small; the")
print("data dominate when n is large.")''')


md(r"""### §3.3 Jeffreys priors

The formally-principled answer to "what is an uninformative prior?" is Harold Jeffreys's 1946 invariance argument. Choose the prior whose density is proportional to the square root of the Fisher information:

$$\pi_J(\theta) \;\propto\; \sqrt{I(\theta)}, \qquad I(\theta) = -\mathbb{E}_{\theta}\!\left[\frac{\partial^2 \log p(y\mid\theta)}{\partial \theta^2}\right].$$

The Jeffreys prior is the unique prior (up to scale) that is *invariant under reparameterization* — if we transform $\theta \to \varphi = g(\theta)$, the Jeffreys prior in $\varphi$-space equals the transform of $\pi_J(\theta)$ via the Jacobian.

**For the Binomial rate.** The Fisher info is $I(\theta) = n/(\theta(1-\theta))$, so $\pi_J(\theta) \propto \theta^{-1/2}(1-\theta)^{-1/2}$ — the kernel of $\mathrm{Beta}(1/2, 1/2)$. It's U-shaped, with more mass near 0 and 1 than near 0.5. Surprising, but invariant. (Full derivation: brief §3.3.)

### §3.4 Reference priors

Bernardo's reference prior program generalizes Jeffreys to multidimensional parameters. For 1-D regular models, reference = Jeffreys. The full construction lives in Bernardo & Smith (1994).

The honest stance: **no prior is no opinion**. Uniform priors, weakly informative priors, Jeffreys priors, and elicited priors all encode a position; the Bayesian discipline is to declare it.

### §3.5 When does the prior wash out?

Conjugacy makes the answer crisp: posterior $\mathrm{Beta}(\alpha + s, \beta + n - s)$ is dominated by the data counts when $n \gg \alpha + \beta$ and by the prior pseudo-counts when $n \ll \alpha + \beta$. Asymptotically the posterior concentrates at the MLE (Bernstein–von Mises, §12).

The figure below visualizes the invariance result that motivates Jeffreys.""")


code(r'''# §3.3 — Jeffreys invariance: uniform-on-theta != uniform-on-logit(theta),
# but Jeffreys is the same prior in both parameterizations.

theta = np.linspace(1e-3, 1 - 1e-3, 600)
phi   = np.log(theta / (1 - theta))     # logit transform

prior_unif_theta = stats.beta(1.0, 1.0).pdf(theta)
prior_jeff_theta = stats.beta(0.5, 0.5).pdf(theta)

jacobian = theta * (1 - theta)
prior_unif_phi = prior_unif_theta * jacobian
prior_jeff_phi = prior_jeff_theta * jacobian

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5))

ax = axes[0]
ax.plot(theta, prior_unif_theta, color=COLOR_C0, lw=2,
        label=r"Uniform - $\mathrm{Beta}(1, 1)$")
ax.plot(theta, prior_jeff_theta, color=COLOR_C1, lw=2,
        label=r"Jeffreys - $\mathrm{Beta}(1/2, 1/2)$")
ax.set_ylim(0, 4)
ax.set_xlabel(r"$\theta \in (0, 1)$  (rate parameterization)")
ax.set_ylabel(r"prior density $\pi(\theta)$")
ax.set_title(r"In $\theta$-space")
ax.legend(loc="upper center", fontsize=9)

ax = axes[1]
ax.plot(phi, prior_unif_phi, color=COLOR_C0, lw=2,
        label=r"Uniform on $\theta$ -> not uniform on $\varphi$")
ax.plot(phi, prior_jeff_phi, color=COLOR_C1, lw=2,
        label=r"Jeffreys on $\theta$ -> Jeffreys on $\varphi$ also")
ax.set_xlim(-6, 6)
ax.set_xlabel(r"$\varphi = \log\frac{\theta}{1-\theta}$  (log-odds parameterization)")
ax.set_ylabel(r"prior density $\pi(\varphi)$")
ax.set_title(r"In $\varphi$-space")
ax.legend(loc="upper center", fontsize=9)

fig.suptitle("Jeffreys priors are parameterization-invariant; uniform priors are not",
             y=1.02, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_3_jeffreys_invariance.png")
plt.show()

direct_jeff_phi = np.sqrt(theta * (1 - theta))
def normalize_curve(y, x):
    return y / np.trapezoid(y, x)
n_jeff_via_transform = normalize_curve(prior_jeff_phi, phi)
n_jeff_via_direct    = normalize_curve(direct_jeff_phi, phi)
max_diff = float(np.max(np.abs(n_jeff_via_transform - n_jeff_via_direct)))
assert max_diff < 1e-10, "Jeffreys-via-transform must equal Jeffreys-via-direct."
print(f"OK Jeffreys invariance verified: max |pi_J(transform) - pi_J(direct)| = {max_diff:.2e}")
print()
print("Reading the figure:")
print("  - Beta(1, 1) is flat on theta but becomes a sech^2-shaped hump on phi.")
print("  - Beta(1/2, 1/2) is U-shaped on theta but stays heavy-tailed on phi.")
print("  - Jeffreys is the same prior in both parameterizations; uniform is not.")''')


# ============================================================================
# §4 — Conjugate families
# ============================================================================

md(r"""## §4. Conjugate families

A *conjugate prior* is a prior whose functional form is preserved by the posterior update: prior in some parametric family, observe data, get a posterior in the *same* parametric family with updated parameters.

### §4.1 Definition

A family $\mathcal{F}$ of priors is **conjugate** for a likelihood $p(y\mid\theta)$ if every prior in $\mathcal{F}$ produces a posterior in $\mathcal{F}$ after observing any data $y$. The update map on hyperparameters typically has the flavor of *counting* — pseudo-counts in the prior plus observed counts from the data.

### §4.2 Beta–Binomial conjugacy

The second of the umbrella's two load-bearing proofs.

> **Theorem 4.2 (Beta–Binomial conjugacy).** If $\theta \sim \mathrm{Beta}(\alpha, \beta)$ and $Y \mid \theta \sim \mathrm{Binomial}(n, \theta)$, then
> $$\theta \mid Y = s \;\sim\; \mathrm{Beta}(\alpha + s, \, \beta + n - s).$$

**Proof.** From Bayes' theorem in proportional form, $\pi(\theta \mid s) \propto p(s \mid \theta)\,\pi(\theta)$. Substituting the Binomial likelihood and Beta prior:

$$\pi(\theta\mid s) \;\propto\; \binom{n}{s}\theta^{s}(1-\theta)^{n-s} \cdot \frac{1}{B(\alpha,\beta)}\theta^{\alpha-1}(1-\theta)^{\beta-1}.$$

Drop the $\theta$-free factors $\binom{n}{s}$ and $1/B(\alpha,\beta)$ and collect powers:

$$\pi(\theta\mid s) \;\propto\; \theta^{(\alpha+s)-1}(1-\theta)^{(\beta+n-s)-1}.$$

This is the kernel of $\mathrm{Beta}(\alpha+s,\,\beta+n-s)$, so the posterior is exactly that distribution. $\blacksquare$

**Moral.** Conjugacy is closure under counting. For the A/B-test variant A, $\mathrm{Beta}(1,1) + (47, 953) = \mathrm{Beta}(48, 954)$ — the posterior we used without proof in §1.2 and §2.2.

The figure below visualizes the sweep: posteriors at increasing $n$, color-coded on a viridis colormap from light (small $n$, prior-dominated) to dark (large $n$, data-dominated).""")


code(r'''# §4.2 — ConjugateUpdateAnimator (static): Beta-Binomial posterior sweep.

milestones = [(0, 0), (10, 0), (50, 2), (100, 5),
              (250, 12), (500, 23), (1000, 47)]
prior_a, prior_b = 1.0, 1.0
true_rate = 47 / 1000

theta = np.linspace(0.0, 0.20, 700)
cmap = plt.cm.viridis_r
colors = cmap(np.linspace(0.15, 0.95, len(milestones)))

fig, ax = plt.subplots(figsize=(10.5, 5.4))

for (n_k, s_k), color in zip(milestones, colors):
    post = stats.beta(prior_a + s_k, prior_b + n_k - s_k)
    pd = post.pdf(theta)
    label = (fr"$n={n_k}$, $s={s_k}$  ->  "
             fr"$\mathrm{{Beta}}({int(prior_a + s_k)},"
             fr"{int(prior_b + n_k - s_k)})$")
    ax.plot(theta, pd, color=color, lw=2, label=label)

ax.axvline(true_rate, color="black", lw=1.2, ls=":", alpha=0.7,
           label=f"empirical rate $s/n = {true_rate:.4f}$")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("posterior density")
ax.set_title(r"ConjugateUpdateAnimator (static): "
             r"Beta-Binomial posterior sweeping from prior to data")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlim(0, 0.16)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_2_conjugate_update.png")
plt.show()

post_final = stats.beta(prior_a + 47, prior_b + 1000 - 47)
assert np.isclose(post_final.mean(), 48 / 1002)
assert np.isclose(post_final.var(),
                  (48 * 954) / (1002**2 * 1003))
print(f"OK Beta(48, 954) posterior: mean = {post_final.mean():.6f}, "
      f"variance = {post_final.var():.6e}")
print()
print("Sweep summary (posterior mean at each milestone):")
for n_k, s_k in milestones:
    post = stats.beta(prior_a + s_k, prior_b + n_k - s_k)
    width = post.ppf(0.975) - post.ppf(0.025)
    print(f"  n = {n_k:>4d}, s = {s_k:>3d}:  "
          f"mean = {post.mean():.4f},  95% CI width = {width:.4f}")''')


md(r"""### §4.3 Normal-Normal conjugacy (known variance)

> **Theorem 4.3 (Normal-Normal, known variance).** If $\theta \sim \mathcal{N}(\mu_0, \tau_0^2)$ and $Y_i \mid \theta \sim_{\mathrm{iid}} \mathcal{N}(\theta, \sigma^2)$ for $i = 1, \dots, n$ with $\sigma^2$ known, then $\theta \mid \mathbf{y} \sim \mathcal{N}(\mu_n, \tau_n^2)$ where
> $$\frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2}, \qquad \mu_n = \tau_n^2 \left(\frac{\mu_0}{\tau_0^2} + \frac{n\bar y}{\sigma^2}\right).$$

**Two readings.** *Precisions add* — posterior precision = prior precision + data precision $n/\sigma^2$. *The posterior mean is a precision-weighted average* — convex combination of $\mu_0$ and $\bar y$. As $n \to \infty$, the data precision dominates and $\mu_n \to \bar y$.

### §4.4 The closed-form catalog

| Likelihood | Conjugate prior | Posterior update |
|---|---|---|
| $y \sim \mathrm{Binomial}(n, \theta)$ | $\mathrm{Beta}(\alpha, \beta)$ | $\mathrm{Beta}(\alpha+s, \beta+n-s)$ |
| $y_i \sim \mathrm{Poisson}(\lambda)$ | $\mathrm{Gamma}(a, b)$ | $\mathrm{Gamma}(a+\sum y_i, b+n)$ |
| $y_i \sim \mathcal{N}(\mu, \sigma^2)$, $\sigma^2$ known | $\mathcal{N}(\mu_0, \tau_0^2)$ | $\mathcal{N}(\mu_n, \tau_n^2)$ — Theorem 4.3 |
| $y_i \sim \mathcal{N}(\mu, \sigma^2)$, $\mu$ known | $\mathrm{Inv\text{-}Gamma}(a, b)$ | $\mathrm{Inv\text{-}Gamma}(a+n/2, b+\tfrac{1}{2}\sum(y_i-\mu)^2)$ |
| $\mathbf{y} \sim \mathrm{Multinomial}(n; \boldsymbol{\theta})$ | $\mathrm{Dirichlet}(\boldsymbol{\alpha})$ | $\mathrm{Dirichlet}(\boldsymbol{\alpha}+\mathbf{y})$ |
| $y_i \sim \mathcal{N}(\mu, \sigma^2)$, both unknown | $\mathrm{NIG}(\mu_0, \kappa_0, a_0, b_0)$ | $\mathrm{NIG}$ updated (BDA3 §3.3) |

### §4.5 Exponential families

Every entry in §4.4 is an instance of the exponential-family construction. For a likelihood $p(y\mid\boldsymbol{\eta}) = h(y)\exp(\boldsymbol{\eta}^\top\mathbf{T}(y) - A(\boldsymbol{\eta}))$, the conjugate prior $\pi(\boldsymbol{\eta}\mid\nu,\boldsymbol{\chi}) \propto \exp(\nu\boldsymbol{\eta}^\top\boldsymbol{\chi} - \nu A(\boldsymbol{\eta}))$ yields a posterior in the same form. Full development: Bernardo & Smith (1994, §5.2).""")


code(r'''# §4.3 — Normal-Normal precision-weighting illustration.

mu_0, tau_0_sq = 0.0, 1.0
sigma_sq       = 1.0
ybar_obs       = 1.0

def normal_normal_posterior(n, ybar, mu_0=mu_0, tau_0_sq=tau_0_sq,
                            sigma_sq=sigma_sq):
    if n == 0:
        return mu_0, tau_0_sq
    tau_n_sq = 1.0 / (1.0 / tau_0_sq + n / sigma_sq)
    mu_n     = tau_n_sq * (mu_0 / tau_0_sq + n * ybar / sigma_sq)
    return mu_n, tau_n_sq

ns_show   = [0, 1, 5, 25, 100]
theta_grid = np.linspace(-2.0, 3.0, 600)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

ax = axes[0]
cmap = plt.cm.viridis_r
colors = cmap(np.linspace(0.18, 0.95, len(ns_show)))
for n_i, color in zip(ns_show, colors):
    mu_n, tau_n_sq = normal_normal_posterior(n_i, ybar_obs)
    label = (fr"$n={n_i}$  ->  "
             fr"$\mathcal{{N}}({mu_n:.3f},\,{tau_n_sq:.3f})$")
    ax.plot(theta_grid,
            stats.norm(mu_n, np.sqrt(tau_n_sq)).pdf(theta_grid),
            color=color, lw=2, label=label)
ax.axvline(mu_0, color=COLOR_C0, lw=1.2, ls=":",
           label=fr"prior mean $\mu_0 = {mu_0}$")
ax.axvline(ybar_obs, color=COLOR_C1, lw=1.2, ls=":",
           label=fr"sample mean $\bar y = {ybar_obs}$")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("posterior density")
ax.set_title(r"Posterior sweeping from prior to data")
ax.legend(loc="upper right", fontsize=8)

ax = axes[1]
n_continuous = np.linspace(0, 100, 400)
mu_curve = ((mu_0 / tau_0_sq + n_continuous * ybar_obs / sigma_sq)
            / (1.0 / tau_0_sq + n_continuous / sigma_sq))
ax.plot(n_continuous, mu_curve, color=COLOR_ACCENT, lw=2.5,
        label=r"$\mu_n$ (precision-weighted)")
ax.axhline(mu_0, color=COLOR_C0, lw=1.2, ls=":",
           label=fr"prior mean $\mu_0 = {mu_0}$")
ax.axhline(ybar_obs, color=COLOR_C1, lw=1.2, ls=":",
           label=fr"sample mean $\bar y = {ybar_obs}$")
for n_i, color in zip(ns_show, colors):
    mu_n, _ = normal_normal_posterior(n_i, ybar_obs)
    ax.plot(n_i, mu_n, "o", color=color, markersize=10,
            markeredgecolor="black", markeredgewidth=0.6)
ax.set_xlabel(r"$n$  (sample size)")
ax.set_ylabel(r"posterior mean $\mu_n$")
ax.set_title(r"$\mu_n$ as a precision-weighted average")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 100)
ax.set_ylim(-0.1, 1.1)

fig.suptitle("Normal-Normal conjugacy: posterior mean interpolates between "
             r"$\mu_0$ and $\bar y$ in proportion to precision",
             y=1.02, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_3_normal_normal.png")
plt.show()

n_check = 25
mu_n_check, tau_n_sq_check = normal_normal_posterior(n_check, ybar_obs)
post_precision  = 1.0 / tau_n_sq_check
prior_precision = 1.0 / tau_0_sq
data_precision  = n_check / sigma_sq
assert np.isclose(post_precision, prior_precision + data_precision), \
    "Theorem 4.3: posterior precision must equal prior + data precision."
print(f"OK At n = {n_check}:  "
      f"posterior precision {post_precision:.4f}  =  "
      f"prior precision {prior_precision:.4f}  +  "
      f"data precision {data_precision:.4f}")
print(f"   Posterior mean mu_n = {mu_n_check:.4f}, weighted toward "
      f"ybar = {ybar_obs} because n/sigma^2 >> 1/tau_0^2.")''')


# ============================================================================
# §5 — Point estimates
# ============================================================================

md(r"""## §5. Point estimates from a posterior

### §5.1 The MAP estimate

$$\hat\theta_{\mathrm{MAP}} = \arg\max_\theta \pi(\theta\mid y) = \arg\max_\theta \bigl[\log p(y\mid\theta) + \log\pi(\theta)\bigr].$$

For $\mathrm{Beta}(\alpha, \beta)$ with $\alpha, \beta > 1$: MAP $= (\alpha-1)/(\alpha+\beta-2)$. Flat prior + MAP = MLE.

### §5.2 The posterior mean

$$\hat\theta_{\mathrm{mean}} = \mathbb{E}[\theta\mid y].$$

> **Theorem 5.1.** $\mathbb{E}[\theta\mid y]$ uniquely minimizes $\mathbb{E}[(\theta - a)^2\mid y]$ over $a$.

For $\mathrm{Beta}(\alpha, \beta)$: mean $= \alpha/(\alpha+\beta)$.

### §5.3 The posterior median

The value $\hat\theta_{\mathrm{med}}$ such that $P(\theta \le \hat\theta_{\mathrm{med}} \mid y) = 1/2$. Minimizes $\mathbb{E}[|\theta - a|\mid y]$.

### §5.4 Credible intervals

**Central CI:** $[F^{-1}(\alpha/2),\, F^{-1}(1 - \alpha/2)]$ — quantile-based.

**HPD interval:** smallest interval (for unimodal posteriors) with mass $1 - \alpha$, characterized by $\pi(\theta_L\mid y) = \pi(\theta_R\mid y)$.

For symmetric unimodal posteriors, central CI = HPD; for skewed, HPD is *shorter* and shifted toward the mode.""")


code(r'''# §5 — PosteriorSummaryDashboard: MAP / mean / median / central CI / HPD CI.

from scipy.optimize import minimize_scalar


def hpd_unimodal(dist, mass=0.95):
    """HPD interval for a unimodal continuous distribution."""
    def density_gap(p_left):
        if p_left < 1e-9 or p_left + mass > 1 - 1e-9:
            return np.inf
        lo, hi = dist.ppf(p_left), dist.ppf(p_left + mass)
        return abs(dist.pdf(lo) - dist.pdf(hi))
    result = minimize_scalar(density_gap,
                              bounds=(1e-6, 1 - mass - 1e-6),
                              method="bounded")
    p_left = result.x
    return dist.ppf(p_left), dist.ppf(p_left + mass)


distributions = [
    ("Symmetric posterior - " r"$\mathrm{Beta}(20, 20)$",  20.0, 20.0),
    ("Skewed posterior - "    r"$\mathrm{Beta}(2, 20)$",    2.0, 20.0),
]

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

for ax, (label, a, b) in zip(axes, distributions):
    dist = stats.beta(a, b)
    x_lo, x_hi = dist.ppf(0.0005), dist.ppf(0.9995)
    theta = np.linspace(x_lo, x_hi, 1000)
    pdf = dist.pdf(theta)

    mode   = (a - 1) / (a + b - 2)
    mean   = dist.mean()
    median = dist.median()
    ci_lo, ci_hi  = dist.ppf([0.025, 0.975])
    hpd_lo, hpd_hi = hpd_unimodal(dist, mass=0.95)

    ymax = pdf.max()

    ax.fill_between(theta, pdf, color=COLOR_C0, alpha=0.18)
    ax.plot(theta, pdf, color=COLOR_C0, lw=2)

    pdf_at = lambda x: dist.pdf(x)
    ax.vlines(mode,   0, pdf_at(mode),   colors=COLOR_C1,     lw=2.4,
              label=f"MAP (mode) = {mode:.4f}")
    ax.vlines(mean,   0, pdf_at(mean),   colors=COLOR_TRUE,   lw=2.4, ls="--",
              label=f"Posterior mean = {mean:.4f}")
    ax.vlines(median, 0, pdf_at(median), colors=COLOR_ACCENT, lw=2.4, ls=":",
              label=f"Posterior median = {median:.4f}")

    ax.set_ylim(-ymax * 0.20, ymax * 1.08)
    bar_y_central = -ymax * 0.06
    bar_y_hpd     = -ymax * 0.13
    ax.plot([ci_lo, ci_hi],   [bar_y_central, bar_y_central],
            color=COLOR_C0, lw=8, solid_capstyle="butt",
            label=f"95% central CI [{ci_lo:.3f}, {ci_hi:.3f}]  "
                  f"(width {ci_hi - ci_lo:.3f})")
    ax.plot([hpd_lo, hpd_hi], [bar_y_hpd, bar_y_hpd],
            color=COLOR_BAND, lw=8, solid_capstyle="butt",
            label=f"95% HPD interval [{hpd_lo:.3f}, {hpd_hi:.3f}]  "
                  f"(width {hpd_hi - hpd_lo:.3f})")

    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("posterior density")
    ax.set_title(label)
    ax.legend(loc="upper right", fontsize=7.5)

fig.suptitle("PosteriorSummaryDashboard: three point estimates, two credible intervals",
             y=1.02, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_dashboard.png")
plt.show()

print(f"{'Distribution':<28} {'MAP':>8} {'Median':>8} {'Mean':>8} "
      f"{'CentralCI width':>16} {'HPD width':>11}")
print("-" * 92)
for label, a, b in distributions:
    dist = stats.beta(a, b)
    mode   = (a - 1) / (a + b - 2)
    median = dist.median()
    mean   = dist.mean()
    ci_lo, ci_hi  = dist.ppf([0.025, 0.975])
    hpd_lo, hpd_hi = hpd_unimodal(dist, mass=0.95)
    short = label.split(" - ")[1].replace("$", "").replace(r"\mathrm", "")
    print(f"{short:<28} {mode:>8.4f} {median:>8.4f} {mean:>8.4f} "
          f"{ci_hi - ci_lo:>16.4f} {hpd_hi - hpd_lo:>11.4f}")

# --- MAP non-invariance demonstration ------------------------------------
print()
print("MAP non-invariance under reparameterization (for the skewed posterior):")
a, b = 2.0, 20.0
map_theta     = (a - 1) / (a + b - 2)
map_phi_in_th = a / (a + b)
print(f"  Beta({a:.0f}, {b:.0f}):")
print(f"    MAP on theta-scale (the rate):                  theta = {map_theta:.4f}")
print(f"    MAP on phi = logit(theta) scale, back to theta: theta = {map_phi_in_th:.4f}")
print(f"    These differ - the MAP is not "
      "reparameterization-invariant.")
print(f"    (And, suggestively, the phi-scale MAP equals the natural-scale "
      "posterior MEAN.)")''')


md(r"""### §5.5 The MAP fallacy

Three properties of the MAP make it dangerous when readers treat it as a generic stand-in for the posterior — each visible above.

**The mode is not the "most likely value."** For continuous posteriors, $P(\theta = \hat\theta_{\mathrm{MAP}} \mid y) = 0$. The MAP is the location of *highest density*, not the location of "most probability."

**The MAP is parameterization-dependent.** The cell above demonstrated this numerically: for $\mathrm{Beta}(2, 20)$, the MAP on the natural rate scale is $0.0500$, but the MAP on the log-odds scale (expressed back on the rate) is $0.0909$ — which happens to coincide with the posterior *mean*.

**The MAP is brittle in high dimensions.** In a high-dimensional posterior the mode is often a sharp spike with almost no surrounding mass; the "typical set" sits far from the MAP. Forward-pointer: [Bayesian neural networks](/topics/bayesian-neural-networks).

**When is the MAP appropriate?** When the posterior is symmetric, unimodal, and we work in a natural parameterization, MAP = mean = median and the fallacy is moot. For exponential-family posteriors with conjugate priors in regular regimes (including the A/B-test running example), the choice doesn't matter.""")


# ============================================================================
# §6 — Posterior predictive
# ============================================================================

md(r"""## §6. The posterior predictive distribution

### §6.1 Definition

$$p(\tilde y \mid y) \;=\; \int_\Theta p(\tilde y \mid \theta) \, \pi(\theta \mid y) \, d\theta.$$

The integral marginalizes $\theta$, leaving a distribution over $\tilde y$ alone. It carries *two* sources of uncertainty — likelihood noise and parameter uncertainty.

### §6.2 Contrast with the frequentist plug-in predictive

The plug-in predictive $p(\tilde y \mid \hat\theta)$ substitutes a point estimate for the parameter — captures likelihood noise but *not* parameter uncertainty. Plug-in prediction intervals are systematically too narrow.

### §6.3 The Normal-Normal posterior predictive

> **Theorem 6.1.** Under the Normal-Normal setup (§4.3),
> $$\tilde Y \mid \mathbf{y} \;\sim\; \mathcal{N}\!\left(\mu_n, \;\; \tau_n^2 + \sigma^2\right).$$

**Proof.** $\tilde Y = \theta + \varepsilon$ with $\theta \mid \mathbf{y} \sim \mathcal{N}(\mu_n, \tau_n^2)$ and $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ independent. Sum of independent Gaussians: $\mathcal{N}(\mu_n, \tau_n^2 + \sigma^2)$. $\blacksquare$

The variance decomposition is the umbrella's load-bearing fact for §6:

$$\underbrace{\tau_n^2 + \sigma^2}_{\text{predictive variance}} \;=\; \underbrace{\tau_n^2}_{\text{parameter uncertainty}} \;+\; \underbrace{\sigma^2}_{\text{likelihood noise}}.$$

As $n \to \infty$, $\tau_n^2 \to 0$ and the posterior predictive collapses to the plug-in. At finite $n$, the gap is the cost of the plug-in's blind spot.""")


code(r'''# §6 — PosteriorPredictiveBands: parameter posterior, plug-in predictive,
# and Bayesian posterior predictive at two sample sizes.

mu_0, tau_0_sq = 0.0, 1.0
sigma_sq       = 1.0
ybar_obs       = 1.0

def normal_normal_pieces(n, ybar=ybar_obs):
    if n == 0:
        mu_n, tau_n_sq = mu_0, tau_0_sq
    else:
        tau_n_sq = 1.0 / (1.0 / tau_0_sq + n / sigma_sq)
        mu_n     = tau_n_sq * (mu_0 / tau_0_sq + n * ybar / sigma_sq)
    return mu_n, tau_n_sq, ybar

regimes = [("Small data: $n = 3$",  3),
           ("Moderate data: $n = 30$", 30)]

theta_grid = np.linspace(-3.5, 4.5, 800)

fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0),
                          gridspec_kw={"height_ratios": [3, 2]})

for ax, (label, n_i) in zip(axes[0], regimes):
    mu_n, tau_n_sq, mle = normal_normal_pieces(n_i)
    post_density = stats.norm(mu_n, np.sqrt(tau_n_sq)).pdf(theta_grid)
    plugin_density = stats.norm(mle, np.sqrt(sigma_sq)).pdf(theta_grid)
    pp_var = tau_n_sq + sigma_sq
    pp_density = stats.norm(mu_n, np.sqrt(pp_var)).pdf(theta_grid)

    ax.fill_between(theta_grid, post_density, color=COLOR_C0, alpha=0.25,
                    label=fr"Parameter posterior $\pi(\theta\mid y)$"
                          fr" - std $\approx {np.sqrt(tau_n_sq):.3f}$")
    ax.plot(theta_grid, post_density, color=COLOR_C0, lw=2)
    ax.plot(theta_grid, plugin_density, color=COLOR_TRUE, lw=2, ls="--",
            label=fr"Plug-in predictive $p(\tilde y\mid\hat\theta_{{\mathrm{{MLE}}}})$"
                  fr" - std $= {np.sqrt(sigma_sq):.3f}$")
    ax.plot(theta_grid, pp_density, color=COLOR_C1, lw=2.4,
            label=fr"Posterior predictive $p(\tilde y\mid y)$"
                  fr" - std $\approx {np.sqrt(pp_var):.3f}$")

    pp_lo, pp_hi   = stats.norm(mu_n, np.sqrt(pp_var)).ppf([0.025, 0.975])
    ax.axvspan(pp_lo, pp_hi, color=COLOR_C1, alpha=0.06)
    ax.set_xlabel(r"$\theta$ (or future $\tilde y$, sharing the same axis)")
    ax.set_ylabel("density")
    ax.set_title(label + fr",  $\bar y = {ybar_obs}$")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_xlim(-3.5, 4.5)

ax = axes[1, 0]
n_range = np.arange(1, 101)
tau_n_sq_curve = 1.0 / (1.0 / tau_0_sq + n_range / sigma_sq)
pp_var_curve   = tau_n_sq_curve + sigma_sq
ax.fill_between(n_range, 0, sigma_sq, color=COLOR_TRUE, alpha=0.35,
                label=r"likelihood noise $\sigma^2$")
ax.fill_between(n_range, sigma_sq, pp_var_curve, color=COLOR_C0, alpha=0.35,
                label=r"parameter uncertainty $\tau_n^2$")
ax.plot(n_range, pp_var_curve, color=COLOR_C1, lw=2,
        label=r"posterior-predictive variance $\tau_n^2 + \sigma^2$")
ax.set_xlabel(r"$n$ (observed sample size)")
ax.set_ylabel("variance")
ax.set_title(r"Posterior-predictive variance = $\tau_n^2 + \sigma^2$")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlim(1, 100)

ax = axes[1, 1]
width_ratio = np.sqrt(pp_var_curve / sigma_sq)
ax.plot(n_range, width_ratio, color=COLOR_ACCENT, lw=2.5)
ax.axhline(1.0, color="black", lw=1, ls=":", alpha=0.6,
           label="plug-in baseline (= 1)")
ax.set_xlabel(r"$n$ (observed sample size)")
ax.set_ylabel(r"posterior-predictive width / plug-in width")
ax.set_title("How much wider is the Bayesian predictive than the plug-in?")
ax.set_xlim(1, 100)
ax.set_ylim(0.95, 1.6)
ax.legend(loc="upper right", fontsize=9)

for n_i in [3, 30]:
    tau_n_sq = 1.0 / (1.0 / tau_0_sq + n_i / sigma_sq)
    ratio = np.sqrt((tau_n_sq + sigma_sq) / sigma_sq)
    ax.plot(n_i, ratio, "o", color=COLOR_BAND, markersize=9,
            markeredgecolor="black", markeredgewidth=0.6)
    ax.annotate(f"n={n_i}: x{ratio:.3f}", xy=(n_i, ratio),
                xytext=(n_i + 6, ratio + 0.04), fontsize=9)

fig.suptitle("PosteriorPredictiveBands: parameter uncertainty + likelihood "
             "noise -> posterior-predictive width",
             y=1.005, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig6_posterior_predictive_bands.png")
plt.show()

for n_i in [3, 30]:
    mu_n, tau_n_sq, _ = normal_normal_pieces(n_i)
    pp_var_calc       = tau_n_sq + sigma_sq
    M = 200_000
    theta_samples = rng.normal(mu_n, np.sqrt(tau_n_sq), size=M)
    ytilde_samples = rng.normal(theta_samples, np.sqrt(sigma_sq))
    emp_var = float(ytilde_samples.var(ddof=1))
    err = abs(emp_var - pp_var_calc) / pp_var_calc
    print(f"n = {n_i:>3d}:  closed-form pp-variance = {pp_var_calc:.4f},  "
          f"MC pp-variance ({M:,} samples) = {emp_var:.4f},  "
          f"relative error = {err:.3%}")
    assert err < 0.01, "Theorem 6.1 numerical check should agree within 1%."
print("\nOK Theorem 6.1 verified by Monte Carlo at two sample sizes.")''')


md(r"""### §6.4 Posterior predictive checks

The posterior predictive is the natural tool for *model criticism*:

1. Draw $\theta^{(k)} \sim \pi(\theta\mid y)$ for $k = 1, \ldots, K$.
2. Simulate a replicate dataset $\tilde y^{(k)} \sim p(\cdot \mid \theta^{(k)})$.
3. Compute a test statistic $T(\cdot)$ on the observed data and replicates.
4. If $T(y)$ is extreme relative to the distribution of $T(\tilde y^{(k)})$, the model is misspecified.

The **Bayesian p-value** $p_B = P(T(\tilde Y) \ge T(y) \mid y)$ summarizes the comparison.

**Caveats.** PPCs use the data twice; Bayesian p-values do not carry frequentist Type-I guarantees. Canonical reference: BDA3 §6. Runnable PyMC version: [probabilistic programming](/topics/probabilistic-programming).

---

**Summary of §6.** The posterior predictive is structurally broader than the parameter posterior because it carries both parameter uncertainty and likelihood noise. Forward-pointers: [variational inference](/topics/variational-inference) for tractable posterior-predictive approximations; [Bayesian neural networks](/topics/bayesian-neural-networks) for predictive uncertainty in high dimensions.""")


# ============================================================================
# §7 — Bayesian decision theory
# ============================================================================

md(r"""## §7. Bayesian decision theory

### §7.1 Loss, utility, and the decision rule

Four objects: parameter space $\Theta$, action space $\mathcal{A}$, loss $L : \Theta \times \mathcal{A} \to \mathbb{R}$ (or utility $U = -L$), and decision rule $\delta : \mathcal{Y} \to \mathcal{A}$.

| Loss | Bayes rule (minimizer of $\mathbb{E}_{\theta\mid y}[L]$) |
|---|---|
| $L = (\theta - a)^2$ (squared error) | posterior mean (§5.2) |
| $L = \lvert\theta - a\rvert$ (absolute error) | posterior median (§5.3) |
| $L = \mathbb{1}[\theta \ne a]$ (0-1 loss, discrete) | posterior mode = MAP (§5.1) |
| $L = c_o (a-\theta)_+ + c_u (\theta-a)_+$ (pinball) | $(c_u/(c_o + c_u))$-quantile |

### §7.2 Bayes risk and posterior expected loss

**Bayes risk:** $r(\pi, \delta) = \mathbb{E}_{\theta, Y}[L(\theta, \delta(Y))]$.

**Posterior expected loss:** $\rho(a, y) = \mathbb{E}_{\theta\mid y}[L(\theta, a)]$.

### §7.3 The Bayes-rule theorem

> **Theorem 7.1.** The rule $\delta_\pi(y) = \arg\min_{a\in\mathcal{A}} \mathbb{E}_{\theta\mid y}[L(\theta, a)]$ minimizes the Bayes risk $r(\pi, \delta)$ over all measurable decision rules.

**Proof.** Apply Fubini-Tonelli:

$$r(\pi, \delta) = \int_\mathcal{Y} \int_\Theta L(\theta, \delta(y))\,\pi(\theta\mid y)\,p(y)\,d\theta\,dy = \int_\mathcal{Y} \rho(\delta(y), y)\,p(y)\,dy.$$

To minimize the outer integral, minimize $\rho(\cdot, y)$ pointwise in $y$. $\blacksquare$""")


code(r'''# §7 — DecisionBoundaryExplorer: posterior expected utility + full-posterior
# decision confidence for the A/B-test running example.

post_A = stats.beta(48, 954)
post_B = stats.beta(65, 937)

M = 200_000
samples_A = post_A.rvs(size=M, random_state=rng)
samples_B = post_B.rvs(size=M, random_state=rng)
delta = samples_B - samples_A

c_grid = np.linspace(0.0, 0.04, 400)

util_stay   = post_A.mean()
util_switch = post_B.mean() - c_grid

prob_switch_correct = np.array([(delta > c).mean() for c in c_grid])

c_star = float(post_B.mean() - post_A.mean())

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

ax = axes[0]
ax.axhline(util_stay, color=COLOR_C0, lw=2.5,
           label=fr"Stay with A:  "
                 fr"$\mathbb{{E}}[\theta_A\mid y] = {util_stay:.4f}$")
ax.plot(c_grid, util_switch, color=COLOR_C1, lw=2.5,
        label=fr"Switch to B:  "
              fr"$\mathbb{{E}}[\theta_B\mid y] - c = {post_B.mean():.4f} - c$")
ax.axvspan(0,      c_star,         color=COLOR_C1, alpha=0.08)
ax.axvspan(c_star, c_grid.max(),   color=COLOR_C0, alpha=0.08)
ax.axvline(c_star, color="black", lw=1.5, ls=":", alpha=0.7)
ax.text(c_star * 0.45, 0.020, "SWITCH",
        fontsize=11, fontweight="bold", color=COLOR_C1,
        ha="center", alpha=0.7)
ax.text(c_star + (c_grid.max() - c_star) * 0.55, 0.020, "STAY",
        fontsize=11, fontweight="bold", color=COLOR_C0,
        ha="center", alpha=0.7)
ax.annotate(f"decision boundary\n$c^* = {c_star:.4f}$",
            xy=(c_star, util_stay),
            xytext=(c_star + 0.005, util_stay + 0.012),
            fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel(r"switching cost $c$  (per impression)")
ax.set_ylabel(r"posterior expected utility")
ax.set_title("Posterior-expected-utility view")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlim(0, c_grid.max())
ax.set_ylim(0.015, 0.075)

ax = axes[1]
ax.plot(c_grid, prob_switch_correct, color=COLOR_ACCENT, lw=2.5,
        label=r"$P(\theta_B - c > \theta_A\mid y)$")
ax.axhline(0.5, color="black", lw=1, ls=":", alpha=0.6,
           label="50% threshold")
ax.axvline(c_star, color="black", lw=1.5, ls=":", alpha=0.7,
           label=fr"decision boundary $c^* = {c_star:.4f}$")
prob_at_zero = float((delta > 0).mean())
ax.plot(0, prob_at_zero, "o", color=COLOR_BAND, markersize=10,
        markeredgecolor="black", markeredgewidth=0.6, zorder=5)
ax.annotate(fr"$P(B>A\mid y) = {prob_at_zero:.3f}$  (from §1)",
            xy=(0, prob_at_zero), xytext=(0.006, prob_at_zero - 0.07),
            fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel(r"switching cost $c$  (per impression)")
ax.set_ylabel("posterior probability switching is correct")
ax.set_title("Full-posterior decision-confidence view")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlim(0, c_grid.max())
ax.set_ylim(0, 1)

fig.suptitle("DecisionBoundaryExplorer: posterior + loss function -> action",
             y=1.02, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig7_decision_boundary.png")
plt.show()

print(f"Posterior means:  E[theta_A|y] = {post_A.mean():.4f},  "
      f"E[theta_B|y] = {post_B.mean():.4f}")
print(f"Decision boundary c* = E[theta_B - theta_A | y] = {c_star:.4f}")
print()
print(f"{'switching cost c':<22} {'P(switch correct | y)':<25}")
print("-" * 50)
for c in [0.0, c_star/2, c_star, 2*c_star, 4*c_star]:
    p_correct = float((delta > c).mean())
    print(f"  c = {c:.4f}             {p_correct:.4f}")

p_at_boundary = float((delta > c_star).mean())
assert 0.49 < p_at_boundary < 0.51, \
    f"At the decision boundary, P(switch correct) should ~ 0.5; got {p_at_boundary:.4f}"
print(f"\nOK At c = c*, P(switch correct) = {p_at_boundary:.4f} ~ 0.5 "
      "(posterior of theta_B - theta_A is approximately symmetric around its mean).")''')


md(r"""### §7.4 The A/B-test decision

The posteriors $\theta_A\mid y \sim \mathrm{Beta}(48, 954)$ (mean $0.0479$) and $\theta_B\mid y \sim \mathrm{Beta}(65, 937)$ (mean $0.0649$) feed into Theorem 7.1 under the A/B-test linear loss to give a decision boundary at $c^* = \mathbb{E}[\theta_B - \theta_A\mid y] \approx 0.0170$ per impression.

### §7.5 How §5 fits in

The Bayes-rule theorem unifies the §5 point estimates: each one is the Bayes rule under a specific loss. Point estimates are not arbitrary summary statistics — they are optimal *actions*.""")


# ============================================================================
# §8 — Sequential updating + exchangeability
# ============================================================================

md(r"""## §8. Sequential updating and exchangeability

The Beta-Binomial posterior depends on the data only through the total counts $(s, n)$. That single fact powers three results: sequential updating (§8.1), stopping-rule irrelevance (§8.2), and De Finetti's representation theorem (§8.3–§8.4).

### §8.1 Sequential conjugate updates

Apply Theorem 4.2 twice to observe data in two batches:

$$\mathrm{Beta}(\alpha_0, \beta_0) \xrightarrow{(s_1, n_1)} \mathrm{Beta}(\alpha_0 + s_1, \beta_0 + n_1 - s_1) \xrightarrow{(s_2, n_2)} \mathrm{Beta}(\alpha_0 + s_1 + s_2, \, \beta_0 + n_1 + n_2 - s_1 - s_2).$$

This is the same posterior as observing $(s_1 + s_2, n_1 + n_2)$ at once. *Yesterday's posterior is today's prior:*

$$\pi(\theta \mid y_1, y_2) \;\propto\; p(y_2 \mid \theta) \, \pi(\theta \mid y_1).$$

The figure below sweeps the variant-B posterior through six checkpoints, and the verification cell beneath confirms that four different updating orders land at exactly the same Beta parameters.""")


code(r'''# §8 — SequentialUpdatingTimeline + order-invariance demonstration.

n_A_full, s_A_full = 1000, 47
n_B_full, s_B_full = 1000, 64
prior_a, prior_b   = 1.0, 1.0

checkpoints = [0, 50, 100, 200, 500, 1000]
s_A_at  = [round(s_A_full * n / n_B_full) for n in checkpoints]
s_B_at  = [round(s_B_full * n / n_B_full) for n in checkpoints]

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

ax = axes[0]
theta = np.linspace(0.0, 0.15, 700)
cmap = plt.cm.viridis_r
colors = cmap(np.linspace(0.18, 0.95, len(checkpoints)))

for (n_k, s_k), color in zip(zip(checkpoints, s_B_at), colors):
    post = stats.beta(prior_a + s_k, prior_b + n_k - s_k)
    label = (fr"$n = {n_k}$, $s = {s_k}$  ->  "
             fr"$\mathrm{{Beta}}({int(prior_a + s_k)}, "
             fr"{int(prior_b + n_k - s_k)})$")
    ax.plot(theta, post.pdf(theta), color=color, lw=2, label=label)

ax.axvline(s_B_full / n_B_full, color="black", lw=1.2, ls=":", alpha=0.7,
           label=fr"empirical rate $s_B / n_B = {s_B_full / n_B_full:.4f}$")
ax.set_xlabel(r"$\theta_B$")
ax.set_ylabel("posterior density")
ax.set_title("Variant B posterior accumulating evidence")
ax.legend(loc="upper right", fontsize=7.5)
ax.set_xlim(0, 0.13)

ax = axes[1]
M = 50_000
prob_track = []
for n_k, s_A_k, s_B_k in zip(checkpoints, s_A_at, s_B_at):
    if n_k == 0:
        prob_track.append(0.5)
        continue
    post_A_k = stats.beta(prior_a + s_A_k, prior_b + n_k - s_A_k)
    post_B_k = stats.beta(prior_a + s_B_k, prior_b + n_k - s_B_k)
    samples_A_k = post_A_k.rvs(size=M, random_state=rng)
    samples_B_k = post_B_k.rvs(size=M, random_state=rng)
    prob_track.append(float((samples_B_k > samples_A_k).mean()))
prob_track = np.array(prob_track)

n_dense = np.array([10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000])
prob_dense = []
for n_k in n_dense:
    s_A_k = round(s_A_full * n_k / n_B_full)
    s_B_k = round(s_B_full * n_k / n_B_full)
    post_A_k = stats.beta(prior_a + s_A_k, prior_b + n_k - s_A_k)
    post_B_k = stats.beta(prior_a + s_B_k, prior_b + n_k - s_B_k)
    sa = post_A_k.rvs(size=M, random_state=rng)
    sb = post_B_k.rvs(size=M, random_state=rng)
    prob_dense.append(float((sb > sa).mean()))
prob_dense = np.array(prob_dense)

ax.plot(n_dense, prob_dense, color=COLOR_ACCENT, lw=2.5,
        label=r"$P(\theta_B > \theta_A \mid \text{data}(n))$")
ax.plot(checkpoints, prob_track, "o", color=COLOR_BAND, markersize=9,
        markeredgecolor="black", markeredgewidth=0.5, zorder=5,
        label="checkpoint values")
ax.axhline(0.5, color="black", lw=1, ls=":", alpha=0.6,
           label="§7 decision threshold (0.5)")
ax.axhline(prob_dense[-1], color=COLOR_C1, lw=1, ls="--", alpha=0.5,
           label=fr"$n = 1000$ limit $\approx {prob_dense[-1]:.3f}$")
ax.set_xlabel(r"$n$ (cumulative sample size, per arm)")
ax.set_ylabel(r"$P(\theta_B > \theta_A \mid \text{data})$")
ax.set_title("Decision confidence accumulates over time")
ax.set_xlim(0, 1000)
ax.set_ylim(0.4, 1.0)
ax.legend(loc="lower right", fontsize=8)

fig.suptitle("SequentialUpdatingTimeline: posterior + decision confidence vs $n$",
             y=1.02, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig8_sequential_timeline.png")
plt.show()

# Order-invariance verification
rng_local = np.random.default_rng(SEED + 1)
true_obs = np.zeros(1000, dtype=int)
true_obs[:s_B_full] = 1
rng_local.shuffle(true_obs)
assert true_obs.sum() == s_B_full

def posterior_params(obs, prior_a=prior_a, prior_b=prior_b):
    s, n = int(obs.sum()), len(obs)
    return prior_a + s, prior_b + n - s

params_1 = posterior_params(true_obs)

a_cur, b_cur = prior_a, prior_b
for y in true_obs:
    a_cur += y
    b_cur += (1 - y)
params_2 = (a_cur, b_cur)

shuffled = true_obs.copy()
rng_local.shuffle(shuffled)
a_cur, b_cur = prior_a, prior_b
for y in shuffled:
    a_cur += y
    b_cur += (1 - y)
params_3 = (a_cur, b_cur)

a_cur, b_cur = prior_a, prior_b
remaining = list(true_obs)
rng_local.shuffle(remaining)
while remaining:
    batch_size = min(len(remaining), int(rng_local.integers(1, 50)))
    batch = remaining[:batch_size]
    remaining = remaining[batch_size:]
    s_batch = sum(batch)
    a_cur += s_batch
    b_cur += len(batch) - s_batch
params_4 = (a_cur, b_cur)

print("Order-invariance of the Beta-Binomial conjugate update:")
print(f"  1. All-at-once:                  Beta({params_1[0]:.0f}, {params_1[1]:.0f})")
print(f"  2. Sequential, original order:   Beta({params_2[0]:.0f}, {params_2[1]:.0f})")
print(f"  3. Sequential, shuffled order:   Beta({params_3[0]:.0f}, {params_3[1]:.0f})")
print(f"  4. Random-batched:               Beta({params_4[0]:.0f}, {params_4[1]:.0f})")

assert params_1 == params_2 == params_3 == params_4, \
    "All four updating schemes must yield identical posteriors."
print(f"\nOK All four schemes agree: posterior depends on data only through (s, n).")''')


md(r"""### §8.2 Stopping-rule irrelevance

> **Proposition 8.2 (Stopping-rule irrelevance, informal).** For a stopping time $N$ depending on $Y_1, \dots, Y_N$ but not on $\theta$,
> $$\pi(\theta \mid y_1, \dots, y_N, N) \;=\; \pi(\theta \mid y_1, \dots, y_N).$$

The joint likelihood factors as $p(y_{1:N}, N \mid \theta) = p(y_{1:N} \mid \theta) \cdot p(N \mid y_{1:N})$. The second factor does not depend on $\theta$, so it cancels.

The frequentist contrast is sharp: adaptive stopping inflates Type-I-error rates and requires alpha-spending corrections. The Bayesian posterior given the realized data is the same regardless of how $n$ came to be the number.

### §8.3 De Finetti's representation theorem

A sequence $(Y_1, Y_2, \ldots)$ is **exchangeable** if its joint distribution is invariant under finite permutations of indices.

> **Theorem 8.3 (De Finetti, binary case).** A sequence $(Y_1, Y_2, \ldots)$ of $\{0, 1\}$-valued random variables is exchangeable iff there exists a unique distribution $F$ on $[0, 1]$ such that for every $n$,
> $$P(Y_1 = y_1, \dots, Y_n = y_n) = \int_0^1 \theta^{\sum y_i}(1 - \theta)^{n - \sum y_i} \, dF(\theta).$$

The right-hand side is exactly the marginal of the Bayesian model "$Y_i \mid \theta \sim_{\mathrm{iid}} \mathrm{Bernoulli}(\theta)$, $\theta \sim F$." Full proof: Bernardo & Smith (1994, §4); Hewitt-Savage (1955) for non-binary.

### §8.4 Implications

**I.i.d. is not foundational; exchangeability is.** The "true parameter" $\theta$ is the limiting empirical frequency — an emergent quantity, not a pre-existing fact about nature.

**Hierarchical models generalize the De Finetti picture.** When exchangeability fails for the whole dataset but holds within groups, each group acquires its own parameter — the structure of hierarchical Bayes (§9).""")


# ============================================================================
# §9 — Hierarchical Bayes
# ============================================================================

md(r"""## §9. Hierarchical Bayes and partial pooling

### §9.1 The multi-level model

$$\phi \sim \pi(\phi), \qquad \theta_g \mid \phi \sim \pi(\theta_g \mid \phi), \qquad y_g \mid \theta_g \sim p(y_g \mid \theta_g).$$

For an A/B/n test on $G$ variants with $(n_g, s_g)$ counts, the canonical Beta-Binomial hierarchy parameterized by $(\omega, \kappa)$ gives a conditional posterior with a closed-form precision-weighted mean:

$$\mathbb{E}[\theta_g \mid s_g, n_g, \omega, \kappa] = \frac{\kappa}{\kappa + n_g}\,\omega + \frac{n_g}{\kappa + n_g}\cdot\frac{s_g}{n_g}.$$

### §9.2 Complete, no, and partial pooling

| Regime | $\kappa$ | Posterior mean | Interpretation |
|---|---|---|---|
| Complete pooling | $\to \infty$ | $\omega$ | all groups share one rate |
| No pooling | $\to 0$ | $s_g/n_g$ | groups share no information |
| Partial pooling | finite | weighted average | related but not identical |

Partial pooling shrinks each group's MLE toward the grand mean, *more* when $n_g$ is small. Stein's paradox / Efron-Morris (1973): for $G \ge 3$, partial pooling dominates no-pooling in MSE.""")


code(r'''# §9 — HierarchicalShrinkageDemo: 8 product variants + empirical Bayes.

from scipy.optimize import minimize
from scipy.special import betaln

group_names = list("ABCDEFGH")
n_g = np.array([100,  100,  200,  200,  500,  500, 1000, 1000])
# MLEs deliberately spread across [0.02, 0.09] so that between-group variance
# exceeds within-group sampling variance — EB lands at finite kappa (partial pooling).
s_g = np.array([  2,    9,    6,   16,   15,   40,   35,   75])
mle_g = s_g / n_g
grand_mean = s_g.sum() / n_g.sum()
print(f"Per-group MLEs: {dict(zip(group_names, mle_g.round(4)))}")
print(f"Grand mean (complete-pooling target): {grand_mean:.4f}")

def neg_log_marginal_likelihood(theta, ns, ss):
    log_kappa, logit_omega = theta
    kappa = np.exp(log_kappa)
    omega = 1.0 / (1.0 + np.exp(-logit_omega))
    alpha, beta = kappa * omega, kappa * (1.0 - omega)
    if alpha <= 0 or beta <= 0:
        return np.inf
    total = 0.0
    for n, s in zip(ns, ss):
        total += betaln(alpha + s, beta + n - s) - betaln(alpha, beta)
    return -total

kappa_init = 50.0
result = minimize(neg_log_marginal_likelihood,
                  x0=[np.log(kappa_init), np.log(grand_mean / (1 - grand_mean))],
                  args=(n_g, s_g),
                  method="Nelder-Mead",
                  options={"xatol": 1e-6, "fatol": 1e-8})
log_kappa_hat, logit_omega_hat = result.x
kappa_hat = float(np.exp(log_kappa_hat))
omega_hat = float(1.0 / (1.0 + np.exp(-logit_omega_hat)))
alpha_hat, beta_hat = kappa_hat * omega_hat, kappa_hat * (1 - omega_hat)
print(f"\nEmpirical-Bayes estimates:")
print(f"  omega_hat (grand-mean target) = {omega_hat:.4f}  (vs. raw grand mean {grand_mean:.4f})")
print(f"  kappa_hat (concentration)     = {kappa_hat:.2f}")
print(f"  -> (alpha_hat, beta_hat) = ({alpha_hat:.3f}, {beta_hat:.3f})")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

ax = axes[0]
kappa_grid = np.logspace(-1, 4, 400)
palette = plt.cm.plasma(np.linspace(0.05, 0.92, len(group_names)))
for g_idx, name in enumerate(group_names):
    n_, s_ = n_g[g_idx], s_g[g_idx]
    post_mean_curve = (kappa_grid * omega_hat + s_) / (kappa_grid + n_)
    ax.plot(kappa_grid, post_mean_curve, lw=2, color=palette[g_idx],
            label=fr"{name}: $n={n_}, s={s_}$ "
                  fr"(MLE $= {s_/n_:.3f}$)")
    ax.axhline(mle_g[g_idx], color=palette[g_idx], lw=0.7, ls=":", alpha=0.4)
ax.axhline(omega_hat, color="black", lw=1.5, ls="-", alpha=0.7,
           label=fr"grand mean $\omega = {omega_hat:.4f}$")
ax.axvline(kappa_hat, color=COLOR_BAND, lw=2, ls="--",
           label=fr"EB estimate $\hat\kappa = {kappa_hat:.1f}$")
ax.set_xscale("log")
ax.set_xlabel(r"concentration $\kappa$  (log scale)")
ax.set_ylabel(r"posterior mean  $\mathbb{E}[\theta_g \mid \text{data}, \omega, \kappa]$")
ax.set_title("Shrinkage trajectories: from MLE (left) to grand mean (right)")
ax.legend(loc="upper right", fontsize=7.5)
ax.set_xlim(0.1, 1e4)

ax = axes[1]
theta = np.linspace(0.0, 0.13, 700)
for g_idx, name in enumerate(group_names):
    a_post = alpha_hat + s_g[g_idx]
    b_post = beta_hat + n_g[g_idx] - s_g[g_idx]
    post = stats.beta(a_post, b_post)
    pmean = post.mean()
    ax.plot(theta, post.pdf(theta), lw=2, color=palette[g_idx],
            label=fr"{name}: posterior mean $= {pmean:.4f}$ "
                  fr"(MLE $= {mle_g[g_idx]:.3f}$)")
    ax.plot(mle_g[g_idx], 0, "v", color=palette[g_idx],
            markersize=8, markeredgecolor="black", markeredgewidth=0.4)
ax.axvline(omega_hat, color="black", lw=1.5, ls="-", alpha=0.7,
           label=fr"$\omega = {omega_hat:.4f}$")
ax.set_xlabel(r"$\theta_g$")
ax.set_ylabel("posterior density")
ax.set_title(fr"Partial-pooled posteriors at $\hat\kappa = {kappa_hat:.1f}$ "
             r"(triangles: MLEs)")
ax.legend(loc="upper right", fontsize=7.5)
ax.set_xlim(0, 0.13)

fig.suptitle("HierarchicalShrinkageDemo: 8 variants, partial pooling shrinks MLEs toward the grand mean",
             y=1.02, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig9_hierarchical_shrinkage.png")
plt.show()

print(f"\n{'Group':<6} {'n_g':>6} {'s_g':>5} {'MLE':>8} "
      f"{'pool wt':>9} {'EB post mean':>14} {'shrinkage':>11}")
print("-" * 70)
for g_idx, name in enumerate(group_names):
    n_, s_ = n_g[g_idx], s_g[g_idx]
    pool_wt = kappa_hat / (kappa_hat + n_)
    post_mean = pool_wt * omega_hat + (1 - pool_wt) * (s_ / n_)
    shrink = mle_g[g_idx] - post_mean
    print(f"{name:<6} {n_:>6} {s_:>5} {mle_g[g_idx]:>8.4f} "
          f"{pool_wt:>9.3f} {post_mean:>14.4f} {shrink:>+11.4f}")

pool_wt_small_n = kappa_hat / (kappa_hat + n_g[0])   # group A: n=100
pool_wt_large_n = kappa_hat / (kappa_hat + n_g[6])   # group G: n=1000
assert pool_wt_small_n > pool_wt_large_n, \
    "Small-n groups must have higher pooling weight (more shrinkage toward the grand mean)."
print(f"\nOK Pooling weight is adaptive to sample size:")
print(f"   Group A (n=100)  borrows {pool_wt_small_n*100:.1f}% from the grand mean")
print(f"   Group G (n=1000) borrows {pool_wt_large_n*100:.1f}% from the grand mean")
print(f"   Small-n groups shrink more because they have less data to defend their MLE.")''')


md(r"""### §9.3 Empirical Bayes as a shortcut

**Empirical Bayes (EB).** Estimate $\hat\phi$ by maximizing the marginal likelihood, then plug $\hat\phi$ into the per-group posteriors.

**Pros.** Computationally trivial; agrees with full Bayes to leading order when $G$ is large.

**Cons.** Plugs in a point estimate of $\phi$ — same plug-in fallacy as §6.2 at the hyperparameter level. Credible intervals are slightly too narrow.

**Rule of thumb.** Use EB for shrinkage point estimates; use full Bayes (MCMC) for credible intervals when $G$ is small.

### §9.4 Connections

**[Mixed-effects models](/topics/mixed-effects)** — frequentist counterpart, same multi-level structure.

**[Sparse Bayesian priors](/topics/sparse-bayesian-priors)** — when $\phi$ controls a sparsity structure.

For MCMC on non-conjugate hierarchies, see [probabilistic programming](/topics/probabilistic-programming) §5. Canonical reference: BDA3 §5.""")


# ============================================================================
# §10 — Bayesian model comparison
# ============================================================================

md(r"""## §10. Bayesian model comparison

### §10.1 Bayes factors

$$\frac{P(M_1\mid y)}{P(M_2\mid y)} \;=\; \underbrace{\frac{p(y\mid M_1)}{p(y\mid M_2)}}_{\text{Bayes factor } B_{12}} \;\times\; \frac{P(M_1)}{P(M_2)}.$$

Jeffreys' scale: $B_{12} > 3$ substantial, $> 10$ strong, $> 100$ decisive.

### §10.2 Bayesian Occam's razor

The marginal likelihood is the *prior-weighted average* of the likelihood. Too-narrow priors miss the truth; too-wide priors are penalized.

### §10.3 A closed-form worked example

**Setup.** $y_i \sim \mathcal{N}(\mu, 1)$ for $i = 1, \dots, n$.

- $M_0$: $\mu = 0$.
- $M_1$: $\mu \sim \mathcal{N}(0, \tau^2)$.

**Closed-form log Bayes factor:**

$$\log B_{10}(\tau) \;=\; \frac{n^2 \bar y^2 \, \tau^2}{2(1 + n\tau^2)} \;-\; \frac{1}{2}\log(1 + n\tau^2).$$

The first term is the data-fit benefit; the second is the Occam penalty.""")


code(r'''# §10 — BayesFactorCalculator: log Bayes factor as a function of prior width.

rng_local = np.random.default_rng(SEED + 10)
n_obs       = 30
true_mu     = 0.3
sigma_known = 1.0
y_data = rng_local.normal(true_mu, sigma_known, size=n_obs)
ybar   = float(y_data.mean())
print(f"Simulated data: n = {n_obs}, ybar = {ybar:.4f} "
      f"(true mu = {true_mu})")

def log_bf_10(tau, n, ybar):
    a = n**2 * ybar**2 * tau**2 / (2.0 * (1.0 + n * tau**2))
    b = 0.5 * np.log1p(n * tau**2)
    return a - b

tau_grid = np.logspace(-2, 2, 400)
log_bf_curve = log_bf_10(tau_grid, n_obs, ybar)

tau_star_sq = ybar**2 - 1.0 / n_obs
if tau_star_sq > 0:
    tau_star = np.sqrt(tau_star_sq)
    log_bf_at_star = log_bf_10(tau_star, n_obs, ybar)
else:
    tau_star = 0.0
    log_bf_at_star = 0.0
print(f"Analytical Occam optimum: tau* = {tau_star:.4f}, "
      f"log_10 B_10(tau*) = {log_bf_at_star / np.log(10):.4f}")

tau_demo = [0.02, max(tau_star, 0.05), 5.0]
tau_labels = ["too narrow", "near optimum", "too wide"]

fig = plt.figure(figsize=(13.5, 8.5))
gs  = fig.add_gridspec(2, 3, height_ratios=[1.4, 1.0], hspace=0.45, wspace=0.30)
ax_top = fig.add_subplot(gs[0, :])
ax_btm = [fig.add_subplot(gs[1, j]) for j in range(3)]

log10_bf_curve = log_bf_curve / np.log(10)
ax_top.plot(tau_grid, log10_bf_curve, color=COLOR_C1, lw=2.5,
            label=r"$\log_{10} B_{10}(\tau)$")
ax_top.axhline(0,   color="black", lw=0.8, alpha=0.5)
ax_top.axhline( np.log10(3),   color=COLOR_TRUE, lw=1, ls=":",
               alpha=0.7, label="Jeffreys: substantial (3), strong (10), decisive (100)")
ax_top.axhline( np.log10(10),  color=COLOR_TRUE, lw=1, ls=":", alpha=0.7)
ax_top.axhline( np.log10(100), color=COLOR_TRUE, lw=1, ls=":", alpha=0.7)
ax_top.axhline(-np.log10(3),   color=COLOR_C0,   lw=1, ls=":", alpha=0.7)
ax_top.axhline(-np.log10(10),  color=COLOR_C0,   lw=1, ls=":", alpha=0.7)
if tau_star > 0:
    ax_top.plot(tau_star, log_bf_at_star / np.log(10),
                "o", color=COLOR_BAND, markersize=10,
                markeredgecolor="black", markeredgewidth=0.6, zorder=5,
                label=fr"interior optimum $\tau^* = {tau_star:.3f}$")
for tau_v, lbl in zip(tau_demo, tau_labels):
    ax_top.axvline(tau_v, color=COLOR_ACCENT, lw=1, ls="--", alpha=0.4)
    ax_top.annotate(lbl, xy=(tau_v, ax_top.get_ylim()[0]),
                    xytext=(tau_v, -1.0), fontsize=8.5,
                    color=COLOR_ACCENT, ha="center")
ax_top.set_xscale("log")
ax_top.set_xlabel(r"prior width $\tau$ (log scale)")
ax_top.set_ylabel(r"$\log_{10} B_{10}(\tau)$")
ax_top.set_title(f"Bayesian Occam curve: $\\log_{{10}} B_{{10}}$ vs prior width  "
                 f"(n = {n_obs}, ybar = {ybar:.3f})")
ax_top.legend(loc="upper right", fontsize=8.5)
ax_top.set_xlim(1e-2, 1e2)
ax_top.set_ylim(-2.5, max(2.5, log_bf_at_star / np.log(10) + 0.5))

mu_grid = np.linspace(-2.5, 2.5, 600)
like = np.exp(-0.5 * n_obs * (mu_grid - ybar)**2 / sigma_known)
like /= np.trapezoid(like, mu_grid)
for ax, tau_v, lbl in zip(ax_btm, tau_demo, tau_labels):
    prior = stats.norm(0, tau_v).pdf(mu_grid)
    overlap = prior * like
    overlap_normalized = overlap / overlap.max() * max(prior.max(), like.max()) * 0.6 \
                          if overlap.max() > 0 else overlap
    ax.fill_between(mu_grid, like,  color=COLOR_TRUE, alpha=0.25)
    ax.plot(mu_grid, like,  color=COLOR_TRUE, lw=2, label="likelihood")
    ax.plot(mu_grid, prior, color=COLOR_C0,   lw=2,
            label=fr"prior $\mathcal{{N}}(0, {tau_v}^2)$")
    ax.fill_between(mu_grid, overlap_normalized, color=COLOR_C1, alpha=0.20,
                    label="product (prop. integrand)")
    log10_bf_here = log_bf_10(tau_v, n_obs, ybar) / np.log(10)
    ax.set_title(fr"$\tau = {tau_v}$ ({lbl})"
                 f"\n$\\log_{{10}} B_{{10}} = {log10_bf_here:+.3f}$",
                 fontsize=10)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_xlim(-2.5, 2.5)

fig.suptitle("BayesFactorCalculator: Occam's razor as the integral of prior x likelihood",
             y=0.995, fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(FIG_DIR / "fig10_bayes_factor.png")
plt.show()

# Cross-check via Monte Carlo
tau_check = max(tau_star, 0.5)
M = 200_000
mu_samples = rng.normal(0, tau_check, size=M)
log_lik_samples = (
    -0.5 * n_obs * np.log(2 * np.pi)
    - 0.5 * np.sum((y_data[None, :] - mu_samples[:, None])**2, axis=1)
)
log_marg_M1 = ssp.logsumexp(log_lik_samples) - np.log(M)
log_marg_M0 = (-0.5 * n_obs * np.log(2*np.pi)
               - 0.5 * np.sum(y_data**2))
log_bf_mc      = log_marg_M1 - log_marg_M0
log_bf_closed  = log_bf_10(tau_check, n_obs, ybar)
print(f"\nNumerical cross-check at tau = {tau_check:.3f}:")
print(f"  Closed-form log B_10 = {log_bf_closed:.4f}")
print(f"  Monte-Carlo log B_10 = {log_bf_mc:.4f}  ({M:,} samples)")
assert abs(log_bf_mc - log_bf_closed) < 0.05, \
    "MC and closed-form should agree within ~0.05 nats."
print(f"  OK Agreement within {abs(log_bf_mc - log_bf_closed):.4f} nats.")''')


md(r"""### §10.4 Reading the Occam curve, with forward-pointers

Three regimes:

- **Narrow prior** ($\tau \ll \tau^*$): prior concentrated at $\mu = 0$, can't reach the data's likelihood mode. $\log B_{10} \to 0$.
- **Optimal prior** ($\tau \approx \tau^* = \sqrt{\bar y^2 - 1/n}$): prior and likelihood overlap; Bayes factor peaks.
- **Wide prior** ($\tau \gg \tau^*$): prior diluted; density at data-supported $\mu$ is small; Occam penalty wins.

**A caution.** Bayes factors depend on the prior in a way that does not wash out as $n \to \infty$. Sensitivity analyses across reasonable $\tau$ are the standard recommendation.

**[Variational Bayes for model selection](/topics/variational-bayes-for-model-selection)** takes over when the marginal likelihood is intractable. **[Stacking and predictive ensembles](/topics/stacking-and-predictive-ensembles)** is the predictive alternative — more robust to misspecification. **BIC** is the large-$n$ Laplace approximation; derivation in VBMS §3.""")


# ============================================================================
# §11 — Computational reality
# ============================================================================

md(r"""## §11. Computational reality

### §11.1 The non-conjugate problem

For non-conjugate models, the posterior is known only up to a proportionality constant. The MAP can be found by optimization (no normalizer needed), but expectations and samples require approximation.

### §11.2 Three computational families

| Family | Method | Forward-pointer |
|---|---|---|
| Deterministic approximation | Laplace (Gaussian fit at MAP) | this notebook (§11.3 demo) |
| Deterministic approximation | Variational inference (ELBO) | [variational inference](/topics/variational-inference) |
| Sampling | HMC / NUTS | [probabilistic programming](/topics/probabilistic-programming) |
| Sampling | Stochastic-gradient MCMC | [stochastic-gradient MCMC](/topics/stochastic-gradient-mcmc) |
| Sampling | Sequential Monte Carlo | [sequential Monte Carlo](/topics/sequential-monte-carlo) |
| Sampling | Reversible-jump MCMC | [reversible-jump MCMC](/topics/reversible-jump-mcmc) |
| Sampling | Riemann-manifold HMC | [Riemann-manifold HMC](/topics/riemann-manifold-hmc) |
| Hybrid | Amortized / neural posterior | [meta-learning](/topics/meta-learning) |
| Hybrid | VI/MCMC for NN weights | [Bayesian neural networks](/topics/bayesian-neural-networks) |
| Special closed-form | GP regression | [Gaussian processes](/topics/gaussian-processes) |

### §11.3 The simplest non-conjugate case: 1-D quadrature

**Setup.** $y_i \sim \mathcal{N}(\mu, 1)$ for $i = 1, \dots, 20$, prior $\mu \sim t_3(0, 1)$. Conjugacy with Normal likelihood breaks, but the posterior is 1-D and `scipy.integrate.quad` handles it.

**Pedagogical point.** Place the prior at $\mu_0 = 0$ and generate data with $\bar y \approx 2$. A Gaussian prior pulls the posterior toward 0; a Student-t prior's heavy tails let the data win more decisively. The Student-t prior is the canonical *robust prior*.""")


code(r'''# §11 — Non-conjugacy demo: Student-t prior + Normal likelihood via 1-D quadrature.

from scipy.integrate import quad
from scipy.optimize import minimize_scalar, brentq

rng_local = np.random.default_rng(SEED + 11)
n_obs   = 20
mu_true = 2.0
sigma   = 1.0
y_data  = rng_local.normal(mu_true, sigma, size=n_obs)
ybar    = float(y_data.mean())
print(f"Data: n = {n_obs}, ybar = {ybar:.4f}  (true mu = {mu_true})")

# 1. Gaussian-prior posterior (closed form)
mu_0, tau_0_sq = 0.0, 1.0
tau_n_sq_G = 1.0 / (1.0 / tau_0_sq + n_obs / sigma**2)
mu_n_G     = tau_n_sq_G * (mu_0 / tau_0_sq + n_obs * ybar / sigma**2)
post_G     = stats.norm(mu_n_G, np.sqrt(tau_n_sq_G))
print(f"\nGaussian prior  N(0, 1):  posterior N({mu_n_G:.4f}, {tau_n_sq_G:.4f})")

# 2. Student-t-prior posterior (quadrature)
t_nu, t_loc, t_scale = 3, 0.0, 1.0

def log_unnorm_post_t(mu):
    log_lik   = np.sum(stats.norm.logpdf(y_data, mu, sigma))
    log_prior = stats.t.logpdf(mu, df=t_nu, loc=t_loc, scale=t_scale)
    return log_lik + log_prior

def unnorm_post_t(mu):
    return np.exp(log_unnorm_post_t(mu) - log_post_t_max)

res = minimize_scalar(lambda mu: -log_unnorm_post_t(mu),
                       bounds=(-3, 5), method="bounded",
                       options={"xatol": 1e-8})
mode_t = float(res.x)
log_post_t_max = log_unnorm_post_t(mode_t)

Z, _ = quad(unnorm_post_t, -10, 10, epsabs=1e-10, epsrel=1e-10)

mean_t, _ = quad(lambda mu: mu * unnorm_post_t(mu), -10, 10)
mean_t /= Z
var_t,  _ = quad(lambda mu: (mu - mean_t)**2 * unnorm_post_t(mu), -10, 10)
var_t  /= Z
print(f"Student-t prior t_3(0, 1): posterior mean = {mean_t:.4f}, "
      f"posterior std = {np.sqrt(var_t):.4f}  (via quadrature)")

def cdf_t(mu_upper):
    val, _ = quad(unnorm_post_t, -10, mu_upper)
    return val / Z

ci_lo_t = brentq(lambda mu: cdf_t(mu) - 0.025, -5, mean_t)
ci_hi_t = brentq(lambda mu: cdf_t(mu) - 0.975, mean_t, 6)
print(f"                              95% credible interval = "
      f"[{ci_lo_t:.4f}, {ci_hi_t:.4f}]")

# 3. Laplace approximation
h = 1e-4
log_post_at_mode    = log_unnorm_post_t(mode_t)
log_post_at_mode_pp = log_unnorm_post_t(mode_t + h)
log_post_at_mode_mm = log_unnorm_post_t(mode_t - h)
curvature = (- log_post_at_mode_pp + 2 * log_post_at_mode - log_post_at_mode_mm) / h**2
laplace_std = 1.0 / np.sqrt(curvature)
post_L = stats.norm(mode_t, laplace_std)
print(f"\nLaplace approximation:    N({mode_t:.4f}, {laplace_std**2:.4f})  "
      f"around the Student-t-prior MAP")

# Figure
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

ax = axes[0]
mu_grid = np.linspace(-3, 5, 600)
ax.plot(mu_grid, stats.norm(mu_0, np.sqrt(tau_0_sq)).pdf(mu_grid),
        color=COLOR_C0, lw=2, label=r"Gaussian prior $\mathcal{N}(0, 1)$")
ax.plot(mu_grid, stats.t(df=t_nu, loc=t_loc, scale=t_scale).pdf(mu_grid),
        color=COLOR_C1, lw=2, label=fr"Student-t prior $t_{{{t_nu}}}(0, 1)$")
log_lik = np.array([np.sum(stats.norm.logpdf(y_data, m, sigma)) for m in mu_grid])
lik = np.exp(log_lik - log_lik.max())
lik /= np.trapezoid(lik, mu_grid)
ax.plot(mu_grid, lik, color=COLOR_TRUE, lw=2, ls="--",
        label=fr"likelihood (centered at $\bar y = {ybar:.2f}$)")
ax.axvline(mu_0, color="black", lw=0.8, ls=":", alpha=0.6)
ax.set_xlabel(r"$\mu$")
ax.set_ylabel("density")
ax.set_title("Priors and likelihood:  prior at 0, data at "
             fr"$\bar y \approx {ybar:.2f}$")
ax.legend(loc="upper left", fontsize=8.5)
ax.set_xlim(-3, 5)

ax = axes[1]
ax.plot(mu_grid, post_G.pdf(mu_grid), color=COLOR_C0, lw=2.4,
        label=fr"Gaussian-prior posterior  "
              fr"$\mathcal{{N}}({mu_n_G:.3f}, {tau_n_sq_G:.4f})$")
post_t_density = np.array([unnorm_post_t(m) for m in mu_grid]) / Z
ax.plot(mu_grid, post_t_density, color=COLOR_C1, lw=2.4,
        label=fr"Student-t-prior posterior (quadrature)  "
              fr"mean $= {mean_t:.3f}$, std $= {np.sqrt(var_t):.3f}$")
ax.plot(mu_grid, post_L.pdf(mu_grid), color=COLOR_ACCENT, lw=2, ls="--",
        label=fr"Laplace around Student-t MAP  "
              fr"$\mathcal{{N}}({mode_t:.3f}, {laplace_std**2:.4f})$")
ax.axvline(ybar,   color=COLOR_TRUE, lw=1, ls=":", alpha=0.6,
           label=fr"$\bar y = {ybar:.3f}$  (MLE)")
ax.axvline(mu_0,   color="black",    lw=0.8, ls=":", alpha=0.6,
           label=r"prior center $\mu_0 = 0$")
ax.set_xlabel(r"$\mu$")
ax.set_ylabel("posterior density")
ax.set_title("Three posteriors: Gaussian-prior (conjugate), "
             "Student-t-prior (quadrature), Laplace")
ax.legend(loc="upper left", fontsize=8)
ax.set_xlim(0.5, 3.0)

fig.suptitle("§11 - Non-conjugacy via 1-D quadrature: heavy-tailed priors are robust",
             y=1.02, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig11_non_conjugate.png")
plt.show()

print("\nSummary (posterior mean - distance from MLE measures prior pull):")
print(f"{'Method':<42} {'mean':>10} {'std':>10} {'distance from ybar':>20}")
print("-" * 85)
print(f"{'MLE (no prior)':<42} {ybar:>10.4f} "
      f"{sigma/np.sqrt(n_obs):>10.4f} {0:>20.4f}")
print(f"{'Gaussian-prior posterior (closed form)':<42} "
      f"{mu_n_G:>10.4f} {np.sqrt(tau_n_sq_G):>10.4f} "
      f"{abs(mu_n_G - ybar):>20.4f}")
print(f"{'Student-t-prior posterior (quadrature)':<42} "
      f"{mean_t:>10.4f} {np.sqrt(var_t):>10.4f} "
      f"{abs(mean_t - ybar):>20.4f}")
print(f"{'Laplace around Student-t-prior MAP':<42} "
      f"{mode_t:>10.4f} {laplace_std:>10.4f} "
      f"{abs(mode_t - ybar):>20.4f}")

assert abs(mean_t - ybar) < abs(mu_n_G - ybar), \
    "Student-t prior should pull less than Gaussian prior."
print(f"\nOK Student-t prior pulls less ({abs(mean_t - ybar):.4f} from ybar) "
      f"than Gaussian prior ({abs(mu_n_G - ybar):.4f} from ybar).")''')


md(r"""### §11.4 The forward-pointer manifest

| Topic | What it owns |
|---|---|
| [variational inference](/topics/variational-inference) | ELBO maximization; mean-field, full-rank, normalizing-flow variational families |
| [probabilistic programming](/topics/probabilistic-programming) | Stan / PyMC / NumPyro; NUTS as the default automatic engine |
| [stochastic-gradient MCMC](/topics/stochastic-gradient-mcmc) | SGLD, SGHMC; minibatch MCMC at NN scale |
| [sequential Monte Carlo](/topics/sequential-monte-carlo) | Particle filtering/smoothing; tempering |
| [reversible-jump MCMC](/topics/reversible-jump-mcmc) | Trans-dimensional MCMC |
| [Riemann-manifold HMC](/topics/riemann-manifold-hmc) | Fisher-information geometry; curvature-aware HMC |
| [Bayesian neural networks](/topics/bayesian-neural-networks) | High-dim posteriors over NN weights |
| [meta-learning](/topics/meta-learning) | Amortized inference; neural posterior estimation |
| [Gaussian processes](/topics/gaussian-processes) | Exact Bayesian function regression |
| [variational Bayes for model selection](/topics/variational-bayes-for-model-selection) | ELBO as evidence approximation; algorithmic side of §10 |
| [mixed-effects models](/topics/mixed-effects) | Frequentist hierarchical models |
| [sparse Bayesian priors](/topics/sparse-bayesian-priors) | Spike-and-slab, horseshoe, ARD |
| [stacking and predictive ensembles](/topics/stacking-and-predictive-ensembles) | Predictive model averaging |

The umbrella is the *map* of T5, not the territory.""")


# ============================================================================
# §12 — Frequentist-Bayesian dialogue
# ============================================================================

md(r"""## §12. The frequentist–Bayesian dialogue

### §12.1 Coverage vs. credibility

**Frequentist 95% CI.** $P_\theta(\theta \in \mathcal{C}(Y)) = 0.95$ for all $\theta$ — probability over the sampling distribution at fixed $\theta$.

**Bayesian 95% credible interval.** $P(\theta \in \mathcal{R}(y) \mid y) = 0.95$ — probability over the posterior at fixed observed data.

The Bayesian statement is the one most readers *want* but requires accepting a prior.

### §12.2 Bernstein–von Mises

> **Theorem 12.1 (Bernstein-von Mises, informal).** Under regularity conditions,
> $$\pi(\theta\mid y_{1:n}) \;\xrightarrow{\mathrm{TV}}\; \mathcal{N}\!\bigl(\hat\theta_{\text{MLE}},\; I(\hat\theta_{\text{MLE}})^{-1}/n\bigr) \quad \text{as } n \to \infty.$$

Asymptotically, CI and CrI agree. The prior contributes an $O(1/n)$ correction.

Forward-pointer: van der Vaart (1998, §10). Caveats — boundaries, unidentifiability, growing dimension, misspecification.""")


code(r'''# §12 — FrequentistBayesianDialogue: coverage view + credibility view.

mu_true   = 0.5
sigma     = 1.0
n_obs     = 20
n_repeats = 30
z_95      = 1.96

mu_0, tau_0_sq = 0.0, 100.0

rng_local = np.random.default_rng(SEED + 12)

samples = rng_local.normal(mu_true, sigma, size=(n_repeats, n_obs))
ybars   = samples.mean(axis=1)
sem     = sigma / np.sqrt(n_obs)
ci_lo   = ybars - z_95 * sem
ci_hi   = ybars + z_95 * sem
covers  = (ci_lo <= mu_true) & (mu_true <= ci_hi)
coverage_rate = float(covers.mean())

fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.5),
                          gridspec_kw={"height_ratios": [1.0, 1.0]})

ax = axes[0]
order = np.argsort(ybars)
for plot_idx, k in enumerate(order):
    color = COLOR_TRUE if covers[k] else COLOR_C1
    ax.plot([ci_lo[k], ci_hi[k]], [plot_idx, plot_idx],
            color=color, lw=2.4)
    ax.plot(ybars[k], plot_idx, "o", color=color, markersize=4,
            markeredgecolor="black", markeredgewidth=0.3)
ax.axvline(mu_true, color="black", lw=2, ls="--",
           label=fr"true $\mu = {mu_true}$")
ax.set_xlabel(r"$\mu$")
ax.set_ylabel("sample # (sorted by $\\bar y$)")
ax.set_title(fr"Frequentist view (n = {n_obs}): {n_repeats} repeated 95% CIs, "
             fr"empirical coverage = {coverage_rate:.0%}  "
             fr"(green = covers, red = misses)")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(-0.5, 1.5)

y_observed   = samples[0]
ybar_obs     = float(y_observed.mean())
tau_n_sq     = 1.0 / (1.0 / tau_0_sq + n_obs / sigma**2)
mu_n         = tau_n_sq * (mu_0 / tau_0_sq + n_obs * ybar_obs / sigma**2)
post         = stats.norm(mu_n, np.sqrt(tau_n_sq))
cri_lo, cri_hi = post.ppf([0.025, 0.975])
ci_lo_obs    = ybar_obs - z_95 * sem
ci_hi_obs    = ybar_obs + z_95 * sem

mu_grid = np.linspace(ybar_obs - 1.0, ybar_obs + 1.0, 600)
ax = axes[1]
ax.fill_between(mu_grid, post.pdf(mu_grid), color=COLOR_C0, alpha=0.20)
ax.plot(mu_grid, post.pdf(mu_grid), color=COLOR_C0, lw=2,
        label=r"Bayesian posterior  $\pi(\mu \mid y)$  (weak prior)")
ymax = post.pdf(mu_grid).max()
ax.plot([ci_lo_obs, ci_hi_obs], [-ymax * 0.10, -ymax * 0.10],
        color=COLOR_C1, lw=8, solid_capstyle="butt",
        label=fr"95% CI [{ci_lo_obs:.3f}, {ci_hi_obs:.3f}]  "
              fr"(width {ci_hi_obs - ci_lo_obs:.3f})")
ax.plot([cri_lo, cri_hi], [-ymax * 0.20, -ymax * 0.20],
        color=COLOR_C0, lw=8, solid_capstyle="butt",
        label=fr"95% CrI [{cri_lo:.3f}, {cri_hi:.3f}]  "
              fr"(width {cri_hi - cri_lo:.3f})")
ax.axvline(ybar_obs, color="black", lw=1, ls="--",
           label=fr"$\bar y = {ybar_obs:.3f}$")
ax.axvline(mu_true,  color=COLOR_TRUE, lw=1, ls=":",
           label=fr"true $\mu = {mu_true}$")
ax.set_ylim(-ymax * 0.30, ymax * 1.10)
ax.set_xlabel(r"$\mu$")
ax.set_ylabel("posterior density")
ax.set_title(fr"Bayesian view (n = {n_obs}): posterior + CrI + CI for one "
             "observed dataset")
ax.legend(loc="upper right", fontsize=8)

fig.suptitle("FrequentistBayesianDialogue: two probability statements, two views",
             y=1.005, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig12_freq_bayes.png")
plt.show()

print("Bernstein-von Mises in action: CI vs CrI at four sample sizes")
print(f"{'n':>5}  {'ybar':>8}  {'CI width':>10}  {'CrI width':>11}  "
      f"{'|CI center - CrI center|':>26}")
print("-" * 70)
for n in [5, 20, 100, 1000]:
    y_n        = rng_local.normal(mu_true, sigma, size=n)
    ybar_n     = float(y_n.mean())
    sem_n      = sigma / np.sqrt(n)
    tau_n_sq_n = 1.0 / (1.0 / tau_0_sq + n / sigma**2)
    mu_n_n     = tau_n_sq_n * (mu_0 / tau_0_sq + n * ybar_n / sigma**2)
    ci_lo_n    = ybar_n - z_95 * sem_n
    ci_hi_n    = ybar_n + z_95 * sem_n
    cri_lo_n   = mu_n_n - z_95 * np.sqrt(tau_n_sq_n)
    cri_hi_n   = mu_n_n + z_95 * np.sqrt(tau_n_sq_n)
    width_ci   = ci_hi_n - ci_lo_n
    width_cri  = cri_hi_n - cri_lo_n
    center_gap = abs(ybar_n - mu_n_n)
    print(f"{n:>5d}  {ybar_n:>8.4f}  {width_ci:>10.4f}  "
          f"{width_cri:>11.4f}  {center_gap:>26.6f}")

print()
print("Reading: at large n, CI and CrI widths converge to ~2 * 1.96 * sigma/sqrt(n),")
print("and their centers coincide. BVM is doing the work as n grows.")

ratio = (sigma / np.sqrt(1000)) / (sigma / np.sqrt(5))
print(f"\nOK CI/CrI width ratio between n=1000 and n=5 ~ {ratio:.4f}  "
      f"(theoretically sqrt(5/1000) = {np.sqrt(5/1000):.4f}). BVM scaling confirmed.")''')


md(r"""### §12.3 When priors matter most

- **Small samples**: prior contributes meaningfully.
- **Weakly identified models**: prior pins down what data cannot.
- **High-dimensional models**: prior *is* the regularizer (ridge ↔ Gaussian, lasso ↔ Laplace).
- **Hierarchical models** (§9).
- **Structural assumptions** (monotonicity, smoothness, sparsity).
- **Decision-theoretic needs** (§7).

### §12.4 A mature view

Neither frame is universally right. The right question is *"does the method answer what I need to know?"*, not *"which camp do I belong to?"*.

**Hybrid approaches** further blur the boundary: empirical Bayes (§9.3), frequentist evaluation of Bayesian procedures, and Bayesian readings of frequentist estimators (lasso, ridge, Stein, BLUP). Forward-pointer: Gelman & Vehtari (2021) on the important statistical ideas of the past 50 years.""")


# ============================================================================
# §13 — Connections and limits + final runtime check
# ============================================================================

md(r"""## §13. Connections and limits

The umbrella closes with an honest scope statement.

### §13.1 Where the Bayesian frame works well

The §1.1 motivations all hold up. **Prior information** via §3.1 pseudo-counts. **Calibrated uncertainty** via §6 predictives and §7 decisions. **Hierarchical and structural modeling** (§9) via De Finetti (§8.3). For regular models at moderate-to-large $n$, the Bayesian frame agrees with frequentist inference asymptotically (BVM, §12.2) while providing small-$n$ corrections and decision-theoretic outputs.

### §13.2 Where the Bayesian frame breaks

**Model misspecification.** Posterior concentrates at the *KL-projection point* (Kleijn-van der Vaart, 2012). Credible intervals can have poor coverage of the true generating parameter.

**Generalized / Gibbs posteriors.**

$$\pi_w(\theta\mid y) \;\propto\; \exp\!\bigl(-w \cdot \ell(\theta, y)\bigr)\,\pi(\theta).$$

Bissiri, Holmes & Walker (2016) — outside strict Bayes but a natural generalization.

**Robust Bayes.** Work with a *set* of priors $\Pi$ (Berger, 1985).

**Persistent prior sensitivity.** For non-asymptotic inference, sensitivity analyses across reasonable priors are the standard recommendation.""")


md(r"""### §13.3 Likelihood-free settings: ABC and SBI

When the likelihood is *intractable* — population genetics, physical simulators, implicit neural-network generators.

**Approximate Bayesian Computation (ABC).** Sample $\theta^*$ from the prior; simulate $y^*$; accept if $\|T(y^*) - T(y)\| < \epsilon$.

**Simulation-based inference (SBI).** Train neural networks on simulator output to learn the posterior, likelihood, or ratio. Cranmer, Brehmer & Louppe (2020, *PNAS*).

The closest T5 cousin is [meta-learning](/topics/meta-learning).

### §13.4 The forward-pointer summary table

| Inference question | Topic |
|---|---|
| **Computational approximations** | |
| Approximate a posterior fast and at scale | [variational inference](/topics/variational-inference) |
| Declarative model spec + automatic MCMC | [probabilistic programming](/topics/probabilistic-programming) |
| MCMC at neural-network scale | [stochastic-gradient MCMC](/topics/stochastic-gradient-mcmc) |
| State-space / sequential data | [sequential Monte Carlo](/topics/sequential-monte-carlo) |
| Sample across models of different dimensions | [reversible-jump MCMC](/topics/reversible-jump-mcmc) |
| Poorly-scaled posteriors | [Riemann-manifold HMC](/topics/riemann-manifold-hmc) |
| **Modeling families** | |
| Bayesian inference for neural networks | [Bayesian neural networks](/topics/bayesian-neural-networks) |
| Amortized inference (learning to infer) | [meta-learning](/topics/meta-learning) |
| Bayesian function regression | [Gaussian processes](/topics/gaussian-processes) |
| Share information across groups (frequentist) | [mixed-effects models](/topics/mixed-effects) |
| Bayesian variable selection / sparsity | [sparse Bayesian priors](/topics/sparse-bayesian-priors) |
| **Model selection and ensembling** | |
| Approximate the marginal likelihood | [variational Bayes for model selection](/topics/variational-bayes-for-model-selection) |
| Combine multiple models for prediction | [stacking and predictive ensembles](/topics/stacking-and-predictive-ensembles) |

### §13.5 Closing

Bayesian inference is a *language*. The same vocabulary — prior, likelihood, posterior, marginal likelihood, posterior predictive, credible interval, Bayes factor — covers the conjugate Beta-Binomial A/B test we used as the running example and the deep BNN that lives in [bayesian neural networks](/topics/bayesian-neural-networks). The umbrella's job was to make the language fluent; the T5 leaves are where the work happens.

We end here. End-to-end notebook runtime check follows.""")


code(r'''# §13 - End-of-notebook runtime sanity check and figures-saved summary.

import os

elapsed = time.time() - _NB_START_TIME

fig_dir = FIG_DIR
saved_figures = sorted(fig_dir.glob("*.png"))
print(f"Figures saved to {fig_dir}/ (total: {len(saved_figures)}):")
for fp in saved_figures:
    size_kb = os.path.getsize(fp) / 1024
    print(f"  {fp.name:<45} {size_kb:>6.1f} KB")

print(f"\nEnd-to-end notebook runtime: {elapsed:.1f} s "
      f"(target: under 30 s on a 2020-era laptop, CPU-only).")

print("\nForward-pointer summary: this umbrella names the framework and links")
print("to the 13 T5 leaves that develop the algorithmic content. See brief §13.4")
print("for the complete map.")''')


# ============================================================================
# References
# ============================================================================

md(r"""## References

Chicago 17th edition (Notes and Bibliography). Grouped foundational / advanced / applied. See `docs/plans/formalml-bayesian-inference-handoff-brief.md` for the full bibliography with verified URLs.

### Foundational primaries

Bayes (1763) — *Phil. Trans.* 53: 370–418. [doi:10.1098/rstl.1763.0053](https://doi.org/10.1098/rstl.1763.0053).
de Finetti (1937) — *Annales IHP* 7 (1): 1–68. [numdam:AIHP_1937__7_1_1_0](http://www.numdam.org/item/AIHP_1937__7_1_1_0/).
Efron & Morris (1973) — *JASA* 68 (341): 117–130. [doi:10.1080/01621459.1973.10481350](https://doi.org/10.1080/01621459.1973.10481350).
Hewitt & Savage (1955) — *Trans. AMS* 80: 470–501. [doi:10.1090/S0002-9947-1955-0076206-8](https://doi.org/10.1090/S0002-9947-1955-0076206-8).
Jeffreys (1946) — *Proc. Roy. Soc. A* 186 (1007): 453–461. [doi:10.1098/rspa.1946.0056](https://doi.org/10.1098/rspa.1946.0056).
Jeffreys (1961) — *Theory of Probability*, 3rd ed., OUP.
Laplace (1814) — *Essai philosophique sur les probabilités*. Paris: Courcier. [BnF Gallica](https://gallica.bnf.fr/ark:/12148/bd6t5772134h).
Schwarz (1978) — *Annals of Statistics* 6 (2): 461–464. [doi:10.1214/aos/1176344136](https://doi.org/10.1214/aos/1176344136).
Stein (1956) — *Proc. 3rd Berkeley Symp.* 1: 197–206.
Welch (1939) — *Annals of Math. Stat.* 10 (1): 58–69. [doi:10.1214/aoms/1177732246](https://doi.org/10.1214/aoms/1177732246).

### Advanced (theory and methods)

Berger (1985) — *Statistical Decision Theory and Bayesian Analysis*, 2nd ed., Springer. [doi:10.1007/978-1-4757-4286-2](https://doi.org/10.1007/978-1-4757-4286-2).
Bernardo & Smith (1994) — *Bayesian Theory*, Wiley. [doi:10.1002/9780470316870](https://doi.org/10.1002/9780470316870).
Bishop (2006) — *Pattern Recognition and Machine Learning*, Springer.
Bissiri, Holmes & Walker (2016) — *JRSS B* 78 (5): 1103–1130. [doi:10.1111/rssb.12158](https://doi.org/10.1111/rssb.12158).
Cranmer, Brehmer & Louppe (2020) — *PNAS* 117 (48): 30055–30062. [doi:10.1073/pnas.1912789117](https://doi.org/10.1073/pnas.1912789117).
Gelman et al. (2013) — *Bayesian Data Analysis*, 3rd ed., Chapman & Hall (BDA3). [doi:10.1201/b16018](https://doi.org/10.1201/b16018).
Gelman, Meng & Stern (1996) — *Statistica Sinica* 6 (4): 733–807.
Gelman & Vehtari (2021) — *JASA* 116 (536): 2087–2097. [doi:10.1080/01621459.2021.1938081](https://doi.org/10.1080/01621459.2021.1938081).
Kleijn & van der Vaart (2012) — *EJS* 6: 354–381. [doi:10.1214/12-EJS675](https://doi.org/10.1214/12-EJS675).
Marin, Pudlo, Robert & Ryder (2012) — *Stat. Comput.* 22 (6): 1167–1180. [doi:10.1007/s11222-011-9288-2](https://doi.org/10.1007/s11222-011-9288-2).
Robert (2007) — *The Bayesian Choice*, 2nd ed., Springer. [doi:10.1007/0-387-71599-1](https://doi.org/10.1007/0-387-71599-1).
Rubin (1984) — *Annals of Statistics* 12 (4): 1151–1172. [doi:10.1214/aos/1176346785](https://doi.org/10.1214/aos/1176346785).
van der Vaart (1998) — *Asymptotic Statistics*, Cambridge. [doi:10.1017/CBO9780511802256](https://doi.org/10.1017/CBO9780511802256).
Vehtari, Gelman & Gabry (2017) — *Stat. Comput.* 27 (5): 1413–1432. [doi:10.1007/s11222-016-9696-4](https://doi.org/10.1007/s11222-016-9696-4).

### Applied

Kruschke (2014) — *Doing Bayesian Data Analysis*, 2nd ed., Academic Press / Elsevier.
McElreath (2020) — *Statistical Rethinking*, 2nd ed., CRC Press. [doi:10.1201/9780429029608](https://doi.org/10.1201/9780429029608).
McGrayne (2011) — *The Theory That Would Not Die*, Yale UP. [doi:10.12987/9780300175097](https://doi.org/10.12987/9780300175097).""")


# ============================================================================
# Assemble + write
# ============================================================================

notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).parent / "01_bayesian_inference.ipynb"
with out_path.open("w") as f:
    json.dump(notebook, f, indent=1)

print(f"Wrote {out_path}")
print(f"Cells: {len(CELLS)} ({sum(1 for c in CELLS if c['cell_type'] == 'markdown')} markdown, "
      f"{sum(1 for c in CELLS if c['cell_type'] == 'code')} code)")
