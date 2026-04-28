// =============================================================================
// bayesian-ml.ts
//
// Shared math/types/palette module for the T5 Bayesian & Probabilistic ML track.
// First consumer: variational-inference topic.
//
// v1 stub — types and a color palette only. No numerical helpers, no D3
// utilities. The variational-inference topic ships with five static PNG figures
// rendered by notebooks/variational-inference/01_variational_inference.ipynb;
// no React/D3 components touch this module yet. As the per-figure interactive
// enhancements outlined in the variational-inference handoff brief §4 land in
// future PRs, the matching helpers (e.g., MC-ELBO trajectory generators,
// reparam vs score-function variance estimators, Real NVP forward-pass
// helpers) will accumulate here. Until then this file is intentionally small.
// =============================================================================

// -----------------------------------------------------------------------------
// Color palette
// -----------------------------------------------------------------------------

/**
 * Color palette mirroring the matplotlib palette declared in the notebook's
 * setup cell. Keeping the TypeScript palette in lockstep with the notebook
 * constants means React enhancements (when they arrive) stay visually
 * continuous with the v1 static PNGs that ship alongside this module.
 *
 * Source of truth: notebooks/variational-inference/01_variational_inference.ipynb
 * setup cell (COLOR_POSTERIOR, COLOR_MEANFIELD, COLOR_FULLRANK, COLOR_FLOW,
 * COLOR_DATA, COLOR_REFERENCE).
 */
export const paletteVI = {
  posterior: '#1f77b4', // blue   — true posterior contours
  meanField: '#d62728', // red    — mean-field variational q
  fullRank: '#ff7f0e', // orange — full-rank Gaussian q
  flow: '#2ca02c', // green  — normalizing-flow q
  data: '#7f7f7f', // gray   — sample / data points
  reference: '#000000', // black  — reference lines (e.g., log p(x))
} as const;

export type VIColorKey = keyof typeof paletteVI;

// -----------------------------------------------------------------------------
// Variational-family types
//
// A small, deliberately incomplete vocabulary of types that future v2
// components will share. Each one mirrors a distinct variational family
// developed across the topic's six sections.
// -----------------------------------------------------------------------------

/**
 * A discrete identifier for the three variational families compared in the
 * variational-inference §5 banana-posterior experiment. `meanField` and
 * `fullRank` are Gaussian families distinguished by covariance structure;
 * `flow` denotes a normalizing-flow family parametrized by an invertible map.
 */
export type VariationalFamily = 'meanField' | 'fullRank' | 'flow';

/**
 * A single sample drawn from a 2D variational distribution, used by future
 * scatter-overlay components in §1 panel (b)/(c), §3 panel (b), and §5
 * panels (a)–(c).
 */
export interface VISample2D {
  x: number;
  y: number;
}

/**
 * A single point on the ELBO-vs-iteration trajectory plotted in §3 panel (c),
 * §4 panel (b), and §5 panel (d). The `family` field is `null` for §3 (CAVI on
 * a Bayesian GMM) and §4 (single-family ADVI), and identifies one of the three
 * banana-posterior runs in §5.
 */
export interface ELBOTrajectoryPoint {
  iteration: number;
  elbo: number;
  family: VariationalFamily | null;
}

/**
 * Closed-form parameters of a 1D Gaussian variational marginal — the §2 and
 * §4.5 worked examples both reduce to comparing such marginals against an
 * exact (§2) or Laplace-reference (§4.5) target.
 */
export interface GaussianMarginal {
  mean: number;
  std: number;
}
