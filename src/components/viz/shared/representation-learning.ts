// =============================================================================
// representation-learning.ts
//
// Shared math module for the T3 Unsupervised & Generative track,
// representation-learning topic. First consumers: viz components in
// src/content/topics/representation-learning.mdx
//   (SufficiencyFactorizationDemo / LinearAEisPCA / DenoisingAEScoreField /
//    ReparametrizationTrickDiagram / InfoNCETemperatureExplorer /
//    IBLagrangianFrontier / RepresentationGeometryViewer /
//    VAELatentSpaceExplorer / AlignmentUniformityScatter /
//    NeuralCollapseSimplex / SufficiencyToContrastiveBridge).
//
// All numerical algorithms are direct translations of the verified Python
// notebook at notebooks/representation-learning/01_representation_learning.ipynb
// (the source of truth). Numerical agreement against notebook printed outputs
// is verified by
// src/components/viz/shared/__tests__/verify-representation-learning.ts.
//
// Cross-module reuse (do NOT redefine):
//   - symEigJacobi from ./double-descent  (cyclic-Jacobi symmetric eig)
//   - logSumExp from ./sequential-monte-carlo  (numerically stable LSE)
//   - mulberry32, gaussianPair from ./bayesian-ml  (RNG, normal pairs)
// =============================================================================

import { symEigJacobi } from './double-descent';
import { logSumExp } from './sequential-monte-carlo';
import { mulberry32, gaussianPair } from './bayesian-ml';

export { symEigJacobi, logSumExp, mulberry32, gaussianPair };

// -----------------------------------------------------------------------------
// Palette — formalML default. Three roles dominate the topic: signal
// (encoder h / discriminative direction), nuisance (color / augmentation axis),
// and theoretical-optimum reference (LDA / sufficient statistic / simplex ETF).
// -----------------------------------------------------------------------------

export const paletteRL = {
  signal: '#1f4e79', // deep blue   — encoder h, discriminative direction
  nuisance: '#c0504d', // red         — color / nuisance axis, projector z
  optimum: '#3a6e3a', // forest      — LDA / sufficient statistic / ETF
  accent: '#9467bd', // purple      — accents
  data: '#7f7f7f', // gray        — raw scatter
  highlight: '#ff7f0e', // orange      — selected / hover
} as const;

// =============================================================================
// §3.2 — PCA via symmetric eigendecomposition + linear-AE gradient descent.
// =============================================================================

export interface PCAResult {
  /** Top-k eigenvectors as a (k × p) row-major matrix. */
  components: Float64Array;
  /** Top-k eigenvalues in descending order. */
  eigenvalues: Float64Array;
  /** Per-feature column means (length p), used to center inputs at probe time. */
  mean: Float64Array;
}

/**
 * Column-center an n × p row-major matrix in place. Returns the means.
 * Centering is the precondition for both PCA and CKA.
 */
export function centerColumns(X: Float64Array, n: number, p: number): Float64Array {
  const mean = new Float64Array(p);
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) mean[j] += X[i * p + j];
  for (let j = 0; j < p; j++) mean[j] /= n;
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) X[i * p + j] -= mean[j];
  return mean;
}

/**
 * Closed-form top-k PCA via cyclic Jacobi on the (1/n) X^T X covariance.
 * Sorts eigenvalues descending — Jacobi output is unsorted (CLAUDE.md
 * §"Jacobi eigendecomp output is unsorted").
 *
 * Input X is assumed already column-centered.
 */
export function pcaTopK(
  X: Float64Array,
  n: number,
  p: number,
  k: number,
): PCAResult {
  // Build (1/n) X^T X (p × p).
  const C = new Float64Array(p * p);
  for (let j = 0; j < p; j++) {
    for (let l = j; l < p; l++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += X[i * p + j] * X[i * p + l];
      C[j * p + l] = s / n;
      C[l * p + j] = C[j * p + l];
    }
  }
  const { values, vectors: V } = symEigJacobi(C, p);
  // Sort indices by descending eigenvalue.
  const order = Array.from({ length: p }, (_, i) => i);
  order.sort((a, b) => values[b] - values[a]);
  const kEff = Math.min(k, p);
  const components = new Float64Array(kEff * p);
  const eigenvalues = new Float64Array(kEff);
  for (let r = 0; r < kEff; r++) {
    eigenvalues[r] = Math.max(values[order[r]], 0);
    // r-th component = column order[r] of V (row-major V is p × p).
    for (let j = 0; j < p; j++) components[r * p + j] = V[j * p + order[r]];
  }
  return { components, eigenvalues, mean: new Float64Array(p) /* caller may overwrite */ };
}

/** Project an n × p row-major X onto a (k × p) row-major components matrix. */
export function pcaProject(
  X: Float64Array,
  components: Float64Array,
  n: number,
  p: number,
  k: number,
): Float64Array {
  const Z = new Float64Array(n * k);
  for (let i = 0; i < n; i++) {
    for (let r = 0; r < k; r++) {
      let s = 0;
      for (let j = 0; j < p; j++) s += X[i * p + j] * components[r * p + j];
      Z[i * k + r] = s;
    }
  }
  return Z;
}

export interface LinearAEHistory {
  /** Reconstruction loss per epoch. */
  loss: Float64Array;
  /** |cos∠(W^T col_0, v_top)| per epoch — convergence to top PC for d = 1. */
  alignTop: Float64Array;
}

export interface LinearAEResult {
  /** Encoder weights W (d × p), row-major. */
  W: Float64Array;
  /** Decoder weights U (p × d), row-major. */
  U: Float64Array;
  history: LinearAEHistory;
}

/**
 * Train a linear autoencoder f(x) = W x, g(z) = U z by gradient descent on
 * E ||x - U W x||^2. Per Baldi-Hornik (Theorem 3.1), the product UW converges
 * to the orthogonal projector onto the top-d eigenspace of the data covariance.
 *
 * Input X is assumed centered. The optional `vTop` is a top-1 reference
 * eigenvector to track alignment against (length p); pass it for the d = 1
 * convergence-to-PCA demo. If d > 1, alignTop is left at zero.
 */
export function linearAEFit(
  X: Float64Array,
  n: number,
  p: number,
  d: number,
  options: { lr?: number; epochs?: number; seed?: number; vTop?: Float64Array } = {},
): LinearAEResult {
  const lr = options.lr ?? 0.05;
  const epochs = options.epochs ?? 200;
  const seed = options.seed ?? 42;
  const rng = mulberry32(seed);
  const W = new Float64Array(d * p);
  const U = new Float64Array(p * d);
  // Small random init.
  for (let i = 0; i < d * p; i++) W[i] = 0.1 * (rng() - 0.5);
  for (let i = 0; i < p * d; i++) U[i] = 0.1 * (rng() - 0.5);
  const loss = new Float64Array(epochs);
  const alignTop = new Float64Array(epochs);
  // Scratch buffers (hoisted out of the loop per CLAUDE.md "hoist allocations").
  const z = new Float64Array(d);
  const xhat = new Float64Array(p);
  const dXhat = new Float64Array(p);
  const dZ = new Float64Array(d);
  const gW = new Float64Array(d * p);
  const gU = new Float64Array(p * d);
  for (let e = 0; e < epochs; e++) {
    let lossEpoch = 0;
    gW.fill(0);
    gU.fill(0);
    for (let i = 0; i < n; i++) {
      const base = i * p;
      // z = W x
      for (let r = 0; r < d; r++) {
        let s = 0;
        for (let j = 0; j < p; j++) s += W[r * p + j] * X[base + j];
        z[r] = s;
      }
      // xhat = U z
      for (let j = 0; j < p; j++) {
        let s = 0;
        for (let r = 0; r < d; r++) s += U[j * d + r] * z[r];
        xhat[j] = s;
      }
      // residual r = xhat - x, loss contribution
      for (let j = 0; j < p; j++) {
        dXhat[j] = xhat[j] - X[base + j];
        lossEpoch += dXhat[j] * dXhat[j];
      }
      // dU = (xhat - x) z^T
      for (let j = 0; j < p; j++) for (let r = 0; r < d; r++) gU[j * d + r] += dXhat[j] * z[r];
      // dZ = U^T (xhat - x)
      for (let r = 0; r < d; r++) {
        let s = 0;
        for (let j = 0; j < p; j++) s += U[j * d + r] * dXhat[j];
        dZ[r] = s;
      }
      // dW = dZ x^T
      for (let r = 0; r < d; r++) for (let j = 0; j < p; j++) gW[r * p + j] += dZ[r] * X[base + j];
    }
    const inv = 1 / n;
    for (let i = 0; i < d * p; i++) W[i] -= lr * gW[i] * inv;
    for (let i = 0; i < p * d; i++) U[i] -= lr * gU[i] * inv;
    loss[e] = lossEpoch / n;
    if (d === 1 && options.vTop) {
      // alignTop[e] = |cos∠(W row 0, vTop)|
      let dot = 0,
        nw = 0,
        nv = 0;
      for (let j = 0; j < p; j++) {
        dot += W[j] * options.vTop[j];
        nw += W[j] * W[j];
        nv += options.vTop[j] * options.vTop[j];
      }
      alignTop[e] = nw > 0 && nv > 0 ? Math.abs(dot) / Math.sqrt(nw * nv) : 0;
    }
  }
  return { W, U, history: { loss, alignTop } };
}

// =============================================================================
// §3.4 — Tweedie identity verification (Gaussian-mixture closed form).
//
// For data X ~ Σ_k w_k δ_{μ_k} (Dirac mixture) and noise η ~ N(0, σ² I),
// X̃ = X + η has density p_σ(x̃) = Σ_k w_k N(x̃; μ_k, σ² I), and
//   r*(x̃) = E[X | X̃ = x̃] = Σ_k γ_k(x̃) μ_k
//   ∇log p_σ(x̃) = Σ_k γ_k(x̃) (μ_k - x̃) / σ²
//                = (r*(x̃) - x̃) / σ²
// where γ_k(x̃) = w_k N(x̃; μ_k, σ² I) / p_σ(x̃) is the posterior responsibility.
// The Tweedie identity r*(x̃) - x̃ = σ² ∇log p_σ(x̃) is therefore exact for the
// Dirac mixture, to machine precision.
// =============================================================================

export interface GMMParams {
  /** K × p row-major component means. */
  mus: Float64Array;
  /** Length-K mixture weights, sum to 1. */
  weights: Float64Array;
  K: number;
  p: number;
}

/** Posterior responsibilities γ_k(x̃) for a Gaussian mixture under isotropic σ². */
export function gmmResponsibilities(
  xTilde: Float64Array,
  gmm: GMMParams,
  sigma: number,
): Float64Array {
  const { mus, weights, K, p } = gmm;
  const inv2s2 = 1 / (2 * sigma * sigma);
  const logW = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    let sq = 0;
    for (let j = 0; j < p; j++) {
      const d = xTilde[j] - mus[k * p + j];
      sq += d * d;
    }
    logW[k] = Math.log(Math.max(weights[k], 1e-300)) - sq * inv2s2;
  }
  const M = Math.max(...logW);
  let Z = 0;
  for (let k = 0; k < K; k++) Z += Math.exp(logW[k] - M);
  const out = new Float64Array(K);
  for (let k = 0; k < K; k++) out[k] = Math.exp(logW[k] - M) / Z;
  return out;
}

/** Optimal denoiser r*(x̃) = E[X | X̃ = x̃] = Σ γ_k(x̃) μ_k. */
export function gmmDenoiser(xTilde: Float64Array, gmm: GMMParams, sigma: number): Float64Array {
  const { mus, K, p } = gmm;
  const g = gmmResponsibilities(xTilde, gmm, sigma);
  const r = new Float64Array(p);
  for (let k = 0; k < K; k++) for (let j = 0; j < p; j++) r[j] += g[k] * mus[k * p + j];
  return r;
}

/** Score ∇log p_σ(x̃) for a Gaussian mixture under isotropic σ². */
export function gmmScore(xTilde: Float64Array, gmm: GMMParams, sigma: number): Float64Array {
  const { mus, K, p } = gmm;
  const g = gmmResponsibilities(xTilde, gmm, sigma);
  const s = new Float64Array(p);
  const inv = 1 / (sigma * sigma);
  for (let k = 0; k < K; k++) {
    for (let j = 0; j < p; j++) s[j] += g[k] * (mus[k * p + j] - xTilde[j]) * inv;
  }
  return s;
}

/**
 * Tweedie identity discrepancy. Returns ||(r*(x̃) - x̃) - σ² ∇log p_σ(x̃)||
 * for a single x̃. Zero up to machine precision for the Dirac mixture.
 */
export function tweedieScore(xTilde: Float64Array, gmm: GMMParams, sigma: number): number {
  const r = gmmDenoiser(xTilde, gmm, sigma);
  const s = gmmScore(xTilde, gmm, sigma);
  let acc = 0;
  for (let j = 0; j < gmm.p; j++) {
    const lhs = r[j] - xTilde[j];
    const rhs = sigma * sigma * s[j];
    const d = lhs - rhs;
    acc += d * d;
  }
  return Math.sqrt(acc);
}

// =============================================================================
// §5.2-§5.3 — InfoNCE machinery.
// =============================================================================

/**
 * InfoNCE / NT-Xent cross-entropy for a batch of similarity scores. Expects
 * an unnormalized K × K similarity matrix `sim` (row-major) with positives
 * on the diagonal. Temperature τ is applied as a divisor. Numerically stable
 * via logSumExp.
 *
 * The returned value is the mean per-anchor cross-entropy — in (0, log K).
 */
export function infoNCELoss(sim: Float64Array, K: number, tau: number): number {
  let total = 0;
  const row = new Array<number>(K);
  for (let i = 0; i < K; i++) {
    for (let j = 0; j < K; j++) row[j] = sim[i * K + j] / tau;
    total += -row[i] + logSumExp(row);
  }
  return total / K;
}

/**
 * Wang-Isola alignment loss in cosine form, on a batch of K positive pairs.
 * z is a (2K × d) row-major matrix with rows [z_1, z_1+, z_2, z_2+, …].
 * Each z_i, z_i+ is assumed already ℓ_2-normalized.
 * Returns -E[<z_i, z_i+>] / τ (negate the cosine alignment, scale by 1/τ).
 */
export function alignmentLoss(z: Float64Array, K: number, d: number, tau: number): number {
  let s = 0;
  for (let i = 0; i < K; i++) {
    const ai = 2 * i * d;
    const pi = (2 * i + 1) * d;
    let dot = 0;
    for (let j = 0; j < d; j++) dot += z[ai + j] * z[pi + j];
    s += dot;
  }
  return -s / (K * tau);
}

/**
 * Wang-Isola uniformity loss in cosine form. Same input layout as alignmentLoss.
 * Returns log E_x log E_x' exp(<z, z'>/τ) averaged over all 2K anchors and all
 * 2K - 1 candidate counterparts (self-pair excluded).
 */
export function uniformityLoss(z: Float64Array, K: number, d: number, tau: number): number {
  const N = 2 * K;
  const expScores = new Array<number>(N - 1);
  let total = 0;
  for (let i = 0; i < N; i++) {
    const bi = i * d;
    let idx = 0;
    for (let j = 0; j < N; j++) {
      if (j === i) continue;
      const bj = j * d;
      let dot = 0;
      for (let k = 0; k < d; k++) dot += z[bi + k] * z[bj + k];
      expScores[idx++] = dot / tau;
    }
    total += logSumExp(expScores) - Math.log(N - 1);
  }
  return total / N;
}

/**
 * Convenience: returns both Wang-Isola metrics in one pass.
 */
export function alignmentUniformity(
  z: Float64Array,
  K: number,
  d: number,
  tau: number,
): { align: number; uniform: number } {
  return { align: alignmentLoss(z, K, d, tau), uniform: uniformityLoss(z, K, d, tau) };
}

// =============================================================================
// §7.2 — Gaussian information-bottleneck frontier (closed form, Theorem 7.1).
//
// For (X, Y) standard bivariate Gaussian with correlation ρ, the IB frontier
//   D(R) = (1/2) log [e^{2R} / (ρ² + (1 - ρ²) e^{2R})]
// where R = I(X; T), D = I(T; Y). Concave; asymptotes at I(X; Y) = -½log(1-ρ²).
// =============================================================================

/** Closed-form Gaussian IB curve. Returns D(R) for each R in the grid. */
export function gaussianIBCurve(rho: number, RGrid: Float64Array): Float64Array {
  const D = new Float64Array(RGrid.length);
  const rhoSq = rho * rho;
  for (let i = 0; i < RGrid.length; i++) {
    const e2R = Math.exp(2 * RGrid[i]);
    D[i] = 0.5 * Math.log(e2R / (rhoSq + (1 - rhoSq) * e2R));
  }
  return D;
}

/** Asymptotic IB ceiling I(X; Y) = -½ log(1 - ρ²) for a bivariate Gaussian. */
export function gaussianMI(rho: number): number {
  return -0.5 * Math.log(1 - rho * rho);
}

// =============================================================================
// §9.2 — Linear centered-kernel alignment (CKA).
//
// CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F · ||Y^T Y||_F), applied to
// column-centered representation matrices X (n × d_x) and Y (n × d_y).
// Takes values in [0, 1]; invariant to orthogonal transformations and
// isotropic scaling. Reference: Kornblith et al. (2019).
// =============================================================================

/**
 * Compute linear CKA between two n × d row-major representations.
 * Inputs are NOT mutated; centering is applied to local copies.
 */
export function cka(
  X: Float64Array,
  Y: Float64Array,
  n: number,
  dx: number,
  dy: number,
): number {
  const Xc = new Float64Array(X);
  const Yc = new Float64Array(Y);
  centerColumns(Xc, n, dx);
  centerColumns(Yc, n, dy);
  // ||Y^T X||_F^2 = sum_{j,l} (Σ_i Y_ij X_il)^2  — work via the dy × dx cross-cov matrix.
  let crossSq = 0;
  for (let j = 0; j < dy; j++) {
    for (let l = 0; l < dx; l++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += Yc[i * dy + j] * Xc[i * dx + l];
      crossSq += s * s;
    }
  }
  // ||X^T X||_F^2 (then sqrt). X^T X is symmetric, so we compute the upper
  // triangle once and double off-diagonal contributions — halves the work.
  let xxSq = 0;
  for (let j = 0; j < dx; j++) {
    for (let l = j; l < dx; l++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += Xc[i * dx + j] * Xc[i * dx + l];
      xxSq += j === l ? s * s : 2 * s * s;
    }
  }
  let yySq = 0;
  for (let j = 0; j < dy; j++) {
    for (let l = j; l < dy; l++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += Yc[i * dy + j] * Yc[i * dy + l];
      yySq += j === l ? s * s : 2 * s * s;
    }
  }
  const denom = Math.sqrt(xxSq) * Math.sqrt(yySq);
  return denom > 0 ? crossSq / denom : 0;
}

// =============================================================================
// §4.3 — Variational autoencoder ELBO (Gaussian encoder + Gaussian decoder).
//
// ELBO = E_q[log p(x | z)] - D_KL(q(z|x) || p(z))
//      = -||x - μ_x(z)||² / (2 σ_x²) + reconst-const
//        - (1/2) Σ_j (μ_z[j]² + σ_z²[j] - 1 - log σ_z²[j])
// The constants for a Gaussian decoder cancel out for comparison/training
// purposes; we return the negative-log-density form for the reconstruction.
// =============================================================================

/** Closed-form Gaussian-vs-standard-normal KL divergence, per-sample. */
export function gaussianKLDivergence(muZ: Float64Array, sigmaZ: Float64Array): number {
  let kl = 0;
  for (let j = 0; j < muZ.length; j++) {
    const m = muZ[j];
    const s = sigmaZ[j];
    kl += m * m + s * s - 1 - 2 * Math.log(s);
  }
  return 0.5 * kl;
}

/**
 * Compute the per-sample ELBO for a single (x, μ_z, σ_z) tuple with Gaussian
 * decoder N(x_recon, σ_x² I). Returns the recon, KL, and ELBO = recon - KL.
 * Reconstruction is the negative squared error normalized by σ_x² — the same
 * up-to-const term as in the β-VAE loss.
 */
export function vaeElbo(
  x: Float64Array,
  xRecon: Float64Array,
  muZ: Float64Array,
  sigmaZ: Float64Array,
  sigmaX: number,
): { recon: number; kl: number; elbo: number } {
  let sq = 0;
  for (let j = 0; j < x.length; j++) {
    const d = x[j] - xRecon[j];
    sq += d * d;
  }
  const recon = -sq / (2 * sigmaX * sigmaX);
  const kl = gaussianKLDivergence(muZ, sigmaZ);
  return { recon, kl, elbo: recon - kl };
}

// =============================================================================
// §12.1 — Fisher discriminant ratio: tr(Σ_B) / tr(Σ_W).
//
// Diagnostic for between-class vs within-class spread in a representation.
// High FDR ⇒ linearly separable; low FDR ⇒ classes smeared.
// =============================================================================

/**
 * Compute the Fisher discriminant ratio for an n × d representation Z with
 * integer class labels y in [0, K). Returns {fdr, betweenTrace, withinTrace}.
 */
export function fisherDiscriminantRatio(
  Z: Float64Array,
  y: Int32Array,
  n: number,
  d: number,
): { fdr: number; betweenTrace: number; withinTrace: number } {
  // Discover K.
  let K = 0;
  for (let i = 0; i < n; i++) if (y[i] + 1 > K) K = y[i] + 1;
  const classMeans = new Float64Array(K * d);
  const classCounts = new Int32Array(K);
  for (let i = 0; i < n; i++) {
    const k = y[i];
    classCounts[k]++;
    for (let j = 0; j < d; j++) classMeans[k * d + j] += Z[i * d + j];
  }
  for (let k = 0; k < K; k++) {
    if (classCounts[k] === 0) continue;
    for (let j = 0; j < d; j++) classMeans[k * d + j] /= classCounts[k];
  }
  const globalMean = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) globalMean[j] += Z[i * d + j];
  for (let j = 0; j < d; j++) globalMean[j] /= n;
  // Between-class trace: Σ_k (n_k/n) ||μ_k - μ||².
  let betweenTrace = 0;
  for (let k = 0; k < K; k++) {
    if (classCounts[k] === 0) continue;
    let sq = 0;
    for (let j = 0; j < d; j++) {
      const dj = classMeans[k * d + j] - globalMean[j];
      sq += dj * dj;
    }
    betweenTrace += (classCounts[k] / n) * sq;
  }
  // Within-class trace: (1/n) Σ_k Σ_{i:y_i=k} ||z_i - μ_k||².
  let withinTrace = 0;
  for (let i = 0; i < n; i++) {
    const k = y[i];
    let sq = 0;
    for (let j = 0; j < d; j++) {
      const dj = Z[i * d + j] - classMeans[k * d + j];
      sq += dj * dj;
    }
    withinTrace += sq;
  }
  withinTrace /= n;
  const fdr = withinTrace > 0 ? betweenTrace / withinTrace : Infinity;
  return { fdr, betweenTrace, withinTrace };
}

// =============================================================================
// §12.3 — Neural collapse diagnostics.
//
// NC1: within-class variance collapses (tr(Σ_W) → 0).
// NC2: class-mean cosine similarities → -1/(K-1) (simplex ETF).
// =============================================================================

/** Mean off-diagonal pairwise cosine similarity of K class means (length-d each). */
export function classMeanCosineSimilarity(
  means: Float64Array,
  K: number,
  d: number,
): number {
  // Center the K means (subtract the global mean) and unit-normalize.
  const centered = new Float64Array(K * d);
  for (let k = 0; k < K; k++) for (let j = 0; j < d; j++) centered[k * d + j] = means[k * d + j];
  const global = new Float64Array(d);
  for (let k = 0; k < K; k++) for (let j = 0; j < d; j++) global[j] += means[k * d + j];
  for (let j = 0; j < d; j++) global[j] /= K;
  for (let k = 0; k < K; k++) for (let j = 0; j < d; j++) centered[k * d + j] -= global[j];
  const norms = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    let s = 0;
    for (let j = 0; j < d; j++) s += centered[k * d + j] * centered[k * d + j];
    norms[k] = Math.sqrt(s);
  }
  let cnt = 0;
  let acc = 0;
  for (let i = 0; i < K; i++) {
    if (norms[i] === 0) continue;
    for (let j = i + 1; j < K; j++) {
      if (norms[j] === 0) continue;
      let dot = 0;
      for (let l = 0; l < d; l++) dot += centered[i * d + l] * centered[j * d + l];
      acc += dot / (norms[i] * norms[j]);
      cnt++;
    }
  }
  return cnt > 0 ? acc / cnt : 0;
}

/** Simplex ETF target cosine: -1/(K-1). */
export function simplexETFTarget(K: number): number {
  return K > 1 ? -1 / (K - 1) : 0;
}

// =============================================================================
// §4.4 — Reparametrization vs score-function gradient variance.
//
// Score (REINFORCE):    grad ≈ f(z) ∇ log q(z; θ)
// Reparametrization:    z = μ + σ ⊙ ε, ε ∼ N(0, I); grad = ∇_θ f(μ + σ ⊙ ε)
// The reparam estimator routinely has order-of-magnitude lower variance on
// smooth f, especially when f's magnitude does not depend on θ.
// =============================================================================

/**
 * Tiny single-example MC variance comparison. For f(z) = ½ ||z||² (a generic
 * stand-in for a smooth VAE reconstruction term), generate B gradient samples
 * via each estimator and return the per-coordinate empirical standard deviation
 * of (∂f/∂μ, ∂f/∂logσ²).
 */
export function reparamVsScoreVariance(
  mu: Float64Array,
  logSigmaSq: Float64Array,
  options: { B?: number; seed?: number } = {},
): {
  reparamMuStd: Float64Array;
  scoreMuStd: Float64Array;
  reparamLogSigmaStd: Float64Array;
  scoreLogSigmaStd: Float64Array;
} {
  const d = mu.length;
  const B = options.B ?? 200;
  const seed = options.seed ?? 1;
  const rng = mulberry32(seed);
  const sigma = new Float64Array(d);
  for (let j = 0; j < d; j++) sigma[j] = Math.exp(0.5 * logSigmaSq[j]);
  // Allocate storage for per-coord per-sample gradients.
  const repMu = new Float64Array(B * d);
  const repLs = new Float64Array(B * d);
  const scMu = new Float64Array(B * d);
  const scLs = new Float64Array(B * d);
  for (let b = 0; b < B; b++) {
    // Draw d independent ε via gaussianPair (handles odd d by discarding last extra).
    const eps = new Float64Array(d);
    for (let j = 0; j < d; j += 2) {
      const [a, c] = gaussianPair(rng);
      eps[j] = a;
      if (j + 1 < d) eps[j + 1] = c;
    }
    const z = new Float64Array(d);
    for (let j = 0; j < d; j++) z[j] = mu[j] + sigma[j] * eps[j];
    // f(z) = ½ ||z||²,  ∇_z f = z.
    const fVal = 0.5 * z.reduce((a, x) => a + x * x, 0);
    // Reparam:  ∂f/∂μ = ∂f/∂z · 1 = z;  ∂f/∂logσ² = ∂f/∂z · (σ ε /2) = z · σ ε / 2.
    for (let j = 0; j < d; j++) {
      repMu[b * d + j] = z[j];
      repLs[b * d + j] = z[j] * sigma[j] * eps[j] * 0.5;
    }
    // Score:   ∂log q/∂μ = (z - μ) / σ²;
    //          ∂log q/∂logσ² = ½ ((z-μ)²/σ² - 1).
    for (let j = 0; j < d; j++) {
      const r = z[j] - mu[j];
      scMu[b * d + j] = fVal * (r / (sigma[j] * sigma[j]));
      scLs[b * d + j] = fVal * 0.5 * ((r * r) / (sigma[j] * sigma[j]) - 1);
    }
  }
  const stdOf = (buf: Float64Array): Float64Array => {
    const s = new Float64Array(d);
    const m = new Float64Array(d);
    for (let b = 0; b < B; b++) for (let j = 0; j < d; j++) m[j] += buf[b * d + j];
    for (let j = 0; j < d; j++) m[j] /= B;
    for (let b = 0; b < B; b++) {
      for (let j = 0; j < d; j++) {
        const v = buf[b * d + j] - m[j];
        s[j] += v * v;
      }
    }
    for (let j = 0; j < d; j++) s[j] = Math.sqrt(s[j] / B);
    return s;
  };
  return {
    reparamMuStd: stdOf(repMu),
    scoreMuStd: stdOf(scMu),
    reparamLogSigmaStd: stdOf(repLs),
    scoreLogSigmaStd: stdOf(scLs),
  };
}
