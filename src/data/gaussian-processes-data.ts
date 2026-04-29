// =============================================================================
// gaussian-processes-data.ts
//
// Canonical fixture data for the gaussian-processes topic. The §3 sparse and
// dense training observations and the §4/§5 dataset constants are hard-coded
// here as they appear in the verified notebook
// (notebooks/gaussian-processes/01_gaussian_processes.ipynb), so that
// TypeScript viz components reproduce the notebook's computations bit-for-bit
// (deterministic under our Mulberry32 PRNG once the underlying y values are
// pinned).
//
// The y-values were extracted by running a separate Python script that
// reproduces the notebook's data-generating recipe verbatim:
//
//   rng_train = np.random.default_rng(SEED + 30)            # SEED = 42
//   X_train_sparse = np.array([-2.6, -1.8, -0.5, 0.7, 1.6, 2.4])  # hand-placed
//   y_train_sparse = f_true(X) + 0.15 * rng_train.standard_normal(6)
//
// Re-running with NumPy's PCG64 reproduces these to >10 decimal places. Mulberry32
// in TS would NOT reproduce them — that's why we pin the canonical values here.
// =============================================================================

/** Ground-truth latent function used for the §3 worked example.
 *  f(x) = sin(x) + 0.3 sin(3x) — two distinct frequency components. */
export function fTrue(x: number): number {
  return Math.sin(x) + 0.3 * Math.sin(3.0 * x);
}

/** Observation-noise standard deviation used throughout §3, §4, §5. */
export const SIGMA_N_TRUE = 0.15;

/** SE-kernel ground-truth output scale used in §3 and §5. */
export const SIGMA_F_TRUE = 1.0;

/** SE-kernel ground-truth lengthscale used in §3 and §5. */
export const ELL_TRUE = 0.6;

// -----------------------------------------------------------------------------
// §3 Sparse training set — six hand-placed points used in panels (a) and (b)
// -----------------------------------------------------------------------------

/** Six hand-placed training inputs for §3 panels (a) and (b). */
export const X_TRAIN_SPARSE: ReadonlyArray<number> = [
  -2.6, -1.8, -0.5, 0.7, 1.6, 2.4,
];

/** Corresponding training observations: f_true(X_TRAIN_SPARSE) + 0.15 * z,
 *  where z are draws from numpy.random.default_rng(72) (SEED + 30) standard
 *  normal — pinned here so TypeScript reproduces the notebook's panel-(a)
 *  posterior μ_* and σ_* curves exactly. */
export const Y_TRAIN_SPARSE: ReadonlyArray<number> = [
  -0.9655632949, -0.7462629395, -0.6908200037,
  0.6501179886, 0.4911493517, 0.7699109818,
];

// -----------------------------------------------------------------------------
// §3 Dense training set — n=30, used in panel (c) and as the §5 hyperparameter
// recovery dataset
// -----------------------------------------------------------------------------

/** Thirty training inputs spaced near linspace(-3, 3, 30) with small
 *  uniform-jitter perturbations. Pinned to the notebook's PCG64 draws. */
export const X_TRAIN_DENSE: ReadonlyArray<number> = [
  -3.0546652953, -2.7541990567, -2.6052874118, -2.36451679, -2.1555916813,
  -2.062856574, -1.7335519104, -1.6000333472, -1.4070360951, -1.0821245752,
  -1.0116757272, -0.8035804343, -0.5593105495, -0.2400444888, -0.0572267613,
  0.0951172462, 0.4006962826, 0.4792425417, 0.6343725147, 0.9645181257,
  1.1487563697, 1.3621021061, 1.4969562618, 1.7868255229, 1.9808724154,
  2.1098458114, 2.4061551555, 2.6001462832, 2.8643250961, 2.9802316249,
];

/** Dense training observations corresponding to X_TRAIN_DENSE. */
export const Y_TRAIN_DENSE: ReadonlyArray<number> = [
  -0.1258516521, -0.5135797611, -0.9952270018, -0.9973088312, -1.0363581574,
  -0.9360804626, -0.5198179754, -0.5956286408, -0.6604883811, -0.8027246369,
  -0.796408218, -0.7071226353, -0.7327478472, -0.212717392, -0.0680310238,
  0.1862099219, 0.7231588459, 0.7183236996, 0.9552637965, 0.5141903385,
  0.7582867695, 0.6500746242, 0.5595637936, 0.8377186369, 0.9696758017,
  1.0399400921, 0.7884099531, 0.887630711, 0.2529317113, 0.2373044007,
];

// -----------------------------------------------------------------------------
// Test-grid factories — convenience wrappers for the canonical evaluation
// grids used across the topic
// -----------------------------------------------------------------------------

/** Linspace utility — closed analog of np.linspace(start, stop, n) inclusive. */
export function linspace(start: number, stop: number, n: number): number[] {
  if (n === 1) return [start];
  const out = new Array<number>(n);
  const dx = (stop - start) / (n - 1);
  for (let i = 0; i < n; i++) out[i] = start + i * dx;
  return out;
}

/** §1 evaluation grid — 200 points in [-3, 3]. */
export const xGridS1 = (): number[] => linspace(-3.0, 3.0, 200);

/** §2 evaluation grid — 300 points in [-3, 3]. */
export const xGridS2 = (): number[] => linspace(-3.0, 3.0, 300);

/** §3, §4, §5 evaluation grid — 300 points in [-3.5, 3.5]. */
export const xGridS345 = (): number[] => linspace(-3.5, 3.5, 300);

/** §1 panel (a) polynomial centers / RBF centers — 6 fixed centers in [-3, 3]. */
export const RBF_CENTERS: ReadonlyArray<number> = [-3, -1.8, -0.6, 0.6, 1.8, 3];

// -----------------------------------------------------------------------------
// §1 polynomial / RBF feature maps — used by GPFunctionSpaceLift
// -----------------------------------------------------------------------------

/** Polynomial feature map: returns a (len(x), D) matrix with columns 1, x, ..., x^{D-1}. */
export function phiPoly(x: number[], D: number): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < x.length; i++) {
    const row = new Array<number>(D);
    let v = 1;
    for (let k = 0; k < D; k++) {
      row[k] = v;
      v *= x[i];
    }
    out.push(row);
  }
  return out;
}

/** RBF feature map at fixed centers: returns a (len(x), K) matrix.
 *  φ(x)_k = exp(-(x - c_k)² / (2 ω²)). */
export function phiRBF(x: number[], centers: ReadonlyArray<number>, omega: number): number[][] {
  const w2 = 2 * omega * omega;
  const out: number[][] = [];
  for (let i = 0; i < x.length; i++) {
    const row = new Array<number>(centers.length);
    const xi = x[i];
    for (let k = 0; k < centers.length; k++) {
      const d = xi - centers[k];
      row[k] = Math.exp(-(d * d) / w2);
    }
    out.push(row);
  }
  return out;
}
