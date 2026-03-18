// Pre-computed SVD data for the Image Compression Demo.
// A 64×64 synthetic image constructed from DCT-II basis vectors with a permuted
// column basis, giving the SVD for free (exact by construction).
// The image has both smooth low-frequency content (captured by leading singular
// values) and fine high-frequency detail (requiring many components).

export const IMAGE_SIZE = 64;

// ─── Singular values: exponential decay + small floor ───
// Designed so rank 5 captures ~60% energy, rank 15 captures ~90%, rank 30 captures ~98%.

export const singularValues: number[] = Array.from({ length: IMAGE_SIZE }, (_, k) =>
  180 * Math.exp(-0.12 * k) + 3,
);

// ─── Column-basis permutation for Vt rows ───
// vtOrder[k] = (k * 7) % 64 is a permutation (gcd(7, 64) = 1).
// Pairs low-frequency row patterns with medium/high-frequency column patterns,
// creating rich 2D structure in the image.

export const vtOrder: number[] = Array.from({ length: IMAGE_SIZE }, (_, k) =>
  (k * 7) % IMAGE_SIZE,
);

// ─── DCT-II basis function ───

function dctBasisValue(N: number, k: number, i: number): number {
  const scale = k === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N);
  return scale * Math.cos((Math.PI * (2 * i + 1) * k) / (2 * N));
}

// ─── Reconstruction ───

/** Compute a single raw pixel value for the rank-r approximation. */
export function reconstructPixelRaw(i: number, j: number, rank: number): number {
  let val = 0;
  const r = Math.min(rank, IMAGE_SIZE);
  for (let k = 0; k < r; k++) {
    val += singularValues[k] * dctBasisValue(IMAGE_SIZE, k, i) * dctBasisValue(IMAGE_SIZE, vtOrder[k], j);
  }
  return val;
}

/** Compute the full raw image (before display scaling). */
function computeRawImage(): { pixels: number[][]; min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  const pixels: number[][] = [];
  for (let i = 0; i < IMAGE_SIZE; i++) {
    const row: number[] = [];
    for (let j = 0; j < IMAGE_SIZE; j++) {
      const v = reconstructPixelRaw(i, j, IMAGE_SIZE);
      if (v < min) min = v;
      if (v > max) max = v;
      row.push(v);
    }
    pixels.push(row);
  }
  return { pixels, min, max };
}

const rawImage = computeRawImage();

/** Raw pixel range (used for display scaling). */
export const pixelMin = rawImage.min;
export const pixelMax = rawImage.max;

/** Map a raw pixel value to display range [0, 255]. */
export function toDisplayValue(raw: number): number {
  const range = pixelMax - pixelMin;
  if (range < 1e-10) return 128;
  return Math.max(0, Math.min(255, Math.round(255 * (raw - pixelMin) / range)));
}

/** Pre-computed original image as display values (0–255). */
export const originalPixels: number[][] = rawImage.pixels.map((row) =>
  row.map(toDisplayValue),
);

// ─── Energy statistics ───

const totalEnergy = singularValues.reduce((s, v) => s + v * v, 0);
const cumulativeEnergy: number[] = [];
{
  let sum = 0;
  for (const s of singularValues) {
    sum += s * s;
    cumulativeEnergy.push(sum);
  }
}

/** Fraction of total energy retained by the rank-k approximation. */
export function energyRetained(k: number): number {
  if (k <= 0) return 0;
  if (k >= IMAGE_SIZE) return 1;
  return cumulativeEnergy[k - 1] / totalEnergy;
}
