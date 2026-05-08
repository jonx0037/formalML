// =============================================================================
// bnn-grid-render.ts
//
// Pure helpers shared across the §§1, 2, 4, 5 BNN viz components. Each viz
// builds a 2D prediction grid over Two Moons input space, runs K MLP forward
// passes (or the variance/mean across them), and paints the result to a
// canvas-backed PNG dataURL embedded in an SVG <image>. These helpers are
// the substrate of that pipeline.
//
// The canonical grid is math-coordinate row-major:
//   gridFlat[(i * res + j) * 2 + 0] = x_i = xMin + (i / (res-1)) * (xMax - xMin)
//   gridFlat[(i * res + j) * 2 + 1] = y_j = yMin + (j / (res-1)) * (yMax - yMin)
// (i indexes x ascending; j indexes y ascending.)
//
// `mathRowMajorToCanvas` flips the y axis so the resulting canvas-row-major
// array lays out top-row first (canvas screen-space convention). The same
// canvas-major array is consumed by both `canvasFromColor` (heatmap painting)
// and `d3.contours` (marching squares for decision boundaries), guaranteeing
// pixel-perfect alignment between the two panel families.
// =============================================================================

import * as d3 from 'd3';

export interface GridConfig {
  res: number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

export const DEFAULT_GRID: GridConfig = {
  res: 60,
  xMin: -2.5,
  xMax: 2.5,
  yMin: -2.0,
  yMax: 2.0,
};

/** Flat (n × 2) row-major grid in math coordinates: x ascending in i, y ascending in j. */
export function buildGridFlat(cfg: GridConfig = DEFAULT_GRID): Float32Array {
  const { res, xMin, xMax, yMin, yMax } = cfg;
  const out = new Float32Array(res * res * 2);
  for (let i = 0; i < res; i++) {
    const x = xMin + (i / (res - 1)) * (xMax - xMin);
    for (let j = 0; j < res; j++) {
      const y = yMin + (j / (res - 1)) * (yMax - yMin);
      const idx = (i * res + j) * 2;
      out[idx] = x;
      out[idx + 1] = y;
    }
  }
  return out;
}

/** Convert math-row-major (i*res + j) to canvas-row-major (canvasRow*res + canvasCol)
 *  with canvas y flipped: canvasRow = res - 1 - j. */
export function mathRowMajorToCanvas(probs: Float32Array, res: number): Float32Array {
  const out = new Float32Array(res * res);
  for (let i = 0; i < res; i++) {
    for (let j = 0; j < res; j++) {
      const canvasRow = res - 1 - j;
      out[canvasRow * res + i] = probs[i * res + j];
    }
  }
  return out;
}

/** Render a canvas-row-major Float32Array to an off-DOM canvas via the supplied
 *  scalar→color map and return a base64 PNG dataURL suitable for an SVG <image>.
 *  Returns '' if no canvas context is available (SSR / non-DOM environment). */
export function canvasFromColor(
  values: Float32Array,
  res: number,
  colorMap: (v: number) => string,
): string {
  if (typeof document === 'undefined') return '';
  const canvas = document.createElement('canvas');
  canvas.width = res;
  canvas.height = res;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';
  const imgData = ctx.createImageData(res, res);
  for (let r = 0; r < res; r++) {
    for (let c = 0; c < res; c++) {
      const v = values[r * res + c];
      const rgb = d3.color(colorMap(v))?.rgb();
      const idx = (r * res + c) * 4;
      if (rgb) {
        imgData.data[idx + 0] = rgb.r;
        imgData.data[idx + 1] = rgb.g;
        imgData.data[idx + 2] = rgb.b;
        imgData.data[idx + 3] = 255;
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL();
}

/** Pointwise mean over the first K elements of a list of equal-length arrays. */
export function meanOverK(arrays: Float32Array[], K: number): Float32Array {
  const n = arrays[0].length;
  const out = new Float32Array(n);
  for (let k = 0; k < K; k++) {
    const a = arrays[k];
    for (let i = 0; i < n; i++) out[i] += a[i];
  }
  for (let i = 0; i < n; i++) out[i] /= K;
  return out;
}

/** Pointwise unbiased sample variance over the first K elements of a list of arrays. */
export function varianceOverK(arrays: Float32Array[], K: number): Float32Array {
  const n = arrays[0].length;
  const mean = meanOverK(arrays, K);
  const out = new Float32Array(n);
  for (let k = 0; k < K; k++) {
    const a = arrays[k];
    for (let i = 0; i < n; i++) {
      const d = a[i] - mean[i];
      out[i] += d * d;
    }
  }
  const denom = Math.max(K - 1, 1);
  for (let i = 0; i < n; i++) out[i] /= denom;
  return out;
}

/** Pointwise standard deviation (unbiased) — convenience wrapper around varianceOverK. */
export function stdOverK(arrays: Float32Array[], K: number): Float32Array {
  const v = varianceOverK(arrays, K);
  for (let i = 0; i < v.length; i++) v[i] = Math.sqrt(v[i]);
  return v;
}

/** Compute the SVG path d-string for the 0.5 isocontour of a single math-row-major
 *  probability grid, scaled to a (panelW, panelH) viewport. */
export function isoContourPath(
  probs: Float32Array,
  res: number,
  panelW: number,
  panelH: number,
  threshold = 0.5,
): string {
  const arr = Array.from(mathRowMajorToCanvas(probs, res));
  const polys = d3.contours().size([res, res]).thresholds([threshold])(arr);
  if (polys.length === 0) return '';
  const cellW = panelW / (res - 1);
  const cellH = panelH / (res - 1);
  const segments: string[] = [];
  for (const polygon of polys[0].coordinates) {
    for (const ring of polygon) {
      if (ring.length === 0) continue;
      const pts = ring.map(([col, row]) => `${(col * cellW).toFixed(2)},${(row * cellH).toFixed(2)}`);
      segments.push(`M ${pts.join(' L ')}`);
    }
  }
  return segments.join(' ');
}
