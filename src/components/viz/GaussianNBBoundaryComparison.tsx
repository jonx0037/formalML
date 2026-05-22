import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  paletteNB,
  mulberry32,
  gaussianSampler,
  gaussianNBFit,
  gaussianNBPredict,
} from './shared/naive-bayes';

// =============================================================================
// GaussianNBBoundaryComparison — §4.3 / §4.4 companion. Two-class 2D Gaussian
// data with adjustable per-class anisotropy and a shared rotation; the viewer
// fits and overlays three decision boundaries:
//
//   GNB  — diagonal class-covariance (Gaussian Naive Bayes)
//   LDA  — shared covariance across classes
//   QDA  — full per-class covariance
//
// The three reduce to the same boundary in special cases (LDA = GNB when the
// shared covariance happens to be diagonal; QDA = GNB when class covariances
// are diagonal). The viz shows how the boundary changes shape as the design
// breaks each assumption.
// =============================================================================

const N_PER_CLASS = 120;
const SCATTER_R = 3;
const GRID = 80;

interface Sample {
  X: Float64Array;
  y: Int32Array;
}

function generateData(
  anisotropy0: number,
  anisotropy1: number,
  rotation: number,
  seed: number,
): Sample {
  const rng = gaussianSampler(mulberry32(seed));
  const X = new Float64Array(2 * N_PER_CLASS * 2);
  const y = new Int32Array(2 * N_PER_CLASS);
  const cos = Math.cos(rotation);
  const sin = Math.sin(rotation);
  // Class 0 at (-1.2, 0), anisotropic with major axis = anisotropy0.
  for (let i = 0; i < N_PER_CLASS; i++) {
    const u = rng() * anisotropy0;
    const v = rng() * (1 / Math.max(anisotropy0, 0.2));
    const xr = u * cos - v * sin;
    const yr = u * sin + v * cos;
    X[i * 2] = xr - 1.2;
    X[i * 2 + 1] = yr;
    y[i] = 0;
  }
  // Class 1 at (+1.2, 0), anisotropic with major axis = anisotropy1.
  for (let i = 0; i < N_PER_CLASS; i++) {
    const u = rng() * anisotropy1;
    const v = rng() * (1 / Math.max(anisotropy1, 0.2));
    const xr = u * cos - v * sin;
    const yr = u * sin + v * cos;
    X[(N_PER_CLASS + i) * 2] = xr + 1.2;
    X[(N_PER_CLASS + i) * 2 + 1] = yr;
    y[N_PER_CLASS + i] = 1;
  }
  return { X, y };
}

// LDA / QDA: closed-form 2D Gaussian fits with optional shared covariance.
interface QuadraticModel {
  // Per-class (mean, 2×2 covariance, log determinant, log prior).
  means: Array<[number, number]>;
  covInv: Array<[number, number, number, number]>; // [a, b, c, d] for 2×2 inverse
  logDet: number[];
  logPrior: number[];
}

function fitQuadratic(X: Float64Array, y: Int32Array, sharedCov: boolean): QuadraticModel {
  const n = y.length;
  const classes = [0, 1];
  const means: Array<[number, number]> = [[0, 0], [0, 0]];
  const counts = [0, 0];
  for (let i = 0; i < n; i++) {
    const c = y[i];
    means[c][0] += X[i * 2];
    means[c][1] += X[i * 2 + 1];
    counts[c]++;
  }
  for (let c = 0; c < 2; c++) {
    means[c][0] /= counts[c];
    means[c][1] /= counts[c];
  }
  // Class covariances.
  const cov: Array<[number, number, number, number]> = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ];
  for (let i = 0; i < n; i++) {
    const c = y[i];
    const dx = X[i * 2] - means[c][0];
    const dy = X[i * 2 + 1] - means[c][1];
    cov[c][0] += dx * dx;
    cov[c][1] += dx * dy;
    cov[c][2] += dx * dy;
    cov[c][3] += dy * dy;
  }
  if (sharedCov) {
    // Pooled covariance.
    const pooled: [number, number, number, number] = [
      (cov[0][0] + cov[1][0]) / n,
      (cov[0][1] + cov[1][1]) / n,
      (cov[0][2] + cov[1][2]) / n,
      (cov[0][3] + cov[1][3]) / n,
    ];
    cov[0] = pooled.slice() as [number, number, number, number];
    cov[1] = pooled.slice() as [number, number, number, number];
  } else {
    for (let c = 0; c < 2; c++) {
      cov[c][0] /= counts[c];
      cov[c][1] /= counts[c];
      cov[c][2] /= counts[c];
      cov[c][3] /= counts[c];
    }
  }
  const covInv: Array<[number, number, number, number]> = [];
  const logDet: number[] = [];
  for (let c = 0; c < 2; c++) {
    const [a, b, , d] = cov[c];
    const det = Math.max(a * d - b * b, 1e-12);
    covInv.push([d / det, -b / det, -b / det, a / det]);
    logDet.push(Math.log(det));
  }
  const logPrior = [Math.log(counts[0] / n), Math.log(counts[1] / n)];
  return { means, covInv, logDet, logPrior };
}

function predictQuadratic(x: number, y: number, m: QuadraticModel): number {
  // Score per class: −0.5 log|Σ| − 0.5 (x−μ)ᵀ Σ⁻¹ (x−μ) + log π
  let best = -Infinity;
  let arg = 0;
  for (let c = 0; c < 2; c++) {
    const dx = x - m.means[c][0];
    const dy = y - m.means[c][1];
    const [a, b, _bb, d] = m.covInv[c];
    const quad = a * dx * dx + 2 * b * dx * dy + d * dy * dy;
    void _bb;
    const s = -0.5 * m.logDet[c] - 0.5 * quad + m.logPrior[c];
    if (s > best) {
      best = s;
      arg = c;
    }
  }
  return arg;
}

type Estimator = 'GNB' | 'LDA' | 'QDA';

export default function GaussianNBBoundaryComparison() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [anisotropy0, setAnisotropy0] = useState(1.4);
  const [anisotropy1, setAnisotropy1] = useState(0.9);
  const [rotation, setRotation] = useState(0.3);
  const [seed, setSeed] = useState(42);
  const [enabled, setEnabled] = useState<Record<Estimator, boolean>>({
    GNB: true,
    LDA: true,
    QDA: true,
  });

  const W = Math.max(360, Math.min(width || 720, 720));
  const H = 400;
  const margin = { top: 14, right: 16, bottom: 36, left: 42 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  const data = useMemo(
    () => generateData(anisotropy0, anisotropy1, rotation, seed),
    [anisotropy0, anisotropy1, rotation, seed],
  );

  // Compute boundary grid for each enabled estimator.
  const boundaries = useMemo(() => {
    const xMin = -5;
    const xMax = 5;
    const yMin = -3;
    const yMax = 3;
    const dx = (xMax - xMin) / GRID;
    const dy = (yMax - yMin) / GRID;
    const result: Record<Estimator, number[][]> = {
      GNB: Array.from({ length: GRID }, () => new Array(GRID).fill(0)),
      LDA: Array.from({ length: GRID }, () => new Array(GRID).fill(0)),
      QDA: Array.from({ length: GRID }, () => new Array(GRID).fill(0)),
    };
    const gnb = gaussianNBFit(data.X, data.y, 2);
    const lda = fitQuadratic(data.X, data.y, true);
    const qda = fitQuadratic(data.X, data.y, false);
    const pt = new Float64Array(2);
    for (let i = 0; i < GRID; i++) {
      for (let j = 0; j < GRID; j++) {
        const x = xMin + i * dx;
        const y = yMin + j * dy;
        pt[0] = x;
        pt[1] = y;
        if (enabled.GNB) result.GNB[i][j] = gaussianNBPredict(pt, gnb)[0];
        if (enabled.LDA) result.LDA[i][j] = predictQuadratic(x, y, lda);
        if (enabled.QDA) result.QDA[i][j] = predictQuadratic(x, y, qda);
      }
    }
    return { result, xMin, xMax, yMin, yMax, dx, dy };
  }, [data, enabled]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const x = d3.scaleLinear().domain([boundaries.xMin, boundaries.xMax]).range([0, innerW]);
      const y = d3.scaleLinear().domain([boundaries.yMin, boundaries.yMax]).range([innerH, 0]);

      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(x).ticks(7));
      g.append('g').call(d3.axisLeft(y).ticks(5));

      // Boundary contours via marching-squares on the 0/1 grid.
      const colors: Record<Estimator, string> = {
        GNB: paletteNB.pos,
        LDA: paletteNB.highlight,
        QDA: paletteNB.ok,
      };
      (Object.keys(boundaries.result) as Estimator[]).forEach((est) => {
        if (!enabled[est]) return;
        const grid = boundaries.result[est];
        // Build a flat values array for d3.contours.
        const values: number[] = new Array(GRID * GRID);
        for (let i = 0; i < GRID; i++) {
          for (let j = 0; j < GRID; j++) values[j * GRID + i] = grid[i][j];
        }
        const contours = d3.contours().size([GRID, GRID]).thresholds([0.5])(values);
        const projector = d3
          .geoPath()
          .projection(
            d3.geoTransform({
              point(this: { stream: { point: (x: number, y: number) => void } }, gx, gy) {
                const sx = x(boundaries.xMin + gx * boundaries.dx);
                const sy = y(boundaries.yMin + gy * boundaries.dy);
                this.stream.point(sx, sy);
              },
            }),
          );
        g.append('g')
          .attr('class', `boundary-${est}`)
          .selectAll('path')
          .data(contours)
          .join('path')
          .attr('d', projector as unknown as (d: d3.ContourMultiPolygon) => string)
          .attr('fill', 'none')
          .attr('stroke-width', 2)
          .attr('opacity', 0.85)
          .style('stroke', colors[est]);
      });

      // Scatter points.
      const pts = g.append('g').attr('class', 'points');
      for (let i = 0; i < data.y.length; i++) {
        pts
          .append('circle')
          .attr('cx', x(data.X[i * 2]))
          .attr('cy', y(data.X[i * 2 + 1]))
          .attr('r', SCATTER_R)
          .attr('fill-opacity', 0.55)
          .style('fill', data.y[i] === 0 ? paletteNB.neg : paletteNB.pos);
      }

      // Legend.
      const legend = g.append('g').attr('transform', `translate(${innerW - 90}, 8)`);
      (Object.keys(colors) as Estimator[]).forEach((est, i) => {
        if (!enabled[est]) return;
        legend
          .append('rect')
          .attr('x', 0)
          .attr('y', i * 16 - 9)
          .attr('width', 14)
          .attr('height', 3)
          .style('fill', colors[est]);
        legend
          .append('text')
          .attr('x', 18)
          .attr('y', i * 16)
          .attr('font-size', 11)
          .style('fill', 'currentColor')
          .text(est);
      });
    },
    [data, boundaries, enabled, W, H, innerW, innerH],
  );

  return (
    <div ref={containerRef} className="not-prose my-6">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-2 text-sm">
        <label className="flex flex-col">
          <span>Class-0 anisotropy = {anisotropy0.toFixed(2)}</span>
          <input
            type="range"
            min={0.4}
            max={2.5}
            step={0.05}
            value={anisotropy0}
            onChange={(e) => setAnisotropy0(Number(e.target.value))}
            aria-label="Class 0 anisotropy"
          />
        </label>
        <label className="flex flex-col">
          <span>Class-1 anisotropy = {anisotropy1.toFixed(2)}</span>
          <input
            type="range"
            min={0.4}
            max={2.5}
            step={0.05}
            value={anisotropy1}
            onChange={(e) => setAnisotropy1(Number(e.target.value))}
            aria-label="Class 1 anisotropy"
          />
        </label>
        <label className="flex flex-col">
          <span>Rotation = {rotation.toFixed(2)} rad</span>
          <input
            type="range"
            min={-1.2}
            max={1.2}
            step={0.02}
            value={rotation}
            onChange={(e) => setRotation(Number(e.target.value))}
            aria-label="Class rotation"
          />
        </label>
      </div>
      <div className="flex flex-wrap items-center gap-3 mb-2 text-sm">
        {(['GNB', 'LDA', 'QDA'] as Estimator[]).map((est) => (
          <label key={est} className="flex items-center gap-1">
            <input
              type="checkbox"
              checked={enabled[est]}
              onChange={(e) => setEnabled({ ...enabled, [est]: e.target.checked })}
              aria-label={`Show ${est} boundary`}
            />
            <span>{est}</span>
          </label>
        ))}
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="ml-auto rounded border px-2 py-1 text-xs"
          aria-label="Resample data"
        >
          Resample (seed {seed})
        </button>
      </div>
      <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label="GNB, LDA, QDA boundaries on 2D Gaussian data"/>
      <div className="text-xs opacity-75 mt-1 text-center">
        GNB assumes a <em>diagonal</em> class-covariance and ignores within-class correlations;
        LDA forces a single covariance across classes; QDA fits a full covariance per class.
        Watch how the three boundaries diverge as you change the anisotropy and rotation.
      </div>
    </div>
  );
}
