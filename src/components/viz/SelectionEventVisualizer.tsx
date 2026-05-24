import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32 } from './shared/normalizing-flows';
import { gaussianFrom } from './shared/generalization-bounds';

// ---------------------------------------------------------------------------
// Toy DGP: n = 50, p = 8, signals at coordinates 0/1/2 with β★ = 1.5.
// Smaller than the notebook (n=100, p=20) to keep the 2-D tessellation
// compute interactive (slider response ≤ 200 ms on a 30x30 grid).
// ---------------------------------------------------------------------------

const N = 50;
const P = 8;
const SUPPORT = [0, 1, 2] as const;
const BETA_STAR_VAL = 1.5;
const SIGMA = 1;
const SEED = 7;

type Toy = {
  X: Float64Array;       // row-major n×p
  y: Float64Array;       // length n
  colNorm2: Float64Array; // length p, ||X_j||²
  beta: Float64Array;    // β★
};

function makeToy(): Toy {
  const rng = mulberry32(SEED);
  const gauss = gaussianFrom(rng);
  const X = new Float64Array(N * P);
  for (let i = 0; i < N * P; i++) X[i] = gauss();
  const beta = new Float64Array(P);
  for (const j of SUPPORT) beta[j] = BETA_STAR_VAL;
  const y = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let mu = 0;
    for (let j = 0; j < P; j++) mu += X[i * P + j] * beta[j];
    y[i] = mu + SIGMA * gauss();
  }
  const colNorm2 = new Float64Array(P);
  for (let j = 0; j < P; j++) {
    let s = 0;
    for (let i = 0; i < N; i++) s += X[i * P + j] * X[i * P + j];
    colNorm2[j] = s;
  }
  return { X, y, colNorm2, beta };
}

// Cyclic coordinate-descent lasso for math objective 1/2 ||y - Xβ||² + λ||β||₁.
// `beta` and `r` are caller-owned scratch buffers reset on entry — hoisting
// them out of the GRID_N² fit loop is what keeps slider-commit interactive.
function lassoFit(
  toy: Toy,
  y: Float64Array,
  lambda: number,
  beta: Float64Array,
  r: Float64Array,
): { S: number[]; z: number[] } {
  const { X, colNorm2 } = toy;
  beta.fill(0);
  r.set(y);

  const MAX_ITER = 80;
  const TOL = 1e-4;
  for (let iter = 0; iter < MAX_ITER; iter++) {
    let maxDelta = 0;
    for (let j = 0; j < P; j++) {
      const bjOld = beta[j];
      // z_j = X_j^T r + ||X_j||² β_j_old   (since r is computed with current β)
      let zj = colNorm2[j] * bjOld;
      for (let i = 0; i < N; i++) zj += X[i * P + j] * r[i];
      // soft-threshold
      let bjNew = 0;
      if (zj > lambda) bjNew = (zj - lambda) / colNorm2[j];
      else if (zj < -lambda) bjNew = (zj + lambda) / colNorm2[j];
      // update residual
      const delta = bjNew - bjOld;
      if (delta !== 0) {
        for (let i = 0; i < N; i++) r[i] -= X[i * P + j] * delta;
        beta[j] = bjNew;
        const ad = Math.abs(delta);
        if (ad > maxDelta) maxDelta = ad;
      }
    }
    if (maxDelta < TOL) break;
  }
  const S: number[] = [];
  const z: number[] = [];
  for (let j = 0; j < P; j++) {
    if (beta[j] !== 0) {
      S.push(j);
      z.push(beta[j] > 0 ? 1 : -1);
    }
  }
  return { S, z };
}

// Stable hash of (S, z) → string key.
function outcomeKey(o: { S: number[]; z: number[] }): string {
  return o.S.map((j, i) => `${j}${o.z[i] > 0 ? '+' : '-'}`).join(',') || '∅';
}

// ---------------------------------------------------------------------------
// 2-D tessellation: perturb y_0 along v1 (signal direction X_0 normalized) and
// v2 (null direction X_{nullIdx} projected orthogonal to span(X_S^★) then
// normalized). Grid resolution GRID_N × GRID_N.
// ---------------------------------------------------------------------------

const GRID_N = 28;
const A_MAX = 18;
const B_MAX = 18;

type Grid = {
  outcomes: string[];          // GRID_N² keys
  unique: string[];            // distinct keys in render order
  centerKey: string;           // outcome at (0, 0)
};

function buildGrid(toy: Toy, lambda: number, signalIdx: number, nullIdx: number): Grid {
  const { X } = toy;
  // v1 = X_{signalIdx} normalized
  const v1 = new Float64Array(N);
  let v1n = 0;
  for (let i = 0; i < N; i++) {
    v1[i] = X[i * P + signalIdx];
    v1n += v1[i] * v1[i];
  }
  v1n = Math.sqrt(v1n);
  for (let i = 0; i < N; i++) v1[i] /= v1n;
  // v2 = X_{nullIdx} − (X_{nullIdx} · v1) v1, normalized
  const v2 = new Float64Array(N);
  let dot = 0;
  for (let i = 0; i < N; i++) dot += X[i * P + nullIdx] * v1[i];
  let v2n = 0;
  for (let i = 0; i < N; i++) {
    v2[i] = X[i * P + nullIdx] - dot * v1[i];
    v2n += v2[i] * v2[i];
  }
  v2n = Math.sqrt(v2n);
  for (let i = 0; i < N; i++) v2[i] /= v2n;

  const perturbed = new Float64Array(N);
  const betaScratch = new Float64Array(P);
  const rScratch = new Float64Array(N);
  const outcomes: string[] = new Array(GRID_N * GRID_N);
  const uniqueOrdered: string[] = [];
  const seen = new Set<string>();
  for (let gi = 0; gi < GRID_N; gi++) {
    const a = -A_MAX + (2 * A_MAX * gi) / (GRID_N - 1);
    for (let gj = 0; gj < GRID_N; gj++) {
      const b = -B_MAX + (2 * B_MAX * gj) / (GRID_N - 1);
      for (let i = 0; i < N; i++) perturbed[i] = toy.y[i] + a * v1[i] + b * v2[i];
      const out = lassoFit(toy, perturbed, lambda, betaScratch, rScratch);
      const key = outcomeKey(out);
      outcomes[gi * GRID_N + gj] = key;
      if (!seen.has(key)) {
        seen.add(key);
        uniqueOrdered.push(key);
      }
    }
  }
  const centerIdx = Math.floor(GRID_N / 2) * GRID_N + Math.floor(GRID_N / 2);
  return { outcomes, unique: uniqueOrdered, centerKey: outcomes[centerIdx] };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const SIGNAL_OPTIONS = SUPPORT.map((j) => ({ value: j, label: `X_${j} (signal)` }));
const NULL_OPTIONS = Array.from({ length: P - SUPPORT.length }, (_, k) => SUPPORT.length + k).map((j) => ({
  value: j,
  label: `X_${j} (null)`,
}));

export default function SelectionEventVisualizer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [lambdaDisplay, setLambdaDisplay] = useState(8);
  const [lambdaCommitted, setLambdaCommitted] = useState(8);
  const [signalIdx, setSignalIdx] = useState(0);
  const [nullIdx, setNullIdx] = useState(5);

  const toy = useMemo(makeToy, []);
  const grid = useMemo(
    () => buildGrid(toy, lambdaCommitted, signalIdx, nullIdx),
    [toy, lambdaCommitted, signalIdx, nullIdx],
  );

  const palette = useMemo(() => {
    const scale = d3.scaleOrdinal<string, string>(d3.schemeTableau10);
    return scale.domain(grid.unique);
  }, [grid.unique]);

  const W = Math.max(280, Math.min(containerWidth || 720, 720));
  const H = Math.round(W * 0.72);
  const M = { t: 28, r: 18, b: 44, l: 50 };
  const iw = W - M.l - M.r;
  const ih = H - M.t - M.b;

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (iw <= 0 || ih <= 0) return;

      const xScale = d3.scaleLinear().domain([-A_MAX, A_MAX]).range([0, iw]);
      const yScale = d3.scaleLinear().domain([-B_MAX, B_MAX]).range([ih, 0]);
      const cellW = iw / GRID_N;
      const cellH = ih / GRID_N;
      const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);

      for (let gi = 0; gi < GRID_N; gi++) {
        for (let gj = 0; gj < GRID_N; gj++) {
          const key = grid.outcomes[gi * GRID_N + gj];
          const a = -A_MAX + (2 * A_MAX * gi) / (GRID_N - 1);
          const b = -B_MAX + (2 * B_MAX * gj) / (GRID_N - 1);
          g.append('rect')
            .attr('x', xScale(a) - cellW / 2)
            .attr('y', yScale(b) - cellH / 2)
            .attr('width', cellW)
            .attr('height', cellH)
            .attr('fill', palette(key))
            .attr('opacity', 0.78);
        }
      }
      // realized cell marker (origin)
      g.append('circle')
        .attr('cx', xScale(0))
        .attr('cy', yScale(0))
        .attr('r', 5)
        .style('fill', 'var(--color-bg)')
        .style('stroke', 'var(--color-text)')
        .attr('stroke-width', 1.8);

      // axes
      const xAxis = d3.axisBottom(xScale).ticks(5);
      const yAxis = d3.axisLeft(yScale).ticks(5);
      g.append('g').attr('transform', `translate(0,${ih})`).call(xAxis).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(yAxis).selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, line').style('stroke', 'var(--color-border)');
      g.append('text')
        .attr('x', iw / 2)
        .attr('y', ih + 34)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '0.78rem')
        .text(`Perturbation along v₁ ∝ X_${signalIdx} (signal direction)`);
      g.append('text')
        .attr('transform', `translate(-36,${ih / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-sans)')
        .style('font-size', '0.78rem')
        .text(`Perturbation along v₂ ⟂ v₁, ∝ X_${nullIdx} residual`);
    },
    [iw, ih, grid, palette, signalIdx, nullIdx],
  );

  return (
    <div ref={containerRef} className="not-prose my-6 w-full">
      <div className="flex flex-col gap-4 sm:flex-row sm:flex-wrap sm:items-end">
        <label className="flex flex-col text-sm" style={{ color: 'var(--color-text-secondary)' }}>
          <span>
            math-λ ={' '}
            <span className="tabular-nums" style={{ color: 'var(--color-text)' }}>
              {lambdaDisplay.toFixed(2)}
            </span>
          </span>
          <input
            type="range"
            min={1}
            max={30}
            step={0.5}
            value={lambdaDisplay}
            onChange={(e) => setLambdaDisplay(parseFloat(e.target.value))}
            onMouseUp={() => setLambdaCommitted(lambdaDisplay)}
            onTouchEnd={() => setLambdaCommitted(lambdaDisplay)}
            onKeyUp={() => setLambdaCommitted(lambdaDisplay)}
            aria-label="Lasso regularization parameter (math convention)"
            className="w-56"
          />
        </label>
        <label className="flex flex-col text-sm" style={{ color: 'var(--color-text-secondary)' }}>
          <span>Horizontal direction</span>
          <select
            value={signalIdx}
            onChange={(e) => setSignalIdx(parseInt(e.target.value, 10))}
            className="rounded border px-2 py-1"
            style={{ background: 'var(--color-surface)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}
          >
            {SIGNAL_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col text-sm" style={{ color: 'var(--color-text-secondary)' }}>
          <span>Vertical direction</span>
          <select
            value={nullIdx}
            onChange={(e) => setNullIdx(parseInt(e.target.value, 10))}
            className="rounded border px-2 py-1"
            style={{ background: 'var(--color-surface)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}
          >
            {NULL_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <svg ref={svgRef} width={W} height={H} className="mt-2" role="img" aria-label="2-D selection-outcome tessellation" />
      <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
        <span style={{ color: 'var(--color-text)' }}>
          {grid.unique.length} distinct outcomes · realized cell at origin: <span className="font-mono">{grid.centerKey}</span>
        </span>
        {grid.unique.slice(0, 8).map((key) => (
          <span key={key} className="inline-flex items-center gap-1">
            <span
              aria-hidden
              style={{
                display: 'inline-block',
                width: 10,
                height: 10,
                background: palette(key),
                borderRadius: 2,
              }}
            />
            <span className="font-mono">{key}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
