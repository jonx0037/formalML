import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32, gaussianSampler } from './shared/nonparametric-ml';
import {
  DEPTH_COLORS,
  projectionDepth,
  tukeyDepth2D,
  type Point2D,
} from '../../data/statistical-depth';

// =============================================================================
// RuntimeScalingExplorer — anchors figure 06 (runtime vs dimension).
//
// Benchmarks two depth-evaluation strategies as d grows:
//   - projection (any d, cost O(K · n) per query)
//   - exact halfspace, restricted to d ∈ {2, 3} per Aloupis 2006 (NP-hard at
//     d ≥ 4)
//
// The exact-3D path uses an inline pair-enumeration over all $\binom{n}{2}$
// hyperplanes through the query — kept inline per the topic-specific notes,
// hard-capped at $d \le 3$, $n \le 200$. The shaded $d \ge 4$ region marks
// the NP-hard regime where exact computation becomes impractical.
//
// Static fallback: 06_runtime_vs_d.png
// =============================================================================

const HEIGHT = 460;
const SM_BREAKPOINT = 640;
const PROJ_DIMS = [2, 3, 5, 10, 20, 30, 50] as const;
const EXACT_DIMS = [2, 3] as const;

interface BenchPoint {
  d: number;
  ms: number;
}

// -----------------------------------------------------------------------------
// Inline exact_halfspace_depth_3d — hard-capped at $d = 3$, $n \le 200$.
// Enumerates $\binom{n}{2}$ hyperplanes through the query and the candidate
// support pairs $(X_i, X_j)$.
// -----------------------------------------------------------------------------

function exactHalfspaceDepth3D(query: number[], X: number[][]): number {
  const n = X.length;
  if (n === 0 || X[0].length !== 3) return 0;
  // Pre-shift: diffs[i] = X[i] − query.
  const dx = new Float64Array(n);
  const dy = new Float64Array(n);
  const dz = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    dx[i] = X[i][0] - query[0];
    dy[i] = X[i][1] - query[1];
    dz[i] = X[i][2] - query[2];
  }

  let minCount = n;
  for (let i = 0; i < n - 1; i++) {
    for (let j = i + 1; j < n; j++) {
      // Normal = diffs[i] × diffs[j]
      const nx = dy[i] * dz[j] - dz[i] * dy[j];
      const ny = dz[i] * dx[j] - dx[i] * dz[j];
      const nz = dx[i] * dy[j] - dy[i] * dx[j];
      const nrm = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (nrm < 1e-12) continue;
      // Project all diffs onto the unit normal and count closed-halfspace
      // memberships in each direction. Boundary points (|p| < tol) lie on
      // both closed halfspaces, so they are counted in both `cPos` and
      // `cNeg` — the smaller of the two counts is the Tukey-depth count
      // for this candidate hyperplane.
      let cPos = 0;
      let cNeg = 0;
      const inv = 1 / nrm;
      for (let k = 0; k < n; k++) {
        const p = (dx[k] * nx + dy[k] * ny + dz[k] * nz) * inv;
        if (p >= -1e-10) cPos++;
        if (p <= 1e-10) cNeg++;
      }
      const c = Math.min(cPos, cNeg);
      if (c < minCount) minCount = c;
    }
  }
  return minCount / n;
}

// -----------------------------------------------------------------------------
// d-dimensional Gaussian sampler for benchmark inputs.
// -----------------------------------------------------------------------------

function sampleGaussianND(n: number, d: number, seed: number): number[][] {
  const rng = mulberry32(seed);
  const gauss = gaussianSampler(rng);
  const out: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(d);
    for (let dd = 0; dd < d; dd++) row[dd] = gauss();
    out[i] = row;
  }
  return out;
}

// -----------------------------------------------------------------------------

function timeRuns(fn: () => void, reps: number): number {
  // Warm-up
  fn();
  const t0 = performance.now();
  for (let i = 0; i < reps; i++) fn();
  return (performance.now() - t0) / reps;
}

export default function RuntimeScalingExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [K, setK] = useState(200);
  const [n, setN] = useState(50);

  const { proj, exact, error } = useMemo(() => {
    if (n > 200) return { proj: [], exact: [], error: 'n capped at 200 for exact halfspace.' };
    const projTimes: BenchPoint[] = [];
    const exactTimes: BenchPoint[] = [];

    for (const d of PROJ_DIMS) {
      const X = sampleGaussianND(n, d, d * 7 + 31);
      const q = new Array<number>(d).fill(0);
      const ms = timeRuns(() => {
        projectionDepth(q, X, K, d * 11);
      }, 5);
      projTimes.push({ d, ms });
    }

    for (const d of EXACT_DIMS) {
      const X = sampleGaussianND(n, d, d * 7 + 31);
      let ms: number;
      if (d === 2) {
        const X2 = X.map((r) => [r[0], r[1]] as Point2D);
        const q: Point2D = [0, 0];
        ms = timeRuns(() => { tukeyDepth2D(q, X2); }, 8);
      } else {
        const q = [0, 0, 0];
        ms = timeRuns(() => { exactHalfspaceDepth3D(q, X); }, 3);
      }
      exactTimes.push({ d, ms });
    }

    return { proj: projTimes, exact: exactTimes, error: null };
  }, [K, n]);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = containerWidth;

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      if (proj.length === 0 && exact.length === 0) return;

      const margin = { top: 30, right: 16, bottom: 46, left: 56 };
      const innerW = w - margin.left - margin.right;
      const innerH = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      // Log-log scales
      const allMs = [...proj.map((p) => p.ms), ...exact.map((p) => p.ms)];
      const yMin = Math.max(d3.min(allMs) ?? 0.001, 0.001);
      const yMax = (d3.max(allMs) ?? 100) * 1.4;
      const xScale = d3.scaleLog().domain([1.8, 60]).range([0, innerW]);
      const yScale = d3.scaleLog().domain([yMin, yMax]).range([innerH, 0]);

      // NP-hard region shading at d ≥ 4
      g.append('rect')
        .attr('x', xScale(4)).attr('y', 0)
        .attr('width', innerW - xScale(4))
        .attr('height', innerH)
        .style('fill', '#DC2626')
        .style('opacity', 0.05);
      g.append('text')
        .attr('x', xScale(15)).attr('y', 24)
        .style('fill', '#B91C1C')
        .style('font-size', '11px')
        .style('text-anchor', 'middle')
        .style('opacity', 0.85)
        .text('exact: NP-hard (Aloupis 2006)');

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(6, '~g'))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(6, '~g'))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

      g.append('text')
        .attr('x', innerW / 2).attr('y', innerH + 36)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '11px')
        .text('dimension d');
      g.append('text')
        .attr('x', -42).attr('y', innerH / 2)
        .style('fill', 'var(--color-text-secondary)')
        .style('text-anchor', 'middle').style('font-size', '11px')
        .attr('transform', `rotate(-90,-42,${innerH / 2})`)
        .text('time per query (ms)');

      // Projection-depth line
      const projLine = d3.line<BenchPoint>()
        .x((p) => xScale(p.d))
        .y((p) => yScale(Math.max(p.ms, yMin)));
      g.append('path')
        .datum(proj)
        .attr('d', projLine)
        .style('fill', 'none')
        .style('stroke', DEPTH_COLORS.projection)
        .style('stroke-width', 2.0);
      g.append('g')
        .selectAll('circle')
        .data(proj)
        .enter().append('circle')
        .attr('cx', (p) => xScale(p.d))
        .attr('cy', (p) => yScale(Math.max(p.ms, yMin)))
        .attr('r', 4)
        .style('fill', DEPTH_COLORS.projection)
        .style('stroke', 'var(--color-bg)')
        .style('stroke-width', 1.2);

      // Exact-halfspace line
      const exactLine = d3.line<BenchPoint>()
        .x((p) => xScale(p.d))
        .y((p) => yScale(Math.max(p.ms, yMin)));
      g.append('path')
        .datum(exact)
        .attr('d', exactLine)
        .style('fill', 'none')
        .style('stroke', DEPTH_COLORS.tukey)
        .style('stroke-width', 2.0);
      g.append('g')
        .selectAll('rect')
        .data(exact)
        .enter().append('rect')
        .attr('x', (p) => xScale(p.d) - 4)
        .attr('y', (p) => yScale(Math.max(p.ms, yMin)) - 4)
        .attr('width', 8).attr('height', 8)
        .style('fill', DEPTH_COLORS.tukey)
        .style('stroke', 'var(--color-bg)')
        .style('stroke-width', 1.2);

      // Legend
      const lg = svg.append('g').attr('transform', `translate(${margin.left + 16},${margin.top + 8})`);
      lg.append('rect')
        .attr('x', 0).attr('y', 0)
        .attr('width', 168).attr('height', 42)
        .style('fill', 'var(--color-bg)')
        .style('opacity', 0.9)
        .style('stroke', 'var(--color-border)');
      lg.append('line')
        .attr('x1', 8).attr('x2', 28).attr('y1', 14).attr('y2', 14)
        .style('stroke', DEPTH_COLORS.projection).style('stroke-width', 2);
      lg.append('text')
        .attr('x', 34).attr('y', 17).style('font-size', '10px')
        .style('fill', 'var(--color-text)')
        .text(`projection (K=${K})`);
      lg.append('line')
        .attr('x1', 8).attr('x2', 28).attr('y1', 32).attr('y2', 32)
        .style('stroke', DEPTH_COLORS.tukey).style('stroke-width', 2);
      lg.append('text')
        .attr('x', 34).attr('y', 35).style('font-size', '10px')
        .style('fill', 'var(--color-text)')
        .text('exact halfspace (d ≤ 3)');

      // Title
      svg.append('text')
        .attr('x', w / 2).attr('y', 16)
        .style('fill', 'var(--color-text)')
        .style('text-anchor', 'middle').style('font-size', '12px')
        .style('font-weight', '600')
        .text(`Runtime vs dimension  (n = ${n}, log-log scale)`);
    },
    [w, proj, exact, K, n],
  );

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] p-4"
    >
      <div
        className="flex flex-wrap items-center gap-4 mb-3"
        style={{ flexDirection: isMobile ? 'column' : 'row', alignItems: isMobile ? 'stretch' : 'center' }}
      >
        <div className="flex items-center gap-2">
          <label htmlFor="rs-k" className="text-sm text-[var(--color-text-secondary)]">K (directions):</label>
          <input
            id="rs-k"
            type="range"
            min={50}
            max={400}
            step={25}
            value={K}
            onChange={(e) => setK(parseInt(e.target.value, 10))}
            className="w-32"
          />
          <span className="text-xs font-mono text-[var(--color-text-secondary)]">{K}</span>
        </div>
        <div className="flex items-center gap-2">
          <label htmlFor="rs-n" className="text-sm text-[var(--color-text-secondary)]">n (sample):</label>
          <input
            id="rs-n"
            type="range"
            min={20}
            max={200}
            step={10}
            value={n}
            onChange={(e) => setN(parseInt(e.target.value, 10))}
            className="w-32"
          />
          <span className="text-xs font-mono text-[var(--color-text-secondary)]">{n}</span>
        </div>
        <p className="text-xs text-[var(--color-text-secondary)] ml-auto max-w-md">
          Per-query timing in milliseconds. Exact halfspace runs only at d ∈ {'{2, 3}'} per the §4.4 NP-hardness result.
        </p>
      </div>
      {error ? (
        <p className="text-sm text-red-600">{error}</p>
      ) : (
        <svg ref={svgRef} width={w} height={HEIGHT} role="img" aria-label="Runtime vs dimension benchmark" />
      )}
      <p className="text-xs text-[var(--color-text-secondary)] mt-2">
        Projection depth&apos;s roughly flat curve in d (cost is O(K · n), independent of dimension after the projection step) is the punchline of §4.5: in practice, the only depth that scales is the approximate one. Increasing K trades accuracy for cost — the random-direction approximation has Op(√(log K / K)) error.
      </p>
    </div>
  );
}
