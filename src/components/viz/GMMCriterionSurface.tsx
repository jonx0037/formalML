import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  RUNNING_EXAMPLE_A,
  RUNNING_EXAMPLE_THETA0,
  affineCriterionSurface,
  gmmCriterion,
  gmmEstimateAffine,
  identityMat,
  matInverse,
  matVec,
  mulberry32,
  gaussianFrom,
  sampleSensorObservations,
  type Mat,
  type Vec,
} from './shared/generalized-method-of-moments';

// -----------------------------------------------------------------------------
// Viz 3.1 — GMM criterion surface (central viz of the topic).
//
// Contour panel for J_n(θ, W) over a 2D (θ_1, θ_2) grid using the running
// example (4 sensors, 2 parameters, A ∈ R^{4×2}). The reader controls:
//   - n (sample size, drives noise in X̄)
//   - σ_3 (sets a per-sensor heteroskedasticity for sensor 3 only)
//   - weighting preset: I, diag(1/σ²), efficient Ω⁻¹, "rank-deficient" demo
//
// Live readout of θ̂_W, J_min = J_n(θ̂_W, W), and ‖θ̂_W − θ₀‖.
//
// CLAUDE.md hot-path notes: contour computation is closed-form (50×50 grid =
// 2500 evaluations of a 2D quadratic, ~50k flops). Cheap enough to refresh on
// slider onChange. No commit-on-release.
// -----------------------------------------------------------------------------

const N_OPTIONS = [50, 100, 200, 500, 1000];
const GRID_RES = 50;
const PANEL_H = 360;
const MARGIN = { top: 24, right: 18, bottom: 50, left: 60 };

type WeightingPreset = 'identity' | 'diagonal' | 'efficient' | 'rank1';

const PRESET_LABEL: Record<WeightingPreset, string> = {
  identity: 'W = I',
  diagonal: 'W = diag(1/σ²)',
  efficient: 'W = Ω⁻¹',
  rank1: 'W = e₁e₁ᵀ (rank-1)',
};

function makeWeighting(preset: WeightingPreset, sigma: Vec): Mat {
  const L = sigma.length;
  if (preset === 'identity') return identityMat(L);
  if (preset === 'diagonal') {
    const W: Mat = Array.from({ length: L }, (_, i) =>
      Array.from({ length: L }, (_, j) => (i === j ? 1 / (sigma[i] * sigma[i]) : 0)),
    );
    return W;
  }
  if (preset === 'efficient') {
    // Population Ω = diag(σ²). Efficient W = Ω⁻¹.
    const W: Mat = Array.from({ length: L }, (_, i) =>
      Array.from({ length: L }, (_, j) => (i === j ? 1 / (sigma[i] * sigma[i]) : 0)),
    );
    return W;
  }
  // rank1: project onto e₁ — collapses identification. Add a tiny diagonal
  // ridge so matInverse doesn't choke on the resulting (Aᵀ W A) being singular.
  const W: Mat = Array.from({ length: L }, () => new Array(L).fill(0));
  W[0][0] = 1;
  for (let i = 0; i < L; i++) W[i][i] += 1e-6;
  return W;
}

interface SurfaceState {
  Xbar: Vec;
  W: Mat;
  thetaHat: Vec;
  Jmin: number;
  distanceToTruth: number;
  surface: ReturnType<typeof affineCriterionSurface>;
}

function computeSurface(n: number, sigma3: number, preset: WeightingPreset, seed: number): SurfaceState {
  const sigma: Vec = [1, 1, sigma3, 1];
  const A = RUNNING_EXAMPLE_A;
  const theta0 = RUNNING_EXAMPLE_THETA0;
  // Sample X̄ once from running-example DGP.
  const rng = mulberry32(seed);
  const X = sampleSensorObservations(rng, theta0, A, n, sigma);
  const Xbar: Vec = [0, 0, 0, 0];
  for (let i = 0; i < n; i++) for (let j = 0; j < 4; j++) Xbar[j] += X[i][j];
  for (let j = 0; j < 4; j++) Xbar[j] /= n;

  const W = makeWeighting(preset, sigma);
  let thetaHat: Vec;
  try {
    thetaHat = gmmEstimateAffine(A, Xbar, W);
  } catch {
    // Rank-deficient case: return the truth as a sentinel; surface will be a
    // flat ridge and the user will see the criterion is degenerate.
    thetaHat = [...theta0];
  }
  const Jmin = gmmCriterion(thetaHat, A, Xbar, W, n);
  const dist = Math.hypot(thetaHat[0] - theta0[0], thetaHat[1] - theta0[1]);
  const surface = affineCriterionSurface(A, Xbar, W, [0.0, 2.0], [0.0, 2.0], GRID_RES, n);
  return { Xbar, W, thetaHat, Jmin, distanceToTruth: dist, surface };
}

export default function GMMCriterionSurface() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nIdx, setNIdx] = useState(2); // n = 200
  const [sigma3, setSigma3] = useState(1.0);
  const [preset, setPreset] = useState<WeightingPreset>('identity');
  const [seed, setSeed] = useState(42);

  const n = N_OPTIONS[nIdx];
  const state = useMemo(() => computeSurface(n, sigma3, preset, seed), [n, sigma3, preset, seed]);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLinear().domain([0.0, 2.0]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0.0, 2.0]).range([innerH, 0]);

      // Flatten the J grid into row-major for d3.contours.
      const flat: number[] = [];
      for (let i = 0; i < GRID_RES; i++) for (let j = 0; j < GRID_RES; j++) flat.push(state.surface.J[i][j]);
      const Jmax = d3.max(flat) ?? 1;
      const Jmin = d3.min(flat) ?? 0;
      // Choose 8 contour levels — geometric near the minimum so the elliptical
      // basin is visible even when Jmin is well below Jmax.
      const levels: number[] = [];
      const span = Jmax - Jmin;
      for (let k = 1; k <= 8; k++) levels.push(Jmin + Math.pow(k / 8, 2.0) * span);

      const contours = d3.contours().size([GRID_RES, GRID_RES]).thresholds(levels)(flat);
      const colorScale = d3.scaleSequential(d3.interpolateBlues).domain([Jmax, Jmin]);
      const cellW = innerW / (GRID_RES - 1);
      const cellH = innerH / (GRID_RES - 1);

      g.append('g')
        .attr('transform', `translate(0, ${innerH}) scale(${cellW}, ${-cellH})`)
        .selectAll('path')
        .data(contours)
        .enter()
        .append('path')
        .attr('d', d3.geoPath() as never)
        .style('fill', (d) => colorScale(d.value))
        .style('stroke', 'var(--color-border)')
        .style('stroke-width', 0.5 / Math.min(cellW, cellH))
        .style('opacity', 0.85);

      // Axes (re-add after contours so they're on top).
      g.append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');

      // θ₀ cross.
      const x0 = xScale(RUNNING_EXAMPLE_THETA0[0]);
      const y0 = yScale(RUNNING_EXAMPLE_THETA0[1]);
      g.append('line').attr('x1', x0 - 8).attr('x2', x0 + 8).attr('y1', y0).attr('y2', y0).style('stroke', 'var(--color-text)').style('stroke-width', 1.6);
      g.append('line').attr('x1', x0).attr('x2', x0).attr('y1', y0 - 8).attr('y2', y0 + 8).style('stroke', 'var(--color-text)').style('stroke-width', 1.6);
      g.append('text').attr('x', x0 + 10).attr('y', y0 - 6).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text('θ₀');

      // θ̂_W dot.
      const xHat = xScale(state.thetaHat[0]);
      const yHat = yScale(state.thetaHat[1]);
      g.append('circle').attr('cx', xHat).attr('cy', yHat).attr('r', 5).style('fill', 'var(--color-accent)').style('stroke', 'var(--color-bg)').style('stroke-width', 1.5);
      g.append('text').attr('x', xHat + 7).attr('y', yHat + 4).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-accent)').text('θ̂_W');

      // Axis labels.
      g.append('text').attr('x', innerW / 2).attr('y', innerH + 38).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('θ_1');
      g.append('text').attr('x', -innerH / 2).attr('y', -44).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('θ_2');
    },
    [state, w],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '160px' }}>
          <span>
            n = <strong>{n}</strong>
          </span>
          <input
            type="range"
            min={0}
            max={N_OPTIONS.length - 1}
            step={1}
            value={nIdx}
            aria-label="Sample size"
            onChange={(e) => setNIdx(Number(e.target.value))}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '160px' }}>
          <span>
            σ_3 = <strong>{sigma3.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min={0.2}
            max={3.0}
            step={0.05}
            value={sigma3}
            aria-label="Sensor 3 noise level"
            onChange={(e) => setSigma3(Number(e.target.value))}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>Weighting</span>
          <select
            value={preset}
            onChange={(e) => setPreset(e.target.value as WeightingPreset)}
            style={{ padding: '4px 8px', fontFamily: 'var(--font-mono)', fontSize: '12px' }}
            aria-label="Weighting matrix preset"
          >
            {(['identity', 'diagonal', 'efficient', 'rank1'] as WeightingPreset[]).map((p) => (
              <option key={p} value={p}>
                {PRESET_LABEL[p]}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          style={{
            padding: '6px 10px',
            fontFamily: 'var(--font-mono)',
            fontSize: '12px',
            border: '1px solid var(--color-border)',
            background: 'var(--color-bg)',
            color: 'var(--color-text)',
            borderRadius: '4px',
            cursor: 'pointer',
            alignSelf: 'flex-end',
          }}
        >
          new seed
        </button>
      </div>
      <svg ref={svgRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} aria-label="GMM criterion surface contours" />
      <div
        style={{
          marginTop: 8,
          padding: '8px 12px',
          fontFamily: 'var(--font-mono)',
          fontSize: '12px',
          color: 'var(--color-text-secondary)',
          border: '1px solid var(--color-border)',
          borderRadius: '4px',
        }}
      >
        θ̂_W = ({state.thetaHat[0].toFixed(3)}, {state.thetaHat[1].toFixed(3)}) &nbsp;·&nbsp;
        J_min = {state.Jmin.toFixed(2)} &nbsp;·&nbsp;
        ‖θ̂_W − θ₀‖ = {state.distanceToTruth.toFixed(3)}
        {preset === 'rank1' && (
          <span style={{ color: 'var(--color-accent)', marginLeft: 8 }}>
            ⚠ rank-1 W collapses identification — only one direction of θ is pinned.
          </span>
        )}
      </div>
    </div>
  );
}
