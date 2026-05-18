// =============================================================================
// LeapfrogReversibilityDemo.tsx
//
// §6.2 Reversibility. Single panel showing that the generalized leapfrog map
// is a momentum-negation involution: N ∘ Φ_ε^L ∘ N ∘ Φ_ε^L = id (Theorem 6.2).
//
// Implementation:
//
//   1. Forward trajectory: (θ_0, p_0) →^{Φ_ε^L} (θ_L, p_L).
//   2. Backward trajectory: (θ_L, -p_L) →^{Φ_ε^L} (θ_{2L}, p_{2L}).
//
// By the theorem, (θ_{2L}, p_{2L}) = (θ_0, -p_0) to fixed-point tolerance.
//
// The component overlays both trajectories on the banana log-density contour;
// the forward path goes olive, the backward path goes pink-dashed, and the
// reader can verify visually that the second trajectory retraces the first
// in reverse. The numerical round-trip error is reported below.
//
// Controls: ε ∈ [0.02, 0.40] slider, L ∈ [10, 80] step slider. Computation is
// cheap (a few hundred GL steps) so we don't bother with display-vs-committed.
//
// Computation: in-browser via shared/riemann-hmc.ts.
// Static fallback: /images/topics/riemann-manifold-hmc/06_leapfrog_reversibility.png
// =============================================================================

import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bananaLogDensity,
  bananaMetric,
  generalizedLeapfrogStep,
  paletteRMHMC,
} from './shared/riemann-hmc';

const SM_BREAKPOINT = 640;
const DEFAULT_EPS = 0.15;
const DEFAULT_L = 40;
const X_MIN = -3.5;
const X_MAX = 3.5;
const Y_MIN = -6;
const Y_MAX = 4;
const BANANA = { a: 1, b: 1 };
const THETA0: [number, number] = [-2, -3];

function seedMomentum(): [number, number] {
  const G = bananaMetric([THETA0[0], THETA0[1]], BANANA);
  const l00 = Math.sqrt(G[0][0]);
  const l10 = G[1][0] / l00;
  const l11 = Math.sqrt(Math.max(G[1][1] - l10 * l10, 1e-12));
  return [l00 * 1.0, l10 * 1.0 + l11 * 0.5];
}

interface ReversibilityResult {
  forward: [number, number][];
  backward: [number, number][];
  roundTripError: number;
  divergent: boolean;
}

function runReversibility(eps: number, L: number): ReversibilityResult {
  let theta: number[] = [THETA0[0], THETA0[1]];
  let mom: number[] = [...seedMomentum()];
  const forward: [number, number][] = [[theta[0], theta[1]]];
  let divergent = false;
  for (let i = 0; i < L; i++) {
    const r = generalizedLeapfrogStep(theta, mom, eps, BANANA);
    if (!r.converged) {
      divergent = true;
      break;
    }
    theta = r.theta;
    mom = r.mom;
    forward.push([theta[0], theta[1]]);
  }
  if (divergent) {
    return { forward, backward: [], roundTripError: Infinity, divergent: true };
  }
  // Negate momentum
  const thetaL = theta.slice();
  const momL = [-mom[0], -mom[1]];
  let theta2 = thetaL.slice();
  let mom2 = momL.slice();
  const backward: [number, number][] = [[theta2[0], theta2[1]]];
  for (let i = 0; i < L; i++) {
    const r = generalizedLeapfrogStep(theta2, mom2, eps, BANANA);
    if (!r.converged) {
      divergent = true;
      break;
    }
    theta2 = r.theta;
    mom2 = r.mom;
    backward.push([theta2[0], theta2[1]]);
  }
  // Compare end of backward to (θ_0, -p_0)
  const target = [THETA0[0], THETA0[1], -seedMomentum()[0], -seedMomentum()[1]];
  const final = [theta2[0], theta2[1], mom2[0], mom2[1]];
  const err =
    Math.hypot(final[0] - target[0], final[1] - target[1]) +
    Math.hypot(final[2] - target[2], final[3] - target[3]);
  return { forward, backward, roundTripError: err, divergent };
}

// Banana contour cache
interface ContourData {
  field: number[];
  nx: number;
  ny: number;
  thresholds: number[];
}
function makeBananaContours(): ContourData {
  const nx = 80;
  const ny = 100;
  const field: number[] = new Array(nx * ny);
  let lo = Infinity;
  let hi = -Infinity;
  for (let j = 0; j < ny; j++) {
    const y = Y_MIN + ((Y_MAX - Y_MIN) * j) / (ny - 1);
    for (let i = 0; i < nx; i++) {
      const x = X_MIN + ((X_MAX - X_MIN) * i) / (nx - 1);
      const v = bananaLogDensity([x, y], BANANA);
      field[j * nx + i] = v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  const nLevels = 8;
  const thresholds: number[] = [];
  for (let i = 0; i < nLevels; i++) thresholds.push(lo + ((hi - lo) * (i + 1)) / (nLevels + 1));
  return { field, nx, ny, thresholds };
}

export default function LeapfrogReversibilityDemo() {
  const [eps, setEps] = useState(DEFAULT_EPS);
  const [L, setL] = useState(DEFAULT_L);
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  const containerWidth = width || 720;
  const isMobile = containerWidth < SM_BREAKPOINT;
  const panelWidth = isMobile ? containerWidth - 24 : Math.min(640, containerWidth - 24);
  const panelHeight = isMobile ? 360 : 460;

  const contours = useMemo(() => makeBananaContours(), []);
  const result = useMemo(() => runReversibility(eps, L), [eps, L]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 30, right: 20, bottom: 40, left: 50 };
      const innerW = panelWidth - margin.left - margin.right;
      const innerH = panelHeight - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xScale = d3.scaleLinear().domain([X_MIN, X_MAX]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([Y_MIN, Y_MAX]).range([innerH, 0]);

      // Contours
      const contourGen = d3.contours().size([contours.nx, contours.ny]).thresholds(contours.thresholds);
      const cs = contourGen(contours.field);
      const colorScale = d3
        .scaleLinear<string>()
        .domain([contours.thresholds[0], contours.thresholds[contours.thresholds.length - 1]])
        .range(['rgba(44, 62, 80, 0.04)', 'rgba(44, 62, 80, 0.28)']);
      const geoPath = d3.geoPath(
        d3.geoTransform({
          point(this: { stream: { point: (x: number, y: number) => void } }, x: number, y: number): void {
            const px = (x / (contours.nx - 1)) * (X_MAX - X_MIN) + X_MIN;
            const py = (y / (contours.ny - 1)) * (Y_MAX - Y_MIN) + Y_MIN;
            this.stream.point(xScale(px), yScale(py));
          },
        }),
      );
      g.append('g')
        .selectAll('path')
        .data(cs)
        .join('path')
        .attr('d', geoPath as unknown as (d: d3.ContourMultiPolygon) => string)
        .style('fill', (d) => colorScale(d.value))
        .style('stroke', 'var(--color-text-secondary, #6B6B6B)')
        .style('stroke-width', 0.3)
        .style('opacity', 0.85);
      g.append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text, #1A1A1A)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text, #1A1A1A)');
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', -8)
        .attr('text-anchor', 'middle')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text, #1A1A1A)')
        .text('Forward + backward trajectories (banana)');
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 32)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .style('fill', 'var(--color-text-secondary, #6B6B6B)')
        .text('θ₁');
      g.append('text')
        .attr('x', -innerH / 2)
        .attr('y', -36)
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .style('font-size', '11px')
        .style('fill', 'var(--color-text-secondary, #6B6B6B)')
        .text('θ₂');

      const line = d3
        .line<[number, number]>()
        .x((p) => xScale(p[0]))
        .y((p) => yScale(p[1]));

      // Forward (solid olive)
      g.append('path')
        .datum(result.forward)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', paletteRMHMC.rmhmc)
        .style('stroke-width', 2.4)
        .style('opacity', 0.85);
      // Backward (dashed pink, slightly thicker to ensure overlap is visible)
      g.append('path')
        .datum(result.backward)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', paletteRMHMC.rmla)
        .style('stroke-width', 1.6)
        .style('stroke-dasharray', '5 4')
        .style('opacity', 0.95);

      // Start marker
      g.append('circle')
        .attr('cx', xScale(THETA0[0]))
        .attr('cy', yScale(THETA0[1]))
        .attr('r', 5)
        .style('fill', paletteRMHMC.target)
        .style('stroke', 'var(--color-surface, #fff)')
        .style('stroke-width', 1.5);
      // End-of-forward marker
      if (result.forward.length > 0) {
        const end = result.forward[result.forward.length - 1];
        g.append('rect')
          .attr('x', xScale(end[0]) - 4)
          .attr('y', yScale(end[1]) - 4)
          .attr('width', 8)
          .attr('height', 8)
          .style('fill', paletteRMHMC.rmhmc)
          .style('stroke', 'var(--color-surface, #fff)')
          .style('stroke-width', 1.5);
      }

      // Legend
      const legend = g.append('g').attr('transform', `translate(${innerW - 220}, 8)`);
      const items = [
        { color: paletteRMHMC.rmhmc, label: 'Forward Φ_ε^L', dashed: false },
        { color: paletteRMHMC.rmla, label: 'Backward Φ_ε^L (-p)', dashed: true },
      ];
      items.forEach((item, i) => {
        legend
          .append('line')
          .attr('x1', 0)
          .attr('x2', 24)
          .attr('y1', i * 16)
          .attr('y2', i * 16)
          .style('stroke', item.color)
          .style('stroke-width', 1.8)
          .style('stroke-dasharray', item.dashed ? '5 4' : 'none');
        legend
          .append('text')
          .attr('x', 28)
          .attr('y', i * 16 + 4)
          .style('font-size', '10.5px')
          .style('fill', 'var(--color-text, #1A1A1A)')
          .text(item.label);
      });
    },
    [contours, result, panelWidth, panelHeight],
  );

  return (
    <figure
      ref={ref}
      role="figure"
      aria-label="Figure 6: Generalized leapfrog reversibility demonstration"
      style={{ margin: '1.5rem 0' }}
    >
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.75rem 1.25rem',
          alignItems: 'center',
          fontSize: '0.875rem',
          color: 'var(--color-text, #333)',
          marginBottom: '0.75rem',
        }}
      >
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
          <span>step size ε:</span>
          <input
            type="range"
            min={0.02}
            max={0.4}
            step={0.01}
            value={eps}
            onChange={(e) => setEps(Number(e.target.value))}
            style={{ width: '140px' }}
            aria-label="Step size epsilon"
          />
          <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '4em' }}>ε = {eps.toFixed(2)}</span>
        </label>
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
          <span>steps L:</span>
          <input
            type="range"
            min={10}
            max={80}
            step={5}
            value={L}
            onChange={(e) => setL(Number(e.target.value))}
            style={{ width: '140px' }}
            aria-label="Steps L"
          />
          <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '3em' }}>L = {L}</span>
        </label>
        <span
          style={{
            color: result.divergent ? '#d62728' : 'var(--color-text-secondary, #6B6B6B)',
            fontSize: '0.85rem',
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          round-trip error:{' '}
          {result.divergent ? 'DIVERGENT' : result.roundTripError.toExponential(2)}
        </span>
      </div>
      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <svg ref={svgRef} width={panelWidth} height={panelHeight} role="img" aria-label="Forward and backward generalized-leapfrog trajectories" />
      </div>
      <figcaption
        style={{
          marginTop: '0.75rem',
          fontSize: '0.85rem',
          color: 'var(--color-text-secondary, #6B6B6B)',
          textAlign: 'center',
        }}
      >
        Figure 6. Theorem 6.2 in action. The forward trajectory (olive) runs L generalized-leapfrog
        steps from (θ_0, p_0). The backward trajectory (pink, dashed) starts at (θ_L, -p_L) — the
        endpoint of the forward path with momentum negated — and runs L more steps. By
        reversibility, it retraces the forward path exactly, landing at (θ_0, -p_0) up to
        fixed-point tolerance. Round-trip error reported above. Crucially, this holds for ANY ε
        for which the implicit solver converges — not just small ε. Reversibility is the
        structural property; energy conservation is only approximate.
      </figcaption>
    </figure>
  );
}
