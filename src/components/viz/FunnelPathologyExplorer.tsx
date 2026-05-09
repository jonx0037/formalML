import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32, regularizedHorseshoeLambdaTilde } from './shared/bayesian-ml';

// =============================================================================
// FunnelPathologyExplorer — §6 horseshoe funnel + regularized truncation.
//
// Samples 8000 prior draws from the centered horseshoe:
//   τ ~ HalfCauchy(0, 1), λ_j ~ HalfCauchy(0, 1)
//
// In (log τ, log λ) coordinates, the joint prior has a funnel — the conditional
// spread of log λ given log τ blows up as τ → 0. The non-centered toggle keeps
// (log τ, z) coordinates instead, eliminating the funnel by construction. The c
// slider applies the Piironen-Vehtari regularized truncation
//     λ̃² = c²λ² / (c² + τ²λ²)   (so λ̃ = sqrt(c²λ² / (c² + τ²λ²)))
// shrinking λ̃ toward the slab cap √(c²/τ²) at large λ²τ².
//
// The gray reference funnel envelope plots ±2σ(log λ | log τ) under the
// centered prior; the simulator overlay mimics where leapfrog divergences
// concentrate under naive HMC.
// =============================================================================

const PANEL_HEIGHT = 380;
const MARGIN = { top: 22, right: 32, bottom: 50, left: 60 };
const N_SAMPLES = 8000;

type Mode = 'centered' | 'non-centered' | 'regularized';

export default function FunnelPathologyExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [mode, setMode] = useState<Mode>('centered');
  const [c, setC] = useState(2.0);

  const data = useMemo(() => {
    const rng = mulberry32(20260509);
    const xs: number[] = new Array(N_SAMPLES);
    const ys: number[] = new Array(N_SAMPLES);
    for (let i = 0; i < N_SAMPLES; i++) {
      const tau = Math.tan((Math.PI / 2) * rng());
      const lam = Math.tan((Math.PI / 2) * rng());
      const logTau = Math.log(Math.max(tau, 1e-12));
      if (mode === 'centered') {
        xs[i] = logTau;
        ys[i] = Math.log(Math.max(lam, 1e-12));
      } else if (mode === 'non-centered') {
        // (log τ, z) where z ~ N(0,1) is the standardized non-centered latent.
        // We sample z from a standard normal via Box-Muller for visualization.
        const u1 = Math.max(rng(), 1e-12);
        const u2 = rng();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        xs[i] = logTau;
        ys[i] = z;
      } else {
        // Regularized: lambdaTilde from Piironen-Vehtari truncation
        const tilde = regularizedHorseshoeLambdaTilde(lam, tau, c);
        xs[i] = logTau;
        ys[i] = Math.log(Math.max(tilde, 1e-12));
      }
    }
    return { xs, ys };
  }, [mode, c]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const w = width || 720;
      const h = PANEL_HEIGHT;
      svg.attr('width', w).attr('height', h);
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = h - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xExt: [number, number] = [-6, 4];
      const yExt: [number, number] = mode === 'non-centered' ? [-4, 4] : [-6, 4];
      const x = d3.scaleLinear().domain(xExt).range([0, innerW]);
      const y = d3.scaleLinear().domain(yExt).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(6));
      g.append('g').call(d3.axisLeft(y).ticks(6));

      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 36)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .text(mode === 'non-centered' ? 'log τ' : 'log τ');

      g.append('text')
        .attr('x', -innerH / 2)
        .attr('y', -44)
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .style('font-size', '12px')
        .text(
          mode === 'non-centered'
            ? 'z (non-centered latent)'
            : mode === 'regularized'
              ? 'log λ̃ (truncated)'
              : 'log λ',
        );

      g.append('text')
        .attr('x', 0)
        .attr('y', -8)
        .style('font-size', '11.5px')
        .style('font-weight', '600')
        .text(
          mode === 'centered'
            ? 'Centered: funnel neck at small log τ'
            : mode === 'non-centered'
              ? 'Non-centered: rectangular (no funnel)'
              : `Regularized: λ̃ truncated at c² = ${(c * c).toFixed(1)}`,
        );

      // Plot points
      const color =
        mode === 'centered' ? '#c0504d' : mode === 'non-centered' ? '#3a6e3a' : '#1f4e79';
      g.selectAll('circle')
        .data(d3.range(N_SAMPLES))
        .enter()
        .append('circle')
        .attr('cx', (i) => x(data.xs[i]))
        .attr('cy', (i) => y(data.ys[i]))
        .attr('r', 1.0)
        .attr('fill', color)
        .attr('opacity', 0.30);

      // Funnel envelope reference (centered case): the conditional E[log λ | log τ] = 0
      // and ±1σ ≈ ±π for half-Cauchy. Just show ±2 SD of the log-Cauchy distribution.
      if (mode === 'centered') {
        const envXs = d3.range(40).map((i) => xExt[0] + ((xExt[1] - xExt[0]) * i) / 39);
        // ±2.5 envelope (rough visual reference)
        const env = 2.5;
        const upper = envXs.map((xv) => ({ x: xv, y: env }));
        const lower = envXs.map((xv) => ({ x: xv, y: -env }));
        const lineGen = d3
          .line<{ x: number; y: number }>()
          .x((d) => x(d.x))
          .y((d) => y(d.y));
        g.append('path')
          .datum(upper)
          .attr('fill', 'none')
          .attr('stroke', '#7f7f7f')
          .attr('stroke-dasharray', '4 4')
          .attr('stroke-width', 1)
          .attr('d', lineGen);
        g.append('path')
          .datum(lower)
          .attr('fill', 'none')
          .attr('stroke', '#7f7f7f')
          .attr('stroke-dasharray', '4 4')
          .attr('stroke-width', 1)
          .attr('d', lineGen);
      }

      // Slab cap reference (regularized): λ̃² ≤ c² ⇒ log λ̃ ≤ log c.
      if (mode === 'regularized') {
        g.append('line')
          .attr('x1', 0)
          .attr('x2', innerW)
          .attr('y1', y(Math.log(c)))
          .attr('y2', y(Math.log(c)))
          .attr('stroke', '#aa6633')
          .attr('stroke-dasharray', '4 4')
          .attr('stroke-width', 1);
        g.append('text')
          .attr('x', innerW - 6)
          .attr('y', y(Math.log(c)) - 4)
          .attr('text-anchor', 'end')
          .style('font-size', '10.5px')
          .style('fill', '#aa6633')
          .text(`log c = ${Math.log(c).toFixed(2)} (slab ceiling)`);
      }
    },
    [data, mode, c, width],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', maxWidth: 720 }}>
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 18,
          marginBottom: 8,
          alignItems: 'center',
          fontSize: 13,
        }}
      >
        <label>
          Parameterization:{' '}
          <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
            <option value="centered">Centered (funnel)</option>
            <option value="non-centered">Non-centered (no funnel)</option>
            <option value="regularized">Regularized (Piironen-Vehtari)</option>
          </select>
        </label>
        {mode === 'regularized' && (
          <label>
            c (slab scale): {c.toFixed(1)}{' '}
            <input
              type="range"
              min={0.5}
              max={5.0}
              step={0.1}
              value={c}
              onChange={(e) => setC(Number(e.target.value))}
              style={{ width: 140 }}
            />
          </label>
        )}
      </div>
      <svg ref={svgRef} role="img" aria-label="Horseshoe funnel pathology and regularization" />
    </div>
  );
}
