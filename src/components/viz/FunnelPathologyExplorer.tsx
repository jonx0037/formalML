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
// In (log τ, β_j) coordinates, with β_j = z·λ_j·τ, the joint prior has a funnel —
// the conditional spread of β_j given log τ (scale λ_j·τ) collapses as τ → 0.
// (Under the prior log λ_j is independent of log τ, so the funnel is not in the
// (log τ, log λ) plane.) The non-centered toggle keeps (log τ, z) coordinates
// instead, eliminating the funnel by construction. The c slider applies the
// Piironen-Vehtari regularized truncation
//     λ̃² = c²λ² / (c² + τ²λ²)   (so λ̃ = sqrt(c²λ² / (c² + τ²λ²)))
// which caps the joint prior scale λ̃·τ at c: the wide mouth of the funnel is
// bounded (95% of β_j | τ within ±1.96c) while the neck at small τ remains.
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
      // z ~ N(0,1), the standardized non-centered latent (Box-Muller).
      const u1 = Math.max(rng(), 1e-12);
      const u2 = rng();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      xs[i] = logTau;
      if (mode === 'centered') {
        // Centered: β_j = z · λ_j · τ
        ys[i] = z * lam * tau;
      } else if (mode === 'non-centered') {
        ys[i] = z;
      } else {
        // Regularized: β_j = z · λ̃_j · τ with the Piironen-Vehtari truncation
        const tilde = regularizedHorseshoeLambdaTilde(lam, tau, c);
        ys[i] = z * tilde * tau;
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
      const yHalf = mode === 'regularized' ? Math.max(6, 2.2 * c) : 6;
      const yExt: [number, number] = mode === 'non-centered' ? [-4, 4] : [-yHalf, yHalf];
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
              ? 'β_j = z · λ̃_j · τ'
              : 'β_j = z · λ_j · τ',
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
              : `Regularized: prior variance λ̃²τ² capped at c² = ${(c * c).toFixed(1)}`,
        );

      // Plot points
      const color =
        mode === 'centered' ? '#c0504d' : mode === 'non-centered' ? '#3a6e3a' : '#1f4e79';
      const inWindow = d3
        .range(N_SAMPLES)
        .filter(
          (i) =>
            data.xs[i] >= xExt[0] &&
            data.xs[i] <= xExt[1] &&
            data.ys[i] >= yExt[0] &&
            data.ys[i] <= yExt[1],
        );
      g.selectAll('circle')
        .data(inWindow)
        .enter()
        .append('circle')
        .attr('cx', (i) => x(data.xs[i]))
        .attr('cy', (i) => y(data.ys[i]))
        .attr('r', 1.0)
        .attr('fill', color)
        .attr('opacity', 0.30);

      // Funnel envelope reference (centered case): ±1.96·τ, the 95% interval of
      // β_j | τ at the typical local scale λ_j = 1 — it collapses as τ → 0 (the neck).
      if (mode === 'centered') {
        const envXs = d3.range(80).map((i) => xExt[0] + ((xExt[1] - xExt[0]) * i) / 79);
        const lineGen = d3
          .line<{ x: number; y: number }>()
          .x((d) => x(d.x))
          .y((d) => y(Math.max(yExt[0], Math.min(yExt[1], d.y))));
        for (const sign of [1, -1]) {
          g.append('path')
            .datum(envXs.map((xv) => ({ x: xv, y: sign * 1.96 * Math.exp(xv) })))
            .attr('fill', 'none')
            .attr('stroke', '#7f7f7f')
            .attr('stroke-dasharray', '4 4')
            .attr('stroke-width', 1)
            .attr('d', lineGen);
        }
      }

      // Slab cap reference (regularized): λ̃²τ² ≤ c², so 95% of β_j | τ lies within ±1.96c.
      if (mode === 'regularized') {
        for (const sign of [1, -1]) {
          g.append('line')
            .attr('x1', 0)
            .attr('x2', innerW)
            .attr('y1', y(sign * 1.96 * c))
            .attr('y2', y(sign * 1.96 * c))
            .attr('stroke', '#aa6633')
            .attr('stroke-dasharray', '4 4')
            .attr('stroke-width', 1);
        }
        g.append('text')
          .attr('x', innerW - 6)
          .attr('y', y(1.96 * c) - 4)
          .attr('text-anchor', 'end')
          .style('font-size', '10.5px')
          .style('fill', '#aa6633')
          .text(`±1.96c = ±${(1.96 * c).toFixed(2)} (slab cap on the prior scale)`);
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
