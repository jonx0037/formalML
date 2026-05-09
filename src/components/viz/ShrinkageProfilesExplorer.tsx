import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bayesianLassoShrinkageMarginal,
  horseshoeShrinkageMarginal,
  ridgeShrinkageConstant,
  spikeSlabShrinkageMarginal,
} from './shared/bayesian-ml';

// =============================================================================
// ShrinkageProfilesExplorer — §1 four-prior shrinkage profile, interactive.
//
// Plots posterior expected shrinkage E[κ_j | y_j] vs signal magnitude |y_j| for
// the canonical four sparse priors:
//   - Ridge (gray, constant horizontal line at 1/(1+τ²))
//   - Bayesian LASSO (brown, exponential-tail shrinkage that persists on signals)
//   - Horseshoe (blue, vanishing shrinkage on signals — the namesake U-shape)
//   - Spike-and-slab (red dashed, discrete idealization)
//
// Sliders:
//   - τ_slab ∈ [0.5, 5] for spike-slab slab variance
//   - π_prior ∈ [0.05, 0.5] for spike-slab prior inclusion probability
//
// Source-of-truth: notebook §1 cell quadrature, mirrored in
// src/components/viz/shared/bayesian-ml.ts.
// =============================================================================

const PANEL_HEIGHT = 380;
const MARGIN = { top: 24, right: 32, bottom: 50, left: 56 };
const N_GRID = 80;
const Y_MIN = 0.05;
const Y_MAX = 8.0;

const RIDGE_COLOR = '#7f7f7f';
const LASSO_COLOR = '#7b3c10';
const HS_COLOR = '#1f4e79';
const SS_COLOR = '#c0504d';

interface ProfilePoint {
  y: number;
  ridge: number;
  lasso: number;
  hs: number;
  ss: number;
}

function computeProfiles(tauSlab: number, piPrior: number): ProfilePoint[] {
  const out: ProfilePoint[] = new Array(N_GRID);
  const ridgeK = ridgeShrinkageConstant(1.0);
  for (let i = 0; i < N_GRID; i++) {
    const y = Y_MIN + ((Y_MAX - Y_MIN) * i) / (N_GRID - 1);
    out[i] = {
      y,
      ridge: ridgeK,
      lasso: bayesianLassoShrinkageMarginal(y, 1.0),
      hs: horseshoeShrinkageMarginal(y, 1.0),
      ss: spikeSlabShrinkageMarginal(y, piPrior, tauSlab),
    };
  }
  return out;
}

export default function ShrinkageProfilesExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [tauSlab, setTauSlab] = useState(3.0);
  const [piPrior, setPiPrior] = useState(0.1);

  const data = useMemo(() => computeProfiles(tauSlab, piPrior), [tauSlab, piPrior]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const w = width || 720;
      const h = PANEL_HEIGHT;
      svg.attr('width', w).attr('height', h);
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = h - MARGIN.top - MARGIN.bottom;
      const g = svg
        .append('g')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLinear().domain([0, Y_MAX]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0, 1.05]).range([innerH, 0]).clamp(true);

      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(8))
        .call((sel) => sel.selectAll('text').style('font-size', '11px'));
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(6))
        .call((sel) => sel.selectAll('text').style('font-size', '11px'));

      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 36)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .text('signal magnitude |y_j| (data-implied posterior)');
      g.append('text')
        .attr('x', -innerH / 2)
        .attr('y', -38)
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .style('font-size', '12px')
        .text('posterior shrinkage  E[κ | y]');

      const lineGen = (key: keyof ProfilePoint) =>
        d3
          .line<ProfilePoint>()
          .x((d) => xScale(d.y))
          .y((d) => yScale(d[key] as number))
          .curve(d3.curveMonotoneX);

      const traces: Array<{
        key: keyof ProfilePoint;
        color: string;
        label: string;
        dash?: string;
        width: number;
      }> = [
        { key: 'ridge', color: RIDGE_COLOR, label: 'Ridge (λ ≡ 1)', width: 2 },
        { key: 'lasso', color: LASSO_COLOR, label: 'Bayesian LASSO (λ² ~ Exp)', width: 2 },
        { key: 'hs', color: HS_COLOR, label: 'Horseshoe (λ ~ HalfCauchy)', width: 2.6 },
        { key: 'ss', color: SS_COLOR, label: 'Spike-and-slab', dash: '6 4', width: 2 },
      ];

      traces.forEach((t) => {
        g.append('path')
          .datum(data)
          .attr('fill', 'none')
          .attr('stroke', t.color)
          .attr('stroke-width', t.width)
          .attr('stroke-dasharray', t.dash ?? '')
          .attr('d', lineGen(t.key));
      });

      // Legend
      const legend = g
        .append('g')
        .attr('transform', `translate(${innerW - 230}, 8)`);
      traces.forEach((t, i) => {
        const row = legend
          .append('g')
          .attr('transform', `translate(0, ${i * 18})`);
        row
          .append('line')
          .attr('x1', 0)
          .attr('x2', 22)
          .attr('y1', 6)
          .attr('y2', 6)
          .attr('stroke', t.color)
          .attr('stroke-width', t.width)
          .attr('stroke-dasharray', t.dash ?? '');
        row
          .append('text')
          .attr('x', 28)
          .attr('y', 10)
          .style('font-size', '11px')
          .style('fill', 'var(--color-text, #222)')
          .text(t.label);
      });
    },
    [data, width],
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
          τ<sub>slab</sub>: <span style={{ fontVariantNumeric: 'tabular-nums' }}>{tauSlab.toFixed(2)}</span>{' '}
          <input
            type="range"
            min={0.5}
            max={5.0}
            step={0.05}
            value={tauSlab}
            onChange={(e) => setTauSlab(Number(e.target.value))}
            style={{ width: 140 }}
          />
        </label>
        <label>
          π<sub>prior</sub>: <span style={{ fontVariantNumeric: 'tabular-nums' }}>{piPrior.toFixed(2)}</span>{' '}
          <input
            type="range"
            min={0.05}
            max={0.5}
            step={0.01}
            value={piPrior}
            onChange={(e) => setPiPrior(Number(e.target.value))}
            style={{ width: 140 }}
          />
        </label>
      </div>
      <svg ref={svgRef} role="img" aria-label="Four-prior posterior shrinkage profiles" />
    </div>
  );
}
