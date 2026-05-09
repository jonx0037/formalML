import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { horseshoeShrinkageMarginal } from './shared/bayesian-ml';

// =============================================================================
// HorseshoeShrinkageProfile — §4 Beta(1/2, 1/2) horseshoe-shape density.
//
// Two overlaid traces on κ ∈ (0, 1):
//   - Closed-form Beta(1/2, 1/2) density (Theorem 3): the U-shape namesake.
//   - Posterior expected shrinkage E[κ | y] for a user-controlled |y| anchor,
//     drawn as a vertical reference line marked on the κ axis.
//
// Slider:
//   - τ ∈ [0.05, 8] global scale (the U-shape persists, the *balance* between
//     arms shifts: small τ → more mass at κ=1, large τ → more mass at κ=0).
//   - |y| ∈ [0.1, 8] data-implied signal magnitude (drives the posterior dot).
// =============================================================================

const PANEL_HEIGHT = 360;
const MARGIN = { top: 20, right: 32, bottom: 50, left: 56 };
const N_GRID = 200;

const PRIOR_COLOR = '#1f4e79';
const POSTERIOR_COLOR = '#c0504d';

export default function HorseshoeShrinkageProfile() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [tau, setTau] = useState(1.0);
  const [yAnchor, setYAnchor] = useState(2.0);

  const data = useMemo(() => {
    // Prior κ density via Monte Carlo (since for general τ, κ is no longer
    // Beta(1/2,1/2) but a transformed version). Use sample-based approach for
    // generality, but the τ=1 case will collapse onto the closed-form Beta
    // density.
    const grid = new Array<number>(N_GRID);
    const beta12 = new Array<number>(N_GRID);
    for (let i = 0; i < N_GRID; i++) {
      const k = (i + 0.5) / N_GRID;
      grid[i] = k;
      // For general τ, the density of κ = 1/(1 + λ²τ²) under λ ~ HalfCauchy(0, 1)
      // can be derived analytically:
      //   p_κ(κ) = (1/π) τ⁻¹ κ⁻¹/² (1-κ)⁻¹/² / (κ + τ²(1-κ))
      // which reduces to Beta(1/2, 1/2) at τ = 1.
      const denom = k + tau * tau * (1 - k);
      beta12[i] =
        (1 / (Math.PI * tau * Math.sqrt(k * (1 - k)))) *
        (tau * tau / denom);
    }
    return { grid, beta12 };
  }, [tau]);

  const posteriorKappa = useMemo(
    () => horseshoeShrinkageMarginal(yAnchor, tau),
    [tau, yAnchor],
  );

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

      const xScale = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
      const yMax = Math.max(d3.max(data.beta12) ?? 4, 4);
      const yScale = d3
        .scaleLinear()
        .domain([0, Math.min(yMax, 6)])
        .range([innerH, 0])
        .clamp(true);

      g.append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(6));
      g.append('g').call(d3.axisLeft(yScale).ticks(5));

      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 36)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .text('shrinkage factor κ');
      g.append('text')
        .attr('x', -innerH / 2)
        .attr('y', -38)
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .style('font-size', '12px')
        .text('p(κ | τ)');

      // Beta(1/2,1/2)-like prior density (transformed for general τ).
      const lineGen = d3
        .line<number>()
        .x((_, i) => xScale(data.grid[i]))
        .y((d) => yScale(d))
        .curve(d3.curveMonotoneX);
      g.append('path')
        .datum(data.beta12)
        .attr('fill', 'none')
        .attr('stroke', PRIOR_COLOR)
        .attr('stroke-width', 2.6)
        .attr('d', lineGen);

      // Posterior κ marker line + dot
      const px = xScale(posteriorKappa);
      g.append('line')
        .attr('x1', px)
        .attr('x2', px)
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', POSTERIOR_COLOR)
        .attr('stroke-width', 1.4)
        .attr('stroke-dasharray', '4 4');
      g.append('circle')
        .attr('cx', px)
        .attr('cy', innerH - 8)
        .attr('r', 5)
        .attr('fill', POSTERIOR_COLOR);
      g.append('text')
        .attr('x', px + 8)
        .attr('y', 14)
        .style('font-size', '11px')
        .style('fill', POSTERIOR_COLOR)
        .text(`E[κ | y=${yAnchor.toFixed(2)}] = ${posteriorKappa.toFixed(3)}`);

      // Static legend
      const legend = g.append('g').attr('transform', `translate(${innerW - 230}, 6)`);
      legend
        .append('line')
        .attr('x1', 0)
        .attr('x2', 22)
        .attr('y1', 6)
        .attr('y2', 6)
        .attr('stroke', PRIOR_COLOR)
        .attr('stroke-width', 2.6);
      legend
        .append('text')
        .attr('x', 28)
        .attr('y', 10)
        .style('font-size', '11px')
        .text('prior p(κ | τ)');
    },
    [data, posteriorKappa, yAnchor, width],
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
          τ: <span style={{ fontVariantNumeric: 'tabular-nums' }}>{tau.toFixed(3)}</span>{' '}
          <input
            type="range"
            min={0.05}
            max={4.0}
            step={0.05}
            value={tau}
            onChange={(e) => setTau(Number(e.target.value))}
            style={{ width: 160 }}
          />
        </label>
        <label>
          |y|: <span style={{ fontVariantNumeric: 'tabular-nums' }}>{yAnchor.toFixed(2)}</span>{' '}
          <input
            type="range"
            min={0.1}
            max={8.0}
            step={0.05}
            value={yAnchor}
            onChange={(e) => setYAnchor(Number(e.target.value))}
            style={{ width: 160 }}
          />
        </label>
      </div>
      <svg ref={svgRef} role="img" aria-label="Horseshoe Beta(1/2,1/2) shrinkage density" />
    </div>
  );
}
