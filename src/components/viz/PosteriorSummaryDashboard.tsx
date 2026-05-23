import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  betaPdf,
  betaMean,
  betaMode,
  betaQuantile,
  betaCentralCI,
  betaHPD,
  COLOR_VARIANT_A,
  COLOR_VARIANT_B,
} from './shared/bayesian-inference';

// =============================================================================
// PosteriorSummaryDashboard — §5 interactive.
//
// Two columns of posteriors: a symmetric Beta and a skewed Beta. Each column
// overlays the three point estimates (MAP, median, mean) on the density and
// renders horizontal bars for the central credible interval and the HPD.
//
// Pedagogical takeaway: MAP / mean / median coincide for symmetric
// posteriors and diverge for skewed ones. HPD shorter than central CI for
// skewed cases. Each estimate is the Bayes rule under a particular loss.
// =============================================================================

const THETA_N = 401;
const theta = Float64Array.from({ length: THETA_N }, (_, i) => i / (THETA_N - 1));

interface PanelStats {
  alpha: number;
  beta: number;
  density: Float64Array;
  map: number;
  median: number;
  mean: number;
  central: { lower: number; upper: number };
  hpd: { lower: number; upper: number };
}

function computePanel(alpha: number, beta: number, level: number): PanelStats {
  const density = new Float64Array(THETA_N);
  for (let i = 0; i < THETA_N; i++) density[i] = betaPdf(theta[i], alpha, beta);
  return {
    alpha,
    beta,
    density,
    map: betaMode(alpha, beta),
    median: betaQuantile(0.5, alpha, beta),
    mean: betaMean(alpha, beta),
    central: betaCentralCI(alpha, beta, level),
    hpd: betaHPD(alpha, beta, level),
  };
}

export default function PosteriorSummaryDashboard() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [skewness, setSkewness] = useState(2); // alpha in the skewed panel
  const [level, setLevel] = useState(0.95);

  const W = Math.max(360, Math.min(width || 720, 800));
  const H = 420;
  const margin = { top: 28, right: 12, bottom: 60, left: 40 };
  const colW = (W - margin.left - margin.right) / 2 - 18;

  const panels = useMemo(
    () => [computePanel(20, 20, level), computePanel(skewness, 20, level)],
    [skewness, level],
  );

  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);

      panels.forEach((p, k) => {
        const xLeft = margin.left + k * (colW + 36);
        const yTop = margin.top;
        const innerH = H - margin.top - margin.bottom - 70;
        const xDomain: [number, number] = k === 0 ? [0, 1] : [0, 0.35];
        const x = d3.scaleLinear().domain(xDomain).range([0, colW]);
        let ymax = 0;
        for (let i = 0; i < THETA_N; i++) {
          const t = theta[i];
          if (t <= xDomain[1] && p.density[i] > ymax) ymax = p.density[i];
        }
        const y = d3.scaleLinear().domain([0, ymax * 1.1]).range([innerH, 0]);
        const g = svg.append('g').attr('transform', `translate(${xLeft}, ${yTop})`);
        const color = k === 0 ? COLOR_VARIANT_A : COLOR_VARIANT_B;

        g.append('text')
          .attr('x', colW / 2)
          .attr('y', -10)
          .attr('text-anchor', 'middle')
          .attr('font-size', 11)
          .attr('font-weight', 600)
          .style('fill', 'var(--color-text)')
          .text(`Beta(${p.alpha}, ${p.beta}) — ${k === 0 ? 'symmetric' : 'skewed'}`);

        // Area
        const area = d3
          .area<number>()
          .x((_, i) => x(theta[i]))
          .y0(innerH)
          .y1((v) => y(v))
          .curve(d3.curveMonotoneX);
        g.append('path')
          .attr('d', area(Array.from(p.density)) ?? '')
          .attr('fill', color)
          .attr('opacity', 0.28);
        g.append('path')
          .attr(
            'd',
            d3
              .line<number>()
              .x((_, i) => x(theta[i]))
              .y((v) => y(v))
              .curve(d3.curveMonotoneX)(Array.from(p.density)) ?? '',
          )
          .attr('fill', 'none')
          .attr('stroke', color)
          .attr('stroke-width', 1.8);

        // Point estimates as vertical lines
        const estimates: { label: string; v: number; color: string; dash: string }[] = [
          { label: 'MAP', v: p.map, color: '#9467bd', dash: '4 2' },
          { label: 'median', v: p.median, color: '#3a6e3a', dash: '2 2' },
          { label: 'mean', v: p.mean, color: '#c0504d', dash: '' },
        ];
        estimates.forEach((e) => {
          g.append('line')
            .attr('x1', x(e.v))
            .attr('x2', x(e.v))
            .attr('y1', 0)
            .attr('y2', innerH)
            .attr('stroke', e.color)
            .attr('stroke-width', 1.5)
            .attr('stroke-dasharray', e.dash || 'none');
          g.append('text')
            .attr('x', x(e.v))
            .attr('y', -2)
            .attr('text-anchor', 'middle')
            .attr('font-size', 9)
            .style('fill', e.color)
            .text(e.label);
        });

        // X axis
        g.append('g')
          .attr('transform', `translate(0, ${innerH})`)
          .call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));

        // CI / HPD bars
        const barY = innerH + 24;
        g.append('text')
          .attr('x', 0)
          .attr('y', barY - 4)
          .attr('font-size', 9)
          .style('fill', 'var(--color-text)')
          .text(`${(level * 100).toFixed(0)}% central CI`);
        g.append('line')
          .attr('x1', x(p.central.lower))
          .attr('x2', x(p.central.upper))
          .attr('y1', barY + 5)
          .attr('y2', barY + 5)
          .attr('stroke', color)
          .attr('stroke-width', 4);
        g.append('text')
          .attr('x', 0)
          .attr('y', barY + 22)
          .attr('font-size', 9)
          .style('fill', 'var(--color-text)')
          .text(`${(level * 100).toFixed(0)}% HPD`);
        g.append('line')
          .attr('x1', x(p.hpd.lower))
          .attr('x2', x(p.hpd.upper))
          .attr('y1', barY + 31)
          .attr('y2', barY + 31)
          .attr('stroke', color)
          .attr('stroke-width', 4)
          .attr('opacity', 0.65);
      });
    },
    [panels, W, H, level],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={ref} role="img" aria-label="Posterior summary dashboard" />
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, minmax(140px, 1fr))',
          gap: '0.75rem',
          marginTop: '0.5rem',
          fontSize: '0.85rem',
        }}
      >
        <label>
          Skewness (right α): <strong>{skewness.toFixed(1)}</strong>
          <input
            type="range"
            min={1.2}
            max={20}
            step={0.1}
            value={skewness}
            onChange={(e) => setSkewness(Number(e.target.value))}
            aria-label="skewness alpha"
            style={{ width: '100%' }}
          />
        </label>
        <label>
          Credibility level: <strong>{(level * 100).toFixed(0)}%</strong>
          <input
            type="range"
            min={0.5}
            max={0.99}
            step={0.01}
            value={level}
            onChange={(e) => setLevel(Number(e.target.value))}
            aria-label="credibility level"
            style={{ width: '100%' }}
          />
        </label>
      </div>
      <p style={{ opacity: 0.7, marginTop: '0.5rem', fontSize: '0.85rem' }}>
        Symmetric panel: MAP = median = mean and HPD = central CI. Skewed panel:
        the three point estimates separate, and the HPD is shifted toward the
        mode (shorter than the central CI).
      </p>
    </div>
  );
}
