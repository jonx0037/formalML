import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  normalPdf,
  normalNormalUpdate,
  normalNormalPosteriorPredictiveVar,
  COLOR_VARIANT_A,
  COLOR_LIKELIHOOD,
  COLOR_ACCENT,
} from './shared/bayesian-inference';

// =============================================================================
// PosteriorPredictiveBands — §6 interactive.
//
// Two panels for the Normal-Normal running example.
// Top: three densities — parameter posterior N(mu_n, tau_n^2),
// plug-in predictive N(yBar, sigma^2), and Bayesian posterior predictive
// N(mu_n, tau_n^2 + sigma^2). 95% intervals annotated.
// Bottom: the ratio (posterior-predictive width) / (plug-in width) as a
// function of n, showing the gap shrinks to 1 as n grows.
//
// Pedagogical takeaway: the Bayesian posterior predictive is structurally
// broader than the plug-in — the gap is the cost of ignoring parameter
// uncertainty.
// =============================================================================

const Y_N = 401;
const X_DOMAIN: [number, number] = [-4, 6];
const yGrid = Float64Array.from({ length: Y_N }, (_, i) =>
  X_DOMAIN[0] + (i * (X_DOMAIN[1] - X_DOMAIN[0])) / (Y_N - 1),
);

export default function PosteriorPredictiveBands() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [n, setN] = useState(10);
  const [tau0Sq, setTau0Sq] = useState(1);
  const [sigmaSq, setSigmaSq] = useState(1);
  const yBar = 1; // fixed for visualization

  const W = Math.max(360, Math.min(width || 720, 760));
  const H = 460;
  const margin = { top: 26, right: 14, bottom: 36, left: 50 };

  const data = useMemo(() => {
    const post = normalNormalUpdate({ mu0: 0, tau0Sq, sigmaSq, yBar, n });
    const ppVar = normalNormalPosteriorPredictiveVar(post, sigmaSq);
    const paramPdf = new Float64Array(Y_N);
    const pluginPdf = new Float64Array(Y_N);
    const ppPdf = new Float64Array(Y_N);
    for (let i = 0; i < Y_N; i++) {
      const y = yGrid[i];
      paramPdf[i] = normalPdf(y, post.muN, Math.sqrt(post.tauNSq));
      pluginPdf[i] = normalPdf(y, yBar, Math.sqrt(sigmaSq));
      ppPdf[i] = normalPdf(y, post.muN, Math.sqrt(ppVar));
    }
    // Width ratio curve as a function of n (1 .. 200)
    const nGrid = d3.range(1, 201);
    const ratio = nGrid.map((nn) => {
      const p = normalNormalUpdate({ mu0: 0, tau0Sq, sigmaSq, yBar, n: nn });
      const ppV = normalNormalPosteriorPredictiveVar(p, sigmaSq);
      return Math.sqrt(ppV / sigmaSq);
    });
    return { post, ppVar, paramPdf, pluginPdf, ppPdf, nGrid, ratio };
  }, [n, tau0Sq, sigmaSq]);

  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);

      const innerW = W - margin.left - margin.right;
      const topH = (H - margin.top - margin.bottom) * 0.6 - 12;
      const botH = (H - margin.top - margin.bottom) * 0.4;

      // Top panel — three densities
      const x = d3.scaleLinear().domain(X_DOMAIN).range([0, innerW]);
      let ymax = 0;
      for (let i = 0; i < Y_N; i++) {
        if (data.paramPdf[i] > ymax) ymax = data.paramPdf[i];
        if (data.pluginPdf[i] > ymax) ymax = data.pluginPdf[i];
        if (data.ppPdf[i] > ymax) ymax = data.ppPdf[i];
      }
      const y = d3.scaleLinear().domain([0, ymax * 1.1]).range([topH, 0]);
      const gTop = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

      const datasets: { label: string; dens: Float64Array; color: string }[] = [
        { label: 'Parameter posterior π(θ|y)', dens: data.paramPdf, color: COLOR_VARIANT_A },
        { label: 'Plug-in predictive p(ỹ|θ̂)', dens: data.pluginPdf, color: COLOR_LIKELIHOOD },
        { label: 'Bayesian posterior predictive p(ỹ|y)', dens: data.ppPdf, color: COLOR_ACCENT },
      ];
      datasets.forEach((d) => {
        gTop.append('path')
          .attr(
            'd',
            d3
              .line<number>()
              .x((_, i) => x(yGrid[i]))
              .y((v) => y(v))
              .curve(d3.curveMonotoneX)(d.dens as unknown as number[]) ?? '',
          )
          .attr('fill', 'none')
          .attr('stroke', d.color)
          .attr('stroke-width', 1.8);
      });
      gTop.append('g')
        .attr('transform', `translate(0, ${topH})`)
        .call(d3.axisBottom(x).ticks(7).tickSizeOuter(0));
      gTop.append('g').call(d3.axisLeft(y).ticks(5).tickSizeOuter(0));

      // Legend
      const legend = gTop.append('g').attr('transform', `translate(${innerW - 220}, 8)`);
      datasets.forEach((d, i) => {
        legend.append('rect')
          .attr('x', 0)
          .attr('y', i * 16)
          .attr('width', 12)
          .attr('height', 8)
          .attr('fill', d.color);
        legend.append('text')
          .attr('x', 16)
          .attr('y', i * 16 + 8)
          .attr('font-size', 9)
          .style('fill', 'var(--color-text)')
          .text(d.label);
      });

      // Bottom panel — width ratio curve
      const gBot = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top + topH + 30})`);
      const xb = d3.scaleLog().domain([1, 200]).range([0, innerW]);
      const yb = d3
        .scaleLinear()
        .domain([1, Math.max(...data.ratio) * 1.05])
        .range([botH, 0]);
      gBot.append('path')
        .attr(
          'd',
          d3
            .line<number>()
            .x((_, i) => xb(data.nGrid[i]))
            .y((v) => yb(v))(data.ratio) ?? '',
        )
        .attr('fill', 'none')
        .attr('stroke', COLOR_ACCENT)
        .attr('stroke-width', 1.8);
      // Reference line at ratio = 1
      gBot.append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yb(1))
        .attr('y2', yb(1))
        .attr('stroke', 'var(--color-text)')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3 3');
      // Marker for current n
      gBot.append('circle')
        .attr('cx', xb(n))
        .attr('cy', yb(Math.sqrt(data.ppVar / sigmaSq)))
        .attr('r', 5)
        .attr('fill', COLOR_ACCENT);
      gBot.append('g')
        .attr('transform', `translate(0, ${botH})`)
        .call(d3.axisBottom(xb).ticks(5, '~s').tickSizeOuter(0));
      gBot.append('g').call(d3.axisLeft(yb).ticks(4).tickSizeOuter(0));
      gBot.append('text')
        .attr('x', innerW / 2)
        .attr('y', botH + 30)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .style('fill', 'var(--color-text)')
        .text('n (log scale) — predictive-width gap shrinks to 1 as n → ∞');
    },
    [data, W, H, n, sigmaSq],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={ref} role="img" aria-label="Posterior predictive bands" />
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, minmax(120px, 1fr))',
          gap: '0.75rem',
          marginTop: '0.5rem',
          fontSize: '0.85rem',
        }}
      >
        <label>
          n: <strong>{n}</strong>
          <input
            type="range"
            min={1}
            max={200}
            step={1}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            aria-label="sample size n"
            style={{ width: '100%' }}
          />
        </label>
        <label>
          τ₀² (prior var): <strong>{tau0Sq.toFixed(2)}</strong>
          <input
            type="range"
            min={0.1}
            max={5}
            step={0.05}
            value={tau0Sq}
            onChange={(e) => setTau0Sq(Number(e.target.value))}
            aria-label="prior variance"
            style={{ width: '100%' }}
          />
        </label>
        <label>
          σ² (noise): <strong>{sigmaSq.toFixed(2)}</strong>
          <input
            type="range"
            min={0.1}
            max={5}
            step={0.05}
            value={sigmaSq}
            onChange={(e) => setSigmaSq(Number(e.target.value))}
            aria-label="likelihood variance"
            style={{ width: '100%' }}
          />
        </label>
      </div>
    </div>
  );
}
