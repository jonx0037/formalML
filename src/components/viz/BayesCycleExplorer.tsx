import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  betaPdf,
  betaBinomialUpdate,
  betaMean,
  betaCentralCI,
  COLOR_VARIANT_A,
  COLOR_LIKELIHOOD,
  COLOR_VARIANT_B,
} from './shared/bayesian-inference';

// =============================================================================
// BayesCycleExplorer — §2.2 interactive.
//
// Three stacked panels on a shared theta-axis show the prior pi(theta),
// the (rescaled) likelihood p(y|theta), and the resulting Beta-Binomial
// posterior. Reader controls alpha, beta, n, s sliders.
//
// Pedagogical takeaway: with little data the prior dominates; with lots
// of data the likelihood dominates. The bottom posterior interpolates
// between the prior (top) and likelihood-mode (middle) accordingly.
// =============================================================================

const THETA_N = 401;
const theta = Float64Array.from({ length: THETA_N }, (_, i) => i / (THETA_N - 1));

function panelDensity(values: Float64Array): { ymax: number } {
  let ymax = 0;
  for (let i = 0; i < values.length; i++) if (values[i] > ymax) ymax = values[i];
  return { ymax: ymax > 0 ? ymax : 1 };
}

export default function BayesCycleExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [alpha, setAlpha] = useState(1);
  const [beta, setBeta] = useState(1);
  const [n, setN] = useState(1000);
  const [s, setS] = useState(47);

  const W = Math.max(360, Math.min(width || 720, 760));
  const H = 480;
  const margin = { top: 30, right: 16, bottom: 36, left: 50 };

  const data = useMemo(() => {
    const prior = new Float64Array(THETA_N);
    const likelihood = new Float64Array(THETA_N);
    const post = new Float64Array(THETA_N);
    const updated = betaBinomialUpdate(alpha, beta, n, s);
    // Log-likelihoods are negative — initialize to -Infinity so the first
    // valid value always becomes the max. The earlier `let maxLik = 0` + `||
    // i === 0` guard never fired the special case because theta[0] = 0
    // fails the `t > 0 && t < 1` check, leaving maxLik stuck at 0 and the
    // exponentiated likelihood ≈ 0 everywhere.
    let maxLik = -Infinity;
    const nMinusS = n - s;
    for (let i = 0; i < THETA_N; i++) {
      const t = theta[i];
      prior[i] = betaPdf(t, alpha, beta);
      // Likelihood proportional to theta^s (1-theta)^(n-s); rescale by max for display
      if (t > 0 && t < 1) {
        const lik = s * Math.log(t) + nMinusS * Math.log(1 - t);
        likelihood[i] = lik;
        if (lik > maxLik) maxLik = lik;
      } else likelihood[i] = -Infinity;
      post[i] = betaPdf(t, updated.alpha, updated.beta);
    }
    // Normalize likelihood for display (shift by max then exponentiate)
    for (let i = 0; i < THETA_N; i++) {
      likelihood[i] = likelihood[i] === -Infinity ? 0 : Math.exp(likelihood[i] - maxLik);
    }
    return {
      prior,
      likelihood,
      post,
      updated,
      mean: betaMean(updated.alpha, updated.beta),
      ci95: betaCentralCI(updated.alpha, updated.beta, 0.95),
    };
  }, [alpha, beta, n, s]);

  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);

      const innerW = W - margin.left - margin.right;
      const panelH = (H - margin.top - margin.bottom) / 3 - 6;

      const x = d3.scaleLinear().domain([0, 0.2]).range([0, innerW]);
      const titles = ['Prior π(θ)', 'Likelihood p(y | θ)  (rescaled)', `Posterior π(θ | y) = Beta(${data.updated.alpha}, ${data.updated.beta})`];
      const colors = [COLOR_VARIANT_A, COLOR_LIKELIHOOD, COLOR_VARIANT_B];
      const datasets = [data.prior, data.likelihood, data.post];

      datasets.forEach((vals, k) => {
        const yTop = margin.top + k * (panelH + 6);
        const { ymax } = panelDensity(vals);
        const y = d3.scaleLinear().domain([0, ymax * 1.1]).range([panelH, 0]);
        const g = svg
          .append('g')
          .attr('transform', `translate(${margin.left}, ${yTop})`);
        g.append('text')
          .attr('x', 0)
          .attr('y', -8)
          .attr('font-size', 11)
          .attr('font-weight', 600)
          .style('fill', 'var(--color-text)')
          .text(titles[k]);
        const area = d3
          .area<number>()
          .x((_, i) => x(theta[i]))
          .y0(panelH)
          .y1((v) => y(v))
          .curve(d3.curveMonotoneX);
        g.append('path')
          .attr('d', area(vals as unknown as number[]) ?? '')
          .attr('fill', colors[k])
          .attr('opacity', 0.35);
        g.append('path')
          .attr(
            'd',
            d3
              .line<number>()
              .x((_, i) => x(theta[i]))
              .y((v) => y(v))
              .curve(d3.curveMonotoneX)(vals as unknown as number[]) ?? '',
          )
          .attr('fill', 'none')
          .attr('stroke', colors[k])
          .attr('stroke-width', 1.8);
        // Vertical guide for posterior mean (panel 3) and 95% CI shading
        if (k === 2) {
          g.append('rect')
            .attr('x', x(data.ci95.lower))
            .attr('width', x(data.ci95.upper) - x(data.ci95.lower))
            .attr('y', 0)
            .attr('height', panelH)
            .attr('fill', COLOR_VARIANT_B)
            .attr('opacity', 0.08);
          g.append('line')
            .attr('x1', x(data.mean))
            .attr('x2', x(data.mean))
            .attr('y1', 0)
            .attr('y2', panelH)
            .attr('stroke', COLOR_VARIANT_B)
            .attr('stroke-width', 1.5)
            .attr('stroke-dasharray', '3 3');
          g.append('text')
            .attr('x', x(data.mean))
            .attr('y', -2)
            .attr('text-anchor', 'middle')
            .attr('font-size', 10)
            .style('fill', COLOR_VARIANT_B)
            .text(`mean = ${data.mean.toFixed(4)}`);
        }
        // axis
        g.append('g')
          .attr('transform', `translate(0, ${panelH})`)
          .call(d3.axisBottom(x).ticks(6).tickSizeOuter(0))
          .selectAll('text')
          .attr('font-size', 9);
      });
    },
    [data, W, H],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={ref} role="img" aria-label="Bayes cycle: prior, likelihood, posterior" />
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, minmax(140px, 1fr))',
          gap: '0.75rem 1.25rem',
          marginTop: '0.5rem',
          fontSize: '0.85rem',
        }}
      >
        <label>
          α (prior pseudo-successes + 1): <strong>{alpha.toFixed(1)}</strong>
          <input
            type="range"
            min={0.5}
            max={50}
            step={0.5}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
            aria-label="prior alpha"
            style={{ width: '100%' }}
          />
        </label>
        <label>
          β (prior pseudo-failures + 1): <strong>{beta.toFixed(1)}</strong>
          <input
            type="range"
            min={0.5}
            max={50}
            step={0.5}
            value={beta}
            onChange={(e) => setBeta(Number(e.target.value))}
            aria-label="prior beta"
            style={{ width: '100%' }}
          />
        </label>
        <label>
          n (trials): <strong>{n}</strong>
          <input
            type="range"
            min={10}
            max={5000}
            step={10}
            value={n}
            onChange={(e) => {
              const newN = Number(e.target.value);
              setN(newN);
              if (s > newN) setS(newN);
            }}
            aria-label="trial count"
            style={{ width: '100%' }}
          />
        </label>
        <label>
          s (successes): <strong>{s}</strong>
          <input
            type="range"
            min={0}
            max={n}
            step={1}
            value={s}
            onChange={(e) => setS(Number(e.target.value))}
            aria-label="success count"
            style={{ width: '100%' }}
          />
        </label>
      </div>
    </div>
  );
}
