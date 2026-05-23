import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  normalPdf,
  normalNormalUpdate,
  mulberry32,
  gaussianSample,
  COLOR_VARIANT_A,
  COLOR_VARIANT_B,
  COLOR_LIKELIHOOD,
} from './shared/bayesian-inference';

// =============================================================================
// FrequentistBayesianDialogue — §12 interactive.
//
// Top: 30 repeated 95% confidence intervals from independent samples
// drawn from N(mu_true, 1). Each interval is colored green if it covers
// mu_true and red if it misses. Reader can sweep n (sample size) to see
// width shrink with sqrt(n).
// Bottom: a single observed dataset's posterior density (Normal-Normal
// with weakly informative prior), the 95% credible interval shaded,
// and the corresponding 95% confidence interval drawn beneath.
//
// Pedagogical takeaway: confidence intervals are sampling-distribution
// statements ("intervals at this rate would cover mu_true"); credible
// intervals are posterior statements ("mu_true lies here with this
// probability given the observed data"). Both refer to "95%" but they
// mean different things. Bernstein-von Mises says the two coincide
// asymptotically.
// =============================================================================

const MU_TRUE = 0.5;
const REPS = 30;

interface Replicate {
  yBar: number;
  ciLower: number;
  ciUpper: number;
  covers: boolean;
}

function simulate(n: number, sigma: number, seed: number): { reps: Replicate[]; observed: { yBar: number } } {
  const rng = mulberry32(seed);
  const reps: Replicate[] = [];
  for (let r = 0; r < REPS; r++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += MU_TRUE + sigma * gaussianSample(rng);
    const yBar = sum / n;
    const halfWidth = 1.96 * sigma / Math.sqrt(n);
    const ciLower = yBar - halfWidth;
    const ciUpper = yBar + halfWidth;
    reps.push({ yBar, ciLower, ciUpper, covers: ciLower <= MU_TRUE && MU_TRUE <= ciUpper });
  }
  // First replicate as the "observed" dataset for the bottom panel
  return { reps, observed: { yBar: reps[0].yBar } };
}

export default function FrequentistBayesianDialogue() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [n, setN] = useState(30);
  const [tau0Sq, setTau0Sq] = useState(2);
  const [seed, setSeed] = useState(42);

  const W = Math.max(360, Math.min(width || 720, 800));
  const H = 460;
  const margin = { top: 28, right: 14, bottom: 36, left: 50 };

  const sim = useMemo(() => simulate(n, 1, seed), [n, seed]);
  const post = useMemo(
    () => normalNormalUpdate({ mu0: 0, tau0Sq, sigmaSq: 1, yBar: sim.observed.yBar, n }),
    [tau0Sq, sim.observed.yBar, n],
  );
  const cri95 = useMemo(() => {
    const sd = Math.sqrt(post.tauNSq);
    return { lower: post.muN - 1.96 * sd, upper: post.muN + 1.96 * sd };
  }, [post]);
  const ci95 = useMemo(() => sim.reps[0], [sim.reps]);

  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);

      const innerW = W - margin.left - margin.right;
      const topH = 230;
      const botH = 130;

      // ---------- Top: 30 confidence intervals ----------
      const gTop = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xExtent = d3.extent(sim.reps.flatMap((r) => [r.ciLower, r.ciUpper])) as [number, number];
      const padding = (xExtent[1] - xExtent[0]) * 0.08;
      const x = d3.scaleLinear().domain([xExtent[0] - padding, xExtent[1] + padding]).range([0, innerW]);
      const y = d3.scaleLinear().domain([0, REPS - 1]).range([10, topH - 10]);

      sim.reps.forEach((r, i) => {
        const color = r.covers ? '#22c55e' : '#ef4444';
        gTop.append('line')
          .attr('x1', x(r.ciLower))
          .attr('x2', x(r.ciUpper))
          .attr('y1', y(i))
          .attr('y2', y(i))
          .attr('stroke', color)
          .attr('stroke-width', 1.5);
        gTop.append('circle')
          .attr('cx', x(r.yBar))
          .attr('cy', y(i))
          .attr('r', 2.5)
          .attr('fill', color);
      });
      // True mu line
      gTop.append('line')
        .attr('x1', x(MU_TRUE))
        .attr('x2', x(MU_TRUE))
        .attr('y1', 0)
        .attr('y2', topH)
        .attr('stroke', '#000')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4 3');
      gTop.append('text')
        .attr('x', x(MU_TRUE))
        .attr('y', -4)
        .attr('text-anchor', 'middle')
        .attr('font-size', 10)
        .style('fill', 'var(--color-text)')
        .text(`μ_true = ${MU_TRUE}`);
      gTop.append('g').attr('transform', `translate(0, ${topH})`).call(d3.axisBottom(x).ticks(7));
      gTop.append('text')
        .attr('x', innerW / 2)
        .attr('y', topH + 28)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .style('fill', 'var(--color-text)')
        .text(`${REPS} repeated 95% confidence intervals (green=covers, red=misses)`);
      const coverage = sim.reps.filter((r) => r.covers).length / REPS;
      gTop.append('text')
        .attr('x', innerW - 4)
        .attr('y', topH + 28)
        .attr('text-anchor', 'end')
        .attr('font-size', 10)
        .style('fill', 'var(--color-text)')
        .text(`observed coverage: ${(coverage * 100).toFixed(0)}%`);

      // ---------- Bottom: posterior + credible interval + confidence interval ----------
      const gBot = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top + topH + 60})`);
      const muGrid = d3.range(x.domain()[0], x.domain()[1], (x.domain()[1] - x.domain()[0]) / 400);
      const dens = muGrid.map((mu) => normalPdf(mu, post.muN, Math.sqrt(post.tauNSq)));
      const ymax = Math.max(...dens) * 1.1;
      const yb = d3.scaleLinear().domain([0, ymax]).range([botH, 0]);
      // Credible interval shading
      gBot.append('rect')
        .attr('x', x(cri95.lower))
        .attr('y', 0)
        .attr('width', x(cri95.upper) - x(cri95.lower))
        .attr('height', botH)
        .attr('fill', COLOR_VARIANT_B)
        .attr('opacity', 0.1);
      // Posterior curve
      gBot.append('path')
        .attr(
          'd',
          d3.line<number>().x((_, i) => x(muGrid[i])).y((v) => yb(v))(dens) ?? '',
        )
        .attr('fill', 'none')
        .attr('stroke', COLOR_VARIANT_B)
        .attr('stroke-width', 1.8);
      // Credible interval bar (top)
      gBot.append('line')
        .attr('x1', x(cri95.lower))
        .attr('x2', x(cri95.upper))
        .attr('y1', yb(ymax * 0.95))
        .attr('y2', yb(ymax * 0.95))
        .attr('stroke', COLOR_VARIANT_B)
        .attr('stroke-width', 4);
      gBot.append('text')
        .attr('x', (x(cri95.lower) + x(cri95.upper)) / 2)
        .attr('y', yb(ymax * 0.95) - 4)
        .attr('text-anchor', 'middle')
        .attr('font-size', 9)
        .style('fill', COLOR_VARIANT_B)
        .text(`95% Bayesian credible interval`);
      // Confidence interval bar (bottom)
      gBot.append('line')
        .attr('x1', x(ci95.ciLower))
        .attr('x2', x(ci95.ciUpper))
        .attr('y1', yb(ymax * 0.1))
        .attr('y2', yb(ymax * 0.1))
        .attr('stroke', COLOR_VARIANT_A)
        .attr('stroke-width', 4);
      gBot.append('text')
        .attr('x', (x(ci95.ciLower) + x(ci95.ciUpper)) / 2)
        .attr('y', yb(ymax * 0.1) - 4)
        .attr('text-anchor', 'middle')
        .attr('font-size', 9)
        .style('fill', COLOR_VARIANT_A)
        .text(`95% frequentist confidence interval (1st replicate)`);
      gBot.append('line')
        .attr('x1', x(MU_TRUE))
        .attr('x2', x(MU_TRUE))
        .attr('y1', 0)
        .attr('y2', botH)
        .attr('stroke', '#000')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4 3');
      gBot.append('g').attr('transform', `translate(0, ${botH})`).call(d3.axisBottom(x).ticks(7));
      gBot.append('text')
        .attr('x', innerW / 2)
        .attr('y', botH + 28)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .style('fill', 'var(--color-text)')
        .text(`Single observed dataset: posterior + 95% credible vs 95% confidence interval`);
    },
    [sim, post, cri95, ci95, W, H],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={ref} role="img" aria-label="Frequentist-Bayesian dialogue" />
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, minmax(140px, 1fr))',
          gap: '0.75rem',
          marginTop: '0.5rem',
          fontSize: '0.85rem',
        }}
      >
        <label>
          n: <strong>{n}</strong>
          <input
            type="range"
            min={5}
            max={500}
            step={1}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            aria-label="sample size"
            style={{ width: '100%' }}
          />
        </label>
        <label>
          τ₀² (prior var): <strong>{tau0Sq.toFixed(2)}</strong>
          <input
            type="range"
            min={0.1}
            max={10}
            step={0.05}
            value={tau0Sq}
            onChange={(e) => setTau0Sq(Number(e.target.value))}
            aria-label="prior variance"
            style={{ width: '100%' }}
          />
        </label>
        <button
          onClick={() => setSeed((s) => s + 1)}
          style={{
            padding: '0.3rem 0.75rem',
            border: '1px solid var(--color-text)',
            borderRadius: 4,
            background: 'transparent',
            color: 'var(--color-text)',
            cursor: 'pointer',
            alignSelf: 'end',
          }}
        >
          New replicates
        </button>
      </div>
      <p style={{ opacity: 0.7, marginTop: '0.5rem', fontSize: '0.85rem' }}>
        Bernstein-von Mises: as n → ∞, the credible-interval (top, red) and
        confidence-interval (top, blue) widths converge to 2 × 1.96 × σ/√n
        and their centers coincide.
      </p>
    </div>
  );
}
