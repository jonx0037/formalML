import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  betaPdf,
  betaBinomialUpdate,
  betaMean,
  COLOR_VARIANT_A,
  COLOR_LIKELIHOOD,
  COLOR_VARIANT_B,
} from './shared/bayesian-inference';

// =============================================================================
// PriorChoiceComparator — §3 interactive.
//
// 3 columns of priors × 3 rows (prior, posterior at small n, posterior at
// large n). Columns: uniform Beta(1,1), weakly informative Beta(2,2),
// strongly (deliberately wrong) informative Beta(20,80) centered at 0.2.
// Slider drives the sample size; success ratio s/n is held near 0.047 to
// match A/B-test variant A.
//
// Pedagogical takeaway: with little data the prior dominates and the three
// posteriors disagree; with a lot of data they converge to the MLE.
// =============================================================================

const THETA_N = 401;
const theta = Float64Array.from({ length: THETA_N }, (_, i) => i / (THETA_N - 1));
const PRIORS: Array<{ name: string; alpha: number; beta: number }> = [
  { name: 'Uniform Beta(1, 1)', alpha: 1, beta: 1 },
  { name: 'Weakly informative Beta(2, 2)', alpha: 2, beta: 2 },
  { name: 'Strongly wrong Beta(20, 80)', alpha: 20, beta: 80 },
];

function density(alpha: number, beta: number): Float64Array {
  const out = new Float64Array(THETA_N);
  for (let i = 0; i < THETA_N; i++) out[i] = betaPdf(theta[i], alpha, beta);
  return out;
}

export default function PriorChoiceComparator() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [n, setN] = useState(1000);
  const s = Math.round(n * 0.047);

  const W = Math.max(360, Math.min(width || 720, 800));
  const H = 480;
  const margin = { top: 28, right: 12, bottom: 28, left: 36 };
  const colW = (W - margin.left - margin.right) / 3 - 12;
  const rowH = (H - margin.top - margin.bottom) / 3 - 12;
  const mle = s / Math.max(n, 1);

  const cells = useMemo(() => {
    const items: { col: number; row: number; alpha: number; beta: number; label: string }[] = [];
    PRIORS.forEach((p, col) => {
      items.push({ col, row: 0, alpha: p.alpha, beta: p.beta, label: p.name });
      const postSmall = betaBinomialUpdate(p.alpha, p.beta, 10, Math.max(0, Math.round(10 * 0.047)));
      items.push({ col, row: 1, alpha: postSmall.alpha, beta: postSmall.beta, label: `n = 10` });
      const postBig = betaBinomialUpdate(p.alpha, p.beta, n, s);
      items.push({ col, row: 2, alpha: postBig.alpha, beta: postBig.beta, label: `n = ${n}` });
    });
    return items;
  }, [n, s]);

  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);

      // Set up a per-row x range — top row priors use [0, 1]; posterior rows zoom to [0, 0.2]
      const xPrior = d3.scaleLinear().domain([0, 1]).range([0, colW]);
      const xPost = d3.scaleLinear().domain([0, 0.2]).range([0, colW]);

      cells.forEach(({ col, row, alpha, beta, label }) => {
        const xLeft = margin.left + col * (colW + 12);
        const yTop = margin.top + row * (rowH + 12);
        const isPrior = row === 0;
        const x = isPrior ? xPrior : xPost;
        const dens = density(alpha, beta);
        let ymax = 0;
        for (let i = 0; i < THETA_N; i++) {
          const t = theta[i];
          if (isPrior ? true : t <= 0.2) {
            if (dens[i] > ymax) ymax = dens[i];
          }
        }
        const y = d3.scaleLinear().domain([0, ymax * 1.1]).range([rowH, 0]);
        const g = svg.append('g').attr('transform', `translate(${xLeft}, ${yTop})`);

        // Title (top row only)
        if (row === 0) {
          g.append('text')
            .attr('x', colW / 2)
            .attr('y', -6)
            .attr('text-anchor', 'middle')
            .attr('font-size', 10)
            .attr('font-weight', 600)
            .style('fill', 'var(--color-text)')
            .text(label);
        } else {
          g.append('text')
            .attr('x', 4)
            .attr('y', 12)
            .attr('font-size', 9)
            .style('fill', 'var(--color-text)')
            .style('opacity', 0.7)
            .text(label);
        }

        // Area
        const area = d3
          .area<number>()
          .x((_, i) => x(theta[i]))
          .y0(rowH)
          .y1((v) => y(v))
          .curve(d3.curveMonotoneX);
        g.append('path')
          .attr('d', area(dens as unknown as number[]) ?? '')
          .attr('fill', col === 0 ? COLOR_VARIANT_A : col === 1 ? COLOR_LIKELIHOOD : COLOR_VARIANT_B)
          .attr('opacity', 0.32);
        g.append('path')
          .attr(
            'd',
            d3
              .line<number>()
              .x((_, i) => x(theta[i]))
              .y((v) => y(v))
              .curve(d3.curveMonotoneX)(dens as unknown as number[]) ?? '',
          )
          .attr('fill', 'none')
          .attr('stroke', col === 0 ? COLOR_VARIANT_A : col === 1 ? COLOR_LIKELIHOOD : COLOR_VARIANT_B)
          .attr('stroke-width', 1.5);

        // Mark posterior mean and MLE on posterior rows
        if (row > 0) {
          const pm = betaMean(alpha, beta);
          g.append('line')
            .attr('x1', x(pm))
            .attr('x2', x(pm))
            .attr('y1', 0)
            .attr('y2', rowH)
            .attr('stroke', 'var(--color-text)')
            .attr('stroke-width', 1)
            .attr('stroke-dasharray', '3 3');
          g.append('line')
            .attr('x1', x(0.047))
            .attr('x2', x(0.047))
            .attr('y1', 0)
            .attr('y2', rowH)
            .attr('stroke', '#888')
            .attr('stroke-width', 1)
            .attr('stroke-dasharray', '2 4');
        }

        // Axis at bottom
        g.append('g')
          .attr('transform', `translate(0, ${rowH})`)
          .call(d3.axisBottom(x).ticks(isPrior ? 5 : 4).tickSizeOuter(0))
          .selectAll('text')
          .attr('font-size', 8);
      });
    },
    [cells, W, H, mle],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={ref} role="img" aria-label="Prior choice comparator: three priors at varying sample sizes" />
      <div style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
        <label>
          n (large-sample row): <strong>{n}</strong> (s = {s}, s/n ≈ {mle.toFixed(3)})
          <input
            type="range"
            min={50}
            max={5000}
            step={50}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            aria-label="large sample size"
            style={{ width: '100%' }}
          />
        </label>
        <p style={{ opacity: 0.7, marginTop: '0.5rem' }}>
          Black dashed line: posterior mean. Gray dotted line: empirical MLE
          {' '}<em>s/n</em> ≈ 0.047. As n grows, the three columns&apos; posteriors converge toward
          the MLE — the prior washes out.
        </p>
      </div>
    </div>
  );
}
