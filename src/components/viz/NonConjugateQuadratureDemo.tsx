import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  studentTLogPdf,
  normalLogPdf,
  normalPdf,
  normalNormalUpdate,
  gridPosterior,
  COLOR_VARIANT_A,
  COLOR_VARIANT_B,
  COLOR_LIKELIHOOD,
  COLOR_ACCENT,
} from './shared/bayesian-inference';

// =============================================================================
// NonConjugateQuadratureDemo — §11 interactive.
//
// Student-t_3(0, 1) prior + Normal(μ, 1) likelihood. The Student-t prior
// has heavier tails than the Gaussian, breaking conjugacy. The notebook
// uses scipy.integrate.quad; we use the in-browser grid quadrature in
// shared/bayesian-inference.ts.
//
// Three curves on a shared μ-axis:
// - Gaussian-prior posterior (closed form via normalNormalUpdate)
// - Student-t-prior posterior (grid quadrature)
// - Likelihood (data alone — the MLE-centered Gaussian)
//
// Pedagogical takeaway: when the data lands far from the prior mean (e.g.
// ȳ = 2 with prior centered at 0), the heavy-tailed Student-t prior lets
// the data win more decisively than the Gaussian-prior posterior.
// =============================================================================

export default function NonConjugateQuadratureDemo() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [yBar, setYBar] = useState(2);
  const [n, setN] = useState(20);
  const [nu, setNu] = useState(3);

  const W = Math.max(360, Math.min(width || 720, 760));
  const H = 380;
  const margin = { top: 28, right: 14, bottom: 36, left: 50 };

  const data = useMemo(() => {
    const sigmaSq = 1;
    const tau0Sq = 1;
    const gaussianPost = normalNormalUpdate({ mu0: 0, tau0Sq, sigmaSq, yBar, n });

    // Grid posterior for Student-t prior + Normal likelihood
    const logLikelihoodAtMu = (mu: number): number => {
      // sum over n observations with mean ybar — sufficient statistic gives
      // log lik = -n*(mu - ybar)^2/(2*sigma^2) (constants absorbed)
      return (-n * (mu - yBar) * (mu - yBar)) / (2 * sigmaSq);
    };
    const studentTPost = gridPosterior({
      logUnnormalized: (mu) => studentTLogPdf(mu, nu, 0, 1) + logLikelihoodAtMu(mu),
      gridStart: -5,
      gridEnd: 8,
      nGrid: 2048,
    });

    // Build grid for plotting
    const muGrid = d3.range(-3, 5.01, 0.02);
    const gaussianPdf = muGrid.map((mu) =>
      normalPdf(mu, gaussianPost.muN, Math.sqrt(gaussianPost.tauNSq)),
    );
    // Interpolate Student-t posterior to muGrid
    const studentTPdf = muGrid.map((mu) => {
      // Linear search (cheap — 401 points, 2048 grid)
      const grid = studentTPost.thetaGrid;
      let i = 0;
      while (i < grid.length - 1 && grid[i + 1] < mu) i++;
      if (i >= grid.length - 1) return 0;
      const w = (mu - grid[i]) / (grid[i + 1] - grid[i]);
      return studentTPost.density[i] * (1 - w) + studentTPost.density[i + 1] * w;
    });
    const likelihoodPdf = muGrid.map((mu) =>
      Math.exp(logLikelihoodAtMu(mu) + n * yBar * yBar / (2 * sigmaSq)),
    );
    // Normalize likelihood for display
    const likMax = Math.max(...likelihoodPdf);
    const likPdfNormalized = likelihoodPdf.map((p) => p / likMax * Math.max(...studentTPdf, ...gaussianPdf));

    return {
      muGrid,
      gaussianPdf,
      studentTPdf,
      likPdfNormalized,
      gaussianPost,
      studentTPost,
    };
  }, [yBar, n, nu]);

  const ref = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      svg.attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);

      const innerW = W - margin.left - margin.right;
      const innerH = H - margin.top - margin.bottom;
      const x = d3.scaleLinear().domain([-3, 5]).range([0, innerW]);
      const ymax = Math.max(...data.gaussianPdf, ...data.studentTPdf) * 1.1;
      const y = d3.scaleLinear().domain([0, ymax]).range([innerH, 0]);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

      // Likelihood (dashed reference)
      g.append('path')
        .attr(
          'd',
          d3.line<number>().x((_, i) => x(data.muGrid[i])).y((v) => y(v))(data.likPdfNormalized) ?? '',
        )
        .attr('fill', 'none')
        .attr('stroke', COLOR_LIKELIHOOD)
        .attr('stroke-width', 1.4)
        .attr('stroke-dasharray', '4 3');

      // Gaussian-prior posterior
      g.append('path')
        .attr(
          'd',
          d3.line<number>().x((_, i) => x(data.muGrid[i])).y((v) => y(v))(data.gaussianPdf) ?? '',
        )
        .attr('fill', 'none')
        .attr('stroke', COLOR_VARIANT_A)
        .attr('stroke-width', 1.8);

      // Student-t-prior posterior
      g.append('path')
        .attr(
          'd',
          d3.line<number>().x((_, i) => x(data.muGrid[i])).y((v) => y(v))(data.studentTPdf) ?? '',
        )
        .attr('fill', 'none')
        .attr('stroke', COLOR_ACCENT)
        .attr('stroke-width', 1.8);

      // Markers
      g.append('line')
        .attr('x1', x(yBar))
        .attr('x2', x(yBar))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', '#000')
        .attr('stroke-width', 0.8)
        .attr('stroke-dasharray', '2 3');
      g.append('text')
        .attr('x', x(yBar))
        .attr('y', -2)
        .attr('text-anchor', 'middle')
        .attr('font-size', 10)
        .style('fill', 'var(--color-text)')
        .text(`ȳ = ${yBar.toFixed(2)}`);

      // Axes
      g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(x).ticks(7).tickSizeOuter(0));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 28)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .style('fill', 'var(--color-text)')
        .text('μ');

      // Legend
      const legend = g.append('g').attr('transform', `translate(${innerW - 220}, 4)`);
      const items = [
        { label: 'Likelihood (data alone)', color: COLOR_LIKELIHOOD, dash: '4 3' },
        { label: 'Gaussian-prior posterior', color: COLOR_VARIANT_A, dash: '' },
        { label: `Student-t₍${nu}₎-prior posterior`, color: COLOR_ACCENT, dash: '' },
      ];
      items.forEach((it, i) => {
        legend.append('line')
          .attr('x1', 0)
          .attr('x2', 16)
          .attr('y1', i * 14 + 6)
          .attr('y2', i * 14 + 6)
          .attr('stroke', it.color)
          .attr('stroke-width', 1.8)
          .attr('stroke-dasharray', it.dash || 'none');
        legend.append('text')
          .attr('x', 20)
          .attr('y', i * 14 + 9)
          .attr('font-size', 9)
          .style('fill', 'var(--color-text)')
          .text(it.label);
      });
    },
    [data, W, H, nu],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={ref} role="img" aria-label="Student-t prior non-conjugate quadrature demo" />
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
          ȳ (data mean): <strong>{yBar.toFixed(2)}</strong>
          <input
            type="range"
            min={0}
            max={4}
            step={0.05}
            value={yBar}
            onChange={(e) => setYBar(Number(e.target.value))}
            aria-label="data mean"
            style={{ width: '100%' }}
          />
        </label>
        <label>
          n: <strong>{n}</strong>
          <input
            type="range"
            min={5}
            max={100}
            step={1}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            aria-label="sample size"
            style={{ width: '100%' }}
          />
        </label>
        <label>
          ν (Student-t dof): <strong>{nu}</strong>
          <input
            type="range"
            min={1}
            max={30}
            step={1}
            value={nu}
            onChange={(e) => setNu(Number(e.target.value))}
            aria-label="degrees of freedom"
            style={{ width: '100%' }}
          />
        </label>
      </div>
      <p style={{ opacity: 0.7, marginTop: '0.5rem', fontSize: '0.85rem' }}>
        Student-t posterior via 2048-point grid quadrature. Gaussian-prior mean ={' '}
        <strong>{data.gaussianPost.muN.toFixed(3)}</strong>, Student-t-prior mean ={' '}
        <strong>{data.studentTPost.mean.toFixed(3)}</strong>. Heavier-tailed prior pulls less; data wins.
      </p>
    </div>
  );
}
