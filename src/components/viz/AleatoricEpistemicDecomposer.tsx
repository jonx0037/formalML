import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bayesPolyPosterior,
  bayesPolyPredict,
  fTrue,
  gaussianFrom,
  linspace,
  mulberry32,
  sigmaTrue,
} from './shared/uncertainty-quantification';

// =============================================================================
// AleatoricEpistemicDecomposer — §2
//
// Closed-form Bayesian polynomial regression on the running heteroscedastic
// toy. Three panels overlaid in one SVG via small-multiples:
//   (a) total ±2σ band on the data, posterior mean, true f(x)
//   (b) aleatoric and epistemic variance curves
// Slider for n; toggle for whether to show training data.
// Static fallback: public/images/topics/uncertainty-quantification/fig_02_decomposition.png
// =============================================================================

const HEIGHT = 460;
const DEGREE = 7;
const X_GRID = linspace(-3.2, 3.2, 220);

function sampleToy(n: number, rng: () => number) {
  const g = gaussianFrom(rng);
  const X: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const xi = -3 + 6 * rng();
    X.push(xi);
    y.push(fTrue(xi) + sigmaTrue(xi) * g());
  }
  return { X, y };
}

export default function AleatoricEpistemicDecomposer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nDisplay, setNDisplay] = useState(200);
  const [nCommitted, setNCommitted] = useState(200);
  const [seedDisplay, setSeedDisplay] = useState(0);
  const [seedCommitted, setSeedCommitted] = useState(0);

  const decomposition = useMemo(() => {
    const rng = mulberry32(20260514 + seedCommitted);
    const { X, y } = sampleToy(nCommitted, rng);
    const sigmaTr = X.map(sigmaTrue);
    const post = bayesPolyPosterior(X, y, DEGREE, sigmaTr);
    const sigmaGrid = X_GRID.map(sigmaTrue);
    const pred = bayesPolyPredict(X_GRID, post, sigmaGrid);
    return { X, y, pred };
  }, [nCommitted, seedCommitted]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 28, right: 24, bottom: 48, left: 56 };
      const panelW = (w - margin.left - margin.right - 32) / 2;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);

      const xScale = d3.scaleLinear().domain([-3.3, 3.3]).range([0, panelW]);
      const { X, y, pred } = decomposition;
      const yMaxBand = Math.max(
        ...pred.fMean.map((m, i) => m + 2 * Math.sqrt(pred.totalVar[i])),
        ...y,
        3,
      );
      const yMinBand = Math.min(
        ...pred.fMean.map((m, i) => m - 2 * Math.sqrt(pred.totalVar[i])),
        ...y,
        -3,
      );
      const yScaleBand = d3.scaleLinear().domain([yMinBand, yMaxBand]).range([H, 0]).nice();

      // ---- Panel (a): data + posterior + band ----
      const gA = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const bandArea = d3.area<number>()
        .x((_, i) => xScale(X_GRID[i]))
        .y0((_, i) => yScaleBand(pred.fMean[i] - 2 * Math.sqrt(pred.totalVar[i])))
        .y1((_, i) => yScaleBand(pred.fMean[i] + 2 * Math.sqrt(pred.totalVar[i])));
      gA.append('path')
        .datum(X_GRID)
        .attr('d', bandArea)
        .style('fill', '#1e293b')
        .style('opacity', 0.18);

      const lineGen = d3.line<number>()
        .x((_, i) => xScale(X_GRID[i]))
        .y((d) => yScaleBand(d));
      gA.append('path').datum(X_GRID.map(fTrue)).attr('d', lineGen).attr('fill', 'none')
        .style('stroke', '#94a3b8').attr('stroke-width', 1.4).attr('stroke-dasharray', '5 3');
      gA.append('path').datum(pred.fMean).attr('d', lineGen).attr('fill', 'none')
        .style('stroke', '#8b5cf6').attr('stroke-width', 2);
      gA.selectAll('.pt').data(X).enter().append('circle').attr('class', 'pt')
        .attr('cx', (d) => xScale(d as number))
        .attr('cy', (_, i) => yScaleBand(y[i]))
        .attr('r', 2.2).style('fill', '#1f2937').style('opacity', 0.45);

      gA.append('g').attr('transform', `translate(0,${H})`)
        .call(d3.axisBottom(xScale).ticks(7))
        .selectAll('text').style('fill', 'var(--color-text)');
      gA.append('g').call(d3.axisLeft(yScaleBand).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text)');
      gA.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      gA.append('text').attr('x', panelW / 2).attr('y', -10)
        .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', 'var(--color-text)')
        .text(`(a) posterior mean ± 2σ total — n = ${nCommitted}`);
      gA.append('text').attr('x', panelW / 2).attr('y', H + 36)
        .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', 'var(--color-text)')
        .text('x');

      // ---- Panel (b): aleatoric vs epistemic vs total ----
      const gB = svg.append('g')
        .attr('transform', `translate(${margin.left + panelW + 32},${margin.top})`);
      const allVar = [...pred.aleatoricVar, ...pred.epistemicVar, ...pred.totalVar];
      const yMaxVar = Math.max(...allVar) * 1.05;
      const yScaleVar = d3.scaleLinear().domain([0, yMaxVar]).range([H, 0]).nice();
      const varLine = (vals: number[]) =>
        d3.line<number>()
          .x((_, i) => xScale(X_GRID[i]))
          .y((d) => yScaleVar(d))(vals);

      gB.append('path').attr('d', varLine(pred.aleatoricVar) || '').attr('fill', 'none')
        .style('stroke', '#0ea5e9').attr('stroke-width', 2);
      gB.append('path').attr('d', varLine(pred.epistemicVar) || '').attr('fill', 'none')
        .style('stroke', '#8b5cf6').attr('stroke-width', 2);
      gB.append('path').attr('d', varLine(pred.totalVar) || '').attr('fill', 'none')
        .style('stroke', '#1e293b').attr('stroke-width', 1.5).attr('stroke-dasharray', '3 3');

      gB.append('g').attr('transform', `translate(0,${H})`)
        .call(d3.axisBottom(xScale).ticks(7))
        .selectAll('text').style('fill', 'var(--color-text)');
      gB.append('g').call(d3.axisLeft(yScaleVar).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text)');
      gB.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      gB.append('text').attr('x', panelW / 2).attr('y', -10)
        .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', 'var(--color-text)')
        .text('(b) variance components');

      // Legend in panel (b).
      const legend = gB.append('g').attr('transform', `translate(${panelW - 110}, 8)`);
      const items = [
        { name: 'aleatoric', color: '#0ea5e9', dash: '' },
        { name: 'epistemic', color: '#8b5cf6', dash: '' },
        { name: 'total = ale + epi', color: '#1e293b', dash: '3 3' },
      ];
      items.forEach((it, i) => {
        legend.append('line').attr('x1', 0).attr('x2', 20).attr('y1', i * 16).attr('y2', i * 16)
          .style('stroke', it.color).attr('stroke-width', 2).attr('stroke-dasharray', it.dash);
        legend.append('text').attr('x', 26).attr('y', i * 16 + 4)
          .style('font-size', '11px').style('fill', 'var(--color-text)').text(it.name);
      });
    },
    [containerWidth, decomposition, nCommitted],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
        aria-label="Aleatoric vs epistemic variance decomposition on the running heteroscedastic toy." />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 220 }}>
          <span>training points n: <strong>{nDisplay}</strong></span>
          <input type="range" min={20} max={500} step={10} value={nDisplay}
            onChange={(e) => setNDisplay(parseInt(e.target.value, 10))}
            onMouseUp={() => setNCommitted(nDisplay)} onTouchEnd={() => setNCommitted(nDisplay)}
            onKeyUp={() => setNCommitted(nDisplay)} aria-label="Training set size n" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 220 }}>
          <span>seed offset: <strong>{seedDisplay}</strong></span>
          <input type="range" min={0} max={20} step={1} value={seedDisplay}
            onChange={(e) => setSeedDisplay(parseInt(e.target.value, 10))}
            onMouseUp={() => setSeedCommitted(seedDisplay)} onTouchEnd={() => setSeedCommitted(seedDisplay)}
            onKeyUp={() => setSeedCommitted(seedDisplay)} aria-label="Random seed offset" />
        </label>
      </div>
      <p style={{ fontSize: '12px', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
        As n grows the epistemic curve (purple) shrinks at rate 1/n by Bernstein–von Mises; the aleatoric
        curve (cyan) — a property of the data-generating process — does not.
      </p>
    </div>
  );
}
