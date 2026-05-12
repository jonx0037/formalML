import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  ermThreshold,
  mulberry32,
  sampleThresholdProblem,
  trueRiskThreshold,
} from './shared/generalization-bounds';

// =============================================================================
// GeneralizationGapDemo — §2.  Empirical-vs-true risk of the threshold ERM
// across a log-spaced grid of n, with shaded confidence bands from B replicates.
// Closed-form true risk; light Monte Carlo (~0.5 s) on each (eta, seed) change.
// =============================================================================

const HEIGHT = 320;
const N_GRID = [10, 20, 50, 100, 200, 500, 1000, 2000];
const B_REPLICATES = 80;
const TAU_STAR = 0.5;

export default function GeneralizationGapDemo() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayEta, setDisplayEta] = useState(0.10);
  const [committedEta, setCommittedEta] = useState(0.10);
  const [seed, setSeed] = useState(20260511);

  const data = useMemo(() => {
    const rng = mulberry32(seed);
    const empMean: number[] = [];
    const empStd: number[] = [];
    const truMean: number[] = [];
    const truStd: number[] = [];
    for (const n of N_GRID) {
      const empArr = new Float64Array(B_REPLICATES);
      const truArr = new Float64Array(B_REPLICATES);
      for (let b = 0; b < B_REPLICATES; b++) {
        const { X, Y } = sampleThresholdProblem(n, committedEta, TAU_STAR, rng);
        const tauHat = ermThreshold(X, Y);
        let emp = 0;
        for (let i = 0; i < n; i++) {
          const pred = X[i] >= tauHat ? 1 : 0;
          if (pred !== Y[i]) emp++;
        }
        empArr[b] = emp / n;
        truArr[b] = trueRiskThreshold(tauHat, committedEta, TAU_STAR);
      }
      const mean = (a: Float64Array) => a.reduce((s, x) => s + x, 0) / a.length;
      const std = (a: Float64Array, m: number) =>
        Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length);
      const eM = mean(empArr); const tM = mean(truArr);
      empMean.push(eM); empStd.push(std(empArr, eM));
      truMean.push(tM); truStd.push(std(truArr, tM));
    }
    return { empMean, empStd, truMean, truStd };
  }, [committedEta, seed]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;
      const margin = { top: 32, right: 90, bottom: 44, left: 56 };
      const w = containerWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLog().domain([10, 2000]).range([0, w]);
      const y = d3.scaleLinear().domain([0, Math.max(0.5, committedEta + 0.4)]).range([h, 0]);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5, '~s'));
      g.append('g').call(d3.axisLeft(y).ticks(6));
      g.append('text').attr('x', w / 2).attr('y', h + 34).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('sample size n');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -42).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('risk');

      // Bayes risk reference
      g.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(committedEta)).attr('y2', y(committedEta))
        .style('stroke', '#94a3b8').style('stroke-dasharray', '4 2');
      g.append('text').attr('x', w + 4).attr('y', y(committedEta) + 4)
        .style('font-size', '10px').style('fill', '#475569').text(`Bayes η=${committedEta.toFixed(2)}`);

      type Pt = { n: number; mean: number; std: number };
      const empData: Pt[] = N_GRID.map((n, i) => ({ n, mean: data.empMean[i], std: data.empStd[i] }));
      const truData: Pt[] = N_GRID.map((n, i) => ({ n, mean: data.truMean[i], std: data.truStd[i] }));

      const area = d3.area<Pt>()
        .x((d) => x(d.n))
        .y0((d) => y(Math.max(0, d.mean - d.std)))
        .y1((d) => y(d.mean + d.std));
      const line = d3.line<Pt>().x((d) => x(d.n)).y((d) => y(d.mean));

      g.append('path').datum(empData).attr('fill', '#3b82f6').attr('fill-opacity', 0.15).attr('d', area);
      g.append('path').datum(empData).attr('fill', 'none').style('stroke', '#3b82f6').style('stroke-width', 2).attr('d', line);

      g.append('path').datum(truData).attr('fill', '#ef4444').attr('fill-opacity', 0.12).attr('d', area);
      g.append('path').datum(truData).attr('fill', 'none').style('stroke', '#ef4444').style('stroke-width', 2).attr('d', line);

      // Legend
      const lg = g.append('g').attr('transform', `translate(${w - 220},${10})`);
      lg.append('line').attr('x1', 0).attr('x2', 20).attr('y1', 0).attr('y2', 0).style('stroke', '#3b82f6').style('stroke-width', 2);
      lg.append('text').attr('x', 24).attr('y', 4).style('font-size', '11px').style('fill', 'var(--color-text)').text('empirical risk');
      lg.append('line').attr('x1', 0).attr('x2', 20).attr('y1', 20).attr('y2', 20).style('stroke', '#ef4444').style('stroke-width', 2);
      lg.append('text').attr('x', 24).attr('y', 24).style('font-size', '11px').style('fill', 'var(--color-text)').text('true risk');
    },
    [data, containerWidth, committedEta],
  );

  return (
    <figure ref={containerRef} className="my-8 not-prose">
      <svg ref={svgRef} width={containerWidth || 720} height={HEIGHT} role="img" aria-label="Empirical vs true risk of threshold ERM across sample sizes." />
      <figcaption style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          label-flip rate η:
          <input
            type="range" min={0} max={0.30} step={0.01} value={displayEta}
            onChange={(e) => setDisplayEta(parseFloat(e.target.value))}
            onMouseUp={() => setCommittedEta(displayEta)}
            onTouchEnd={() => setCommittedEta(displayEta)}
            onKeyUp={() => setCommittedEta(displayEta)}
            style={{ marginLeft: 8, verticalAlign: 'middle' }}
            aria-label="label flip rate eta"
          />
          <span style={{ marginLeft: 6 }}>{displayEta.toFixed(2)}</span>
        </label>
        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          style={{
            fontSize: 12, padding: '4px 10px', borderRadius: 6,
            border: '1px solid var(--color-border)',
            background: 'var(--color-surface)', color: 'var(--color-text)',
            cursor: 'pointer',
          }}
        >
          re-roll replicates
        </button>
      </figcaption>
    </figure>
  );
}
