import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  canonicalBound,
  empiricalRademacherMC,
  mulberry32,
  sampleThresholdProblem,
  thresholdClassMatrixMinimal,
  trueRiskThreshold,
} from './shared/generalization-bounds';

// =============================================================================
// UniformConvergenceBound — §7.  Cor 3 bound vs the empirical worst-case gap
// (across the threshold class on a noisy problem), as a function of n.
//
// Commit-on-release.
// =============================================================================

const HEIGHT = 320;
const N_GRID = [30, 100, 300, 1000, 3000];
const B_REPL = 60;
const TAU_STAR = 0.5;

export default function UniformConvergenceBound() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayDelta, setDisplayDelta] = useState(0.05);
  const [committedDelta, setCommittedDelta] = useState(0.05);
  const [displayEta, setDisplayEta] = useState(0.10);
  const [committedEta, setCommittedEta] = useState(0.10);

  const data = useMemo(() => {
    const rng = mulberry32(20260511);
    return N_GRID.map((n) => {
      // Estimate Rademacher of the threshold class on a representative sample of size n.
      const Xrep = new Float64Array(n);
      for (let i = 0; i < n; i++) Xrep[i] = rng();
      const H = thresholdClassMatrixMinimal(Xrep);
      const radEst = empiricalRademacherMC(H, n + 1, n, 200, rng);
      const bound = canonicalBound(radEst.mean, n, committedDelta);

      // Empirical worst-case gap across the threshold class — O(B·n log n) via
      // a single sort plus a left-to-right sweep over candidate thresholds
      // (the naive double-loop is O(B·n²); at n=3000, B=60 that's ~5×10⁸
      // iterations per useMemo, which freezes the UI on slider release).
      //
      // After sorting X, the candidate thresholds (one per gap between sorted
      // points) split the sample into "predict 0" (indices < k) and "predict 1"
      // (indices ≥ k).  As tau increases, exactly one point moves from the
      // right partition to the left at each step, so empirical risk updates
      // by ±1/n based on that point's true label.  True risk has a closed form
      // we already use.
      const gapArr = new Float64Array(B_REPL);
      for (let b = 0; b < B_REPL; b++) {
        const { X, Y } = sampleThresholdProblem(n, committedEta, TAU_STAR, rng);
        // Sort indices by X ascending; Ys[order[i]] is the label at the i-th sorted X.
        const order = new Uint32Array(n);
        for (let i = 0; i < n; i++) order[i] = i;
        const sortedIdx = order.slice().sort((a, b_) => X[a] - X[b_]);
        const sortedX = new Float64Array(n);
        const sortedY = new Int8Array(n);
        for (let i = 0; i < n; i++) {
          sortedX[i] = X[sortedIdx[i]];
          sortedY[i] = Y[sortedIdx[i]];
        }
        // Start with tau = 0: every point predicts 1.  Empirical errors = #{Y_i = 0}.
        let totalOnes = 0;
        for (let i = 0; i < n; i++) totalOnes += sortedY[i];
        const totalZeros = n - totalOnes;
        let errs = totalZeros;  // initial errors at tau = 0
        let maxGap = Math.abs(trueRiskThreshold(0, committedEta, TAU_STAR) - errs / n);
        // Walk tau through each sorted X: at step k, sortedX[k-1] flips from "right" to "left".
        // If sortedY[k-1] == 1, this used to be correct (pred 1, label 1) and is now wrong (pred 0, label 1) → +1 error.
        // If sortedY[k-1] == 0, this used to be wrong and is now correct → -1 error.
        for (let k = 1; k <= n; k++) {
          errs += sortedY[k - 1] === 1 ? 1 : -1;
          const empR = errs / n;
          const truR = trueRiskThreshold(sortedX[k - 1], committedEta, TAU_STAR);
          const g = Math.abs(truR - empR);
          if (g > maxGap) maxGap = g;
        }
        gapArr[b] = maxGap;
      }
      const empMean = gapArr.reduce((s, x) => s + x, 0) / gapArr.length;
      const sortedGaps = Array.from(gapArr).sort((a, b) => a - b);
      const q95 = sortedGaps[Math.min(B_REPL - 1, Math.floor(0.95 * B_REPL))];
      return { n, bound, empMean, q95, rad: radEst.mean };
    });
  }, [committedDelta, committedEta]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;
      const margin = { top: 32, right: 16, bottom: 44, left: 50 };
      const w = containerWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLog().domain([30, 3000]).range([0, w]);
      const y = d3.scaleLinear().domain([0, Math.max(0.6, (d3.max(data, (d) => d.bound) ?? 1) * 1.05)]).range([h, 0]);
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5, '~s'));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('sample size n');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -36).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)')
        .text('sup_h |R(h) − R̂_S(h)| / bound');

      type Pt = { n: number; bound: number; empMean: number; q95: number; rad: number };
      const empLine = d3.line<Pt>().x((d) => x(d.n)).y((d) => y(d.empMean));
      const q95Line = d3.line<Pt>().x((d) => x(d.n)).y((d) => y(d.q95));
      const boundLine = d3.line<Pt>().x((d) => x(d.n)).y((d) => y(d.bound));

      g.append('path').datum(data).attr('fill', 'none').style('stroke', '#3b82f6').style('stroke-width', 2).attr('d', empLine);
      g.append('path').datum(data).attr('fill', 'none').style('stroke', '#1e40af').style('stroke-width', 1.5).style('stroke-dasharray', '4 3').attr('d', q95Line);
      g.append('path').datum(data).attr('fill', 'none').style('stroke', '#ef4444').style('stroke-width', 2).attr('d', boundLine);

      data.forEach((d) => {
        g.append('circle').attr('cx', x(d.n)).attr('cy', y(d.empMean)).attr('r', 3.5).style('fill', '#3b82f6');
        g.append('circle').attr('cx', x(d.n)).attr('cy', y(d.bound)).attr('r', 3.5).style('fill', '#ef4444');
      });

      const lg = g.append('g').attr('transform', `translate(${w - 220},${10})`);
      lg.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 0).attr('y2', 0).style('stroke', '#3b82f6').style('stroke-width', 2);
      lg.append('text').attr('x', 22).attr('y', 4).style('font-size', '11px').style('fill', 'var(--color-text)').text('empirical mean gap');
      lg.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 18).attr('y2', 18).style('stroke', '#1e40af').style('stroke-width', 1.5).style('stroke-dasharray', '4 3');
      lg.append('text').attr('x', 22).attr('y', 22).style('font-size', '11px').style('fill', 'var(--color-text)').text('95th percentile gap');
      lg.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 36).attr('y2', 36).style('stroke', '#ef4444').style('stroke-width', 2);
      lg.append('text').attr('x', 22).attr('y', 40).style('font-size', '11px').style('fill', 'var(--color-text)').text(`Cor 3 bound at δ=${committedDelta}`);
    },
    [data, containerWidth, committedDelta],
  );

  return (
    <figure ref={containerRef} className="my-8 not-prose">
      <svg ref={svgRef} width={containerWidth || 720} height={HEIGHT} role="img" aria-label="Cor 3 bound vs empirical worst-case gap." />
      <figcaption style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          confidence δ:
          <input
            type="range" min={0.005} max={0.3} step={0.005} value={displayDelta}
            onChange={(e) => setDisplayDelta(parseFloat(e.target.value))}
            onMouseUp={() => setCommittedDelta(displayDelta)}
            onTouchEnd={() => setCommittedDelta(displayDelta)}
            onKeyUp={() => setCommittedDelta(displayDelta)}
            style={{ marginLeft: 8, verticalAlign: 'middle' }}
            aria-label="confidence delta"
          />
          <span style={{ marginLeft: 6 }}>{displayDelta.toFixed(3)}</span>
        </label>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          label noise η:
          <input
            type="range" min={0} max={0.3} step={0.01} value={displayEta}
            onChange={(e) => setDisplayEta(parseFloat(e.target.value))}
            onMouseUp={() => setCommittedEta(displayEta)}
            onTouchEnd={() => setCommittedEta(displayEta)}
            onKeyUp={() => setCommittedEta(displayEta)}
            style={{ marginLeft: 8, verticalAlign: 'middle' }}
            aria-label="label noise eta"
          />
          <span style={{ marginLeft: 6 }}>{displayEta.toFixed(2)}</span>
        </label>
      </figcaption>
    </figure>
  );
}
