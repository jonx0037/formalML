import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  empiricalRademacherMC,
  massartBound,
  mulberry32,
  thresholdClassMatrixMinimal,
} from './shared/generalization-bounds';

// =============================================================================
// RademacherComplexityExplorer — §5.
//
// Two panels:
//   Left  — for a fixed sample (n = 30) and a single sigma, the inner product
//           (1/n) Σ σ_i h_τ(X_i) as a function of τ, with the supremum marked.
//           "Re-roll σ" redraws the labels.
//   Right — Empirical Rademacher complexity vs. n (commit-on-release MC).
//
// All in-browser TS.
// =============================================================================

const HEIGHT = 320;
const SM_BREAKPOINT = 900;
const N_FIXED = 30;
const N_GRID = [20, 50, 100, 200, 500, 1000];

export default function RademacherComplexityExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [sigmaSeed, setSigmaSeed] = useState(20260511);
  const [displayN, setDisplayN] = useState(100);
  const [committedN, setCommittedN] = useState(100);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked ? containerWidth : Math.floor(containerWidth / 2);

  // Left panel: a single (X, sigma) draw, inner-product over the τ grid.
  const leftData = useMemo(() => {
    const rng = mulberry32(sigmaSeed);
    const X = new Float64Array(N_FIXED);
    for (let i = 0; i < N_FIXED; i++) X[i] = rng();
    const sortedX = X.slice().sort();
    const taus = new Float64Array(N_FIXED + 1);
    taus[0] = 0;
    for (let i = 0; i < N_FIXED; i++) taus[i + 1] = sortedX[i];
    const sigma = new Int8Array(N_FIXED);
    for (let i = 0; i < N_FIXED; i++) sigma[i] = rng() < 0.5 ? -1 : 1;
    const innerByTau = new Float64Array(taus.length);
    let supIdx = 0;
    let supVal = -Infinity;
    for (let k = 0; k < taus.length; k++) {
      let acc = 0;
      for (let i = 0; i < N_FIXED; i++) {
        const h = X[i] >= taus[k] ? 1 : -1;
        acc += sigma[i] * h;
      }
      const v = acc / N_FIXED;
      innerByTau[k] = v;
      if (v > supVal) { supVal = v; supIdx = k; }
    }
    return { taus, innerByTau, supTau: taus[supIdx], supVal };
  }, [sigmaSeed]);

  // Right panel: Rademacher complexity at committedN with Massart bound overlay.
  const rightData = useMemo(() => {
    const rng = mulberry32(20260511);
    const points = N_GRID.map((n) => {
      const X = new Float64Array(n);
      for (let i = 0; i < n; i++) X[i] = rng();
      const H = thresholdClassMatrixMinimal(X);
      const est = empiricalRademacherMC(H, n + 1, n, 250, rng);
      return { n, rad: est.mean, se: est.se, massart: massartBound(n + 1, n) };
    });
    // Point at committedN if not already on grid
    if (!N_GRID.includes(committedN)) {
      const X = new Float64Array(committedN);
      for (let i = 0; i < committedN; i++) X[i] = rng();
      const H = thresholdClassMatrixMinimal(X);
      const est = empiricalRademacherMC(H, committedN + 1, committedN, 250, rng);
      points.push({ n: committedN, rad: est.mean, se: est.se, massart: massartBound(committedN + 1, committedN) });
      points.sort((a, b) => a.n - b.n);
    }
    return points;
  }, [committedN]);

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const margin = { top: 32, right: 12, bottom: 40, left: 50 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
      const y = d3.scaleLinear().domain([-1, 1]).range([h, 0]);
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('threshold τ');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -36)
        .attr('text-anchor', 'middle').style('font-size', '11px').style('fill', 'var(--color-text-secondary)')
        .text('(1/n) Σ σᵢ h_τ(Xᵢ)');
      g.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('font-weight', '600').style('fill', 'var(--color-text)')
        .text('Inner product vs τ on one sample of size n=30');

      // Zero line
      g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0)).attr('y2', y(0))
        .style('stroke', '#94a3b8').style('stroke-width', 1);

      const pts = Array.from(leftData.taus).map((t, i) => ({ t, v: leftData.innerByTau[i] }));
      const line = d3.line<{ t: number; v: number }>().x((d) => x(d.t)).y((d) => y(d.v));
      g.append('path').datum(pts).attr('fill', 'none').style('stroke', '#3b82f6').style('stroke-width', 2).attr('d', line);

      // Sup marker
      g.append('circle').attr('cx', x(leftData.supTau)).attr('cy', y(leftData.supVal)).attr('r', 5)
        .style('fill', '#dc2626');
      g.append('text').attr('x', x(leftData.supTau) + 8).attr('y', y(leftData.supVal) - 6)
        .style('font-size', '10px').style('fill', '#dc2626')
        .text(`sup ${leftData.supVal.toFixed(3)} @ τ=${leftData.supTau.toFixed(2)}`);
    },
    [leftData, panelWidth],
  );

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const margin = { top: 32, right: 16, bottom: 40, left: 50 };
      const w = panelWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLog().domain([20, 1000]).range([0, w]);
      const y = d3.scaleLog().domain([0.02, 1]).range([h, 0]);
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5, '~s'));
      g.append('g').call(d3.axisLeft(y).ticks(5, '~g'));
      g.append('text').attr('x', w / 2).attr('y', h + 32).attr('text-anchor', 'middle')
        .style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text('sample size n');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -36)
        .attr('text-anchor', 'middle').style('font-size', '11px').style('fill', 'var(--color-text-secondary)')
        .text('Rademacher complexity');
      g.append('text').attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('font-weight', '600').style('fill', 'var(--color-text)')
        .text('Empirical Rademacher vs Massart bound');

      type Pt = { n: number; rad: number; se: number; massart: number };
      const radLine = d3.line<Pt>().x((d) => x(d.n)).y((d) => y(Math.max(0.02, d.rad)));
      const masLine = d3.line<Pt>().x((d) => x(d.n)).y((d) => y(d.massart));
      g.append('path').datum(rightData).attr('fill', 'none').style('stroke', '#3b82f6').style('stroke-width', 2).attr('d', radLine);
      g.append('path').datum(rightData).attr('fill', 'none').style('stroke', '#ef4444').style('stroke-width', 2).style('stroke-dasharray', '4 3').attr('d', masLine);

      rightData.forEach((p) => {
        g.append('circle').attr('cx', x(p.n)).attr('cy', y(Math.max(0.02, p.rad))).attr('r', 3.5).style('fill', '#3b82f6');
      });

      // Legend
      const lg = g.append('g').attr('transform', `translate(${w - 175},${10})`);
      lg.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 0).attr('y2', 0).style('stroke', '#3b82f6').style('stroke-width', 2);
      lg.append('text').attr('x', 22).attr('y', 4).style('font-size', '11px').style('fill', 'var(--color-text)').text('empirical Rademacher');
      lg.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 20).attr('y2', 20).style('stroke', '#ef4444').style('stroke-width', 2).style('stroke-dasharray', '4 3');
      lg.append('text').attr('x', 22).attr('y', 24).style('font-size', '11px').style('fill', 'var(--color-text)').text('Massart bound');
    },
    [rightData, panelWidth],
  );

  return (
    <figure ref={containerRef} className="my-8 not-prose">
      <div style={{ display: 'grid', gridTemplateColumns: isStacked ? '1fr' : '1fr 1fr', gap: 8 }}>
        <svg ref={leftRef} width={panelWidth || 360} height={HEIGHT} role="img" aria-label="Per-tau inner product on a fixed sample, sup highlighted." />
        <svg ref={rightRef} width={panelWidth || 360} height={HEIGHT} role="img" aria-label="Empirical Rademacher complexity vs sample size." />
      </div>
      <figcaption style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
        <button
          type="button"
          onClick={() => setSigmaSeed((s) => s + 1)}
          style={{
            fontSize: 12, padding: '4px 10px', borderRadius: 6,
            border: '1px solid var(--color-border)',
            background: 'var(--color-surface)', color: 'var(--color-text)',
            cursor: 'pointer',
          }}
        >
          re-roll σ
        </button>
        <label style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          right-panel highlight n:
          <input
            type="range" min={20} max={1000} step={10} value={displayN}
            onChange={(e) => setDisplayN(parseInt(e.target.value, 10))}
            onMouseUp={() => setCommittedN(displayN)}
            onTouchEnd={() => setCommittedN(displayN)}
            onKeyUp={() => setCommittedN(displayN)}
            style={{ marginLeft: 8, verticalAlign: 'middle' }}
            aria-label="highlight sample size"
          />
          <span style={{ marginLeft: 6 }}>{displayN}</span>
        </label>
      </figcaption>
    </figure>
  );
}
