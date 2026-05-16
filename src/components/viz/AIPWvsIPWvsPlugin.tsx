import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  crossFitAipwAte,
  mulberry32,
  paletteSemi,
  sampleAte,
} from './shared/semiparametric-inference';

// =============================================================================
// AIPWvsIPWvsPlugin — §9.1 (ATE under unconfoundedness, n=600)
//
// Three side-by-side histograms across MC replications:
//   - "naive diff-in-means": Ȳ_{D=1} − Ȳ_{D=0} (confounded — biased)
//   - "AIPW (oracle)": truth-substituted AIPW (efficient at the BKRW bound)
//   - "cross-fit DML": K=5 cross-fit AIPW with polynomial-degree-2 nuisance
//
// The brief originally specified a 2×2 (m̂ × π̂) grid of misspecifications;
// that pattern is now realized by the §13.3 MisspecificationStressTest viz.
// This panel focuses on the naive-vs-oracle-AIPW-vs-DML comparison that §9.1
// of the topic motivates. The plug-in OR estimator and standalone IPW would
// land between naive and AIPW; they're omitted here to keep the visual at
// three panels (consistent with the static fallback figure).
// =============================================================================

const HEIGHT_PANEL = 200;
const SM_BREAKPOINT = 640;
const N = 600;
const TRUE_ATE = 1.0;

function naiveDiff(D: Int8Array, Y: Float64Array): number {
  let sum1 = 0, n1 = 0, sum0 = 0, n0 = 0;
  for (let i = 0; i < D.length; i++) {
    if (D[i] === 1) { sum1 += Y[i]; n1 += 1; }
    else { sum0 += Y[i]; n0 += 1; }
  }
  return sum1 / n1 - sum0 / n0;
}

function aipwTruth(
  D: Int8Array,
  Y: Float64Array,
  pi: Float64Array,
  mu0: Float64Array,
  mu1: Float64Array,
): number {
  let s = 0;
  const n = D.length;
  for (let i = 0; i < n; i++) {
    const e = Math.max(Math.min(pi[i], 0.95), 0.05);
    s += mu1[i] - mu0[i] + (D[i] * (Y[i] - mu1[i])) / e - ((1 - D[i]) * (Y[i] - mu0[i])) / (1 - e);
  }
  return s / n;
}

export default function AIPWvsIPWvsPlugin() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayB, setDisplayB] = useState(100);
  const [committedB, setCommittedB] = useState(100);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const data = useMemo(() => {
    const naive = new Float64Array(committedB);
    const aipwOracle = new Float64Array(committedB);
    const dml = new Float64Array(committedB);
    for (let b = 0; b < committedB; b++) {
      const rng = mulberry32(20260515 + b * 17);
      const sample = sampleAte(N, rng);
      naive[b] = naiveDiff(sample.D, sample.Y);
      aipwOracle[b] = aipwTruth(sample.D, sample.Y, sample.piOracle, sample.mu0Oracle, sample.mu1Oracle);
      const dr = crossFitAipwAte(sample.X, sample.D, sample.Y, 5, rng, 2);
      dml[b] = dr.tau;
    }
    return { naive, aipwOracle, dml };
  }, [committedB]);

  const renderHistogram = (
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    panelW: number,
    arr: Float64Array,
    color: string,
    title: string,
  ) => {
    const margin = { top: 26, right: 12, bottom: 30, left: 44 };
    const W = panelW - margin.left - margin.right;
    const H = HEIGHT_PANEL - margin.top - margin.bottom;
    svg.selectAll('*').remove();
    if (W <= 0) return;
    svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT_PANEL}`);
    const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
    const a = Array.from(arr);
    const lo = Math.min(...a, TRUE_ATE - 0.5);
    const hi = Math.max(...a, TRUE_ATE + 0.5);
    const xScale = d3.scaleLinear().domain([lo, hi]).range([0, W]).nice();
    const bins = d3.bin().domain(xScale.domain() as [number, number]).thresholds(18)(a);
    const yMax = d3.max(bins, (d) => d.length)! * 1.1 + 1;
    const yScale = d3.scaleLinear().domain([0, yMax]).range([H, 0]).nice();
    g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(4));
    g.append('g').call(d3.axisLeft(yScale).ticks(3));
    g.selectAll(null).data(bins).join('rect')
      .attr('x', (d) => xScale(d.x0!) + 1)
      .attr('y', (d) => yScale(d.length))
      .attr('width', (d) => Math.max(xScale(d.x1!) - xScale(d.x0!) - 1, 0))
      .attr('height', (d) => H - yScale(d.length))
      .style('fill', color).style('opacity', 0.7);
    g.append('line').attr('x1', xScale(TRUE_ATE)).attr('x2', xScale(TRUE_ATE))
      .attr('y1', 0).attr('y2', H)
      .style('stroke', paletteSemi.theoryLine).style('stroke-width', 1.5).style('stroke-dasharray', '4,3');
    const mu = d3.mean(a)!;
    const sd = d3.deviation(a) ?? 0;
    g.append('text').attr('x', 0).attr('y', -12)
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-mono)').style('font-size', 11)
      .text(`${title}`);
    g.append('text').attr('x', 0).attr('y', 0)
      .style('fill', 'var(--color-text-secondary)')
      .style('font-family', 'var(--font-mono)').style('font-size', 10)
      .text(`mean=${mu.toFixed(3)}, SD=${sd.toFixed(3)}`);
  };

  const ref1 = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w / 2 - 8;
      renderHistogram(svg, panelW, data.naive, paletteSemi.naive, 'Naive diff-in-means');
    },
    [data, containerWidth, isMobile],
  );
  const ref2 = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w / 2 - 8;
      renderHistogram(svg, panelW, data.aipwOracle, paletteSemi.oneStep, 'AIPW (truth-substituted nuisances)');
    },
    [data, containerWidth, isMobile],
  );
  const ref3 = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w / 2 - 8;
      renderHistogram(svg, panelW, data.dml, paletteSemi.dml, 'Cross-fit DML (K=5, poly-degree-2 nuisance)');
    },
    [data, containerWidth, isMobile],
  );

  return (
    <div ref={containerRef} className="viz-container" style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 12 }}>
        <label style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>
          MC replications B: <strong>{displayB}</strong>
          <input
            type="range"
            min={50}
            max={200}
            step={50}
            value={displayB}
            onChange={(e) => setDisplayB(+e.target.value)}
            onMouseUp={() => setCommittedB(displayB)}
            onTouchEnd={() => setCommittedB(displayB)}
            onKeyUp={() => setCommittedB(displayB)}
            aria-label="Monte Carlo replications"
            style={{ width: '100%', marginTop: 4 }}
          />
        </label>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 8 }}>
        <svg ref={ref1} style={{ width: isMobile ? '100%' : '50%', height: HEIGHT_PANEL, display: 'block' }} />
        <svg ref={ref2} style={{ width: isMobile ? '100%' : '50%', height: HEIGHT_PANEL, display: 'block' }} />
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 8, marginTop: 8 }}>
        <svg ref={ref3} style={{ width: isMobile ? '100%' : '50%', height: HEIGHT_PANEL, display: 'block' }} />
        <div style={{ width: isMobile ? '100%' : '50%', padding: '8px 12px', fontSize: 12, lineHeight: 1.5, color: 'var(--color-text-secondary)' }}>
          <strong style={{ color: 'var(--color-text)', display: 'block', marginBottom: 6 }}>What to look for</strong>
          The naive estimator (grey) is biased — the histogram center sits off the true ATE=1 because D and X are confounded. AIPW with truth-substituted nuisances (red) is unbiased and asymptotically efficient. Cross-fit DML with polynomial nuisance (purple) approaches the same efficient distribution as B grows, demonstrating that ML-fitted nuisance clears the rate condition on this smooth DGP.
        </div>
      </div>
    </div>
  );
}
