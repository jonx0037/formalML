import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  ipwMean,
  mean,
  mulberry32,
  oneStepMar,
  paletteSemi,
  sampleMar,
  stddev,
} from './shared/semiparametric-inference';

// =============================================================================
// InfluenceFunctionVisualizer — §3
// Left panel: per-observation EIF values φ*(O_i) sorted by index.
// Right panel: cumulative mean (1/k)Σ_{i≤k} φ*(O_i) with ±1.96·σ_φ/√k CLT bands.
// Toggle compares EIF (AIPW) vs IPW influence function (RY/π − ψ) for variance.
// =============================================================================

const HEIGHT = 320;
const SM_BREAKPOINT = 640;

export default function InfluenceFunctionVisualizer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayN, setDisplayN] = useState(1000);
  const [committedN, setCommittedN] = useState(1000);
  const [showIPW, setShowIPW] = useState(false);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const data = useMemo(() => {
    const rng = mulberry32(20260515);
    const sample = sampleMar(committedN, rng);
    const { psi, eif } = oneStepMar(sample.R, sample.Y, sample.mOracle, sample.piOracle);
    // IPW influence function: RY/π − ψ_IPW (non-efficient).
    const psiIpw = ipwMean(sample.R, sample.Y, sample.piOracle);
    const eifIpw = new Float64Array(sample.n);
    for (let i = 0; i < sample.n; i++) {
      const pi = Math.max(sample.piOracle[i], 1e-6);
      eifIpw[i] = (sample.R[i] * sample.Y[i]) / pi - psiIpw;
    }
    return { psi, eif, eifIpw };
  }, [committedN]);

  const eifArr = showIPW ? data.eifIpw : data.eif;
  const eifSd = stddev(eifArr);
  const eifMean = mean(eifArr);
  const label = showIPW ? 'IPW IF' : 'AIPW EIF';
  const color = showIPW ? paletteSemi.ipw : paletteSemi.oneStep;

  const refL = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w / 2 - 8;
      const margin = { top: 16, right: 14, bottom: 36, left: 50 };
      const W = panelW - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const arr = Array.from(eifArr);
      const xScale = d3.scaleLinear().domain([0, arr.length - 1]).range([0, W]);
      const ext = d3.extent(arr) as [number, number];
      const padded: [number, number] = [
        Math.min(ext[0], -3 * eifSd),
        Math.max(ext[1], 3 * eifSd),
      ];
      const yScale = d3.scaleLinear().domain(padded).range([H, 0]).nice();
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(5));
      g.append('g').call(d3.axisLeft(yScale).ticks(5));
      g.append('line').attr('x1', 0).attr('x2', W).attr('y1', yScale(0)).attr('y2', yScale(0))
        .style('stroke', 'var(--color-border)').style('stroke-width', 1);
      g.selectAll('circle.eif').data(arr).join('circle').attr('class', 'eif')
        .attr('cx', (_d, i) => xScale(i))
        .attr('cy', (d) => yScale(d))
        .attr('r', 1.6)
        .style('fill', color)
        .style('opacity', 0.55);
      g.append('text').attr('x', 0).attr('y', -4)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-mono)')
        .style('font-size', 12)
        .text(`Per-observation ${label}: mean = ${eifMean.toExponential(2)}, SD = ${eifSd.toFixed(3)}`);
      g.append('text').attr('x', W / 2).attr('y', H + 28).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text('observation index i');
    },
    [eifArr, containerWidth, isMobile, color, eifSd, eifMean, label],
  );

  const refR = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const panelW = isMobile ? w : w / 2 - 8;
      const margin = { top: 16, right: 14, bottom: 36, left: 50 };
      const W = panelW - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${panelW} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const arr = Array.from(eifArr);
      const n = arr.length;
      const cum = new Array<number>(n);
      let s = 0;
      for (let i = 0; i < n; i++) {
        s += arr[i];
        cum[i] = s / (i + 1);
      }
      const xScale = d3.scaleLinear().domain([1, n]).range([0, W]);
      // Compute the maximum band width across i to set the y-scale.
      const bandWidthMax = 1.96 * eifSd / Math.sqrt(1);
      const yMax = Math.max(Math.max(...cum.map(Math.abs)), bandWidthMax);
      const yScale = d3.scaleLinear().domain([-yMax * 1.05, yMax * 1.05]).range([H, 0]).nice();
      g.append('g').attr('transform', `translate(0, ${H})`).call(d3.axisBottom(xScale).ticks(5));
      g.append('g').call(d3.axisLeft(yScale).ticks(5));
      // CLT bands: ±1.96·σ_φ/√k.
      const bandData = d3.range(1, n + 1).map((k) => ({
        k,
        upper: 1.96 * eifSd / Math.sqrt(k),
        lower: -1.96 * eifSd / Math.sqrt(k),
      }));
      const areaGen = d3.area<{ k: number; upper: number; lower: number }>()
        .x((d) => xScale(d.k))
        .y0((d) => yScale(d.lower))
        .y1((d) => yScale(d.upper));
      g.append('path').datum(bandData).attr('d', areaGen)
        .style('fill', paletteSemi.ciBand).style('opacity', 0.35);
      // Zero line.
      g.append('line').attr('x1', 0).attr('x2', W).attr('y1', yScale(0)).attr('y2', yScale(0))
        .style('stroke', paletteSemi.theoryLine).style('stroke-width', 1);
      // Cumulative-mean line.
      const lineGen = d3.line<number>().x((_d, i) => xScale(i + 1)).y((d) => yScale(d));
      g.append('path').datum(cum).attr('d', lineGen)
        .style('fill', 'none').style('stroke', color).style('stroke-width', 1.8);
      g.append('text').attr('x', 0).attr('y', -4)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-mono)').style('font-size', 12)
        .text(`Cumulative mean (1/k) Σ ${label}_i — converges to 0 inside ±1.96σ/√k bands`);
      g.append('text').attr('x', W / 2).attr('y', H + 28).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-family', 'var(--font-mono)').style('font-size', 11)
        .text('sample size k');
    },
    [eifArr, containerWidth, isMobile, color, eifSd, label],
  );

  return (
    <div ref={containerRef} className="viz-container" style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 12 }}>
        <label style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>
          Sample size n: <strong>{displayN}</strong>
          <input
            type="range"
            min={50}
            max={3000}
            step={50}
            value={displayN}
            onChange={(e) => setDisplayN(+e.target.value)}
            onMouseUp={() => setCommittedN(displayN)}
            onTouchEnd={() => setCommittedN(displayN)}
            onKeyUp={() => setCommittedN(displayN)}
            aria-label="Sample size"
            style={{ width: '100%', marginTop: 4 }}
          />
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontFamily: 'var(--font-mono)', fontSize: 13 }}>
          <input type="checkbox" checked={showIPW} onChange={(e) => setShowIPW(e.target.checked)} />
          Compare against non-efficient IPW influence function (RY/π − ψ_IPW)
        </label>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 8 }}>
        <svg ref={refL} style={{ width: isMobile ? '100%' : '50%', height: HEIGHT, display: 'block' }} />
        <svg ref={refR} style={{ width: isMobile ? '100%' : '50%', height: HEIGHT, display: 'block' }} />
      </div>
    </div>
  );
}
