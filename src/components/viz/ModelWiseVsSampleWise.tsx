import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  excessRiskSweep,
  gaussianRng,
  hastieRisk,
  mulberry32,
  sampleSphereBetaStar,
} from './shared/double-descent';

// =============================================================================
// ModelWiseVsSampleWise — §7.3 (Figure 8)
//
// Two-panel side-by-side: (a) model-wise sweep p at fixed n; (b) sample-wise
// sweep n at fixed p. Both overlay the Hastie 2022 analytic curve.
// Static fallback: 08_modelwise_vs_samplewise.png
// =============================================================================

const HEIGHT = 480;

export default function ModelWiseVsSampleWise() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [snrLog, setSnrLog] = useState(0);
  const [snrLogCommitted, setSnrLogCommitted] = useState(0);
  const [nFixed, setNFixed] = useState(50);
  const [nFixedCommitted, setNFixedCommitted] = useState(50);
  const [pFixed, setPFixed] = useState(50);
  const [pFixedCommitted, setPFixedCommitted] = useState(50);
  const [showAnalytic, setShowAnalytic] = useState(true);

  // Model-wise: fix n, sweep p
  const modelWise = useMemo(() => {
    const snr = Math.pow(10, snrLogCommitted);
    const sigma = 1;
    const r = Math.sqrt(snr);
    const n = nFixedCommitted;
    const pMax = 200;
    const pGrid = new Int32Array(Array.from({ length: 41 }, (_, i) => Math.max(1, Math.round((i / 40) * pMax))));
    const rng = mulberry32(2024);
    const gauss = gaussianRng(rng);
    const betaStar = sampleSphereBetaStar(pMax, r, gauss);
    const out = excessRiskSweep({
      n, pMax, pGrid, betaStarFull: betaStar, sigma, B: 15, rng: mulberry32(7919),
    });
    return Array.from(pGrid).map((p, i) => ({
      x: p, gamma: p / n, mean: out.mean[i],
    }));
  }, [nFixedCommitted, snrLogCommitted]);

  // Sample-wise: fix p, sweep n. Each n needs its own MC run with fresh design matrix
  const sampleWise = useMemo(() => {
    const snr = Math.pow(10, snrLogCommitted);
    const sigma = 1;
    const r = Math.sqrt(snr);
    const p = pFixedCommitted;
    const nGrid: number[] = [];
    for (let nVal = 5; nVal <= 200; nVal += 5) nGrid.push(nVal);
    const result: { x: number; gamma: number; mean: number }[] = [];
    for (const nVal of nGrid) {
      const pGrid = new Int32Array([p]);
      const rng = mulberry32(2024);
      const gauss = gaussianRng(rng);
      const betaStar = sampleSphereBetaStar(p, r, gauss);
      const out = excessRiskSweep({
        n: nVal, pMax: p, pGrid, betaStarFull: betaStar, sigma, B: 12, rng: mulberry32(7919 + nVal),
      });
      result.push({ x: nVal, gamma: p / nVal, mean: out.mean[0] });
    }
    return result;
  }, [pFixedCommitted, snrLogCommitted]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 28, right: 16, bottom: 56, left: 56 };
      const gap = 32;
      const panelW = (w - margin.left - margin.right - gap) / 2;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);

      const drawPanel = (
        offsetX: number,
        title: string,
        xLabel: string,
        data: { x: number; gamma: number; mean: number }[],
        xMax: number,
        fixedMark: number,
      ) => {
        const g = svg.append('g').attr('transform', `translate(${offsetX},${margin.top})`);
        const yMin = Math.max(1e-2, d3.min(data.map((d) => d.mean).filter((v) => v > 0)) ?? 1);
        const yMax = Math.min(1e5, d3.max(data.map((d) => d.mean)) ?? 100);
        const xScale = d3.scaleLinear().domain([0, xMax]).range([0, panelW]);
        const yScale = d3.scaleLog().domain([yMin * 0.7, yMax * 1.4]).range([H, 0]);
        g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xScale).ticks(6))
          .selectAll('text').style('fill', 'var(--color-text)');
        g.append('g').call(d3.axisLeft(yScale).ticks(6, '~g'))
          .selectAll('text').style('fill', 'var(--color-text)');
        g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
        g.append('text').attr('x', panelW / 2).attr('y', -8).attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text)').style('font-size', '12px').style('font-weight', '600').text(title);
        g.append('text').attr('x', panelW / 2).attr('y', H + 36).attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text)').style('font-size', '12px').text(xLabel);
        // Spike marker (where γ = 1)
        g.append('line').attr('x1', xScale(fixedMark)).attr('x2', xScale(fixedMark))
          .attr('y1', 0).attr('y2', H).style('stroke', 'var(--color-accent)')
          .attr('stroke-dasharray', '4 4').attr('opacity', 0.5);

        const lineGen = d3.line<{ x: number; mean: number }>().x((d) => xScale(d.x))
          .y((d) => yScale(Math.max(yMin * 0.7, d.mean)))
          .defined((d) => d.mean > 0);
        g.append('path').datum(data).attr('d', lineGen).attr('fill', 'none')
          .style('stroke', 'var(--color-accent)').attr('stroke-width', 2);
        g.selectAll('circle.mc').data(data).enter().append('circle')
          .attr('cx', (d) => xScale(d.x)).attr('cy', (d) => yScale(Math.max(yMin * 0.7, d.mean)))
          .attr('r', 2.2).style('fill', 'var(--color-accent)').filter((d) => d.mean > 0);

        if (showAnalytic) {
          // Hastie analytic curve at each gamma in the panel
          const snr = Math.pow(10, snrLogCommitted);
          const analyticData = data.map((d) => ({ x: d.x, mean: hastieRisk(d.gamma, snr, 1) }));
          const aGen = d3.line<{ x: number; mean: number }>().x((d) => xScale(d.x))
            .y((d) => yScale(Math.max(yMin * 0.7, d.mean)))
            .defined((d) => Number.isFinite(d.mean) && d.mean > 0);
          g.append('path').datum(analyticData).attr('d', aGen).attr('fill', 'none')
            .style('stroke', '#9B1D20').attr('stroke-width', 1.8).attr('stroke-dasharray', '5 3');
        }
      };

      drawPanel(margin.left, '(a) model-wise (fix n, sweep p)', `model size p   (n = ${nFixedCommitted})`,
        modelWise, 200, nFixedCommitted);
      drawPanel(margin.left + panelW + gap, '(b) sample-wise (fix p, sweep n)', `training-set size n   (p = ${pFixedCommitted})`,
        sampleWise, 200, pFixedCommitted);

      // y-axis label (shared)
      svg.append('text').attr('transform', 'rotate(-90)').attr('x', -HEIGHT / 2 - 6).attr('y', 16)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px')
        .text('excess risk (log scale)');
    },
    [containerWidth, modelWise, sampleWise, nFixedCommitted, pFixedCommitted, showAnalytic, snrLogCommitted],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
        aria-label="Two-panel model-wise vs sample-wise sweeps with Hastie analytic overlay." />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>SNR: <strong>{Math.pow(10, snrLog).toFixed(2)}</strong></span>
          <input type="range" min={-1} max={1} step={0.05} value={snrLog}
            onChange={(e) => setSnrLog(parseFloat(e.target.value))}
            onMouseUp={() => setSnrLogCommitted(snrLog)} onTouchEnd={() => setSnrLogCommitted(snrLog)}
            onKeyUp={() => setSnrLogCommitted(snrLog)} aria-label="SNR" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>n (left panel): <strong>{nFixed}</strong></span>
          <input type="range" min={20} max={100} step={5} value={nFixed}
            onChange={(e) => setNFixed(parseInt(e.target.value, 10))}
            onMouseUp={() => setNFixedCommitted(nFixed)} onTouchEnd={() => setNFixedCommitted(nFixed)}
            onKeyUp={() => setNFixedCommitted(nFixed)} aria-label="Fixed n for left panel" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>p (right panel): <strong>{pFixed}</strong></span>
          <input type="range" min={20} max={100} step={5} value={pFixed}
            onChange={(e) => setPFixed(parseInt(e.target.value, 10))}
            onMouseUp={() => setPFixedCommitted(pFixed)} onTouchEnd={() => setPFixedCommitted(pFixed)}
            onKeyUp={() => setPFixedCommitted(pFixed)} aria-label="Fixed p for right panel" />
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <input type="checkbox" checked={showAnalytic} onChange={(e) => setShowAnalytic(e.target.checked)} />
          <span>show Hastie analytic overlay</span>
        </label>
      </div>
    </div>
  );
}
