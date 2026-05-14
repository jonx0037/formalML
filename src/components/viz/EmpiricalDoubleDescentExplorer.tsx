import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  excessRiskSweep,
  gaussianRng,
  mulberry32,
  sampleSphereBetaStar,
} from './shared/double-descent';

// =============================================================================
// EmpiricalDoubleDescentExplorer — §1.2 (Figure 1)
//
// Monte Carlo empirical excess-risk curve for the §1 misspecified-Hastie setup.
// Static fallback: public/images/topics/double-descent/01_empirical_double_descent.png
//
// Sliders:
//   - n (training-set size, commit-on-release): default 50, range [10, 100].
//   - SNR (signal-to-noise ratio, log scale, commit-on-release): default 1.0,
//     range [0.1, 10].
//   - B (Monte Carlo replicates, commit-on-release): default 30, range [10, 100].
//   - λ (ridge overlay, log scale, live): default 0 (ridgeless), range [0, 10].
// =============================================================================

const HEIGHT = 460;
const P_GRID = Array.from({ length: 41 }, (_, i) => Math.max(1, Math.round((i / 40) * 200)));

// Map slider value t ∈ [0, 1] → λ ∈ [lo, hi] on log scale.
function logSliderToLambda(t: number, lo: number, hi: number): number {
  return Math.exp(Math.log(lo) + t * (Math.log(hi) - Math.log(lo)));
}

// Inverse of logSliderToLambda: map λ → t ∈ [0, 1].
function lambdaToSliderT(lambda: number, lo: number, hi: number): number {
  return (Math.log(lambda) - Math.log(lo)) / (Math.log(hi) - Math.log(lo));
}

export default function EmpiricalDoubleDescentExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  // Display values update live; committed values drive the MC sweep.
  const [nDisplay, setNDisplay] = useState(50);
  const [nCommitted, setNCommitted] = useState(50);
  const [snrLogDisplay, setSnrLogDisplay] = useState(0); // log10(SNR), default SNR = 1
  const [snrLogCommitted, setSnrLogCommitted] = useState(0);
  const [BDisplay, setBDisplay] = useState(30);
  const [BCommitted, setBCommitted] = useState(30);
  const [lambdaDisplay, setLambdaDisplay] = useState(0); // 0 ⇒ ridgeless
  const [showRidge, setShowRidge] = useState(false);

  const { curve, ridgeCurve, pGrid } = useMemo(() => {
    const n = nCommitted;
    const snr = Math.pow(10, snrLogCommitted);
    const sigma = 1;
    const rNorm = Math.sqrt(snr); // ‖β*‖² = SNR · σ² = SNR
    const pMax = 200;
    const pGrid = new Int32Array(P_GRID.filter((p) => p <= pMax));
    const baseRng = mulberry32(2024);
    const gauss = gaussianRng(baseRng);
    const betaStar = sampleSphereBetaStar(pMax, rNorm, gauss);
    const lambdas = showRidge && lambdaDisplay > 0
      ? new Float64Array([0, lambdaDisplay])
      : undefined;
    const out = excessRiskSweep({
      n,
      pMax,
      pGrid,
      betaStarFull: betaStar,
      sigma,
      B: BCommitted,
      rng: mulberry32(7919),
      ridgeLambdas: lambdas,
    });
    if (lambdas) {
      const nL = lambdas.length;
      const ridgeless = Array.from(pGrid).map((p, gi) => ({
        p,
        mean: out.mean[gi * nL + 0],
        low: out.iqrLow[gi * nL + 0],
        high: out.iqrHigh[gi * nL + 0],
      }));
      const ridge = Array.from(pGrid).map((p, gi) => ({
        p,
        mean: out.mean[gi * nL + 1],
        low: out.iqrLow[gi * nL + 1],
        high: out.iqrHigh[gi * nL + 1],
      }));
      return { curve: ridgeless, ridgeCurve: ridge, pGrid };
    }
    const single = Array.from(pGrid).map((p, gi) => ({
      p,
      mean: out.mean[gi],
      low: out.iqrLow[gi],
      high: out.iqrHigh[gi],
    }));
    return { curve: single, ridgeCurve: null, pGrid };
  }, [nCommitted, snrLogCommitted, BCommitted, showRidge, lambdaDisplay]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 28, right: 32, bottom: 56, left: 64 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const allMeans = curve.map((d) => d.mean);
      const allHigh = curve.map((d) => d.high);
      const allLow = curve.map((d) => d.low);
      const ridgeMeans = ridgeCurve ? ridgeCurve.map((d) => d.mean) : [];
      const yMin = Math.max(1e-2, d3.min([...allMeans, ...allLow, ...ridgeMeans]) ?? 1);
      const yMax = d3.max([...allMeans, ...allHigh, ...ridgeMeans]) ?? 100;

      const xScale = d3.scaleLinear().domain([1, 200]).range([0, W]);
      const yScale = d3.scaleLog().domain([yMin * 0.7, yMax * 1.4]).range([H, 0]);

      const xAxis = d3.axisBottom(xScale).ticks(8).tickSizeOuter(0);
      const yAxis = d3.axisLeft(yScale).ticks(6, '~g').tickSizeOuter(0);
      g.append('g').attr('transform', `translate(0,${H})`).call(xAxis).selectAll('text')
        .style('fill', 'var(--color-text)');
      g.append('g').call(yAxis).selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

      g.append('text').attr('x', W / 2).attr('y', H + 40).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '12px')
        .text('model size p (number of features)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -48)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)').style('font-size', '12px')
        .text('excess test risk (log scale)');

      // Threshold marker p = n.
      g.append('line').attr('x1', xScale(nCommitted)).attr('x2', xScale(nCommitted))
        .attr('y1', 0).attr('y2', H).style('stroke', 'var(--color-accent)')
        .attr('stroke-dasharray', '4 4').attr('opacity', 0.6);
      g.append('text').attr('x', xScale(nCommitted) + 4).attr('y', 12)
        .style('fill', 'var(--color-accent)').style('font-family', 'var(--font-sans)').style('font-size', '11px')
        .text(`p = n = ${nCommitted}`);

      // IQR band
      const areaGen = d3.area<{ p: number; low: number; high: number }>()
        .x((d) => xScale(d.p))
        .y0((d) => yScale(Math.max(yMin * 0.7, d.low)))
        .y1((d) => yScale(d.high))
        .curve(d3.curveMonotoneX);
      g.append('path').datum(curve).attr('d', areaGen)
        .style('fill', 'var(--color-accent)').attr('opacity', 0.15);

      // Mean line
      const lineGen = d3.line<{ p: number; mean: number }>()
        .x((d) => xScale(d.p))
        .y((d) => yScale(Math.max(yMin * 0.7, d.mean)))
        .curve(d3.curveMonotoneX);
      g.append('path').datum(curve).attr('d', lineGen).attr('fill', 'none')
        .style('stroke', 'var(--color-accent)').attr('stroke-width', 2.2);

      // Ridge overlay
      if (ridgeCurve) {
        g.append('path').datum(ridgeCurve).attr('d', lineGen).attr('fill', 'none')
          .style('stroke', '#D97706').attr('stroke-width', 2).attr('stroke-dasharray', '5 3');
        g.append('text').attr('x', W - 10).attr('y', 14).attr('text-anchor', 'end')
          .style('fill', '#D97706').style('font-family', 'var(--font-sans)').style('font-size', '11px')
          .text(`λ = ${lambdaDisplay.toFixed(3)} (ridge)`);
      }

      // Mean-point dots
      g.selectAll('circle.mean').data(curve).enter().append('circle')
        .attr('cx', (d) => xScale(d.p)).attr('cy', (d) => yScale(Math.max(yMin * 0.7, d.mean)))
        .attr('r', 2.5).style('fill', 'var(--color-accent)');
    },
    [containerWidth, curve, ridgeCurve, nCommitted, lambdaDisplay],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
        aria-label="Empirical excess-risk curve with the interpolation-threshold spike at p = n." />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>n (training-set size): <strong>{nDisplay}</strong></span>
          <input type="range" min={10} max={100} step={5} value={nDisplay}
            onChange={(e) => setNDisplay(parseInt(e.target.value, 10))}
            onMouseUp={() => setNCommitted(nDisplay)}
            onTouchEnd={() => setNCommitted(nDisplay)}
            onKeyUp={() => setNCommitted(nDisplay)}
            aria-label="Training set size n" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>SNR: <strong>{Math.pow(10, snrLogDisplay).toFixed(2)}</strong></span>
          <input type="range" min={-1} max={1} step={0.05} value={snrLogDisplay}
            onChange={(e) => setSnrLogDisplay(parseFloat(e.target.value))}
            onMouseUp={() => setSnrLogCommitted(snrLogDisplay)}
            onTouchEnd={() => setSnrLogCommitted(snrLogDisplay)}
            onKeyUp={() => setSnrLogCommitted(snrLogDisplay)}
            aria-label="Signal-to-noise ratio, log scale" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>B (replicates): <strong>{BDisplay}</strong></span>
          <input type="range" min={10} max={100} step={5} value={BDisplay}
            onChange={(e) => setBDisplay(parseInt(e.target.value, 10))}
            onMouseUp={() => setBCommitted(BDisplay)}
            onTouchEnd={() => setBCommitted(BDisplay)}
            onKeyUp={() => setBCommitted(BDisplay)}
            aria-label="Monte Carlo replicates B" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 200 }}>
          <span>
            <input type="checkbox" checked={showRidge}
              onChange={(e) => setShowRidge(e.target.checked)} /> {' '}
            ridge overlay (λ = <strong>{lambdaDisplay.toFixed(3)}</strong>)
          </span>
          <input type="range" min={0} max={1} step={0.02}
            value={lambdaDisplay > 0 ? lambdaToSliderT(lambdaDisplay, 0.01, 10) : 0}
            onChange={(e) => {
              const t = parseFloat(e.target.value);
              setLambdaDisplay(t === 0 ? 0 : logSliderToLambda(t, 0.01, 10));
            }}
            disabled={!showRidge} aria-label="Ridge penalty lambda, log scale" />
        </label>
      </div>
    </div>
  );
}
