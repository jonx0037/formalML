import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  ece,
  eceEqFreq,
  equalFreqBins,
  fitIsotonic,
  mce,
  mulberry32,
  predictIsotonic,
  reliabilityBins,
  sharpness,
} from './shared/uncertainty-quantification';

// =============================================================================
// ReliabilityDiagramExplorer — §3
//
// Builds a synthetic overconfident binary classifier (truth probability p_true
// stretched toward {0,1} by a sharpness parameter τ). Sliders for τ (how
// overconfident), bin count K, and toggle equal-width vs equal-frequency.
// Reads ECE / MCE / sharpness and the reliability curve live.
// Static fallback: fig_03_reliability_eq_width_vs_eq_freq.png
// =============================================================================

const HEIGHT = 440;

function buildClassifier(n: number, tau: number, rng: () => number) {
  // True conditional positive probability draws from a Beta-shaped target.
  // Forecast is p_hat = sigma((logit(p_true) * tau)) — τ > 1 sharpens (overconfident).
  const yTrue: number[] = [];
  const pHat: number[] = [];
  for (let i = 0; i < n; i++) {
    // Latent score uniform on [0, 1].
    const u = rng();
    const pTrue = u;
    const y = rng() < pTrue ? 1 : 0;
    const eps = 1e-6;
    const z = Math.log((pTrue + eps) / (1 - pTrue + eps));
    const phat = 1 / (1 + Math.exp(-tau * z));
    yTrue.push(y);
    pHat.push(phat);
  }
  return { yTrue, pHat };
}

export default function ReliabilityDiagramExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [tauDisplay, setTauDisplay] = useState(2.5);
  const [tauCommitted, setTauCommitted] = useState(2.5);
  const [kDisplay, setKDisplay] = useState(10);
  const [kCommitted, setKCommitted] = useState(10);
  const [eqFreq, setEqFreq] = useState(false);
  const [showRecalibrated, setShowRecalibrated] = useState(false);

  const stats = useMemo(() => {
    const rng = mulberry32(20260514);
    // Draw twice: a held-out calibration sample for fitting isotonic, and a
    // held-out evaluation sample for measuring ECE/MCE/sharpness. Without the
    // split, isotonic-recalibrated ECE is in-sample and overly optimistic.
    const cal = buildClassifier(2000, tauCommitted, rng);
    const evalSample = buildClassifier(2000, tauCommitted, rng);
    const yTrue = evalSample.yTrue;
    const pHat = evalSample.pHat;
    const eceVal = (eqFreq ? eceEqFreq : ece)(yTrue, pHat, kCommitted);
    const mceVal = mce(yTrue, pHat, kCommitted);
    const shVal = sharpness(pHat);
    const bins = eqFreq
      ? equalFreqBins(yTrue, pHat, kCommitted)
      : reliabilityBins(yTrue, pHat, kCommitted);
    // Fit isotonic on (pHat_cal, y_cal); evaluate on the held-out pHat.
    let recal: number[] | null = null;
    let eceRecal: number | null = null;
    let binsRecal: ReturnType<typeof reliabilityBins> | null = null;
    if (showRecalibrated) {
      const isoFit = fitIsotonic(cal.pHat, cal.yTrue);
      recal = predictIsotonic(isoFit, pHat);
      eceRecal = (eqFreq ? eceEqFreq : ece)(yTrue, recal, kCommitted);
      binsRecal = eqFreq
        ? equalFreqBins(yTrue, recal, kCommitted)
        : reliabilityBins(yTrue, recal, kCommitted);
    }
    return { yTrue, pHat, eceVal, mceVal, shVal, bins, recal, eceRecal, binsRecal };
  }, [tauCommitted, kCommitted, eqFreq, showRecalibrated]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 24, right: 24, bottom: 48, left: 56 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([0, 1]).range([0, W]);
      const yScale = d3.scaleLinear().domain([0, 1]).range([H, 0]);

      // Perfect-calibration diagonal.
      g.append('line').attr('x1', xScale(0)).attr('y1', yScale(0))
        .attr('x2', xScale(1)).attr('y2', yScale(1))
        .style('stroke', '#94a3b8').attr('stroke-dasharray', '4 3').attr('stroke-width', 1.4);

      const drawBins = (bins: { binConf: number[]; binAcc: number[]; binW: number[] },
                       color: string, marker: 'circle' | 'square') => {
        for (let b = 0; b < bins.binConf.length; b++) {
          if (Number.isNaN(bins.binConf[b]) || bins.binW[b] === 0) continue;
          // Gap segment.
          g.append('line').attr('x1', xScale(bins.binConf[b]))
            .attr('y1', yScale(Math.min(bins.binConf[b], bins.binAcc[b])))
            .attr('x2', xScale(bins.binConf[b]))
            .attr('y2', yScale(Math.max(bins.binConf[b], bins.binAcc[b])))
            .style('stroke', '#f97316').attr('stroke-width', 2).style('opacity', 0.55);
        }
        const validBins = bins.binConf.map((c, i) => ({ c, a: bins.binAcc[i], w: bins.binW[i] }))
          .filter((d) => !Number.isNaN(d.c) && d.w > 0);
        // Line through bin centers.
        const lineGen = d3.line<{ c: number; a: number }>()
          .x((d) => xScale(d.c)).y((d) => yScale(d.a));
        g.append('path').datum(validBins).attr('d', lineGen).attr('fill', 'none')
          .style('stroke', color).attr('stroke-width', 1.6);
        // Markers.
        for (const d of validBins) {
          if (marker === 'circle') {
            g.append('circle').attr('cx', xScale(d.c)).attr('cy', yScale(d.a))
              .attr('r', 4 + 8 * d.w).style('fill', color).style('opacity', 0.85);
          } else {
            const s = 5 + 6 * d.w;
            g.append('rect').attr('x', xScale(d.c) - s).attr('y', yScale(d.a) - s)
              .attr('width', 2 * s).attr('height', 2 * s)
              .style('fill', color).style('opacity', 0.65);
          }
        }
      };

      drawBins(stats.bins, '#dc2626', 'circle');
      if (stats.binsRecal) drawBins(stats.binsRecal, '#8b5cf6', 'square');

      g.append('g').attr('transform', `translate(0,${H})`)
        .call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', W / 2).attr('y', H + 36)
        .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', 'var(--color-text)')
        .text('mean predicted probability  p̂');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -40)
        .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', 'var(--color-text)')
        .text('empirical accuracy');

      // Stats badge.
      const badge = g.append('g').attr('transform', `translate(${W - 200}, 8)`);
      const lines = [
        `ECE = ${stats.eceVal.toFixed(3)}`,
        `MCE = ${stats.mceVal.toFixed(3)}`,
        `sharpness = ${stats.shVal.toFixed(3)}`,
        stats.eceRecal !== null ? `ECE (recal) = ${stats.eceRecal.toFixed(3)}` : '',
      ].filter(Boolean);
      lines.forEach((t, i) => {
        badge.append('text').attr('x', 0).attr('y', i * 16)
          .style('font-size', '12px').style('fill', 'var(--color-text)').text(t);
      });
    },
    [containerWidth, stats],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
        aria-label="Interactive reliability diagram with calibration gap segments and live ECE/MCE/sharpness readouts." />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 200 }}>
          <span>sharpness τ (overconfidence): <strong>{tauDisplay.toFixed(1)}</strong></span>
          <input type="range" min={0.5} max={4} step={0.1} value={tauDisplay}
            onChange={(e) => setTauDisplay(parseFloat(e.target.value))}
            onMouseUp={() => setTauCommitted(tauDisplay)} onTouchEnd={() => setTauCommitted(tauDisplay)}
            onKeyUp={() => setTauCommitted(tauDisplay)} aria-label="Classifier sharpness τ" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 200 }}>
          <span>bins K: <strong>{kDisplay}</strong></span>
          <input type="range" min={2} max={30} step={1} value={kDisplay}
            onChange={(e) => setKDisplay(parseInt(e.target.value, 10))}
            onMouseUp={() => setKCommitted(kDisplay)} onTouchEnd={() => setKCommitted(kDisplay)}
            onKeyUp={() => setKCommitted(kDisplay)} aria-label="Number of bins K" />
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <input type="checkbox" checked={eqFreq} onChange={(e) => setEqFreq(e.target.checked)} />
          <span>equal-frequency bins</span>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <input type="checkbox" checked={showRecalibrated} onChange={(e) => setShowRecalibrated(e.target.checked)} />
          <span>overlay isotonic-recalibrated</span>
        </label>
      </div>
    </div>
  );
}
