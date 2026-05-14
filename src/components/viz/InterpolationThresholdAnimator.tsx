import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  betaMinNorm,
  conditionNumber,
  excessRiskMisspecified,
  fillIsotropicGaussian,
  gaussianRng,
  matVec,
  mpSupport,
  mulberry32,
  sampleSphereBetaStar,
  thinSVD,
} from './shared/double-descent';

// =============================================================================
// InterpolationThresholdAnimator — §3.4 (Figure 3, three-panel synchronized)
//
// Panels (top to bottom, shared x = p):
//   - σ_min, σ_max with MP overlay |1-√γ|, 1+√γ
//   - ‖β̂‖² (log scale)
//   - excess test risk (log scale)
// Static fallback: 03_interpolation_threshold_synced.png
// =============================================================================

const PANEL_HEIGHT = 180;
const PANELS = 3;
const HEIGHT = PANEL_HEIGHT * PANELS + 40;
const P_GRID = (() => {
  const out: number[] = [];
  for (let p = 1; p <= 200; p += 2) out.push(p);
  return out;
})();

export default function InterpolationThresholdAnimator() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nDisplay, setNDisplay] = useState(50);
  const [nCommitted, setNCommitted] = useState(50);
  const [snrLog, setSnrLog] = useState(0); // log10(SNR)
  const [snrLogCommitted, setSnrLogCommitted] = useState(0);
  const [showMP, setShowMP] = useState(true);

  const data = useMemo(() => {
    const n = nCommitted;
    const snr = Math.pow(10, snrLogCommitted);
    const sigma = 1;
    const r = Math.sqrt(snr);
    const pMax = 200;
    const rng = mulberry32(2024);
    const gauss = gaussianRng(rng);
    const betaStar = sampleSphereBetaStar(pMax, r, gauss);
    const X = new Float64Array(n * pMax);
    fillIsotropicGaussian(X, n, pMax, gauss);
    const y = new Float64Array(n);
    matVec(X, betaStar, n, pMax, y);
    for (let i = 0; i < n; i++) y[i] += sigma * gauss();
    const Xslice = new Float64Array(n * pMax);

    const out: { p: number; sMin: number; sMax: number; betaNorm: number; risk: number; gamma: number }[] = [];
    for (const p of P_GRID) {
      // Build Xslice (n × p)
      for (let i = 0; i < n; i++) {
        const sb = i * pMax;
        const db = i * p;
        for (let j = 0; j < p; j++) Xslice[db + j] = X[sb + j];
      }
      const svd = thinSVD(Xslice, n, p);
      let sMin = Infinity;
      let sMax = 0;
      for (let k = 0; k < svd.S.length; k++) {
        if (svd.S[k] < sMin) sMin = svd.S[k];
        if (svd.S[k] > sMax) sMax = svd.S[k];
      }
      const betaHat = betaMinNorm(Xslice, y, n, p);
      let bn = 0;
      for (let j = 0; j < p; j++) bn += betaHat[j] * betaHat[j];
      const risk = excessRiskMisspecified(betaHat, betaStar, p);
      out.push({ p, sMin: sMin / Math.sqrt(n), sMax: sMax / Math.sqrt(n), betaNorm: bn, risk, gamma: p / n });
    }
    return out;
  }, [nCommitted, snrLogCommitted]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 18, right: 32, bottom: 38, left: 64 };
      const W = w - margin.left - margin.right;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);

      const xScale = d3.scaleLinear().domain([0, 200]).range([0, W]);

      const ps = data.map((d) => d.p);

      const drawPanel = (
        idx: number,
        yVals: number[],
        label: string,
        logY: boolean,
        overlays?: { color: string; vals: number[]; label: string }[],
        markX?: number,
      ) => {
        const yTop = margin.top + idx * PANEL_HEIGHT;
        const g = svg.append('g').attr('transform', `translate(${margin.left},${yTop})`);
        const H = PANEL_HEIGHT - 20;
        const valid = yVals.filter((v) => Number.isFinite(v) && (!logY || v > 0));
        const yMin = logY ? Math.max(1e-6, d3.min(valid) ?? 1e-6) : (d3.min(yVals) ?? 0);
        const yMax = d3.max([...valid, ...(overlays?.flatMap((o) => o.vals.filter((v) => Number.isFinite(v))) ?? [])]) ?? 1;
        const yScale = logY
          ? d3.scaleLog().domain([yMin * 0.7, yMax * 1.4]).range([H, 0])
          : d3.scaleLinear().domain([Math.min(0, yMin) * 1.05, yMax * 1.2]).range([H, 0]);

        g.append('g').attr('transform', `translate(0,${H})`)
          .call(d3.axisBottom(xScale).ticks(8).tickSizeOuter(0))
          .selectAll('text').style('fill', 'var(--color-text)');
        g.append('g').call(d3.axisLeft(yScale).ticks(4, '~g').tickSizeOuter(0))
          .selectAll('text').style('fill', 'var(--color-text)');
        g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
        g.append('text').attr('x', -50).attr('y', H / 2).attr('text-anchor', 'middle')
          .attr('transform', `rotate(-90, -50, ${H / 2})`)
          .style('fill', 'var(--color-text)').style('font-size', '11px').text(label);

        // Threshold marker
        if (markX !== undefined) {
          g.append('line').attr('x1', xScale(markX)).attr('x2', xScale(markX))
            .attr('y1', 0).attr('y2', H).style('stroke', 'var(--color-accent)')
            .attr('stroke-dasharray', '4 4').attr('opacity', 0.5);
        }

        const lineGen = d3.line<[number, number]>().x((d) => xScale(d[0]))
          .y((d) => yScale(Math.max(logY ? yMin * 0.7 : -Infinity, d[1])))
          .defined((d) => Number.isFinite(d[1]) && (!logY || d[1] > 0));
        const pairs: [number, number][] = ps.map((p, i) => [p, yVals[i]]);
        g.append('path').datum(pairs).attr('d', lineGen).attr('fill', 'none')
          .style('stroke', 'var(--color-accent)').attr('stroke-width', 2);

        overlays?.forEach((ov) => {
          const ovPairs: [number, number][] = ps.map((p, i) => [p, ov.vals[i]]);
          g.append('path').datum(ovPairs).attr('d', lineGen).attr('fill', 'none')
            .style('stroke', ov.color).attr('stroke-width', 1.5).attr('stroke-dasharray', '5 3');
        });
      };

      // Panel 1: σ_min, σ_max (linear)
      // Combine into two series; we'll plot σ_min on bottom and σ_max on same axis.
      const sMins = data.map((d) => d.sMin);
      const sMaxes = data.map((d) => d.sMax);
      // For MP overlay: |1-√γ|, 1+√γ
      const mpMin = data.map((d) => Math.abs(1 - Math.sqrt(d.gamma)));
      const mpMax = data.map((d) => 1 + Math.sqrt(d.gamma));
      // Draw both σ_min and σ_max as two lines in one panel
      const yTop1 = margin.top;
      const g1 = svg.append('g').attr('transform', `translate(${margin.left},${yTop1})`);
      const H1 = PANEL_HEIGHT - 20;
      const yMax1 = d3.max([...sMaxes, ...mpMax]) ?? 3;
      const yScale1 = d3.scaleLinear().domain([0, yMax1 * 1.2]).range([H1, 0]);
      g1.append('g').attr('transform', `translate(0,${H1})`).call(d3.axisBottom(xScale).ticks(8))
        .selectAll('text').style('fill', 'var(--color-text)');
      g1.append('g').call(d3.axisLeft(yScale1).ticks(4))
        .selectAll('text').style('fill', 'var(--color-text)');
      g1.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      g1.append('text').attr('x', -50).attr('y', H1 / 2)
        .attr('transform', `rotate(-90, -50, ${H1 / 2})`)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '11px').text('σ_min, σ_max / √n');
      g1.append('line').attr('x1', xScale(nCommitted)).attr('x2', xScale(nCommitted))
        .attr('y1', 0).attr('y2', H1).style('stroke', 'var(--color-accent)')
        .attr('stroke-dasharray', '4 4').attr('opacity', 0.5);
      const lineGen1 = d3.line<[number, number]>().x((d) => xScale(d[0])).y((d) => yScale1(d[1]));
      g1.append('path').datum(ps.map((p, i) => [p, sMins[i]] as [number, number]))
        .attr('d', lineGen1).attr('fill', 'none').style('stroke', '#534AB7').attr('stroke-width', 2);
      g1.append('path').datum(ps.map((p, i) => [p, sMaxes[i]] as [number, number]))
        .attr('d', lineGen1).attr('fill', 'none').style('stroke', 'var(--color-accent)').attr('stroke-width', 2);
      if (showMP) {
        g1.append('path').datum(ps.map((p, i) => [p, mpMin[i]] as [number, number]))
          .attr('d', lineGen1).attr('fill', 'none').style('stroke', '#534AB7')
          .attr('stroke-width', 1).attr('stroke-dasharray', '5 3');
        g1.append('path').datum(ps.map((p, i) => [p, mpMax[i]] as [number, number]))
          .attr('d', lineGen1).attr('fill', 'none').style('stroke', 'var(--color-accent)')
          .attr('stroke-width', 1).attr('stroke-dasharray', '5 3');
      }
      g1.append('text').attr('x', W - 6).attr('y', 12).attr('text-anchor', 'end')
        .style('fill', 'var(--color-accent)').style('font-size', '11px').text('σ_max');
      g1.append('text').attr('x', W - 6).attr('y', 26).attr('text-anchor', 'end')
        .style('fill', '#534AB7').style('font-size', '11px').text('σ_min');

      // Panel 2: ‖β̂‖²
      drawPanel(1, data.map((d) => d.betaNorm), '‖β̂‖² (log)', true, undefined, nCommitted);
      // Panel 3: excess risk
      drawPanel(2, data.map((d) => d.risk), 'excess risk (log)', true, undefined, nCommitted);

      // x-axis label
      svg.append('text').attr('x', margin.left + W / 2).attr('y', HEIGHT - 6)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px')
        .text('model size p');
    },
    [containerWidth, data, nCommitted, showMP],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
        aria-label="Three-panel diagnostic: singular values with MP overlay, coefficient norm, excess risk." />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>n: <strong>{nDisplay}</strong></span>
          <input type="range" min={20} max={100} step={5} value={nDisplay}
            onChange={(e) => setNDisplay(parseInt(e.target.value, 10))}
            onMouseUp={() => setNCommitted(nDisplay)} onTouchEnd={() => setNCommitted(nDisplay)}
            onKeyUp={() => setNCommitted(nDisplay)} aria-label="Training set size" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>SNR: <strong>{Math.pow(10, snrLog).toFixed(2)}</strong></span>
          <input type="range" min={-1} max={1} step={0.05} value={snrLog}
            onChange={(e) => setSnrLog(parseFloat(e.target.value))}
            onMouseUp={() => setSnrLogCommitted(snrLog)}
            onTouchEnd={() => setSnrLogCommitted(snrLog)}
            onKeyUp={() => setSnrLogCommitted(snrLog)} aria-label="Signal-to-noise ratio" />
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <input type="checkbox" checked={showMP} onChange={(e) => setShowMP(e.target.checked)} />
          <span>show MP overlay |1 ± √γ|</span>
        </label>
      </div>
    </div>
  );
}
