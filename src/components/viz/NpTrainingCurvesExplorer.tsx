// =============================================================================
// NpTrainingCurvesExplorer.tsx
//
// §4.5 CNP and Latent NP training curves over 1000 epochs, with smoothing
// window controlled by a slider. Also renders the §4.5 held-out task
// comparison panel (exact GP posterior + CNP + LatentNP traces) at the
// notebook's snapshot.
//
// Data: /sample-data/meta-learning/np_curves.json
// Static fallback: /images/topics/meta-learning/{08,09}_*.png
// =============================================================================

import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { META_PALETTE } from './shared/meta-learning';

interface NpCurvesPayload {
  epochs: number;
  cnpLoss: number[];
  lnpLoss: number[];
  smoothedCnpLoss: number[];
  smoothedLnpLoss: number[];
  smoothedXStart: number;
  heldOut: {
    ell: number;
    xFull: number[];
    yFull: number[];
    xContext: number[];
    yContext: number[];
    xDense: number[];
    muGp: number[];
    sdGp: number[];
    muCnp: number[];
    sdCnp: number[];
    muLnp: number[];
    sdLnp: number[];
    lnpSamples: number[][];
  };
}

const SM_BREAKPOINT = 640;

function movingAverage(arr: number[], w: number): { x: number[]; y: number[] } {
  if (arr.length < w) return { x: arr.map((_, i) => i), y: arr };
  const out: number[] = [];
  let sum = 0;
  for (let i = 0; i < w; i++) sum += arr[i];
  out.push(sum / w);
  for (let i = w; i < arr.length; i++) {
    sum += arr[i] - arr[i - w];
    out.push(sum / w);
  }
  const x = out.map((_, i) => i + Math.floor(w / 2));
  return { x, y: out };
}

export default function NpTrainingCurvesExplorer(): React.JSX.Element {
  const [payload, setPayload] = useState<NpCurvesPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [smoothWindow, setSmoothWindow] = useState(25);
  const [showLatentSamples, setShowLatentSamples] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetch('/sample-data/meta-learning/np_curves.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: NpCurvesPayload) => {
        if (!cancelled) setPayload(data);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const smoothedCnp = useMemo(() => (payload ? movingAverage(payload.cnpLoss, smoothWindow) : null), [payload, smoothWindow]);
  const smoothedLnp = useMemo(() => (payload ? movingAverage(payload.lnpLoss, smoothWindow) : null), [payload, smoothWindow]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const isMobile = (containerWidth || 800) < SM_BREAKPOINT;
  const panelWidth = isMobile ? containerWidth || 320 : (containerWidth || 800) / 2 - 12;
  const panelHeight = 240;

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (!payload || !smoothedCnp || !smoothedLnp || panelWidth <= 0) return;
      const margin = { top: 22, right: 12, bottom: 36, left: 50 };
      const innerW = panelWidth - margin.left - margin.right;
      const innerH = panelHeight - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const xScale = d3.scaleLinear().domain([0, payload.epochs - 1]).range([0, innerW]);
      const allY = [...payload.cnpLoss, ...payload.lnpLoss];
      const yScale = d3.scaleLinear().domain([Math.min(...allY) * 0.9, Math.max(...allY) * 1.05]).range([innerH, 0]);
      g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format('d'))).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text('epoch (1 task / epoch)');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -innerH / 2).attr('y', -36).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text('NLL / −ELBO');
      g.append('text').attr('x', innerW / 2).attr('y', -8).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px').text('CNP NLL & Latent NP −ELBO');
      // Raw curves (faint)
      const rawLineC = d3.line<number>().x((_, i) => xScale(i)).y((d) => yScale(d));
      g.append('path').datum(payload.cnpLoss).attr('d', rawLineC).style('fill', 'none').style('stroke', META_PALETTE[0]).style('stroke-width', 0.6).style('opacity', 0.35);
      g.append('path').datum(payload.lnpLoss).attr('d', rawLineC).style('fill', 'none').style('stroke', META_PALETTE[1]).style('stroke-width', 0.6).style('opacity', 0.35);
      // Smoothed curves
      const smLine = (pts: { x: number[]; y: number[] }) =>
        d3.line<number>().x((_, i) => xScale(pts.x[i])).y((d) => yScale(d))(pts.y);
      g.append('path').attr('d', smLine(smoothedCnp)).style('fill', 'none').style('stroke', META_PALETTE[0]).style('stroke-width', 1.6);
      g.append('path').attr('d', smLine(smoothedLnp)).style('fill', 'none').style('stroke', META_PALETTE[1]).style('stroke-width', 1.6);
      // legend
      const legendG = g.append('g').attr('transform', `translate(${innerW - 130}, 0)`);
      legendG.append('rect').attr('width', 12).attr('height', 3).attr('y', 5).style('fill', META_PALETTE[0]);
      legendG.append('text').attr('x', 16).attr('y', 8).style('fill', 'var(--color-text)').style('font-size', '10px').text('CNP NLL');
      legendG.append('rect').attr('width', 12).attr('height', 3).attr('y', 19).style('fill', META_PALETTE[1]);
      legendG.append('text').attr('x', 16).attr('y', 22).style('fill', 'var(--color-text)').style('font-size', '10px').text('Latent NP −ELBO');
    },
    [payload, smoothedCnp, smoothedLnp, panelWidth],
  );

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (!payload || panelWidth <= 0) return;
      const margin = { top: 22, right: 12, bottom: 36, left: 50 };
      const innerW = panelWidth - margin.left - margin.right;
      const innerH = panelHeight - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const xScale = d3.scaleLinear().domain([-3, 3]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([-3, 3]).range([innerH, 0]);
      g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text(`x  (held-out GP, ℓ=${payload.heldOut.ell.toFixed(2)})`);
      g.append('text').attr('x', innerW / 2).attr('y', -8).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px').text('Held-out task: GP vs CNP vs Latent NP');
      // True function
      const tLine = d3.line<number>().x((_, i) => xScale(payload.heldOut.xFull[i])).y((d) => yScale(d));
      g.append('path').datum(payload.heldOut.yFull).attr('d', tLine).style('fill', 'none').style('stroke', 'var(--color-text)').style('stroke-width', 1).style('opacity', 0.4);
      // GP predictive
      const xD = payload.heldOut.xDense;
      const gpLine = d3.line<number>().x((_, i) => xScale(xD[i])).y((d) => yScale(d));
      const gpBand = d3.area<number>().x((_, i) => xScale(xD[i])).y0((_, i) => yScale(payload.heldOut.muGp[i] - 1.96 * payload.heldOut.sdGp[i])).y1((_, i) => yScale(payload.heldOut.muGp[i] + 1.96 * payload.heldOut.sdGp[i]));
      g.append('path').datum(payload.heldOut.muGp).attr('d', gpBand).style('fill', META_PALETTE[2]).style('opacity', 0.15);
      g.append('path').datum(payload.heldOut.muGp).attr('d', gpLine).style('fill', 'none').style('stroke', META_PALETTE[2]).style('stroke-width', 1.4).style('stroke-dasharray', '4 3');
      // CNP
      g.append('path').datum(payload.heldOut.muCnp).attr('d', gpLine).style('fill', 'none').style('stroke', META_PALETTE[0]).style('stroke-width', 1.4);
      // Latent NP mean + samples
      g.append('path').datum(payload.heldOut.muLnp).attr('d', gpLine).style('fill', 'none').style('stroke', META_PALETTE[1]).style('stroke-width', 1.4);
      if (showLatentSamples) {
        payload.heldOut.lnpSamples.forEach((sample) => {
          g.append('path').datum(sample).attr('d', gpLine).style('fill', 'none').style('stroke', META_PALETTE[1]).style('stroke-width', 0.6).style('opacity', 0.4);
        });
      }
      // Context points
      payload.heldOut.xContext.forEach((x, i) => {
        g.append('circle').attr('cx', xScale(x)).attr('cy', yScale(payload.heldOut.yContext[i])).attr('r', 4).style('fill', META_PALETTE[3]).style('stroke', 'white').style('stroke-width', 0.8);
      });
      // legend
      const legendG = g.append('g').attr('transform', `translate(${innerW - 130}, 0)`);
      ['exact GP', 'CNP', 'Latent NP'].forEach((lbl, idx) => {
        legendG.append('rect').attr('width', 12).attr('height', 3).attr('y', 5 + idx * 13).style('fill', META_PALETTE[idx === 0 ? 2 : idx === 1 ? 0 : 1]);
        legendG.append('text').attr('x', 16).attr('y', 8 + idx * 13).style('fill', 'var(--color-text)').style('font-size', '10px').text(lbl);
      });
    },
    [payload, showLatentSamples, panelWidth],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'flex', gap: 12, marginBottom: 8, alignItems: 'center', fontSize: 11 }}>
        {payload ? (
          <>
            <label style={{ flex: 1 }}>
              smoothing window: <strong>{smoothWindow}</strong>
              <input type="range" min={1} max={101} step={2} value={smoothWindow} onChange={(e) => setSmoothWindow(Number(e.target.value))} style={{ width: '100%' }} aria-label="smoothing" />
            </label>
            <label>
              <input type="checkbox" checked={showLatentSamples} onChange={(e) => setShowLatentSamples(e.target.checked)} /> show Latent NP samples
            </label>
          </>
        ) : (
          <span style={{ color: 'var(--color-text-secondary)' }}>{error ? `Error loading data: ${error}` : 'Loading…'}</span>
        )}
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16 }}>
        <svg ref={leftRef} width={panelWidth} height={panelHeight} role="img" aria-label="NP training curves" />
        <svg ref={rightRef} width={panelWidth} height={panelHeight} role="img" aria-label="held-out NP comparison" />
      </div>
    </div>
  );
}
