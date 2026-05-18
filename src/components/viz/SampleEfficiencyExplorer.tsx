// =============================================================================
// SampleEfficiencyExplorer.tsx
//
// §9.2 Sample-efficiency sweep: test NLL vs context size for CNP and Latent NP
// on 20 held-out GP tasks. Slider lets the reader exclude certain K_ctx values
// (truncate the x-axis range) to focus on the steep-descent or plateau region.
//
// Data: /sample-data/meta-learning/sample_efficiency.json
// Static fallback: /images/topics/meta-learning/15_sample_efficiency.png
// =============================================================================

import { useEffect, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { META_PALETTE } from './shared/meta-learning';

interface SampleEfficiencyPayload {
  K_ctx: number[];
  cnpNllMean: number[];
  cnpNllSE: number[];
  lnpNllMean: number[];
  lnpNllSE: number[];
}

export default function SampleEfficiencyExplorer(): React.JSX.Element {
  const [payload, setPayload] = useState<SampleEfficiencyPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showCnp, setShowCnp] = useState(true);
  const [showLnp, setShowLnp] = useState(true);
  const [errBarScale, setErrBarScale] = useState(1.96);

  useEffect(() => {
    let cancelled = false;
    fetch('/sample-data/meta-learning/sample_efficiency.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d: SampleEfficiencyPayload) => {
        if (!cancelled) setPayload(d);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const width = containerWidth || 700;
  const height = 320;

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (!payload || width <= 0) return;
      const margin = { top: 18, right: 16, bottom: 40, left: 60 };
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const xScale = d3.scaleLinear().domain([0, Math.max(...payload.K_ctx) + 2]).range([0, innerW]);
      const allY = [...payload.cnpNllMean, ...payload.lnpNllMean];
      const allSE = [...payload.cnpNllSE, ...payload.lnpNllSE];
      const yMax = Math.max(...allY) + errBarScale * Math.max(...allSE) * 1.1;
      const yMin = Math.min(...allY) - errBarScale * Math.max(...allSE) * 1.1;
      const yScale = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]);
      g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xScale).ticks(7).tickFormat(d3.format('d'))).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('text').attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text('context size K_ctx');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -innerH / 2).attr('y', -46).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text('test NLL per target (mean ± errBars)');

      const drawSeries = (means: number[], SEs: number[], color: string, label: string) => {
        const line = d3.line<number>().x((_, i) => xScale(payload.K_ctx[i])).y((d) => yScale(d));
        g.append('path').datum(means).attr('d', line).style('fill', 'none').style('stroke', color).style('stroke-width', 2);
        means.forEach((m, i) => {
          const x = xScale(payload.K_ctx[i]);
          const y = yScale(m);
          const eb = errBarScale * SEs[i];
          g.append('line').attr('x1', x).attr('x2', x).attr('y1', yScale(m - eb)).attr('y2', yScale(m + eb)).style('stroke', color).style('stroke-width', 1.5);
          g.append('circle').attr('cx', x).attr('cy', y).attr('r', 5).style('fill', color).style('stroke', 'white').style('stroke-width', 0.8);
        });
        return label;
      };
      if (showCnp) drawSeries(payload.cnpNllMean, payload.cnpNllSE, META_PALETTE[0], 'CNP');
      if (showLnp) drawSeries(payload.lnpNllMean, payload.lnpNllSE, META_PALETTE[1], 'Latent NP');

      // legend
      const legendG = g.append('g').attr('transform', `translate(${innerW - 130}, 8)`);
      if (showCnp) {
        legendG.append('rect').attr('width', 14).attr('height', 3).attr('y', 5).style('fill', META_PALETTE[0]);
        legendG.append('text').attr('x', 18).attr('y', 8).style('fill', 'var(--color-text)').style('font-size', '11px').text('CNP');
      }
      if (showLnp) {
        legendG.append('rect').attr('width', 14).attr('height', 3).attr('y', 23).style('fill', META_PALETTE[1]);
        legendG.append('text').attr('x', 18).attr('y', 26).style('fill', 'var(--color-text)').style('font-size', '11px').text('Latent NP');
      }
    },
    [payload, width, showCnp, showLnp, errBarScale],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'flex', gap: 12, marginBottom: 8, alignItems: 'center', fontSize: 11, flexWrap: 'wrap' }}>
        {payload ? (
          <>
            <label>
              <input type="checkbox" checked={showCnp} onChange={(e) => setShowCnp(e.target.checked)} /> CNP
            </label>
            <label>
              <input type="checkbox" checked={showLnp} onChange={(e) => setShowLnp(e.target.checked)} /> Latent NP
            </label>
            <label style={{ flex: 1 }}>
              error-bar scale ({errBarScale.toFixed(2)} × SE): {errBarScale === 1.96 ? '95% CI' : errBarScale === 1.0 ? '1 SE' : ''}
              <input type="range" min={0.5} max={3} step={0.5} value={errBarScale} onChange={(e) => setErrBarScale(Number(e.target.value))} style={{ width: '100%', maxWidth: 240 }} aria-label="error bar scale" />
            </label>
          </>
        ) : (
          <span style={{ color: 'var(--color-text-secondary)' }}>{error ? `Error loading data: ${error}` : 'Loading…'}</span>
        )}
      </div>
      <svg ref={svgRef} width={width} height={height} role="img" aria-label="sample efficiency curves" />
    </div>
  );
}
