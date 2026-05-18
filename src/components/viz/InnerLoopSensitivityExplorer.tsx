// =============================================================================
// InnerLoopSensitivityExplorer.tsx
//
// §9.3 Inner-loop step-count sensitivity sweep. Toggle which methods to
// display, slider for the y-axis log/linear toggle, and a checkbox for the
// meta-training N=5 reference line.
//
// Data: /sample-data/meta-learning/inner_loop_sens.json
// Static fallback: /images/topics/meta-learning/16_inner_loop_sensitivity.png
// =============================================================================

import { useEffect, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { META_PALETTE } from './shared/meta-learning';

interface InnerLoopPayload {
  N_test: number[];
  MAML: number[];
  FOMAML: number[];
  Reptile: number[];
}

type Method = 'MAML' | 'FOMAML' | 'Reptile';
const METHODS: Method[] = ['MAML', 'FOMAML', 'Reptile'];

export default function InnerLoopSensitivityExplorer(): React.JSX.Element {
  const [payload, setPayload] = useState<InnerLoopPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [enabled, setEnabled] = useState<Record<Method, boolean>>({ MAML: true, FOMAML: true, Reptile: true });
  const [logY, setLogY] = useState(false);
  const [showRefLine, setShowRefLine] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetch('/sample-data/meta-learning/inner_loop_sens.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d: InnerLoopPayload) => {
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
      const margin = { top: 18, right: 16, bottom: 40, left: 56 };
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const xScale = d3.scaleLinear().domain([0, Math.max(...payload.N_test) + 1]).range([0, innerW]);
      const enabledArrays: number[][] = METHODS.filter((m) => enabled[m]).map((m) => payload[m]);
      const allY = enabledArrays.flat();
      if (allY.length === 0) {
        g.append('text').attr('x', innerW / 2).attr('y', innerH / 2).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '12px').text('Toggle at least one method to display.');
        return;
      }
      const yMin = Math.max(logY ? 0.5 : 0, Math.min(...allY) * 0.9);
      const yMax = Math.max(...allY) * 1.05;
      const yScale = logY ? d3.scaleLog().domain([yMin, yMax]).range([innerH, 0]) : d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]);
      g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xScale).tickValues(payload.N_test)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('text').attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text('test-time inner-loop step count N_test');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -innerH / 2).attr('y', -42).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text(`mean held-out query MSE${logY ? ' (log)' : ''}`);

      if (showRefLine) {
        g.append('line').attr('x1', xScale(5)).attr('x2', xScale(5)).attr('y1', 0).attr('y2', innerH).style('stroke', 'var(--color-text-secondary)').style('stroke-dasharray', '4 4').style('opacity', 0.7);
        g.append('text').attr('x', xScale(5) + 4).attr('y', 14).style('fill', 'var(--color-text-secondary)').style('font-size', '10px').text('N_meta = 5');
      }

      const markers: Record<Method, [string, string]> = {
        MAML: [META_PALETTE[0], 'circle'],
        FOMAML: [META_PALETTE[1], 'square'],
        Reptile: [META_PALETTE[2], 'triangle'],
      };
      METHODS.forEach((m) => {
        if (!enabled[m]) return;
        const [color] = markers[m];
        const data = payload[m];
        const line = d3.line<number>().x((_, i) => xScale(payload.N_test[i])).y((d) => yScale(d));
        g.append('path').datum(data).attr('d', line).style('fill', 'none').style('stroke', color).style('stroke-width', 1.8);
        data.forEach((v, i) => {
          g.append('circle').attr('cx', xScale(payload.N_test[i])).attr('cy', yScale(v)).attr('r', 5).style('fill', color).style('stroke', 'white').style('stroke-width', 0.8);
        });
      });
      // legend
      const legendG = g.append('g').attr('transform', `translate(${innerW - 100}, 8)`);
      METHODS.forEach((m, idx) => {
        if (!enabled[m]) return;
        const [color] = markers[m];
        legendG.append('rect').attr('width', 14).attr('height', 3).attr('y', 5 + idx * 16).style('fill', color);
        legendG.append('text').attr('x', 18).attr('y', 8 + idx * 16).style('fill', 'var(--color-text)').style('font-size', '11px').text(m);
      });
    },
    [payload, width, enabled, logY, showRefLine],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'flex', gap: 12, marginBottom: 8, alignItems: 'center', fontSize: 11, flexWrap: 'wrap' }}>
        {payload ? (
          <>
            {METHODS.map((m) => (
              <label key={m}>
                <input type="checkbox" checked={enabled[m]} onChange={(e) => setEnabled({ ...enabled, [m]: e.target.checked })} /> {m}
              </label>
            ))}
            <label>
              <input type="checkbox" checked={logY} onChange={(e) => setLogY(e.target.checked)} /> log y
            </label>
            <label>
              <input type="checkbox" checked={showRefLine} onChange={(e) => setShowRefLine(e.target.checked)} /> show N_meta=5 line
            </label>
          </>
        ) : (
          <span style={{ color: 'var(--color-text-secondary)' }}>{error ? `Error loading data: ${error}` : 'Loading…'}</span>
        )}
      </div>
      <svg ref={svgRef} width={width} height={height} role="img" aria-label="inner-loop sensitivity sweep" />
    </div>
  );
}
