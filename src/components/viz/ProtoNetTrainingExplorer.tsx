// =============================================================================
// ProtoNetTrainingExplorer.tsx
//
// §5.5 Prototypical Network training curve + held-out decision regions.
// Left panel: loss curve over 500 meta-iters (raw + smoothed); right panel:
// the decision-region map for one of two precomputed held-out tasks.
//
// Data: /sample-data/meta-learning/protonet_training.json
// Static fallback: /images/topics/meta-learning/{10,11}_*.png
// =============================================================================

import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { META_PALETTE } from './shared/meta-learning';

interface HeldOutTask {
  Xs: number[][];
  ys: number[];
  Xq: number[][];
  yq: number[];
  queryAcc: number;
  predGrid: number[];
}

interface ProtoNetPayload {
  metaIters: number;
  lossCurve: number[];
  accCurve: number[];
  smoothedLoss: number[];
  smoothedAcc: number[];
  gridX: number[];
  gridY: number[];
  heldOutTasks: HeldOutTask[];
}

const SM_BREAKPOINT = 640;

export default function ProtoNetTrainingExplorer(): React.JSX.Element {
  const [payload, setPayload] = useState<ProtoNetPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [taskIdx, setTaskIdx] = useState(0);
  const [showAcc, setShowAcc] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetch('/sample-data/meta-learning/protonet_training.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d: ProtoNetPayload) => {
        if (!cancelled) setPayload(d);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const task = payload?.heldOutTasks[taskIdx] || null;

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const isMobile = (containerWidth || 800) < SM_BREAKPOINT;
  const panelWidth = isMobile ? containerWidth || 320 : (containerWidth || 800) / 2 - 12;
  const panelHeight = 280;

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (!payload || panelWidth <= 0) return;
      const margin = { top: 22, right: 50, bottom: 36, left: 50 };
      const innerW = panelWidth - margin.left - margin.right;
      const innerH = panelHeight - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const xScale = d3.scaleLinear().domain([0, payload.metaIters]).range([0, innerW]);
      const lossYScale = d3.scaleLinear().domain([0, Math.max(...payload.lossCurve) * 1.05]).range([innerH, 0]);
      g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(lossYScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text('meta-iteration');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -innerH / 2).attr('y', -36).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-size', '11px').text('cross-entropy loss');
      const lossLine = d3.line<number>().x((_, i) => xScale(i)).y((d) => lossYScale(d));
      g.append('path').datum(payload.lossCurve).attr('d', lossLine).style('fill', 'none').style('stroke', META_PALETTE[0]).style('stroke-width', 0.5).style('opacity', 0.35);
      const w = payload.smoothedLoss.length < payload.lossCurve.length ? Math.floor((payload.lossCurve.length - payload.smoothedLoss.length) / 2) : 0;
      const xSm = payload.smoothedLoss.map((_, i) => i + w);
      const smLine = d3.line<number>().x((_, i) => xScale(xSm[i])).y((d) => lossYScale(d));
      g.append('path').datum(payload.smoothedLoss).attr('d', smLine).style('fill', 'none').style('stroke', META_PALETTE[0]).style('stroke-width', 1.5);
      if (showAcc) {
        const accYScale = d3.scaleLinear().domain([0, 1.05]).range([innerH, 0]);
        const accAxis = d3.axisRight(accYScale).ticks(5);
        g.append('g').attr('transform', `translate(${innerW}, 0)`).call(accAxis).selectAll('text').style('fill', META_PALETTE[2]);
        g.append('text').attr('transform', `translate(${innerW + 36}, ${innerH / 2}) rotate(90)`).attr('text-anchor', 'middle').style('fill', META_PALETTE[2]).style('font-size', '11px').text('query accuracy');
        const accLine = d3.line<number>().x((_, i) => xScale(i)).y((d) => accYScale(d));
        g.append('path').datum(payload.accCurve).attr('d', accLine).style('fill', 'none').style('stroke', META_PALETTE[2]).style('stroke-width', 1).style('opacity', 0.5);
        // chance line
        g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', accYScale(0.2)).attr('y2', accYScale(0.2)).style('stroke', 'var(--color-text-secondary)').style('stroke-dasharray', '3 3').style('opacity', 0.5);
        g.append('text').attr('x', 4).attr('y', accYScale(0.2) - 3).style('fill', 'var(--color-text-secondary)').style('font-size', '9px').text('chance');
      }
      g.append('text').attr('x', innerW / 2).attr('y', -8).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px').text('ProtoNet training');
    },
    [payload, showAcc, panelWidth],
  );

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (!payload || !task || panelWidth <= 0) return;
      const margin = { top: 22, right: 12, bottom: 18, left: 18 };
      const innerW = panelWidth - margin.left - margin.right;
      const innerH = panelHeight - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      // Compress grid (220×220 → ~60×60) and render as colored rects
      const gridW = payload.gridX.length;
      const gridH = payload.gridY.length;
      const cellW = innerW / gridW;
      const cellH = innerH / gridH;
      // Convert flat predGrid to lookup
      for (let i = 0; i < gridH; i++) {
        for (let j = 0; j < gridW; j++) {
          const c = task.predGrid[i * gridW + j];
          g.append('rect').attr('x', j * cellW).attr('y', i * cellH).attr('width', cellW + 0.5).attr('height', cellH + 0.5).style('fill', META_PALETTE[c]).style('opacity', 0.22);
        }
      }
      // Support points
      const xScale = d3.scaleLinear().domain([-4, 4]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([-4, 4]).range([innerH, 0]); // y axis points up
      task.Xs.forEach((pt, i) => {
        g.append('circle').attr('cx', xScale(pt[0])).attr('cy', yScale(pt[1])).attr('r', 4).style('fill', META_PALETTE[task.ys[i]]).style('stroke', 'white').style('stroke-width', 0.8);
      });
      task.Xq.forEach((pt, i) => {
        g.append('circle').attr('cx', xScale(pt[0])).attr('cy', yScale(pt[1])).attr('r', 1.5).style('fill', META_PALETTE[task.yq[i]]).style('opacity', 0.55);
      });
      g.append('text').attr('x', innerW / 2).attr('y', -8).attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '11px').text(`held-out task ${taskIdx + 1} — query acc = ${task.queryAcc.toFixed(2)}`);
    },
    [payload, task, taskIdx, panelWidth],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'flex', gap: 12, marginBottom: 8, alignItems: 'center', fontSize: 11 }}>
        {payload ? (
          <>
            <label>
              Held-out task:
              {payload.heldOutTasks.map((_, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => setTaskIdx(i)}
                  style={{
                    marginLeft: 6,
                    padding: '3px 9px',
                    border: '1px solid var(--color-border)',
                    borderRadius: 3,
                    background: taskIdx === i ? 'var(--color-accent)' : 'transparent',
                    color: taskIdx === i ? 'white' : 'var(--color-text)',
                    cursor: 'pointer',
                    fontSize: 11,
                  }}
                >
                  {i + 1}
                </button>
              ))}
            </label>
            <label style={{ marginLeft: 'auto' }}>
              <input type="checkbox" checked={showAcc} onChange={(e) => setShowAcc(e.target.checked)} /> show accuracy
            </label>
          </>
        ) : (
          <span style={{ color: 'var(--color-text-secondary)' }}>{error ? `Error loading data: ${error}` : 'Loading…'}</span>
        )}
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16 }}>
        <svg ref={leftRef} width={panelWidth} height={panelHeight} role="img" aria-label="ProtoNet training curve" />
        <svg ref={rightRef} width={panelWidth} height={panelHeight} role="img" aria-label="ProtoNet held-out decision regions" />
      </div>
    </div>
  );
}
