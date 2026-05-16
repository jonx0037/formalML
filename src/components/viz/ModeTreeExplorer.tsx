import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { bandwidthSelectorForMeanShift, modeFinder } from './shared/unsupervised';
import { BLOB3, BLOB4, makeMoons, CLUSTER_COLORS, type Dataset } from '../../data/clustering-data';

// =============================================================================
// ModeTreeExplorer — §5.3 / §5.5. Left: M(h) step curve. Right: scatter with
// labels at the selected h. Honest blips honored (no smoothing).
// =============================================================================

const HEIGHT = 400;
const SM_BP = 640;

type DatasetKey = 'moons' | '3-blob' | '4-blob';

function getDataset(key: DatasetKey): Dataset {
  switch (key) {
    case 'moons': return makeMoons(200, 0.1, 42);
    case '3-blob': return BLOB3();
    case '4-blob': return BLOB4();
  }
}

export default function ModeTreeExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('moons');
  const [h, setH] = useState(0.40);
  const [committedH, setCommittedH] = useState(0.40);

  const dataset = useMemo(() => getDataset(datasetKey), [datasetKey]);

  const { hGrid, modeCounts, silverman } = useMemo(() => {
    // Build the log-spaced sweep ourselves to recover h values.
    const lo = Math.log(0.05);
    const hi = Math.log(2.0);
    const N = 50;
    const hG: number[] = [];
    for (let i = 0; i < N; i++) hG.push(Math.exp(lo + ((hi - lo) * i) / (N - 1)));
    const result = bandwidthSelectorForMeanShift(dataset.X, { mode: 'scree', hGrid: hG });
    return { hGrid: hG, modeCounts: result.modeCounts ?? [], silverman: result.silverman };
  }, [dataset]);

  const scatter = useMemo(() => {
    return modeFinder(dataset.X, committedH, { dedupTol: 1e-3 });
  }, [dataset, committedH]);

  const w = containerWidth;
  const isMobile = w > 0 && w < SM_BP;
  const panelW = (isMobile ? w : w / 2) - 8;

  const renderLeft = useD3<SVGSVGElement>((svg) => {
    svg.selectAll('*').remove();
    if (panelW <= 0 || modeCounts.length === 0) return;
    const m = { top: 32, right: 12, bottom: 40, left: 48 };
    const innerW = panelW - m.left - m.right;
    const innerH = HEIGHT - m.top - m.bottom;
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    svg.append('text').attr('x', panelW / 2).attr('y', 18).attr('text-anchor', 'middle')
      .style('font', '600 12px var(--font-sans)').style('fill', 'var(--color-text)')
      .text(`M(h) sweep — ${dataset.label}`);

    const x = d3.scaleLog().domain([hGrid[0], hGrid[hGrid.length - 1]]).range([0, innerW]);
    const maxM = Math.max(...modeCounts);
    const yMax = Math.min(maxM + 1, 30);
    const y = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

    g.append('g').attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(5, '.2f')).style('color', 'var(--color-text-secondary)');
    g.append('g').call(d3.axisLeft(y).ticks(5)).style('color', 'var(--color-text-secondary)');

    g.append('text').attr('x', innerW / 2).attr('y', innerH + 30)
      .attr('text-anchor', 'middle').style('font', '11px var(--font-sans)')
      .style('fill', 'var(--color-text-secondary)').text('bandwidth h (log)');
    g.append('text').attr('x', -innerH / 2).attr('y', -36).attr('transform', 'rotate(-90)')
      .attr('text-anchor', 'middle').style('font', '11px var(--font-sans)')
      .style('fill', 'var(--color-text-secondary)').text('M(h)');

    // Step curve.
    const line = d3.line<number>().x((_, i) => x(hGrid[i])).y((v) => y(Math.min(v, yMax))).curve(d3.curveStepAfter);
    g.append('path').attr('d', line(modeCounts))
      .style('fill', 'none').style('stroke', 'var(--color-accent)').style('stroke-width', 2);
    // Points to mark exact mode counts.
    g.selectAll('.pt').data(modeCounts).enter().append('circle')
      .attr('cx', (_, i) => x(hGrid[i]))
      .attr('cy', (d) => y(Math.min(d, yMax)))
      .attr('r', 2.5)
      .style('fill', 'var(--color-accent)');

    // Silverman marker.
    if (silverman > hGrid[0] && silverman < hGrid[hGrid.length - 1]) {
      g.append('line').attr('x1', x(silverman)).attr('y1', 0).attr('x2', x(silverman)).attr('y2', innerH)
        .style('stroke', 'var(--color-text)')
        .style('stroke-width', 1).style('stroke-dasharray', '4 4').style('opacity', 0.6);
      g.append('text').attr('x', x(silverman) + 4).attr('y', 12)
        .style('font', '10px var(--font-sans)').style('fill', 'var(--color-text-secondary)')
        .text(`h_S=${silverman.toFixed(2)}`);
    }

    // Current h marker.
    g.append('line').attr('x1', x(committedH)).attr('y1', 0).attr('x2', x(committedH)).attr('y2', innerH)
      .style('stroke', '#DC2626').style('stroke-width', 1.5).style('opacity', 0.7);
  }, [hGrid, modeCounts, silverman, committedH, panelW, dataset]);

  const renderRight = useD3<SVGSVGElement>((svg) => {
    svg.selectAll('*').remove();
    if (panelW <= 0) return;
    const m = { top: 32, right: 12, bottom: 40, left: 40 };
    const innerW = panelW - m.left - m.right;
    const innerH = HEIGHT - m.top - m.bottom;
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    svg.append('text').attr('x', panelW / 2).attr('y', 18).attr('text-anchor', 'middle')
      .style('font', '600 12px var(--font-sans)').style('fill', 'var(--color-text)')
      .text(`h=${committedH.toFixed(2)}, M(h)=${scatter.modes.length}`);

    const x = d3.scaleLinear().domain(dataset.xDomain).range([0, innerW]);
    const y = d3.scaleLinear().domain(dataset.yDomain).range([innerH, 0]);

    g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(5)).style('color', 'var(--color-text-secondary)');
    g.append('g').call(d3.axisLeft(y).ticks(5)).style('color', 'var(--color-text-secondary)');

    g.selectAll('.pt').data(dataset.X).enter().append('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1]))
      .attr('r', 2.5)
      .style('fill', (_, i) => CLUSTER_COLORS[scatter.labels[i] % CLUSTER_COLORS.length])
      .style('stroke', 'var(--color-bg)').style('stroke-width', 0.3);

    const cross = d3.symbol().type(d3.symbolCross).size(120);
    g.selectAll('.mode').data(scatter.modes).enter().append('path')
      .attr('d', cross)
      .attr('transform', (d) => `translate(${x(d[0])},${y(d[1])}) rotate(45)`)
      .style('fill', 'var(--color-text)').style('stroke', 'white').style('stroke-width', 1.5);
  }, [dataset, scatter, committedH, panelW]);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap', marginBottom: 12 }}>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 200 }}>
          h: <strong style={{ color: 'var(--color-text)' }}>{h.toFixed(2)}</strong> {h !== committedH && <em>(release to apply)</em>}
          <input type="range" min={0.05} max={2.0} step={0.01}
            value={h}
            onChange={(e) => setH(+e.target.value)}
            onMouseUp={() => setCommittedH(h)}
            onTouchEnd={() => setCommittedH(h)}
            onKeyUp={() => setCommittedH(h)}
            aria-label="bandwidth h" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)' }}>
          dataset:
          <select value={datasetKey} onChange={(e) => setDatasetKey(e.target.value as DatasetKey)} aria-label="dataset">
            <option value="moons">moons</option>
            <option value="3-blob">3-blob</option>
            <option value="4-blob">4-blob (varied std)</option>
          </select>
        </label>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16 }}>
        <svg ref={renderLeft} width={panelW} height={HEIGHT} style={{ background: 'var(--color-surface)', borderRadius: 8 }} />
        <svg ref={renderRight} width={panelW} height={HEIGHT} style={{ background: 'var(--color-surface)', borderRadius: 8 }} />
      </div>
    </div>
  );
}
