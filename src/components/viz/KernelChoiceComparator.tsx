import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { meanShiftKernel, type KernelName } from './shared/unsupervised';
import { ari, BLOB3, makeMoons, CLUSTER_COLORS, type Dataset, type Pt } from '../../data/clustering-data';

// =============================================================================
// KernelChoiceComparator — §6.3. Three side-by-side mean-shift panels with
// different kernels at the same bandwidth.
// =============================================================================

const HEIGHT = 320;
const SM_BP = 640;

type DatasetKey = 'moons' | '3-blob';

function getDataset(key: DatasetKey): Dataset {
  return key === 'moons' ? makeMoons(200, 0.1, 42) : BLOB3();
}

function dedupAndLabel(endpoints: Pt[], tol: number): { labels: number[]; modes: Pt[] } {
  const modes: Pt[] = [];
  const labels: number[] = [];
  const t2 = tol * tol;
  for (const p of endpoints) {
    let matched = -1;
    for (let m = 0; m < modes.length; m++) {
      const dx = p[0] - modes[m][0];
      const dy = p[1] - modes[m][1];
      if (dx * dx + dy * dy < t2) { matched = m; break; }
    }
    if (matched < 0) { modes.push([p[0], p[1]]); labels.push(modes.length - 1); }
    else labels.push(matched);
  }
  return { labels, modes };
}

const KERNELS: KernelName[] = ['gaussian', 'epanechnikov', 'biweight', 'triweight'];

export default function KernelChoiceComparator() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('moons');
  const [h, setH] = useState(0.40);
  const [committedH, setCommittedH] = useState(0.40);

  const dataset = useMemo(() => getDataset(datasetKey), [datasetKey]);

  const panels = useMemo(() => {
    return KERNELS.map((k) => {
      // For compact-support kernels, h needs to be larger (their effective radius
      // is the bandwidth ball, not the Gaussian's effective ~3σ). Scale by 2.
      const hLocal = k === 'gaussian' ? committedH : committedH * 2;
      const result = meanShiftKernel(dataset.X.map((p) => [p[0], p[1]]), dataset.X.map((p) => [p[0], p[1]]), hLocal, k);
      const { labels, modes } = dedupAndLabel(result.finalPositions.map((p) => [p[0], p[1]]) as Pt[], 1e-3);
      return { kernel: k, labels, modes, hLocal };
    });
  }, [dataset, committedH]);

  const referenceLabels = panels[0].labels;
  const w = containerWidth;
  const isMobile = w > 0 && w < SM_BP;
  const cols = isMobile ? 2 : 4;
  const panelW = w / cols - 8;

  const renderPanel = (idx: number) => useD3<SVGSVGElement>((svg) => {
    svg.selectAll('*').remove();
    if (panelW <= 0 || !panels[idx]) return;
    const { kernel, labels, modes, hLocal } = panels[idx];
    const m = { top: 36, right: 8, bottom: 28, left: 32 };
    const innerW = panelW - m.left - m.right;
    const innerH = HEIGHT - m.top - m.bottom;
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    svg.append('text').attr('x', panelW / 2).attr('y', 16).attr('text-anchor', 'middle')
      .style('font', '600 11px var(--font-sans)').style('fill', 'var(--color-text)')
      .text(`${kernel} (h=${hLocal.toFixed(2)}, M=${modes.length})`);
    const ariVsGauss = idx === 0 ? 1.0 : ari(referenceLabels, labels);
    svg.append('text').attr('x', panelW / 2).attr('y', 30).attr('text-anchor', 'middle')
      .style('font', '10px var(--font-sans)').style('fill', 'var(--color-text-secondary)')
      .text(idx === 0 ? 'ref' : `ARI vs Gaussian = ${ariVsGauss.toFixed(3)}`);

    const x = d3.scaleLinear().domain(dataset.xDomain).range([0, innerW]);
    const y = d3.scaleLinear().domain(dataset.yDomain).range([innerH, 0]);

    g.selectAll('.pt').data(dataset.X).enter().append('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1]))
      .attr('r', 2)
      .style('fill', (_, i) => CLUSTER_COLORS[labels[i] % CLUSTER_COLORS.length]);

    const cross = d3.symbol().type(d3.symbolCross).size(80);
    g.selectAll('.mode').data(modes).enter().append('path')
      .attr('d', cross)
      .attr('transform', (d) => `translate(${x(d[0])},${y(d[1])}) rotate(45)`)
      .style('fill', 'var(--color-text)').style('stroke', 'white').style('stroke-width', 1);
  }, [panels, referenceLabels, panelW, dataset]);

  const ref0 = renderPanel(0);
  const ref1 = renderPanel(1);
  const ref2 = renderPanel(2);
  const ref3 = renderPanel(3);
  const refs = [ref0, ref1, ref2, ref3];

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap', marginBottom: 12 }}>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 200 }}>
          h (Gaussian; compact kernels scaled ×2): <strong style={{ color: 'var(--color-text)' }}>{h.toFixed(2)}</strong> {h !== committedH && <em>(release to apply)</em>}
          <input type="range" min={0.10} max={1.5} step={0.01}
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
          </select>
        </label>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: 8 }}>
        {refs.map((r, i) => (
          <svg key={i} ref={r} width={panelW} height={HEIGHT} style={{ background: 'var(--color-surface)', borderRadius: 8 }} />
        ))}
      </div>
    </div>
  );
}
