import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { basinOfAttractionMap } from './shared/unsupervised';
import { BLOB3, makeMoons, CLUSTER_COLORS, type Dataset, type Pt } from '../../data/clustering-data';

// =============================================================================
// BasinOfAttractionMap — §7.5 signature visualization.
// Filled raster of basin labels via canvas-style data-URI image (NOT n×n rects).
// Bandwidth slider uses commit-on-release per CLAUDE.md slider-perf rule.
// =============================================================================

const HEIGHT = 480;
const SM_BP = 640;

type DatasetKey = 'moons' | '3-blob';

function getDataset(key: DatasetKey): Dataset {
  return key === 'moons' ? makeMoons(200, 0.1, 42) : BLOB3();
}

function hexToRgb(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff];
}

function basinPng(labels: number[][], modes: Pt[]): string {
  const ny = labels.length;
  const nx = labels[0]?.length ?? 0;
  if (!nx || !ny) return '';
  // Component is client:only="react", so document is always defined here.
  // OffscreenCanvas doesn't support synchronous toDataURL, so we stick with
  // a regular canvas.
  const canvas = document.createElement('canvas');
  canvas.width = nx;
  canvas.height = ny;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(nx, ny);
  const colors = modes.map((_, i) => hexToRgb(CLUSTER_COLORS[i % CLUSTER_COLORS.length]));
  for (let r = 0; r < ny; r++) {
    for (let c = 0; c < nx; c++) {
      // Flip Y: image row 0 is the top, but our data row 0 is yLo (bottom).
      const src = ny - 1 - r;
      const lab = labels[src][c];
      const [R, G, B] = colors[lab] || [200, 200, 200];
      const idx = (r * nx + c) * 4;
      img.data[idx] = R;
      img.data[idx + 1] = G;
      img.data[idx + 2] = B;
      img.data[idx + 3] = 100; // semi-transparent so scatter overlays
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL();
}

export default function BasinOfAttractionMap() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('moons');
  const [h, setH] = useState(0.40);
  const [committedH, setCommittedH] = useState(0.40);
  const [gridRes, setGridRes] = useState(80);
  const [committedGridRes, setCommittedGridRes] = useState(80);

  const dataset = useMemo(() => getDataset(datasetKey), [datasetKey]);

  const basin = useMemo(() => {
    return basinOfAttractionMap(
      dataset.X.map((p) => [p[0], p[1]]),
      committedH,
      {
        x: dataset.xDomain,
        y: dataset.yDomain,
        nx: committedGridRes,
        ny: committedGridRes,
      },
    );
  }, [dataset, committedH, committedGridRes]);

  const dataUri = useMemo(() => basinPng(basin.labels, basin.modes as Pt[]), [basin]);

  const w = containerWidth;
  const isMobile = w > 0 && w < SM_BP;

  const renderRef = useD3<SVGSVGElement>((svg) => {
    svg.selectAll('*').remove();
    if (w <= 0) return;
    const m = { top: 36, right: 16, bottom: 36, left: 40 };
    const innerW = w - m.left - m.right;
    const innerH = HEIGHT - m.top - m.bottom;
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    svg.append('text').attr('x', w / 2).attr('y', 22).attr('text-anchor', 'middle')
      .style('font', '600 13px var(--font-sans)').style('fill', 'var(--color-text)')
      .text(`${dataset.label}, h=${committedH.toFixed(2)}, ${committedGridRes}×${committedGridRes} grid, M(h)=${basin.modes.length}`);

    const x = d3.scaleLinear().domain(dataset.xDomain).range([0, innerW]);
    const y = d3.scaleLinear().domain(dataset.yDomain).range([innerH, 0]);

    if (dataUri) {
      g.append('image')
        .attr('href', dataUri)
        .attr('x', 0).attr('y', 0)
        .attr('width', innerW).attr('height', innerH)
        .attr('preserveAspectRatio', 'none');
    }

    g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(5)).style('color', 'var(--color-text-secondary)');
    g.append('g').call(d3.axisLeft(y).ticks(5)).style('color', 'var(--color-text-secondary)');

    // Data scatter colored by their data-mode label.
    // Look up each data point's basin from the grid (nearest cell).
    const xDom = dataset.xDomain, yDom = dataset.yDomain;
    const cellX = (xDom[1] - xDom[0]) / (committedGridRes - 1);
    const cellY = (yDom[1] - yDom[0]) / (committedGridRes - 1);
    g.selectAll('.pt').data(dataset.X).enter().append('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1]))
      .attr('r', 2.5)
      .style('fill', (d) => {
        const cx = Math.max(0, Math.min(committedGridRes - 1, Math.round((d[0] - xDom[0]) / cellX)));
        const cy = Math.max(0, Math.min(committedGridRes - 1, Math.round((d[1] - yDom[0]) / cellY)));
        return CLUSTER_COLORS[basin.labels[cy][cx] % CLUSTER_COLORS.length];
      })
      .style('stroke', 'white').style('stroke-width', 0.7);

    // Modes as large crosses.
    const cross = d3.symbol().type(d3.symbolCross).size(180);
    g.selectAll('.mode').data(basin.modes).enter().append('path')
      .attr('d', cross)
      .attr('transform', (d) => `translate(${x(d[0])},${y(d[1])}) rotate(45)`)
      .style('fill', 'var(--color-text)').style('stroke', 'white').style('stroke-width', 2);
  }, [basin, dataUri, dataset, w, committedH, committedGridRes]);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap', marginBottom: 12 }}>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 200 }}>
          h (bandwidth): <strong style={{ color: 'var(--color-text)' }}>{h.toFixed(2)}</strong> {h !== committedH && <em>(release to apply)</em>}
          <input type="range" min={0.05} max={1.0} step={0.01}
            value={h}
            onChange={(e) => setH(+e.target.value)}
            onMouseUp={() => setCommittedH(h)}
            onTouchEnd={() => setCommittedH(h)}
            onKeyUp={() => setCommittedH(h)}
            aria-label="bandwidth h" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 180 }}>
          grid resolution: <strong style={{ color: 'var(--color-text)' }}>{gridRes}</strong> {gridRes !== committedGridRes && <em>(release to apply)</em>}
          <input type="range" min={40} max={160} step={10}
            value={gridRes}
            onChange={(e) => setGridRes(+e.target.value)}
            onMouseUp={() => setCommittedGridRes(gridRes)}
            onTouchEnd={() => setCommittedGridRes(gridRes)}
            onKeyUp={() => setCommittedGridRes(gridRes)}
            aria-label="grid resolution" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)' }}>
          dataset:
          <select value={datasetKey} onChange={(e) => setDatasetKey(e.target.value as DatasetKey)} aria-label="dataset">
            <option value="moons">moons</option>
            <option value="3-blob">3-blob</option>
          </select>
        </label>
      </div>
      <svg ref={renderRef} width={w} height={HEIGHT} style={{ background: 'var(--color-surface)', borderRadius: 8 }} />
    </div>
  );
}
