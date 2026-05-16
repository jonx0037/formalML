import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { basinOfAttractionMap } from './shared/unsupervised';
import { ari, BLOB3, kmeans, makeMoons, CLUSTER_COLORS, type Dataset, type Pt } from '../../data/clustering-data';

// =============================================================================
// KMeansVoronoiLimit — §9.4. Row of mean-shift basin panels at decreasing h
// alongside a fixed k-means Voronoi panel.
// =============================================================================

const HEIGHT_ROW = 220;
const HEIGHT_ARI = 160;
const SM_BP = 640;
const GRID_RES = 60;

type DatasetKey = '3-blob' | 'moons';
function getDataset(key: DatasetKey): Dataset {
  return key === '3-blob' ? BLOB3() : makeMoons(200, 0.1, 42);
}

function hexToRgb(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff];
}

function rasterPng(labels: number[][]): string {
  const ny = labels.length, nx = labels[0]?.length ?? 0;
  if (!nx || !ny) return '';
  const canvas = document.createElement('canvas');
  canvas.width = nx;
  canvas.height = ny;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(nx, ny);
  for (let r = 0; r < ny; r++) {
    for (let c = 0; c < nx; c++) {
      const src = ny - 1 - r;
      const [R, G, B] = hexToRgb(CLUSTER_COLORS[labels[src][c] % CLUSTER_COLORS.length]);
      const i = (r * nx + c) * 4;
      img.data[i] = R; img.data[i + 1] = G; img.data[i + 2] = B; img.data[i + 3] = 110;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL();
}

// Build Voronoi labels by nearest-centroid assignment on the same grid.
function voronoiLabels(centers: Pt[], xDom: [number, number], yDom: [number, number], nx: number, ny: number): number[][] {
  const out: number[][] = [];
  const cellX = (xDom[1] - xDom[0]) / (nx - 1);
  const cellY = (yDom[1] - yDom[0]) / (ny - 1);
  for (let r = 0; r < ny; r++) {
    const row: number[] = [];
    const cy = yDom[0] + r * cellY;
    for (let c = 0; c < nx; c++) {
      const cx = xDom[0] + c * cellX;
      let best = 0, bestD = Infinity;
      for (let m = 0; m < centers.length; m++) {
        const dx = cx - centers[m][0];
        const dy = cy - centers[m][1];
        const d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = m; }
      }
      row.push(best);
    }
    out.push(row);
  }
  return out;
}

export default function KMeansVoronoiLimit() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('3-blob');
  const [K, setK] = useState(3);
  const [hCurrent, setHCurrent] = useState(0.65);
  const [committedH, setCommittedH] = useState(0.65);

  const dataset = useMemo(() => getDataset(datasetKey), [datasetKey]);

  // Three pre-set h values for the "decreasing h" row.
  const fixedHs = [committedH, committedH / 2, committedH / 4];

  const basinSet = useMemo(() => {
    return fixedHs.map((h) => basinOfAttractionMap(
      dataset.X.map((p) => [p[0], p[1]]),
      h,
      { x: dataset.xDomain, y: dataset.yDomain, nx: GRID_RES, ny: GRID_RES },
    ));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, committedH]);

  const km = useMemo(() => kmeans(dataset.X, K, 7), [dataset, K]);

  const voronoi = useMemo(() => voronoiLabels(km.centers, dataset.xDomain, dataset.yDomain, GRID_RES, GRID_RES), [km, dataset]);

  // ARI sweep.
  const ariSweep = useMemo(() => {
    const lo = 0.05, hi = 1.5, N = 30;
    const hs: number[] = [];
    const aris: number[] = [];
    for (let i = 0; i < N; i++) {
      const h = lo + ((hi - lo) * i) / (N - 1);
      hs.push(h);
      // Sample-as-queries basin map for label comparison.
      const grid = basinOfAttractionMap(dataset.X.map((p) => [p[0], p[1]]), h,
        { x: dataset.xDomain, y: dataset.yDomain, nx: 40, ny: 40 });
      // Get per-data-point labels by nearest grid cell.
      const xDom = dataset.xDomain, yDom = dataset.yDomain;
      const cellX = (xDom[1] - xDom[0]) / 39;
      const cellY = (yDom[1] - yDom[0]) / 39;
      const labels: number[] = [];
      for (const p of dataset.X) {
        const cx = Math.max(0, Math.min(39, Math.round((p[0] - xDom[0]) / cellX)));
        const cy = Math.max(0, Math.min(39, Math.round((p[1] - yDom[0]) / cellY)));
        labels.push(grid.labels[cy][cx]);
      }
      aris.push(ari(labels, km.labels));
    }
    return { hs, aris };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, K]);

  const w = containerWidth;
  const isMobile = w > 0 && w < SM_BP;
  const cols = isMobile ? 2 : 4; // 3 basins + 1 voronoi
  const panelW = w / cols - 6;

  const renderPanel = (data: { labels: number[][]; modes: number[][] } | null, label: string) =>
    useD3<SVGSVGElement>((svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0 || !data) return;
      const uri = rasterPng(data.labels);
      const m = { top: 28, right: 6, bottom: 24, left: 26 };
      const innerW = panelW - m.left - m.right;
      const innerH = HEIGHT_ROW - m.top - m.bottom;
      const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

      svg.append('text').attr('x', panelW / 2).attr('y', 16).attr('text-anchor', 'middle')
        .style('font', '600 11px var(--font-sans)').style('fill', 'var(--color-text)')
        .text(label);

      const x = d3.scaleLinear().domain(dataset.xDomain).range([0, innerW]);
      const y = d3.scaleLinear().domain(dataset.yDomain).range([innerH, 0]);
      if (uri) g.append('image').attr('href', uri).attr('width', innerW).attr('height', innerH).attr('preserveAspectRatio', 'none');
      g.selectAll('.pt').data(dataset.X).enter().append('circle')
        .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1]))
        .attr('r', 1.5).style('fill', 'var(--color-text)');
    }, [data, panelW, dataset]);

  const refs = [
    renderPanel(basinSet[0], `MS h=${fixedHs[0].toFixed(2)}`),
    renderPanel(basinSet[1], `MS h=${fixedHs[1].toFixed(2)}`),
    renderPanel(basinSet[2], `MS h=${fixedHs[2].toFixed(2)}`),
    renderPanel({ labels: voronoi, modes: km.centers }, `k-means K=${K} Voronoi`),
  ];

  const renderAri = useD3<SVGSVGElement>((svg) => {
    svg.selectAll('*').remove();
    if (w <= 0) return;
    const m = { top: 28, right: 16, bottom: 36, left: 50 };
    const innerW = w - m.left - m.right;
    const innerH = HEIGHT_ARI - m.top - m.bottom;
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    svg.append('text').attr('x', w / 2).attr('y', 16).attr('text-anchor', 'middle')
      .style('font', '600 11px var(--font-sans)').style('fill', 'var(--color-text)')
      .text(`ARI(mean-shift, k-means K=${K})  vs  h`);

    const x = d3.scaleLinear().domain([ariSweep.hs[0], ariSweep.hs[ariSweep.hs.length - 1]]).range([0, innerW]);
    const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);
    g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(6)).style('color', 'var(--color-text-secondary)');
    g.append('g').call(d3.axisLeft(y).ticks(5)).style('color', 'var(--color-text-secondary)');

    const line = d3.line<number>().x((_, i) => x(ariSweep.hs[i])).y((v) => y(v));
    g.append('path').attr('d', line(ariSweep.aris)).style('fill', 'none').style('stroke', 'var(--color-accent)').style('stroke-width', 2);
  }, [ariSweep, w, K]);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap', marginBottom: 12 }}>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 180 }}>
          h (mean-shift): <strong style={{ color: 'var(--color-text)' }}>{hCurrent.toFixed(2)}</strong> {hCurrent !== committedH && <em>(release to apply)</em>}
          <input type="range" min={0.10} max={1.5} step={0.01}
            value={hCurrent}
            onChange={(e) => setHCurrent(+e.target.value)}
            onMouseUp={() => setCommittedH(hCurrent)}
            onTouchEnd={() => setCommittedH(hCurrent)}
            onKeyUp={() => setCommittedH(hCurrent)}
            aria-label="bandwidth h" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 140 }}>
          K (k-means): <strong style={{ color: 'var(--color-text)' }}>{K}</strong>
          <input type="range" min={2} max={6} step={1} value={K} onChange={(e) => setK(+e.target.value)} aria-label="K for k-means" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)' }}>
          dataset:
          <select value={datasetKey} onChange={(e) => setDatasetKey(e.target.value as DatasetKey)} aria-label="dataset">
            <option value="3-blob">3-blob</option>
            <option value="moons">moons</option>
          </select>
        </label>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: 6, marginBottom: 12 }}>
        {refs.map((r, i) => (
          <svg key={i} ref={r} width={panelW} height={HEIGHT_ROW} style={{ background: 'var(--color-surface)', borderRadius: 8 }} />
        ))}
      </div>
      <svg ref={renderAri} width={w} height={HEIGHT_ARI} style={{ background: 'var(--color-surface)', borderRadius: 8 }} />
    </div>
  );
}
