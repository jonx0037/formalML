import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { gaussianKdeGrid, findGridModes, BLOB3, BLOB4, makeMoons, makeCircles, type Dataset } from '../../data/clustering-data';

// =============================================================================
// KDESurfaceWithModes — §2.4. Filled-contour heatmap of $\hat f_h$ with overlaid
// data scatter and discovered modes (red crosses).
// =============================================================================

const HEIGHT = 460;
const SM_BP = 640;
const GRID_N = 100;
const CONTOUR_COUNT = 14;

type DatasetKey = '3-blob' | '4-blob' | 'moons' | 'circles';

function getDataset(key: DatasetKey): Dataset {
  switch (key) {
    case '3-blob': return BLOB3();
    case '4-blob': return BLOB4();
    case 'moons': return makeMoons(200, 0.1, 42);
    case 'circles': return makeCircles(200, 0.5, 0.05, 42);
  }
}

export default function KDESurfaceWithModes() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('3-blob');
  const [h, setH] = useState(0.40);
  const [committedH, setCommittedH] = useState(0.40);
  const isMobile = containerWidth > 0 && containerWidth < SM_BP;
  const w = containerWidth;

  const dataset = useMemo(() => getDataset(datasetKey), [datasetKey]);

  const { values, xVals, yVals, modes } = useMemo(() => {
    const [xLo, xHi] = dataset.xDomain;
    const [yLo, yHi] = dataset.yDomain;
    const grid = gaussianKdeGrid(dataset.X, committedH, xLo, xHi, GRID_N, yLo, yHi, GRID_N);
    // Threshold modes by 25% of grid maximum to suppress spurious low-density bumps.
    let maxV = 0;
    for (const row of grid.values) for (const v of row) if (v > maxV) maxV = v;
    const threshold = maxV * 0.05;
    const modes = findGridModes(grid.values, grid.xVals, grid.yVals, threshold);
    return { values: grid.values, xVals: grid.xVals, yVals: grid.yVals, modes };
  }, [dataset, committedH]);

  const renderRef = useD3<SVGSVGElement>((svg) => {
    svg.selectAll('*').remove();
    if (w <= 0) return;
    const m = { top: 36, right: 16, bottom: 36, left: 40 };
    const innerW = w - m.left - m.right;
    const innerH = HEIGHT - m.top - m.bottom;
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    svg.append('text')
      .attr('x', w / 2).attr('y', 22)
      .attr('text-anchor', 'middle')
      .style('font', '600 13px var(--font-sans)')
      .style('fill', 'var(--color-text)')
      .text(`${dataset.label}, Gaussian kernel, h=${committedH.toFixed(2)}, M(h)=${modes.length}`);

    const x = d3.scaleLinear().domain(dataset.xDomain).range([0, innerW]);
    const y = d3.scaleLinear().domain(dataset.yDomain).range([innerH, 0]);

    // Flatten values for d3.contours. Float64Array minimizes GC pressure
    // versus a plain Array<number>; cast satisfies d3.contours' typing.
    const flat = new Float64Array(GRID_N * GRID_N);
    let maxV = 0;
    for (let r = 0; r < GRID_N; r++) {
      for (let c = 0; c < GRID_N; c++) {
        flat[r * GRID_N + c] = values[r][c];
        if (values[r][c] > maxV) maxV = values[r][c];
      }
    }
    const thresholds = d3.range(1, CONTOUR_COUNT + 1).map((i) => (i / (CONTOUR_COUNT + 1)) * maxV);
    const contoursGen = d3.contours().size([GRID_N, GRID_N]).thresholds(thresholds);
    const polys = contoursGen(flat as unknown as number[]);
    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, maxV]);

    // Path projection from grid coords [0, GRID_N] to data coords then to pixels.
    const xProj = (gx: number) => x(dataset.xDomain[0] + ((dataset.xDomain[1] - dataset.xDomain[0]) * gx) / (GRID_N - 1));
    const yProj = (gy: number) => y(dataset.yDomain[0] + ((dataset.yDomain[1] - dataset.yDomain[0]) * gy) / (GRID_N - 1));
    const proj = d3.geoTransform({
      point(gx: number, gy: number) {
        (this as unknown as { stream: d3.GeoStream }).stream.point(xProj(gx), yProj(gy));
      },
    });
    const path = d3.geoPath(proj as unknown as d3.GeoProjection);

    g.append('g')
      .selectAll('path')
      .data(polys)
      .enter()
      .append('path')
      .attr('d', path as any)
      .style('fill', (d) => colorScale(d.value))
      .style('stroke', 'var(--color-bg)')
      .style('stroke-opacity', 0.3)
      .style('stroke-width', 0.5);

    // Axes.
    g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(5)).style('color', 'var(--color-text-secondary)');
    g.append('g').call(d3.axisLeft(y).ticks(5)).style('color', 'var(--color-text-secondary)');

    // Data scatter.
    g.selectAll('.pt').data(dataset.X).enter().append('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1]))
      .attr('r', 2)
      .style('fill', 'white')
      .style('stroke', 'black')
      .style('stroke-width', 0.5);

    // Modes as red crosses.
    const cross = d3.symbol().type(d3.symbolCross).size(150);
    g.selectAll('.mode').data(modes).enter().append('path')
      .attr('d', cross)
      .attr('transform', (d) => `translate(${x(d[0])},${y(d[1])}) rotate(45)`)
      .style('fill', '#DC2626')
      .style('stroke', 'white')
      .style('stroke-width', 1.5);
  }, [values, xVals, yVals, modes, dataset, w, committedH]);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap', marginBottom: 12 }}>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 200 }}>
          h (bandwidth): <strong style={{ color: 'var(--color-text)' }}>{h.toFixed(2)}</strong> {h !== committedH && <em>(release to apply)</em>}
          <input type="range" min={0.05} max={2.0} step={0.01}
            value={h}
            onChange={(e) => setH(+e.target.value)}
            onMouseUp={() => setCommittedH(h)}
            onTouchEnd={() => setCommittedH(h)}
            onKeyUp={() => setCommittedH(h)}
            aria-label="bandwidth h" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 140 }}>
          dataset:
          <select value={datasetKey} onChange={(e) => setDatasetKey(e.target.value as DatasetKey)} aria-label="dataset">
            <option value="3-blob">3-blob</option>
            <option value="4-blob">4-blob (varied std)</option>
            <option value="moons">moons</option>
            <option value="circles">circles</option>
          </select>
        </label>
      </div>
      <svg ref={renderRef} width={w} height={HEIGHT} style={{ background: 'var(--color-surface)', borderRadius: 8 }} />
    </div>
  );
}
