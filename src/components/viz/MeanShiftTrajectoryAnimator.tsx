import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { meanShift } from './shared/unsupervised';
import { makeMoons, BLOB3, gaussianKdeGrid, CLUSTER_COLORS, type Dataset } from '../../data/clustering-data';

// =============================================================================
// MeanShiftTrajectoryAnimator — §3.5. Animated mean-shift trajectories with
// per-step play/pause/step controls.
// =============================================================================

const HEIGHT = 460;
const SM_BP = 640;
const GRID_N = 80;
const CONTOUR_COUNT = 10;

type DatasetKey = 'moons' | '3-blob';

function getDataset(key: DatasetKey): Dataset {
  return key === 'moons' ? makeMoons(200, 0.1, 42) : BLOB3();
}

export default function MeanShiftTrajectoryAnimator() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('moons');
  const [h, setH] = useState(0.30);
  const [committedH, setCommittedH] = useState(0.30);
  const [nQueries, setNQueries] = useState(12);
  const [showVector, setShowVector] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [step, setStep] = useState(0);

  const dataset = useMemo(() => getDataset(datasetKey), [datasetKey]);

  // Generate query starting points on a sparse grid.
  const startQueries = useMemo(() => {
    const queries: number[][] = [];
    const aspect = (dataset.xDomain[1] - dataset.xDomain[0]) / (dataset.yDomain[1] - dataset.yDomain[0]);
    const cols = Math.max(2, Math.round(Math.sqrt(nQueries * aspect)));
    const rows = Math.max(2, Math.ceil(nQueries / cols));
    for (let r = 0; r < rows && queries.length < nQueries; r++) {
      for (let c = 0; c < cols && queries.length < nQueries; c++) {
        const fx = (c + 0.5) / cols;
        const fy = (r + 0.5) / rows;
        const xv = dataset.xDomain[0] + fx * (dataset.xDomain[1] - dataset.xDomain[0]);
        const yv = dataset.yDomain[0] + fy * (dataset.yDomain[1] - dataset.yDomain[0]);
        queries.push([xv, yv]);
      }
    }
    return queries;
  }, [dataset, nQueries]);

  const trajectories = useMemo(() => {
    const X = dataset.X.map((p) => [p[0], p[1]]);
    const result = meanShift(X, startQueries, committedH, { maxIter: 60, returnHistory: true });
    return result.trajectories ?? [];
  }, [dataset, startQueries, committedH]);

  const maxStep = Math.max(0, trajectories.length - 1);

  // KDE contours.
  const kde = useMemo(() => {
    return gaussianKdeGrid(
      dataset.X, committedH,
      dataset.xDomain[0], dataset.xDomain[1], GRID_N,
      dataset.yDomain[0], dataset.yDomain[1], GRID_N,
    );
  }, [dataset, committedH]);

  // Animation loop.
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    if (!playing) {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }
    intervalRef.current = setInterval(() => {
      setStep((s) => {
        if (s >= maxStep) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, 120);
    return () => {
      if (intervalRef.current !== null) clearInterval(intervalRef.current);
    };
  }, [playing, maxStep]);

  useEffect(() => { setStep(0); setPlaying(false); }, [datasetKey, committedH, nQueries]);
  const w = containerWidth;
  const isMobile = w > 0 && w < SM_BP;

  const renderRef = useD3<SVGSVGElement>((svg) => {
    svg.selectAll('*').remove();
    if (w <= 0 || trajectories.length === 0) return;
    const m = { top: 36, right: 16, bottom: 36, left: 40 };
    const innerW = w - m.left - m.right;
    const innerH = HEIGHT - m.top - m.bottom;
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    svg.append('text')
      .attr('x', w / 2).attr('y', 22)
      .attr('text-anchor', 'middle')
      .style('font', '600 13px var(--font-sans)')
      .style('fill', 'var(--color-text)')
      .text(`step ${step}/${maxStep}, h=${committedH.toFixed(2)}, ${nQueries} trajectories`);

    const x = d3.scaleLinear().domain(dataset.xDomain).range([0, innerW]);
    const y = d3.scaleLinear().domain(dataset.yDomain).range([innerH, 0]);

    // Faint KDE backdrop.
    let maxV = 0;
    const flat = new Array<number>(GRID_N * GRID_N);
    for (let r = 0; r < GRID_N; r++) for (let c = 0; c < GRID_N; c++) {
      flat[r * GRID_N + c] = kde.values[r][c];
      if (kde.values[r][c] > maxV) maxV = kde.values[r][c];
    }
    const thresholds = d3.range(1, CONTOUR_COUNT + 1).map((i) => (i / (CONTOUR_COUNT + 1)) * maxV);
    const contoursGen = d3.contours().size([GRID_N, GRID_N]).thresholds(thresholds);
    const polys = contoursGen(flat);
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
      .style('fill', 'none')
      .style('stroke', 'var(--color-border)')
      .style('stroke-opacity', 0.6)
      .style('stroke-width', 0.7);

    g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(5)).style('color', 'var(--color-text-secondary)');
    g.append('g').call(d3.axisLeft(y).ticks(5)).style('color', 'var(--color-text-secondary)');

    // Data scatter (small dark dots).
    g.selectAll('.pt').data(dataset.X).enter().append('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1]))
      .attr('r', 1.5)
      .style('fill', 'var(--color-text-secondary)')
      .style('fill-opacity', 0.5);

    // Trajectories: render up to current step.
    const line = d3.line<[number, number]>().x((p) => x(p[0])).y((p) => y(p[1]));
    for (let q = 0; q < nQueries; q++) {
      const pts: [number, number][] = [];
      for (let t = 0; t <= step && t < trajectories.length; t++) {
        const p = trajectories[t][q];
        pts.push([p[0], p[1]]);
      }
      g.append('path')
        .attr('d', line(pts))
        .style('fill', 'none')
        .style('stroke', CLUSTER_COLORS[q % CLUSTER_COLORS.length])
        .style('stroke-width', 1.5)
        .style('stroke-opacity', 0.8);
      // Start circle.
      g.append('circle')
        .attr('cx', x(trajectories[0][q][0]))
        .attr('cy', y(trajectories[0][q][1]))
        .attr('r', 3.5)
        .style('fill', CLUSTER_COLORS[q % CLUSTER_COLORS.length])
        .style('stroke', 'white').style('stroke-width', 1);
      // Current-step marker.
      const cur = trajectories[Math.min(step, trajectories.length - 1)][q];
      g.append('path')
        .attr('d', d3.symbol().type(d3.symbolCross).size(80)())
        .attr('transform', `translate(${x(cur[0])},${y(cur[1])}) rotate(45)`)
        .style('fill', CLUSTER_COLORS[q % CLUSTER_COLORS.length])
        .style('stroke', 'white').style('stroke-width', 1);

      // Shift vector arrow on the active trajectory (last one for visibility).
      if (showVector && step > 0 && step < trajectories.length && q === 0) {
        const prev = trajectories[step - 1][q];
        const next = trajectories[step][q];
        const dx = next[0] - prev[0];
        const dy = next[1] - prev[1];
        const mag = Math.sqrt(dx * dx + dy * dy);
        if (mag > 1e-5) {
          g.append('line')
            .attr('x1', x(prev[0])).attr('y1', y(prev[1]))
            .attr('x2', x(next[0])).attr('y2', y(next[1]))
            .style('stroke', '#DC2626')
            .style('stroke-width', 2);
        }
      }
    }
  }, [trajectories, kde, step, dataset, w, committedH, nQueries, showVector, maxStep]);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap', marginBottom: 12 }}>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 180 }}>
          h (bandwidth): <strong style={{ color: 'var(--color-text)' }}>{h.toFixed(2)}</strong> {h !== committedH && <em>(release to apply)</em>}
          <input type="range" min={0.05} max={1.0} step={0.01}
            value={h}
            onChange={(e) => setH(+e.target.value)}
            onMouseUp={() => setCommittedH(h)}
            onTouchEnd={() => setCommittedH(h)}
            onKeyUp={() => setCommittedH(h)}
            aria-label="bandwidth h" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 160 }}>
          # trajectories: <strong style={{ color: 'var(--color-text)' }}>{nQueries}</strong>
          <input type="range" min={1} max={25} step={1} value={nQueries} onChange={(e) => setNQueries(+e.target.value)} aria-label="number of trajectories" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)' }}>
          dataset:
          <select value={datasetKey} onChange={(e) => setDatasetKey(e.target.value as DatasetKey)} aria-label="dataset">
            <option value="moons">moons</option>
            <option value="3-blob">3-blob</option>
          </select>
        </label>
        <div style={{ display: 'flex', gap: 6, alignItems: 'flex-end' }}>
          <button onClick={() => setPlaying((p) => !p)} aria-label="play or pause"
            style={{ padding: '6px 12px', borderRadius: 4, background: 'var(--color-accent)', color: 'white', border: 'none', fontSize: 12 }}>
            {playing ? 'pause' : 'play'}
          </button>
          <button onClick={() => setStep((s) => Math.min(s + 1, maxStep))} disabled={step >= maxStep}
            style={{ padding: '6px 12px', borderRadius: 4, background: 'var(--color-surface)', color: 'var(--color-text)', border: '1px solid var(--color-border)', fontSize: 12 }}>
            step
          </button>
          <button onClick={() => { setStep(0); setPlaying(false); }}
            style={{ padding: '6px 12px', borderRadius: 4, background: 'var(--color-surface)', color: 'var(--color-text)', border: '1px solid var(--color-border)', fontSize: 12 }}>
            reset
          </button>
        </div>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--color-text-secondary)', marginTop: 12 }}>
          <input type="checkbox" checked={showVector} onChange={(e) => setShowVector(e.target.checked)} />
          shift vector
        </label>
      </div>
      <svg ref={renderRef} width={w} height={HEIGHT} style={{ background: 'var(--color-surface)', borderRadius: 8 }} />
    </div>
  );
}
