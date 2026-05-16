import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { meanShift, gaussianKdeLogSurrogate } from './shared/unsupervised';
import { BLOB3, makeMoons, mulberry32, type Dataset } from '../../data/clustering-data';

// =============================================================================
// ConvergenceDiagnostic — §4.5. Two panels: log f̂_h vs t (monotone-ascent) and
// log step-size vs t (geometric decay near a mode).
// =============================================================================

const HEIGHT = 360;
const SM_BP = 640;

type DatasetKey = '3-blob' | 'moons';

function getDataset(key: DatasetKey): Dataset {
  return key === '3-blob' ? BLOB3() : makeMoons(200, 0.1, 42);
}

export default function ConvergenceDiagnostic() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('3-blob');
  const [h, setH] = useState(0.65);
  const [committedH, setCommittedH] = useState(0.65);
  const [B, setB] = useState(50);

  const dataset = useMemo(() => getDataset(datasetKey), [datasetKey]);

  const { trajectories, densityCurves, stepCurves } = useMemo(() => {
    // B random starts within the data's bbox.
    const rng = mulberry32(2026);
    const [xLo, xHi] = dataset.xDomain;
    const [yLo, yHi] = dataset.yDomain;
    const starts: number[][] = [];
    for (let i = 0; i < B; i++) {
      starts.push([xLo + (xHi - xLo) * rng(), yLo + (yHi - yLo) * rng()]);
    }
    const result = meanShift(dataset.X, starts, committedH, { maxIter: 60, returnHistory: true });
    const trajs = result.trajectories ?? [];
    // Density curves (log surrogate) and step-size curves.
    const dens: number[][] = [];
    const steps: number[][] = [];
    for (let q = 0; q < B; q++) {
      const dCurve: number[] = [];
      const sCurve: number[] = [];
      for (let t = 0; t < trajs.length; t++) {
        const pt = trajs[t][q];
        dCurve.push(gaussianKdeLogSurrogate(dataset.X, pt, committedH));
        if (t > 0) {
          const prev = trajs[t - 1][q];
          const dx = pt[0] - prev[0];
          const dy = pt[1] - prev[1];
          sCurve.push(Math.sqrt(dx * dx + dy * dy));
        }
      }
      dens.push(dCurve);
      steps.push(sCurve);
    }
    return { trajectories: trajs, densityCurves: dens, stepCurves: steps };
  }, [dataset, committedH, B]);

  const w = containerWidth;
  const isMobile = w > 0 && w < SM_BP;
  const panelW = (isMobile ? w : w / 2) - 8;

  const renderLeft = useD3<SVGSVGElement>((svg) => {
    svg.selectAll('*').remove();
    if (panelW <= 0 || densityCurves.length === 0) return;
    const m = { top: 30, right: 12, bottom: 36, left: 50 };
    const innerW = panelW - m.left - m.right;
    const innerH = HEIGHT - m.top - m.bottom;
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    svg.append('text').attr('x', panelW / 2).attr('y', 16).attr('text-anchor', 'middle')
      .style('font', '600 12px var(--font-sans)').style('fill', 'var(--color-text)')
      .text('log f̂_h(x_t)  vs  iteration t');

    const maxT = densityCurves[0].length - 1;
    let yMin = Infinity, yMax = -Infinity;
    for (const c of densityCurves) for (const v of c) {
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
    const x = d3.scaleLinear().domain([0, maxT]).range([0, innerW]);
    const y = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]);

    g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(6)).style('color', 'var(--color-text-secondary)');
    g.append('g').call(d3.axisLeft(y).ticks(5)).style('color', 'var(--color-text-secondary)');

    const line = d3.line<number>().x((_, i) => x(i)).y((v) => y(v));
    g.selectAll('.curve').data(densityCurves).enter().append('path')
      .attr('d', (d) => line(d))
      .style('fill', 'none')
      .style('stroke', 'var(--color-accent)')
      .style('stroke-opacity', 0.3)
      .style('stroke-width', 1);
  }, [densityCurves, panelW]);

  const renderRight = useD3<SVGSVGElement>((svg) => {
    svg.selectAll('*').remove();
    if (panelW <= 0 || stepCurves.length === 0) return;
    const m = { top: 30, right: 12, bottom: 36, left: 50 };
    const innerW = panelW - m.left - m.right;
    const innerH = HEIGHT - m.top - m.bottom;
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    svg.append('text').attr('x', panelW / 2).attr('y', 16).attr('text-anchor', 'middle')
      .style('font', '600 12px var(--font-sans)').style('fill', 'var(--color-text)')
      .text('log₁₀ ||x_{t+1} - x_t||  vs  iteration t');

    const maxT = stepCurves[0].length;
    let logMin = Infinity, logMax = -Infinity;
    for (const c of stepCurves) for (const v of c) {
      if (v > 0) {
        const lv = Math.log10(v);
        if (lv < logMin) logMin = lv;
        if (lv > logMax) logMax = lv;
      }
    }
    if (!Number.isFinite(logMin)) logMin = -6;
    if (!Number.isFinite(logMax)) logMax = 0;
    const x = d3.scaleLinear().domain([0, maxT - 1]).range([0, innerW]);
    const y = d3.scaleLinear().domain([logMin, logMax]).range([innerH, 0]);

    g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(6)).style('color', 'var(--color-text-secondary)');
    g.append('g').call(d3.axisLeft(y).ticks(5)).style('color', 'var(--color-text-secondary)');

    const line = d3.line<number>().defined((v) => v > 0).x((_, i) => x(i)).y((v) => y(Math.log10(v)));
    g.selectAll('.curve').data(stepCurves).enter().append('path')
      .attr('d', (d) => line(d))
      .style('fill', 'none')
      .style('stroke', 'var(--color-accent)')
      .style('stroke-opacity', 0.3)
      .style('stroke-width', 1);

    // Tolerance dashed line.
    g.append('line')
      .attr('x1', 0).attr('y1', y(-6)).attr('x2', innerW).attr('y2', y(-6))
      .style('stroke', 'var(--color-text-secondary)')
      .style('stroke-width', 1)
      .style('stroke-dasharray', '4 4');
    g.append('text')
      .attr('x', innerW - 4).attr('y', y(-6) - 4)
      .attr('text-anchor', 'end')
      .style('font', '10px var(--font-sans)')
      .style('fill', 'var(--color-text-secondary)')
      .text('convergence tol');
  }, [stepCurves, panelW]);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap', marginBottom: 12 }}>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 180 }}>
          h: <strong style={{ color: 'var(--color-text)' }}>{h.toFixed(2)}</strong> {h !== committedH && <em>(release to apply)</em>}
          <input type="range" min={0.1} max={1.5} step={0.01}
            value={h}
            onChange={(e) => setH(+e.target.value)}
            onMouseUp={() => setCommittedH(h)}
            onTouchEnd={() => setCommittedH(h)}
            onKeyUp={() => setCommittedH(h)}
            aria-label="bandwidth h" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 160 }}>
          # trajectories B: <strong style={{ color: 'var(--color-text)' }}>{B}</strong>
          <input type="range" min={10} max={200} step={10} value={B} onChange={(e) => setB(+e.target.value)} aria-label="trajectory count" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)' }}>
          dataset:
          <select value={datasetKey} onChange={(e) => setDatasetKey(e.target.value as DatasetKey)} aria-label="dataset">
            <option value="3-blob">3-blob</option>
            <option value="moons">moons</option>
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
