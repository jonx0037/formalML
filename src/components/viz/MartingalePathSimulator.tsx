import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

const MARGIN = { top: 16, right: 12, bottom: 36, left: 50 };

// Seeded LCG PRNG
function lcg(index: number, seed: number): number {
  let s = (seed * 2147483647 + index * 16807 + 12345) & 0x7fffffff;
  s = (s * 16807 + 12345) & 0x7fffffff;
  s = (s * 16807 + 12345) & 0x7fffffff;
  return (s & 0x7fffffff) / 0x7fffffff;
}

function seededNormal(i: number, seed: number): number {
  const u1 = Math.max(lcg(i * 2, seed), 1e-10);
  const u2 = lcg(i * 2 + 1, seed);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ─── Process generators ───

type ProcessType = 'random-walk' | 'drifted-walk' | 'polya-urn' | 'gbm' | 'regime-switching';

interface ProcessConfig {
  label: string;
  description: string;
  isMartingale: boolean;
  steps: number;
}

const PROCESSES: Record<ProcessType, ProcessConfig> = {
  'random-walk': { label: 'Simple Random Walk', description: 'Martingale — symmetric ±1 steps', isMartingale: true, steps: 500 },
  'drifted-walk': { label: 'Random Walk + Drift', description: 'Submartingale — upward drift +0.02/step', isMartingale: false, steps: 500 },
  'polya-urn': { label: 'Pólya Urn', description: 'Martingale — fraction of red balls converges a.s.', isMartingale: true, steps: 200 },
  'gbm': { label: 'Geometric Brownian Motion', description: 'Martingale after discounting — log-normal asset price', isMartingale: true, steps: 252 },
  'regime-switching': { label: 'Regime-Switching Volatility', description: 'Non-constant conditional variance — regime colors show state', isMartingale: true, steps: 500 },
};

interface PathData {
  values: number[];
  regimes?: number[]; // 0 = low vol, 1 = high vol
}

function generatePath(type: ProcessType, pathIdx: number, seed: number): PathData {
  const cfg = PROCESSES[type];
  const n = cfg.steps;
  const baseSeed = seed * 1000 + pathIdx * 137;

  switch (type) {
    case 'random-walk': {
      const values = [0];
      for (let i = 1; i <= n; i++) {
        const step = lcg(i, baseSeed) < 0.5 ? 1 : -1;
        values.push(values[i - 1] + step);
      }
      return { values };
    }
    case 'drifted-walk': {
      const values = [0];
      for (let i = 1; i <= n; i++) {
        const step = lcg(i, baseSeed) < 0.5 ? 1 : -1;
        values.push(values[i - 1] + step + 0.02);
      }
      return { values };
    }
    case 'polya-urn': {
      let red = 1, blue = 1;
      const values = [0.5]; // initial fraction
      for (let i = 1; i <= n; i++) {
        const u = lcg(i, baseSeed);
        if (u < red / (red + blue)) red++; else blue++;
        values.push(red / (red + blue));
      }
      return { values };
    }
    case 'gbm': {
      const sigma = 0.2;
      const dt = 1 / 252;
      const values = [100];
      for (let i = 1; i <= n; i++) {
        const z = seededNormal(i, baseSeed);
        // Martingale: drift = -sigma^2/2 so E[S_{t+1}|F_t] = S_t
        values.push(values[i - 1] * Math.exp(-0.5 * sigma * sigma * dt + sigma * Math.sqrt(dt) * z));
      }
      return { values };
    }
    case 'regime-switching': {
      const sigmaLow = 0.1, sigmaHigh = 0.4;
      const pSwitch = 0.02;
      let regime = 0; // start in low vol
      const values = [0];
      const regimes = [0];
      for (let i = 1; i <= n; i++) {
        // Switch regime?
        if (lcg(i * 3, baseSeed) < pSwitch) regime = 1 - regime;
        const sigma = regime === 0 ? sigmaLow : sigmaHigh;
        const z = seededNormal(i, baseSeed);
        values.push(values[i - 1] + sigma * z);
        regimes.push(regime);
      }
      return { values, regimes };
    }
  }
}

// ─── Color palette for paths ───
const PATH_COLORS = [
  '#0F6E56', '#534AB7', '#D97706', '#DC2626', '#2563EB',
  '#059669', '#7C3AED', '#EA580C', '#E11D48', '#0284C7',
  '#10B981', '#8B5CF6', '#F59E0B', '#F43F5E', '#3B82F6',
  '#14B8A6', '#A855F7', '#FB923C', '#F87171', '#60A5FA',
];
const REGIME_COLORS = ['#22C55E', '#F59E0B']; // green = low, orange = high

export default function MartingalePathSimulator() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const [processType, setProcessType] = useState<ProcessType>('random-walk');
  const [numPaths, setNumPaths] = useState(5);
  const [seed, setSeed] = useState(42);

  const isDesktop = (containerWidth || 0) > 700;
  const readoutW = isDesktop ? 200 : containerWidth;
  const chartW = isDesktop ? containerWidth - readoutW - 24 : containerWidth;
  const chartH = Math.min(360, Math.max(240, chartW * 0.55));

  const paths = useMemo(() => {
    const result: PathData[] = [];
    for (let p = 0; p < numPaths; p++) {
      result.push(generatePath(processType, p, seed));
    }
    return result;
  }, [processType, numPaths, seed]);

  // Compute stats
  const stats = useMemo(() => {
    const lastValues = paths.map(p => p.values[p.values.length - 1]);
    const mean = lastValues.reduce((a, b) => a + b, 0) / lastValues.length;
    const initial = paths[0].values[0];
    return { mean, initial };
  }, [paths]);

  // D3 rendering
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || chartW <= 0) return;

    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const cfg = PROCESSES[processType];
    const steps = cfg.steps;

    // Compute y extent across all paths
    let yMin = Infinity, yMax = -Infinity;
    for (const p of paths) {
      for (const v of p.values) {
        if (v < yMin) yMin = v;
        if (v > yMax) yMax = v;
      }
    }
    const yPad = (yMax - yMin) * 0.08 || 1;
    yMin -= yPad;
    yMax += yPad;

    const xScale = d3.scaleLinear().domain([0, steps]).range([MARGIN.left, chartW - MARGIN.right]);
    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([chartH - MARGIN.bottom, MARGIN.top]);

    // Axes
    sel.append('g')
      .attr('transform', `translate(0,${chartH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(6).tickSize(-chartH + MARGIN.top + MARGIN.bottom))
      .call(g => g.selectAll('.tick line').style('stroke', 'var(--color-border)').style('stroke-dasharray', '3 3'))
      .call(g => g.selectAll('.tick text').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'))
      .call(g => g.select('.domain').remove());

    sel.append('g')
      .attr('transform', `translate(${MARGIN.left},0)`)
      .call(d3.axisLeft(yScale).ticks(6).tickSize(-chartW + MARGIN.left + MARGIN.right))
      .call(g => g.selectAll('.tick line').style('stroke', 'var(--color-border)').style('stroke-dasharray', '3 3'))
      .call(g => g.selectAll('.tick text').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'))
      .call(g => g.select('.domain').remove());

    // Axis labels
    sel.append('text').attr('x', chartW / 2).attr('y', chartH - 2)
      .attr('text-anchor', 'middle').style('font-size', '11px').style('font-family', 'var(--font-serif)').style('fill', 'var(--color-text-secondary)')
      .text('Step n');

    // Initial value reference line
    const initVal = paths[0].values[0];
    sel.append('line')
      .attr('x1', MARGIN.left).attr('x2', chartW - MARGIN.right)
      .attr('y1', yScale(initVal)).attr('y2', yScale(initVal))
      .style('stroke', 'var(--color-text-secondary)').style('stroke-width', 1).style('stroke-dasharray', '6 4').style('opacity', 0.5);

    // Draw paths
    const lineGen = d3.line<number>()
      .x((_, i) => xScale(i))
      .y(d => yScale(d));

    for (let p = 0; p < paths.length; p++) {
      const pathData = paths[p];
      if (processType === 'regime-switching' && pathData.regimes) {
        // Split by regime for coloring
        let segStart = 0;
        for (let i = 1; i <= pathData.values.length; i++) {
          const currentRegime = pathData.regimes[Math.min(i, pathData.regimes.length - 1)];
          const prevRegime = pathData.regimes[i - 1];
          if (i === pathData.values.length || currentRegime !== prevRegime) {
            const segValues = pathData.values.slice(segStart, i);
            sel.append('path')
              .datum(segValues)
              .attr('d', d3.line<number>().x((_, j) => xScale(segStart + j)).y(d => yScale(d)))
              .style('fill', 'none')
              .style('stroke', REGIME_COLORS[prevRegime])
              .style('stroke-width', 1.5)
              .style('opacity', numPaths > 10 ? 0.5 : 0.75);
            segStart = i - 1;
          }
        }
      } else {
        sel.append('path')
          .datum(pathData.values)
          .attr('d', lineGen)
          .style('fill', 'none')
          .style('stroke', PATH_COLORS[p % PATH_COLORS.length])
          .style('stroke-width', 1.5)
          .style('opacity', numPaths > 10 ? 0.5 : 0.75);
      }
    }
  }, [paths, processType, chartW, chartH, numPaths]);

  const handleProcess = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setProcessType(e.target.value as ProcessType);
  }, []);

  if (!containerWidth) {
    return <div ref={containerRef} style={{ minHeight: 300 }} />;
  }

  const cfg = PROCESSES[processType];

  return (
    <div
      ref={containerRef}
      style={{
        border: '1px solid var(--color-border)',
        borderRadius: '8px',
        padding: '16px',
        background: 'var(--color-surface)',
        marginTop: '1.5rem',
        marginBottom: '1.5rem',
      }}
    >
      <div style={{ fontFamily: 'var(--font-sans)', fontSize: '13px', fontWeight: 600, marginBottom: '12px', color: 'var(--color-text)' }}>
        Martingale Path Simulator
      </div>

      <div style={{ display: 'flex', flexDirection: isDesktop ? 'row' : 'column', gap: '12px' }}>
        {/* Chart */}
        <svg role="img" aria-label="Martingale path simulator visualization" ref={svgRef} width={chartW} height={chartH} style={{ border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)' }} />

        {/* Readout */}
        <div style={{ width: readoutW, fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
          <div style={{ padding: '10px', background: 'var(--color-muted-bg)', borderRadius: '6px', border: '1px solid var(--color-border)' }}>
            <div style={{ fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: '11px', color: 'var(--color-text)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Properties
            </div>
            <div style={{ marginBottom: '6px' }}>
              <span style={{ color: 'var(--color-text)' }}>Type:</span>{' '}
              <span style={{ color: cfg.isMartingale ? 'var(--color-accent)' : '#D97706' }}>
                {cfg.isMartingale ? 'Martingale' : 'Submartingale'}
              </span>
            </div>
            <div style={{ marginBottom: '6px' }}>
              Initial: <strong style={{ color: 'var(--color-text)' }}>{stats.initial.toFixed(2)}</strong>
            </div>
            <div style={{ marginBottom: '6px' }}>
              Mean (final): <strong style={{ color: 'var(--color-text)' }}>{stats.mean.toFixed(2)}</strong>
            </div>
            <div style={{ marginBottom: '6px' }}>
              E[M_{'{n+1}'}|ℱₙ] {cfg.isMartingale ? '=' : '≥'} Mₙ
            </div>
            {processType === 'regime-switching' && (
              <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid var(--color-border)' }}>
                <div style={{ marginBottom: '4px' }}>Regime legend:</div>
                <div style={{ display: 'flex', gap: '12px' }}>
                  <span><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: REGIME_COLORS[0], marginRight: 4, verticalAlign: 'middle' }} />Low σ</span>
                  <span><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: REGIME_COLORS[1], marginRight: 4, verticalAlign: 'middle' }} />High σ</span>
                </div>
              </div>
            )}
          </div>
          <div style={{ marginTop: '8px', fontSize: '11px', lineHeight: 1.5, color: 'var(--color-text-secondary)' }}>
            {cfg.description}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div style={{ marginTop: '14px', display: 'flex', flexWrap: 'wrap', gap: '16px', alignItems: 'center' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
          Process:
          <select value={processType} onChange={handleProcess} style={{
            padding: '3px 8px', fontSize: '12px', fontFamily: 'var(--font-mono)',
            border: '1px solid var(--color-border)', borderRadius: '4px',
            background: 'var(--color-surface)', color: 'var(--color-text)',
          }}>
            {(Object.keys(PROCESSES) as ProcessType[]).map(k => (
              <option key={k} value={k}>{PROCESSES[k].label}</option>
            ))}
          </select>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
          Paths: <strong style={{ color: 'var(--color-text)', minWidth: '20px' }}>{numPaths}</strong>
          <input type="range" min={1} max={20} step={1} value={numPaths}
            onChange={e => setNumPaths(parseInt(e.target.value))}
            style={{ width: '100px', accentColor: 'var(--color-accent)' }} />
        </label>
        <button
          onClick={() => setSeed(s => s + 1)}
          style={{
            padding: '4px 12px', fontSize: '12px', fontFamily: 'var(--font-mono)',
            border: '1px solid var(--color-border)', borderRadius: '4px',
            background: 'transparent', color: 'var(--color-text-secondary)', cursor: 'pointer',
          }}
        >
          New Sample
        </button>
      </div>
    </div>
  );
}
