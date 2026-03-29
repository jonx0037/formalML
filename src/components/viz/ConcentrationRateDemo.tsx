import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

const MARGIN = { top: 12, right: 8, bottom: 28, left: 40 };
const MAX_N = 1000;
const NUM_PATHS = 10;
const NUM_MC = 500;

// Seeded LCG PRNG (same as ConvergenceModesDemo)
function lcg(index: number, seed: number): number {
  let s = (seed * 2147483647 + index * 16807 + 12345) & 0x7fffffff;
  s = (s * 16807 + 12345) & 0x7fffffff;
  s = (s * 16807 + 12345) & 0x7fffffff;
  return (s & 0x7fffffff) / 0x7fffffff;
}

type DistKey = 'uniform' | 'beta' | 'bernoulli';

interface DistConfig {
  label: string;
  mean: number;
  variance: number;
  range: number; // b - a for Hoeffding
  sample: (i: number, seed: number) => number;
}

const DISTRIBUTIONS: Record<DistKey, DistConfig> = {
  uniform: {
    label: 'Uniform[0,1]',
    mean: 0.5,
    variance: 1 / 12,
    range: 1,
    sample: (i, seed) => lcg(i, seed),
  },
  beta: {
    label: 'Beta(2,5)',
    mean: 2 / 7,
    variance: (2 * 5) / (49 * 8),
    range: 1,
    sample: (i, seed) => {
      // Rejection sampling approximation for Beta(2,5)
      // Use inverse CDF via bisection for seeded reproducibility
      const u = lcg(i, seed);
      // Simple approximation: use Kumaraswamy(2,5) as close match
      // CDF: 1 - (1 - x^2)^5, inverse: (1 - (1-u)^{1/5})^{1/2}
      return Math.sqrt(1 - Math.pow(1 - u, 0.2));
    },
  },
  bernoulli: {
    label: 'Bernoulli(0.5)',
    mean: 0.5,
    variance: 0.25,
    range: 1,
    sample: (i, seed) => (lcg(i, seed) < 0.5 ? 0 : 1),
  },
};

const PANEL_COLORS = ['#0F6E56', '#534AB7', '#2563EB', '#D97706'];
const PATH_PALETTE = ['#0F6E56', '#534AB7', '#D97706', '#DC2626', '#2563EB', '#059669', '#7C3AED', '#EA580C', '#0891B2', '#4F46E5'];

export default function ConcentrationRateDemo() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgRefs = [useRef<SVGSVGElement>(null), useRef<SVGSVGElement>(null), useRef<SVGSVGElement>(null), useRef<SVGSVGElement>(null)];

  const [dist, setDist] = useState<DistKey>('uniform');
  const [epsilon, setEpsilon] = useState(0.1);
  const [delta, setDelta] = useState(0.05);
  const [currentN, setCurrentN] = useState(MAX_N);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(3);
  const [seed, setSeed] = useState(42);
  const stepRef = useRef(MAX_N);
  const animRef = useRef<number>(0);

  const isDesktop = (containerWidth || 0) > 640;
  const panelW = isDesktop ? Math.floor((containerWidth - 12) / 2) : containerWidth;
  const panelH = Math.min(220, Math.max(160, panelW * 0.55));

  const cfg = DISTRIBUTIONS[dist];

  // ─── Pre-compute sample paths & MC data ───

  const pathData = useMemo(() => {
    // Generate NUM_PATHS sample paths of running means
    const paths: number[][] = [];
    for (let p = 0; p < NUM_PATHS; p++) {
      const vals: number[] = [];
      let sum = 0;
      for (let n = 1; n <= MAX_N; n++) {
        sum += cfg.sample(p * MAX_N + n, seed);
        vals.push(sum / n);
      }
      paths.push(vals);
    }
    return paths;
  }, [cfg, seed]);

  const mcDeviations = useMemo(() => {
    // For each n, compute 500 deviations |X_bar_n - mu|
    // Pre-compute cumulative sums for each MC run
    const runs: Float64Array[] = [];
    for (let r = 0; r < NUM_MC; r++) {
      const cumSum = new Float64Array(MAX_N);
      let sum = 0;
      for (let n = 0; n < MAX_N; n++) {
        sum += cfg.sample(r * MAX_N + n + NUM_PATHS * MAX_N, seed);
        cumSum[n] = sum;
      }
      runs.push(cumSum);
    }
    return runs;
  }, [cfg, seed]);

  // ─── Heatmap data (static per distribution) ───

  const heatmapData = useMemo(() => {
    const gridSize = 50;
    const nVals = Array.from({ length: gridSize }, (_, i) => 10 + (990) * i / (gridSize - 1));
    const epsVals = Array.from({ length: gridSize }, (_, i) => 0.01 + 0.49 * i / (gridSize - 1));
    const data: { n: number; eps: number; ratio: number }[] = [];

    for (const nVal of nVals) {
      for (const eps of epsVals) {
        const hoeffding = 2 * Math.exp(-2 * nVal * eps * eps / (cfg.range * cfg.range));
        const chebyshev = Math.min(cfg.variance / (nVal * eps * eps), 1);
        const ratio = chebyshev > 1e-15 ? hoeffding / chebyshev : 1;
        data.push({ n: nVal, eps, ratio: Math.min(ratio, 2) });
      }
    }
    return { data, nVals, epsVals, gridSize };
  }, [cfg]);

  // ─── Animation ───

  useEffect(() => {
    if (!isPlaying) return;
    let frame = 0;
    const tick = () => {
      frame++;
      if (frame % Math.max(1, 6 - speed) === 0) {
        stepRef.current = Math.min(stepRef.current + 1, MAX_N);
        setCurrentN(stepRef.current);
      }
      if (stepRef.current >= MAX_N) {
        setIsPlaying(false);
        return;
      }
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animRef.current);
  }, [isPlaying, speed]);

  const handlePlay = useCallback(() => {
    if (stepRef.current >= MAX_N) {
      stepRef.current = 1;
      setCurrentN(1);
    }
    setIsPlaying(true);
  }, []);

  const handleReset = useCallback(() => {
    setIsPlaying(false);
    stepRef.current = MAX_N;
    setCurrentN(MAX_N);
  }, []);

  const handleDist = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setDist(e.target.value as DistKey);
    setIsPlaying(false);
    stepRef.current = MAX_N;
    setCurrentN(MAX_N);
  }, []);

  // ─── Panel 1: Sample paths with confidence bands ───

  useEffect(() => {
    const svg = svgRefs[0].current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const n = currentN;
    const xScale = d3.scaleLinear().domain([0, MAX_N]).range([MARGIN.left, panelW - MARGIN.right]);
    const yMax = Math.max(cfg.mean + 0.5, 1);
    const yMin = Math.min(cfg.mean - 0.5, 0);
    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([panelH - MARGIN.bottom, MARGIN.top]);

    // Axes
    sel.append('g').attr('transform', `translate(0,${panelH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(4).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').remove(); });

    // Mean line
    sel.append('line').attr('x1', MARGIN.left).attr('x2', panelW - MARGIN.right)
      .attr('y1', yScale(cfg.mean)).attr('y2', yScale(cfg.mean))
      .style('stroke', 'var(--color-text-secondary)').style('stroke-dasharray', '4 3').style('opacity', 0.5);

    // Confidence bands (from current n onward, show width at n)
    if (n > 1) {
      const sigma = Math.sqrt(cfg.variance);
      const chebHalf = sigma / (Math.sqrt(n) * Math.sqrt(delta));
      const hoeffHalf = Math.sqrt(Math.log(2 / delta) / (2 * n)) * cfg.range;

      // Chebyshev band
      const chebTop = Math.min(cfg.mean + chebHalf, yMax);
      const chebBot = Math.max(cfg.mean - chebHalf, yMin);
      sel.append('rect')
        .attr('x', MARGIN.left).attr('width', panelW - MARGIN.left - MARGIN.right)
        .attr('y', yScale(chebTop)).attr('height', Math.max(yScale(chebBot) - yScale(chebTop), 0))
        .style('fill', PANEL_COLORS[1]).style('opacity', 0.08);

      // Hoeffding band (tighter)
      const hoeffTop = Math.min(cfg.mean + hoeffHalf, yMax);
      const hoeffBot = Math.max(cfg.mean - hoeffHalf, yMin);
      sel.append('rect')
        .attr('x', MARGIN.left).attr('width', panelW - MARGIN.left - MARGIN.right)
        .attr('y', yScale(hoeffTop)).attr('height', Math.max(yScale(hoeffBot) - yScale(hoeffTop), 0))
        .style('fill', PANEL_COLORS[0]).style('opacity', 0.12);
    }

    // Sample paths (up to currentN)
    const lineGen = d3.line<number>().x((_, i) => xScale(i + 1)).y(d => yScale(d));
    for (let p = 0; p < pathData.length; p++) {
      sel.append('path')
        .datum(pathData[p].slice(0, n))
        .attr('d', lineGen)
        .style('fill', 'none').style('stroke', PATH_PALETTE[p % PATH_PALETTE.length]).style('stroke-width', 1.2).style('opacity', 0.7);
    }

    // Title
    sel.append('text').attr('x', MARGIN.left + 4).attr('y', MARGIN.top + 10)
      .style('font-size', '10px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', PANEL_COLORS[0])
      .text('Sample Paths & Bands');
  }, [pathData, currentN, panelW, panelH, cfg, delta]);

  // ─── Panel 2: Tail histogram at current n ───

  useEffect(() => {
    const svg = svgRefs[1].current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const n = currentN;
    // Compute deviations at this n
    const deviations: number[] = [];
    for (let r = 0; r < NUM_MC; r++) {
      const mean = mcDeviations[r][n - 1] / n;
      deviations.push(Math.abs(mean - cfg.mean));
    }

    const maxDev = d3.max(deviations) || 0.5;
    const bins = d3.bin().domain([0, maxDev]).thresholds(30)(deviations);

    const xScale = d3.scaleLinear().domain([0, maxDev]).range([MARGIN.left, panelW - MARGIN.right]);
    const yScale = d3.scaleLinear().domain([0, d3.max(bins, b => b.length) || 1]).range([panelH - MARGIN.bottom, MARGIN.top]);

    // Axes
    sel.append('g').attr('transform', `translate(0,${panelH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(4).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').remove(); });

    // Histogram bars
    sel.selectAll('.bar').data(bins).join('rect')
      .attr('class', 'bar')
      .attr('x', d => xScale(d.x0 ?? 0))
      .attr('y', d => yScale(d.length))
      .attr('width', d => Math.max(xScale(d.x1 ?? 0) - xScale(d.x0 ?? 0) - 1, 1))
      .attr('height', d => panelH - MARGIN.bottom - yScale(d.length))
      .style('fill', PANEL_COLORS[1]).style('opacity', 0.6);

    // Threshold lines
    const sigma = Math.sqrt(cfg.variance);
    const chebThresh = sigma / (Math.sqrt(n) * Math.sqrt(delta));
    const hoeffThresh = Math.sqrt(Math.log(2 / delta) / (2 * n)) * cfg.range;

    if (chebThresh <= maxDev) {
      sel.append('line')
        .attr('x1', xScale(chebThresh)).attr('x2', xScale(chebThresh))
        .attr('y1', MARGIN.top).attr('y2', panelH - MARGIN.bottom)
        .style('stroke', PANEL_COLORS[1]).style('stroke-width', 2).style('stroke-dasharray', '4 3');
    }
    if (hoeffThresh <= maxDev) {
      sel.append('line')
        .attr('x1', xScale(hoeffThresh)).attr('x2', xScale(hoeffThresh))
        .attr('y1', MARGIN.top).attr('y2', panelH - MARGIN.bottom)
        .style('stroke', PANEL_COLORS[0]).style('stroke-width', 2);
    }

    sel.append('text').attr('x', MARGIN.left + 4).attr('y', MARGIN.top + 10)
      .style('font-size', '10px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', PANEL_COLORS[1])
      .text(`Deviation Histogram (n=${n})`);
  }, [mcDeviations, currentN, panelW, panelH, cfg, delta]);

  // ─── Panel 3: Log-scale convergence rates ───

  useEffect(() => {
    const svg = svgRefs[2].current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const nPts = 100;
    const ns = Array.from({ length: nPts }, (_, i) => 10 + (MAX_N - 10) * i / (nPts - 1));

    const chebBound = ns.map(n => Math.min(cfg.variance / (n * epsilon * epsilon), 1));
    const hoeffBound = ns.map(n => Math.min(2 * Math.exp(-2 * n * epsilon * epsilon / (cfg.range * cfg.range)), 1));

    const xScale = d3.scaleLog().domain([10, MAX_N]).range([MARGIN.left, panelW - MARGIN.right]);
    const yScale = d3.scaleLog().domain([1e-8, 2]).range([panelH - MARGIN.bottom, MARGIN.top]).clamp(true);

    sel.append('g').attr('transform', `translate(0,${panelH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(3, ',.0f').tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').remove(); });

    sel.append('g').attr('transform', `translate(${MARGIN.left},0)`)
      .call(d3.axisLeft(yScale).ticks(4, '.0e').tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').remove(); });

    const lineGen = d3.line<number>()
      .defined(d => d > 1e-15 && isFinite(d))
      .x((_, i) => xScale(ns[i]))
      .y(d => yScale(d));

    sel.append('path').datum(chebBound).attr('d', lineGen)
      .style('fill', 'none').style('stroke', PANEL_COLORS[1]).style('stroke-width', 2).style('stroke-dasharray', '6 3');
    sel.append('path').datum(hoeffBound).attr('d', lineGen)
      .style('fill', 'none').style('stroke', PANEL_COLORS[0]).style('stroke-width', 2);

    // Current n marker
    if (currentN >= 10) {
      const chebAtN = Math.min(cfg.variance / (currentN * epsilon * epsilon), 1);
      const hoeffAtN = Math.min(2 * Math.exp(-2 * currentN * epsilon * epsilon / (cfg.range * cfg.range)), 1);
      if (chebAtN > 1e-8) {
        sel.append('circle').attr('cx', xScale(currentN)).attr('cy', yScale(chebAtN))
          .attr('r', 3).style('fill', PANEL_COLORS[1]);
      }
      if (hoeffAtN > 1e-8) {
        sel.append('circle').attr('cx', xScale(currentN)).attr('cy', yScale(hoeffAtN))
          .attr('r', 3).style('fill', PANEL_COLORS[0]);
      }
    }

    sel.append('text').attr('x', MARGIN.left + 4).attr('y', MARGIN.top + 10)
      .style('font-size', '10px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', PANEL_COLORS[2])
      .text(`Convergence Rates (ε=${epsilon})`);
  }, [cfg, epsilon, currentN, panelW, panelH]);

  // ─── Panel 4: Heatmap (Hoeffding / Chebyshev ratio) ───

  useEffect(() => {
    const svg = svgRefs[3].current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const { data, nVals, epsVals, gridSize } = heatmapData;

    const xScale = d3.scaleLinear().domain([nVals[0], nVals[nVals.length - 1]]).range([MARGIN.left, panelW - MARGIN.right]);
    const yScale = d3.scaleLinear().domain([epsVals[0], epsVals[epsVals.length - 1]]).range([panelH - MARGIN.bottom, MARGIN.top]);

    const colorScale = d3.scaleSequential(d3.interpolateRdYlGn).domain([1.5, 0]); // green = Hoeffding wins

    const cellW = (panelW - MARGIN.left - MARGIN.right) / gridSize;
    const cellH = (panelH - MARGIN.top - MARGIN.bottom) / gridSize;

    sel.selectAll('.cell').data(data).join('rect')
      .attr('class', 'cell')
      .attr('x', (_, i) => xScale(nVals[Math.floor(i / gridSize)]))
      .attr('y', (_, i) => yScale(epsVals[i % gridSize]) - cellH)
      .attr('width', cellW + 0.5)
      .attr('height', cellH + 0.5)
      .style('fill', d => colorScale(d.ratio));

    // Axes
    sel.append('g').attr('transform', `translate(0,${panelH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(4).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').remove(); });

    sel.append('g').attr('transform', `translate(${MARGIN.left},0)`)
      .call(d3.axisLeft(yScale).ticks(4).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').remove(); });

    sel.append('text').attr('x', MARGIN.left + 4).attr('y', MARGIN.top + 10)
      .style('font-size', '10px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', PANEL_COLORS[3])
      .text('Hoeffding/Chebyshev Ratio');

    // Color legend
    const legendW = 60;
    const legendH = 8;
    const lx = panelW - MARGIN.right - legendW - 4;
    const ly = MARGIN.top + 4;
    const defs = sel.append('defs');
    const grad = defs.append('linearGradient').attr('id', 'heatmap-grad');
    grad.append('stop').attr('offset', '0%').attr('stop-color', colorScale(0));
    grad.append('stop').attr('offset', '100%').attr('stop-color', colorScale(1.5));
    sel.append('rect').attr('x', lx).attr('y', ly).attr('width', legendW).attr('height', legendH)
      .style('fill', 'url(#heatmap-grad)').style('stroke', 'var(--color-border)').style('stroke-width', 0.5);
    sel.append('text').attr('x', lx - 2).attr('y', ly + legendH + 8)
      .style('font-size', '7px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('H wins');
    sel.append('text').attr('x', lx + legendW + 2).attr('y', ly + legendH + 8)
      .attr('text-anchor', 'end').style('font-size', '7px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('C wins');
  }, [heatmapData, panelW, panelH]);

  // ─── Render ───

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
        Concentration Rate Demo
      </div>

      {/* 2×2 grid */}
      <div style={{ display: 'grid', gridTemplateColumns: isDesktop ? '1fr 1fr' : '1fr', gap: '8px' }}>
        {svgRefs.map((ref, i) => (
          <svg role="img" aria-label={`Concentration rate demo — ${['Sample paths and bands', 'Deviation histogram', 'Convergence rates', 'Hoeffding/Chebyshev ratio'][i]}`}
            key={i}
            ref={ref}
            width={panelW}
            height={panelH}
            style={{
              border: '1px solid var(--color-border)',
              borderRadius: '6px',
              background: 'var(--color-muted-bg)',
            }}
          />
        ))}
      </div>

      {/* Legend */}
      <div style={{ marginTop: '8px', display: 'flex', flexWrap: 'wrap', gap: '12px', fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
        <span><span style={{ display: 'inline-block', width: 12, height: 12, background: PANEL_COLORS[0], borderRadius: 2, verticalAlign: 'middle', marginRight: 4 }} /> Hoeffding</span>
        <span><span style={{ display: 'inline-block', width: 12, height: 12, background: PANEL_COLORS[1], borderRadius: 2, verticalAlign: 'middle', marginRight: 4 }} /> Chebyshev</span>
      </div>

      {/* Controls */}
      <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '14px', alignItems: 'center' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
          Distribution:
          <select value={dist} onChange={handleDist} style={{
            padding: '3px 8px', fontSize: '12px', fontFamily: 'var(--font-mono)',
            border: '1px solid var(--color-border)', borderRadius: '4px',
            background: 'var(--color-surface)', color: 'var(--color-text)',
          }}>
            {(Object.keys(DISTRIBUTIONS) as DistKey[]).map(k => (
              <option key={k} value={k}>{DISTRIBUTIONS[k].label}</option>
            ))}
          </select>
        </label>

        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
          ε = {epsilon.toFixed(2)}
          <input type="range" min={0.02} max={0.5} step={0.02} value={epsilon}
            onChange={e => setEpsilon(parseFloat(e.target.value))}
            style={{ width: '70px', accentColor: 'var(--color-accent)' }} />
        </label>

        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
          δ = {delta.toFixed(2)}
          <input type="range" min={0.01} max={0.5} step={0.01} value={delta}
            onChange={e => setDelta(parseFloat(e.target.value))}
            style={{ width: '70px', accentColor: 'var(--color-accent)' }} />
        </label>

        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
          Speed:
          <input type="range" min={1} max={5} step={1} value={speed} onChange={e => setSpeed(parseInt(e.target.value))}
            style={{ width: '60px', accentColor: 'var(--color-accent)' }} />
        </label>

        <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
          n = <strong style={{ color: 'var(--color-text)' }}>{currentN}</strong>
        </span>

        <div style={{ display: 'flex', gap: '6px' }}>
          <button onClick={isPlaying ? () => setIsPlaying(false) : handlePlay} style={{
            padding: '4px 12px', fontSize: '12px', fontFamily: 'var(--font-mono)',
            border: '1px solid var(--color-border)', borderRadius: '4px',
            background: isPlaying ? 'var(--color-definition-bg)' : 'transparent',
            color: isPlaying ? 'var(--color-accent)' : 'var(--color-text-secondary)', cursor: 'pointer',
          }}>
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button onClick={handleReset} style={{
            padding: '4px 12px', fontSize: '12px', fontFamily: 'var(--font-mono)',
            border: '1px solid var(--color-border)', borderRadius: '4px',
            background: 'transparent', color: 'var(--color-text-secondary)', cursor: 'pointer',
          }}>
            Reset
          </button>
          <button onClick={() => setSeed(s => s + 1)} style={{
            padding: '4px 12px', fontSize: '12px', fontFamily: 'var(--font-mono)',
            border: '1px solid var(--color-border)', borderRadius: '4px',
            background: 'transparent', color: 'var(--color-text-secondary)', cursor: 'pointer',
          }}>
            New Sample
          </button>
        </div>
      </div>
    </div>
  );
}
