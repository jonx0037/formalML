import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { getTypewriterSequence } from '../../data/convergence-data';

const MARGIN = { top: 12, right: 8, bottom: 28, left: 40 };
const NUM_PATHS = 8;
const MAX_N = 500;

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

// Distribution samplers
type DistributionType = 'exponential' | 'uniform' | 'bernoulli';

interface DistConfig {
  label: string;
  mean: number;
  variance: number;
  sample: (i: number, seed: number) => number;
}

const DISTRIBUTIONS: Record<DistributionType, DistConfig> = {
  exponential: {
    label: 'Exponential(1)',
    mean: 1,
    variance: 1,
    sample: (i, seed) => -Math.log(Math.max(lcg(i, seed), 1e-10)),
  },
  uniform: {
    label: 'Uniform(0,1)',
    mean: 0.5,
    variance: 1 / 12,
    sample: (i, seed) => lcg(i, seed),
  },
  bernoulli: {
    label: 'Bernoulli(0.5)',
    mean: 0.5,
    variance: 0.25,
    sample: (i, seed) => lcg(i, seed) < 0.5 ? 0 : 1,
  },
};

const PANEL_COLORS = ['#0F6E56', '#534AB7', '#2563EB', '#D97706'];
const PATH_PALETTE = ['#0F6E56', '#534AB7', '#D97706', '#DC2626', '#2563EB', '#059669', '#7C3AED', '#EA580C'];

export default function ConvergenceModesDemo() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgRefs = [useRef<SVGSVGElement>(null), useRef<SVGSVGElement>(null), useRef<SVGSVGElement>(null), useRef<SVGSVGElement>(null)];
  const [dist, setDist] = useState<DistributionType>('exponential');
  const [currentN, setCurrentN] = useState(MAX_N);
  const [isPlaying, setIsPlaying] = useState(false);
  const [seed, setSeed] = useState(42);
  const [hoveredPanel, setHoveredPanel] = useState<number | null>(null);
  const stepRef = useRef(MAX_N);
  const animRef = useRef<number>(0);
  const [speed, setSpeed] = useState(3); // frames per animation tick

  const isDesktop = (containerWidth || 0) > 640;
  const panelW = isDesktop ? Math.floor((containerWidth - 12) / 2) : containerWidth;
  const panelH = Math.min(220, Math.max(160, panelW * 0.55));

  const distCfg = DISTRIBUTIONS[dist];

  // ─── Pre-compute data ───

  // Panel 1: A.S. convergence — X_n = Z_n / n → 0
  const asData = useMemo(() => {
    const paths: number[][] = [];
    for (let p = 0; p < NUM_PATHS; p++) {
      const vals = [0];
      for (let i = 1; i <= MAX_N; i++) {
        vals.push(seededNormal(i, seed + p * 1000) / i);
      }
      paths.push(vals);
    }
    return paths;
  }, [seed]);

  // Panel 2: Typewriter sequence (deterministic)
  const twData = useMemo(() => getTypewriterSequence(MAX_N), []);

  // Panel 3: SLLN — running average of i.i.d.
  const sllnData = useMemo(() => {
    const paths: number[][] = [];
    for (let p = 0; p < Math.min(NUM_PATHS, 4); p++) {
      const vals = [0];
      let cumsum = 0;
      for (let i = 1; i <= MAX_N; i++) {
        cumsum += distCfg.sample(i, seed + p * 1000);
        vals.push(cumsum / i);
      }
      paths.push(vals);
    }
    return paths;
  }, [seed, distCfg]);

  // Panel 4: CLT — histograms at increasing n
  const cltData = useMemo(() => {
    const sampleSizes = [5, 20, 50, 100, 200, 500];
    const nSamples = 2000;
    const histograms: { n: number; bins: { x0: number; x1: number; count: number }[] }[] = [];

    for (const n of sampleSizes) {
      const means: number[] = [];
      for (let s = 0; s < nSamples; s++) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
          sum += distCfg.sample(s * n + i, seed + 99);
        }
        const barX = sum / n;
        // Standardize: Z = (barX - mu) / (sigma / sqrt(n))
        const z = (barX - distCfg.mean) / (Math.sqrt(distCfg.variance) / Math.sqrt(n));
        means.push(z);
      }
      // Bin into histogram
      const binGen = d3.bin().domain([-4, 4]).thresholds(30);
      const bins = binGen(means);
      histograms.push({
        n,
        bins: bins.map(b => ({ x0: b.x0 ?? -4, x1: b.x1 ?? 4, count: b.length / nSamples })),
      });
    }
    return { sampleSizes, histograms };
  }, [seed, distCfg]);

  // Pick the CLT histogram closest to currentN
  const cltHistIdx = useMemo(() => {
    let best = 0;
    for (let i = 1; i < cltData.sampleSizes.length; i++) {
      if (cltData.sampleSizes[i] <= currentN) best = i;
    }
    return best;
  }, [currentN, cltData]);

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

  // ─── Render panels ───

  // Panel 1: A.S. convergence
  useEffect(() => {
    const svg = svgRefs[0].current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const n = currentN;
    const xScale = d3.scaleLinear().domain([0, MAX_N]).range([MARGIN.left, panelW - MARGIN.right]);
    const yScale = d3.scaleLinear().domain([-1.5, 1.5]).range([panelH - MARGIN.bottom, MARGIN.top]);

    // Axes
    sel.append('g').attr('transform', `translate(0,${panelH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(4).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').remove(); });

    // Zero line
    sel.append('line').attr('x1', MARGIN.left).attr('x2', panelW - MARGIN.right)
      .attr('y1', yScale(0)).attr('y2', yScale(0))
      .style('stroke', 'var(--color-text-secondary)').style('stroke-dasharray', '4 3').style('opacity', 0.5);

    // Tolerance band
    if (n > 10) {
      const bandY = 2 / Math.sqrt(n);
      sel.append('rect')
        .attr('x', MARGIN.left).attr('width', panelW - MARGIN.left - MARGIN.right)
        .attr('y', yScale(bandY)).attr('height', yScale(-bandY) - yScale(bandY))
        .style('fill', PANEL_COLORS[0]).style('opacity', 0.08);
    }

    // Paths (up to currentN)
    const lineGen = d3.line<number>().x((_, i) => xScale(i)).y(d => yScale(d));
    for (let p = 0; p < asData.length; p++) {
      sel.append('path')
        .datum(asData[p].slice(0, n + 1))
        .attr('d', lineGen)
        .style('fill', 'none').style('stroke', PATH_PALETTE[p % PATH_PALETTE.length]).style('stroke-width', 1.2).style('opacity', 0.7);
    }

    // Title
    sel.append('text').attr('x', MARGIN.left + 4).attr('y', MARGIN.top + 10)
      .style('font-size', '10px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', PANEL_COLORS[0])
      .text('A.S. Convergence');
  }, [asData, currentN, panelW, panelH]);

  // Panel 2: Typewriter
  useEffect(() => {
    const svg = svgRefs[1].current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const n = Math.min(currentN, twData.length - 1);
    const xScale = d3.scaleLinear().domain([0, 1]).range([MARGIN.left, panelW - MARGIN.right]);
    const yScale = d3.scaleLinear().domain([0, 1.3]).range([panelH - MARGIN.bottom, MARGIN.top]);

    sel.append('g').attr('transform', `translate(0,${panelH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(5).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').remove(); });

    // Current interval indicator
    if (n >= 0 && n < twData.length) {
      const interval = twData[n];
      sel.append('rect')
        .attr('x', xScale(interval.start)).attr('width', xScale(interval.end) - xScale(interval.start))
        .attr('y', yScale(1)).attr('height', yScale(0) - yScale(1))
        .style('fill', PANEL_COLORS[1]).style('opacity', 0.4);

      // Show a few recent intervals fading
      for (let k = 1; k <= Math.min(5, n); k++) {
        const prev = twData[n - k];
        sel.append('rect')
          .attr('x', xScale(prev.start)).attr('width', xScale(prev.end) - xScale(prev.start))
          .attr('y', yScale(1)).attr('height', yScale(0) - yScale(1))
          .style('fill', PANEL_COLORS[1]).style('opacity', 0.08);
      }

      // Width annotation
      sel.append('text').attr('x', panelW - MARGIN.right - 4).attr('y', panelH - MARGIN.bottom - 6)
        .style('text-anchor', 'end').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
        .text(`width = ${(interval.end - interval.start).toFixed(3)}`);
    }

    sel.append('text').attr('x', MARGIN.left + 4).attr('y', MARGIN.top + 10)
      .style('font-size', '10px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', PANEL_COLORS[1])
      .text('Typewriter (in prob, not a.s.)');
  }, [currentN, twData, panelW, panelH]);

  // Panel 3: SLLN
  useEffect(() => {
    const svg = svgRefs[2].current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const n = currentN;
    const mu = distCfg.mean;
    const sigma = Math.sqrt(distCfg.variance);
    const xScale = d3.scaleLinear().domain([1, MAX_N]).range([MARGIN.left, panelW - MARGIN.right]);
    const yPad = Math.max(sigma * 1.5, 0.5);
    const yScale = d3.scaleLinear().domain([mu - yPad, mu + yPad]).range([panelH - MARGIN.bottom, MARGIN.top]);

    sel.append('g').attr('transform', `translate(0,${panelH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(4).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').remove(); });

    // Mean line
    sel.append('line').attr('x1', MARGIN.left).attr('x2', panelW - MARGIN.right)
      .attr('y1', yScale(mu)).attr('y2', yScale(mu))
      .style('stroke', '#DC2626').style('stroke-width', 1.5).style('stroke-dasharray', '6 3');

    // ±2σ/√n bands
    const bandPath = d3.area<number>()
      .x((_, i) => xScale(i + 1))
      .y0((_, i) => yScale(mu - 2 * sigma / Math.sqrt(i + 1)))
      .y1((_, i) => yScale(mu + 2 * sigma / Math.sqrt(i + 1)));
    const bandIndices = Array.from({ length: Math.min(n, MAX_N) }, (_, i) => i);
    sel.append('path').datum(bandIndices).attr('d', bandPath)
      .style('fill', PANEL_COLORS[2]).style('opacity', 0.1);

    // Paths
    const lineGen = d3.line<number>().x((_, i) => xScale(i)).y(d => yScale(d));
    for (let p = 0; p < sllnData.length; p++) {
      sel.append('path')
        .datum(sllnData[p].slice(1, n + 1))
        .attr('d', lineGen)
        .style('fill', 'none').style('stroke', PATH_PALETTE[p % PATH_PALETTE.length]).style('stroke-width', 1.2).style('opacity', 0.7);
    }

    // Label
    sel.append('text').attr('x', panelW - MARGIN.right - 4).attr('y', yScale(mu) - 6)
      .style('text-anchor', 'end').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', '#DC2626')
      .text(`μ = ${mu}`);

    sel.append('text').attr('x', MARGIN.left + 4).attr('y', MARGIN.top + 10)
      .style('font-size', '10px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', PANEL_COLORS[2])
      .text('SLLN');
  }, [sllnData, currentN, distCfg, panelW, panelH]);

  // Panel 4: CLT histograms
  useEffect(() => {
    const svg = svgRefs[3].current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const hist = cltData.histograms[cltHistIdx];
    const xScale = d3.scaleLinear().domain([-4, 4]).range([MARGIN.left, panelW - MARGIN.right]);
    const maxCount = d3.max(hist.bins, b => b.count) || 0.1;
    const yScale = d3.scaleLinear().domain([0, maxCount * 1.15]).range([panelH - MARGIN.bottom, MARGIN.top]);

    sel.append('g').attr('transform', `translate(0,${panelH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(5).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').remove(); });

    // Histogram bars
    for (const bin of hist.bins) {
      sel.append('rect')
        .attr('x', xScale(bin.x0) + 0.5).attr('width', Math.max(0, xScale(bin.x1) - xScale(bin.x0) - 1))
        .attr('y', yScale(bin.count)).attr('height', Math.max(0, yScale(0) - yScale(bin.count)))
        .style('fill', PANEL_COLORS[3]).style('opacity', 0.6);
    }

    // N(0,1) overlay
    const nPoints = 100;
    const normalLine = d3.line<number>()
      .x(d => xScale(d))
      .y(d => {
        const binWidth = 8 / 30;
        const density = Math.exp(-d * d / 2) / Math.sqrt(2 * Math.PI) * binWidth;
        return yScale(density);
      });
    const zValues = Array.from({ length: nPoints + 1 }, (_, i) => -4 + 8 * i / nPoints);
    sel.append('path').datum(zValues).attr('d', normalLine)
      .style('fill', 'none').style('stroke', '#DC2626').style('stroke-width', 2).style('stroke-dasharray', '5 3');

    // Label
    sel.append('text').attr('x', MARGIN.left + 4).attr('y', MARGIN.top + 10)
      .style('font-size', '10px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', PANEL_COLORS[3])
      .text(`CLT (n=${hist.n})`);
  }, [cltData, cltHistIdx, panelW, panelH]);

  const handleDist = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setDist(e.target.value as DistributionType);
  }, []);

  if (!containerWidth) {
    return <div ref={containerRef} style={{ minHeight: 400 }} />;
  }

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
        Convergence Modes Demo
      </div>

      {/* 2×2 grid */}
      <div style={{ display: 'grid', gridTemplateColumns: isDesktop ? '1fr 1fr' : '1fr', gap: '8px' }}>
        {svgRefs.map((ref, i) => (
          <svg
            key={i}
            ref={ref}
            width={panelW}
            height={panelH}
            onMouseEnter={() => setHoveredPanel(i)}
            onMouseLeave={() => setHoveredPanel(null)}
            style={{
              border: `1px solid ${hoveredPanel === i ? PANEL_COLORS[i] : 'var(--color-border)'}`,
              borderRadius: '6px',
              background: 'var(--color-muted-bg)',
              transition: 'border-color 0.15s',
            }}
          />
        ))}
      </div>

      {/* Hierarchy diagram */}
      <div style={{ marginTop: '12px' }}>
        <svg width={containerWidth} height={60} style={{ display: 'block' }}>
          {/* Layout: Lp -> in prob -> in dist, a.s. -> in prob */}
          {(() => {
            const cx = containerWidth / 2;
            const nodeW = isDesktop ? 80 : 55;
            const gap = isDesktop ? 40 : 20;
            const nodes = [
              { label: 'L^p', x: cx - nodeW - gap, y: 14, color: PANEL_COLORS[3], panelIdx: null },
              { label: 'a.s.', x: cx - nodeW - gap, y: 44, color: PANEL_COLORS[0], panelIdx: 0 },
              { label: 'in prob', x: cx, y: 29, color: PANEL_COLORS[1], panelIdx: 1 },
              { label: 'in dist', x: cx + nodeW + gap, y: 29, color: PANEL_COLORS[3], panelIdx: 3 },
            ];
            const arrows = [
              { from: 0, to: 2 }, // Lp -> in prob
              { from: 1, to: 2 }, // a.s. -> in prob
              { from: 2, to: 3 }, // in prob -> in dist
            ];
            const falseConverses = [
              { from: 2, to: 1, label: '✗' }, // in prob -/-> a.s.
              { from: 3, to: 2, label: '✗' }, // in dist -/-> in prob
            ];
            return (
              <>
                {arrows.map((a, i) => (
                  <line key={`a-${i}`}
                    x1={nodes[a.from].x + nodeW / 2 - 4} y1={nodes[a.from].y}
                    x2={nodes[a.to].x - nodeW / 2 + 4} y2={nodes[a.to].y}
                    stroke="var(--color-text-secondary)" strokeWidth={1.5} markerEnd="url(#arrowhead)" />
                ))}
                {falseConverses.map((fc, i) => (
                  <g key={`fc-${i}`}>
                    <text
                      x={(nodes[fc.from].x + nodes[fc.to].x) / 2}
                      y={(nodes[fc.from].y + nodes[fc.to].y) / 2 + (i === 0 ? 14 : -8)}
                      textAnchor="middle" fontSize={12} fill="#DC2626" fontWeight={700}>
                      {fc.label}
                    </text>
                  </g>
                ))}
                {nodes.map((n, i) => (
                  <g key={`n-${i}`}>
                    <rect x={n.x - nodeW / 2} y={n.y - 10} width={nodeW} height={20} rx={4}
                      fill={hoveredPanel === n.panelIdx ? n.color : 'var(--color-muted-bg)'}
                      stroke={n.color} strokeWidth={1.5} opacity={hoveredPanel === n.panelIdx ? 0.2 : 1} />
                    <text x={n.x} y={n.y + 4} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fontWeight={600}
                      fill={n.color}>
                      {n.label}
                    </text>
                  </g>
                ))}
                <defs>
                  <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
                    <path d="M0,0 L8,3 L0,6" fill="var(--color-text-secondary)" />
                  </marker>
                </defs>
              </>
            );
          })()}
        </svg>
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
            {(Object.keys(DISTRIBUTIONS) as DistributionType[]).map(k => (
              <option key={k} value={k}>{DISTRIBUTIONS[k].label}</option>
            ))}
          </select>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
          Speed:
          <input type="range" min={1} max={5} step={1} value={speed} onChange={e => setSpeed(parseInt(e.target.value))}
            style={{ width: '80px', accentColor: 'var(--color-accent)' }} />
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
