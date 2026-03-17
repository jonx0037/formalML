import { useState, useMemo, useRef, useCallback, useEffect } from 'react';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { consistencyColorScale, dimensionColors } from './shared/colorScales';
import * as d3 from 'd3';

// ─── Graph topology: 5-node "bowtie" graph ───

interface GNode {
  id: number;
  label: string;
  x: number;
  y: number;
}

interface GEdge {
  source: number;
  target: number;
}

const GRAPH_NODES: GNode[] = [
  { id: 0, label: 'v₀', x: 0.15, y: 0.3 },
  { id: 1, label: 'v₁', x: 0.15, y: 0.7 },
  { id: 2, label: 'v₂', x: 0.5, y: 0.5 },
  { id: 3, label: 'v₃', x: 0.85, y: 0.3 },
  { id: 4, label: 'v₄', x: 0.85, y: 0.7 },
];

const GRAPH_EDGES: GEdge[] = [
  { source: 0, target: 1 },
  { source: 0, target: 2 },
  { source: 1, target: 2 },
  { source: 2, target: 3 },
  { source: 2, target: 4 },
  { source: 3, target: 4 },
];

const N_NODES = GRAPH_NODES.length;
const N_EDGES = GRAPH_EDGES.length;
const STALK_DIM = 2;
const STATE_DIM = N_NODES * STALK_DIM; // 10

type SheafPreset = 'constant' | 'rotation';

// ─── Linear algebra helpers ───

function rotMat(theta: number): number[][] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [[c, -s], [s, c]];
}

const IDENTITY: number[][] = [[1, 0], [0, 1]];

/** Build the sheaf Laplacian L_F as a flat STATE_DIM × STATE_DIM array. */
function buildSheafLaplacian(preset: SheafPreset, theta: number): number[] {
  const dim = STATE_DIM;
  const L = new Array(dim * dim).fill(0);

  // For each edge, accumulate L_F = δ₀ᵀ δ₀
  for (const edge of GRAPH_EDGES) {
    const s = edge.source;
    const t = edge.target;

    // Restriction maps: ρ_s and ρ_t map stalks into edge stalk
    let rhoS: number[][];
    let rhoT: number[][];

    if (preset === 'constant') {
      rhoS = IDENTITY;
      rhoT = IDENTITY;
    } else {
      rhoS = IDENTITY;
      rhoT = rotMat(theta);
    }

    // The coboundary contribution for this edge is:
    // δ₀ x |_e = ρ_t x_t − ρ_s x_s
    // L_F += (ρ_t^T ρ_t) at (t,t) + (ρ_s^T ρ_s) at (s,s)
    //       − (ρ_t^T ρ_s) at (t,s) − (ρ_s^T ρ_t) at (s,t)

    for (let i = 0; i < STALK_DIM; i++) {
      for (let j = 0; j < STALK_DIM; j++) {
        // ρ_s^T ρ_s
        let rsTrs = 0;
        for (let k = 0; k < STALK_DIM; k++) rsTrs += rhoS[k][i] * rhoS[k][j];
        // ρ_t^T ρ_t
        let rtTrt = 0;
        for (let k = 0; k < STALK_DIM; k++) rtTrt += rhoT[k][i] * rhoT[k][j];
        // ρ_s^T ρ_t
        let rsTrt = 0;
        for (let k = 0; k < STALK_DIM; k++) rsTrt += rhoS[k][i] * rhoT[k][j];
        // ρ_t^T ρ_s
        let rtTrs = 0;
        for (let k = 0; k < STALK_DIM; k++) rtTrs += rhoT[k][i] * rhoS[k][j];

        L[(s * STALK_DIM + i) * dim + (s * STALK_DIM + j)] += rsTrs;
        L[(t * STALK_DIM + i) * dim + (t * STALK_DIM + j)] += rtTrt;
        L[(s * STALK_DIM + i) * dim + (t * STALK_DIM + j)] -= rsTrt;
        L[(t * STALK_DIM + i) * dim + (s * STALK_DIM + j)] -= rtTrs;
      }
    }
  }

  return L;
}

/** Compute x^T L x (Laplacian energy). */
function laplacianEnergy(L: number[], x: number[]): number {
  const n = x.length;
  let energy = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      energy += x[i] * L[i * n + j] * x[j];
    }
  }
  return energy;
}

/** One step of forward Euler diffusion: x_{t+1} = x_t − α L_F x_t. */
function diffusionStep(L: number[], x: number[], alpha: number): number[] {
  const n = x.length;
  const Lx = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      Lx[i] += L[i * n + j] * x[j];
    }
  }
  return x.map((xi, i) => xi - alpha * Lx[i]);
}

/** Deterministic seeded random. */
function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return ((s >>> 0) / 0xffffffff) * 2 - 1;
  };
}

function generateInitialState(): number[] {
  const rng = seededRandom(17);
  return Array.from({ length: STATE_DIM }, () => rng());
}

const INITIAL_STATE = generateInitialState();

// ─── Component ───

export default function SheafDiffusionSimulator() {
  const [preset, setPreset] = useState<SheafPreset>('constant');
  const [theta, setTheta] = useState(Math.PI / 4);
  const [alpha, setAlpha] = useState(0.08);
  const [isPlaying, setIsPlaying] = useState(false);
  const [state, setState] = useState<number[]>(INITIAL_STATE);
  const [energyHistory, setEnergyHistory] = useState<number[]>([]);
  const animFrameRef = useRef<number>(0);
  const stateRef = useRef(state);
  stateRef.current = state;

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const panelWidth = Math.min(((containerWidth || 700) - 16) / 2, 340);
  const svgHeight = 280;

  const L = useMemo(() => buildSheafLaplacian(preset, theta), [preset, theta]);

  const currentEnergy = useMemo(() => laplacianEnergy(L, state), [L, state]);

  // Reset when sheaf changes
  const handleReset = useCallback(() => {
    setIsPlaying(false);
    setState(INITIAL_STATE);
    setEnergyHistory([]);
  }, []);

  useEffect(() => {
    handleReset();
  }, [preset, theta]);

  // Single step
  const handleStep = useCallback(() => {
    setState((prev) => {
      const next = diffusionStep(L, prev, alpha);
      setEnergyHistory((h) => [...h, laplacianEnergy(L, next)]);
      return next;
    });
  }, [L, alpha]);

  // Animation loop
  useEffect(() => {
    if (!isPlaying) return;

    let running = true;
    const step = () => {
      if (!running) return;
      setState((prev) => {
        const next = diffusionStep(L, prev, alpha);
        setEnergyHistory((h) => {
          const entry = laplacianEnergy(L, next);
          if (h.length > 200) return [...h.slice(-199), entry];
          return [...h, entry];
        });
        return next;
      });
      animFrameRef.current = requestAnimationFrame(step);
    };
    animFrameRef.current = requestAnimationFrame(step);

    return () => {
      running = false;
      cancelAnimationFrame(animFrameRef.current);
    };
  }, [isPlaying, L, alpha]);

  // ─── Graph rendering ───

  const graphRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const pad = 40;
      const w = panelWidth - 2 * pad;
      const h = svgHeight - 2 * pad;
      const xScale = (v: number) => pad + v * w;
      const yScale = (v: number) => pad + v * h;
      const arrowLen = 22;

      // Edges
      for (const edge of GRAPH_EDGES) {
        const s = GRAPH_NODES[edge.source];
        const t = GRAPH_NODES[edge.target];

        // Compute edge inconsistency for coloring
        const sv = state.slice(s.id * STALK_DIM, (s.id + 1) * STALK_DIM);
        const tv = state.slice(t.id * STALK_DIM, (t.id + 1) * STALK_DIM);
        const diff = sv.map((v, i) => v - tv[i]);
        const inc = Math.sqrt(diff.reduce((a, x) => a + x * x, 0));
        const normInc = Math.min(inc / 1.5, 1);

        svg
          .append('line')
          .attr('x1', xScale(s.x))
          .attr('y1', yScale(s.y))
          .attr('x2', xScale(t.x))
          .attr('y2', yScale(t.y))
          .attr('stroke', consistencyColorScale(normInc))
          .attr('stroke-width', 2.5)
          .attr('stroke-opacity', 0.7);
      }

      // Nodes with vector arrows
      for (const node of GRAPH_NODES) {
        const cx = xScale(node.x);
        const cy = yScale(node.y);
        const val = state.slice(node.id * STALK_DIM, (node.id + 1) * STALK_DIM);
        const vNorm = Math.sqrt(val[0] * val[0] + val[1] * val[1]);

        // Node circle
        svg
          .append('circle')
          .attr('cx', cx)
          .attr('cy', cy)
          .attr('r', 26)
          .attr('fill', 'var(--color-surface)')
          .attr('stroke', 'var(--color-border)')
          .attr('stroke-width', 1.5);

        // Vector arrow
        if (vNorm > 0.01) {
          const scale = Math.min(vNorm, 1);
          svg
            .append('line')
            .attr('x1', cx)
            .attr('y1', cy)
            .attr('x2', cx + (val[0] / vNorm) * scale * arrowLen)
            .attr('y2', cy - (val[1] / vNorm) * scale * arrowLen)
            .attr('stroke', '#6366f1')
            .attr('stroke-width', 2)
            .attr('stroke-linecap', 'round');

          svg
            .append('circle')
            .attr('cx', cx + (val[0] / vNorm) * scale * arrowLen)
            .attr('cy', cy - (val[1] / vNorm) * scale * arrowLen)
            .attr('r', 2.5)
            .attr('fill', '#6366f1');
        }

        // Label
        svg
          .append('text')
          .attr('x', cx)
          .attr('y', cy + 38)
          .attr('text-anchor', 'middle')
          .style('font-size', '11px')
          .style('font-family', 'var(--font-sans)')
          .style('font-weight', '600')
          .style('fill', 'var(--color-text)')
          .text(node.label);
      }
    },
    [panelWidth, svgHeight, state],
  );

  // ─── Energy chart rendering ───

  const chartRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 24, right: 16, bottom: 30, left: 46 };
      const w = panelWidth - margin.left - margin.right;
      const h = svgHeight - margin.top - margin.bottom;

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const data = energyHistory.length > 0 ? energyHistory : [currentEnergy];
      const maxE = Math.max(d3.max(data) ?? 1, 0.1);

      const xScale = d3.scaleLinear().domain([0, Math.max(data.length - 1, 1)]).range([0, w]);
      const yScale = d3.scaleLinear().domain([0, maxE * 1.1]).range([h, 0]);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format('d')))
        .selectAll('text')
        .style('font-size', '10px')
        .style('font-family', 'var(--font-mono, monospace)');

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format('.2f')))
        .selectAll('text')
        .style('font-size', '10px')
        .style('font-family', 'var(--font-mono, monospace)');

      // Axis labels
      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 26)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .style('font-family', 'var(--font-sans)')
        .style('fill', 'var(--color-text-secondary)')
        .text('Step');

      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -h / 2)
        .attr('y', -36)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .style('font-family', 'var(--font-sans)')
        .style('fill', 'var(--color-text-secondary)')
        .text('Energy x\u1D40L_Fx');

      // Title
      svg
        .append('text')
        .attr('x', margin.left + w / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-family', 'var(--font-sans)')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('Laplacian Energy');

      // Line
      if (data.length > 1) {
        const line = d3
          .line<number>()
          .x((_, i) => xScale(i))
          .y((d) => yScale(d));

        g.append('path')
          .datum(data)
          .attr('d', line)
          .attr('fill', 'none')
          .attr('stroke', dimensionColors[1])
          .attr('stroke-width', 2);
      }

      // Current energy dot
      if (data.length > 0) {
        g.append('circle')
          .attr('cx', xScale(data.length - 1))
          .attr('cy', yScale(data[data.length - 1]))
          .attr('r', 4)
          .attr('fill', dimensionColors[1]);
      }
    },
    [panelWidth, svgHeight, energyHistory, currentEnergy],
  );

  return (
    <div ref={containerRef} className="w-full">
      {/* Controls */}
      <div className="mb-3 flex flex-wrap items-center gap-3" style={{ fontFamily: 'var(--font-sans)' }}>
        <select
          value={preset}
          onChange={(e) => setPreset(e.target.value as SheafPreset)}
          className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-2 py-1 text-sm"
        >
          <option value="constant">Constant sheaf</option>
          <option value="rotation">Rotation sheaf</option>
        </select>

        {preset === 'rotation' && (
          <>
            <label className="text-xs font-medium">θ</label>
            <input
              type="range"
              min={0.1}
              max={Math.PI}
              step={0.05}
              value={theta}
              onChange={(e) => setTheta(parseFloat(e.target.value))}
              className="w-20"
            />
            <span className="w-10 font-mono text-xs">{((theta * 180) / Math.PI).toFixed(0)}°</span>
          </>
        )}

        <div className="flex gap-1">
          <button
            onClick={() => setIsPlaying((p) => !p)}
            className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-1 text-sm font-medium hover:bg-[var(--color-muted-bg)] transition-colors"
          >
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            onClick={handleStep}
            disabled={isPlaying}
            className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-1 text-sm font-medium hover:bg-[var(--color-muted-bg)] transition-colors disabled:opacity-40"
          >
            Step
          </button>
          <button
            onClick={handleReset}
            className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-1 text-sm font-medium hover:bg-[var(--color-muted-bg)] transition-colors"
          >
            Reset
          </button>
        </div>

        <label className="text-xs font-medium">α</label>
        <input
          type="range"
          min={0.01}
          max={0.2}
          step={0.005}
          value={alpha}
          onChange={(e) => setAlpha(parseFloat(e.target.value))}
          className="w-20"
        />
        <span className="w-10 font-mono text-xs">{alpha.toFixed(3)}</span>
      </div>

      {/* Dual-panel display */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        <div>
          <svg
            ref={graphRef}
            width={panelWidth}
            height={svgHeight}
            className="mx-auto rounded-lg border border-[var(--color-border)]"
          />
        </div>
        <div>
          <svg
            ref={chartRef}
            width={panelWidth}
            height={svgHeight}
            className="mx-auto rounded-lg border border-[var(--color-border)]"
          />
        </div>
      </div>

      {/* Status */}
      <div
        className="mt-3 rounded-md bg-[var(--color-muted-bg)] px-4 py-3"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm">
          <span>
            <span className="font-medium">Energy</span>{' '}
            <span className="font-mono">{currentEnergy.toFixed(4)}</span>
          </span>
          <span>
            <span className="font-medium">Steps</span>{' '}
            <span className="font-mono">{energyHistory.length}</span>
          </span>
          <span className={currentEnergy < 0.01 ? 'text-green-600' : 'text-[var(--color-text-secondary)]'}>
            {currentEnergy < 0.01 ? 'Converged to H⁰' : 'Diffusing...'}
          </span>
        </div>
        <p className="mt-1 text-xs text-[var(--color-text-secondary)]">
          Sheaf diffusion drives each node's vector toward consistency with its neighbors via the restriction maps.
          The energy x<sup>T</sup>L<sub>F</sub>x measures total inconsistency and decays monotonically to zero.
        </p>
      </div>
    </div>
  );
}
