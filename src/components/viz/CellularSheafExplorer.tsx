import { useState, useMemo, useCallback } from 'react';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { consistencyColorScale } from './shared/colorScales';

// ─── Graph layout: equilateral triangle (K₃) ───

interface GraphNode {
  id: string;
  x: number;
  y: number;
}

interface GraphEdge {
  source: string;
  target: string;
  id: string;
}

const NODES: GraphNode[] = [
  { id: 'A', x: 0.5, y: 0.12 },
  { id: 'B', x: 0.15, y: 0.82 },
  { id: 'C', x: 0.85, y: 0.82 },
];

const EDGES: GraphEdge[] = [
  { source: 'A', target: 'B', id: 'AB' },
  { source: 'B', target: 'C', id: 'BC' },
  { source: 'A', target: 'C', id: 'AC' },
];

type SheafType = 'constant' | 'rotation';

// ─── Linear algebra helpers (2×2) ───

function rotationMatrix(theta: number): number[][] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    [c, -s],
    [s, c],
  ];
}

function matVec(M: number[][], v: number[]): number[] {
  return M.map((row) => row.reduce((sum, val, j) => sum + val * v[j], 0));
}

function vecSub(a: number[], b: number[]): number[] {
  return a.map((v, i) => v - b[i]);
}

function vecNorm(v: number[]): number {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

/** Build restriction maps for the selected sheaf type. */
function buildRestrictionMaps(
  type: SheafType,
  theta: number,
): Record<string, { source: number[][]; target: number[][] }> {
  const I = [
    [1, 0],
    [0, 1],
  ];

  if (type === 'constant') {
    return {
      AB: { source: I, target: I },
      BC: { source: I, target: I },
      AC: { source: I, target: I },
    };
  }

  // Rotation sheaf: restriction maps are rotations around the triangle
  const R = rotationMatrix(theta);
  return {
    AB: { source: I, target: R },
    BC: { source: I, target: R },
    AC: { source: I, target: R },
  };
}

/** Compute coboundary on each edge: ρ_target(x_target) − ρ_source(x_source). */
function computeEdgeInconsistency(
  nodeValues: Record<string, number[]>,
  maps: Record<string, { source: number[][]; target: number[][] }>,
): Record<string, { diff: number[]; norm: number }> {
  const result: Record<string, { diff: number[]; norm: number }> = {};
  for (const edge of EDGES) {
    const m = maps[edge.id];
    const mappedSource = matVec(m.source, nodeValues[edge.source]);
    const mappedTarget = matVec(m.target, nodeValues[edge.target]);
    const diff = vecSub(mappedTarget, mappedSource);
    result[edge.id] = { diff, norm: vecNorm(diff) };
  }
  return result;
}

/** Format a 2×2 matrix as a compact string. */
function matrixLabel(M: number[][]): string {
  const fmt = (x: number) => {
    if (Math.abs(x) < 0.005) return '0';
    if (Math.abs(x - 1) < 0.005) return '1';
    if (Math.abs(x + 1) < 0.005) return '−1';
    return x.toFixed(2);
  };
  return `[${fmt(M[0][0])}, ${fmt(M[0][1])}; ${fmt(M[1][0])}, ${fmt(M[1][1])}]`;
}

// ─── Component ───

export default function CellularSheafExplorer() {
  const [sheafType, setSheafType] = useState<SheafType>('constant');
  const [theta, setTheta] = useState(Math.PI / 3);
  const [nodeValues, setNodeValues] = useState<Record<string, number[]>>({
    A: [1, 0],
    B: [1, 0],
    C: [1, 0],
  });

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgWidth = Math.min(containerWidth || 480, 480);
  const svgHeight = 380;
  const pad = 60;

  const nodeMap = useMemo(() => new Map(NODES.map((n) => [n.id, n])), []);
  const maps = useMemo(() => buildRestrictionMaps(sheafType, theta), [sheafType, theta]);
  const inconsistency = useMemo(
    () => computeEdgeInconsistency(nodeValues, maps),
    [nodeValues, maps],
  );
  const totalInconsistency = useMemo(
    () => Object.values(inconsistency).reduce((s, e) => s + e.norm * e.norm, 0),
    [inconsistency],
  );
  const isGlobalSection = totalInconsistency < 1e-6;

  const handleNodeAngle = useCallback(
    (nodeId: string, angle: number) => {
      setNodeValues((prev) => ({
        ...prev,
        [nodeId]: [Math.cos(angle), Math.sin(angle)],
      }));
    },
    [],
  );

  // ─── D3 rendering ───

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const w = svgWidth - 2 * pad;
      const h = svgHeight - 2 * pad;
      const xScale = (v: number) => pad + v * w;
      const yScale = (v: number) => pad + v * h;

      // Draw edges
      for (const edge of EDGES) {
        const s = nodeMap.get(edge.source);
        const t = nodeMap.get(edge.target);
        if (!s || !t) continue;
        const inc = inconsistency[edge.id];
        const normClamped = Math.min(inc.norm / 1.5, 1);

        svg
          .append('line')
          .attr('x1', xScale(s.x))
          .attr('y1', yScale(s.y))
          .attr('x2', xScale(t.x))
          .attr('y2', yScale(t.y))
          .attr('stroke', consistencyColorScale(normClamped))
          .attr('stroke-width', 3)
          .attr('stroke-opacity', 0.7);

        // Edge label (restriction map info)
        const mx = (xScale(s.x) + xScale(t.x)) / 2;
        const my = (yScale(s.y) + yScale(t.y)) / 2;
        const dx = xScale(t.x) - xScale(s.x);
        const dy = yScale(t.y) - yScale(s.y);
        const len = Math.sqrt(dx * dx + dy * dy);
        const nx = -dy / len;
        const ny = dx / len;
        const labelOffset = 16;

        svg
          .append('text')
          .attr('x', mx + nx * labelOffset)
          .attr('y', my + ny * labelOffset)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .style('font-size', '10px')
          .style('font-family', 'var(--font-mono, monospace)')
          .style('fill', 'var(--color-text-secondary)')
          .text(`δ=${inc.norm.toFixed(2)}`);
      }

      // Draw nodes with vector arrows
      const arrowLen = 28;

      for (const node of NODES) {
        const cx = xScale(node.x);
        const cy = yScale(node.y);
        const val = nodeValues[node.id];

        // Node circle (stalk indicator)
        svg
          .append('circle')
          .attr('cx', cx)
          .attr('cy', cy)
          .attr('r', 32)
          .style('fill', 'var(--color-surface)')
          .style('stroke', 'var(--color-border)')
          .attr('stroke-width', 2);

        // Vector arrow inside node
        svg
          .append('line')
          .attr('x1', cx)
          .attr('y1', cy)
          .attr('x2', cx + val[0] * arrowLen)
          .attr('y2', cy - val[1] * arrowLen)
          .attr('stroke', '#6366f1')
          .attr('stroke-width', 2.5)
          .attr('stroke-linecap', 'round');

        // Arrowhead
        const ax = cx + val[0] * arrowLen;
        const ay = cy - val[1] * arrowLen;
        svg
          .append('circle')
          .attr('cx', ax)
          .attr('cy', ay)
          .attr('r', 3)
          .attr('fill', '#6366f1');

        // Node label
        svg
          .append('text')
          .attr('x', cx)
          .attr('y', cy + 46)
          .attr('text-anchor', 'middle')
          .style('font-size', '13px')
          .style('font-family', 'var(--font-sans)')
          .style('font-weight', '600')
          .style('fill', 'var(--color-text)')
          .text(node.id);
      }
    },
    [svgWidth, svgHeight, nodeValues, inconsistency],
  );

  return (
    <div ref={containerRef} className="w-full">
      {/* Controls */}
      <div className="mb-3 flex flex-wrap items-center gap-4" style={{ fontFamily: 'var(--font-sans)' }}>
        <label className="text-sm font-medium">Sheaf type</label>
        <select
          value={sheafType}
          onChange={(e) => setSheafType(e.target.value as SheafType)}
          className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-2 py-1 text-sm"
        >
          <option value="constant">Constant (identity maps)</option>
          <option value="rotation">Rotation (angle θ)</option>
        </select>

        {sheafType === 'rotation' && (
          <>
            <label className="text-sm font-medium">θ</label>
            <input
              type="range"
              min={0}
              max={Math.PI}
              step={0.01}
              value={theta}
              onChange={(e) => setTheta(parseFloat(e.target.value))}
              className="w-28"
            />
            <span className="w-14 font-mono text-sm">{((theta * 180) / Math.PI).toFixed(0)}°</span>
          </>
        )}
      </div>

      {/* Graph visualization */}
      <svg role="img" aria-label="Cellular sheaf explorer visualization"
        ref={svgRef}
        width={svgWidth}
        height={svgHeight}
        className="mx-auto rounded-lg border border-[var(--color-border)]"
      />

      {/* Node vector angle controls */}
      <div className="mt-3 grid grid-cols-3 gap-3" style={{ fontFamily: 'var(--font-sans)' }}>
        {NODES.map((node) => {
          const val = nodeValues[node.id];
          const angle = Math.atan2(val[1], val[0]);
          return (
            <div key={node.id} className="text-center">
              <label className="text-xs font-medium">
                Vector at {node.id}
              </label>
              <input
                type="range"
                min={-Math.PI}
                max={Math.PI}
                step={0.05}
                value={angle}
                onChange={(e) => handleNodeAngle(node.id, parseFloat(e.target.value))}
                className="w-full"
              />
              <span className="text-xs font-mono text-[var(--color-text-secondary)]">
                ({val[0].toFixed(2)}, {val[1].toFixed(2)})
              </span>
            </div>
          );
        })}
      </div>

      {/* Status display */}
      <div
        className="mt-3 rounded-md bg-[var(--color-muted-bg)] px-4 py-3"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm">
          <span>
            <span className="font-medium">Total inconsistency</span>{' '}
            <span className="font-mono">x<sup>T</sup>L<sub>F</sub>x = {totalInconsistency.toFixed(4)}</span>
          </span>
          <span className={isGlobalSection ? 'text-green-600' : 'text-red-500'}>
            {isGlobalSection ? 'Global section (in ker δ₀)' : 'Not a global section'}
          </span>
        </div>
        {sheafType === 'rotation' && (
          <p className="mt-1 text-xs text-[var(--color-text-secondary)]">
            Restriction maps: ρ<sub>source</sub> = I, ρ<sub>target</sub> = R(θ).{' '}
            {matrixLabel(rotationMatrix(theta))}
          </p>
        )}
        {sheafType === 'constant' && (
          <p className="mt-1 text-xs text-[var(--color-text-secondary)]">
            All restriction maps are the identity. A global section exists when all node vectors agree.
          </p>
        )}
      </div>
    </div>
  );
}
