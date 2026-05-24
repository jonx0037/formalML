import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

type Clustering = {
  id: string;
  label: string;
  shortLabel: string;
  assignments: number[];
  description: string;
};

const N_DOCS = 8;

const TOPICS: number[] = [0, 0, 0, 0, 1, 1, 1, 1];

const CLUSTERINGS: Clustering[] = [
  {
    id: 'const',
    label: 'T = const  (full compression)',
    shortLabel: 'T = const',
    assignments: [0, 0, 0, 0, 0, 0, 0, 0],
    description: 'All eight documents collapsed into a single cluster. Compression is total; predictiveness is zero.',
  },
  {
    id: 'topic',
    label: 'T = topic  (T = Y)',
    shortLabel: 'T = Y',
    assignments: [0, 0, 0, 0, 1, 1, 1, 1],
    description: 'Two clusters aligned with the two topics. Captures all of Y at the smallest possible I(X;T) — sits on the optimal corner of the IB plane.',
  },
  {
    id: 'pairs',
    label: '4 clusters, topic-pure pairs',
    shortLabel: '4-cluster pairs',
    assignments: [0, 0, 1, 1, 2, 2, 3, 3],
    description: 'Four clusters of two docs each, with each cluster entirely inside one topic. Doubles I(X;T) over T = Y but adds nothing to I(T;Y) — wasted compression budget.',
  },
  {
    id: 'identity',
    label: 'T = X  (identity)',
    shortLabel: 'T = X',
    assignments: [0, 1, 2, 3, 4, 5, 6, 7],
    description: 'Each document is its own cluster. Maximum I(X;T) = 3 bits; predictiveness is still capped at I(X;Y) = 1 bit. The extra two bits of X carry no Y-information.',
  },
  {
    id: 'misaligned',
    label: 'Mis-aligned 2-cluster split',
    shortLabel: 'Mis-aligned',
    assignments: [0, 0, 1, 1, 0, 0, 1, 1],
    description: 'Two clusters that mix topics. Compression budget I(X;T) = 1 bit, but the encoder threw away the bit that mattered: I(T;Y) = 0.',
  },
];

// All five colors are CSS-variable accents in the dark theme.
const CLUSTER_COLORS = ['#0F6E56', '#534AB7', '#D97706', '#0EA5E9', '#DC2626'];

// Compute (I(X;T), I(T;Y)) in bits from an assignment array.
function computeIBCoords(assignments: number[]): { IXT: number; ITY: number } {
  const k = Math.max(...assignments) + 1;
  const pxt = new Array(k * 2).fill(0); // p(t, y) folded into k*2
  const pt = new Array(k).fill(0);
  const py = [0.5, 0.5];
  for (let x = 0; x < N_DOCS; x++) {
    const t = assignments[x];
    const y = TOPICS[x];
    const w = 1 / N_DOCS;
    pt[t] += w;
    pxt[t * 2 + y] += w;
  }
  // I(X;T) = H(T) since p(t|x) is deterministic for these reference clusterings:
  //   I(X;T) = H(X) - H(X|T) = H(T) - H(T|X) = H(T) - 0 = H(T).
  let IXT = 0;
  for (let t = 0; t < k; t++) {
    if (pt[t] > 0) IXT -= pt[t] * Math.log2(pt[t]);
  }
  // I(T;Y) = sum_{t,y} p(t,y) log [p(t,y) / (p(t) p(y))]
  let ITY = 0;
  for (let t = 0; t < k; t++) {
    for (let y = 0; y < 2; y++) {
      const ptu = pxt[t * 2 + y];
      if (ptu > 0) {
        ITY += ptu * Math.log2(ptu / (pt[t] * py[y]));
      }
    }
  }
  return { IXT, ITY };
}

export default function IBPlaneExplorer() {
  const [activeId, setActiveId] = useState<string>('topic');
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const points = useMemo(
    () => CLUSTERINGS.map((c) => ({ ...c, ...computeIBCoords(c.assignments) })),
    [],
  );
  const active = points.find((p) => p.id === activeId) ?? points[0];

  const W = Math.max(containerWidth || 720, 300);
  const isMobile = W < 640;
  const planeW = isMobile ? W : Math.min(W * 0.65, 560);
  const planeH = Math.round(planeW * 0.75);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const margin = { top: 28, right: 24, bottom: 48, left: 56 };
      const innerW = planeW - margin.left - margin.right;
      const innerH = planeH - margin.top - margin.bottom;
      if (innerW <= 0 || innerH <= 0) return;

      svg.attr('viewBox', `0 0 ${planeW} ${planeH}`).attr('width', '100%').attr('height', planeH);

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([0, 3.2]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0, 1.2]).range([innerH, 0]);

      // Achievable region (below diagonal, below ceiling) — light shading
      const regionPath = d3.path();
      regionPath.moveTo(xScale(0), yScale(0));
      regionPath.lineTo(xScale(1), yScale(1));
      regionPath.lineTo(xScale(3.2), yScale(1));
      regionPath.lineTo(xScale(3.2), yScale(0));
      regionPath.closePath();
      g.append('path')
        .attr('d', regionPath.toString())
        .style('fill', 'var(--color-accent)')
        .attr('opacity', 0.04);

      // DPI ceiling at I(X;Y) = 1 bit
      g.append('line')
        .attr('x1', xScale(0))
        .attr('x2', xScale(3.2))
        .attr('y1', yScale(1))
        .attr('y2', yScale(1))
        .style('stroke', 'var(--color-text-secondary)')
        .attr('stroke-dasharray', '4 3')
        .attr('stroke-width', 1.2);
      g.append('text')
        .attr('x', xScale(3.0))
        .attr('y', yScale(1) - 6)
        .attr('text-anchor', 'end')
        .style('fill', 'var(--color-text-secondary)')
        .attr('font-size', 11)
        .text('I(X;Y) = 1 bit  (DPI ceiling)');

      // Diagonal I(T;Y) ≤ I(X;T)
      g.append('line')
        .attr('x1', xScale(0))
        .attr('x2', xScale(1.2))
        .attr('y1', yScale(0))
        .attr('y2', yScale(1.2))
        .style('stroke', 'var(--color-border)')
        .attr('stroke-dasharray', '3 3')
        .attr('stroke-width', 1.0);
      g.append('text')
        .attr('x', xScale(0.85))
        .attr('y', yScale(0.95))
        .attr('text-anchor', 'start')
        .style('fill', 'var(--color-text-secondary)')
        .attr('font-size', 11)
        .text('I(T;Y) ≤ I(X;T)');

      // Axes
      const xAxis = d3.axisBottom(xScale).ticks(7);
      const yAxis = d3.axisLeft(yScale).ticks(6);
      const xAxisG = g.append('g').attr('transform', `translate(0,${innerH})`).call(xAxis);
      const yAxisG = g.append('g').call(yAxis);
      xAxisG.selectAll('text').style('fill', 'var(--color-text)').attr('font-size', 11);
      yAxisG.selectAll('text').style('fill', 'var(--color-text)').attr('font-size', 11);
      xAxisG.selectAll('line, path').style('stroke', 'var(--color-border)');
      yAxisG.selectAll('line, path').style('stroke', 'var(--color-border)');

      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 36)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .attr('font-size', 12)
        .text('I(X;T) — compression cost (bits)');
      g.append('text')
        .attr('transform', `translate(-40,${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .attr('font-size', 12)
        .text('I(T;Y) — predictiveness (bits)');

      // Plot the 5 reference clusterings
      points.forEach((p, i) => {
        const isActive = p.id === activeId;
        const cx = xScale(p.IXT);
        const cy = yScale(p.ITY);
        g.append('circle')
          .attr('cx', cx)
          .attr('cy', cy)
          .attr('r', isActive ? 9 : 6)
          .style('fill', CLUSTER_COLORS[i])
          .attr('opacity', isActive ? 1.0 : 0.55)
          .attr('stroke', isActive ? 'var(--color-text)' : 'none')
          .attr('stroke-width', isActive ? 1.5 : 0);
        // Label offsets, hand-tuned to avoid overlap
        const labelOffsets: Record<string, [number, number, string]> = {
          const: [12, -8, 'start'],
          topic: [-12, -10, 'end'],
          pairs: [12, -6, 'start'],
          identity: [-12, -10, 'end'],
          misaligned: [12, 14, 'start'],
        };
        const [dx, dy, anchor] = labelOffsets[p.id];
        g.append('text')
          .attr('x', cx + dx)
          .attr('y', cy + dy)
          .attr('text-anchor', anchor)
          .style('fill', isActive ? CLUSTER_COLORS[i] : 'var(--color-text-secondary)')
          .attr('font-size', isActive ? 12 : 10)
          .attr('font-weight', isActive ? 600 : 400)
          .text(`${p.shortLabel}  (${p.IXT.toFixed(2)}, ${p.ITY.toFixed(2)})`);
      });
    },
    [activeId, planeW, planeH, points],
  );

  // Assignment heatmap: 8 docs × max(k) clusters
  const k = active.assignments.reduce((m, v) => Math.max(m, v), 0) + 1;
  const cellSize = isMobile ? 26 : 30;
  const heatW = N_DOCS * cellSize + 80;
  const heatH = k * cellSize + 80;

  return (
    <div
      ref={containerRef}
      style={{
        display: 'flex',
        flexDirection: isMobile ? 'column' : 'row',
        gap: '1.5rem',
        alignItems: 'flex-start',
        padding: '1rem 0',
      }}
    >
      <div style={{ flex: isMobile ? '1' : '0 0 65%', minWidth: 0, width: '100%' }}>
        <div style={{ marginBottom: '0.75rem' }}>
          <label
            htmlFor="ib-plane-clustering"
            style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)', display: 'block', marginBottom: '0.25rem' }}
          >
            Reference clustering
          </label>
          <select
            id="ib-plane-clustering"
            value={activeId}
            onChange={(e) => setActiveId(e.target.value)}
            style={{
              padding: '0.4rem 0.6rem',
              fontSize: '0.95rem',
              fontFamily: 'var(--font-sans)',
              background: 'var(--color-surface)',
              color: 'var(--color-text)',
              border: '1px solid var(--color-border)',
              borderRadius: '4px',
              width: '100%',
              maxWidth: 420,
            }}
          >
            {CLUSTERINGS.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>
        </div>
        <svg ref={svgRef} role="img" aria-label="IB plane with five reference clusterings" />
        <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', maxWidth: 520 }}>
          {active.description}
        </p>
      </div>

      <div style={{ flex: isMobile ? '1' : '0 0 30%', minWidth: 0 }}>
        <div style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)', marginBottom: '0.5rem' }}>
          Assignment p(t&nbsp;|&nbsp;x)
        </div>
        <svg
          viewBox={`0 0 ${heatW} ${heatH}`}
          style={{ width: '100%', maxWidth: heatW, height: 'auto' }}
          role="img"
          aria-label="Cluster assignment heatmap for the active clustering"
        >
          {/* Cluster row labels */}
          {Array.from({ length: k }).map((_, t) => (
            <text
              key={`row-${t}`}
              x={36}
              y={50 + t * cellSize + cellSize / 2}
              fontSize={11}
              dominantBaseline="middle"
              textAnchor="end"
              style={{ fill: 'var(--color-text)' }}
            >
              t = {t}
            </text>
          ))}
          {/* Document column labels */}
          {Array.from({ length: N_DOCS }).map((_, x) => (
            <text
              key={`col-${x}`}
              x={48 + x * cellSize + cellSize / 2}
              y={38}
              fontSize={11}
              textAnchor="middle"
              style={{ fill: 'var(--color-text)' }}
            >
              x{x}
            </text>
          ))}
          {/* Cells */}
          {active.assignments.map((tAssigned, x) => (
            <g key={`cell-${x}`}>
              {Array.from({ length: k }).map((_, t) => {
                const filled = t === tAssigned;
                return (
                  <rect
                    key={`cell-${x}-${t}`}
                    x={48 + x * cellSize}
                    y={50 + t * cellSize}
                    width={cellSize - 2}
                    height={cellSize - 2}
                    style={{ fill: filled ? CLUSTER_COLORS[points.findIndex((p) => p.id === active.id)] : 'transparent' }}
                    opacity={filled ? 0.85 : 1}
                    stroke="var(--color-border)"
                    strokeWidth={1}
                  />
                );
              })}
            </g>
          ))}
          {/* Topic row indicator below heatmap */}
          {Array.from({ length: N_DOCS }).map((_, x) => (
            <text
              key={`topic-${x}`}
              x={48 + x * cellSize + cellSize / 2}
              y={50 + k * cellSize + 18}
              fontSize={10}
              textAnchor="middle"
              style={{ fill: 'var(--color-text-secondary)' }}
            >
              y={TOPICS[x]}
            </text>
          ))}
        </svg>
        <div style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', marginTop: '0.6rem' }}>
          <div>I(X;T) = <strong style={{ color: 'var(--color-text)' }}>{active.IXT.toFixed(3)}</strong> bits</div>
          <div>I(T;Y) = <strong style={{ color: 'var(--color-text)' }}>{active.ITY.toFixed(3)}</strong> bits</div>
        </div>
      </div>
    </div>
  );
}
