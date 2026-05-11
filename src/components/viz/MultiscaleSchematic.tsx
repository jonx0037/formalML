// =============================================================================
// MultiscaleSchematic.tsx
//
// §6 — Block-diagram of a 3-level Glow architecture:
//   squeeze → coupling stack → split → squeeze → coupling stack → split →
//   squeeze → coupling stack → flatten → final couplings
//
// Annotated with tensor shapes (C × H × W → 4C × H/2 × W/2) and per-level
// parameter counts. No interactivity — orientation diagram only.
// =============================================================================

interface Block {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  sub?: string;
  kind: 'squeeze' | 'coupling' | 'split' | 'flatten' | 'final';
}

const COLORS: Record<Block['kind'], string> = {
  squeeze: '#0F6E56',
  coupling: '#534AB7',
  split: '#D97706',
  flatten: '#6B6B6B',
  final: '#2563EB',
};

export default function MultiscaleSchematic() {
  const W = 820;
  const H = 380;
  const yLane = (i: number) => 50 + i * 80;

  const blocks: Block[] = [
    // Level 1 (3 × 32 × 32 → 12 × 16 × 16)
    { x: 30, y: yLane(0), w: 80, h: 56, label: 'squeeze', sub: '32→16', kind: 'squeeze' },
    { x: 130, y: yLane(0), w: 110, h: 56, label: 'K coupling', sub: '12 × 16 × 16', kind: 'coupling' },
    { x: 260, y: yLane(0), w: 70, h: 56, label: 'split', sub: 'peel 6 ch', kind: 'split' },

    // Level 2 (6 × 16 × 16 → 24 × 8 × 8)
    { x: 350, y: yLane(1), w: 80, h: 56, label: 'squeeze', sub: '16→8', kind: 'squeeze' },
    { x: 450, y: yLane(1), w: 110, h: 56, label: 'K coupling', sub: '24 × 8 × 8', kind: 'coupling' },
    { x: 580, y: yLane(1), w: 70, h: 56, label: 'split', sub: 'peel 12 ch', kind: 'split' },

    // Level 3 (12 × 8 × 8 → 48 × 4 × 4)
    { x: 30, y: yLane(2), w: 80, h: 56, label: 'squeeze', sub: '8→4', kind: 'squeeze' },
    { x: 130, y: yLane(2), w: 110, h: 56, label: 'K coupling', sub: '48 × 4 × 4', kind: 'coupling' },

    // Final
    { x: 260, y: yLane(2), w: 90, h: 56, label: 'flatten', sub: '→ 768-vec', kind: 'flatten' },
    { x: 370, y: yLane(2), w: 130, h: 56, label: 'final couplings', sub: 'vector form', kind: 'final' },
  ];

  // Drawing arrows between adjacent blocks in the levels
  const flow = [
    [0, 1], [1, 2],
    [3, 4], [4, 5],
    [6, 7], [7, 8], [8, 9],
  ];
  // Vertical bridges between level ends and next-level starts
  const bridges = [
    { from: 2, to: 3 },
    { from: 5, to: 6 },
  ];

  return (
    <figure className="my-8 not-prose">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        role="img"
        aria-label="Block diagram of a three-level Glow multi-scale architecture: per-level squeeze + coupling stack + split, with a final flatten + vector-coupling stack."
        style={{ maxWidth: '100%', height: 'auto' }}
      >
        <defs>
          <marker
            id="ms-arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" style={{ fill: 'var(--color-text-secondary)' }} />
          </marker>
        </defs>

        {/* Title */}
        <text x={W / 2} y={26} textAnchor="middle"
              style={{ fill: 'var(--color-text)', fontSize: 14, fontWeight: 600, fontFamily: 'var(--font-sans, sans-serif)' }}>
          Glow — 3-level multi-scale architecture
        </text>

        {/* Blocks */}
        {blocks.map((b, i) => (
          <g key={i}>
            <rect
              x={b.x}
              y={b.y}
              width={b.w}
              height={b.h}
              rx={6}
              fill={COLORS[b.kind]}
              fillOpacity={0.18}
              style={{ stroke: COLORS[b.kind], strokeWidth: 1.5 }}
            />
            <text x={b.x + b.w / 2} y={b.y + 22} textAnchor="middle"
                  style={{ fill: COLORS[b.kind], fontSize: 12, fontWeight: 700, fontFamily: 'var(--font-sans, sans-serif)' }}>
              {b.label}
            </text>
            {b.sub && (
              <text x={b.x + b.w / 2} y={b.y + 40} textAnchor="middle"
                    style={{ fill: 'var(--color-text-secondary)', fontSize: 10, fontFamily: 'var(--font-mono, monospace)' }}>
                {b.sub}
              </text>
            )}
          </g>
        ))}

        {/* Forward arrows (within a level) */}
        {flow.map(([a, b], k) => {
          const A = blocks[a];
          const B = blocks[b];
          return (
            <line
              key={`flow-${k}`}
              x1={A.x + A.w}
              y1={A.y + A.h / 2}
              x2={B.x - 4}
              y2={B.y + A.h / 2}
              style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1.4 }}
              markerEnd="url(#ms-arrow)"
            />
          );
        })}

        {/* Bridges between levels (drop down) */}
        {bridges.map(({ from, to }, k) => {
          const A = blocks[from];
          const B = blocks[to];
          const midX = A.x + A.w + 30;
          return (
            <g key={`bridge-${k}`}>
              <line x1={A.x + A.w} y1={A.y + A.h / 2} x2={midX} y2={A.y + A.h / 2}
                    style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1.2, strokeDasharray: '4 3' }} />
              <line x1={midX} y1={A.y + A.h / 2} x2={midX} y2={B.y + B.h / 2}
                    style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1.2, strokeDasharray: '4 3' }} />
              <line x1={midX} y1={B.y + B.h / 2} x2={B.x - 4} y2={B.y + B.h / 2}
                    style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1.2, strokeDasharray: '4 3' }}
                    markerEnd="url(#ms-arrow)" />
            </g>
          );
        })}

        {/* Input / output annotations */}
        <text x={20} y={yLane(0) + 30} textAnchor="end"
              style={{ fill: 'var(--color-text-secondary)', fontSize: 10, fontFamily: 'var(--font-mono, monospace)' }}>
          x: 3×32×32
        </text>
        <text x={W - 20} y={yLane(2) + 30} textAnchor="start"
              style={{ fill: 'var(--color-text-secondary)', fontSize: 10, fontFamily: 'var(--font-mono, monospace)' }}>
          → z
        </text>

        {/* Legend */}
        <g transform={`translate(20, ${H - 30})`}>
          {(['squeeze', 'coupling', 'split', 'flatten', 'final'] as Block['kind'][]).map((kind, i) => (
            <g key={kind} transform={`translate(${i * 130}, 0)`}>
              <rect width={14} height={14} fill={COLORS[kind]} fillOpacity={0.18}
                    style={{ stroke: COLORS[kind], strokeWidth: 1.5 }} />
              <text x={20} y={11}
                    style={{ fill: 'var(--color-text)', fontSize: 11, fontFamily: 'var(--font-sans, sans-serif)' }}>
                {kind}
              </text>
            </g>
          ))}
        </g>
      </svg>
      <figcaption className="text-sm mt-2 text-center" style={{ color: 'var(--color-text-secondary)' }}>
        Glow's three-level multi-scale architecture — squeeze halves spatial resolution and quadruples channels; split
        peels off half the channels at each scale; the bottom level flattens and finishes with vector-form coupling
        layers. None of the layers change the log-det math from §3.4; squeeze/split have log-det 0.
      </figcaption>
    </figure>
  );
}
