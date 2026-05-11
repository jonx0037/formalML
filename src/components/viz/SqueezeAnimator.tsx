// =============================================================================
// SqueezeAnimator.tsx
//
// §6.1 — Animation of the squeeze operation on a small C × H × W RGB image.
// The reader sees 2×2 blocks of pixels getting "lifted" into the channel
// axis as the spatial resolution halves and channel count quadruples. A
// slider drags the interpolation parameter α from 0 (input shape) to 1
// (output shape).
//
// No real math module needed — this is a pure permutation visualization.
// Demonstrates: squeeze is bijective, log-det = 0, no information lost.
// =============================================================================

import { useState, useMemo } from 'react';

const C = 3; // input channels
const H = 4; // input spatial
const Wd = 4;

// Color palette for visualizing channel identity (12 output channels = 4 × 3).
const PALETTE = [
  '#0F6E56',
  '#534AB7',
  '#D97706',
  '#2563EB',
  '#DC2626',
  '#16A34A',
  '#9333EA',
  '#EA580C',
  '#0891B2',
  '#BE185D',
  '#65A30D',
  '#A16207',
];

interface CellMeta {
  channel: number; // 0..C-1
  i: number;
  j: number;
  outputChannel: number; // 0..4C-1
  outputI: number;
  outputJ: number;
}

function buildLayout(): CellMeta[] {
  const cells: CellMeta[] = [];
  for (let c = 0; c < C; c++) {
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < Wd; j++) {
        const alpha = i % 2;
        const beta = j % 2;
        const outI = Math.floor(i / 2);
        const outJ = Math.floor(j / 2);
        const outC = 4 * c + 2 * alpha + beta;
        cells.push({
          channel: c,
          i,
          j,
          outputChannel: outC,
          outputI: outI,
          outputJ: outJ,
        });
      }
    }
  }
  return cells;
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

export default function SqueezeAnimator() {
  const [t, setT] = useState(0); // 0 = input, 1 = output
  const cells = useMemo(() => buildLayout(), []);

  const W = 760;
  const Hsvg = 320;
  const cellSize = 24;

  // Output channel layout: 12 channels arranged in a 4 × 3 grid (4 channels per
  // input channel, 3 input channels). Each output channel is 2×2 spatial.
  const outputColX = (outC: number) => 40 + outC * 56;
  const outputColY = 200;

  return (
    <div className="my-8 not-prose" style={{ border: '1px solid var(--color-border)', borderRadius: 8, padding: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
        <span style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>squeeze:</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={t}
          onChange={(e) => setT(Number(e.target.value))}
          aria-label="Squeeze interpolation parameter"
          style={{ flex: 1, maxWidth: 360 }}
        />
        <span style={{ fontFamily: 'var(--font-mono, monospace)', fontSize: 12, color: 'var(--color-text-secondary)' }}>
          α = {t.toFixed(2)}
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${Hsvg}`} width="100%" style={{ maxWidth: '100%', height: 'auto' }}>
        {/* Input shape label */}
        <text x={195} y={18} textAnchor="middle"
              style={{ fill: t < 0.5 ? 'var(--color-text)' : 'var(--color-text-secondary)', fontSize: 12, fontFamily: 'var(--font-mono, monospace)' }}>
          input: 3 × 4 × 4
        </text>
        <text x={W - 200} y={18} textAnchor="middle"
              style={{ fill: t > 0.5 ? 'var(--color-text)' : 'var(--color-text-secondary)', fontSize: 12, fontFamily: 'var(--font-mono, monospace)' }}>
          output: 12 × 2 × 2
        </text>

        {/* Render the 48 cells, interpolating from input pose to output pose */}
        {cells.map((cell, k) => {
          const fromX = 60 + cell.channel * 130 + cell.j * cellSize;
          const fromY = 40 + cell.i * cellSize;
          const toX = outputColX(cell.outputChannel) + cell.outputJ * cellSize;
          const toY = outputColY + cell.outputI * cellSize;
          const x = lerp(fromX, toX, t);
          const y = lerp(fromY, toY, t);
          return (
            <rect
              key={k}
              x={x}
              y={y}
              width={cellSize - 1}
              height={cellSize - 1}
              fill={PALETTE[cell.outputChannel]}
              fillOpacity={0.65}
              style={{ stroke: 'var(--color-bg)', strokeWidth: 0.5, transition: 'fill 0.2s' }}
            />
          );
        })}

        {/* Input channel labels */}
        {t < 0.6 && Array.from({ length: C }).map((_, c) => (
          <text
            key={`in-${c}`}
            x={60 + c * 130 + (Wd * cellSize) / 2}
            y={40 + H * cellSize + 18}
            textAnchor="middle"
            style={{ fill: 'var(--color-text-secondary)', fontSize: 11, fontFamily: 'var(--font-mono, monospace)', opacity: 1 - t }}
          >
            c = {c}
          </text>
        ))}

        {/* Output channel labels */}
        {t > 0.4 && Array.from({ length: 4 * C }).map((_, c) => (
          <text
            key={`out-${c}`}
            x={outputColX(c) + cellSize}
            y={outputColY + 2 * cellSize + 18}
            textAnchor="middle"
            style={{ fill: 'var(--color-text-secondary)', fontSize: 10, fontFamily: 'var(--font-mono, monospace)', opacity: t }}
          >
            {c}
          </text>
        ))}

        {/* Log-det readout */}
        <text x={W / 2} y={Hsvg - 6} textAnchor="middle"
              style={{ fill: 'var(--color-accent)', fontSize: 12, fontFamily: 'var(--font-mono, monospace)' }}>
          log |det ∂y/∂x| = 0 — squeeze is a permutation
        </text>
      </svg>
    </div>
  );
}
