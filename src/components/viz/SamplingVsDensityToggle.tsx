// =============================================================================
// SamplingVsDensityToggle.tsx
//
// §3.3 — Two-panel side-by-side diagram. Reader toggles between "sample" mode
// and "evaluate density" mode; in each mode the relevant operations highlight
// along the layer stack (forward arrows light up for sampling, inverse arrows
// for density). Three architectures (coupling / MAF / IAF) selectable.
//
// Pure React + SVG. The arrow highlighting and the parallel-vs-sequential
// readout are the educational payload.
// =============================================================================

import { useState } from 'react';

type Mode = 'sample' | 'density';
type Arch = 'coupling' | 'maf' | 'iaf';

const ARCH_LABELS: Record<Arch, string> = {
  coupling: 'Coupling (RealNVP)',
  maf: 'MAF',
  iaf: 'IAF',
};

interface Cost {
  sample: string;
  density: string;
  sampleParallel: boolean;
  densityParallel: boolean;
}

const COSTS: Record<Arch, Cost> = {
  coupling: { sample: '1 forward pass', density: '1 inverse pass', sampleParallel: true, densityParallel: true },
  maf: { sample: 'd sequential passes', density: '1 forward pass', sampleParallel: false, densityParallel: true },
  iaf: { sample: '1 forward pass', density: 'd sequential passes', sampleParallel: true, densityParallel: false },
};

export default function SamplingVsDensityToggle() {
  const [mode, setMode] = useState<Mode>('sample');
  const [arch, setArch] = useState<Arch>('coupling');

  const cost = COSTS[arch];
  const direction = mode === 'sample' ? 'forward' : 'reverse';
  const layerCount = 4;
  const layerWidth = 90;
  const spacing = 30;
  const x0 = 60;
  const yMid = 110;

  const isHighlighted = (which: 'forward' | 'reverse') => which === direction;

  return (
    <div className="my-8 not-prose" style={{ border: '1px solid var(--color-border)', borderRadius: 8, padding: 20 }}>
      {/* Top controls */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginBottom: 16, alignItems: 'center' }}>
        <div>
          <span style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginRight: 8 }}>Mode:</span>
          {(['sample', 'density'] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              aria-pressed={mode === m}
              style={{
                marginRight: 6,
                padding: '4px 12px',
                fontSize: 13,
                borderRadius: 4,
                border: '1px solid var(--color-border)',
                background: mode === m ? 'var(--color-accent)' : 'transparent',
                color: mode === m ? 'white' : 'var(--color-text)',
                cursor: 'pointer',
              }}
            >
              {m === 'sample' ? 'Sample' : 'Evaluate density'}
            </button>
          ))}
        </div>
        <div>
          <span style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginRight: 8 }}>Architecture:</span>
          {(Object.keys(ARCH_LABELS) as Arch[]).map((a) => (
            <button
              key={a}
              onClick={() => setArch(a)}
              aria-pressed={arch === a}
              style={{
                marginRight: 6,
                padding: '4px 10px',
                fontSize: 12,
                borderRadius: 4,
                border: '1px solid var(--color-border)',
                background: arch === a ? 'var(--color-text)' : 'transparent',
                color: arch === a ? 'var(--color-bg)' : 'var(--color-text)',
                cursor: 'pointer',
              }}
            >
              {ARCH_LABELS[a]}
            </button>
          ))}
        </div>
      </div>

      {/* Layer stack visualization */}
      <svg viewBox="0 0 680 220" width="100%" style={{ maxWidth: '100%', height: 'auto', maxHeight: 280 }}>
        <defs>
          <marker id="svd-arrow-active" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8"
                  orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" style={{ fill: 'var(--color-accent)' }} />
          </marker>
          <marker id="svd-arrow-dim" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8"
                  orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" style={{ fill: 'var(--color-border)' }} />
          </marker>
        </defs>

        {/* z box (left) */}
        <rect x={x0 - 50} y={yMid - 25} width={50} height={50} rx={4} fill="none"
              style={{ stroke: 'var(--color-border)', strokeWidth: 1.5 }} />
        <text x={x0 - 25} y={yMid + 5} textAnchor="middle"
              style={{ fill: 'var(--color-text)', fontSize: 14, fontFamily: 'var(--font-serif, serif)' }}>z</text>

        {/* Layer boxes T_1, T_2, T_3, T_4 */}
        {Array.from({ length: layerCount }).map((_, k) => {
          const x = x0 + k * (layerWidth + spacing);
          return (
            <g key={k}>
              <rect x={x} y={yMid - 25} width={layerWidth} height={50} rx={4}
                    fill={'color-mix(in srgb, var(--color-accent) 12%, transparent)'}
                    style={{ stroke: 'var(--color-accent)', strokeWidth: 1.5 }} />
              <text x={x + layerWidth / 2} y={yMid + 5} textAnchor="middle"
                    style={{ fill: 'var(--color-text)', fontSize: 13, fontFamily: 'var(--font-serif, serif)' }}>
                T_{k + 1}
              </text>
            </g>
          );
        })}

        {/* x box (right) */}
        <rect x={x0 + layerCount * (layerWidth + spacing)} y={yMid - 25} width={50} height={50} rx={4} fill="none"
              style={{ stroke: 'var(--color-border)', strokeWidth: 1.5 }} />
        <text x={x0 + layerCount * (layerWidth + spacing) + 25} y={yMid + 5} textAnchor="middle"
              style={{ fill: 'var(--color-text)', fontSize: 14, fontFamily: 'var(--font-serif, serif)' }}>x</text>

        {/* Forward arrows (above) */}
        {Array.from({ length: layerCount + 1 }).map((_, k) => {
          const x1 = k === 0 ? x0 - 50 + 50 : x0 + (k - 1) * (layerWidth + spacing) + layerWidth;
          const x2 = k === layerCount ? x0 + layerCount * (layerWidth + spacing) : x0 + k * (layerWidth + spacing);
          const active = isHighlighted('forward');
          return (
            <line
              key={`fwd-${k}`}
              x1={x1 + 4}
              y1={yMid - 50}
              x2={x2 - 4}
              y2={yMid - 50}
              style={{ stroke: active ? 'var(--color-accent)' : 'var(--color-border)', strokeWidth: active ? 2 : 1.2 }}
              markerEnd={active ? 'url(#svd-arrow-active)' : 'url(#svd-arrow-dim)'}
            />
          );
        })}
        <text x={x0 + (layerCount * (layerWidth + spacing)) / 2} y={yMid - 60} textAnchor="middle"
              style={{
                fill: isHighlighted('forward') ? 'var(--color-accent)' : 'var(--color-text-secondary)',
                fontSize: 12,
                fontFamily: 'var(--font-sans, sans-serif)',
                fontWeight: isHighlighted('forward') ? 600 : 400,
              }}>
          forward — sampling
        </text>

        {/* Reverse arrows (below) */}
        {Array.from({ length: layerCount + 1 }).map((_, k) => {
          const x1 = k === 0 ? x0 - 50 + 50 : x0 + (k - 1) * (layerWidth + spacing) + layerWidth;
          const x2 = k === layerCount ? x0 + layerCount * (layerWidth + spacing) : x0 + k * (layerWidth + spacing);
          const active = isHighlighted('reverse');
          return (
            <line
              key={`rev-${k}`}
              x1={x2 - 4}
              y1={yMid + 50}
              x2={x1 + 4}
              y2={yMid + 50}
              style={{ stroke: active ? 'var(--color-accent)' : 'var(--color-border)', strokeWidth: active ? 2 : 1.2 }}
              markerEnd={active ? 'url(#svd-arrow-active)' : 'url(#svd-arrow-dim)'}
            />
          );
        })}
        <text x={x0 + (layerCount * (layerWidth + spacing)) / 2} y={yMid + 75} textAnchor="middle"
              style={{
                fill: isHighlighted('reverse') ? 'var(--color-accent)' : 'var(--color-text-secondary)',
                fontSize: 12,
                fontFamily: 'var(--font-sans, sans-serif)',
                fontWeight: isHighlighted('reverse') ? 600 : 400,
              }}>
          reverse — density evaluation
        </text>
      </svg>

      {/* Cost readout */}
      <div style={{ marginTop: 8, padding: 12, background: 'color-mix(in srgb, var(--color-accent) 6%, transparent)', borderRadius: 6 }}>
        <div style={{ fontSize: 13, color: 'var(--color-text)' }}>
          <strong>{ARCH_LABELS[arch]}</strong> — {mode === 'sample' ? 'sampling cost: ' : 'density cost: '}
          <span style={{
            fontFamily: 'var(--font-mono, monospace)',
            color: mode === 'sample' ? (cost.sampleParallel ? 'var(--color-accent)' : '#D97706') : (cost.densityParallel ? 'var(--color-accent)' : '#D97706'),
          }}>
            {mode === 'sample' ? cost.sample : cost.density}
          </span>
          <span style={{ marginLeft: 12, fontSize: 12, fontStyle: 'italic', color: 'var(--color-text-secondary)' }}>
            {mode === 'sample'
              ? cost.sampleParallel ? '(parallel across dimensions)' : '(sequential across dimensions)'
              : cost.densityParallel ? '(parallel across dimensions)' : '(sequential across dimensions)'}
          </span>
        </div>
      </div>
    </div>
  );
}
