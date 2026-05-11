// =============================================================================
// AutoregressiveFlowExplorer.tsx
//
// §5 — Side-by-side MAF and IAF demonstration on a 4-D toy. Reader toggles
// between sample / density modes and between MAF / IAF. The panel animates
// the operations down the dimension axis: for MAF sampling, a sequential
// cascade highlights one dim at a time (each waiting for the previous);
// for MAF density, all four dim operations highlight simultaneously.
// =============================================================================

import { useState, useEffect } from 'react';

type Mode = 'sample' | 'density';
type Arch = 'maf' | 'iaf';

const D = 4;

export default function AutoregressiveFlowExplorer() {
  const [mode, setMode] = useState<Mode>('sample');
  const [arch, setArch] = useState<Arch>('maf');
  const [step, setStep] = useState(0);

  // Determine whether the (mode, arch) combination is sequential.
  const sequential =
    (arch === 'maf' && mode === 'sample') || (arch === 'iaf' && mode === 'density');

  // Animate sequential mode: cycle step from 0 to D.
  useEffect(() => {
    if (!sequential) {
      setStep(D);
      return;
    }
    setStep(0);
    const id = setInterval(() => {
      setStep((s) => (s >= D ? 0 : s + 1));
    }, 700);
    return () => clearInterval(id);
  }, [sequential, mode, arch]);

  const W = 720;
  const H = 240;
  const PAD = 30;
  const dimGap = (W - PAD * 2) / (D + 1);

  // For each dim i, the "highlight" status depends on mode.
  const isActive = (i: number) => (sequential ? i < step : true);

  // Cost readout
  const cost = sequential ? `${D} sequential MADE passes` : '1 MADE pass';
  const costColor = sequential ? '#D97706' : 'var(--color-accent)';

  return (
    <div className="my-8 not-prose" style={{ border: '1px solid var(--color-border)', borderRadius: 8, padding: 20 }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginBottom: 16, alignItems: 'center' }}>
        <div>
          <span style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginRight: 8 }}>Architecture:</span>
          {(['maf', 'iaf'] as Arch[]).map((a) => (
            <button
              key={a}
              onClick={() => setArch(a)}
              aria-pressed={arch === a}
              style={{
                marginRight: 6,
                padding: '4px 12px',
                fontSize: 12,
                borderRadius: 4,
                border: '1px solid var(--color-border)',
                background: arch === a ? 'var(--color-text)' : 'transparent',
                color: arch === a ? 'var(--color-bg)' : 'var(--color-text)',
                cursor: 'pointer',
              }}
            >
              {a.toUpperCase()}
            </button>
          ))}
        </div>
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
                fontSize: 12,
                borderRadius: 4,
                border: '1px solid var(--color-border)',
                background: mode === m ? 'var(--color-accent)' : 'transparent',
                color: mode === m ? 'white' : 'var(--color-text)',
                cursor: 'pointer',
              }}
            >
              {m === 'sample' ? 'Sample (z → x)' : 'Density (x → z)'}
            </button>
          ))}
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: '100%', height: 'auto' }}>
        {/* Title bar */}
        <text x={W / 2} y={26} textAnchor="middle"
              style={{ fill: 'var(--color-text)', fontSize: 13, fontWeight: 600, fontFamily: 'var(--font-sans, sans-serif)' }}>
          {arch.toUpperCase()} — {mode === 'sample' ? 'sampling' : 'density evaluation'} ({sequential ? 'sequential' : 'parallel'})
        </text>

        {/* Dim columns */}
        {Array.from({ length: D }).map((_, i) => {
          const x = PAD + (i + 0.5) * dimGap + 30;
          const active = isActive(i);
          const labelTop = mode === 'sample' ? 'z' : 'x';
          const labelBot = mode === 'sample' ? 'x' : 'z';
          return (
            <g key={i}>
              {/* Top label (z_i or x_i) */}
              <rect x={x - 18} y={70} width={36} height={30} rx={4}
                    fill={active ? 'color-mix(in srgb, var(--color-accent) 22%, transparent)' : 'transparent'}
                    style={{ stroke: active ? 'var(--color-accent)' : 'var(--color-border)', strokeWidth: 1.2 }} />
              <text x={x} y={90} textAnchor="middle"
                    style={{ fill: 'var(--color-text)', fontSize: 12, fontFamily: 'var(--font-serif, serif)' }}>
                {labelTop}_{i}
              </text>

              {/* Arrow */}
              <line x1={x} y1={108} x2={x} y2={140}
                    style={{ stroke: active ? 'var(--color-accent)' : 'var(--color-border)', strokeWidth: active ? 2 : 1.2 }} />

              {/* Bottom label */}
              <rect x={x - 18} y={140} width={36} height={30} rx={4}
                    fill={active ? 'color-mix(in srgb, var(--color-accent) 18%, transparent)' : 'transparent'}
                    style={{ stroke: active ? 'var(--color-accent)' : 'var(--color-border)', strokeWidth: 1.2 }} />
              <text x={x} y={160} textAnchor="middle"
                    style={{ fill: 'var(--color-text)', fontSize: 12, fontFamily: 'var(--font-serif, serif)' }}>
                {labelBot}_{i}
              </text>
            </g>
          );
        })}

        {/* Conditioning hint */}
        <text x={W / 2} y={H - 28} textAnchor="middle"
              style={{ fill: 'var(--color-text-secondary)', fontSize: 11, fontFamily: 'var(--font-sans, sans-serif)', fontStyle: 'italic' }}>
          {arch === 'maf' && mode === 'sample' && 'x_i depends on x_{<i} — must compute in order'}
          {arch === 'maf' && mode === 'density' && 'z_i = (x_i - t(x_{<i})) · exp(-s(x_{<i})) — parallel over i'}
          {arch === 'iaf' && mode === 'sample' && 'x_i = z_i · exp(s(z_{<i})) + t(z_{<i}) — parallel over i'}
          {arch === 'iaf' && mode === 'density' && 'z_i depends on z_{<i} which depends on x — must compute in order'}
        </text>

        <text x={W / 2} y={H - 8} textAnchor="middle"
              style={{ fill: costColor, fontSize: 12, fontFamily: 'var(--font-mono, monospace)', fontWeight: 600 }}>
          cost: {cost}
        </text>
      </svg>
    </div>
  );
}
