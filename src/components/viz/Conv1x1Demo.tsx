// =============================================================================
// Conv1x1Demo.tsx
//
// §6.2 — Three channel-mixing strategies on a small C = 4 toy:
//   (a) fixed permutation (NICE / RealNVP)
//   (b) dense W (1×1 conv with O(C^3) log-det)
//   (c) LU-parameterized W (Glow's O(C) version)
//
// Reader sees the W matrix as a 4×4 heatmap, the resulting channel-mixing
// pattern, and the log-det readout in each case.
// =============================================================================

import { useState, useMemo } from 'react';
import { Conv1x1Dense, Conv1x1LU, slogdetAbs } from './shared/normalizing-flows';

type Strategy = 'permutation' | 'dense' | 'lu';

const C = 4;

function buildPermutationW(seed: number): Float64Array[] {
  // Deterministic permutation from seed.
  const perm = Array.from({ length: C }, (_, i) => i);
  // Fisher-Yates with a simple seeded LCG.
  let s = seed;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  for (let i = C - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = perm[i];
    perm[i] = perm[j];
    perm[j] = tmp;
  }
  const W: Float64Array[] = [];
  for (let i = 0; i < C; i++) {
    const row = new Float64Array(C);
    row[perm[i]] = 1;
    W.push(row);
  }
  return W;
}

export default function Conv1x1Demo() {
  const [strategy, setStrategy] = useState<Strategy>('lu');
  const [seed, setSeed] = useState(7);

  const { W, logDet, paramCount, costLabel } = useMemo(() => {
    if (strategy === 'permutation') {
      const W = buildPermutationW(seed);
      return { W, logDet: 0, paramCount: 0, costLabel: 'O(1)' };
    }
    if (strategy === 'dense') {
      const conv = new Conv1x1Dense({ C, seed, init: 'gaussian' });
      return {
        W: conv.W,
        logDet: conv.logDetW(),
        paramCount: C * C,
        costLabel: `O(C³) = ${C * C * C}`,
      };
    }
    const conv = new Conv1x1LU({ C, seed });
    const W = conv.buildW();
    return {
      W,
      logDet: conv.logDetW(),
      paramCount: C * C,
      costLabel: `O(C) = ${C}`,
    };
  }, [strategy, seed]);

  // Compute slogdet via LU for verification readout (matches logDet up to numerical precision)
  const slogdet = useMemo(() => slogdetAbs(W), [W]);

  // Max-abs value for color scaling
  const maxAbs = Math.max(...W.flatMap((row) => Array.from(row).map((v) => Math.abs(v))));

  const cellSize = 56;

  return (
    <div className="my-8 not-prose" style={{ border: '1px solid var(--color-border)', borderRadius: 8, padding: 20 }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginBottom: 16, alignItems: 'center' }}>
        <div>
          <span style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginRight: 8 }}>Strategy:</span>
          {(['permutation', 'dense', 'lu'] as Strategy[]).map((s) => (
            <button
              key={s}
              onClick={() => setStrategy(s)}
              aria-pressed={strategy === s}
              style={{
                marginRight: 6,
                padding: '4px 10px',
                fontSize: 12,
                borderRadius: 4,
                border: '1px solid var(--color-border)',
                background: strategy === s ? 'var(--color-accent)' : 'transparent',
                color: strategy === s ? 'white' : 'var(--color-text)',
                cursor: 'pointer',
              }}
            >
              {s === 'permutation' ? 'Fixed permutation' : s === 'dense' ? 'Dense W' : 'LU param (Glow)'}
            </button>
          ))}
        </div>
        <div style={{ flex: 1 }} />
        <div>
          <span style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginRight: 8 }}>seed:</span>
          <input type="range" min={1} max={20} step={1} value={seed} onChange={(e) => setSeed(Number(e.target.value))}
                 aria-label="Seed" style={{ width: 140, verticalAlign: 'middle' }} />
          <span style={{ marginLeft: 6, fontFamily: 'var(--font-mono, monospace)', fontSize: 12, color: 'var(--color-text-secondary)' }}>{seed}</span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {/* Heatmap of W */}
        <div style={{ padding: 12, border: '1px solid var(--color-border)', borderRadius: 6 }}>
          <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 8 }}>W ∈ R^{C}×{C}</div>
          <div style={{ display: 'inline-grid', gridTemplateColumns: `repeat(${C}, ${cellSize}px)`, gap: 2 }}>
            {W.map((row, i) =>
              Array.from(row).map((v, j) => {
                const intensity = maxAbs > 0 ? Math.min(Math.abs(v) / maxAbs, 1) : 0;
                const color = v >= 0
                  ? `rgba(15, 110, 86, ${0.15 + 0.75 * intensity})`
                  : `rgba(220, 38, 38, ${0.15 + 0.75 * intensity})`;
                return (
                  <div
                    key={`${i}-${j}`}
                    style={{
                      width: cellSize,
                      height: cellSize,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      background: color,
                      fontFamily: 'var(--font-mono, monospace)',
                      fontSize: 11,
                      color: intensity > 0.5 ? 'white' : 'var(--color-text)',
                    }}
                  >
                    {v.toFixed(2)}
                  </div>
                );
              }),
            )}
          </div>
        </div>

        {/* Stats panel */}
        <div style={{ padding: 12, border: '1px solid var(--color-border)', borderRadius: 6 }}>
          <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 8 }}>Properties</div>
          <div style={{ fontFamily: 'var(--font-mono, monospace)', fontSize: 13, color: 'var(--color-text)', lineHeight: 1.8 }}>
            <div>
              log|det W| ={' '}
              <span style={{ color: 'var(--color-accent)', fontWeight: 600 }}>
                {logDet.toFixed(4)}
              </span>
            </div>
            <div>
              slogdet(W) ={' '}
              <span style={{ color: 'var(--color-text-secondary)' }}>{slogdet.toFixed(4)}</span>
            </div>
            <div style={{ marginTop: 12, fontSize: 11, color: 'var(--color-text-secondary)' }}>parameters</div>
            <div>count = {paramCount}</div>
            <div>log-det cost = {costLabel}</div>
          </div>
          <div style={{ marginTop: 16, fontSize: 11, color: 'var(--color-text-secondary)', fontStyle: 'italic' }}>
            {strategy === 'permutation' &&
              'A fixed permutation matrix — exactly one 1 per row/column. log|det| = 0; channel mixing is parameter-free.'}
            {strategy === 'dense' &&
              'A generic dense weight. Same C² parameters as the LU form, but log-det requires the O(C³) LU decomposition every step.'}
            {strategy === 'lu' &&
              'Glow\'s LU parameterization: same C² total parameters, but log|det W| = Σ s_log_i in O(C) — the headline identity (6.4).'}
          </div>
        </div>
      </div>
    </div>
  );
}
