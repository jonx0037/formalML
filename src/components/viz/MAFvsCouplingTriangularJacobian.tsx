// =============================================================================
// MAFvsCouplingTriangularJacobian.tsx
//
// §5 — A 2-panel comparison. Top: coupling layer's Jacobian on d=4 toy, with
// upper-left identity and lower-right diagonal blocks highlighted. Bottom:
// MAF layer's Jacobian showing the full lower-triangular structure. The
// log-det formulas are displayed under each.
// =============================================================================

import { useMemo, useState } from 'react';
import {
  AffineCoupling,
  MAFLayer,
  alternatingMask,
  sampleStandardNormalBatch,
} from './shared/normalizing-flows';
import { useResizeObserver } from './shared/useResizeObserver';

const D = 4;

export default function MAFvsCouplingTriangularJacobian() {
  const [seed, setSeed] = useState(7);
  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const W = Math.max(420, Math.min(cw || 720, 800));

  const { Jcoup, Jmaf, ldCoup, ldMaf, A, B } = useMemo(() => {
    const z = sampleStandardNormalBatch(1, D, seed * 31)[0];
    const mask = alternatingMask(D, 0);
    const coup = new AffineCoupling({ d: D, mask, seed: seed * 17, scaleAmp: 1.0, paramScale: 0.7 });
    const Jcoup = coup.jacobian(z);
    const ldCoup = coup.forward(z).logDet;
    const maf = new MAFLayer({ d: D, seed: seed * 19, scaleAmp: 1.0, paramScale: 0.7 });
    const Jmaf = maf.jacobian(z);
    const ldMaf = maf.forward(z).logDet;
    const A = Array.from({ length: D }).map((_, i) => mask[i] === 1);
    const B = A.map((a) => !a);
    return { Jcoup, Jmaf, ldCoup, ldMaf, A, B };
  }, [seed]);

  const cellSize = 44;
  const matrixOffset = 40;

  function MatrixDisplay({
    title,
    J,
    classify,
    legend,
  }: {
    title: string;
    J: Float64Array[];
    classify: (i: number, j: number, val: number) => 'identity' | 'diag' | 'lower' | 'zero' | 'other';
    legend: { label: string; color: string }[];
  }) {
    return (
      <div style={{ flex: 1, minWidth: 280, padding: 12, border: '1px solid var(--color-border)', borderRadius: 6 }}>
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 10, color: 'var(--color-text)' }}>{title}</div>
        <div style={{ display: 'inline-grid', gridTemplateColumns: `repeat(${D}, ${cellSize}px)`, gap: 4 }}>
          {J.map((row, i) =>
            Array.from(row).map((cell, j) => {
              const kind = classify(i, j, cell);
              const bg =
                kind === 'identity' ? 'color-mix(in srgb, #534AB7 22%, transparent)'
                : kind === 'diag' ? 'color-mix(in srgb, #D97706 22%, transparent)'
                : kind === 'lower' ? 'color-mix(in srgb, #2563EB 14%, transparent)'
                : kind === 'zero' ? 'color-mix(in srgb, var(--color-accent) 22%, transparent)'
                : 'transparent';
              const display = Math.abs(cell) < 1e-10 ? '0' : cell.toFixed(2);
              return (
                <div
                  key={`${i}-${j}`}
                  style={{
                    width: cellSize,
                    height: cellSize,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontFamily: 'var(--font-mono, monospace)',
                    fontSize: 11,
                    color: 'var(--color-text)',
                    background: bg,
                    border: '1px solid var(--color-border)',
                    borderRadius: 3,
                  }}
                >
                  {display}
                </div>
              );
            }),
          )}
        </div>
        <div style={{ display: 'flex', gap: 12, marginTop: 12, flexWrap: 'wrap' }}>
          {legend.map((l, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 12, height: 12, background: l.color, borderRadius: 2 }} />
              <span style={{ fontSize: 11, color: 'var(--color-text-secondary)' }}>{l.label}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="my-8 not-prose" style={{ border: '1px solid var(--color-border)', borderRadius: 8, padding: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <span style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>seed:</span>
        <input
          type="range"
          min={1}
          max={20}
          step={1}
          value={seed}
          onChange={(e) => setSeed(Number(e.target.value))}
          aria-label="Random seed"
          style={{ width: 180 }}
        />
        <span style={{ fontFamily: 'var(--font-mono, monospace)', fontSize: 12, color: 'var(--color-text-secondary)' }}>
          {seed}
        </span>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16 }}>
        <MatrixDisplay
          title={`Coupling layer (mask = [1,1,0,0])    log|det| = Σ_{i ∈ B} s_i = ${ldCoup.toFixed(3)}`}
          J={Jcoup}
          classify={(i, j) => {
            const aI = A[i];
            const aJ = A[j];
            if (aI && !aJ) return 'zero'; // upper-right zero block
            if (aI && aJ && i === j) return 'identity'; // pass-through identity
            if (!aI && !aJ && i === j) return 'diag'; // transformed diagonal
            return 'other';
          }}
          legend={[
            { label: 'identity (pass-through)', color: 'color-mix(in srgb, #534AB7 22%, transparent)' },
            { label: 'zero block (UR)', color: 'color-mix(in srgb, var(--color-accent) 22%, transparent)' },
            { label: 'diag exp(s_i) (LR)', color: 'color-mix(in srgb, #D97706 22%, transparent)' },
          ]}
        />
        <MatrixDisplay
          title={`MAF layer    log|det| = Σ_{i=1}^d s_i(x_{<i}) = ${ldMaf.toFixed(3)}`}
          J={Jmaf}
          classify={(i, j, _val) => {
            if (j > i) return 'zero'; // strict upper = 0
            if (i === j) return 'diag';
            return 'lower';
          }}
          legend={[
            { label: 'strict upper = 0', color: 'color-mix(in srgb, var(--color-accent) 22%, transparent)' },
            { label: 'diag exp(s_i)', color: 'color-mix(in srgb, #D97706 22%, transparent)' },
            { label: 'lower-triangular fill', color: 'color-mix(in srgb, #2563EB 14%, transparent)' },
          ]}
        />
      </div>

      <div style={{ marginTop: 12, fontSize: 12, color: 'var(--color-text-secondary)', fontStyle: 'italic' }}>
        Same triangular-Jacobian trick, two different partitions. Coupling: block-triangular with identity in the
        pass-through block; MAF: full lower-triangular. Both deliver O(d) log-det as a sum of d scalars.
      </div>
    </div>
  );
}
