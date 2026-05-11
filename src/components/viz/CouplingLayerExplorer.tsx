// =============================================================================
// CouplingLayerExplorer.tsx
//
// §4 — Interactive split-and-transform on 2-D data. Reader chooses the mask
// via a 2-cell toggle (mask = [1, 0] vs [0, 1]), toggles affine vs additive
// coupling, and sees forward / inverse computations on a single point. A
// 2×2 Jacobian display highlights the block-triangular structure.
// =============================================================================

import { useState, useMemo } from 'react';
import { AffineCoupling } from './shared/normalizing-flows';
import { useResizeObserver } from './shared/useResizeObserver';

export default function CouplingLayerExplorer() {
  const [maskIdx, setMaskIdx] = useState<0 | 1>(0); // 0 = [1, 0]; 1 = [0, 1]
  const [coupType, setCoupType] = useState<'additive' | 'affine'>('affine');
  const [z0, setZ0] = useState(0.6);
  const [z1, setZ1] = useState(-0.4);

  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const W = Math.max(420, Math.min(cw || 780, 820));

  const mask = useMemo(() => (maskIdx === 0 ? new Float64Array([1, 0]) : new Float64Array([0, 1])), [maskIdx]);

  const { x, jacobian, logDet, s, t } = useMemo(() => {
    // For "additive" we set scaleAmp = 0 so exp(s) = 1, i.e., x_B = z_B + t.
    const scaleAmp = coupType === 'additive' ? 0.0001 : 1.0;
    const layer = new AffineCoupling({ d: 2, mask, seed: 314 + maskIdx * 13, scaleAmp, paramScale: 0.9 });
    const zVec = new Float64Array([z0, z1]);
    const fwd = layer.forward(zVec);
    const J = layer.jacobian(zVec);
    // Pull out the s, t values for display.
    let sVal = 0;
    let tVal = 0;
    if (mask[0] === 1) {
      // mask = [1, 0]: A = {0}, B = {1}; s_1, t_1 are functions of z_A = z_0.
      const dotS = layer.bs[0] + layer.Ws[0][0] * z0;
      const dotT = layer.bt[0] + layer.Wt[0][0] * z0;
      sVal = layer.scaleAmp * Math.tanh(dotS);
      tVal = dotT;
    } else {
      // mask = [0, 1]: A = {1}, B = {0}; s_0, t_0 are functions of z_A = z_1.
      const dotS = layer.bs[0] + layer.Ws[0][0] * z1;
      const dotT = layer.bt[0] + layer.Wt[0][0] * z1;
      sVal = layer.scaleAmp * Math.tanh(dotS);
      tVal = dotT;
    }
    return { x: fwd.x, jacobian: J, logDet: fwd.logDet, s: sVal, t: tVal };
  }, [mask, coupType, maskIdx, z0, z1]);

  return (
    <div ref={containerRef} className="my-8 not-prose" style={{ border: '1px solid var(--color-border)', borderRadius: 8, padding: 20 }}>
      {/* Controls row */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginBottom: 16, alignItems: 'center' }}>
        <div>
          <span style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginRight: 8 }}>Mask:</span>
          {[0, 1].map((idx) => (
            <button
              key={idx}
              onClick={() => setMaskIdx(idx as 0 | 1)}
              aria-pressed={maskIdx === idx}
              style={{
                marginRight: 6,
                padding: '4px 10px',
                fontSize: 12,
                borderRadius: 4,
                border: '1px solid var(--color-border)',
                background: maskIdx === idx ? 'var(--color-accent)' : 'transparent',
                color: maskIdx === idx ? 'white' : 'var(--color-text)',
                cursor: 'pointer',
                fontFamily: 'var(--font-mono, monospace)',
              }}
            >
              [{idx === 0 ? '1, 0' : '0, 1'}]
            </button>
          ))}
        </div>
        <div>
          <span style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginRight: 8 }}>Coupling:</span>
          {(['additive', 'affine'] as const).map((c) => (
            <button
              key={c}
              onClick={() => setCoupType(c)}
              aria-pressed={coupType === c}
              style={{
                marginRight: 6,
                padding: '4px 10px',
                fontSize: 12,
                borderRadius: 4,
                border: '1px solid var(--color-border)',
                background: coupType === c ? 'var(--color-text)' : 'transparent',
                color: coupType === c ? 'var(--color-bg)' : 'var(--color-text)',
                cursor: 'pointer',
              }}
            >
              {c === 'additive' ? 'Additive (NICE)' : 'Affine (RealNVP)'}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginBottom: 14, alignItems: 'center' }}>
        <span style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>z_0:</span>
        <input type="range" min={-2} max={2} step={0.05} value={z0} onChange={(e) => setZ0(Number(e.target.value))}
               style={{ flex: 1, maxWidth: 200 }} aria-label="z_0" />
        <span style={{ fontFamily: 'var(--font-mono, monospace)', fontSize: 12, color: 'var(--color-text-secondary)' }}>
          {z0.toFixed(2)}
        </span>
        <span style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginLeft: 16 }}>z_1:</span>
        <input type="range" min={-2} max={2} step={0.05} value={z1} onChange={(e) => setZ1(Number(e.target.value))}
               style={{ flex: 1, maxWidth: 200 }} aria-label="z_1" />
        <span style={{ fontFamily: 'var(--font-mono, monospace)', fontSize: 12, color: 'var(--color-text-secondary)' }}>
          {z1.toFixed(2)}
        </span>
      </div>

      {/* Three panel display: z, transformation breakdown, x */}
      <div style={{ display: 'grid', gridTemplateColumns: W < 640 ? '1fr' : 'repeat(3, 1fr)', gap: 16 }}>
        <div style={{ padding: 12, border: '1px solid var(--color-border)', borderRadius: 6 }}>
          <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 8 }}>Input z</div>
          <div style={{ fontFamily: 'var(--font-mono, monospace)', fontSize: 14, color: 'var(--color-text)' }}>
            z_0 = {z0.toFixed(3)} {mask[0] === 1 && <span style={{ color: 'var(--color-accent)', fontSize: 10 }}> ← passes through</span>}
            <br />
            z_1 = {z1.toFixed(3)} {mask[1] === 1 && <span style={{ color: 'var(--color-accent)', fontSize: 10 }}> ← passes through</span>}
          </div>
        </div>

        <div style={{ padding: 12, border: '1px solid var(--color-border)', borderRadius: 6 }}>
          <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 8 }}>s, t evaluated</div>
          <div style={{ fontFamily: 'var(--font-mono, monospace)', fontSize: 12, color: 'var(--color-text)', lineHeight: 1.6 }}>
            s(z_A) = {coupType === 'additive' ? '0' : s.toFixed(4)}<br />
            t(z_A) = {t.toFixed(4)}<br />
            log|det| = {logDet.toFixed(4)}
          </div>
        </div>

        <div style={{ padding: 12, border: '1px solid var(--color-border)', borderRadius: 6 }}>
          <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 8 }}>Output x</div>
          <div style={{ fontFamily: 'var(--font-mono, monospace)', fontSize: 14, color: 'var(--color-text)' }}>
            x_0 = {x[0].toFixed(3)}
            <br />
            x_1 = {x[1].toFixed(3)}
          </div>
        </div>
      </div>

      {/* Jacobian display */}
      <div style={{ marginTop: 16, padding: 12, border: '1px solid var(--color-border)', borderRadius: 6 }}>
        <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 8 }}>
          Jacobian ∂x/∂z (block-triangular: upper-right zero highlighted in green)
        </div>
        <div style={{ display: 'inline-grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontFamily: 'var(--font-mono, monospace)', fontSize: 14 }}>
          {jacobian.map((row, i) =>
            Array.from(row).map((cell, j) => {
              const passThroughI = mask[i] === 1;
              const passThroughJ = mask[j] === 1;
              const isUpperRightZero = passThroughI && !passThroughJ;
              const isDiag = i === j && !passThroughI;
              const isLowerLeft = !passThroughI && passThroughJ;
              return (
                <div
                  key={`${i}-${j}`}
                  style={{
                    padding: '6px 12px',
                    background: isUpperRightZero
                      ? 'color-mix(in srgb, var(--color-accent) 25%, transparent)'
                      : isDiag
                      ? 'color-mix(in srgb, #D97706 18%, transparent)'
                      : isLowerLeft
                      ? 'color-mix(in srgb, #534AB7 18%, transparent)'
                      : 'transparent',
                    color: 'var(--color-text)',
                    textAlign: 'center',
                    border: '1px solid var(--color-border)',
                  }}
                >
                  {Math.abs(cell) < 1e-10 ? '0' : cell.toFixed(4)}
                </div>
              );
            }),
          )}
        </div>
        <div style={{ marginTop: 8, fontSize: 11, color: 'var(--color-text-secondary)', fontStyle: 'italic' }}>
          Identity on pass-through rows (purple shading where they appear), upper-right zero block (green) — the only
          fact (4.10) needs about the Jacobian. The lower-right diagonal entry (amber) is exp(s); its log is log|det|.
        </div>
      </div>
    </div>
  );
}
