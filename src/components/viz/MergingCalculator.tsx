import { useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';

// -----------------------------------------------------------------------------
// Viz 10.1 — Interactive e-value merging calculator.
//
// Dropdown for m. Sliders for individual E^(i). Displays:
//   arithmetic mean, geometric mean, product (with "indep. required" badge),
//   Bonferroni p-equivalent.
//
// Pure arithmetic — no charts.
// -----------------------------------------------------------------------------

const M_OPTIONS = [2, 5, 10, 20];

export default function MergingCalculator() {
  const { ref: containerRef } = useResizeObserver<HTMLDivElement>();
  const [m, setM] = useState(5);
  const [values, setValues] = useState<number[]>(() => Array(5).fill(5));

  // resync values if m changes
  const adjustedValues = useMemo(() => {
    const out = [...values];
    while (out.length < m) out.push(5);
    out.length = m;
    return out;
  }, [m, values]);

  const am = adjustedValues.reduce((s, v) => s + v, 0) / m;
  const prod = adjustedValues.reduce((s, v) => s * v, 1);
  const gm = Math.pow(prod, 1 / m);
  const minE = Math.min(...adjustedValues);
  // Bonferroni "p-equivalent": apply Vovk-1/2 calibration to each E (gives p),
  // then min*m. Vovk-1/2 calibrator: f_{1/2}(p) = 0.5 p^{-1/2} ⇒ inverting,
  // p ≈ (2/E)² for E ≥ 0.5; clamp to (0, 1].
  const pEachVovk = adjustedValues.map((E) => Math.min(1, (2 / Math.max(E, 0.5)) ** 2));
  const pMin = Math.min(...pEachVovk);
  const pBonf = Math.min(1, m * pMin);

  // p-equivalent of arithmetic mean via same Vovk-1/2: p = (2/E_AM)² clamped.
  const pAM = Math.min(1, (2 / Math.max(am, 0.5)) ** 2);
  const pProd = Math.min(1, (2 / Math.max(prod, 0.5)) ** 2);

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap', alignItems: 'center' }}>
        <label>
          <span>Number of e-values m: </span>
          <select value={m} onChange={(e) => setM(Number(e.target.value))} style={{ padding: '4px 8px', border: '1px solid var(--color-border)', borderRadius: '4px' }}>
            {M_OPTIONS.map((mo) => <option key={mo} value={mo}>{mo}</option>)}
          </select>
        </label>
        <button
          type="button"
          onClick={() => setValues(Array(m).fill(5))}
          style={{ padding: '4px 10px', border: '1px solid var(--color-border)', borderRadius: '4px', background: 'var(--color-surface)', cursor: 'pointer', fontSize: '12px' }}
        >
          reset all to 5
        </button>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '8px', padding: '8px 0' }}>
        {adjustedValues.map((v, i) => (
          <label key={i} style={{ fontSize: '12px', display: 'flex', flexDirection: 'column' }}>
            <span>E<sup>({i + 1})</sup> = <strong>{v.toFixed(1)}</strong></span>
            <input
              type="range"
              min={0}
              max={50}
              step={0.5}
              value={v}
              aria-label={`E ${i + 1}`}
              onChange={(e) => {
                const newVal = Number(e.target.value);
                const next = [...adjustedValues];
                next[i] = newVal;
                setValues(next);
              }}
            />
          </label>
        ))}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '10px', padding: '10px 0', fontFamily: 'var(--font-mono)', fontSize: '13px' }}>
        <div style={{ padding: '10px 14px', border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-surface)' }}>
          <div style={{ fontWeight: 600, color: 'var(--color-accent)' }}>Arithmetic mean E^AM</div>
          <div style={{ marginTop: '6px' }}>E^AM = <strong>{am.toFixed(3)}</strong></div>
          <div style={{ fontSize: '11px', color: 'var(--color-text-secondary)' }}>p-equiv (Vovk-1/2): {pAM.toExponential(2)}</div>
          <div style={{ fontSize: '11px', color: 'var(--color-text-secondary)' }}>Valid under ANY dependence.</div>
        </div>
        <div style={{ padding: '10px 14px', border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-surface)' }}>
          <div style={{ fontWeight: 600, color: '#7c3aed' }}>Product E^prod <span style={{ fontSize: '10px', color: '#c44545' }}>indep. required</span></div>
          <div style={{ marginTop: '6px' }}>E^prod = <strong>{prod.toExponential(3)}</strong></div>
          <div style={{ fontSize: '11px', color: 'var(--color-text-secondary)' }}>p-equiv (Vovk-1/2): {pProd.toExponential(2)}</div>
          <div style={{ fontSize: '11px', color: 'var(--color-text-secondary)' }}>Invalid under correlation — see Viz 10.2.</div>
        </div>
        <div style={{ padding: '10px 14px', border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-surface)' }}>
          <div style={{ fontWeight: 600, color: 'var(--color-text-secondary)' }}>Geometric mean E^GM</div>
          <div style={{ marginTop: '6px' }}>E^GM = <strong>{gm.toFixed(3)}</strong></div>
          <div style={{ fontSize: '11px', color: 'var(--color-text-secondary)' }}>Dominated by E^AM (AM–GM).</div>
        </div>
        <div style={{ padding: '10px 14px', border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-surface)' }}>
          <div style={{ fontWeight: 600, color: '#c44545' }}>Bonferroni p (p-value path)</div>
          <div style={{ marginTop: '6px' }}>min p × m = <strong>{pBonf.toExponential(2)}</strong></div>
          <div style={{ fontSize: '11px', color: 'var(--color-text-secondary)' }}>Always loses factor of m. Use E^AM instead.</div>
        </div>
      </div>
    </div>
  );
}
