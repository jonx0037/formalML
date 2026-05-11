// =============================================================================
// InvertibilityConstraintExplorer.tsx
//
// §2.3 — Demonstrates the fold-and-density-blowup pathology. Reader perturbs
// a 1-D map from injective (T(z) = z + 0.5 tanh(z)) through the bifurcation
// at T'(z*) = 0 to non-injective (T(z) = z - 1.2 tanh(z)) via a slider.
//
// As the coefficient crosses the critical value, a fold appears in the
// graph of T; the pushed-forward density develops a vertical asymptote at
// the fold's image; and the formula (2.2) visibly breaks. A live readout
// shows min_z |T'(z)| in green when positive and red when zero.
// =============================================================================

import { useState, useMemo } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';

const Z_MIN = -4;
const Z_MAX = 4;
const N_GRID = 401;

function gaussianPDF(z: number): number {
  return Math.exp(-(z * z) / 2) / Math.sqrt(2 * Math.PI);
}

// T_α(z) = z + α tanh(z); for α > -1 this is injective.
// For α < -1 it folds (T'(0) = 1 + α < 0, but T'(±∞) → 1, so two roots of T').
function Talpha(z: number, alpha: number): number {
  return z + alpha * Math.tanh(z);
}
function TalphaPrime(z: number, alpha: number): number {
  const th = Math.tanh(z);
  return 1 + alpha * (1 - th * th);
}

export default function InvertibilityConstraintExplorer() {
  const [alpha, setAlpha] = useState(-0.5);
  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const W = Math.max(380, Math.min(cw || 720, 800));
  const H = 380;

  const { zPts, tVals, primeMin, primeMinSigned, density } = useMemo(() => {
    const zPts: number[] = [];
    const tVals: number[] = [];
    const primeVals: number[] = [];
    for (let i = 0; i < N_GRID; i++) {
      const z = Z_MIN + ((Z_MAX - Z_MIN) * i) / (N_GRID - 1);
      zPts.push(z);
      tVals.push(Talpha(z, alpha));
      primeVals.push(TalphaPrime(z, alpha));
    }
    // The fold-pathology bifurcation is "T'(z) crosses zero somewhere," not
    // "T'(z) becomes negative." Track both the signed minimum (for the readout
    // that distinguishes injective from folded) and the absolute minimum
    // (for the `min |T'|` label).
    const primeMinSigned = Math.min(...primeVals);
    const primeMin = Math.min(...primeVals.map(Math.abs));

    // Compute density along T(z) values via change of variables, accumulating
    // mass when multiple z's map to the same x bin.
    const xMin = -6;
    const xMax = 6;
    const nBins = 200;
    const binW = (xMax - xMin) / nBins;
    const density = new Float64Array(nBins);
    for (let i = 0; i < zPts.length - 1; i++) {
      const xMid = 0.5 * (tVals[i] + tVals[i + 1]);
      const bin = Math.floor((xMid - xMin) / binW);
      if (bin < 0 || bin >= nBins) continue;
      const pBase = gaussianPDF(zPts[i]);
      const tp = Math.abs(0.5 * (primeVals[i] + primeVals[i + 1]));
      if (tp > 1e-6) density[bin] += (pBase / tp) * binW * ((Z_MAX - Z_MIN) / (N_GRID - 1)) / binW;
      else density[bin] += 100; // visualize blowup
    }
    return { zPts, tVals, primeMin, primeMinSigned, density };
  }, [alpha]);

  const PAD_LEFT = 50;
  const PAD_RIGHT = 24;
  const PAD_TOP = 30;
  const PAD_BOT = 50;
  const plotW = (W - PAD_LEFT - PAD_RIGHT) / 2 - 12;
  const plotH = H - PAD_TOP - PAD_BOT;

  // Left plot: graph of T(z)
  const xL0 = PAD_LEFT;
  const xR0 = PAD_LEFT + plotW + 24;
  const scaleZ = (z: number) => xL0 + ((z - Z_MIN) / (Z_MAX - Z_MIN)) * plotW;
  const scaleT = (t: number) => PAD_TOP + plotH - ((t + 8) / 16) * plotH;
  const tCurve = zPts.map((z, i) => `${i === 0 ? 'M' : 'L'} ${scaleZ(z)} ${scaleT(tVals[i])}`).join(' ');

  // Right plot: density on x-axis (bin midpoints)
  const xMin = -6;
  const xMax = 6;
  const nBins = 200;
  const binW = (xMax - xMin) / nBins;
  const scaleX = (x: number) => xR0 + ((x - xMin) / (xMax - xMin)) * plotW;
  const maxDensityCap = 0.9;
  const scaleD = (d: number) => PAD_TOP + plotH - (Math.min(d, maxDensityCap) / maxDensityCap) * plotH;
  const dCurve = Array.from({ length: nBins }, (_, b) => {
    const x = xMin + (b + 0.5) * binW;
    return `${b === 0 ? 'M' : 'L'} ${scaleX(x)} ${scaleD(density[b])}`;
  }).join(' ');

  // The bifurcation happens at T'(z*) = 0 — so we treat the boundary case as
  // "folded" too (a small epsilon absorbs floating-point round-off in the
  // grid evaluation).
  const folded = primeMinSigned <= 1e-9;

  return (
    <div ref={containerRef} className="my-8 not-prose" style={{ border: '1px solid var(--color-border)', borderRadius: 8, padding: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>α:</span>
        <input
          type="range"
          min={-1.8}
          max={1}
          step={0.05}
          value={alpha}
          onChange={(e) => setAlpha(Number(e.target.value))}
          aria-label="Map coefficient α"
          style={{ flex: 1, maxWidth: 360 }}
        />
        <span style={{ fontFamily: 'var(--font-mono, monospace)', fontSize: 12, color: 'var(--color-text-secondary)' }}>
          T(z) = z + {alpha.toFixed(2)} · tanh(z)
        </span>
        <span style={{
          fontFamily: 'var(--font-mono, monospace)',
          fontSize: 12,
          color: folded ? '#DC2626' : 'var(--color-accent)',
          fontWeight: 600,
        }}>
          min |T'| = {primeMin.toFixed(3)} {folded ? '⟵ folded' : '⟵ injective'}
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: '100%', height: 'auto' }}>
        {/* Left plot frame */}
        <line x1={xL0} y1={PAD_TOP + plotH} x2={xL0 + plotW} y2={PAD_TOP + plotH}
              style={{ stroke: 'var(--color-border)', strokeWidth: 1 }} />
        <line x1={xL0} y1={PAD_TOP} x2={xL0} y2={PAD_TOP + plotH}
              style={{ stroke: 'var(--color-border)', strokeWidth: 1 }} />
        <text x={xL0 + plotW / 2} y={PAD_TOP - 10} textAnchor="middle"
              style={{ fill: 'var(--color-text)', fontSize: 11, fontFamily: 'var(--font-sans, sans-serif)' }}>
          Graph of T(z)
        </text>
        <line x1={xL0} y1={scaleT(0)} x2={xL0 + plotW} y2={scaleT(0)}
              style={{ stroke: 'var(--color-border)', strokeWidth: 0.5, strokeDasharray: '3 3' }} />
        <path d={tCurve} fill="none" style={{ stroke: folded ? '#DC2626' : 'var(--color-accent)', strokeWidth: 1.8 }} />

        {/* Right plot frame */}
        <line x1={xR0} y1={PAD_TOP + plotH} x2={xR0 + plotW} y2={PAD_TOP + plotH}
              style={{ stroke: 'var(--color-border)', strokeWidth: 1 }} />
        <line x1={xR0} y1={PAD_TOP} x2={xR0} y2={PAD_TOP + plotH}
              style={{ stroke: 'var(--color-border)', strokeWidth: 1 }} />
        <text x={xR0 + plotW / 2} y={PAD_TOP - 10} textAnchor="middle"
              style={{ fill: 'var(--color-text)', fontSize: 11, fontFamily: 'var(--font-sans, sans-serif)' }}>
          Pushed-forward density p_X(x)
        </text>
        <path d={dCurve} fill="none" style={{ stroke: folded ? '#DC2626' : 'var(--color-accent)', strokeWidth: 1.8 }} />

        {/* Bottom axis labels */}
        <text x={xL0} y={H - 16}
              style={{ fill: 'var(--color-text-secondary)', fontSize: 10, fontFamily: 'var(--font-mono, monospace)' }}>
          z = {Z_MIN}
        </text>
        <text x={xL0 + plotW} y={H - 16} textAnchor="end"
              style={{ fill: 'var(--color-text-secondary)', fontSize: 10, fontFamily: 'var(--font-mono, monospace)' }}>
          z = {Z_MAX}
        </text>
        <text x={xR0} y={H - 16}
              style={{ fill: 'var(--color-text-secondary)', fontSize: 10, fontFamily: 'var(--font-mono, monospace)' }}>
          x = {xMin}
        </text>
        <text x={xR0 + plotW} y={H - 16} textAnchor="end"
              style={{ fill: 'var(--color-text-secondary)', fontSize: 10, fontFamily: 'var(--font-mono, monospace)' }}>
          x = {xMax}
        </text>
      </svg>

      <div style={{ marginTop: 8, fontSize: 12, color: 'var(--color-text-secondary)', fontStyle: 'italic' }}>
        Drag α below -1 to fold T. The pushed-forward density develops a vertical asymptote at the fold image — the
        change-of-variables formula (2.2) literally divides by zero where T'(z) vanishes. Flow architectures avoid
        this by parameterizing T with strictly positive Jacobian.
      </div>
    </div>
  );
}
