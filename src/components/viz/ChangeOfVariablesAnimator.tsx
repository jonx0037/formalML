// =============================================================================
// ChangeOfVariablesAnimator.tsx
//
// §2 — 1-D interactive animator showing the geometric content of equations
// (2.2) and (2.3). Slider parameterizes a one-parameter family of monotone
// maps T_θ(z) = z + θ tanh(z). As the reader drags θ, the pushforward
// density updates in real time. The local stretch factor 1/|T_θ'(z)| is
// shown as a colored band on the z-axis.
//
// Pure math, closed-form everywhere — no shared module needed beyond
// standard-Gaussian evaluation.
// =============================================================================

import { useState, useMemo, useRef } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';

const Z_MIN = -4;
const Z_MAX = 4;
const X_MIN = -6;
const X_MAX = 6;
const N_GRID = 401;

function gaussianPDF(z: number): number {
  return Math.exp(-(z * z) / 2) / Math.sqrt(2 * Math.PI);
}

// T_θ(z) = z + θ tanh(z); T'_θ(z) = 1 + θ sech²(z) = 1 + θ (1 - tanh²(z))
function Tofz(z: number, theta: number): number {
  return z + theta * Math.tanh(z);
}
function TPrime(z: number, theta: number): number {
  const th = Math.tanh(z);
  return 1 + theta * (1 - th * th);
}

// Inverse via bisection (monotone increasing when θ > -1, which we enforce).
function Tinv(x: number, theta: number): number {
  let lo = Z_MIN * 1.4;
  let hi = Z_MAX * 1.4;
  for (let i = 0; i < 60; i++) {
    const mid = 0.5 * (lo + hi);
    if (Tofz(mid, theta) < x) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

export default function ChangeOfVariablesAnimator() {
  const [theta, setTheta] = useState(1.0);
  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const widthRef = useRef(720);
  const W = Math.max(360, Math.min(cw || 720, 800));
  widthRef.current = W;
  const H = 360;

  const { zPts, basePdf, xPts, pushPdf, stretchColors } = useMemo(() => {
    const zPts: number[] = [];
    const basePdf: number[] = [];
    const xPtsLocal: number[] = [];
    const pushPdf: number[] = [];
    const stretchColors: { z: number; rho: number }[] = [];
    for (let i = 0; i < N_GRID; i++) {
      const z = Z_MIN + ((Z_MAX - Z_MIN) * i) / (N_GRID - 1);
      zPts.push(z);
      basePdf.push(gaussianPDF(z));
      // Stretch factor 1/|T'(z)|: > 1 means compressed (mass piles up), < 1 means stretched.
      const rho = 1 / Math.abs(TPrime(z, theta));
      stretchColors.push({ z, rho });
    }
    for (let i = 0; i < N_GRID; i++) {
      const x = X_MIN + ((X_MAX - X_MIN) * i) / (N_GRID - 1);
      xPtsLocal.push(x);
      const z = Tinv(x, theta);
      const pdf = gaussianPDF(z) / Math.abs(TPrime(z, theta));
      pushPdf.push(pdf);
    }
    return { zPts, basePdf, xPts: xPtsLocal, pushPdf, stretchColors };
  }, [theta]);

  const PAD_LEFT = 50;
  const PAD_RIGHT = 24;
  const PAD_TOP = 24;
  const PAD_BOT = 50;
  const plotW = W - PAD_LEFT - PAD_RIGHT;
  const plotTopH = (H - PAD_TOP - PAD_BOT) * 0.4;
  const plotBotH = (H - PAD_TOP - PAD_BOT) * 0.55;
  const gapH = (H - PAD_TOP - PAD_BOT) * 0.05;
  const topY0 = PAD_TOP;
  const botY0 = PAD_TOP + plotTopH + gapH;

  const maxY = 0.55; // big enough for theta = -0.5..3

  const scaleZ = (z: number) => PAD_LEFT + ((z - Z_MIN) / (Z_MAX - Z_MIN)) * plotW;
  const scaleX = (x: number) => PAD_LEFT + ((x - X_MIN) / (X_MAX - X_MIN)) * plotW;
  const scaleYTop = (y: number) => topY0 + plotTopH - (y / maxY) * plotTopH;
  const scaleYBot = (y: number) => botY0 + plotBotH - (y / maxY) * plotBotH;

  const baseCurvePath = zPts
    .map((z, i) => `${i === 0 ? 'M' : 'L'} ${scaleZ(z)} ${scaleYTop(basePdf[i])}`)
    .join(' ');
  const pushCurvePath = xPts
    .map((x, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(x)} ${scaleYBot(pushPdf[i])}`)
    .join(' ');

  return (
    <div ref={containerRef} className="my-8 not-prose" style={{ border: '1px solid var(--color-border)', borderRadius: 8, padding: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
        <span style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>θ:</span>
        <input
          type="range"
          min={-0.8}
          max={3}
          step={0.05}
          value={theta}
          onChange={(e) => setTheta(Number(e.target.value))}
          aria-label="Map parameter θ"
          style={{ flex: 1, maxWidth: 360 }}
        />
        <span style={{ fontFamily: 'var(--font-mono, monospace)', fontSize: 12, color: 'var(--color-text-secondary)' }}>
          T_θ(z) = z + {theta.toFixed(2)} · tanh(z)
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: '100%', height: 'auto' }}>
        {/* Top plot: base density */}
        <line x1={PAD_LEFT} y1={scaleYTop(0)} x2={W - PAD_RIGHT} y2={scaleYTop(0)}
              style={{ stroke: 'var(--color-border)', strokeWidth: 1 }} />
        <text x={PAD_LEFT} y={topY0 - 6}
              style={{ fill: 'var(--color-text-secondary)', fontSize: 11, fontFamily: 'var(--font-sans, sans-serif)' }}>
          base density p_Z(z) = N(0, 1)
        </text>
        <path d={baseCurvePath} fill="none" style={{ stroke: 'var(--color-accent)', strokeWidth: 1.8 }} />

        {/* Stretch-factor color band along the z-axis (between the two plots) */}
        {stretchColors
          .filter((_, i) => i % 4 === 0)
          .map((s, k) => {
            const x = scaleZ(s.z);
            // Map rho ∈ [0.5, 2] to a teal-to-amber gradient via interpolation.
            const intensity = Math.min(Math.max((s.rho - 0.4) / 1.6, 0), 1);
            const color = intensity < 0.5
              ? `rgba(15, 110, 86, ${1 - intensity * 2 + 0.1})`
              : `rgba(217, 119, 6, ${(intensity - 0.5) * 2 + 0.1})`;
            return <rect key={k} x={x} y={topY0 + plotTopH + 2} width={6} height={gapH - 4} fill={color} />;
          })}
        <text x={PAD_LEFT} y={topY0 + plotTopH + gapH / 2 + 4}
              style={{ fill: 'var(--color-text-secondary)', fontSize: 9, fontFamily: 'var(--font-sans, sans-serif)' }}>
          1/|T'(z)|
        </text>

        {/* Bottom plot: pushed-forward density */}
        <line x1={PAD_LEFT} y1={scaleYBot(0)} x2={W - PAD_RIGHT} y2={scaleYBot(0)}
              style={{ stroke: 'var(--color-border)', strokeWidth: 1 }} />
        <text x={PAD_LEFT} y={botY0 - 6}
              style={{ fill: 'var(--color-text-secondary)', fontSize: 11, fontFamily: 'var(--font-sans, sans-serif)' }}>
          pushforward p_X(x) = p_Z(T⁻¹(x)) / |T'(T⁻¹(x))|
        </text>
        <path d={pushCurvePath} fill="none" style={{ stroke: 'var(--color-accent)', strokeWidth: 1.8 }} />

        {/* X-axis labels */}
        <text x={PAD_LEFT} y={H - 16}
              style={{ fill: 'var(--color-text-secondary)', fontSize: 10, fontFamily: 'var(--font-mono, monospace)' }}>
          x = {X_MIN}
        </text>
        <text x={W - PAD_RIGHT} y={H - 16} textAnchor="end"
              style={{ fill: 'var(--color-text-secondary)', fontSize: 10, fontFamily: 'var(--font-mono, monospace)' }}>
          x = {X_MAX}
        </text>
      </svg>

      <div style={{ marginTop: 8, fontSize: 12, color: 'var(--color-text-secondary)', fontStyle: 'italic' }}>
        Drag θ to deform T. The pushforward density (bottom) is the base density (top) divided by the local stretch
        factor T'(z); the colored band between them visualizes 1/|T'(z)| — green where T stretches, amber where T
        compresses.
      </div>
    </div>
  );
}
