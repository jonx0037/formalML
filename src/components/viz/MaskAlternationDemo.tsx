// =============================================================================
// MaskAlternationDemo.tsx
//
// §4.4 — Multi-panel pushforward visualization: a slider controls the number
// of layers from 1 to 8, and the panels animate to show the scatter at each
// h_k. A toggle switches between additive and affine coupling, demonstrating
// that additive (volume-preserving) cannot stretch the Gaussian to a
// non-Gaussian shape while affine can.
// =============================================================================

import { useState, useMemo } from 'react';
import { CouplingFlow, sampleStandardNormalBatch } from './shared/normalizing-flows';
import { useResizeObserver } from './shared/useResizeObserver';

const N_POINTS = 800;
const D = 2;

export default function MaskAlternationDemo() {
  const [K, setK] = useState(4);
  const [coupType, setCoupType] = useState<'additive' | 'affine'>('affine');

  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const W = Math.max(420, Math.min(cw || 760, 820));

  const intermediates = useMemo(() => {
    const z = sampleStandardNormalBatch(N_POINTS, D, 11);
    const scaleAmp = coupType === 'additive' ? 0.0001 : 1.2;
    const flow = new CouplingFlow({ d: D, nLayers: K, seed: 4242, scaleAmp, paramScale: 1.0 });
    const states: number[][][] = [];
    states.push(z.map((r) => Array.from(r)));
    let current = z.map((row) => Array.from(row));
    for (let k = 0; k < K; k++) {
      const next: number[][] = [];
      for (const row of current) {
        next.push(Array.from(flow.layers[k].forward(new Float64Array(row)).x));
      }
      current = next;
      states.push(current.map((r) => [...r]));
    }
    return states;
  }, [K, coupType]);

  // Display up to 4 panels evenly spaced through the K+1 intermediates.
  const nPanels = Math.min(intermediates.length, 5);
  const indices = Array.from({ length: nPanels }, (_, i) =>
    Math.round((i * (intermediates.length - 1)) / Math.max(nPanels - 1, 1)),
  );
  const panelGap = 12;
  const panelSize = Math.max(120, Math.min((W - 40 - panelGap * (nPanels - 1)) / nPanels, 170));
  const Hsvg = panelSize + 90;

  const scatterScale = panelSize / 8;
  const xOfPoint = (px: number) => panelSize / 2 + px * scatterScale;
  const yOfPoint = (py: number) => panelSize / 2 - py * scatterScale;

  return (
    <div ref={containerRef} className="my-8 not-prose" style={{ border: '1px solid var(--color-border)', borderRadius: 8, padding: 20 }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginBottom: 16, alignItems: 'center' }}>
        <div>
          <span style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginRight: 8 }}>Depth K:</span>
          <input
            type="range"
            min={1}
            max={8}
            step={1}
            value={K}
            onChange={(e) => setK(Number(e.target.value))}
            aria-label="Depth K"
            style={{ width: 200, verticalAlign: 'middle' }}
          />
          <span style={{ marginLeft: 8, fontFamily: 'var(--font-mono, monospace)', fontSize: 12, color: 'var(--color-text-secondary)' }}>
            K = {K}
          </span>
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
                background: coupType === c ? 'var(--color-accent)' : 'transparent',
                color: coupType === c ? 'white' : 'var(--color-text)',
                cursor: 'pointer',
              }}
            >
              {c === 'additive' ? 'Additive (NICE)' : 'Affine (RealNVP)'}
            </button>
          ))}
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${Hsvg}`} width="100%" style={{ maxWidth: '100%', height: 'auto' }}>
        {indices.map((idx, i) => {
          const ox = 20 + i * (panelSize + panelGap);
          const oy = 40;
          const pts = intermediates[idx];
          return (
            <g key={idx}>
              <rect x={ox} y={oy} width={panelSize} height={panelSize} fill="none"
                    style={{ stroke: 'var(--color-border)', strokeWidth: 1 }} />
              {pts.map((p, k) => (
                <circle key={k} cx={ox + xOfPoint(p[0])} cy={oy + yOfPoint(p[1])} r={1.2}
                        style={{ fill: idx === 0 ? '#6B7280' : 'var(--color-accent)', opacity: 0.6 }} />
              ))}
              <text x={ox + panelSize / 2} y={oy - 8} textAnchor="middle"
                    style={{ fill: 'var(--color-text)', fontSize: 11, fontFamily: 'var(--font-serif, serif)' }}>
                {idx === 0 ? 'h_0 = z' : idx === intermediates.length - 1 ? `h_${idx} = x` : `h_${idx}`}
              </text>
            </g>
          );
        })}
      </svg>

      <div style={{ marginTop: 8, fontSize: 12, color: 'var(--color-text-secondary)', fontStyle: 'italic' }}>
        {coupType === 'additive' ? (
          <>
            Additive coupling is volume-preserving — the pushforward only redistributes mass, never concentrates or
            dilates it. Try affine to see the difference.
          </>
        ) : (
          <>
            Affine coupling with alternating masks lets every dimension be transformed across the stack. By K = 4 the
            base Gaussian is already noticeably warped; deeper stacks fit complex targets like 2-moons (§11).
          </>
        )}
      </div>
    </div>
  );
}
