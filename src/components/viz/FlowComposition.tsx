// =============================================================================
// FlowComposition.tsx
//
// §3.2 — Interactive visualization of a K-layer flow as a sequence of
// pushforwards. The reader sees K+1 scatter panels arranged horizontally,
// each showing the data at intermediate state h_k = T_k(h_{k-1}). A slider
// varies K from 1 to 6. A side panel displays the running log-det sum.
//
// Uses the shared CouplingFlow module — illustrates the additivity of
// log-det across composition (eq. 3.4).
// =============================================================================

import { useState, useMemo } from 'react';
import {
  CouplingFlow,
  sampleStandardNormalBatch,
} from './shared/normalizing-flows';
import { useResizeObserver } from './shared/useResizeObserver';

const N_POINTS = 600;
const D = 2;

export default function FlowComposition() {
  const [K, setK] = useState(3);
  const { ref: containerRef, width: cw } = useResizeObserver<HTMLDivElement>();
  const W = Math.max(420, Math.min(cw || 760, 820));

  const { intermediates, logDets, totalLogDet } = useMemo(() => {
    const z = sampleStandardNormalBatch(N_POINTS, D, 7);
    const flow = new CouplingFlow({ d: D, nLayers: K, seed: 2025, scaleAmp: 1.2, paramScale: 0.85 });
    // Apply layer by layer to get intermediate states and the per-layer
    // *batch-mean* log-det E_z[log|det dT_k/dh_{k-1}|]. We report the mean
    // rather than a single-z evaluation so the readout matches the typical
    // density-estimation training objective (which integrates against the
    // empirical distribution).
    const states: number[][][] = [];
    const layerMeanLogDets: number[] = [];
    let current = z.map((row) => Array.from(row));
    states.push(current.map((r) => [...r]));
    for (let k = 0; k < K; k++) {
      let cumLogDet = 0;
      const nextState: number[][] = [];
      for (const row of current) {
        const r = flow.layers[k].forward(new Float64Array(row));
        nextState.push(Array.from(r.x));
        cumLogDet += r.logDet;
      }
      layerMeanLogDets.push(cumLogDet / current.length);
      current = nextState;
      states.push(current.map((r) => [...r]));
    }
    const totalLogDet = layerMeanLogDets.reduce((a, b) => a + b, 0);
    return { intermediates: states, logDets: layerMeanLogDets, totalLogDet };
  }, [K]);

  const numPanels = K + 1;
  const panelGap = 12;
  const panelSize = Math.max(80, Math.min((W - 40 - panelGap * (numPanels - 1)) / numPanels, 140));
  const Hsvg = panelSize + 90;

  const scatterScale = panelSize / 7; // map [-3.5, 3.5] → [0, panelSize]
  const panelOriginX = (i: number) => 20 + i * (panelSize + panelGap);
  const panelOriginY = 40;

  const xOfPoint = (px: number) => panelSize / 2 + px * scatterScale;
  const yOfPoint = (py: number) => panelSize / 2 - py * scatterScale;

  return (
    <div ref={containerRef} className="my-8 not-prose" style={{ border: '1px solid var(--color-border)', borderRadius: 8, padding: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
        <span style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>K:</span>
        <input
          type="range"
          min={1}
          max={6}
          step={1}
          value={K}
          onChange={(e) => setK(Number(e.target.value))}
          aria-label="Number of layers K"
          style={{ flex: 1, maxWidth: 280 }}
        />
        <span style={{ fontFamily: 'var(--font-mono, monospace)', fontSize: 12, color: 'var(--color-text-secondary)' }}>
          K = {K} layers
        </span>
        <span style={{ flex: 1 }} />
        <span style={{ fontFamily: 'var(--font-mono, monospace)', fontSize: 12, color: 'var(--color-accent)', fontWeight: 600 }}>
          E[Σ log|det dT_k|] = {totalLogDet.toFixed(3)}
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${Hsvg}`} width="100%" style={{ maxWidth: '100%', height: 'auto' }}>
        {intermediates.map((pts, i) => {
          const ox = panelOriginX(i);
          const oy = panelOriginY;
          const labelY = oy + panelSize + 18;
          const cumLD = logDets.slice(0, i).reduce((a, b) => a + b, 0);
          return (
            <g key={i}>
              <rect x={ox} y={oy} width={panelSize} height={panelSize} fill="none"
                    style={{ stroke: 'var(--color-border)', strokeWidth: 1 }} />
              {pts.map((p, k) => (
                <circle key={k} cx={ox + xOfPoint(p[0])} cy={oy + yOfPoint(p[1])} r={1.2}
                        style={{ fill: 'var(--color-accent)', opacity: 0.55 }} />
              ))}
              <text x={ox + panelSize / 2} y={oy - 6} textAnchor="middle"
                    style={{ fill: 'var(--color-text)', fontSize: 10, fontFamily: 'var(--font-serif, serif)' }}>
                h_{i}
              </text>
              <text x={ox + panelSize / 2} y={labelY} textAnchor="middle"
                    style={{ fill: 'var(--color-text-secondary)', fontSize: 9, fontFamily: 'var(--font-mono, monospace)' }}>
                Σ ld = {cumLD.toFixed(2)}
              </text>
            </g>
          );
        })}
      </svg>

      <div style={{ marginTop: 8, fontSize: 12, color: 'var(--color-text-secondary)', fontStyle: 'italic' }}>
        Each panel shows the data after another coupling layer. The cumulative log-det across all K layers
        (top-right readout) is the sum of the per-layer log-dets shown beneath each panel — equation (3.4).
        Composition is additive in log-det; depth doesn't introduce any cross-layer determinant computation.
      </div>
    </div>
  );
}
