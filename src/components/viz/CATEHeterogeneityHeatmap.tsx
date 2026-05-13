import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  drLearnerKernel,
  heterogeneousDGP,
  mulberry32,
  tLearnerKernel,
} from './shared/causal-inference-methods';

// =============================================================================
// CATEHeterogeneityHeatmap — §11 heterogeneous treatment effects.
//
// Three heatmaps over (X_1, X_2) grid (40×40):
//  (A) true τ(x) = 1 + 0.5 x_1 - 0.3 x_2 + 0.4 x_1 x_2
//  (B) T-learner estimate (kernel-smoother proxy)
//  (C) DR-learner estimate (cross-fitted AIPW pseudo-outcomes + kernel smoother)
//
// Rendered via canvas → image data URL for fast grid evaluation.
// =============================================================================

const PANEL_HEIGHT = 280;
const PANEL_WIDTH = 280;
const GRID = 40;
const SM_BREAKPOINT = 920;

type CATEResult = {
  true_: Float64Array;
  t: Float64Array;
  dr: Float64Array;
  minVal: number;
  maxVal: number;
};

function gridEvaluate(nTrain: number): CATEResult {
  const rng = mulberry32(20260512 + 71);
  const sample = heterogeneousDGP(nTrain, 5, rng);
  const { X, D, Y, n, p } = sample;
  const xs: number[] = [];
  for (let i = 0; i < GRID; i++) xs.push(-2 + (4 * i) / (GRID - 1));
  const Xte = new Float64Array(GRID * GRID * p);
  for (let i = 0; i < GRID; i++) {
    for (let j = 0; j < GRID; j++) {
      const idx = (i * GRID + j) * p;
      Xte[idx + 0] = xs[j];
      Xte[idx + 1] = xs[i];
      // Other features at 0.
    }
  }
  const nTe = GRID * GRID;
  const trueX = new Float64Array(nTe);
  for (let i = 0; i < GRID; i++) {
    for (let j = 0; j < GRID; j++) {
      const x0 = xs[j], x1 = xs[i];
      trueX[i * GRID + j] = 1 + 0.5 * x0 - 0.3 * x1 + 0.4 * x0 * x1;
    }
  }
  const tPred = tLearnerKernel(X, D, Y, n, p)(Xte, nTe);
  const drPred = drLearnerKernel(X, D, Y, n, p, mulberry32(20260512 + 73), 3)(Xte, nTe);
  let minV = Infinity, maxV = -Infinity;
  for (const arr of [trueX, tPred, drPred]) {
    for (let i = 0; i < nTe; i++) {
      if (arr[i] < minV) minV = arr[i];
      if (arr[i] > maxV) maxV = arr[i];
    }
  }
  return { true_: trueX, t: tPred, dr: drPred, minVal: minV, maxVal: maxV };
}

function renderHeatmap(values: Float64Array, minV: number, maxV: number): string {
  const canvas = document.createElement('canvas');
  canvas.width = GRID; canvas.height = GRID;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';
  const img = ctx.createImageData(GRID, GRID);
  // Diverging RdBu around 1.
  const color = d3.scaleLinear<string>()
    .domain([minV, (minV + maxV) / 2, maxV])
    .range(['#2c7fb8', '#f7f7f7', '#c0504d'])
    .clamp(true);
  for (let i = 0; i < GRID; i++) {
    for (let j = 0; j < GRID; j++) {
      const v = values[i * GRID + j];
      const c = d3.color(color(v)) as d3.RGBColor;
      const k = (i * GRID + j) * 4;
      img.data[k + 0] = c.r;
      img.data[k + 1] = c.g;
      img.data[k + 2] = c.b;
      img.data[k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL();
}

export default function CATEHeterogeneityHeatmap() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nTrain, setNTrain] = useState<500 | 1000 | 2000>(1000);
  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const result = useMemo(() => gridEvaluate(nTrain), [nTrain]);

  const trueUrl = useMemo(() => typeof document !== 'undefined' ? renderHeatmap(result.true_, result.minVal, result.maxVal) : '', [result]);
  const tUrl = useMemo(() => typeof document !== 'undefined' ? renderHeatmap(result.t, result.minVal, result.maxVal) : '', [result]);
  const drUrl = useMemo(() => typeof document !== 'undefined' ? renderHeatmap(result.dr, result.minVal, result.maxVal) : '', [result]);

  const panels: { label: string; url: string }[] = [
    { label: `True CATE τ(x)`, url: trueUrl },
    { label: `T-learner (kernel smoother)`, url: tUrl },
    { label: `DR-learner (cross-fit AIPW + smoother)`, url: drUrl },
  ];

  return (
    <div ref={containerRef} style={{ marginBlock: '1.25rem' }}>
      <div style={{ marginBottom: '0.75rem', display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.9rem', color: 'var(--color-text)' }}>
          Training n =
          <select value={nTrain} onChange={(e) => setNTrain(parseInt(e.target.value) as 500 | 1000 | 2000)}
            style={{ padding: '0.2rem 0.4rem' }}>
            <option value={500}>500</option>
            <option value={1000}>1000</option>
            <option value={2000}>2000</option>
          </select>
        </label>
        <span style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
          DGP: τ(x) = 1 + 0.5 x<sub>1</sub> − 0.3 x<sub>2</sub> + 0.4 x<sub>1</sub> x<sub>2</sub>
        </span>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '1rem', justifyContent: 'space-between', flexWrap: 'wrap' }}>
        {panels.map((p) => (
          <div key={p.label} style={{ flex: 1, minWidth: 240 }}>
            <div style={{ fontSize: '0.85rem', color: 'var(--color-text)', textAlign: 'center', marginBottom: '0.25rem' }}>{p.label}</div>
            <img src={p.url} alt={p.label}
                 style={{ width: '100%', maxWidth: PANEL_WIDTH, height: PANEL_HEIGHT, imageRendering: 'pixelated', border: '1px solid var(--color-border)', display: 'block', margin: '0 auto' }} />
          </div>
        ))}
      </div>
      <div style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem', textAlign: 'center' }}>
        x<sub>1</sub> on horizontal axis, x<sub>2</sub> on vertical, both in [−2, 2]. Color scale: blue ↔ red, centered at the grid midpoint.
      </div>
    </div>
  );
}
