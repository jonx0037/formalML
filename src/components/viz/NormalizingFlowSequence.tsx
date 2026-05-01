import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// NormalizingFlowSequence — embedded after §5.4's worked-example prose in the
// variational-inference topic. Visualizes the §5.3 normalizing-flow recipe on
// the §5.4 banana target by composing 4 hand-designed invertible coupling
// layers that take standard-Normal samples to a banana-shaped distribution
// matching the §5.4 target log p(z₁, z₂) = −z₁²/8 − ½(z₂ + z₁²/2)².
//
// The §5.4 target's generative process inverts to ε = T⁻¹(z) with
// ε₁ = z₁/2 and ε₂ = z₂ + z₁²/2, so the forward flow is
//   z₁ = 2ε₁,    z₂ = ε₂ − 2ε₁².
// This composite is split into 4 alternating coupling layers:
//   Layer 1: y₁ = √2 · ε₁,  y₂ = ε₂                 (scale dim 1)
//   Layer 2: y₁ = √2 · y₁,  y₂ = y₂                 (scale dim 1 again)
//   Layer 3: y₁ = y₁,       y₂ = y₂ − y₁²/4         (shift dim 2 by quadratic)
//   Layer 4: y₁ = y₁,       y₂ = y₂ − y₁²/4         (shift dim 2 by quadratic again)
//
// After all 4 layers, z₁ = 2ε₁ and z₂ = ε₂ − ε₁²·2 = ε₂ − z₁²/2 — exactly
// the §5.4 banana inverse. Per-layer Jacobian determinants are constants or
// volume-preserving:
//   Layers 1, 2: |det J| = √2  (scale, shrinks volume in dim 1)
//   Layers 3, 4: |det J| = 1   (shift is volume-preserving)
//   Cumulative:  |det J| = 2   (the 2D volume scaling)
//
// Slider for layer index k ∈ {0, 1, 2, 3, 4}: at k=0 samples are standard-
// Normal (circle); at k=4 they trace the banana. Background contour is the
// §5.4 target. The reader watches one of VI's most powerful tricks — flexible
// posterior approximation by stacking simple invertible maps — happen one
// step at a time.
//
// No precompute: hand-designed transformations + closed-form target contour.
// The pedagogical purpose is the LAYER COMPOSITION, not training a real flow.
// =============================================================================

const PANEL_HEIGHT = 460;
const N_SAMPLES = 1500;
const VIEW_DOMAIN_X: [number, number] = [-6, 6];
const VIEW_DOMAIN_Y: [number, number] = [-12, 4];
const N_CONTOUR_GRID = 80;

const COLORS = {
  samples: '#2563eb',
  target: '#dbeafe',
  targetEdge: '#1d4ed8',
  axis: '#374151',
};

// Mulberry32 + Box-Muller — deterministic for a given seed.
function makeNormalSampler(seed: number) {
  let s = seed >>> 0;
  const next = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let cached: number | null = null;
  return () => {
    if (cached !== null) {
      const v = cached;
      cached = null;
      return v;
    }
    let u1 = 0;
    while (u1 === 0) u1 = next();
    const u2 = next();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    cached = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
}

// Banana target log-density from §5.4: −z₁²/8 − ½(z₂ + z₁²/2)² + const.
function targetLogPdf(z1: number, z2: number): number {
  return -0.125 * z1 * z1 - 0.5 * (z2 + 0.5 * z1 * z1) ** 2;
}

// Apply k-th flow layer in place. Layers are 1-indexed in this function.
// Layer parameters chosen so the composition of all 4 produces the §5.4
// banana exactly: z₁ = 2ε₁, z₂ = ε₂ − z₁²/2.
function applyLayer(layer: number, p: [number, number]): [number, number] {
  if (layer === 1) return [Math.SQRT2 * p[0], p[1]];           // scale dim 1
  if (layer === 2) return [Math.SQRT2 * p[0], p[1]];           // scale dim 1 again
  if (layer === 3) return [p[0], p[1] - 0.25 * p[0] * p[0]];   // bend dim 2
  if (layer === 4) return [p[0], p[1] - 0.25 * p[0] * p[0]];   // bend dim 2 again
  return p;
}

// Per-layer log |det J|, evaluated at the input p just before that layer.
// Layers 1,2 (scale): log √2 = 0.5 log 2.
// Layers 3,4 (shift): 0.
function layerLogJacobian(layer: number): number {
  if (layer === 1 || layer === 2) return 0.5 * Math.log(2);
  return 0;
}

const LAYER_LABELS = [
  'Standard normal (k=0)',
  'k=1: y₁ = √2·ε₁',
  'k=2: y₁ = √2·y₁  (cum. ×2)',
  'k=3: y₂ = y₂ − y₁²/4',
  'k=4: y₂ = y₂ − y₁²/4 (cum. −y₁²/2)',
];

export default function NormalizingFlowSequence() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [k, setK] = useState<number>(4);

  // Standard-normal base samples — fixed across slider changes so the
  // morph is a function of layer choice rather than fresh randomness.
  const baseSamples = useMemo<{ e1: Float64Array; e2: Float64Array }>(() => {
    const sample = makeNormalSampler(20260430);
    const e1 = new Float64Array(N_SAMPLES);
    const e2 = new Float64Array(N_SAMPLES);
    for (let i = 0; i < N_SAMPLES; i++) {
      e1[i] = sample();
      e2[i] = sample();
    }
    return { e1, e2 };
  }, []);

  // Apply layers 1..k to each sample.
  const transformed = useMemo<{ x: Float64Array; y: Float64Array }>(() => {
    const x = new Float64Array(N_SAMPLES);
    const y = new Float64Array(N_SAMPLES);
    for (let i = 0; i < N_SAMPLES; i++) {
      let p: [number, number] = [baseSamples.e1[i], baseSamples.e2[i]];
      for (let layer = 1; layer <= k; layer++) p = applyLayer(layer, p);
      x[i] = p[0];
      y[i] = p[1];
    }
    return { x, y };
  }, [baseSamples, k]);

  // Cumulative log |det J| (constant across samples for these particular
  // layer choices).
  const cumLogJac = useMemo<number>(() => {
    let s = 0;
    for (let layer = 1; layer <= k; layer++) s += layerLogJacobian(layer);
    return s;
  }, [k]);

  // Pre-compute target log-density on a 80×80 grid for the contour.
  const contourValues = useMemo<Float64Array>(() => {
    const out = new Float64Array(N_CONTOUR_GRID * N_CONTOUR_GRID);
    const xSpan = VIEW_DOMAIN_X[1] - VIEW_DOMAIN_X[0];
    const ySpan = VIEW_DOMAIN_Y[1] - VIEW_DOMAIN_Y[0];
    for (let i = 0; i < N_CONTOUR_GRID; i++) {
      const z1 = VIEW_DOMAIN_X[0] + (i / (N_CONTOUR_GRID - 1)) * xSpan;
      for (let j = 0; j < N_CONTOUR_GRID; j++) {
        const z2 = VIEW_DOMAIN_Y[0] + (j / (N_CONTOUR_GRID - 1)) * ySpan;
        out[j * N_CONTOUR_GRID + i] = targetLogPdf(z1, z2);
      }
    }
    return out;
  }, []);

  const ref = useD3(
    (svg) => {
      const w = width || 720;
      const h = PANEL_HEIGHT;
      const margin = { top: 24, right: 16, bottom: 44, left: 56 };
      svg.attr('width', w).attr('height', h);
      svg.selectAll('*').remove();

      const innerW = w - margin.left - margin.right;
      const innerH = h - margin.top - margin.bottom;

      const xScale = d3.scaleLinear().domain(VIEW_DOMAIN_X).range([0, innerW]);
      const yScale = d3.scaleLinear().domain(VIEW_DOMAIN_Y).range([innerH, 0]);

      const root = svg
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

      // Banana target contours via d3-contour.
      const contoursGen = d3
        .contours()
        .size([N_CONTOUR_GRID, N_CONTOUR_GRID])
        .thresholds(8);
      // d3-contour expects values in row-major (y outer, x inner) order, which
      // is how we built contourValues. Map grid coords through xScale/yScale.
      const contours = contoursGen(Array.from(contourValues));
      const cellW = innerW / (N_CONTOUR_GRID - 1);
      const cellH = innerH / (N_CONTOUR_GRID - 1);
      const path = d3.geoPath(
        d3.geoIdentity().scale(1).translate([0, 0]),
      );
      // Compose a transform that maps (i, j) grid coords to pixel space.
      root
        .append('g')
        .attr('transform', `translate(0,0) scale(${cellW}, ${cellH})`)
        .selectAll('path')
        .data(contours)
        .join('path')
        .attr('d', path)
        .attr('fill', 'none')
        .attr('stroke', COLORS.targetEdge)
        .attr('stroke-width', 0.6)
        .attr('opacity', 0.45)
        .attr('vector-effect', 'non-scaling-stroke');

      // Fill the highest-density contour with light blue for visual mass.
      if (contours.length > 0) {
        const highest = contours[contours.length - 1];
        root
          .append('g')
          .attr('transform', `translate(0,0) scale(${cellW}, ${cellH})`)
          .append('path')
          .datum(highest)
          .attr('d', path)
          .attr('fill', COLORS.target)
          .attr('opacity', 0.45)
          .attr('vector-effect', 'non-scaling-stroke');
      }

      // Axes.
      root
        .append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(6).tickSizeOuter(0));
      root.append('g').call(d3.axisLeft(yScale).ticks(6).tickSizeOuter(0));
      root
        .append('text')
        .attr('transform', `translate(${innerW / 2},${innerH + 36})`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('y₁');
      root
        .append('text')
        .attr('transform', `translate(-44,${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('y₂');

      // Title showing current layer.
      root
        .append('text')
        .attr('x', innerW / 2)
        .attr('y', -8)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .attr('font-weight', 600)
        .text(LAYER_LABELS[k]);

      // Sample scatter.
      const sampleGroup = root.append('g');
      for (let i = 0; i < N_SAMPLES; i++) {
        const xv = transformed.x[i];
        const yv = transformed.y[i];
        if (xv < VIEW_DOMAIN_X[0] || xv > VIEW_DOMAIN_X[1]) continue;
        if (yv < VIEW_DOMAIN_Y[0] || yv > VIEW_DOMAIN_Y[1]) continue;
        sampleGroup
          .append('circle')
          .attr('cx', xScale(xv))
          .attr('cy', yScale(yv))
          .attr('r', 1.4)
          .attr('fill', COLORS.samples)
          .attr('opacity', 0.32);
      }
    },
    [transformed, contourValues, k, width],
  );

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2">
          flow layer k:
          <input
            type="range"
            min={0}
            max={4}
            step={1}
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            className="w-32"
            aria-label="flow layer slider"
          />
          <span className="tabular-nums w-6 text-right text-xs text-[var(--color-text-muted)]">
            {k}
          </span>
        </label>
      </div>
      <div ref={containerRef} className="w-full">
        <svg ref={ref} />
      </div>
      <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-1 text-xs text-[var(--color-text-muted)]">
        <span>
          <strong>cumulative log |det J_T|:</strong>{' '}
          <span className="tabular-nums">{cumLogJac.toFixed(4)}</span>
        </span>
        <span>
          <strong>cumulative |det J_T|:</strong>{' '}
          <span className="tabular-nums">{Math.exp(cumLogJac).toFixed(4)}</span>{' '}
          (1 = volume-preserving, &gt; 1 = stretched)
        </span>
      </div>
      <div className="mt-2 text-xs text-[var(--color-text-muted)] leading-relaxed">
        Target (light blue): the §5.4 banana posterior, log p(z₁, z₂) =
        −z₁²/8 − ½(z₂ + z₁²/2)². Standard-Normal base samples (k=0) form a
        circular blob; each subsequent slider step composes one more
        invertible coupling layer onto the previous output. Layers 1 and 2
        scale y₁ by √2 each (cumulative ×2), widening the blob. Layers 3
        and 4 shift y₂ by −y₁²/4 each (cumulative −y₁²/2), bending the
        scattered points downward. After all four layers the samples trace
        the banana, matching the target contours exactly. Per-layer log
        |det J| is closed-form: ½ log 2 for the scaling layers,
        0 for the volume-preserving shifts; cumulatively, the flow scales
        2D volume by 2 = exp(log 2). The §5.3 promise — that simple
        invertible maps compose into expressive variational families with
        cheap Jacobians — is exactly what the slider exhibits step by step.
      </div>
    </div>
  );
}
