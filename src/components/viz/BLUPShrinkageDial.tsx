import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  blupShrinkage,
  generateSixClassrooms,
  PALETTE_CLASSROOMS,
  partialPooledIntercept,
} from './shared/mixed-effects';

// =============================================================================
// BLUPShrinkageDial — embedded after Proposition 2 in §3.3 of the
// mixed-effects topic. Twin-panel viz that makes the BLUP shrinkage formula
// λⱼ = τ²·nⱼ / (τ²·nⱼ + σ²) tactile by showing it from two angles
// simultaneously:
//
//   Left panel  — λ as a function of group size n at the current τ²/σ², with
//                 the six classroom sizes (4, 6, 8, 10, 12, 20) marked along
//                 the curve. Dragging the slider warps the curve and the six
//                 markers slide along it; the size-dependent shape of
//                 shrinkage becomes visible.
//   Right panel — for each classroom j a horizontal "shrinkage line" from
//                 the complete-pool intercept α̂ on the left to the no-pool
//                 intercept α̂ⱼ^(no-pool) on the right, with the
//                 partial-pool intercept α̂ⱼ + λⱼ·r̄ⱼ marked as a filled
//                 colored circle. The position of that circle along the line
//                 is exactly λⱼ — the abstract shrinkage factor becomes a
//                 spatial coordinate.
//
// Same six-classroom DGP and slider semantics as A1's PoolingSpectrumExplorer
// (shared in src/components/viz/shared/mixed-effects.ts), so the reader who
// scrolled past the §1.2 viz sees the same six numbers reappear in §3.3 with
// the formula now named.
// =============================================================================

const PANEL_HEIGHT = 380;
const PANEL_GAP = 36;
const N_GRID = 200;
const N_DOMAIN: [number, number] = [1, 30];

export default function BLUPShrinkageDial() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [u, setU] = useState<number>(Math.sqrt(0.07 / 5));

  const data = useMemo(() => generateSixClassrooms(), []);

  const tauSqOverSigmaSq = useMemo<number>(() => 5 * u * u, [u]);

  const lambdas = useMemo<Float64Array>(() => {
    const J = data.classrooms.length;
    const out = new Float64Array(J);
    for (let j = 0; j < J; j++) {
      out[j] = blupShrinkage(data.classrooms[j].nj, tauSqOverSigmaSq);
    }
    return out;
  }, [data, tauSqOverSigmaSq]);

  const alphasPartial = useMemo<Float64Array>(() => {
    const J = data.classrooms.length;
    const out = new Float64Array(J);
    for (let j = 0; j < J; j++) {
      out[j] = partialPooledIntercept(
        data.alphasNoPool[j],
        data.alphaPool,
        lambdas[j],
      );
    }
    return out;
  }, [data, lambdas]);

  // Continuous λ-curve over n ∈ [1, 30] for the left panel.
  const lambdaCurve = useMemo<Float64Array>(() => {
    const c = new Float64Array(N_GRID);
    const span = N_DOMAIN[1] - N_DOMAIN[0];
    for (let i = 0; i < N_GRID; i++) {
      const n = N_DOMAIN[0] + (i / (N_GRID - 1)) * span;
      c[i] = blupShrinkage(n, tauSqOverSigmaSq);
    }
    return c;
  }, [tauSqOverSigmaSq]);

  // Right-panel x-axis: span the complete-pool intercept and all no-pool
  // intercepts, with a small pad so endpoints aren't right at the edges.
  const alphaDomain = useMemo<[number, number]>(() => {
    let lo = data.alphaPool;
    let hi = data.alphaPool;
    for (let j = 0; j < data.alphasNoPool.length; j++) {
      const v = data.alphasNoPool[j];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const pad = (hi - lo) * 0.08;
    return [lo - pad, hi + pad];
  }, [data]);

  const ref = useD3(
    (svg) => {
      const w = width || 720;
      const h = PANEL_HEIGHT;
      const margin = { top: 24, right: 18, bottom: 48, left: 52 };
      svg.attr('width', w).attr('height', h);
      svg.selectAll('*').remove();

      const innerW = w - margin.left - margin.right;
      const innerH = h - margin.top - margin.bottom;
      const panelW = (innerW - PANEL_GAP) / 2;

      const root = svg
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

      // ── Left panel: λ vs n ────────────────────────────────────────────
      const leftG = root.append('g');
      const nScale = d3.scaleLinear().domain(N_DOMAIN).range([0, panelW]);
      const lScale = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

      leftG
        .append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(nScale).ticks(6).tickSizeOuter(0));
      leftG.append('g').call(d3.axisLeft(lScale).ticks(5).tickSizeOuter(0));

      leftG
        .append('text')
        .attr('transform', `translate(${panelW / 2},${innerH + 34})`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('group size nⱼ');

      leftG
        .append('text')
        .attr('transform', `translate(-40,${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('shrinkage λⱼ');

      leftG
        .append('text')
        .attr('x', panelW / 2)
        .attr('y', -8)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .attr('font-weight', 600)
        .text('λⱼ = τ²·nⱼ / (τ²·nⱼ + σ²)');

      // λ = 0.5 reference line.
      leftG
        .append('line')
        .attr('x1', 0)
        .attr('x2', panelW)
        .attr('y1', lScale(0.5))
        .attr('y2', lScale(0.5))
        .attr('stroke', '#9ca3af')
        .attr('stroke-dasharray', '2 3')
        .attr('stroke-width', 0.8);

      const lambdaLine = d3
        .line<number>()
        .x((_, i) => nScale(N_DOMAIN[0] + (i / (N_GRID - 1)) * (N_DOMAIN[1] - N_DOMAIN[0])))
        .y((d) => lScale(d))
        .curve(d3.curveMonotoneX);

      leftG
        .append('path')
        .datum(Array.from(lambdaCurve))
        .attr('fill', 'none')
        .attr('stroke', '#1f2937')
        .attr('stroke-width', 2)
        .attr('d', lambdaLine);

      // Six classroom markers on the curve.
      data.classrooms.forEach((c, j) => {
        leftG
          .append('circle')
          .attr('cx', nScale(c.nj))
          .attr('cy', lScale(lambdas[j]))
          .attr('r', 5)
          .attr('fill', PALETTE_CLASSROOMS[j])
          .attr('stroke', '#fff')
          .attr('stroke-width', 1.5);
        leftG
          .append('text')
          .attr('x', nScale(c.nj))
          .attr('y', lScale(lambdas[j]) - 10)
          .attr('text-anchor', 'middle')
          .attr('font-size', 10)
          .attr('fill', PALETTE_CLASSROOMS[j])
          .attr('font-weight', 600)
          .text(`n=${c.nj}`);
      });

      // ── Right panel: shrinkage thermometers ───────────────────────────
      const rightG = root
        .append('g')
        .attr('transform', `translate(${panelW + PANEL_GAP},0)`);
      const aScale = d3.scaleLinear().domain(alphaDomain).range([0, panelW]);

      rightG
        .append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(aScale).ticks(5).tickSizeOuter(0));

      rightG
        .append('text')
        .attr('transform', `translate(${panelW / 2},${innerH + 34})`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('intercept α̂ⱼ');

      rightG
        .append('text')
        .attr('x', panelW / 2)
        .attr('y', -8)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .attr('font-weight', 600)
        .text('α̂ⱼ = λⱼ·α̂ⱼ^(no-pool) + (1 − λⱼ)·α̂');

      // Vertical guide at the complete-pool intercept.
      rightG
        .append('line')
        .attr('x1', aScale(data.alphaPool))
        .attr('x2', aScale(data.alphaPool))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', '#374151')
        .attr('stroke-dasharray', '3 3')
        .attr('stroke-width', 1)
        .attr('opacity', 0.7);
      rightG
        .append('text')
        .attr('x', aScale(data.alphaPool))
        .attr('y', -2)
        .attr('text-anchor', 'middle')
        .attr('font-size', 10)
        .attr('fill', '#374151')
        .text(`α̂ = ${data.alphaPool.toFixed(2)}`);

      // One row per classroom — sorted small-n on top so the eye reads
      // "small classrooms shrink hard" from top to bottom.
      const rowH = innerH / data.classrooms.length;
      data.classrooms.forEach((c, j) => {
        const rowY = (j + 0.5) * rowH;
        const aPool = data.alphaPool;
        const aNo = data.alphasNoPool[j];
        const aPart = alphasPartial[j];
        const color = PALETTE_CLASSROOMS[j];

        // Connecting line between the two endpoints.
        rightG
          .append('line')
          .attr('x1', aScale(aPool))
          .attr('x2', aScale(aNo))
          .attr('y1', rowY)
          .attr('y2', rowY)
          .attr('stroke', color)
          .attr('stroke-width', 1)
          .attr('opacity', 0.45);

        // Open circle at α̂_pool endpoint.
        rightG
          .append('circle')
          .attr('cx', aScale(aPool))
          .attr('cy', rowY)
          .attr('r', 3.5)
          .attr('fill', '#fff')
          .attr('stroke', color)
          .attr('stroke-width', 1.4);

        // Open circle at α̂_j^(no-pool) endpoint.
        rightG
          .append('circle')
          .attr('cx', aScale(aNo))
          .attr('cy', rowY)
          .attr('r', 3.5)
          .attr('fill', '#fff')
          .attr('stroke', color)
          .attr('stroke-width', 1.4);

        // Filled colored circle at α̂_j_partial — position = λⱼ along the line.
        rightG
          .append('circle')
          .attr('cx', aScale(aPart))
          .attr('cy', rowY)
          .attr('r', 5.5)
          .attr('fill', color)
          .attr('stroke', '#fff')
          .attr('stroke-width', 1.5);

        // Per-row label showing classroom info.
        rightG
          .append('text')
          .attr('x', panelW)
          .attr('y', rowY - 7)
          .attr('text-anchor', 'end')
          .attr('font-size', 10)
          .attr('fill', color)
          .attr('font-weight', 600)
          .text(`n = ${c.nj}, λ = ${lambdas[j].toFixed(2)}`);
      });
    },
    [data, lambdas, alphasPartial, lambdaCurve, alphaDomain, width],
  );

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2">
          variance ratio τ²/σ²:
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={u}
            onChange={(e) => setU(Number(e.target.value))}
            className="w-40"
            aria-label="variance ratio slider"
          />
          <span className="tabular-nums w-20 text-right text-xs text-[var(--color-text-muted)]">
            {tauSqOverSigmaSq.toFixed(3)}
          </span>
        </label>
      </div>
      <div ref={containerRef} className="w-full">
        <svg ref={ref} />
      </div>
      <div className="mt-3 text-xs text-[var(--color-text-muted)] leading-relaxed">
        <strong>Left:</strong> λⱼ as a function of group size n at the current
        variance ratio. The curve is concave-saturating in n — small groups
        sit on the steep ascent (sharp marginal returns to data), large groups
        sit on the plateau (diminishing returns). The six dots are the actual
        classroom sizes; their λⱼ values are the y-coordinates.{' '}
        <strong>Right:</strong> for each classroom, an open dot on the dashed
        vertical at the complete-pool intercept α̂, an open dot at the
        no-pool intercept α̂ⱼ<sup>(no-pool)</sup>, and a filled circle at the
        partial-pool intercept α̂ⱼ. The filled circle's position along the
        row <em>is</em> λⱼ — left endpoint is λ = 0 (complete pooling), right
        endpoint is λ = 1 (no pooling). Drag the slider and the whole picture
        breathes: the curve on the left changes shape and the dots move; the
        partial-pool circles on the right slide along their rows; small
        classrooms always travel a larger fraction of the way back to α̂.
      </div>
    </div>
  );
}
