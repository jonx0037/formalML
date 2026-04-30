import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  blupShrinkage,
  generateSixClassrooms,
  PALETTE_CLASSROOMS,
  partialPooledIntercept,
  X_DOMAIN,
} from './shared/mixed-effects';

// =============================================================================
// PoolingSpectrumExplorer — embedded after §1.2 of the mixed-effects topic.
// One panel: scatter of N=60 student-level (prep-hours, exam-score) pairs
// across J=6 classrooms with sizes (4, 6, 8, 10, 12, 20), color-coded.
// Six classroom lines share a slope and use intercepts that morph between
// no-pooling (per-classroom) and complete-pooling (overall mean) under the
// random-intercept BLUP shrinkage rule
//   λⱼ = τ²·nⱼ / (τ²·nⱼ + σ²),
// with σ² fixed at the within-group residual variance and τ² controlled by a
// slider. At τ² = 0 the six lines collapse onto the population-mean line
// (complete pooling); at τ² → ∞ each line returns to its OLS fit (no pooling).
// The default τ²/σ² ≈ 0.07 reproduces the REML shrinkage factors (~0.22 for
// n=4, ~0.59 for n=20) the topic reports at line 249. The size-dependent
// drift — small classrooms collapsing hard, large classrooms barely moving —
// is the §1.2 pedagogical payoff before §3 names the BLUP formula.
//
// DGP and shrinkage helpers are imported from shared/mixed-effects.ts so the
// §3.3 BLUPShrinkageDial (A2) sees the same six classrooms.
// =============================================================================

const PANEL_HEIGHT = 380;

export default function PoolingSpectrumExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  // τ²/σ² slider — square-mapped from u ∈ [0, 1] for fine resolution near the
  // REML estimate.
  const [u, setU] = useState<number>(Math.sqrt(0.07 / 5));
  const [showPopulationLine, setShowPopulationLine] = useState<boolean>(true);

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

  const ref = useD3(
    (svg) => {
      const w = width || 720;
      const h = PANEL_HEIGHT;
      const margin = { top: 20, right: 20, bottom: 48, left: 52 };
      svg.attr('width', w).attr('height', h);
      svg.selectAll('*').remove();

      const innerW = w - margin.left - margin.right;
      const innerH = h - margin.top - margin.bottom;

      let yMin = Infinity;
      let yMax = -Infinity;
      for (const c of data.classrooms) {
        for (let i = 0; i < c.nj; i++) {
          if (c.y[i] < yMin) yMin = c.y[i];
          if (c.y[i] > yMax) yMax = c.y[i];
        }
      }
      const yPad = (yMax - yMin) * 0.06;
      const yDomain: [number, number] = [yMin - yPad, yMax + yPad];

      const xScale = d3.scaleLinear().domain(X_DOMAIN).range([0, innerW]);
      const yScale = d3.scaleLinear().domain(yDomain).range([innerH, 0]);

      const root = svg
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

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
        .text('prep-hours index');

      root
        .append('text')
        .attr('transform', `translate(-40,${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('exam score');

      if (showPopulationLine) {
        const x0 = X_DOMAIN[0];
        const x1 = X_DOMAIN[1];
        root
          .append('line')
          .attr('x1', xScale(x0))
          .attr('x2', xScale(x1))
          .attr('y1', yScale(data.alphaPool + data.betaWithin * x0))
          .attr('y2', yScale(data.alphaPool + data.betaWithin * x1))
          .attr('stroke', '#374151')
          .attr('stroke-dasharray', '3 4')
          .attr('stroke-width', 1.2)
          .attr('opacity', 0.85);
      }

      for (let j = 0; j < data.classrooms.length; j++) {
        const a = alphasPartial[j];
        const b = data.betaWithin;
        const x0 = X_DOMAIN[0];
        const x1 = X_DOMAIN[1];
        root
          .append('line')
          .attr('x1', xScale(x0))
          .attr('x2', xScale(x1))
          .attr('y1', yScale(a + b * x0))
          .attr('y2', yScale(a + b * x1))
          .attr('stroke', PALETTE_CLASSROOMS[j])
          .attr('stroke-width', 2.2)
          .attr('opacity', 0.92);
      }

      for (let j = 0; j < data.classrooms.length; j++) {
        const c = data.classrooms[j];
        const grp = root.append('g');
        for (let i = 0; i < c.nj; i++) {
          grp
            .append('circle')
            .attr('cx', xScale(c.x[i]))
            .attr('cy', yScale(c.y[i]))
            .attr('r', 3.2)
            .attr('fill', PALETTE_CLASSROOMS[j])
            .attr('opacity', 0.7);
        }
      }

      const legend = root.append('g').attr('transform', `translate(${innerW - 200},10)`);
      legend
        .append('text')
        .attr('x', 0)
        .attr('y', 0)
        .attr('font-size', 10)
        .attr('font-weight', 600)
        .text('Classroom (nⱼ, λⱼ)');
      data.classrooms.forEach((c, j) => {
        const row = legend.append('g').attr('transform', `translate(0,${10 + (j + 1) * 13})`);
        row
          .append('line')
          .attr('x1', 0)
          .attr('x2', 18)
          .attr('y1', 4)
          .attr('y2', 4)
          .attr('stroke', PALETTE_CLASSROOMS[j])
          .attr('stroke-width', 2.2);
        row
          .append('text')
          .attr('x', 24)
          .attr('y', 7)
          .attr('font-size', 10)
          .text(`n = ${c.nj}, λ = ${lambdas[j].toFixed(2)}`);
      });
    },
    [data, alphasPartial, lambdas, showPopulationLine, width],
  );

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2">
          pooling strength:
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={u}
            onChange={(e) => setU(Number(e.target.value))}
            className="w-40"
            aria-label="pooling strength slider"
          />
          <span className="tabular-nums w-24 text-right text-xs text-[var(--color-text-muted)]">
            τ²/σ² = {tauSqOverSigmaSq.toFixed(3)}
          </span>
        </label>
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={showPopulationLine}
            onChange={(e) => setShowPopulationLine(e.target.checked)}
          />
          <span>Show population-mean line</span>
        </label>
      </div>
      <div ref={containerRef} className="w-full">
        <svg ref={ref} />
      </div>
      <div className="mt-3 text-xs text-[var(--color-text-muted)] leading-relaxed">
        Six classrooms, sizes (4, 6, 8, 10, 12, 20). Slide pooling strength
        toward 0 — the classroom-level lines collapse onto the dashed
        population-mean line (complete pooling). Slide it toward 1 — each line
        returns to the OLS fit through its own data (no pooling). Anywhere in
        between, the smallest classroom (n = 4) shrinks toward the dashed line
        much faster than the largest (n = 20) because λⱼ depends on group
        size. The default position reproduces the topic's REML shrinkage
        factors (~0.22 for n = 4, rising to ~0.59 for n = 20).
      </div>
    </div>
  );
}
