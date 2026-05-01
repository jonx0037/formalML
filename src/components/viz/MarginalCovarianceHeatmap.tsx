import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { GROUP_SIZES, PALETTE_CLASSROOMS } from './shared/mixed-effects';

// =============================================================================
// MarginalCovarianceHeatmap — embedded after §2.3's ICC discussion in the
// mixed-effects topic. 60×60 heatmap of V = ZGZᵀ + R with the random-intercept
// special case (G = τ²·I_J, R = σ²·I_N), so V is block-diagonal with six
// compound-symmetric blocks of sizes (4, 6, 8, 10, 12, 20).
//
// Slider controls τ²/σ² (the variance ratio that drives ICC = τ²/(τ²+σ²)).
// Three view buttons let the reader DECOMPOSE the covariance:
//   - "Full V"   — V = ZGZᵀ + R, the marginal covariance the reader observes
//   - "ZGZᵀ"     — between-group block structure alone (off-diagonal blocks
//                  vanish; within-block off-diagonals are τ²)
//   - "R"        — within-group residual alone (diagonal of σ², zero elsewhere)
//
// The decomposition is the §2.3 pedagogical payload made interactive. A
// reader who slides τ²/σ² to 0 watches V collapse to σ²·I (only R contributes);
// at τ²/σ² → ∞, V is dominated by the block-of-ones structure (ZGZᵀ
// dominates). At the topic's truth (τ²=25, σ²=64, ratio≈0.39, ICC≈0.281),
// each block has 89 on the diagonal and 25 in the within-block off-diagonals.
// Shared group-sizes from src/components/viz/shared/mixed-effects.ts.
// =============================================================================

const PANEL_HEIGHT = 540;
const N = GROUP_SIZES.reduce<number>((acc, n) => acc + n, 0); // 60

const VIEWS = ['full', 'between', 'within'] as const;
type MatrixView = (typeof VIEWS)[number];
const VIEW_LABEL: Record<MatrixView, string> = {
  full: 'Full V',
  between: 'ZGZᵀ',
  within: 'R',
};

const COLORS = {
  empty: '#f3f4f6',  // gray for true zero cells
  border: '#374151',
  blockBoundary: '#ffffff',
};

export default function MarginalCovarianceHeatmap() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  // Default τ²/σ² = 0.39 (truth: τ²=25, σ²=64). u² mapping covers [0, 1].
  const [u, setU] = useState<number>(Math.sqrt(0.39));
  const [view, setView] = useState<MatrixView>('full');

  // u ∈ [0, 1] → τ²/σ² ∈ [0, 1] via u². Square mapping concentrates
  // resolution near small ratios where the block structure first appears.
  const tauSqOverSigmaSq = useMemo<number>(() => u * u, [u]);
  // Fix σ² = 1 internally and let τ² = ratio. Visual structure is invariant
  // to overall scale; only the ratio matters.
  const sigmaSq = 1;
  const tauSq = tauSqOverSigmaSq * sigmaSq;

  // Classroom membership for each of N=60 observations.
  const classroomMembership = useMemo<Int32Array>(() => {
    const c = new Int32Array(N);
    let idx = 0;
    GROUP_SIZES.forEach((nj, j) => {
      for (let i = 0; i < nj; i++) c[idx++] = j;
    });
    return c;
  }, []);

  // Block boundaries for the cumulative-index grid lines.
  const blockBoundaries = useMemo<number[]>(() => {
    const out: number[] = [];
    let cum = 0;
    GROUP_SIZES.forEach((n, j) => {
      cum += n;
      if (j < GROUP_SIZES.length - 1) out.push(cum);
    });
    return out;
  }, []);

  const icc = useMemo<number>(
    () => tauSq / (tauSq + sigmaSq),
    [tauSq, sigmaSq],
  );

  // Compute V[i][j] under the chosen view.
  const matrixValue = (i: number, j: number): number => {
    const sameClass = classroomMembership[i] === classroomMembership[j];
    const sameObs = i === j;
    let v = 0;
    if (view === 'full' || view === 'between') {
      if (sameClass) v += tauSq;
    }
    if (view === 'full' || view === 'within') {
      if (sameObs) v += sigmaSq;
    }
    return v;
  };

  // Color-scale max under the current view — keeps within-view contrast
  // consistent across slider positions.
  const maxValue = useMemo<number>(() => {
    if (view === 'within') return sigmaSq;
    if (view === 'between') return Math.max(tauSq, 1e-6);
    // full: max is on the diagonal = τ² + σ²
    return tauSq + sigmaSq;
  }, [view, tauSq, sigmaSq]);

  const ref = useD3(
    (svg) => {
      const w = width || 720;
      const h = PANEL_HEIGHT;
      const margin = { top: 24, right: 90, bottom: 28, left: 36 };
      svg.attr('width', w).attr('height', h);
      svg.selectAll('*').remove();

      const innerW = w - margin.left - margin.right;
      const innerH = h - margin.top - margin.bottom;
      // Square heatmap; pick min dimension so it never overflows on mobile.
      const size = Math.max(60, Math.min(innerW, innerH));
      const cellSize = size / N;

      const root = svg
        .append('g')
        .attr(
          'transform',
          `translate(${margin.left + Math.max(0, (innerW - size) / 2)},${margin.top})`,
        );

      const colorScale = d3
        .scaleSequential(d3.interpolateBlues)
        .domain([0, maxValue]);

      // Cell rectangles. Skip true-zero cells (off-block) to reduce DOM size
      // when view is "between" or "within" — only nonzero cells get a rect.
      // For "full", every cell gets a rect to make the gray "empty" off-block
      // grid visible.
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const v = matrixValue(i, j);
          if (v === 0 && view !== 'full') continue;
          root
            .append('rect')
            .attr('x', j * cellSize)
            .attr('y', i * cellSize)
            .attr('width', cellSize + 0.5)  // hairline overlap to remove gaps
            .attr('height', cellSize + 0.5)
            .attr('fill', v > 0 ? colorScale(v) : COLORS.empty);
        }
      }

      // Block boundaries — thin white lines separating classroom blocks.
      blockBoundaries.forEach((boundary) => {
        const pos = boundary * cellSize;
        root
          .append('line')
          .attr('x1', pos)
          .attr('x2', pos)
          .attr('y1', 0)
          .attr('y2', size)
          .attr('stroke', COLORS.blockBoundary)
          .attr('stroke-width', 1.2);
        root
          .append('line')
          .attr('x1', 0)
          .attr('x2', size)
          .attr('y1', pos)
          .attr('y2', pos)
          .attr('stroke', COLORS.blockBoundary)
          .attr('stroke-width', 1.2);
      });

      // Outer border.
      root
        .append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', size)
        .attr('height', size)
        .attr('fill', 'none')
        .attr('stroke', COLORS.border)
        .attr('stroke-width', 1.4);

      // Classroom indices on top axis (block centers).
      let cum = 0;
      GROUP_SIZES.forEach((n, j) => {
        const center = (cum + n / 2) * cellSize;
        cum += n;
        root
          .append('text')
          .attr('x', center)
          .attr('y', -8)
          .attr('text-anchor', 'middle')
          .attr('font-size', 10)
          .attr('fill', PALETTE_CLASSROOMS[j])
          .attr('font-weight', 600)
          .text(`n=${n}`);
      });

      // Color-scale legend on the right.
      const legendX = size + 16;
      const legendW = 14;
      const legendH = Math.min(size, 240);
      const legendY = (size - legendH) / 2;
      const nStops = 50;
      for (let k = 0; k < nStops; k++) {
        const t = k / (nStops - 1);
        root
          .append('rect')
          .attr('x', legendX)
          .attr('y', legendY + (1 - t) * legendH * (1 - 1 / nStops))
          .attr('width', legendW)
          .attr('height', legendH / nStops + 0.5)
          .attr('fill', colorScale(t * maxValue));
      }
      root
        .append('rect')
        .attr('x', legendX)
        .attr('y', legendY)
        .attr('width', legendW)
        .attr('height', legendH)
        .attr('fill', 'none')
        .attr('stroke', COLORS.border)
        .attr('stroke-width', 1);
      root
        .append('text')
        .attr('x', legendX + legendW + 4)
        .attr('y', legendY + 4)
        .attr('font-size', 10)
        .text(maxValue.toFixed(2));
      root
        .append('text')
        .attr('x', legendX + legendW + 4)
        .attr('y', legendY + legendH)
        .attr('font-size', 10)
        .text('0');
    },
    [
      tauSq,
      sigmaSq,
      view,
      maxValue,
      classroomMembership,
      blockBoundaries,
      width,
    ],
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
          <span className="tabular-nums w-16 text-right text-xs text-[var(--color-text-muted)]">
            {tauSqOverSigmaSq.toFixed(3)}
          </span>
        </label>
        <div className="flex items-center gap-1">
          <span className="text-[var(--color-text-muted)]">view:</span>
          {VIEWS.map((v) => (
            <button
              key={v}
              type="button"
              onClick={() => setView(v)}
              className={`rounded border px-2 py-0.5 text-xs ${
                view === v
                  ? 'bg-[var(--color-text-primary)] text-[var(--color-bg)]'
                  : 'bg-white hover:bg-gray-50'
              }`}
              aria-pressed={view === v}
            >
              {VIEW_LABEL[v]}
            </button>
          ))}
        </div>
      </div>
      <div ref={containerRef} className="w-full">
        <svg ref={ref} />
      </div>
      <div className="mt-3 flex flex-wrap items-baseline gap-x-5 gap-y-1 text-xs text-[var(--color-text-muted)]">
        <span>
          <strong>τ²:</strong> <span className="tabular-nums">{tauSq.toFixed(3)}</span>
        </span>
        <span>
          <strong>σ²:</strong> <span className="tabular-nums">{sigmaSq.toFixed(3)}</span>
        </span>
        <span>
          <strong>diagonal of V:</strong>{' '}
          <span className="tabular-nums">{(tauSq + sigmaSq).toFixed(3)}</span>
        </span>
        <span>
          <strong>within-block off-diagonal:</strong>{' '}
          <span className="tabular-nums">{tauSq.toFixed(3)}</span>
        </span>
        <span>
          <strong>ICC = τ²/(τ²+σ²):</strong>{' '}
          <span className="tabular-nums">{icc.toFixed(3)}</span>
        </span>
      </div>
      <div className="mt-2 text-xs text-[var(--color-text-muted)] leading-relaxed">
        Six block-diagonal blocks of sizes (4, 6, 8, 10, 12, 20). Each block
        is compound-symmetric — τ²+σ² on the diagonal, τ² on every within-block
        off-diagonal, zero off-block. Slide τ²/σ² → 0: V collapses to σ²·I_N
        and the block structure disappears (between-classroom contribution
        gone). Slide τ²/σ² → 1: the within-block off-diagonals climb to match
        the diagonal as the random-effects prior dominates the residual.
        Toggle <strong>ZGZᵀ</strong> to see the between-group structure
        alone — six dense blocks of τ² with off-block zeros and no diagonal
        residual. Toggle <strong>R</strong> to see what's left when the
        random effects are stripped — a plain σ²·I_N diagonal with no block
        structure at all.
      </div>
    </div>
  );
}
