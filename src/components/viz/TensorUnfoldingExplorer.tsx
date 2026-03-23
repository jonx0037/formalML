import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

// ─── Data ───

const TENSOR_DATA: number[][][] = [
  [[1, 13], [5, 17], [9, 21]],
  [[2, 14], [6, 18], [10, 22]],
  [[3, 15], [7, 19], [11, 23]],
  [[4, 16], [8, 20], [12, 24]],
];
// Shape: [I1=4][I2=3][I3=2], so TENSOR_DATA[i1][i2][i3]

const I1 = 4, I2 = 3, I3 = 2;

// ─── Layout ───

const SM_BREAKPOINT = 640;
const MARGIN = { top: 28, right: 8, bottom: 8, left: 36 };
const SLICE_GAP = 28;

// ─── Unfolding logic ───

function modeNUnfold(data: number[][][], mode: number): number[][] {
  const dims = [I1, I2, I3];
  const modeSize = dims[mode];
  const otherDims = dims.filter((_, i) => i !== mode);
  const cols = otherDims.reduce((a, b) => a * b, 1);
  const result: number[][] = Array.from({ length: modeSize }, () => []);

  // Column ordering: cycle through remaining modes in order
  for (let row = 0; row < modeSize; row++) {
    for (let col = 0; col < cols; col++) {
      const idx = [0, 0, 0];
      idx[mode] = row;
      // Map col to indices of the other dimensions
      const otherIndices: number[] = [];
      let rem = col;
      for (let d = otherDims.length - 1; d >= 0; d--) {
        otherIndices[d] = rem % otherDims[d];
        rem = Math.floor(rem / otherDims[d]);
      }
      let oi = 0;
      for (let d = 0; d < 3; d++) {
        if (d !== mode) {
          idx[d] = otherIndices[oi++];
        }
      }
      result[row].push(data[idx[0]][idx[1]][idx[2]]);
    }
  }
  return result;
}

// ─── Color scale ───

function colorScale(v: number) {
  // Reversed RdBu so low=blue, high=red
  const t = (v - 1) / 23; // normalize 1–24 to 0–1
  return d3.interpolateRdBu(1 - t);
}

function textColor(v: number): string {
  const t = (v - 1) / 23;
  return t > 0.35 && t < 0.65 ? '#111' : '#fff';
}

// ─── Component ───

export default function TensorUnfoldingExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const tensorRef = useRef<SVGSVGElement>(null);
  const unfoldRef = useRef<SVGSVGElement>(null);

  const [mode, setMode] = useState(0);
  const [hoveredCell, setHoveredCell] = useState<[number, number, number] | null>(null);

  const unfoldedMatrix = useMemo(() => modeNUnfold(TENSOR_DATA, mode), [mode]);
  const unfoldRows = unfoldedMatrix.length;
  const unfoldCols = unfoldedMatrix[0].length;

  // ─── Sizing ───

  const isWide = containerWidth >= SM_BREAKPOINT;
  const cellSize = useMemo(() => {
    if (!containerWidth) return 28;
    const available = isWide ? containerWidth * 0.45 : containerWidth - 16;
    const maxDim = Math.max(unfoldCols, I2);
    return Math.max(16, Math.min(Math.floor((available - MARGIN.left - MARGIN.right) / maxDim), 36));
  }, [containerWidth, isWide, unfoldCols]);

  // Tensor view dimensions
  const tensorW = MARGIN.left + I2 * cellSize + SLICE_GAP + I2 * cellSize + MARGIN.right;
  const tensorH = MARGIN.top + I1 * cellSize + MARGIN.bottom;

  // Unfolding view dimensions
  const unfoldW = MARGIN.left + unfoldCols * cellSize + MARGIN.right;
  const unfoldH = MARGIN.top + unfoldRows * cellSize + MARGIN.bottom;

  // ─── Highlight logic ───

  const isCellHighlighted = useCallback(
    (i1: number, i2: number, i3: number) => {
      if (!hoveredCell) return false;
      const [h1, h2, h3] = hoveredCell;
      if (mode === 0) return i1 === h1;
      if (mode === 1) return i2 === h2;
      return i3 === h3;
    },
    [hoveredCell, mode],
  );

  // ─── Tensor 3D view ───

  useEffect(() => {
    if (!tensorRef.current) return;
    const svg = d3.select(tensorRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // Render two frontal slices side by side
    for (let k = 0; k < I3; k++) {
      const offsetX = k * (I2 * cellSize + SLICE_GAP);
      const sliceG = g.append('g').attr('transform', `translate(${offsetX},0)`);

      // Slice label
      sliceG.append('text')
        .attr('x', (I2 * cellSize) / 2)
        .attr('y', -8)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-muted)')
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 10)
        .text(`k = ${k + 1}`);

      for (let i = 0; i < I1; i++) {
        for (let j = 0; j < I2; j++) {
          const v = TENSOR_DATA[i][j][k];
          const highlighted = isCellHighlighted(i, j, k);
          const dimmed = hoveredCell && !highlighted;

          sliceG.append('rect')
            .attr('x', j * cellSize)
            .attr('y', i * cellSize)
            .attr('width', cellSize - 1)
            .attr('height', cellSize - 1)
            .attr('rx', 2)
            .attr('fill', colorScale(v))
            .attr('opacity', dimmed ? 0.25 : 1)
            .attr('stroke', highlighted ? 'var(--color-accent)' : 'none')
            .attr('stroke-width', highlighted ? 2.5 : 0)
            .style('cursor', 'pointer')
            .on('mouseenter', () => setHoveredCell([i, j, k]))
            .on('mouseleave', () => setHoveredCell(null));

          sliceG.append('text')
            .attr('x', j * cellSize + (cellSize - 1) / 2)
            .attr('y', i * cellSize + (cellSize - 1) / 2)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .style('fill', textColor(v))
            .style('font-family', 'var(--font-sans)')
            .attr('font-size', Math.max(9, cellSize * 0.35))
            .attr('font-weight', 600)
            .attr('opacity', dimmed ? 0.3 : 1)
            .attr('pointer-events', 'none')
            .text(v);
        }
      }
    }

    // Row labels (i1)
    for (let i = 0; i < I1; i++) {
      g.append('text')
        .attr('x', -6)
        .attr('y', i * cellSize + (cellSize - 1) / 2)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'central')
        .style('fill', 'var(--color-muted)')
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 9)
        .text(`i₁=${i + 1}`);
    }
  }, [cellSize, hoveredCell, isCellHighlighted, mode]);

  // ─── Unfolding heatmap ───

  useEffect(() => {
    if (!unfoldRef.current) return;
    const svg = d3.select(unfoldRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // Title
    const modeLabels = ['Mode-1 (4×6)', 'Mode-2 (3×8)', 'Mode-3 (2×12)'];
    g.append('text')
      .attr('x', (unfoldCols * cellSize) / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .text(modeLabels[mode]);

    // Map unfolded (row, col) back to tensor indices
    const unfoldToTensor = (row: number, col: number): [number, number, number] => {
      const dims = [I1, I2, I3];
      const otherDims = dims.filter((_, i) => i !== mode);
      const idx: [number, number, number] = [0, 0, 0];
      idx[mode] = row;
      const otherIndices: number[] = [];
      let rem = col;
      for (let d = otherDims.length - 1; d >= 0; d--) {
        otherIndices[d] = rem % otherDims[d];
        rem = Math.floor(rem / otherDims[d]);
      }
      let oi = 0;
      for (let d = 0; d < 3; d++) {
        if (d !== mode) idx[d] = otherIndices[oi++];
      }
      return idx;
    };

    for (let r = 0; r < unfoldRows; r++) {
      for (let c = 0; c < unfoldCols; c++) {
        const v = unfoldedMatrix[r][c];
        const [ti, tj, tk] = unfoldToTensor(r, c);
        const highlighted = isCellHighlighted(ti, tj, tk);
        const dimmed = hoveredCell && !highlighted;

        g.append('rect')
          .attr('x', c * cellSize)
          .attr('y', r * cellSize)
          .attr('width', cellSize - 1)
          .attr('height', cellSize - 1)
          .attr('rx', 2)
          .attr('fill', colorScale(v))
          .attr('opacity', dimmed ? 0.25 : 1)
          .attr('stroke', highlighted ? 'var(--color-accent)' : 'none')
          .attr('stroke-width', highlighted ? 2.5 : 0)
          .style('cursor', 'pointer')
          .on('mouseenter', () => setHoveredCell([ti, tj, tk]))
          .on('mouseleave', () => setHoveredCell(null));

        g.append('text')
          .attr('x', c * cellSize + (cellSize - 1) / 2)
          .attr('y', r * cellSize + (cellSize - 1) / 2)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'central')
          .style('fill', textColor(v))
          .style('font-family', 'var(--font-sans)')
          .attr('font-size', Math.max(9, cellSize * 0.35))
          .attr('font-weight', 600)
          .attr('opacity', dimmed ? 0.3 : 1)
          .attr('pointer-events', 'none')
          .text(v);
      }
    }

    // Row labels
    const rowLabels = ['i₁', 'i₂', 'i₃'];
    for (let r = 0; r < unfoldRows; r++) {
      g.append('text')
        .attr('x', -6)
        .attr('y', r * cellSize + (cellSize - 1) / 2)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'central')
        .style('fill', 'var(--color-muted)')
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 9)
        .text(`${rowLabels[mode]}=${r + 1}`);
    }

    // Column indices
    for (let c = 0; c < unfoldCols; c++) {
      g.append('text')
        .attr('x', c * cellSize + (cellSize - 1) / 2)
        .attr('y', unfoldRows * cellSize + 12)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-muted)')
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 8)
        .text(c + 1);
    }
  }, [mode, cellSize, unfoldedMatrix, unfoldRows, unfoldCols, hoveredCell, isCellHighlighted]);

  // ─── Render ───

  return (
    <div ref={containerRef} className="w-full space-y-3">
      <div className={`flex gap-4 ${isWide ? 'flex-row items-start' : 'flex-col items-center'}`}>
        {/* 3D tensor view */}
        <div className="shrink-0">
          <p
            className="mb-1 text-center text-xs font-medium"
            style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-text)' }}
          >
            Tensor (4 × 3 × 2)
          </p>
          <svg
            ref={tensorRef}
            width={tensorW}
            height={tensorH}
            className="rounded-lg border"
            style={{ borderColor: 'var(--color-border)' }}
          />
        </div>

        {/* Unfolding heatmap */}
        <div className="min-w-0 overflow-x-auto">
          <p
            className="mb-1 text-center text-xs font-medium"
            style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-text)' }}
          >
            Unfolding
          </p>
          <svg
            ref={unfoldRef}
            width={unfoldW}
            height={unfoldH + 16}
            className="rounded-lg border"
            style={{ borderColor: 'var(--color-border)' }}
          />
        </div>
      </div>

      {/* Mode selector */}
      <div
        className="flex flex-wrap items-center gap-4 rounded-lg border px-4 py-2"
        style={{ borderColor: 'var(--color-border)', fontFamily: 'var(--font-sans)' }}
      >
        <span className="text-xs font-medium" style={{ color: 'var(--color-text)' }}>
          Unfold along:
        </span>
        {[0, 1, 2].map((m) => (
          <label key={m} className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--color-text)' }}>
            <input
              type="radio"
              name="tensor-mode"
              checked={mode === m}
              onChange={() => { setMode(m); setHoveredCell(null); }}
              className="accent-[var(--color-accent)]"
            />
            Mode {m + 1}
          </label>
        ))}
        <span className="text-xs" style={{ color: 'var(--color-muted)' }}>
          → {unfoldRows} × {unfoldCols} matrix
        </span>
      </div>
    </div>
  );
}
