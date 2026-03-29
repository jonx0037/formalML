import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { getTuckerResults, type TuckerResult } from '../../data/tensor-decompositions-data';

// ─── Layout ───

const HEAT_MARGIN = { top: 20, right: 4, bottom: 4, left: 4 };
const MODE_LABELS = ['U⁽¹⁾', 'U⁽²⁾', 'U⁽³⁾'];

// ─── Color scale ───

function divergingScale(extent: number) {
  return d3.scaleSequential(d3.interpolateRdBu).domain([extent, -extent]);
}

// ─── Heatmap renderer ───

function renderHeatmap(
  svgEl: SVGSVGElement,
  data: number[][],
  title: string,
  cellSize: number,
) {
  const svg = d3.select(svgEl);
  svg.selectAll('*').remove();

  const rows = data.length;
  const cols = data[0].length;
  const flat = data.flat();
  const absMax = Math.max(Math.abs(Math.min(...flat)), Math.abs(Math.max(...flat)), 0.01);
  const color = divergingScale(absMax);

  const g = svg.append('g').attr('transform', `translate(${HEAT_MARGIN.left},${HEAT_MARGIN.top})`);

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      g.append('rect')
        .attr('x', j * cellSize)
        .attr('y', i * cellSize)
        .attr('width', cellSize - 0.5)
        .attr('height', cellSize - 0.5)
        .attr('fill', color(data[i][j]));
    }
  }

  g.append('text')
    .attr('x', (cols * cellSize) / 2)
    .attr('y', -5)
    .attr('text-anchor', 'middle')
    .style('fill', 'var(--color-text)')
    .style('font-family', 'var(--font-sans)')
    .attr('font-size', 10)
    .attr('font-weight', 500)
    .text(title);
}

// ─── Component ───

export default function TuckerCoreExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [selectedIndex, setSelectedIndex] = useState(0);

  const results = useMemo(() => getTuckerResults(), []);
  const result = results[selectedIndex];

  // Factor matrix SVG refs
  const factorRefs = [
    useRef<SVGSVGElement>(null),
    useRef<SVGSVGElement>(null),
    useRef<SVGSVGElement>(null),
  ];

  // Core slice SVG refs — one per frontal slice
  const coreSliceCount = result.core[0][0].length;
  const coreRefs = useRef<(SVGSVGElement | null)[]>([]);

  // ─── Responsive cell sizing ───

  const cellSize = useMemo(() => {
    if (!containerWidth) return 10;
    // Factor heatmaps share the top row; core slices share the bottom-left
    const available = Math.max(containerWidth / 4, 80);
    const maxDim = Math.max(
      ...result.factors.map((f) => Math.max(f.length, f[0].length)),
      result.core.length,
      result.core[0].length,
    );
    return Math.max(4, Math.min(Math.floor((available - HEAT_MARGIN.left - HEAT_MARGIN.right) / maxDim), 18));
  }, [containerWidth, result]);

  // ─── Factor matrix dimensions ───

  const factorDims = useMemo(
    () =>
      result.factors.map((f) => {
        const rows = f.length;
        const cols = f[0].length;
        return {
          rows,
          cols,
          w: cols * cellSize + HEAT_MARGIN.left + HEAT_MARGIN.right,
          h: rows * cellSize + HEAT_MARGIN.top + HEAT_MARGIN.bottom,
        };
      }),
    [result, cellSize],
  );

  // ─── Core slice dimensions ───

  const coreDims = useMemo(() => {
    const rows = result.core.length;
    const cols = result.core[0].length;
    return {
      rows,
      cols,
      w: cols * cellSize + HEAT_MARGIN.left + HEAT_MARGIN.right,
      h: rows * cellSize + HEAT_MARGIN.top + HEAT_MARGIN.bottom,
    };
  }, [result, cellSize]);

  // ─── Render factor heatmaps ───

  useEffect(() => {
    result.factors.forEach((factor, mode) => {
      const el = factorRefs[mode].current;
      if (!el) return;
      const dims = factorDims[mode];
      const label = `${MODE_LABELS[mode]}  (${dims.rows}×${dims.cols})`;
      renderHeatmap(el, factor, label, cellSize);
    });
  }, [result, cellSize, factorDims]);

  // ─── Render core slices ───

  useEffect(() => {
    const nSlices = result.core[0][0].length;
    for (let k = 0; k < nSlices; k++) {
      const el = coreRefs.current[k];
      if (!el) continue;
      // Extract frontal slice [:, :, k]
      const slice = result.core.map((row) => row.map((col) => col[k]));
      renderHeatmap(el, slice, `G[:,:,${k + 1}]`, cellSize);
    }
  }, [result, cellSize]);

  // ─── Stats ───

  const stats = useMemo(() => {
    const [R1, R2, R3] = result.ranks;
    const coreEntries = R1 * R2 * R3;
    // Factor dimensions come from data: I_n × R_n
    const factorEntries = result.factors.reduce((sum, f) => sum + f.length * f[0].length, 0);
    const totalStored = coreEntries + factorEntries;
    // Full tensor size from reconstruction
    const recon = result.reconstruction;
    const fullEntries = recon.length * recon[0].length * recon[0][0].length;

    return {
      relativeError: (result.relativeError * 100).toFixed(2),
      compressionRatio: result.compressionRatio.toFixed(1),
      coreEntries,
      factorEntries,
      totalStored,
      fullEntries,
    };
  }, [result]);

  const handleSelect = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => setSelectedIndex(Number(e.target.value)),
    [],
  );

  // ─── Render ───

  return (
    <div ref={containerRef} className="w-full space-y-4">
      {/* Rank selector */}
      <div className="flex items-center gap-3">
        <label
          className="whitespace-nowrap text-xs font-medium"
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          Multilinear rank (R₁, R₂, R₃)
        </label>
        <select
          value={selectedIndex}
          onChange={handleSelect}
          className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-2 py-1 text-xs"
          style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-text)' }}
        >
          {results.map((r, i) => (
            <option key={i} value={i}>
              ({r.ranks.join(', ')})
            </option>
          ))}
        </select>
      </div>

      {/* Factor matrices row */}
      <div>
        <p
          className="mb-1 text-xs font-medium"
          style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-muted)' }}
        >
          Factor matrices
        </p>
        <div className="flex flex-wrap justify-start gap-3">
          {factorDims.map((dim, mode) => (
            <svg role="img" aria-label="Tucker core explorer visualization (panel 1 of 2)"
              key={mode}
              ref={factorRefs[mode]}
              width={dim.w}
              height={dim.h}
              className="rounded border border-[var(--color-border)]"
            />
          ))}
        </div>
      </div>

      {/* Bottom row: core tensor + stats */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start">
        {/* Core tensor slices */}
        <div>
          <p
            className="mb-1 text-xs font-medium"
            style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-muted)' }}
          >
            Core tensor 𝒢 — frontal slices
          </p>
          <div className="flex flex-wrap gap-2">
            {Array.from({ length: coreSliceCount }, (_, k) => (
              <svg role="img" aria-label="Tucker core explorer visualization (panel 2 of 2)"
                key={k}
                ref={(el) => { coreRefs.current[k] = el; }}
                width={coreDims.w}
                height={coreDims.h}
                className="rounded border border-[var(--color-border)]"
              />
            ))}
          </div>
        </div>

        {/* Stats panel */}
        <div
          className="shrink-0 rounded-lg border border-[var(--color-border)] px-4 py-3 text-xs"
          style={{ fontFamily: 'var(--font-sans)', minWidth: 180 }}
        >
          <p className="mb-2 text-xs font-semibold" style={{ color: 'var(--color-text)' }}>
            Decomposition stats
          </p>
          <ul className="space-y-1" style={{ color: 'var(--color-text)' }}>
            <li>
              Relative error: <strong>{stats.relativeError}%</strong>
            </li>
            <li>
              Compression: <strong>{stats.compressionRatio}×</strong>
            </li>
            <li>
              Core entries: <strong>{stats.coreEntries}</strong>
            </li>
            <li>
              Factor entries: <strong>{stats.factorEntries}</strong>
            </li>
            <li>
              Total stored: <strong>{stats.totalStored}</strong>{' '}
              <span style={{ color: 'var(--color-muted)' }}>/ {stats.fullEntries}</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
