import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  IMAGE_SIZE, singularValues, getOriginalPixels, reconstructPixelRaw,
  toDisplayValue, energyRetained,
} from '../../data/svd-image-data';

// ─── Layout ───

const CANVAS_SIZE = 192;
const SPECTRUM_WIDTH = 240;
const SPECTRUM_HEIGHT = 160;
const SPEC_MARGIN = { top: 20, right: 12, bottom: 28, left: 40 };

// ─── Component ───

export default function ImageCompressionDemo() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const origCanvasRef = useRef<HTMLCanvasElement>(null);
  const compCanvasRef = useRef<HTMLCanvasElement>(null);
  const spectrumRef = useRef<SVGSVGElement>(null);

  const [rank, setRank] = useState(10);

  // ─── Compute compressed image pixels ───

  const compressedPixels = useMemo(() => {
    const pixels: number[][] = [];
    for (let i = 0; i < IMAGE_SIZE; i++) {
      const row: number[] = [];
      for (let j = 0; j < IMAGE_SIZE; j++) {
        row.push(toDisplayValue(reconstructPixelRaw(i, j, rank)));
      }
      pixels.push(row);
    }
    return pixels;
  }, [rank]);

  // ─── Statistics ───

  const stats = useMemo(() => {
    const storageOriginal = IMAGE_SIZE * IMAGE_SIZE;
    const storageTruncated = rank * (IMAGE_SIZE + IMAGE_SIZE + 1);
    const energy = energyRetained(rank);

    // Relative error (Frobenius): sqrt(1 - energy retained)
    const relativeError = Math.sqrt(1 - energy);

    return {
      compressionRatio: storageTruncated > 0 ? storageOriginal / storageTruncated : Infinity,
      energyPct: energy * 100,
      relativeError,
      storageOriginal,
      storageTruncated,
    };
  }, [rank]);

  // ─── Canvas sizing ───

  const canvasDisplay = useMemo(() => {
    if (!containerWidth) return CANVAS_SIZE;
    return Math.min(CANVAS_SIZE, Math.floor(containerWidth / 2.5));
  }, [containerWidth]);

  const specWidth = useMemo(() => {
    if (!containerWidth) return SPECTRUM_WIDTH;
    return Math.min(SPECTRUM_WIDTH, containerWidth - 16);
  }, [containerWidth]);

  // ─── Compute original pixels lazily (only when component mounts) ───

  const originalPixels = useMemo(() => getOriginalPixels(), []);

  // ─── Render original image to canvas ───

  useEffect(() => {
    const canvas = origCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const imgData = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE);
    for (let i = 0; i < IMAGE_SIZE; i++) {
      for (let j = 0; j < IMAGE_SIZE; j++) {
        const idx = (i * IMAGE_SIZE + j) * 4;
        const v = originalPixels[i][j];
        imgData.data[idx] = v;
        imgData.data[idx + 1] = v;
        imgData.data[idx + 2] = v;
        imgData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }, [originalPixels]);

  // ─── Render compressed image to canvas ───

  useEffect(() => {
    const canvas = compCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const imgData = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE);
    for (let i = 0; i < IMAGE_SIZE; i++) {
      for (let j = 0; j < IMAGE_SIZE; j++) {
        const idx = (i * IMAGE_SIZE + j) * 4;
        const v = compressedPixels[i][j];
        imgData.data[idx] = v;
        imgData.data[idx + 1] = v;
        imgData.data[idx + 2] = v;
        imgData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }, [compressedPixels]);

  // ─── Spectrum chart ───

  useEffect(() => {
    if (!spectrumRef.current || specWidth === 0) return;

    const svg = d3.select(spectrumRef.current);
    svg.selectAll('*').remove();

    const w = specWidth - SPEC_MARGIN.left - SPEC_MARGIN.right;
    const h = SPECTRUM_HEIGHT - SPEC_MARGIN.top - SPEC_MARGIN.bottom;
    const g = svg.append('g').attr('transform', `translate(${SPEC_MARGIN.left},${SPEC_MARGIN.top})`);

    // Log scale for singular values
    const xScale = d3.scaleLinear().domain([0, IMAGE_SIZE - 1]).range([0, w]);
    const yScale = d3.scaleLog()
      .domain([Math.max(singularValues[IMAGE_SIZE - 1], 0.5), singularValues[0] * 1.1])
      .range([h, 0]);

    // Spectrum line
    const line = d3.line<number>()
      .x((_, i) => xScale(i))
      .y((d) => yScale(Math.max(d, 0.5)));

    g.append('path')
      .datum(singularValues)
      .attr('d', line)
      .attr('fill', 'none')
      .style('stroke', 'var(--color-text)')
      .attr('stroke-width', 1.5);

    // Fill area for retained components
    const areaGen = d3.area<number>()
      .x((_, i) => xScale(i))
      .y0(h)
      .y1((d) => yScale(Math.max(d, 0.5)));

    g.append('path')
      .datum(singularValues.slice(0, rank))
      .attr('d', areaGen)
      .style('fill', 'var(--color-accent)')
      .attr('opacity', 0.15);

    // Rank cutoff line
    g.append('line')
      .attr('x1', xScale(rank - 0.5))
      .attr('y1', 0)
      .attr('x2', xScale(rank - 0.5))
      .attr('y2', h)
      .attr('stroke', '#ef4444')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,3');

    // Rank label
    g.append('text')
      .attr('x', xScale(rank - 0.5) + 4)
      .attr('y', 12)
      .style('fill', '#ef4444')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 9)
      .attr('font-weight', 600)
      .text(`k=${rank}`);

    // Axes
    const xAxis = d3.axisBottom(xScale).ticks(6).tickSize(-h);
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(xAxis)
      .call((g) => g.select('.domain').remove())
      .call((g) => g.selectAll('.tick line').style('stroke', 'var(--color-muted)').attr('opacity', 0.15))
      .call((g) => g.selectAll('.tick text').style('fill', 'var(--color-muted)').attr('font-size', 9));

    const yAxis = d3.axisLeft(yScale).ticks(4, '.0f').tickSize(-w);
    g.append('g')
      .call(yAxis)
      .call((g) => g.select('.domain').remove())
      .call((g) => g.selectAll('.tick line').style('stroke', 'var(--color-muted)').attr('opacity', 0.15))
      .call((g) => g.selectAll('.tick text').style('fill', 'var(--color-muted)').attr('font-size', 9));

    // Labels
    g.append('text')
      .attr('x', w / 2)
      .attr('y', h + 22)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-muted)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 9)
      .text('Index');

    g.append('text')
      .attr('x', -6)
      .attr('y', -8)
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 10)
      .attr('font-weight', 600)
      .text('Singular value spectrum (log scale)');
  }, [rank, specWidth]);

  // ─── Render ───

  return (
    <div ref={containerRef} className="w-full space-y-3">
      {/* Images + Spectrum */}
      <div className="flex flex-wrap items-start justify-center gap-4">
        {/* Original */}
        <div className="text-center">
          <div className="mb-1 text-xs font-medium opacity-60" style={{ fontFamily: 'var(--font-sans)' }}>
            Original ({IMAGE_SIZE}×{IMAGE_SIZE})
          </div>
          <canvas
            ref={origCanvasRef}
            width={IMAGE_SIZE}
            height={IMAGE_SIZE}
            className="rounded border border-[var(--color-border)]"
            style={{ width: canvasDisplay, height: canvasDisplay, imageRendering: 'pixelated' }}
          />
        </div>

        {/* Compressed */}
        <div className="text-center">
          <div className="mb-1 text-xs font-medium opacity-60" style={{ fontFamily: 'var(--font-sans)' }}>
            Rank-{rank} approximation
          </div>
          <canvas
            ref={compCanvasRef}
            width={IMAGE_SIZE}
            height={IMAGE_SIZE}
            className="rounded border border-[var(--color-border)]"
            style={{ width: canvasDisplay, height: canvasDisplay, imageRendering: 'pixelated' }}
          />
        </div>

        {/* Spectrum */}
        <svg role="img" aria-label="Image compression demo visualization"
          ref={spectrumRef}
          width={specWidth}
          height={SPECTRUM_HEIGHT}
          className="rounded-lg border border-[var(--color-border)]"
        />
      </div>

      {/* Statistics */}
      <div
        className="flex flex-wrap gap-x-6 gap-y-1 rounded-lg border border-[var(--color-border)] px-4 py-2 text-xs"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        <span>Energy retained: <strong>{stats.energyPct.toFixed(1)}%</strong></span>
        <span>Relative error: <strong>{(stats.relativeError * 100).toFixed(1)}%</strong></span>
        <span>
          Storage: <strong>{stats.storageTruncated}</strong> / {stats.storageOriginal}
          {' '}({stats.compressionRatio.toFixed(1)}× compression)
        </span>
      </div>

      {/* Rank slider */}
      <div className="flex items-center gap-3">
        <label
          className="min-w-[100px] whitespace-nowrap text-xs font-medium"
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          Rank k = {rank}
        </label>
        <input
          type="range"
          min={1}
          max={IMAGE_SIZE}
          step={1}
          value={rank}
          onChange={useCallback((e: React.ChangeEvent<HTMLInputElement>) => setRank(parseInt(e.target.value)), [])}
          className="w-full accent-[var(--color-accent)]"
        />
      </div>
    </div>
  );
}
