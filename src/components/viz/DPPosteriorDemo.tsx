import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

const MARGIN = { top: 30, right: 16, bottom: 40, left: 56 };
const GRID_SIZE = 200;
const X_MIN = -4;
const X_MAX = 4;
const MAX_DATA = 20;
/**
 * Bandwidth for the Gaussian smoothing kernel applied to each data point.
 * Controls the width of the bump around each observation in the posterior
 * predictive. Smaller values give sharper peaks; larger values give smoother
 * curves. This is a visualization aid — the true DP predictive has point masses
 * at observed atoms, which we smooth for visual clarity.
 */
const KERNEL_BW = 0.3;
const SQRT_2PI = Math.sqrt(2 * Math.PI);

const COLORS = {
  prior: '#6B7280',       // gray for G0
  posterior: '#2563EB',   // blue for posterior predictive (smoothed)
  posteriorFill: 'rgba(37, 99, 235, 0.3)',
  dataPoint: '#DC2626',   // red dots
} as const;

/** Standard normal PDF */
function normalPDF(x: number): number {
  return Math.exp(-0.5 * x * x) / SQRT_2PI;
}

/**
 * Gaussian kernel centered at xi with bandwidth h.
 * Used to smooth the discrete point masses in the DP predictive distribution
 * into a continuous density for visualization. The bandwidth h controls how
 * spread out each bump is (h = KERNEL_BW = 0.3 by default).
 */
function gaussianKernel(x: number, xi: number, h: number): number {
  const z = (x - xi) / h;
  return Math.exp(-0.5 * z * z) / (h * SQRT_2PI);
}

export default function DPPosteriorDemo() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  const [alpha, setAlpha] = useState(1.0);
  const [dataPoints, setDataPoints] = useState<number[]>([]);

  const panelW = containerWidth || 0;
  const panelH = Math.min(320, Math.max(220, panelW * 0.5));

  // ─── Evaluation grid and densities ───

  const { grid, g0, posterior, maxDensity } = useMemo(() => {
    const grid = Array.from({ length: GRID_SIZE }, (_, i) => X_MIN + (X_MAX - X_MIN) * i / (GRID_SIZE - 1));
    const g0 = grid.map(normalPDF);
    const n = dataPoints.length;

    let posterior: number[];
    if (n === 0) {
      posterior = g0.slice();
    } else {
      posterior = grid.map((x) => {
        const priorTerm = (alpha / (alpha + n)) * normalPDF(x);
        let kernelSum = 0;
        for (let j = 0; j < n; j++) {
          kernelSum += gaussianKernel(x, dataPoints[j], KERNEL_BW);
        }
        const dataTerm = (1 / (alpha + n)) * kernelSum;
        return priorTerm + dataTerm;
      });
    }

    const maxDensity = Math.max(
      d3.max(g0) || 0.5,
      d3.max(posterior) || 0.5,
    );

    return { grid, g0, posterior, maxDensity };
  }, [alpha, dataPoints]);

  // ─── D3 rendering ───

  const handleClick = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    if (dataPoints.length >= MAX_DATA) return;
    const svg = svgRef.current;
    if (!svg || panelW <= 0) return;

    const rect = svg.getBoundingClientRect();
    const px = event.clientX - rect.left;

    const xScale = d3.scaleLinear().domain([X_MIN, X_MAX]).range([MARGIN.left, panelW - MARGIN.right]);
    const xVal = xScale.invert(px);

    if (xVal >= X_MIN && xVal <= X_MAX) {
      setDataPoints(prev => [...prev, xVal]);
    }
  }, [dataPoints.length, panelW]);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const xScale = d3.scaleLinear().domain([X_MIN, X_MAX]).range([MARGIN.left, panelW - MARGIN.right]);
    const yScale = d3.scaleLinear().domain([0, maxDensity * 1.1]).range([panelH - MARGIN.bottom, MARGIN.top]);

    // ─── Axes ───

    sel.append('g').attr('transform', `translate(0,${panelH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(8).tickSize(3))
      .call(g => {
        g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)');
        g.select('.domain').style('stroke', 'var(--color-border)');
        g.selectAll('.tick line').style('stroke', 'var(--color-border)');
      });

    sel.append('g').attr('transform', `translate(${MARGIN.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5).tickSize(3))
      .call(g => {
        g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)');
        g.select('.domain').style('stroke', 'var(--color-border)');
        g.selectAll('.tick line').style('stroke', 'var(--color-border)');
      });

    // Axis labels
    sel.append('text').attr('x', panelW / 2).attr('y', panelH - 4)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('x');
    sel.append('text').attr('transform', 'rotate(-90)').attr('x', -(panelH / 2)).attr('y', 14)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('density');

    // ─── Posterior predictive filled area ───

    const areaGen = d3.area<number>()
      .x((_, i) => xScale(grid[i]))
      .y0(yScale(0))
      .y1((_, i) => yScale(posterior[i]));

    sel.append('path').datum(posterior)
      .attr('d', areaGen)
      .style('fill', COLORS.posteriorFill)
      .style('stroke', 'none');

    // ─── Posterior predictive line ───

    const posteriorLine = d3.line<number>()
      .x((_, i) => xScale(grid[i]))
      .y((_, i) => yScale(posterior[i]));

    sel.append('path').datum(posterior)
      .attr('d', posteriorLine)
      .style('fill', 'none')
      .style('stroke', COLORS.posterior)
      .style('stroke-width', 2);

    // ─── G0 prior (dashed) ───

    const g0Line = d3.line<number>()
      .x((_, i) => xScale(grid[i]))
      .y((_, i) => yScale(g0[i]));

    sel.append('path').datum(g0)
      .attr('d', g0Line)
      .style('fill', 'none')
      .style('stroke', COLORS.prior)
      .style('stroke-width', 1.8)
      .style('stroke-dasharray', '6 3');

    // ─── Data points on x-axis ───

    const axisY = yScale(0);
    dataPoints.forEach((xi) => {
      sel.append('circle')
        .attr('cx', xScale(xi))
        .attr('cy', axisY - 6)
        .attr('r', 4)
        .style('fill', COLORS.dataPoint)
        .style('stroke', 'white')
        .style('stroke-width', 1);
    });

    // ─── Title ───

    sel.append('text').attr('x', MARGIN.left + 4).attr('y', MARGIN.top - 10)
      .style('font-size', '11px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', 'var(--color-text)')
      .text('DP Posterior Predictive (kernel-smoothed)');

    // ─── Legend ───

    const legendX = panelW - MARGIN.right - 160;
    const legendY = MARGIN.top + 6;

    // Prior G0
    sel.append('line').attr('x1', legendX).attr('x2', legendX + 18).attr('y1', legendY).attr('y2', legendY)
      .style('stroke', COLORS.prior).style('stroke-width', 1.8).style('stroke-dasharray', '6 3');
    sel.append('text').attr('x', legendX + 22).attr('y', legendY + 4)
      .style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('Prior G\u2080');

    // Posterior predictive
    sel.append('line').attr('x1', legendX).attr('x2', legendX + 18).attr('y1', legendY + 16).attr('y2', legendY + 16)
      .style('stroke', COLORS.posterior).style('stroke-width', 2);
    sel.append('rect').attr('x', legendX).attr('y', legendY + 12).attr('width', 18).attr('height', 8)
      .style('fill', COLORS.posteriorFill).style('stroke', 'none');
    sel.append('text').attr('x', legendX + 22).attr('y', legendY + 20)
      .style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('Posterior predictive (smoothed)');

    // Click-area overlay (transparent, for cursor hint)
    sel.append('rect')
      .attr('x', MARGIN.left)
      .attr('y', MARGIN.top)
      .attr('width', panelW - MARGIN.left - MARGIN.right)
      .attr('height', panelH - MARGIN.top - MARGIN.bottom)
      .style('fill', 'transparent')
      .style('cursor', 'crosshair');

  }, [grid, g0, posterior, maxDensity, dataPoints, panelW, panelH]);

  // ─── Render ───

  return (
    <div
      ref={containerRef}
      style={{
        border: '1px solid var(--color-border)',
        borderRadius: '8px',
        padding: '16px',
        background: 'var(--color-surface)',
        marginTop: '1.5rem',
        marginBottom: '1.5rem',
      }}
    >
      <div style={{ fontFamily: 'var(--font-sans)', fontSize: '13px', fontWeight: 600, marginBottom: '12px', color: 'var(--color-text)' }}>
        DP Posterior Updating
      </div>

      <svg
        ref={svgRef}
        width={panelW}
        height={panelH}
        onClick={handleClick}
        style={{
          border: '1px solid var(--color-border)',
          borderRadius: '6px',
          background: 'var(--color-muted-bg)',
          cursor: 'crosshair',
          display: 'block',
        }}
      />

      <div style={{
        marginTop: '6px',
        fontSize: '10px',
        fontFamily: 'var(--font-mono)',
        color: 'var(--color-text-secondary)',
        fontStyle: 'italic',
      }}>
        Click on the plot to add data points (max {MAX_DATA})
      </div>

      {/* Controls */}
      <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '14px', alignItems: 'center' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
          α = {alpha.toFixed(1)}
          <input
            type="range"
            min={0.1}
            max={50}
            step={0.1}
            value={alpha}
            onChange={e => setAlpha(parseFloat(e.target.value))}
            style={{ width: '120px', accentColor: 'var(--color-accent)' }}
          />
        </label>

        <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
          n = {dataPoints.length} observations
        </span>

        <button
          onClick={() => setDataPoints([])}
          style={{
            padding: '3px 10px',
            fontSize: '12px',
            fontFamily: 'var(--font-mono)',
            border: '1px solid var(--color-border)',
            borderRadius: '4px',
            background: 'var(--color-surface)',
            color: 'var(--color-text)',
            cursor: 'pointer',
          }}
        >
          Clear
        </button>
      </div>
    </div>
  );
}
