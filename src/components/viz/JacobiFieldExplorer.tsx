import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import { jacobiFieldMagnitude } from './shared/manifoldGeometry';

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 300;
const PLOT_HEIGHT = 120;
const SM_BREAKPOINT = 640;

const TEAL = dimensionColors[0];
const PURPLE = dimensionColors[1];
const SLATE = '#64748b';
const RED = '#DC2626';

const fmt = (x: number) => x.toFixed(2);

// ── Component ────────────────────────────────────────────────────────

export default function JacobiFieldExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [kMagnitude, setKMagnitude] = useState(1.0);
  const [showConjugate, setShowConjugate] = useState(true);
  const [showNormPlot, setShowNormPlot] = useState(false);

  const tMax = kMagnitude > 0.01 ? Math.min(4, Math.PI / Math.sqrt(kMagnitude) * 1.3) : 4;

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked ? containerWidth : Math.floor((containerWidth - 16) / 3);

  // ── Jacobi field data ──────────────────────────────────────────────
  const fields = useMemo(() => {
    const nPts = 150;
    const positive: { t: number; j: number }[] = [];
    const zero: { t: number; j: number }[] = [];
    const negative: { t: number; j: number }[] = [];

    const posK = kMagnitude < 0.01 ? 0.01 : kMagnitude;

    for (let i = 0; i <= nPts; i++) {
      const t = (i / nPts) * tMax;
      positive.push({ t, j: jacobiFieldMagnitude(posK, t) });
      zero.push({ t, j: jacobiFieldMagnitude(0, t) });
      negative.push({ t, j: jacobiFieldMagnitude(-posK, t) });
    }

    return { positive, zero, negative };
  }, [kMagnitude, tMax]);

  // ── Conjugate point location ───────────────────────────────────────
  const conjugateT = kMagnitude > 0.01 ? Math.PI / Math.sqrt(kMagnitude) : null;

  // ── Draw a single Jacobi field panel ───────────────────────────────
  function drawPanel(
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    data: { t: number; j: number }[],
    color: string,
    label: string,
    curvSign: string,
    w: number,
    h: number,
    showConj: boolean,
    conjT: number | null
  ) {
    svg.selectAll('*').remove();

    const margin = { top: 28, right: 12, bottom: 20, left: 12 };
    const iw = w - margin.left - margin.right;
    const ih = h - margin.top - margin.bottom;

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Find max J for scaling
    const maxJ = Math.max(1, d3.max(data, (d) => Math.abs(d.j)) || 1);
    const geodesicY = ih / 2;

    const xScale = d3.scaleLinear().domain([0, tMax]).range([0, iw]);
    const jScale = d3.scaleLinear().domain([-maxJ * 1.1, maxJ * 1.1]).range([ih, 0]);

    // Central geodesic
    g.append('line')
      .attr('x1', 0).attr('y1', geodesicY)
      .attr('x2', iw).attr('y2', geodesicY)
      .style('stroke', color)
      .style('stroke-width', '2');

    // Neighboring geodesics (displaced by Jacobi field)
    const nNeighbors = 3;
    for (let n = 1; n <= nNeighbors; n++) {
      const sign = n % 2 === 0 ? 1 : -1;
      const amplitude = (Math.ceil(n / 2)) * 0.3;
      const lineData = data.map((d) => ({
        x: xScale(d.t),
        y: geodesicY - sign * amplitude * d.j * (ih * 0.35) / maxJ,
      }));
      const lineGen = d3.line<{ x: number; y: number }>()
        .x((d) => d.x).y((d) => d.y);
      g.append('path')
        .attr('d', lineGen(lineData))
        .attr('fill', 'none')
        .style('stroke', color)
        .style('stroke-width', '1')
        .style('opacity', `${0.6 - n * 0.1}`);
    }

    // Conjugate point
    if (showConj && conjT !== null && conjT <= tMax) {
      const cx = xScale(conjT);
      g.append('circle')
        .attr('cx', cx).attr('cy', geodesicY)
        .attr('r', 5)
        .style('fill', RED)
        .style('stroke', '#fff')
        .style('stroke-width', '1.5');
      g.append('text')
        .attr('x', cx).attr('y', geodesicY - 10)
        .attr('text-anchor', 'middle')
        .style('fill', RED)
        .style('font-size', '9px')
        .text(`t = ${fmt(conjT)}`);
    }

    // Start point
    g.append('circle')
      .attr('cx', 0).attr('cy', geodesicY)
      .attr('r', 4)
      .style('fill', color);

    // Labels
    svg.append('text')
      .attr('x', w / 2).attr('y', 16)
      .attr('text-anchor', 'middle')
      .style('fill', color)
      .style('font-size', '11px')
      .style('font-weight', 'bold')
      .text(label);

    svg.append('text')
      .attr('x', w / 2).attr('y', h - 4)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-muted, #666)')
      .style('font-size', '9px')
      .text(curvSign);
  }

  // ── Three panel refs ───────────────────────────────────────────────
  const posRef = useD3<SVGSVGElement>(
    (svg) => {
      if (panelWidth <= 0) return;
      drawPanel(
        svg,
        fields.positive, TEAL, 'K > 0 (converge)',
        `K = +${fmt(kMagnitude)}`,
        panelWidth, HEIGHT,
        showConjugate, conjugateT
      );
    },
    [fields.positive, panelWidth, showConjugate, conjugateT, kMagnitude]
  );

  const zeroRef = useD3<SVGSVGElement>(
    (svg) => {
      if (panelWidth <= 0) return;
      drawPanel(
        svg,
        fields.zero, SLATE, 'K = 0 (linear)',
        'K = 0',
        panelWidth, HEIGHT,
        false, null
      );
    },
    [fields.zero, panelWidth]
  );

  const negRef = useD3<SVGSVGElement>(
    (svg) => {
      if (panelWidth <= 0) return;
      drawPanel(
        svg,
        fields.negative, PURPLE, 'K < 0 (diverge)',
        `K = −${fmt(kMagnitude)}`,
        panelWidth, HEIGHT,
        false, null
      );
    },
    [fields.negative, panelWidth, kMagnitude]
  );

  // ── Norm plot ──────────────────────────────────────────────────────
  const normRef = useD3<SVGSVGElement>(
    (svg) => {
      if (!showNormPlot || containerWidth <= 0) return;
      svg.selectAll('*').remove();

      const w = containerWidth;
      const h = PLOT_HEIGHT;
      const margin = { top: 16, right: 16, bottom: 24, left: 40 };
      const iw = w - margin.left - margin.right;
      const ih = h - margin.top - margin.bottom;

      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const maxJ = Math.max(
        d3.max(fields.positive, (d) => d.j) || 1,
        d3.max(fields.zero, (d) => d.j) || 1,
        d3.max(fields.negative, (d) => d.j) || 1,
        1
      );

      const xScale = d3.scaleLinear().domain([0, tMax]).range([0, iw]);
      const yScale = d3.scaleLinear().domain([0, maxJ * 1.1]).range([ih, 0]);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${ih})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text').style('font-size', '9px');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(4))
        .selectAll('text').style('font-size', '9px');

      // Axis labels
      g.append('text')
        .attr('x', iw / 2).attr('y', ih + 20)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-muted, #666)')
        .style('font-size', '9px')
        .text('t');
      g.append('text')
        .attr('x', -8).attr('y', -6)
        .style('fill', 'var(--color-text-muted, #666)')
        .style('font-size', '9px')
        .text('|J(t)|');

      const drawLine = (data: { t: number; j: number }[], color: string) => {
        const lineGen = d3.line<{ t: number; j: number }>()
          .x((d) => xScale(d.t))
          .y((d) => yScale(Math.abs(d.j)));
        g.append('path')
          .attr('d', lineGen(data))
          .attr('fill', 'none')
          .style('stroke', color)
          .style('stroke-width', '2');
      };

      drawLine(fields.positive, TEAL);
      drawLine(fields.zero, SLATE);
      drawLine(fields.negative, PURPLE);

      // Conjugate point on plot
      if (showConjugate && conjugateT !== null && conjugateT <= tMax) {
        g.append('circle')
          .attr('cx', xScale(conjugateT))
          .attr('cy', yScale(0))
          .attr('r', 4)
          .style('fill', RED);
      }
    },
    [showNormPlot, fields, containerWidth, tMax, showConjugate, conjugateT]
  );

  // ── Render ─────────────────────────────────────────────────────────
  return (
    <div
      ref={containerRef}
      className="my-6 rounded-lg border p-3"
      style={{ borderColor: 'var(--color-border, #e5e7eb)', background: 'var(--color-bg-secondary, #f9fafb)' }}
    >
      <div className="mb-2 text-sm font-semibold" style={{ color: 'var(--color-text-primary, #111)' }}>
        Jacobi Field Explorer
      </div>

      {/* Three-column layout */}
      <div style={{ display: 'flex', flexDirection: isStacked ? 'column' : 'row', gap: '4px' }}>
        <svg role="img" aria-label="Jacobi field explorer visualization (panel 1 of 4)"
          ref={posRef}
          width={panelWidth}
          height={HEIGHT}
          style={{ background: 'var(--color-bg-primary, #fff)', borderRadius: '6px' }}
        />
        <svg role="img" aria-label="Jacobi field explorer visualization (panel 2 of 4)"
          ref={zeroRef}
          width={panelWidth}
          height={HEIGHT}
          style={{ background: 'var(--color-bg-primary, #fff)', borderRadius: '6px' }}
        />
        <svg role="img" aria-label="Jacobi field explorer visualization (panel 3 of 4)"
          ref={negRef}
          width={panelWidth}
          height={HEIGHT}
          style={{ background: 'var(--color-bg-primary, #fff)', borderRadius: '6px' }}
        />
      </div>

      {/* Norm plot */}
      {showNormPlot && (
        <svg role="img" aria-label="Jacobi field explorer visualization (panel 4 of 4)"
          ref={normRef}
          width={containerWidth}
          height={PLOT_HEIGHT}
          style={{ background: 'var(--color-bg-primary, #fff)', borderRadius: '6px', marginTop: 4 }}
        />
      )}

      {/* Controls */}
      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-1.5">
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Curvature |K|:</span>
          <input
            type="range" min={0.1} max={2} step={0.05} value={kMagnitude}
            onChange={(e) => setKMagnitude(parseFloat(e.target.value))}
            className="w-24"
          />
          <span className="font-mono w-10">{fmt(kMagnitude)}</span>
        </label>

        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={showConjugate} onChange={(e) => setShowConjugate(e.target.checked)} />
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Show conjugate points</span>
        </label>

        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={showNormPlot} onChange={(e) => setShowNormPlot(e.target.checked)} />
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Show |J(t)| plot</span>
        </label>

        {showConjugate && conjugateT !== null && (
          <span className="font-mono" style={{ color: RED }}>
            Conjugate at t = {fmt(conjugateT)}
          </span>
        )}
      </div>
    </div>
  );
}
