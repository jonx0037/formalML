import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';

// ── Constants ────────────────────────────────────────────────────────

const PANEL_HEIGHT = 320;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 20, bottom: 40, left: 55 };

const BLUE = dimensionColors[0];
const RED = '#DC2626';
const PURPLE = dimensionColors[1];
const AMBER = '#D97706';

// ── Component ────────────────────────────────────────────────────────

export default function VariationalBoundsExplorer() {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();

  const [slopeS, setSlopeS] = useState(1.5);
  const [muP, setMuP] = useState(2);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  // ── Left panel: Fenchel conjugate of f(t) = t log t ──────────────
  const fenchelWidth = isStacked
    ? containerWidth
    : Math.floor(containerWidth / 2);

  const fenchelRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (fenchelWidth <= 0) return;

      const w = fenchelWidth - MARGIN.left - MARGIN.right;
      const h = PANEL_HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg
        .append('g')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLinear().domain([0.01, 5]).range([0, w]);
      const yScale = d3.scaleLinear().domain([-2, 6]).range([h, 0]);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');

      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 30)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('t');

      const tGrid = d3.range(0.05, 5, 0.02);

      // f(t) = t log t
      const fData = tGrid.map((t) => ({ t, v: t * Math.log(t) }));
      g.append('path')
        .datum(fData.filter((d) => d.v >= -2 && d.v <= 6))
        .attr(
          'd',
          d3
            .line<{ t: number; v: number }>()
            .x((d) => xScale(d.t))
            .y((d) => yScale(d.v))
            .curve(d3.curveBasis)
        )
        .attr('fill', 'none')
        .attr('stroke', BLUE)
        .attr('stroke-width', 2.5);

      // Linear function s*t
      const lineData = tGrid.map((t) => ({ t, v: slopeS * t }));
      g.append('path')
        .datum(lineData.filter((d) => d.v >= -2 && d.v <= 6))
        .attr(
          'd',
          d3
            .line<{ t: number; v: number }>()
            .x((d) => xScale(d.t))
            .y((d) => yScale(d.v))
            .curve(d3.curveLinear)
        )
        .attr('fill', 'none')
        .attr('stroke', RED)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '6,3');

      // Shade gap st - f(t) where positive
      const gapData = tGrid
        .map((t) => {
          const ft = t * Math.log(t);
          const st = slopeS * t;
          return { t, st, ft, gap: st - ft };
        })
        .filter((d) => d.gap > 0 && d.ft >= -2 && d.st <= 6);

      if (gapData.length > 1) {
        g.append('path')
          .datum(gapData)
          .attr(
            'd',
            d3
              .area<{ t: number; st: number; ft: number }>()
              .x((d) => xScale(d.t))
              .y0((d) => yScale(d.ft))
              .y1((d) => yScale(d.st))
              .curve(d3.curveBasis)
          )
          .attr('fill', AMBER)
          .attr('opacity', 0.2);
      }

      // Optimal t* = e^{s-1} (where gap is maximized)
      const tStar = Math.exp(slopeS - 1);
      const fStar = Math.exp(slopeS - 1); // f*(s) = e^{s-1}

      if (tStar > 0.01 && tStar < 5) {
        g.append('circle')
          .attr('cx', xScale(tStar))
          .attr('cy', yScale(tStar * Math.log(tStar)))
          .attr('r', 5)
          .attr('fill', AMBER)
          .attr('stroke', '#fff')
          .attr('stroke-width', 1.5);

        g.append('text')
          .attr('x', xScale(tStar) + 8)
          .attr('y', yScale(tStar * Math.log(tStar)))
          .attr('dy', '-4')
          .style('fill', AMBER)
          .style('font-size', '11px')
          .style('font-weight', '600')
          .text(`t* = e^(s−1) = ${tStar.toFixed(2)}`);
      }

      // f*(s) annotation
      g.append('text')
        .attr('x', w - 5)
        .attr('y', yScale(5.5))
        .attr('text-anchor', 'end')
        .style('fill', AMBER)
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text(`f*(${slopeS.toFixed(1)}) = e^(s−1) = ${fStar.toFixed(2)}`);

      // Labels
      g.append('text')
        .attr('x', xScale(4.5))
        .attr('y', yScale(4.5 * Math.log(4.5)) - 8)
        .style('fill', BLUE)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text('f(t) = t log t');

      g.append('text')
        .attr('x', xScale(3))
        .attr('y', yScale(slopeS * 3) - 8)
        .style('fill', RED)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text(`st (s = ${slopeS.toFixed(1)})`);

      // Title
      g.append('text')
        .attr('x', w / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text('Fenchel conjugate: f*(s) = sup{st − f(t)}');

      g.selectAll('.domain').style('stroke', 'var(--color-border)');
      g.selectAll('.tick line').style('stroke', 'var(--color-border)');
    },
    [fenchelWidth, slopeS]
  );

  // ── Right panel: NWJ bound for Gaussians ─────────────────────────
  const nwjWidth = isStacked
    ? containerWidth
    : containerWidth - fenchelWidth;

  // True KL divergence: KL(N(mu_p, 1) || N(0, 1)) = mu_p^2 / 2
  const trueKL = useMemo(() => (muP * muP) / 2, [muP]);

  const nwjRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (nwjWidth <= 0) return;

      const w = nwjWidth - MARGIN.left - MARGIN.right;
      const h = PANEL_HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg
        .append('g')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      // NWJ bound for linear T(x) = ax: bound = a*mu_p - a^2/2
      const aMax = Math.max(muP * 2, 4);
      const aGrid = d3.range(-aMax, aMax, 0.05);
      const boundData = aGrid.map((a) => ({
        a,
        bound: a * muP - (a * a) / 2,
      }));

      const maxBound = Math.max(trueKL * 1.3, 1);
      const xScale = d3.scaleLinear().domain([-aMax, aMax]).range([0, w]);
      const yScale = d3
        .scaleLinear()
        .domain([Math.min(-1, ...boundData.map((d) => d.bound)), maxBound])
        .range([h, 0]);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '11px');

      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 30)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('Critic slope a (T(x) = ax)');

      // Bound curve
      g.append('path')
        .datum(boundData)
        .attr(
          'd',
          d3
            .line<{ a: number; bound: number }>()
            .x((d) => xScale(d.a))
            .y((d) => yScale(d.bound))
            .curve(d3.curveBasis)
        )
        .attr('fill', 'none')
        .attr('stroke', PURPLE)
        .attr('stroke-width', 2.5);

      // True KL horizontal line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', w)
        .attr('y1', yScale(trueKL))
        .attr('y2', yScale(trueKL))
        .attr('stroke', AMBER)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '6,3');

      g.append('text')
        .attr('x', w - 5)
        .attr('y', yScale(trueKL) - 6)
        .attr('text-anchor', 'end')
        .style('fill', AMBER)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text(`True D_KL = ${trueKL.toFixed(3)} nats`);

      // Optimal a* = mu_p (where bound peaks)
      const optA = muP;
      const optBound = optA * muP - (optA * optA) / 2;
      g.append('circle')
        .attr('cx', xScale(optA))
        .attr('cy', yScale(optBound))
        .attr('r', 5)
        .attr('fill', PURPLE)
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5);

      g.append('text')
        .attr('x', xScale(optA) + 8)
        .attr('y', yScale(optBound) + 4)
        .style('fill', PURPLE)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text(`a* = μ_p = ${muP.toFixed(1)}`);

      // Zero line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', w)
        .attr('y1', yScale(0))
        .attr('y2', yScale(0))
        .attr('stroke', 'var(--color-border)')
        .attr('stroke-dasharray', '3,3');

      // Title
      g.append('text')
        .attr('x', w / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text('NWJ bound: E_p[T] − E_q[e^(T−1)]');

      g.selectAll('.domain').style('stroke', 'var(--color-border)');
      g.selectAll('.tick line').style('stroke', 'var(--color-border)');
    },
    [nwjWidth, muP, trueKL]
  );

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        background: 'var(--color-surface, #fff)',
        borderRadius: '8px',
        border: '1px solid var(--color-border, #e5e7eb)',
        padding: '16px',
      }}
    >
      {/* Controls */}
      <div
        style={{
          display: 'flex',
          gap: '20px',
          flexWrap: 'wrap',
          marginBottom: '12px',
          alignItems: 'center',
        }}
      >
        <label
          style={{ fontSize: '13px', color: 'var(--color-text-secondary)' }}
        >
          Slope s:
          <input
            type="range"
            min={-1}
            max={3}
            step={0.1}
            value={slopeS}
            onChange={(e) => setSlopeS(Number(e.target.value))}
            style={{
              marginLeft: '8px',
              width: '120px',
              verticalAlign: 'middle',
            }}
          />
          <span style={{ marginLeft: '4px' }}>{slopeS.toFixed(1)}</span>
        </label>
        <label
          style={{ fontSize: '13px', color: 'var(--color-text-secondary)' }}
        >
          μ_p (mean of p):
          <input
            type="range"
            min={0}
            max={4}
            step={0.1}
            value={muP}
            onChange={(e) => setMuP(Number(e.target.value))}
            style={{
              marginLeft: '8px',
              width: '120px',
              verticalAlign: 'middle',
            }}
          />
          <span style={{ marginLeft: '4px' }}>{muP.toFixed(1)}</span>
        </label>
      </div>

      {/* Two panels */}
      <div
        style={{
          display: 'flex',
          flexDirection: isStacked ? 'column' : 'row',
        }}
      >
        <svg role="img" aria-label="Variational bounds explorer visualization (panel 1 of 2)"
          ref={fenchelRef}
          width={fenchelWidth}
          height={PANEL_HEIGHT}
          style={{ overflow: 'visible' }}
        />
        <svg role="img" aria-label="Variational bounds explorer visualization (panel 2 of 2)"
          ref={nwjRef}
          width={nwjWidth}
          height={PANEL_HEIGHT}
          style={{ overflow: 'visible' }}
        />
      </div>
    </div>
  );
}
