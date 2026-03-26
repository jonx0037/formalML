import { useState, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import {
  entropy,
  crossEntropy,
  klDivergence,
  normalize,
} from './shared/informationTheory';

// ── Constants ────────────────────────────────────────────────────────

const DIST_HEIGHT = 300;
const ASYM_HEIGHT = 120;
const PW_HEIGHT = 220;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 20, bottom: 40, left: 55 };

const BLUE = dimensionColors[0];
const RED = '#DC2626';
const PURPLE = dimensionColors[1];
const AMBER = '#D97706';

const K_OPTIONS = [2, 3, 4, 6];

const fmt = (x: number) =>
  Number.isFinite(x) ? x.toFixed(3) : '∞';

function uniformDist(k: number): number[] {
  return new Array(k).fill(1 / k);
}

function initialP(k: number): number[] {
  if (k === 2) return normalize([0.7, 0.3]);
  if (k === 3) return normalize([0.5, 0.3, 0.2]);
  if (k === 4) return normalize([0.4, 0.3, 0.2, 0.1]);
  return normalize(Array.from({ length: k }, (_, i) => k - i));
}

// ── Component ────────────────────────────────────────────────────────

export default function KLDivergenceExplorer() {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();

  const [k, setK] = useState(4);
  const [pDist, setPDist] = useState(() => initialP(4));
  const [qDist, setQDist] = useState(() => uniformDist(4));

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const handleKChange = useCallback((newK: number) => {
    setK(newK);
    setPDist(initialP(newK));
    setQDist(uniformDist(newK));
  }, []);

  const handleMatch = useCallback(() => {
    setQDist([...pDist]);
  }, [pDist]);

  const handleUniformQ = useCallback(() => {
    setQDist(uniformDist(k));
  }, [k]);

  // ── Computed values ──────────────────────────────────────────────
  const Hp = useMemo(() => entropy(pDist), [pDist]);
  const Hpq = useMemo(() => crossEntropy(pDist, qDist), [pDist, qDist]);
  const klPQ = useMemo(() => klDivergence(pDist, qDist), [pDist, qDist]);
  const klQP = useMemo(() => klDivergence(qDist, pDist), [pDist, qDist]);

  const pointwiseKL = useMemo(
    () =>
      pDist.map((pi, i) => {
        if (pi <= 0) return 0;
        if (qDist[i] <= 0) return Infinity;
        return pi * Math.log2(pi / qDist[i]);
      }),
    [pDist, qDist]
  );

  // ── Distribution bar chart ───────────────────────────────────────
  const distWidth = isStacked
    ? containerWidth
    : Math.floor(containerWidth * 0.6);

  const distRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (distWidth <= 0) return;

      const w = distWidth - MARGIN.left - MARGIN.right;
      const h = DIST_HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg
        .append('g')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3
        .scaleBand<number>()
        .domain(pDist.map((_, i) => i))
        .range([0, w])
        .padding(0.25);

      const yScale = d3.scaleLinear().domain([0, 1]).range([h, 0]);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(
          d3.axisBottom(xScale).tickFormat((i) => `x${(i as number) + 1}`)
        )
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px');

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px');

      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -h / 2)
        .attr('y', -40)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('Probability');

      const bw = xScale.bandwidth();
      const halfBw = bw / 2;

      // q bars (background, red)
      const qBarData = qDist.map((q, i) => ({ v: q, i }));
      const qBars = g
        .selectAll<SVGRectElement, { v: number; i: number }>('.q-bar')
        .data(qBarData)
        .enter()
        .append('rect')
        .attr('class', 'q-bar')
        .attr('x', (d) => xScale(d.i)! + halfBw)
        .attr('y', (d) => yScale(d.v))
        .attr('width', halfBw)
        .attr('height', (d) => h - yScale(d.v))
        .attr('fill', RED)
        .attr('opacity', 0.65)
        .attr('rx', 2)
        .style('cursor', 'ns-resize');

      // p bars (foreground, blue)
      const pBarData = pDist.map((p, i) => ({ v: p, i }));
      const pBars = g
        .selectAll<SVGRectElement, { v: number; i: number }>('.p-bar')
        .data(pBarData)
        .enter()
        .append('rect')
        .attr('class', 'p-bar')
        .attr('x', (d) => xScale(d.i)!)
        .attr('y', (d) => yScale(d.v))
        .attr('width', halfBw)
        .attr('height', (d) => h - yScale(d.v))
        .attr('fill', BLUE)
        .attr('opacity', 0.75)
        .attr('rx', 2)
        .style('cursor', 'ns-resize');

      // Drag p
      pBars.call(
        d3
          .drag<SVGRectElement, { v: number; i: number }>()
          .on('drag', function (event, d) {
            const newVal = Math.max(
              0.001,
              yScale.invert(Math.max(0, Math.min(h, event.y)))
            );
            const updated = [...pDist];
            updated[d.i] = newVal;
            setPDist(normalize(updated));
          })
      );

      // Drag q
      qBars.call(
        d3
          .drag<SVGRectElement, { v: number; i: number }>()
          .on('drag', function (event, d) {
            const newVal = Math.max(
              0.001,
              yScale.invert(Math.max(0, Math.min(h, event.y)))
            );
            const updated = [...qDist];
            updated[d.i] = newVal;
            setQDist(normalize(updated));
          })
      );

      // Legend
      const legend = g
        .append('g')
        .attr('transform', `translate(${w - 80}, -15)`);
      legend
        .append('rect')
        .attr('width', 12)
        .attr('height', 12)
        .attr('fill', BLUE)
        .attr('opacity', 0.75);
      legend
        .append('text')
        .attr('x', 16)
        .attr('y', 10)
        .style('fill', 'var(--color-text)')
        .style('font-size', '12px')
        .text('p');
      legend
        .append('rect')
        .attr('x', 35)
        .attr('width', 12)
        .attr('height', 12)
        .attr('fill', RED)
        .attr('opacity', 0.65);
      legend
        .append('text')
        .attr('x', 51)
        .attr('y', 10)
        .style('fill', 'var(--color-text)')
        .style('font-size', '12px')
        .text('q');

      // Info text
      const info = g
        .append('text')
        .attr('x', 0)
        .attr('y', -10)
        .style('font-size', '13px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)');

      info.text(
        `H(p) = ${fmt(Hp)}   H(p,q) = ${fmt(Hpq)}   D_KL(p‖q) = ${fmt(klPQ)}`
      );

      // Tick styling
      g.selectAll('.domain').style('stroke', 'var(--color-border)');
      g.selectAll('.tick line').style('stroke', 'var(--color-border)');
    },
    [distWidth, pDist, qDist, Hp, Hpq, klPQ, k]
  );

  // ── Asymmetry bars (right panel) ─────────────────────────────────
  const asymWidth = isStacked ? containerWidth : containerWidth - distWidth;

  const asymRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (asymWidth <= 0) return;

      const m = { top: 30, right: 20, bottom: 10, left: 20 };
      const w = asymWidth - m.left - m.right;
      const h = ASYM_HEIGHT - m.top - m.bottom;
      const g = svg
        .append('g')
        .attr('transform', `translate(${m.left},${m.top})`);

      const maxVal = Math.max(
        Number.isFinite(klPQ) ? klPQ : 5,
        Number.isFinite(klQP) ? klQP : 5,
        0.1
      );
      const xScale = d3.scaleLinear().domain([0, maxVal * 1.15]).range([0, w]);

      const barH = 24;
      const gap = 16;

      // Forward KL bar
      g.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', Number.isFinite(klPQ) ? xScale(klPQ) : w)
        .attr('height', barH)
        .attr('fill', BLUE)
        .attr('opacity', 0.7)
        .attr('rx', 3);

      g.append('text')
        .attr('x', 0)
        .attr('y', -6)
        .style('fill', BLUE)
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text(`D_KL(p‖q) = ${fmt(klPQ)} bits`);

      // Reverse KL bar
      g.append('rect')
        .attr('x', 0)
        .attr('y', barH + gap)
        .attr('width', Number.isFinite(klQP) ? xScale(klQP) : w)
        .attr('height', barH)
        .attr('fill', RED)
        .attr('opacity', 0.7)
        .attr('rx', 3);

      g.append('text')
        .attr('x', 0)
        .attr('y', barH + gap - 6)
        .style('fill', RED)
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text(`D_KL(q‖p) = ${fmt(klQP)} bits`);

      // Asymmetry annotation
      const ratio =
        Number.isFinite(klPQ) && Number.isFinite(klQP) && klQP > 0.001
          ? klPQ / klQP
          : NaN;
      const asymmetric =
        Number.isFinite(ratio) && (ratio > 1.2 || ratio < 0.8);

      g.append('text')
        .attr('x', w / 2)
        .attr('y', 2 * barH + gap + 14)
        .attr('text-anchor', 'middle')
        .style('fill', asymmetric ? AMBER : 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .style('font-weight', asymmetric ? '600' : '400')
        .text(
          asymmetric
            ? `KL is NOT symmetric (ratio ≈ ${ratio.toFixed(2)})`
            : Number.isFinite(ratio)
              ? `ratio ≈ ${ratio.toFixed(2)}`
              : ''
        );
    },
    [asymWidth, klPQ, klQP]
  );

  // ── Pointwise KL chart (bottom panel) ────────────────────────────
  const pwRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;

      const w = containerWidth - MARGIN.left - MARGIN.right;
      const h = PW_HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg
        .append('g')
        .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3
        .scaleBand<number>()
        .domain(pDist.map((_, i) => i))
        .range([0, w])
        .padding(0.25);

      const finiteVals = pointwiseKL.filter(Number.isFinite);
      const maxPW = finiteVals.length > 0 ? Math.max(...finiteVals, 0.01) : 1;
      const yScale = d3
        .scaleLinear()
        .domain([Math.min(0, ...finiteVals), maxPW * 1.15])
        .range([h, 0]);

      const colorScale = d3
        .scaleSequential(d3.interpolateYlOrRd)
        .domain([0, maxPW]);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(
          d3.axisBottom(xScale).tickFormat((i) => `x${(i as number) + 1}`)
        )
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px');

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px');

      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -h / 2)
        .attr('y', -40)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('p(x) log₂(p(x)/q(x))');

      // Zero line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', w)
        .attr('y1', yScale(0))
        .attr('y2', yScale(0))
        .attr('stroke', 'var(--color-border)')
        .attr('stroke-dasharray', '4,3');

      // Bars
      pointwiseKL.forEach((val, i) => {
        const isInf = !Number.isFinite(val);
        const displayVal = isInf ? maxPW * 1.1 : val;
        const barY = displayVal >= 0 ? yScale(displayVal) : yScale(0);
        const barH = Math.abs(yScale(displayVal) - yScale(0));

        g.append('rect')
          .attr('x', xScale(i)!)
          .attr('y', barY)
          .attr('width', xScale.bandwidth())
          .attr('height', Math.max(barH, 1))
          .attr('fill', isInf ? '#EF4444' : colorScale(Math.abs(val)))
          .attr('opacity', 0.8)
          .attr('rx', 2);

        // Value labels
        g.append('text')
          .attr('x', xScale(i)! + xScale.bandwidth() / 2)
          .attr('y', barY - 4)
          .attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text)')
          .style('font-size', '10px')
          .text(isInf ? '∞' : val.toFixed(3));
      });

      // Title
      g.append('text')
        .attr('x', w / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text(
          `Pointwise KL contributions (sum = ${fmt(klPQ)} bits)`
        );

      g.selectAll('.domain').style('stroke', 'var(--color-border)');
      g.selectAll('.tick line').style('stroke', 'var(--color-border)');
    },
    [containerWidth, pDist, qDist, pointwiseKL, klPQ]
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
          gap: '12px',
          flexWrap: 'wrap',
          marginBottom: '12px',
          alignItems: 'center',
        }}
      >
        <label style={{ fontSize: '13px', color: 'var(--color-text-secondary)' }}>
          Outcomes k:
          <select
            value={k}
            onChange={(e) => handleKChange(Number(e.target.value))}
            style={{
              marginLeft: '6px',
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid var(--color-border)',
              background: 'var(--color-surface)',
              color: 'var(--color-text)',
              fontSize: '13px',
            }}
          >
            {K_OPTIONS.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <button
          onClick={handleMatch}
          style={{
            padding: '4px 12px',
            borderRadius: '4px',
            border: '1px solid var(--color-border)',
            background: 'var(--color-surface)',
            color: 'var(--color-text)',
            fontSize: '13px',
            cursor: 'pointer',
          }}
        >
          Match (q = p)
        </button>
        <button
          onClick={handleUniformQ}
          style={{
            padding: '4px 12px',
            borderRadius: '4px',
            border: '1px solid var(--color-border)',
            background: 'var(--color-surface)',
            color: 'var(--color-text)',
            fontSize: '13px',
            cursor: 'pointer',
          }}
        >
          Uniform q
        </button>
        <span
          style={{
            fontSize: '12px',
            color: 'var(--color-text-secondary)',
            fontStyle: 'italic',
          }}
        >
          Drag bars to adjust p (blue) and q (red)
        </span>
      </div>

      {/* Distribution chart + Asymmetry bars */}
      <div
        style={{
          display: 'flex',
          flexDirection: isStacked ? 'column' : 'row',
          gap: '0px',
        }}
      >
        <svg
          ref={distRef}
          width={distWidth}
          height={DIST_HEIGHT}
          style={{ overflow: 'visible' }}
        />
        <svg
          ref={asymRef}
          width={asymWidth}
          height={ASYM_HEIGHT}
          style={{ overflow: 'visible', alignSelf: 'center' }}
        />
      </div>

      {/* Pointwise KL chart */}
      <svg
        ref={pwRef}
        width={containerWidth}
        height={PW_HEIGHT}
        style={{ overflow: 'visible' }}
      />
    </div>
  );
}
