import { useState, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import { entropy, binaryEntropy, normalize } from './shared/informationTheory';

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 320;
const BINARY_HEIGHT = 260;
const SURPRISE_HEIGHT = 220;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 20, bottom: 40, left: 50 };

const TEAL = dimensionColors[0];
const PURPLE = dimensionColors[1];
const AMBER = '#D97706';

const K_OPTIONS = [2, 3, 4, 6, 8];

const fmt = (x: number) => x.toFixed(3);

function uniformDist(k: number): number[] {
  return new Array(k).fill(1 / k);
}

function peakedDist(k: number): number[] {
  const rest = 0.1 / (k - 1);
  return [0.9, ...new Array(k - 1).fill(rest)];
}

// Warm-to-cool color scale for surprise magnitudes
const surpriseColor = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, 6]);

// ── Component ────────────────────────────────────────────────────────

export default function EntropyExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [k, setK] = useState(4);
  const [probs, setProbs] = useState(() => uniformDist(4));
  const [binaryP, setBinaryP] = useState(0.5);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  // Update k and reset distribution
  const handleKChange = useCallback((newK: number) => {
    setK(newK);
    setProbs(uniformDist(newK));
  }, []);

  // ── Computed values ──────────────────────────────────────────────
  const H = useMemo(() => entropy(probs), [probs]);
  const maxH = useMemo(() => Math.log2(k), [k]);
  const surprises = useMemo(() => probs.map((p) => (p > 0 ? -Math.log2(p) : 0)), [probs]);

  // ── Distribution bar chart (left panel) ─────────────────────────
  const distWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.55);

  const distRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (distWidth <= 0) return;

      const w = distWidth - MARGIN.left - MARGIN.right;
      const h = HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleBand<number>()
        .domain(probs.map((_, i) => i))
        .range([0, w])
        .padding(0.2);

      const yScale = d3.scaleLinear().domain([0, 1]).range([h, 0]);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).tickFormat((i) => `x${(i as number) + 1}`))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px');

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px');

      // Axis labels
      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 35)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('Outcome');

      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -h / 2)
        .attr('y', -38)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('p(x)');

      // Max entropy dashed line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', w)
        .attr('y1', yScale(1 / k))
        .attr('y2', yScale(1 / k))
        .attr('stroke', PURPLE)
        .attr('stroke-dasharray', '5,4')
        .attr('stroke-width', 1.5)
        .attr('opacity', 0.6);

      g.append('text')
        .attr('x', w + 2)
        .attr('y', yScale(1 / k))
        .attr('dy', '0.35em')
        .style('fill', PURPLE)
        .style('font-size', '10px')
        .text('1/k');

      // Draggable bars — bind {prob, index} to avoid indexOf ambiguity
      const barData = probs.map((p, i) => ({ p, i }));
      const bars = g.selectAll<SVGRectElement, { p: number; i: number }>('.bar')
        .data(barData)
        .enter()
        .append('rect')
        .attr('class', 'bar')
        .attr('x', (d) => xScale(d.i)!)
        .attr('y', (d) => yScale(d.p))
        .attr('width', xScale.bandwidth())
        .attr('height', (d) => h - yScale(d.p))
        .attr('fill', TEAL)
        .attr('rx', 2)
        .style('cursor', 'ns-resize');

      // Drag behavior
      const drag = d3.drag<SVGRectElement, { p: number; i: number }>()
        .on('drag', function (event, d) {
          const newVal = Math.max(0.001, yScale.invert(Math.max(0, Math.min(h, event.y))));
          const updated = [...probs];
          updated[d.i] = newVal;
          setProbs(normalize(updated));
        });

      bars.call(drag);

      // Entropy display
      const entropyText = g.append('text')
        .attr('x', w / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle');

      entropyText.append('tspan')
        .style('fill', 'var(--color-text)')
        .style('font-size', '14px')
        .style('font-weight', '600')
        .text(`H(X) = ${fmt(H)} bits`);

      entropyText.append('tspan')
        .attr('dx', '0.8em')
        .style('fill', PURPLE)
        .style('font-size', '11px')
        .style('font-weight', '500')
        .text(`(max = ${fmt(maxH)})`);

      // Tick styling
      g.selectAll('.domain').style('stroke', 'var(--color-border)');
      g.selectAll('.tick line').style('stroke', 'var(--color-border)');
    },
    [distWidth, probs, H, maxH, k]
  );

  // ── Binary entropy curve (right panel) ──────────────────────────
  const binaryWidth = isStacked ? containerWidth : containerWidth - distWidth;

  const binaryRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (binaryWidth <= 0) return;

      const w = binaryWidth - MARGIN.left - MARGIN.right;
      const h = BINARY_HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLinear().domain([0, 1]).range([0, w]);
      const yScale = d3.scaleLinear().domain([0, 1.05]).range([h, 0]);

      // Generate curve
      const curveData: [number, number][] = [];
      for (let p = 0; p <= 1; p += 0.005) {
        curveData.push([p, binaryEntropy(p)]);
      }

      const line = d3.line<[number, number]>()
        .x((d) => xScale(d[0]))
        .y((d) => yScale(d[1]))
        .curve(d3.curveCatmullRom);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px');

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px');

      // Axis labels
      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 35)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('p');

      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -h / 2)
        .attr('y', -38)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('Hb(p)');

      // Curve
      g.append('path')
        .datum(curveData)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', AMBER)
        .attr('stroke-width', 2.5);

      // Maximum annotation at p=0.5
      g.append('line')
        .attr('x1', xScale(0.5))
        .attr('x2', xScale(0.5))
        .attr('y1', yScale(1))
        .attr('y2', h)
        .attr('stroke', 'var(--color-border)')
        .attr('stroke-dasharray', '3,3');

      g.append('text')
        .attr('x', xScale(0.5))
        .attr('y', yScale(1) - 6)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '10px')
        .text('max = 1 bit');

      // Draggable point
      const currentH = binaryEntropy(binaryP);
      const pointG = g.append('g').style('cursor', 'ew-resize');

      // Vertical dotted line from point to x-axis
      pointG.append('line')
        .attr('x1', xScale(binaryP))
        .attr('x2', xScale(binaryP))
        .attr('y1', yScale(currentH))
        .attr('y2', h)
        .attr('stroke', TEAL)
        .attr('stroke-dasharray', '2,2')
        .attr('opacity', 0.5);

      pointG.append('circle')
        .attr('cx', xScale(binaryP))
        .attr('cy', yScale(currentH))
        .attr('r', 7)
        .attr('fill', TEAL)
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);

      pointG.append('text')
        .attr('x', xScale(binaryP))
        .attr('y', yScale(currentH) - 12)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text(`p=${binaryP.toFixed(2)}, Hb=${currentH.toFixed(3)}`);

      // Drag on the SVG for binary entropy
      const dragBinary = d3.drag<SVGSVGElement, unknown>()
        .on('drag', (event) => {
          const [mx] = d3.pointer(event, g.node());
          const newP = Math.max(0.01, Math.min(0.99, xScale.invert(mx)));
          setBinaryP(newP);
        });

      svg.call(dragBinary);

      // Tick styling
      g.selectAll('.domain').style('stroke', 'var(--color-border)');
      g.selectAll('.tick line').style('stroke', 'var(--color-border)');
    },
    [binaryWidth, binaryP]
  );

  // ── Surprise bar chart (bottom panel) ───────────────────────────
  const surpriseRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;

      const w = containerWidth - MARGIN.left - MARGIN.right;
      const h = SURPRISE_HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const maxSurprise = Math.max(...surprises, 1);

      const xScale = d3.scaleBand<number>()
        .domain(probs.map((_, i) => i))
        .range([0, w])
        .padding(0.2);

      const yScale = d3.scaleLinear().domain([0, maxSurprise * 1.15]).range([h, 0]);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).tickFormat((i) => `x${(i as number) + 1}`))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px');

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(4))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px');

      // Axis label
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -h / 2)
        .attr('y', -38)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('−log₂ p(x)');

      // Surprise bars (color-coded by magnitude)
      g.selectAll('.surprise-bar')
        .data(surprises)
        .enter()
        .append('rect')
        .attr('x', (_, i) => xScale(i)!)
        .attr('y', (d) => yScale(d))
        .attr('width', xScale.bandwidth())
        .attr('height', (d) => h - yScale(d))
        .attr('fill', (d) => surpriseColor(d))
        .attr('rx', 2);

      // Value labels on bars
      g.selectAll('.surprise-label')
        .data(surprises)
        .enter()
        .append('text')
        .attr('x', (_, i) => xScale(i)! + xScale.bandwidth() / 2)
        .attr('y', (d) => yScale(d) - 4)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '10px')
        .text((d) => d.toFixed(2));

      // Entropy annotation line (weighted average)
      g.append('line')
        .attr('x1', 0)
        .attr('x2', w)
        .attr('y1', yScale(H))
        .attr('y2', yScale(H))
        .attr('stroke', TEAL)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '6,3');

      g.append('text')
        .attr('x', w + 2)
        .attr('y', yScale(H))
        .attr('dy', '0.35em')
        .style('fill', TEAL)
        .style('font-size', '10px')
        .style('font-weight', '600')
        .text(`H = ${fmt(H)}`);

      // Title
      g.append('text')
        .attr('x', w / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '13px')
        .style('font-weight', '500')
        .text('Surprise: −log₂ p(xᵢ)   (weighted avg = entropy)');

      // Tick styling
      g.selectAll('.domain').style('stroke', 'var(--color-border)');
      g.selectAll('.tick line').style('stroke', 'var(--color-border)');
    },
    [containerWidth, surprises, H, probs]
  );

  return (
    <div ref={containerRef} className="w-full">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3 mb-4 text-sm">
        <label className="flex items-center gap-1.5" style={{ color: 'var(--color-text-secondary)' }}>
          Outcomes (k):
          <select
            value={k}
            onChange={(e) => handleKChange(Number(e.target.value))}
            className="rounded border px-2 py-1 text-sm"
            style={{
              borderColor: 'var(--color-border)',
              backgroundColor: 'var(--color-bg)',
              color: 'var(--color-text)',
            }}
          >
            {K_OPTIONS.map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>

        <button
          onClick={() => setProbs(uniformDist(k))}
          className="rounded px-3 py-1 text-sm font-medium transition-colors"
          style={{
            backgroundColor: 'var(--color-definition-bg)',
            color: TEAL,
            border: `1px solid ${TEAL}`,
          }}
        >
          Uniform
        </button>

        <button
          onClick={() => setProbs(peakedDist(k))}
          className="rounded px-3 py-1 text-sm font-medium transition-colors"
          style={{
            backgroundColor: 'var(--color-theorem-bg)',
            color: PURPLE,
            border: `1px solid ${PURPLE}`,
          }}
        >
          Peaked
        </button>
      </div>

      {/* Top panels: distribution + binary entropy */}
      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-2`}>
        <svg role="img" aria-label="Entropy explorer visualization (panel 1 of 3)" ref={distRef} width={distWidth} height={HEIGHT} />
        <svg role="img" aria-label="Entropy explorer visualization (panel 2 of 3)" ref={binaryRef} width={binaryWidth} height={BINARY_HEIGHT} />
      </div>

      {/* Bottom panel: surprise chart */}
      <svg role="img" aria-label="Entropy explorer visualization (panel 3 of 3)" ref={surpriseRef} width={containerWidth} height={SURPRISE_HEIGHT} />
    </div>
  );
}
