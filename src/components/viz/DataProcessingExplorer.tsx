import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import { binaryEntropy } from './shared/informationTheory';

// ── Constants ────────────────────────────────────────────────────────

const DIAGRAM_HEIGHT = 180;
const PLOT_HEIGHT = 260;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 30, bottom: 40, left: 55 };

const TEAL = dimensionColors[0];
const PURPLE = dimensionColors[1];
const AMBER = '#D97706';

type Channel = 'bsc' | 'gaussian' | 'quantization' | 'deterministic';

const fmt = (x: number) => x.toFixed(3);

/**
 * Compute I(X;Y) and I(X;Z) for each channel type given a noise parameter.
 *
 * Binary symmetric channel (BSC):
 *   X uniform Bernoulli, Y = X ⊕ noise(ε₁), Z = Y ⊕ noise(ε₂)
 *   I(X;Y) = 1 - H_b(ε₁)
 *   I(X;Z) = 1 - H_b(ε₁ * (1-ε₂) + (1-ε₁) * ε₂)  (composed channel)
 *
 * Gaussian channel:
 *   I(X;Y) = 0.5 * log2(1 + SNR)  where SNR = 1/noise²
 *
 * Quantization: I(X;Z) = fraction of I(X;Y) (modeled as lossy)
 *
 * Deterministic: I(X;Z) = I(X;Y) (equality case — sufficient statistic)
 */
function computeMI(channel: Channel, noise: number): { iXY: number; iXZ: number; iYZ: number } {
  const eps = Math.max(0.001, Math.min(0.499, noise));

  switch (channel) {
    case 'bsc': {
      const eps2 = 0.15; // fixed second-stage noise
      const composed = eps * (1 - eps2) + (1 - eps) * eps2;
      const iXY = 1 - binaryEntropy(eps);
      const iXZ = 1 - binaryEntropy(composed);
      const iYZ = 1 - binaryEntropy(eps2);
      return { iXY, iXZ, iYZ };
    }
    case 'gaussian': {
      const sigma1 = Math.max(0.01, noise);
      const sigma2 = 0.5; // fixed second-stage noise
      const snr1 = 1 / (sigma1 * sigma1);
      const snr_total = 1 / (sigma1 * sigma1 + sigma2 * sigma2);
      const snr2 = 1 / (sigma2 * sigma2);
      const iXY = 0.5 * Math.log2(1 + snr1);
      const iXZ = 0.5 * Math.log2(1 + snr_total);
      const iYZ = 0.5 * Math.log2(1 + snr2);
      return { iXY, iXZ, iYZ };
    }
    case 'quantization': {
      // Y is continuous, Z = floor(Y) loses information
      const iXY = Math.max(0, 1 - noise);
      const loss = noise * 0.3 + noise * noise * 0.5;
      const iXZ = Math.max(0, iXY - loss);
      const iYZ = Math.max(0, iXY - loss * 0.5);
      return { iXY, iXZ, iYZ };
    }
    case 'deterministic': {
      // Z = f(Y) deterministic and invertible → I(X;Z) = I(X;Y)
      const iXY = Math.max(0, 1 - noise);
      return { iXY, iXZ: iXY, iYZ: iXY };
    }
  }
}

// ── Component ────────────────────────────────────────────────────────

export default function DataProcessingExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [channel, setChannel] = useState<Channel>('bsc');
  const [noise, setNoise] = useState(0.15);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const { iXY, iXZ, iYZ } = useMemo(() => computeMI(channel, noise), [channel, noise]);

  // ── Markov chain diagram + MI bars ──────────────────────────────
  const diagramRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;

      const w = containerWidth;
      const h = DIAGRAM_HEIGHT;
      const g = svg.append('g');

      const nodeY = 50;
      const nodePositions = [
        { x: w * 0.2, label: 'X', sub: 'Source' },
        { x: w * 0.5, label: 'Y', sub: 'Observation' },
        { x: w * 0.8, label: 'Z', sub: 'Processed' },
      ];

      // Edges
      for (let i = 0; i < 2; i++) {
        const x1 = nodePositions[i].x + 22;
        const x2 = nodePositions[i + 1].x - 22;
        g.append('line')
          .attr('x1', x1).attr('y1', nodeY)
          .attr('x2', x2).attr('y2', nodeY)
          .attr('stroke', 'var(--color-border)')
          .attr('stroke-width', 2)
          .attr('marker-end', 'url(#arrow)');
      }

      // Arrow marker
      svg.append('defs').append('marker')
        .attr('id', 'arrow')
        .attr('viewBox', '0 0 10 10')
        .attr('refX', 9).attr('refY', 5)
        .attr('markerWidth', 8).attr('markerHeight', 8)
        .attr('orient', 'auto-start-reverse')
        .append('path')
        .attr('d', 'M 0 0 L 10 5 L 0 10 z')
        .attr('fill', 'var(--color-border)');

      // Nodes
      for (const node of nodePositions) {
        g.append('circle')
          .attr('cx', node.x).attr('cy', nodeY).attr('r', 22)
          .attr('fill', 'var(--color-bg)')
          .attr('stroke', TEAL)
          .attr('stroke-width', 2);

        g.append('text')
          .attr('x', node.x).attr('y', nodeY + 1)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'central')
          .style('fill', 'var(--color-text)')
          .style('font-size', '16px')
          .style('font-weight', '600')
          .text(node.label);

        g.append('text')
          .attr('x', node.x).attr('y', nodeY + 38)
          .attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '11px')
          .text(node.sub);
      }

      // MI bars below the diagram
      const barY = 105;
      const barH = 18;
      const maxBar = w * 0.35;
      const maxMI = Math.max(iXY, iXZ, iYZ, 0.01);

      const bars = [
        { label: 'I(X;Y)', val: iXY, color: TEAL },
        { label: 'I(X;Z)', val: iXZ, color: AMBER },
        { label: 'I(Y;Z)', val: iYZ, color: PURPLE },
      ];

      bars.forEach((bar, i) => {
        const y = barY + i * (barH + 10);
        g.append('text')
          .attr('x', w * 0.12)
          .attr('y', y + barH / 2)
          .attr('text-anchor', 'end')
          .attr('dominant-baseline', 'central')
          .style('fill', bar.color)
          .style('font-size', '12px')
          .style('font-weight', '600')
          .text(bar.label);

        g.append('rect')
          .attr('x', w * 0.14)
          .attr('y', y)
          .attr('width', Math.max(2, (bar.val / maxMI) * maxBar))
          .attr('height', barH)
          .attr('fill', bar.color)
          .attr('rx', 3)
          .attr('opacity', 0.75);

        g.append('text')
          .attr('x', w * 0.14 + (bar.val / maxMI) * maxBar + 8)
          .attr('y', y + barH / 2)
          .attr('dominant-baseline', 'central')
          .style('fill', 'var(--color-text)')
          .style('font-size', '11px')
          .text(`${fmt(bar.val)} bits`);
      });

      // DPI annotation
      const dpiGap = iXY - iXZ;
      g.append('text')
        .attr('x', w * 0.75)
        .attr('y', barY + barH / 2)
        .attr('dominant-baseline', 'central')
        .style('fill', dpiGap >= -0.001 ? TEAL : '#ef4444')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text(`DPI gap: ${fmt(Math.max(0, dpiGap))} bits`);

      if (channel === 'deterministic') {
        g.append('text')
          .attr('x', w * 0.75)
          .attr('y', barY + barH / 2 + 18)
          .attr('dominant-baseline', 'central')
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '11px')
          .style('font-style', 'italic')
          .text('(equality — sufficient statistic)');
      }
    },
    [containerWidth, iXY, iXZ, iYZ, channel]
  );

  // ── MI vs. noise plot ───────────────────────────────────────────
  const plotRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (containerWidth <= 0) return;

      const w = containerWidth - MARGIN.left - MARGIN.right;
      const h = PLOT_HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const noiseRange = channel === 'gaussian' ? [0.01, 2] as [number, number] : [0.001, 0.499] as [number, number];
      const xScale = d3.scaleLinear().domain(noiseRange).range([0, w]);
      const nSteps = 200;
      const step = (noiseRange[1] - noiseRange[0]) / nSteps;

      // Compute curves
      const curveXY: [number, number][] = [];
      const curveXZ: [number, number][] = [];
      for (let n = noiseRange[0]; n <= noiseRange[1]; n += step) {
        const mi = computeMI(channel, n);
        curveXY.push([n, mi.iXY]);
        curveXZ.push([n, mi.iXZ]);
      }

      const maxY = Math.max(1, ...curveXY.map((d) => d[1]), ...curveXZ.map((d) => d[1]));
      const yScale = d3.scaleLinear().domain([0, maxY * 1.05]).range([h, 0]);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px');

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px');

      g.append('text')
        .attr('x', w / 2).attr('y', h + 35)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text(channel === 'gaussian' ? 'Noise σ' : 'Noise ε');

      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -h / 2).attr('y', -42)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px')
        .text('Mutual Information (bits)');

      // DPI gap shading (area between curves)
      const areaGen = d3.area<[number, number]>()
        .x((d) => xScale(d[0]))
        .y0((_, i) => yScale(curveXZ[i]?.[1] ?? 0))
        .y1((d) => yScale(d[1]))
        .curve(d3.curveCatmullRom);

      g.append('path')
        .datum(curveXY)
        .attr('d', areaGen)
        .attr('fill', AMBER)
        .attr('opacity', 0.12);

      // Lines
      const line = d3.line<[number, number]>()
        .x((d) => xScale(d[0]))
        .y((d) => yScale(d[1]))
        .curve(d3.curveCatmullRom);

      g.append('path')
        .datum(curveXY)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', TEAL)
        .attr('stroke-width', 2.5);

      g.append('path')
        .datum(curveXZ)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', AMBER)
        .attr('stroke-width', 2.5);

      // Current noise indicator
      g.append('line')
        .attr('x1', xScale(noise)).attr('x2', xScale(noise))
        .attr('y1', 0).attr('y2', h)
        .attr('stroke', 'var(--color-text-secondary)')
        .attr('stroke-dasharray', '4,3')
        .attr('opacity', 0.5);

      g.append('circle')
        .attr('cx', xScale(noise))
        .attr('cy', yScale(iXY))
        .attr('r', 5)
        .attr('fill', TEAL);

      g.append('circle')
        .attr('cx', xScale(noise))
        .attr('cy', yScale(iXZ))
        .attr('r', 5)
        .attr('fill', AMBER);

      // Legend
      const legend = g.append('g').attr('transform', `translate(${w - 120}, 5)`);

      [
        { label: 'I(X;Y)', color: TEAL },
        { label: 'I(X;Z)', color: AMBER },
        { label: 'DPI gap', color: AMBER, dashed: true },
      ].forEach((item, i) => {
        const ly = i * 18;
        if (item.dashed) {
          legend.append('rect')
            .attr('x', 0).attr('y', ly + 2)
            .attr('width', 16).attr('height', 10)
            .attr('fill', item.color)
            .attr('opacity', 0.2);
        } else {
          legend.append('line')
            .attr('x1', 0).attr('x2', 16)
            .attr('y1', ly + 7).attr('y2', ly + 7)
            .attr('stroke', item.color)
            .attr('stroke-width', 2.5);
        }
        legend.append('text')
          .attr('x', 22).attr('y', ly + 7)
          .attr('dominant-baseline', 'central')
          .style('fill', 'var(--color-text-secondary)')
          .style('font-size', '11px')
          .text(item.label);
      });

      // Title
      g.append('text')
        .attr('x', w / 2).attr('y', -12)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '13px')
        .style('font-weight', '500')
        .text('I(X;Y) ≥ I(X;Z)  — Data Processing Inequality');

      // Tick styling
      g.selectAll('.domain').style('stroke', 'var(--color-border)');
      g.selectAll('.tick line').style('stroke', 'var(--color-border)');
    },
    [containerWidth, channel, noise, iXY, iXZ]
  );

  return (
    <div ref={containerRef} className="w-full">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3 mb-4 text-sm">
        <label className="flex items-center gap-1.5" style={{ color: 'var(--color-text-secondary)' }}>
          Channel:
          <select
            value={channel}
            onChange={(e) => { setChannel(e.target.value as Channel); setNoise(0.15); }}
            className="rounded border px-2 py-1 text-sm"
            style={{ borderColor: 'var(--color-border)', backgroundColor: 'var(--color-bg)', color: 'var(--color-text)' }}
          >
            <option value="bsc">Binary Symmetric</option>
            <option value="gaussian">Gaussian</option>
            <option value="quantization">Quantization</option>
            <option value="deterministic">Deterministic (equality)</option>
          </select>
        </label>

        <label className="flex items-center gap-2" style={{ color: 'var(--color-text-secondary)' }}>
          {channel === 'gaussian' ? 'σ' : 'ε'}:
          <input
            type="range"
            min={channel === 'gaussian' ? 0.01 : 0.001}
            max={channel === 'gaussian' ? 2 : 0.499}
            step={0.005}
            value={noise}
            onChange={(e) => setNoise(Number(e.target.value))}
            className="w-36"
          />
          <span className="text-xs font-mono w-12">{noise.toFixed(3)}</span>
        </label>
      </div>

      {/* Markov chain diagram */}
      <svg ref={diagramRef} width={containerWidth} height={DIAGRAM_HEIGHT} />

      {/* MI vs. noise plot */}
      <svg ref={plotRef} width={containerWidth} height={PLOT_HEIGHT} />
    </div>
  );
}
