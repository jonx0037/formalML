import { useState, useMemo, useRef } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import {
  klDivGaussian,
  hellingerDistGaussian,
  fisherRaoDistanceGaussian,
  clamp,
} from './shared/manifoldGeometry';

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 380;
const SM_BREAKPOINT = 640;

const TEAL = dimensionColors[0];
const PURPLE = dimensionColors[1];
const AMBER = '#D97706';

const MU_RANGE: [number, number] = [-3, 3];
const SIG_RANGE: [number, number] = [0, 3];
const DRAG_PAD_X = 0.2;
const DRAG_PAD_SIG_LO = 0.15;
const DRAG_PAD_SIG_HI = 0.1;

type DivType = 'kl-forward' | 'kl-reverse' | 'hellinger' | 'fisher-rao';

const fmt = (x: number) => x.toFixed(4);
const fmt2 = (x: number) => x.toFixed(2);

/** Gaussian PDF */
function gaussianPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

// ── Component ────────────────────────────────────────────────────────

export default function DivergenceExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [mu1, setMu1] = useState(-0.5);
  const [sig1, setSig1] = useState(0.8);
  const [mu2, setMu2] = useState(1.0);
  const [sig2, setSig2] = useState(1.5);
  const [divType, setDivType] = useState<DivType>('kl-forward');
  const draggingPRef = useRef(true);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const leftWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.5);
  const rightWidth = isStacked ? containerWidth : containerWidth - leftWidth;

  // ── Divergence value ─────────────────────────────────────────────
  const divValue = useMemo(() => {
    switch (divType) {
      case 'kl-forward': return klDivGaussian(mu1, sig1, mu2, sig2);
      case 'kl-reverse': return klDivGaussian(mu2, sig2, mu1, sig1);
      case 'hellinger': return hellingerDistGaussian(mu1, sig1, mu2, sig2);
      case 'fisher-rao': return fisherRaoDistanceGaussian(mu1, sig1, mu2, sig2);
    }
  }, [mu1, sig1, mu2, sig2, divType]);

  const divLabel = useMemo(() => {
    switch (divType) {
      case 'kl-forward': return 'D_KL(p ‖ q)';
      case 'kl-reverse': return 'D_KL(q ‖ p)';
      case 'hellinger': return 'H²(p, q)';
      case 'fisher-rao': return 'd_FR(p, q)';
    }
  }, [divType]);

  // ── Left panel: parameter space ──────────────────────────────────
  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (leftWidth <= 0) return;

      const margin = { top: 30, right: 20, bottom: 40, left: 50 };
      const w = leftWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain(MU_RANGE).range([0, w]);
      const yScale = d3.scaleLinear().domain(SIG_RANGE).range([h, 0]);

      // Axes
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', w / 2).attr('y', h + 35)
        .style('fill', 'var(--color-text-secondary)').style('text-anchor', 'middle').style('font-size', '12px').text('μ');
      g.append('text').attr('x', -35).attr('y', h / 2)
        .style('fill', 'var(--color-text-secondary)').style('text-anchor', 'middle')
        .style('font-size', '12px').attr('transform', `rotate(-90,-35,${h / 2})`).text('σ');

      // Connecting line
      g.append('line')
        .attr('x1', xScale(mu1)).attr('y1', yScale(sig1))
        .attr('x2', xScale(mu2)).attr('y2', yScale(sig2))
        .style('stroke', 'var(--color-border)').style('stroke-width', 1.5).style('stroke-dasharray', '4,3');

      // Point 1 (p)
      g.append('circle')
        .attr('cx', xScale(mu1)).attr('cy', yScale(sig1))
        .attr('r', 8).style('fill', TEAL).style('stroke', '#fff').style('stroke-width', 2).style('cursor', 'grab');

      // Point 2 (q)
      g.append('circle')
        .attr('cx', xScale(mu2)).attr('cy', yScale(sig2))
        .attr('r', 8).style('fill', PURPLE).style('stroke', '#fff').style('stroke-width', 2).style('cursor', 'grab');

      // Invisible overlay for drag — limited to the plot area
      const overlay = g.append('rect')
        .attr('width', w).attr('height', h)
        .style('fill', 'none').style('pointer-events', 'all').style('cursor', 'grab');

      overlay.call(d3.drag<SVGRectElement, unknown>()
        .on('start', (event) => {
          const mx = xScale.invert(event.x);
          const my = yScale.invert(event.y);
          const d1 = (mx - mu1) ** 2 + (my - sig1) ** 2;
          const d2 = (mx - mu2) ** 2 + (my - sig2) ** 2;
          draggingPRef.current = d1 <= d2;
        })
        .on('drag', (event) => {
          const mu = clamp(xScale.invert(event.x), MU_RANGE[0] + DRAG_PAD_X, MU_RANGE[1] - DRAG_PAD_X);
          const sig = clamp(yScale.invert(event.y), SIG_RANGE[0] + DRAG_PAD_SIG_LO, SIG_RANGE[1] - DRAG_PAD_SIG_HI);
          if (draggingPRef.current) { setMu1(mu); setSig1(sig); }
          else { setMu2(mu); setSig2(sig); }
        }));

      // Labels
      g.append('text').attr('x', xScale(mu1) - 12).attr('y', yScale(sig1) - 12)
        .style('fill', TEAL).style('font-size', '12px').style('font-weight', '600').text('p');
      g.append('text').attr('x', xScale(mu2) + 8).attr('y', yScale(sig2) - 12)
        .style('fill', PURPLE).style('font-size', '12px').style('font-weight', '600').text('q');

      // Divergence value display
      g.append('text')
        .attr('x', w / 2).attr('y', 15)
        .style('fill', AMBER).style('text-anchor', 'middle')
        .style('font-size', '13px').style('font-weight', '600')
        .text(`${divLabel} = ${fmt(divValue)}`);

      // Title
      svg.append('text').attr('x', leftWidth / 2).attr('y', 16)
        .style('fill', 'var(--color-text-primary)').style('text-anchor', 'middle')
        .style('font-size', '13px').style('font-weight', '600')
        .text('Parameter Space');
    },
    [leftWidth, mu1, sig1, mu2, sig2, divValue, divLabel]
  );

  // ── Right panel: PDF overlay ─────────────────────────────────────
  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (rightWidth <= 0) return;

      const margin = { top: 30, right: 20, bottom: 40, left: 50 };
      const w = rightWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xRange: [number, number] = [
        Math.min(mu1, mu2) - 3 * Math.max(sig1, sig2),
        Math.max(mu1, mu2) + 3 * Math.max(sig1, sig2),
      ];
      const xScale = d3.scaleLinear().domain(xRange).range([0, w]);

      const nPts = 200;
      const xs = Array.from({ length: nPts }, (_, i) => xRange[0] + (xRange[1] - xRange[0]) * i / (nPts - 1));
      const p1 = xs.map((x) => gaussianPdf(x, mu1, sig1));
      const p2 = xs.map((x) => gaussianPdf(x, mu2, sig2));
      const yMax = Math.max(d3.max(p1) || 1, d3.max(p2) || 1) * 1.1;
      const yScale = d3.scaleLinear().domain([0, yMax]).range([h, 0]);

      // Axes
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(6))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(yScale).ticks(4))
        .selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', w / 2).attr('y', h + 35)
        .style('fill', 'var(--color-text-secondary)').style('text-anchor', 'middle').style('font-size', '12px').text('x');

      // Area between PDFs (shaded)
      const area = d3.area<number>()
        .x((_, i) => xScale(xs[i]))
        .y0((_, i) => yScale(Math.min(p1[i], p2[i])))
        .y1((_, i) => yScale(Math.max(p1[i], p2[i])));

      g.append('path')
        .datum(d3.range(nPts))
        .attr('d', area)
        .style('fill', AMBER)
        .style('opacity', 0.12);

      // PDF p
      const line1 = d3.line<number>()
        .x((_, i) => xScale(xs[i]))
        .y((d) => yScale(d));
      g.append('path').datum(p1).attr('d', line1)
        .style('fill', 'none').style('stroke', TEAL).style('stroke-width', 2.5);

      // PDF q
      g.append('path').datum(p2).attr('d', line1)
        .style('fill', 'none').style('stroke', PURPLE).style('stroke-width', 2.5);

      // Legend
      g.append('text').attr('x', w - 5).attr('y', 10).style('fill', TEAL)
        .style('text-anchor', 'end').style('font-size', '11px')
        .text(`p = N(${fmt2(mu1)}, ${fmt2(sig1)}²)`);
      g.append('text').attr('x', w - 5).attr('y', 24).style('fill', PURPLE)
        .style('text-anchor', 'end').style('font-size', '11px')
        .text(`q = N(${fmt2(mu2)}, ${fmt2(sig2)}²)`);

      // Asymmetry note for KL
      if (divType === 'kl-forward' || divType === 'kl-reverse') {
        const other = divType === 'kl-forward'
          ? klDivGaussian(mu2, sig2, mu1, sig1)
          : klDivGaussian(mu1, sig1, mu2, sig2);
        const otherLabel = divType === 'kl-forward' ? 'D_KL(q ‖ p)' : 'D_KL(p ‖ q)';
        g.append('text').attr('x', w / 2).attr('y', h - 10)
          .style('fill', 'var(--color-text-tertiary)').style('text-anchor', 'middle').style('font-size', '10px')
          .text(`${otherLabel} = ${fmt(other)} (asymmetric!)`);
      }

      // Title
      svg.append('text').attr('x', rightWidth / 2).attr('y', 16)
        .style('fill', 'var(--color-text-primary)').style('text-anchor', 'middle')
        .style('font-size', '13px').style('font-weight', '600')
        .text('Probability Density Functions');
    },
    [rightWidth, mu1, sig1, mu2, sig2, divType]
  );

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-3">
      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-1`}>
        <svg ref={leftRef} width={leftWidth} height={HEIGHT} />
        <svg ref={rightRef} width={rightWidth} height={HEIGHT} />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-4 text-sm">
        <span className="text-[var(--color-text-secondary)]">Divergence:</span>
        {(['kl-forward', 'kl-reverse', 'hellinger', 'fisher-rao'] as const).map((dt) => (
          <label key={dt} className="flex items-center gap-1 text-[var(--color-text-secondary)]">
            <input
              type="radio"
              name="divType"
              checked={divType === dt}
              onChange={() => setDivType(dt)}
              className="accent-[var(--color-accent)]"
            />
            {dt === 'kl-forward' ? 'KL(p‖q)' : dt === 'kl-reverse' ? 'KL(q‖p)' : dt === 'hellinger' ? 'Hellinger' : 'Fisher-Rao'}
          </label>
        ))}
      </div>

      <p className="mt-2 text-xs text-[var(--color-text-tertiary)]">
        Drag the two points to compare divergences between Gaussians. KL divergence is asymmetric — swapping p and q changes the value. The Fisher-Rao distance is a true metric (symmetric, triangle inequality).
      </p>
    </div>
  );
}
