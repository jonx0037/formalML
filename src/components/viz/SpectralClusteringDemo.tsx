import { useState, useMemo, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  moonsPoints, trueLabels, kmeansLabels, spectralLabels,
  laplacianEigenvalues, spectralEmbedding,
  spectralAccuracy, kmeansAccuracy,
  SIGMA,
} from '../../data/spectral-clustering-data';

const STEP_LABELS = ['Data', 'Similarity', 'Laplacian', 'Embedding', 'Result'];

const STEP_DESCRIPTIONS = [
  '100 points forming two interleaving crescents. K-means fails on non-convex shapes like these.',
  'Connect nearby points with edge weights from a Gaussian kernel (σ = 0.15).',
  'The spectral gap after λ₁ reveals 2 clusters. The Spectral Theorem guarantees these eigenvalues are real and non-negative.',
  'Plotting eigenvectors v₂ and v₃ as coordinates. The non-convex clusters become linearly separable.',
  `K-means on the spectral embedding perfectly separates the moons (accuracy: ${(spectralAccuracy * 100).toFixed(0)}%).`,
];

const STEP_TITLES = [
  'Two Moons Dataset',
  'Similarity Graph',
  'Laplacian Eigenvalues',
  'Spectral Embedding',
  'Spectral Clustering Result',
];

const COLORS = ['#2171b5', '#d94801'] as const;
const MARGIN = { top: 24, right: 20, bottom: 40, left: 44 };
const PANEL_HEIGHT = 350;
const SIM_THRESHOLD = 0.01;
const EDGE_DIST = 0.5;

function gaussian(d: number, sigma: number): number {
  return Math.exp(-(d * d) / (2 * sigma * sigma));
}

export default function SpectralClusteringDemo() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const [step, setStep] = useState(0);

  const panelWidth = useMemo(() => {
    if (!containerWidth) return 500;
    return Math.min(containerWidth - 16, 640);
  }, [containerWidth]);

  const innerW = panelWidth - MARGIN.left - MARGIN.right;
  const innerH = PANEL_HEIGHT - MARGIN.top - MARGIN.bottom;

  // Pre-compute similarity edges once
  const edges = useMemo(() => {
    const result: { i: number; j: number; sim: number }[] = [];
    for (let i = 0; i < moonsPoints.length; i++) {
      for (let j = i + 1; j < moonsPoints.length; j++) {
        const dx = moonsPoints[i].x - moonsPoints[j].x;
        const dy = moonsPoints[i].y - moonsPoints[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist <= EDGE_DIST) {
          const sim = gaussian(dist, SIGMA);
          if (sim > SIM_THRESHOLD) {
            result.push({ i, j, sim });
          }
        }
      }
    }
    return result;
  }, []);

  // Scatter extents for original points
  const xExtent = useMemo(
    () => d3.extent(moonsPoints, (p) => p.x) as [number, number],
    [],
  );
  const yExtent = useMemo(
    () => d3.extent(moonsPoints, (p) => p.y) as [number, number],
    [],
  );

  // Embedding extents
  const embXExtent = useMemo(
    () => d3.extent(spectralEmbedding, (p) => p.v2) as [number, number],
    [],
  );
  const embYExtent = useMemo(
    () => d3.extent(spectralEmbedding, (p) => p.v3) as [number, number],
    [],
  );

  useEffect(() => {
    if (!svgRef.current || panelWidth === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g');
    const pad = 0.15;

    // Title
    svg
      .append('text')
      .attr('x', panelWidth / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-family', 'var(--font-sans)')
      .attr('font-size', 13)
      .attr('font-weight', 600)
      .text(STEP_TITLES[step]);

    if (step === 2) {
      // ── Bar chart of eigenvalues ──
      const barData = laplacianEigenvalues;
      const xScale = d3
        .scaleBand<number>()
        .domain(barData.map((_, i) => i))
        .range([MARGIN.left, MARGIN.left + innerW])
        .padding(0.2);

      const yMax = d3.max(barData) as number;
      const yScale = d3
        .scaleLinear()
        .domain([0, yMax * 1.1])
        .range([MARGIN.top + innerH, MARGIN.top]);

      // Bars
      g.selectAll('.bar')
        .data(barData)
        .join('rect')
        .attr('class', 'bar')
        .attr('x', (_, i) => xScale(i)!)
        .attr('y', (d) => yScale(d))
        .attr('width', xScale.bandwidth())
        .attr('height', (d) => MARGIN.top + innerH - yScale(d))
        .attr('fill', (_, i) => (i === 0 ? '#999' : COLORS[0]))
        .attr('fill-opacity', 0.8)
        .attr('rx', 2);

      // Spectral gap annotation: bracket between bar 1 and bar 2
      const x1 = xScale(1)! + xScale.bandwidth() / 2;
      const x2 = xScale(2)! + xScale.bandwidth() / 2;
      const y1 = yScale(barData[1]);
      const y2 = yScale(barData[2]);
      const bracketX = (x1 + x2) / 2;
      const bracketTop = Math.min(y1, y2) - 16;

      g.append('line')
        .attr('x1', x1).attr('y1', y1 - 4)
        .attr('x2', x1).attr('y2', bracketTop)
        .style('stroke', 'var(--color-text)')
        .attr('stroke-width', 1.5)
        .attr('stroke-opacity', 0.6);
      g.append('line')
        .attr('x1', x1).attr('y1', bracketTop)
        .attr('x2', x2).attr('y2', bracketTop)
        .style('stroke', 'var(--color-text)')
        .attr('stroke-width', 1.5)
        .attr('stroke-opacity', 0.6);
      g.append('line')
        .attr('x1', x2).attr('y1', bracketTop)
        .attr('x2', x2).attr('y2', y2 - 4)
        .style('stroke', 'var(--color-text)')
        .attr('stroke-width', 1.5)
        .attr('stroke-opacity', 0.6);
      g.append('text')
        .attr('x', bracketX)
        .attr('y', bracketTop - 6)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 10)
        .attr('font-weight', 600)
        .text('spectral gap');

      // X axis labels
      g.selectAll('.xlabel')
        .data(barData)
        .join('text')
        .attr('class', 'xlabel')
        .attr('x', (_, i) => xScale(i)! + xScale.bandwidth() / 2)
        .attr('y', MARGIN.top + innerH + 16)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 10)
        .attr('opacity', 0.7)
        .text((_, i) => `λ${i}`);

      // Y axis
      const yAxis = d3.axisLeft(yScale).ticks(5).tickSize(-innerW);
      const yAxisG = g
        .append('g')
        .attr('transform', `translate(${MARGIN.left}, 0)`)
        .call(yAxis);
      yAxisG.selectAll('line').style('stroke', 'var(--color-border)').attr('stroke-opacity', 0.3);
      yAxisG.selectAll('path').style('stroke', 'none');
      yAxisG.selectAll('text')
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-sans)')
        .attr('font-size', 9)
        .attr('opacity', 0.6);
    } else {
      // ── Scatter plot steps: 0, 1, 3, 4 ──
      let xDomain: [number, number];
      let yDomain: [number, number];
      let xLabel = '';
      let yLabel = '';

      if (step === 3) {
        // Spectral embedding
        xDomain = [embXExtent[0] - pad, embXExtent[1] + pad];
        yDomain = [embYExtent[0] - pad, embYExtent[1] + pad];
        xLabel = 'v₂';
        yLabel = 'v₃';
      } else {
        // Original space
        xDomain = [xExtent[0] - pad, xExtent[1] + pad];
        yDomain = [yExtent[0] - pad, yExtent[1] + pad];
      }

      const xScale = d3.scaleLinear().domain(xDomain).range([MARGIN.left, MARGIN.left + innerW]);
      const yScale = d3.scaleLinear().domain(yDomain).range([MARGIN.top + innerH, MARGIN.top]);

      // Similarity edges (step 1 only)
      if (step === 1) {
        g.selectAll('.edge')
          .data(edges)
          .join('line')
          .attr('class', 'edge')
          .attr('x1', (e) => xScale(moonsPoints[e.i].x))
          .attr('y1', (e) => yScale(moonsPoints[e.i].y))
          .attr('x2', (e) => xScale(moonsPoints[e.j].x))
          .attr('y2', (e) => yScale(moonsPoints[e.j].y))
          .style('stroke', 'var(--color-text)')
          .attr('stroke-opacity', (e) => Math.min(e.sim * 0.8, 0.6))
          .attr('stroke-width', (e) => 0.5 + e.sim * 0.5);
      }

      // Choose labels and data for scatter
      let labels: number[];
      let pts: { x: number; y: number }[];

      if (step === 3) {
        pts = spectralEmbedding.map((p) => ({ x: p.v2, y: p.v3 }));
        labels = spectralLabels;
      } else if (step === 4) {
        pts = moonsPoints;
        labels = spectralLabels;
      } else {
        pts = moonsPoints;
        labels = trueLabels;
      }

      // Points
      g.selectAll('.point')
        .data(pts)
        .join('circle')
        .attr('class', 'point')
        .attr('cx', (p) => xScale(p.x))
        .attr('cy', (p) => yScale(p.y))
        .attr('r', 4)
        .attr('fill', (_, i) => COLORS[labels[i]])
        .attr('fill-opacity', 0.85)
        .style('stroke', 'var(--color-surface)')
        .attr('stroke-width', 1);

      // Axis labels for embedding
      if (step === 3) {
        g.append('text')
          .attr('x', MARGIN.left + innerW / 2)
          .attr('y', MARGIN.top + innerH + 32)
          .attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text)')
          .style('font-family', 'var(--font-sans)')
          .attr('font-size', 11)
          .attr('opacity', 0.6)
          .text(xLabel);

        g.append('text')
          .attr('x', 14)
          .attr('y', MARGIN.top + innerH / 2)
          .attr('text-anchor', 'middle')
          .attr('transform', `rotate(-90, 14, ${MARGIN.top + innerH / 2})`)
          .style('fill', 'var(--color-text)')
          .style('font-family', 'var(--font-sans)')
          .attr('font-size', 11)
          .attr('opacity', 0.6)
          .text(yLabel);
      }

      // K-means accuracy note on step 0
      if (step === 0) {
        svg
          .append('text')
          .attr('x', panelWidth / 2)
          .attr('y', MARGIN.top + innerH + 32)
          .attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text)')
          .style('font-family', 'var(--font-sans)')
          .attr('font-size', 10)
          .attr('opacity', 0.5)
          .text(`K-means accuracy on this dataset: ${(kmeansAccuracy * 100).toFixed(0)}%`);
      }

      // Result accuracy note on step 4
      if (step === 4) {
        svg
          .append('text')
          .attr('x', panelWidth / 2)
          .attr('y', MARGIN.top + innerH + 32)
          .attr('text-anchor', 'middle')
          .style('fill', 'var(--color-text)')
          .style('font-family', 'var(--font-sans)')
          .attr('font-size', 10)
          .attr('opacity', 0.5)
          .text(`Spectral clustering accuracy: ${(spectralAccuracy * 100).toFixed(0)}%`);
      }
    }
  }, [step, panelWidth, innerW, innerH, edges, xExtent, yExtent, embXExtent, embYExtent]);

  return (
    <div ref={containerRef} className="w-full space-y-3">
      {/* Step navigation */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => setStep((s) => Math.max(0, s - 1))}
          disabled={step === 0}
          className="rounded-md px-2.5 py-1 text-xs font-medium transition-colors disabled:opacity-30"
          style={{
            fontFamily: 'var(--font-sans)',
            background: 'var(--color-surface)',
            color: 'var(--color-text)',
            border: '1px solid var(--color-border)',
          }}
        >
          Previous
        </button>

        <div className="flex flex-wrap gap-1.5">
          {STEP_LABELS.map((label, i) => (
            <button
              key={i}
              onClick={() => setStep(i)}
              className="rounded-md px-2.5 py-1 text-xs font-medium transition-colors"
              style={{
                fontFamily: 'var(--font-sans)',
                background: i === step ? 'var(--color-accent)' : 'var(--color-surface)',
                color: i === step ? '#fff' : 'var(--color-text)',
                border: `1px solid ${i === step ? 'var(--color-accent)' : 'var(--color-border)'}`,
                opacity: i <= step ? 1 : 0.5,
              }}
            >
              {label}
            </button>
          ))}
        </div>

        <button
          onClick={() => setStep((s) => Math.min(STEP_LABELS.length - 1, s + 1))}
          disabled={step === STEP_LABELS.length - 1}
          className="rounded-md px-2.5 py-1 text-xs font-medium transition-colors disabled:opacity-30"
          style={{
            fontFamily: 'var(--font-sans)',
            background: 'var(--color-surface)',
            color: 'var(--color-text)',
            border: '1px solid var(--color-border)',
          }}
        >
          Next
        </button>
      </div>

      {/* Visualization */}
      <svg
        ref={svgRef}
        width={panelWidth}
        height={PANEL_HEIGHT}
        className="rounded-lg border border-[var(--color-border)]"
      />

      {/* Step description */}
      <p
        className="text-sm opacity-70"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        {STEP_DESCRIPTIONS[step]}
      </p>
    </div>
  );
}
