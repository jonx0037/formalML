import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import {
  fisherMetricGaussian,
  fisherMetricBernoulli,
  fisherMetricExponential,
  metricEigendecomp,
} from './shared/manifoldGeometry';

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 380;
const SM_BREAKPOINT = 640;

const TEAL = dimensionColors[0];
const PURPLE = dimensionColors[1];
const AMBER = '#D97706';

const GAUSSIAN_MU_RANGE: [number, number] = [-3, 3];
const GAUSSIAN_SIG_RANGE: [number, number] = [0, 3];
const GAUSSIAN_SIG_PAD = 0.15;
const BERNOULLI_DOMAIN: [number, number] = [0.02, 0.98];
const EXPONENTIAL_DOMAIN: [number, number] = [0.1, 5];
const ELLIPSE_MU_RANGE: [number, number] = [-2.5, 2.5];
const ELLIPSE_SIG_RANGE: [number, number] = [0.4, 2.8];
const ELLIPSE_SCALE = 0.15;
const CURRENT_ELLIPSE_SCALE = 0.25;

type Family = 'gaussian' | 'bernoulli' | 'exponential';

const fmt = (x: number) => x.toFixed(3);

// ── Component ────────────────────────────────────────────────────────

export default function FisherMetricExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [family, setFamily] = useState<Family>('gaussian');
  const [paramX, setParamX] = useState(0);      // μ for Gaussian, p for Bernoulli, λ for Exponential
  const [paramY, setParamY] = useState(1);       // σ for Gaussian (unused for 1D families)
  const [showEllipses, setShowEllipses] = useState(false);
  const [showScore, setShowScore] = useState(false);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const svgWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.58);
  const rightWidth = isStacked ? containerWidth : containerWidth - svgWidth;

  // ── Metric data ──────────────────────────────────────────────────
  const metricData = useMemo(() => {
    if (family === 'gaussian') {
      const metric = fisherMetricGaussian(paramY);
      const eigen = metricEigendecomp(metric.g);
      return { metric, eigen, dim: 2 as const };
    } else if (family === 'bernoulli') {
      const g = fisherMetricBernoulli(paramX);
      return { scalar: g, dim: 1 as const };
    } else {
      const g = fisherMetricExponential(paramX);
      return { scalar: g, dim: 1 as const };
    }
  }, [family, paramX, paramY]);

  // ── Ellipse grid for Gaussian ────────────────────────────────────
  const ellipseGrid = useMemo(() => {
    if (!showEllipses || family !== 'gaussian') return [];
    const grid: { cx: number; cy: number; rx: number; ry: number }[] = [];
    for (let mu = ELLIPSE_MU_RANGE[0]; mu <= ELLIPSE_MU_RANGE[1]; mu += 1) {
      for (let sig = ELLIPSE_SIG_RANGE[0]; sig <= ELLIPSE_SIG_RANGE[1]; sig += 0.4) {
        const m = fisherMetricGaussian(sig);
        const e = metricEigendecomp(m.g);
        // Semi-axes proportional to 1/sqrt(eigenvalue) — the "unit ball" in the metric
        grid.push({
          cx: mu,
          cy: sig,
          rx: ELLIPSE_SCALE / Math.sqrt(e.eigenvalues[0]),
          ry: ELLIPSE_SCALE / Math.sqrt(e.eigenvalues[1]),
        });
      }
    }
    return grid;
  }, [showEllipses, family]);

  // ── Score function data ──────────────────────────────────────────
  const scoreData = useMemo(() => {
    if (!showScore) return null;
    const xs = Array.from({ length: 200 }, (_, i) => -4 + (8 * i) / 199);
    if (family === 'gaussian') {
      const s2 = paramY * paramY;
      return {
        xs,
        scoreMu: xs.map((x) => (x - paramX) / s2),
        scoreSigma: xs.map((x) => -1 / paramY + (x - paramX) ** 2 / (paramY * s2)),
        labels: ['s_μ', 's_σ'],
      };
    } else if (family === 'bernoulli') {
      const p = Math.max(0.01, Math.min(0.99, paramX));
      // Score for Bernoulli: s_p(x) = x/p - (1-x)/(1-p)
      return {
        xs: [0, 1],
        scoreMu: [
          -1 / (1 - p),  // s_p(x=0) = 0/p - 1/(1-p)
          1 / p,          // s_p(x=1) = 1/p - 0/(1-p)
        ],
        scoreSigma: null,
        labels: ['s_p(x=0)', 's_p(x=1)'],
      };
    }
    return null;
  }, [showScore, family, paramX, paramY]);

  // ── Left panel: parameter space ──────────────────────────────────
  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (svgWidth <= 0) return;

      const margin = { top: 30, right: 20, bottom: 40, left: 50 };
      const w = svgWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      if (family === 'gaussian') {
        const xScale = d3.scaleLinear().domain(GAUSSIAN_MU_RANGE).range([0, w]);
        const yScale = d3.scaleLinear().domain(GAUSSIAN_SIG_RANGE).range([h, 0]);

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

        // Ellipse grid
        ellipseGrid.forEach((e) => {
          g.append('ellipse')
            .attr('cx', xScale(e.cx))
            .attr('cy', yScale(e.cy))
            .attr('rx', Math.abs(xScale(e.cx + e.rx) - xScale(e.cx)))
            .attr('ry', Math.abs(yScale(e.cy + e.ry) - yScale(e.cy)))
            .style('fill', 'none')
            .style('stroke', PURPLE)
            .style('stroke-width', 0.8)
            .style('opacity', 0.4);
        });

        // Current point ellipse (highlighted)
        if (metricData.dim === 2) {
          const eigen = metricData.eigen;
          const rx = CURRENT_ELLIPSE_SCALE / Math.sqrt(eigen.eigenvalues[0]);
          const ry = CURRENT_ELLIPSE_SCALE / Math.sqrt(eigen.eigenvalues[1]);
          g.append('ellipse')
            .attr('cx', xScale(paramX))
            .attr('cy', yScale(paramY))
            .attr('rx', Math.abs(xScale(paramX + rx) - xScale(paramX)))
            .attr('ry', Math.abs(yScale(paramY + ry) - yScale(paramY)))
            .style('fill', TEAL)
            .style('fill-opacity', 0.15)
            .style('stroke', TEAL)
            .style('stroke-width', 2);
        }

        // Draggable point
        const dot = g.append('circle')
          .attr('cx', xScale(paramX))
          .attr('cy', yScale(paramY))
          .attr('r', 7)
          .style('fill', TEAL)
          .style('stroke', '#fff')
          .style('stroke-width', 2)
          .style('cursor', 'grab');

        const drag = d3.drag<SVGCircleElement, unknown>()
          .on('drag', (event) => {
            const mu = xScale.invert(event.x);
            const sig = yScale.invert(event.y);
            setParamX(Math.max(GAUSSIAN_MU_RANGE[0], Math.min(GAUSSIAN_MU_RANGE[1], mu)));
            setParamY(Math.max(GAUSSIAN_SIG_RANGE[0] + GAUSSIAN_SIG_PAD, Math.min(GAUSSIAN_SIG_RANGE[1] - 0.1, sig)));
          });
        dot.call(drag);

      } else {
        // 1D families: Bernoulli or Exponential
        const domain: [number, number] = family === 'bernoulli' ? BERNOULLI_DOMAIN : EXPONENTIAL_DOMAIN;
        const xScale = d3.scaleLinear().domain(domain).range([0, w]);
        const yScale = d3.scaleLinear().domain([0, family === 'bernoulli' ? 30 : 120]).range([h, 0]);

        g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(6))
          .selectAll('text').style('fill', 'var(--color-text-secondary)');
        g.append('g').call(d3.axisLeft(yScale).ticks(6))
          .selectAll('text').style('fill', 'var(--color-text-secondary)');
        g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');

        const label = family === 'bernoulli' ? 'p' : 'λ';
        g.append('text').attr('x', w / 2).attr('y', h + 35)
          .style('fill', 'var(--color-text-secondary)').style('text-anchor', 'middle').style('font-size', '12px').text(label);
        g.append('text').attr('x', -35).attr('y', h / 2)
          .style('fill', 'var(--color-text-secondary)').style('text-anchor', 'middle')
          .style('font-size', '12px').attr('transform', `rotate(-90,-35,${h / 2})`).text('g(θ)');

        // Fisher info curve
        const nPts = 200;
        const fisherFn = family === 'bernoulli' ? fisherMetricBernoulli : fisherMetricExponential;
        const line = d3.line<number>()
          .x((_, i) => xScale(domain[0] + (domain[1] - domain[0]) * i / (nPts - 1)))
          .y((_, i) => {
            const t = domain[0] + (domain[1] - domain[0]) * i / (nPts - 1);
            return yScale(Math.min(fisherFn(t), yScale.domain()[1]));
          });

        g.append('path')
          .datum(d3.range(nPts))
          .attr('d', line)
          .style('fill', 'none')
          .style('stroke', PURPLE)
          .style('stroke-width', 2.5);

        // Current point
        const gVal = fisherFn(paramX);
        g.append('circle')
          .attr('cx', xScale(paramX))
          .attr('cy', yScale(Math.min(gVal, yScale.domain()[1])))
          .attr('r', 7)
          .style('fill', TEAL)
          .style('stroke', '#fff')
          .style('stroke-width', 2)
          .style('cursor', 'grab');

        const dotSel = g.select<SVGCircleElement>('circle');
        const drag = d3.drag<SVGCircleElement, unknown>()
          .on('drag', (event) => {
            const t = xScale.invert(event.x);
            setParamX(Math.max(domain[0], Math.min(domain[1], t)));
          });
        dotSel.call(drag);
      }

      // Title
      svg.append('text')
        .attr('x', svgWidth / 2).attr('y', 16)
        .style('fill', 'var(--color-text-primary)').style('text-anchor', 'middle')
        .style('font-size', '13px').style('font-weight', '600')
        .text(family === 'gaussian' ? 'Parameter Space (μ, σ)' :
              family === 'bernoulli' ? 'Fisher Information g(p)' : 'Fisher Information g(λ)');
    },
    [family, paramX, paramY, showEllipses, ellipseGrid, metricData, svgWidth]
  );

  // ── Right panel: score / eigenvalues ─────────────────────────────
  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (rightWidth <= 0) return;

      const margin = { top: 30, right: 20, bottom: 40, left: 50 };
      const w = rightWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      if (showScore && scoreData && family === 'gaussian') {
        // Plot score functions
        const xScale = d3.scaleLinear().domain([-4, 4]).range([0, w]);
        const allVals = [...scoreData.scoreMu, ...scoreData.scoreSigma!];
        const yMax = Math.min(10, d3.max(allVals.map(Math.abs)) || 5);
        const yScale = d3.scaleLinear().domain([-yMax, yMax]).range([h, 0]);

        g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(6))
          .selectAll('text').style('fill', 'var(--color-text-secondary)');
        g.append('g').call(d3.axisLeft(yScale).ticks(6))
          .selectAll('text').style('fill', 'var(--color-text-secondary)');
        g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
        g.append('text').attr('x', w / 2).attr('y', h + 35)
          .style('fill', 'var(--color-text-secondary)').style('text-anchor', 'middle').style('font-size', '12px').text('x');

        // Score mu
        const lineMu = d3.line<number>()
          .x((_, i) => xScale(scoreData.xs[i]))
          .y((d) => yScale(Math.max(-yMax, Math.min(yMax, d))));
        g.append('path').datum(scoreData.scoreMu).attr('d', lineMu)
          .style('fill', 'none').style('stroke', TEAL).style('stroke-width', 2.5);

        // Score sigma
        const lineSig = d3.line<number>()
          .x((_, i) => xScale(scoreData.xs[i]))
          .y((d) => yScale(Math.max(-yMax, Math.min(yMax, d))));
        g.append('path').datum(scoreData.scoreSigma!).attr('d', lineSig)
          .style('fill', 'none').style('stroke', PURPLE).style('stroke-width', 2.5);

        // Legend
        g.append('text').attr('x', w - 5).attr('y', 10).style('fill', TEAL)
          .style('text-anchor', 'end').style('font-size', '11px').text('s_μ(x; θ)');
        g.append('text').attr('x', w - 5).attr('y', 24).style('fill', PURPLE)
          .style('text-anchor', 'end').style('font-size', '11px').text('sσ(x; θ)');

        // Zero line
        g.append('line').attr('x1', 0).attr('x2', w)
          .attr('y1', yScale(0)).attr('y2', yScale(0))
          .style('stroke', 'var(--color-border)').style('stroke-dasharray', '4,3');

        svg.append('text').attr('x', rightWidth / 2).attr('y', 16)
          .style('fill', 'var(--color-text-primary)').style('text-anchor', 'middle')
          .style('font-size', '13px').style('font-weight', '600').text('Score Functions');

      } else {
        // Eigenvalue / metric info display
        const yMid = h / 2;

        if (family === 'gaussian' && metricData.dim === 2) {
          const eigen = metricData.eigen;
          const lines = [
            `Fisher Metric at (μ=${fmt(paramX)}, σ=${fmt(paramY)})`,
            '',
            `g₁₁ = 1/σ² = ${fmt(metricData.metric.g[0][0])}`,
            `g₂₂ = 2/σ² = ${fmt(metricData.metric.g[1][1])}`,
            `g₁₂ = 0`,
            '',
            `Eigenvalues: λ₁ = ${fmt(eigen.eigenvalues[0])}, λ₂ = ${fmt(eigen.eigenvalues[1])}`,
            `det(g) = ${fmt(metricData.metric.det)}`,
            '',
            `Curvature: K = -1/2`,
          ];
          lines.forEach((line, i) => {
            g.append('text')
              .attr('x', w / 2).attr('y', yMid - (lines.length / 2 - i) * 20)
              .style('fill', i === 0 ? 'var(--color-text-primary)' : 'var(--color-text-secondary)')
              .style('text-anchor', 'middle')
              .style('font-size', i === 0 ? '13px' : '12px')
              .style('font-weight', i === 0 ? '600' : '400')
              .style('font-family', i >= 2 && line ? 'var(--font-mono, monospace)' : 'inherit')
              .text(line);
          });
        } else {
          const gVal = family === 'bernoulli'
            ? fisherMetricBernoulli(paramX)
            : fisherMetricExponential(paramX);
          const paramLabel = family === 'bernoulli' ? `p = ${fmt(paramX)}` : `λ = ${fmt(paramX)}`;
          const formula = family === 'bernoulli' ? '1/(p(1-p))' : '1/λ²';
          const lines = [
            `Fisher Information at ${paramLabel}`,
            '',
            `g(θ) = ${formula} = ${fmt(gVal)}`,
            '',
            `Cramér-Rao bound:`,
            `Var(T) ≥ 1/g = ${fmt(1 / gVal)}`,
          ];
          lines.forEach((line, i) => {
            g.append('text')
              .attr('x', w / 2).attr('y', yMid - (lines.length / 2 - i) * 22)
              .style('fill', i === 0 ? 'var(--color-text-primary)' : 'var(--color-text-secondary)')
              .style('text-anchor', 'middle')
              .style('font-size', i === 0 ? '13px' : '12px')
              .style('font-weight', i === 0 ? '600' : '400')
              .style('font-family', i >= 2 && line ? 'var(--font-mono, monospace)' : 'inherit')
              .text(line);
          });
        }
        svg.append('text').attr('x', rightWidth / 2).attr('y', 16)
          .style('fill', 'var(--color-text-primary)').style('text-anchor', 'middle')
          .style('font-size', '13px').style('font-weight', '600')
          .text(showScore && scoreData && family === 'gaussian' ? 'Score Functions' : 'Metric Information');
      }
    },
    [family, paramX, paramY, showScore, scoreData, metricData, rightWidth]
  );

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-3">
      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-1`}>
        <svg ref={svgRef} width={svgWidth} height={HEIGHT} />
        <svg ref={rightRef} width={rightWidth} height={HEIGHT} />
      </div>

      {/* Controls */}
      <div className="mt-3 flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-1.5 text-[var(--color-text-secondary)]">
          Family:
          <select
            value={family}
            onChange={(e) => {
              const f = e.target.value as Family;
              setFamily(f);
              if (f === 'gaussian') { setParamX(0); setParamY(1); }
              else if (f === 'bernoulli') { setParamX(0.5); setShowScore(false); }
              else { setParamX(1); setShowScore(false); }
            }}
            className="rounded border border-[var(--color-border)] bg-[var(--color-bg-primary)] px-2 py-0.5 text-[var(--color-text-primary)]"
          >
            <option value="gaussian">Gaussian N(μ, σ²)</option>
            <option value="bernoulli">Bernoulli Ber(p)</option>
            <option value="exponential">Exponential Exp(λ)</option>
          </select>
        </label>

        {family === 'gaussian' && (
          <label className="flex items-center gap-1.5 text-[var(--color-text-secondary)]">
            <input
              type="checkbox"
              checked={showEllipses}
              onChange={(e) => setShowEllipses(e.target.checked)}
              className="accent-[var(--color-accent)]"
            />
            Show metric ellipses
          </label>
        )}

        {family === 'gaussian' && (
          <label className="flex items-center gap-1.5 text-[var(--color-text-secondary)]">
            <input
              type="checkbox"
              checked={showScore}
              onChange={(e) => setShowScore(e.target.checked)}
              className="accent-[var(--color-accent)]"
            />
            Show score functions
          </label>
        )}
      </div>

      <p className="mt-2 text-xs text-[var(--color-text-tertiary)]">
        Drag the point to explore how the Fisher metric varies across parameter space.
        {family === 'gaussian' && ' Metric ellipses show the local unit ball in the Fisher metric — smaller ellipses mean higher information.'}
      </p>
    </div>
  );
}
