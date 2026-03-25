import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import {
  fisherMetricGaussian,
  klDivGaussian,
  klDivGradient,
  metricEigendecomp,
} from './shared/manifoldGeometry';

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 420;

const TEAL = dimensionColors[0];
const PURPLE = dimensionColors[1];
const AMBER = '#D97706';

interface Target {
  label: string;
  mu0: number;
  sigma0: number;
}

const TARGETS: Target[] = [
  { label: 'N(0, 1)', mu0: 0, sigma0: 1 },
  { label: 'N(1, 0.5)', mu0: 1, sigma0: 0.5 },
  { label: 'N(-1, 2)', mu0: -1, sigma0: 2 },
];

interface TrajectoryPoint {
  mu: number;
  sigma: number;
}

function runGradientDescent(
  mu0: number,
  sigma0: number,
  target: Target,
  lr: number,
  natural: boolean,
  nSteps = 80,
): TrajectoryPoint[] {
  const pts: TrajectoryPoint[] = [{ mu: mu0, sigma: sigma0 }];
  let mu = mu0;
  let sigma = sigma0;

  for (let i = 0; i < nSteps; i++) {
    const [dMu, dSigma] = klDivGradient(mu, sigma, target.mu0, target.sigma0);

    if (natural) {
      const metric = fisherMetricGaussian(sigma);
      // Natural gradient = g^{-1} * Euclidean gradient
      const natMu = metric.inv[0][0] * dMu;
      const natSigma = metric.inv[1][1] * dSigma;
      mu -= lr * natMu;
      sigma -= lr * natSigma;
    } else {
      mu -= lr * dMu;
      sigma -= lr * dSigma;
    }

    sigma = Math.max(sigma, 0.05); // Clamp sigma > 0
    pts.push({ mu, sigma });

    // Early stop if close
    if (Math.abs(mu - target.mu0) < 0.01 && Math.abs(sigma - target.sigma0) < 0.01) break;
  }

  return pts;
}

// ── Component ────────────────────────────────────────────────────────

export default function NaturalGradientExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [targetIdx, setTargetIdx] = useState(0);
  const [lr, setLr] = useState(0.15);
  const [showMetricEllipses, setShowMetricEllipses] = useState(false);
  const [startPoints, setStartPoints] = useState<[number, number][]>([
    [-2, 2.5],
    [2, 0.3],
    [0, 3],
  ]);

  const target = TARGETS[targetIdx];
  const svgWidth = containerWidth > 0 ? containerWidth : 400;

  // ── Compute trajectories ─────────────────────────────────────────
  const trajectories = useMemo(() => {
    return startPoints.map(([mu, sigma]) => ({
      euclidean: runGradientDescent(mu, sigma, target, lr, false),
      natural: runGradientDescent(mu, sigma, target, lr, true),
    }));
  }, [startPoints, target, lr]);

  // ── SVG ──────────────────────────────────────────────────────────
  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      if (svgWidth <= 0) return;
      svg.selectAll('*').remove();

      const w = svgWidth;
      const h = HEIGHT;
      const margin = { top: 25, right: 20, bottom: 35, left: 45 };

      const muRange: [number, number] = [-3.5, 3.5];
      const sigmaRange: [number, number] = [0.1, 4];

      const xScale = d3.scaleLinear().domain(muRange).range([margin.left, w - margin.right]);
      const yScale = d3.scaleLinear().domain(sigmaRange).range([h - margin.bottom, margin.top]);

      // KL divergence contours
      const gridN = 80;
      const klGrid: number[] = [];
      const muStep = (muRange[1] - muRange[0]) / gridN;
      const sigStep = (sigmaRange[1] - sigmaRange[0]) / gridN;

      for (let j = 0; j < gridN; j++) {
        for (let i = 0; i < gridN; i++) {
          const mu = muRange[0] + i * muStep;
          const sigma = sigmaRange[0] + j * sigStep;
          klGrid.push(klDivGaussian(mu, sigma, target.mu0, target.sigma0));
        }
      }

      const contourGen = d3.contours()
        .size([gridN, gridN])
        .thresholds([0.1, 0.3, 0.5, 1, 2, 3, 5, 8, 12]);

      const contourData = contourGen(klGrid);
      const colorScale = d3.scaleSequential(d3.interpolateBlues).domain([0, 12]);

      const contourPath = d3.geoPath().projection(
        d3.geoTransform({
          point: function (px, py) {
            const mu = muRange[0] + (px / gridN) * (muRange[1] - muRange[0]);
            const sigma = sigmaRange[0] + (py / gridN) * (sigmaRange[1] - sigmaRange[0]);
            this.stream.point(xScale(mu), yScale(sigma));
          },
        }),
      );

      svg.selectAll('.contour')
        .data(contourData)
        .join('path')
        .attr('class', 'contour')
        .attr('d', contourPath as any)
        .attr('fill', (d) => colorScale(d.value))
        .style('fill-opacity', '0.15')
        .style('stroke', (d) => colorScale(d.value))
        .style('stroke-width', '0.5')
        .style('stroke-opacity', '0.4');

      // Axes
      svg.append('g')
        .attr('transform', `translate(0, ${h - margin.bottom})`)
        .call(d3.axisBottom(xScale).ticks(7))
        .style('font-size', '10px')
        .style('color', 'var(--color-text-muted, #888)');

      svg.append('g')
        .attr('transform', `translate(${margin.left}, 0)`)
        .call(d3.axisLeft(yScale).ticks(5))
        .style('font-size', '10px')
        .style('color', 'var(--color-text-muted, #888)');

      svg.append('text')
        .attr('x', w / 2).attr('y', h - 3)
        .attr('text-anchor', 'middle')
        .text('μ')
        .style('fill', 'var(--color-text-muted, #888)').style('font-size', '12px');

      svg.append('text')
        .attr('x', 14).attr('y', h / 2)
        .attr('text-anchor', 'middle')
        .attr('transform', `rotate(-90, 14, ${h / 2})`)
        .text('σ')
        .style('fill', 'var(--color-text-muted, #888)').style('font-size', '12px');

      // Fisher metric ellipses
      if (showMetricEllipses) {
        const muPts = d3.range(-3, 3.5, 1);
        const sigPts = d3.range(0.5, 4, 0.7);
        for (const mu of muPts) {
          for (const sigma of sigPts) {
            const metric = fisherMetricGaussian(sigma);
            const { eigenvalues } = metricEigendecomp(metric.g);
            const ellipseScale = 12;
            const rx = ellipseScale / Math.sqrt(eigenvalues[0]);
            const ry = ellipseScale / Math.sqrt(eigenvalues[1]);
            svg.append('ellipse')
              .attr('cx', xScale(mu)).attr('cy', yScale(sigma))
              .attr('rx', Math.min(rx, 30)).attr('ry', Math.min(ry, 30))
              .attr('fill', 'none')
              .style('stroke', AMBER)
              .style('stroke-width', '0.8')
              .style('opacity', '0.5');
          }
        }
      }

      // Target point
      svg.append('circle')
        .attr('cx', xScale(target.mu0)).attr('cy', yScale(target.sigma0))
        .attr('r', 7)
        .attr('fill', 'none')
        .style('stroke', '#ef4444')
        .style('stroke-width', '2.5');
      svg.append('circle')
        .attr('cx', xScale(target.mu0)).attr('cy', yScale(target.sigma0))
        .attr('r', 3)
        .attr('fill', '#ef4444');

      // Trajectories
      const lineGen = d3.line<TrajectoryPoint>()
        .x((d) => xScale(d.mu))
        .y((d) => yScale(d.sigma));

      trajectories.forEach(({ euclidean, natural }, idx) => {
        // Euclidean (dashed)
        svg.append('path')
          .datum(euclidean)
          .attr('d', lineGen)
          .attr('fill', 'none')
          .style('stroke', 'var(--color-text-muted, #999)')
          .style('stroke-width', '1.5')
          .style('stroke-dasharray', '5,3');

        // Natural (solid)
        svg.append('path')
          .datum(natural)
          .attr('d', lineGen)
          .attr('fill', 'none')
          .style('stroke', TEAL)
          .style('stroke-width', '2');

        // Start point
        const [mu, sigma] = [euclidean[0].mu, euclidean[0].sigma];
        svg.append('circle')
          .attr('cx', xScale(mu)).attr('cy', yScale(sigma))
          .attr('r', 4)
          .attr('fill', PURPLE)
          .style('stroke', '#fff')
          .style('stroke-width', '1');
      });

      // Legend
      const legendY = margin.top + 5;
      svg.append('line')
        .attr('x1', w - margin.right - 150).attr('y1', legendY)
        .attr('x2', w - margin.right - 125).attr('y2', legendY)
        .style('stroke', TEAL).style('stroke-width', '2');
      svg.append('text')
        .attr('x', w - margin.right - 120).attr('y', legendY + 4)
        .text('Natural gradient')
        .style('fill', TEAL).style('font-size', '10px');

      svg.append('line')
        .attr('x1', w - margin.right - 150).attr('y1', legendY + 16)
        .attr('x2', w - margin.right - 125).attr('y2', legendY + 16)
        .style('stroke', 'var(--color-text-muted, #999)').style('stroke-width', '1.5').style('stroke-dasharray', '5,3');
      svg.append('text')
        .attr('x', w - margin.right - 120).attr('y', legendY + 20)
        .text('Euclidean gradient')
        .style('fill', 'var(--color-text-muted, #999)').style('font-size', '10px');

      // Click to add start point
      svg.on('click', (event) => {
        const [mx, my] = d3.pointer(event);
        const mu = xScale.invert(mx);
        const sigma = yScale.invert(my);
        if (mu >= muRange[0] && mu <= muRange[1] && sigma >= sigmaRange[0] && sigma <= sigmaRange[1]) {
          setStartPoints((prev) => [...prev, [mu, sigma]]);
        }
      });
    },
    [trajectories, target, showMetricEllipses, svgWidth, lr],
  );

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-[var(--color-border,#e5e7eb)] bg-[var(--color-bg-secondary,#f9fafb)] p-4">
      <div className="mb-3 text-sm font-semibold" style={{ color: 'var(--color-text-primary, #111)' }}>
        Natural Gradient Explorer
      </div>

      <svg ref={svgRef} width={svgWidth} height={HEIGHT} />

      <div className="mt-1 text-xs" style={{ color: 'var(--color-text-muted, #888)' }}>
        Click anywhere on the plot to add a new starting point.
      </div>

      {/* Controls */}
      <div className="mt-2 flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-1.5">
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Target:</span>
          <select
            value={targetIdx}
            onChange={(e) => {
              setTargetIdx(parseInt(e.target.value));
              setStartPoints([[-2, 2.5], [2, 0.3], [0, 3]]);
            }}
            className="rounded border px-1.5 py-0.5 text-xs"
            style={{ borderColor: 'var(--color-border, #d1d5db)', background: 'var(--color-bg-primary, #fff)', color: 'var(--color-text-primary, #111)' }}
          >
            {TARGETS.map((t, i) => (
              <option key={i} value={i}>{t.label}</option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-1.5">
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Learning rate:</span>
          <input
            type="range" min={0.01} max={0.5} step={0.01}
            value={lr} onChange={(e) => setLr(parseFloat(e.target.value))}
            className="w-24"
          />
          <span className="font-mono w-10">{lr.toFixed(2)}</span>
        </label>

        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={showMetricEllipses} onChange={(e) => setShowMetricEllipses(e.target.checked)} />
          <span style={{ color: 'var(--color-text-muted, #666)' }}>Show metric ellipses</span>
        </label>

        <button
          onClick={() => setStartPoints([[-2, 2.5], [2, 0.3], [0, 3]])}
          className="rounded border px-2 py-0.5 text-xs"
          style={{ borderColor: 'var(--color-border, #d1d5db)', background: 'var(--color-bg-primary, #fff)', color: 'var(--color-text-muted, #666)' }}
        >
          Reset points
        </button>
      </div>
    </div>
  );
}
