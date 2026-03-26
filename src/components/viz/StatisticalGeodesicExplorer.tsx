import { useState, useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import {
  solveGeodesicGaussian,
  fisherMetricGaussian,
  klDivGaussian,
  klDivGradient,
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
const SIG_RANGE: [number, number] = [0, 3.5];
const DRAG_PAD = 0.5;
const DRAG_PAD_SIG_LO = 0.2;
const DRAG_PAD_SIG_HI = 0.3;
const GEODESIC_T_MAX = 2.5;
const GEODESIC_INITIAL_SPEED = 0.5;
const KL_CONTOUR_GRID_SIZE = 60;
const KL_CONTOUR_THRESHOLDS = [0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0];
const CONVERGENCE_THRESHOLD = 0.001;
const GRADIENT_LEARNING_RATE = 0.05;
const GRADIENT_STARTS = [
  { mu: -2, sigma: 2.5 },
  { mu: 2, sigma: 0.3 },
  { mu: -1.5, sigma: 0.5 },
  { mu: 1.5, sigma: 2.5 },
];

const fmt = (x: number) => x.toFixed(2);

interface TrajectoryPoint {
  mu: number;
  sigma: number;
}

/** Run gradient descent (Euclidean or natural) on KL divergence */
function runGradientDescent(
  mu0: number, sigma0: number,
  targetMu: number, targetSig: number,
  lr: number, natural: boolean, nSteps = 80
): TrajectoryPoint[] {
  const pts: TrajectoryPoint[] = [{ mu: mu0, sigma: sigma0 }];
  let mu = mu0;
  let sigma = sigma0;

  for (let i = 0; i < nSteps; i++) {
    const [dMu, dSigma] = klDivGradient(mu, sigma, targetMu, targetSig);

    if (natural) {
      const metric = fisherMetricGaussian(sigma);
      mu -= lr * metric.inv[0][0] * dMu;
      sigma -= lr * metric.inv[1][1] * dSigma;
    } else {
      mu -= lr * dMu;
      sigma -= lr * dSigma;
    }

    sigma = Math.max(0.05, sigma);
    pts.push({ mu, sigma });

    // Convergence check
    if (Math.abs(mu - targetMu) < CONVERGENCE_THRESHOLD && Math.abs(sigma - targetSig) < CONVERGENCE_THRESHOLD) break;
  }
  return pts;
}

// ── Component ────────────────────────────────────────────────────────

export default function StatisticalGeodesicExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [startMu, setStartMu] = useState(0);
  const [startSig, setStartSig] = useState(1.5);
  const [showFan, setShowFan] = useState(false);
  const [showKLContours, setShowKLContours] = useState(false);
  const [showNatGrad, setShowNatGrad] = useState(false);
  const [targetMu, setTargetMu] = useState(0);
  const [targetSig, setTargetSig] = useState(1);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const leftWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.58);
  const rightWidth = isStacked ? containerWidth : containerWidth - leftWidth;

  // ── Geodesics from start point ───────────────────────────────────
  const geodesics = useMemo(() => {
    if (showNatGrad) return [];
    const nRays = showFan ? 12 : 1;
    const rays: { x: number; y: number }[][] = [];

    for (let r = 0; r < nRays; r++) {
      const angle = showFan ? (2 * Math.PI * r) / nRays : Math.PI / 4;
      const speed = GEODESIC_INITIAL_SPEED;
      const dmu = speed * Math.cos(angle);
      const dsig = speed * Math.sin(angle);
      const pts = solveGeodesicGaussian(startMu, startSig, dmu, dsig, GEODESIC_T_MAX, 200);
      rays.push(
        pts
          .filter((p) => p.y > 0.05 && p.y < 4 && Math.abs(p.x) < 4)
          .map((p) => ({ x: p.x, y: p.y }))
      );
    }
    return rays;
  }, [startMu, startSig, showFan, showNatGrad]);

  // ── KL contour data ──────────────────────────────────────────────
  const klContours = useMemo(() => {
    if (!showKLContours) return null;
    const nGrid = KL_CONTOUR_GRID_SIZE;
    const muRange: [number, number] = MU_RANGE;
    const sigRange: [number, number] = [SIG_RANGE[0] + DRAG_PAD_SIG_LO, SIG_RANGE[1]];
    const grid: number[] = [];

    for (let j = 0; j < nGrid; j++) {
      for (let i = 0; i < nGrid; i++) {
        const mu = muRange[0] + (muRange[1] - muRange[0]) * i / (nGrid - 1);
        const sig = sigRange[0] + (sigRange[1] - sigRange[0]) * j / (nGrid - 1);
        grid.push(klDivGaussian(startMu, startSig, mu, sig));
      }
    }

    const contourGen = d3.contours().size([nGrid, nGrid])
      .thresholds(KL_CONTOUR_THRESHOLDS);
    return { contours: contourGen(grid), nGrid, muRange, sigRange };
  }, [showKLContours, startMu, startSig]);

  // ── Natural gradient trajectories ────────────────────────────────
  const gradTrajectories = useMemo(() => {
    if (!showNatGrad) return { euclidean: [], natural: [] };

    const starts = GRADIENT_STARTS;
    const lr = GRADIENT_LEARNING_RATE;

    const euclidean = starts.map((s) =>
      runGradientDescent(s.mu, s.sigma, targetMu, targetSig, lr, false)
    );
    const natural = starts.map((s) =>
      runGradientDescent(s.mu, s.sigma, targetMu, targetSig, lr, true)
    );

    return { euclidean, natural };
  }, [showNatGrad, targetMu, targetSig]);

  // ── Distance along geodesic ──────────────────────────────────────
  const distancePlot = useMemo(() => {
    if (showNatGrad || geodesics.length === 0) return null;
    const geo = geodesics[0];
    if (geo.length < 2) return null;

    const dists: { t: number; fr: number; kl: number }[] = [];
    let arcLen = 0;
    for (let i = 0; i < geo.length; i++) {
      if (i > 0) {
        const dx = geo[i].x - geo[i - 1].x;
        const dy = geo[i].y - geo[i - 1].y;
        const sig = (geo[i].y + geo[i - 1].y) / 2;
        arcLen += Math.sqrt((dx * dx + 2 * dy * dy) / (sig * sig));
      }
      const fr = fisherRaoDistanceGaussian(startMu, startSig, geo[i].x, geo[i].y);
      const kl = klDivGaussian(startMu, startSig, geo[i].x, geo[i].y);
      dists.push({ t: arcLen, fr, kl });
    }
    return dists;
  }, [geodesics, startMu, startSig, showNatGrad]);

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

      const lineGen = d3.line<{ x: number; y: number }>()
        .x((d) => xScale(d.x))
        .y((d) => yScale(d.y));

      // KL contours
      if (klContours) {
        const { contours, nGrid, muRange, sigRange } = klContours;
        const cxScale = d3.scaleLinear().domain([0, nGrid - 1]).range([xScale(muRange[0]), xScale(muRange[1])]);
        const cyScale = d3.scaleLinear().domain([0, nGrid - 1]).range([yScale(sigRange[0]), yScale(sigRange[1])]);

        const colorScale = d3.scaleSequential(d3.interpolateBlues).domain([0, 5]);

        contours.forEach((contour) => {
          g.append('path')
            .attr('d', d3.geoPath(
              d3.geoTransform({
                point(x, y) { this.stream.point(cxScale(x), cyScale(y)); },
              })
            )(contour))
            .style('fill', 'none')
            .style('stroke', colorScale(contour.value))
            .style('stroke-width', 1)
            .style('opacity', 0.6);
        });
      }

      if (showNatGrad) {
        // Gradient descent trajectories
        const trajLineGen = d3.line<TrajectoryPoint>()
          .x((d) => xScale(d.mu))
          .y((d) => yScale(d.sigma));

        gradTrajectories.euclidean.forEach((traj) => {
          g.append('path').datum(traj).attr('d', trajLineGen)
            .style('fill', 'none').style('stroke', PURPLE).style('stroke-width', 1.5).style('stroke-dasharray', '4,3');
        });
        gradTrajectories.natural.forEach((traj) => {
          g.append('path').datum(traj).attr('d', trajLineGen)
            .style('fill', 'none').style('stroke', TEAL).style('stroke-width', 2);
        });

        // Target point
        g.append('circle')
          .attr('cx', xScale(targetMu)).attr('cy', yScale(targetSig))
          .attr('r', 8).style('fill', AMBER).style('stroke', '#fff').style('stroke-width', 2).style('cursor', 'grab');

        const overlay = g.append('rect')
          .attr('width', w).attr('height', h)
          .style('fill', 'none').style('pointer-events', 'all').style('cursor', 'grab');

        overlay.call(d3.drag<SVGRectElement, unknown>().on('drag', (event) => {
          setTargetMu(clamp(xScale.invert(event.x), MU_RANGE[0] + DRAG_PAD, MU_RANGE[1] - DRAG_PAD));
          setTargetSig(clamp(yScale.invert(event.y), SIG_RANGE[0] + DRAG_PAD_SIG_LO, SIG_RANGE[1] - DRAG_PAD_SIG_HI));
        }));

        g.append('text').attr('x', xScale(targetMu) + 10).attr('y', yScale(targetSig) - 10)
          .style('fill', AMBER).style('font-size', '11px').text('target');

        // Legend
        g.append('line').attr('x1', 10).attr('x2', 30).attr('y1', 10).attr('y2', 10)
          .style('stroke', TEAL).style('stroke-width', 2);
        g.append('text').attr('x', 33).attr('y', 14)
          .style('fill', TEAL).style('font-size', '10px').text('Natural gradient');
        g.append('line').attr('x1', 10).attr('x2', 30).attr('y1', 26).attr('y2', 26)
          .style('stroke', PURPLE).style('stroke-width', 1.5).style('stroke-dasharray', '4,3');
        g.append('text').attr('x', 33).attr('y', 30)
          .style('fill', PURPLE).style('font-size', '10px').text('Euclidean gradient');

      } else {
        // Geodesics
        geodesics.forEach((ray, i) => {
          if (ray.length < 2) return;
          g.append('path').datum(ray).attr('d', lineGen)
            .style('fill', 'none')
            .style('stroke', i === 0 ? TEAL : TEAL)
            .style('stroke-width', i === 0 ? 2.5 : 1.5)
            .style('opacity', i === 0 ? 1 : 0.5);
        });

        // Start point
        g.append('circle')
          .attr('cx', xScale(startMu)).attr('cy', yScale(startSig))
          .attr('r', 8).style('fill', AMBER).style('stroke', '#fff').style('stroke-width', 2).style('cursor', 'grab');

        const overlay = g.append('rect')
          .attr('width', w).attr('height', h)
          .style('fill', 'none').style('pointer-events', 'all').style('cursor', 'grab');

        overlay.call(d3.drag<SVGRectElement, unknown>().on('drag', (event) => {
          setStartMu(clamp(xScale.invert(event.x), MU_RANGE[0] + DRAG_PAD, MU_RANGE[1] - DRAG_PAD));
          setStartSig(clamp(yScale.invert(event.y), SIG_RANGE[0] + DRAG_PAD_SIG_LO, SIG_RANGE[1] - DRAG_PAD_SIG_HI));
        }));
      }

      // Title
      svg.append('text').attr('x', leftWidth / 2).attr('y', 16)
        .style('fill', 'var(--color-text-primary)').style('text-anchor', 'middle')
        .style('font-size', '13px').style('font-weight', '600')
        .text(showNatGrad ? 'Gradient Descent Trajectories' : 'Fisher-Rao Geodesics');
    },
    [leftWidth, startMu, startSig, geodesics, klContours, showNatGrad, gradTrajectories, targetMu, targetSig]
  );

  // ── Right panel: distance plot ───────────────────────────────────
  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (rightWidth <= 0) return;

      const margin = { top: 30, right: 20, bottom: 40, left: 55 };
      const w = rightWidth - margin.left - margin.right;
      const h = HEIGHT - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      if (showNatGrad) {
        // Convergence plot: KL divergence over steps
        const eucTraj = gradTrajectories.euclidean[0] || [];
        const natTraj = gradTrajectories.natural[0] || [];

        const eucKL = eucTraj.map((p) => klDivGaussian(p.mu, p.sigma, targetMu, targetSig));
        const natKL = natTraj.map((p) => klDivGaussian(p.mu, p.sigma, targetMu, targetSig));

        const maxSteps = Math.max(eucKL.length, natKL.length);
        const xScale = d3.scaleLinear().domain([0, maxSteps]).range([0, w]);
        const yMax = Math.max(d3.max(eucKL) || 1, d3.max(natKL) || 1) * 1.1;
        const yScale = d3.scaleLinear().domain([0, yMax]).range([h, 0]);

        g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(5))
          .selectAll('text').style('fill', 'var(--color-text-secondary)');
        g.append('g').call(d3.axisLeft(yScale).ticks(5))
          .selectAll('text').style('fill', 'var(--color-text-secondary)');
        g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
        g.append('text').attr('x', w / 2).attr('y', h + 35)
          .style('fill', 'var(--color-text-secondary)').style('text-anchor', 'middle').style('font-size', '12px').text('Step');
        g.append('text').attr('x', -40).attr('y', h / 2)
          .style('fill', 'var(--color-text-secondary)').style('text-anchor', 'middle')
          .style('font-size', '12px').attr('transform', `rotate(-90,-40,${h / 2})`).text('D_KL to target');

        const lineEuc = d3.line<number>().x((_, i) => xScale(i)).y((d) => yScale(d));
        g.append('path').datum(eucKL).attr('d', lineEuc)
          .style('fill', 'none').style('stroke', PURPLE).style('stroke-width', 2).style('stroke-dasharray', '4,3');
        g.append('path').datum(natKL).attr('d', lineEuc)
          .style('fill', 'none').style('stroke', TEAL).style('stroke-width', 2);

        svg.append('text').attr('x', rightWidth / 2).attr('y', 16)
          .style('fill', 'var(--color-text-primary)').style('text-anchor', 'middle')
          .style('font-size', '13px').style('font-weight', '600').text('Convergence');

      } else if (distancePlot && distancePlot.length > 1) {
        const xScale = d3.scaleLinear().domain([0, d3.max(distancePlot, (d) => d.t) || 1]).range([0, w]);
        const yMax = Math.max(
          d3.max(distancePlot, (d) => d.fr) || 1,
          d3.max(distancePlot, (d) => d.kl) || 1
        ) * 1.1;
        const yScale = d3.scaleLinear().domain([0, yMax]).range([h, 0]);

        g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(5))
          .selectAll('text').style('fill', 'var(--color-text-secondary)');
        g.append('g').call(d3.axisLeft(yScale).ticks(5))
          .selectAll('text').style('fill', 'var(--color-text-secondary)');
        g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
        g.append('text').attr('x', w / 2).attr('y', h + 35)
          .style('fill', 'var(--color-text-secondary)').style('text-anchor', 'middle').style('font-size', '12px').text('Arc length');

        // Fisher-Rao distance
        const lineFR = d3.line<typeof distancePlot[0]>()
          .x((d) => xScale(d.t))
          .y((d) => yScale(d.fr));
        g.append('path').datum(distancePlot).attr('d', lineFR)
          .style('fill', 'none').style('stroke', TEAL).style('stroke-width', 2.5);

        // KL divergence
        const lineKL = d3.line<typeof distancePlot[0]>()
          .x((d) => xScale(d.t))
          .y((d) => yScale(d.kl));
        g.append('path').datum(distancePlot).attr('d', lineKL)
          .style('fill', 'none').style('stroke', PURPLE).style('stroke-width', 2).style('stroke-dasharray', '4,3');

        // Legend
        g.append('text').attr('x', w - 5).attr('y', 10).style('fill', TEAL)
          .style('text-anchor', 'end').style('font-size', '10px').text('d_FR');
        g.append('text').attr('x', w - 5).attr('y', 24).style('fill', PURPLE)
          .style('text-anchor', 'end').style('font-size', '10px').text('D_KL');

        svg.append('text').attr('x', rightWidth / 2).attr('y', 16)
          .style('fill', 'var(--color-text-primary)').style('text-anchor', 'middle')
          .style('font-size', '13px').style('font-weight', '600').text('Distance Along Geodesic');

      } else {
        g.append('text').attr('x', w / 2).attr('y', h / 2)
          .style('fill', 'var(--color-text-tertiary)').style('text-anchor', 'middle')
          .style('font-size', '12px').text('Distance plot');
      }
    },
    [rightWidth, distancePlot, showNatGrad, gradTrajectories, targetMu, targetSig]
  );

  return (
    <div ref={containerRef} className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-3">
      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-1`}>
        <svg ref={leftRef} width={leftWidth} height={HEIGHT} />
        <svg ref={rightRef} width={rightWidth} height={HEIGHT} />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-4 text-sm">
        {!showNatGrad && (
          <>
            <label className="flex items-center gap-1.5 text-[var(--color-text-secondary)]">
              <input type="checkbox" checked={showFan} onChange={(e) => setShowFan(e.target.checked)} className="accent-[var(--color-accent)]" />
              Geodesic fan
            </label>
            <label className="flex items-center gap-1.5 text-[var(--color-text-secondary)]">
              <input type="checkbox" checked={showKLContours} onChange={(e) => setShowKLContours(e.target.checked)} className="accent-[var(--color-accent)]" />
              KL contours
            </label>
          </>
        )}
        <label className="flex items-center gap-1.5 text-[var(--color-text-secondary)]">
          <input type="checkbox" checked={showNatGrad} onChange={(e) => setShowNatGrad(e.target.checked)} className="accent-[var(--color-accent)]" />
          Natural gradient mode
        </label>
      </div>

      <p className="mt-2 text-xs text-[var(--color-text-tertiary)]">
        {showNatGrad
          ? 'Drag the target to compare natural gradient (solid) vs Euclidean gradient (dashed) descent. Natural gradient follows the intrinsic geometry and converges faster.'
          : 'Drag the start point to explore Fisher-Rao geodesics on the Gaussian manifold. These are semicircles in the Poincaré half-plane model.'}
      </p>
    </div>
  );
}
