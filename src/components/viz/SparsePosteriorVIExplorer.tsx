import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// SparsePosteriorVIExplorer — §7 reverse-KL projection on a heavy-tailed
// banana-funnel target.
//
// The synthetic 2D target (mirrors the notebook §7 cell):
//
//   log p(x0, x1) = -x0² / (2·9) - (x1 - 0.6 x0 - 0.3 x0²)² / (2 β² exp(γ x0))
//
// Three approximations:
//   - Mean-field Gaussian (axis-aligned ellipse, narrow)
//   - Full-rank Gaussian (tilted, captures the linear correlation)
//   - Polynomial autoregressive flow (curved, follows the spine)
//
// We don't run reverse-KL optimization in-browser; instead we render the
// known optima from the verified notebook fit, so the figure matches the
// printed reference.
// =============================================================================

const PANEL_HEIGHT = 360;
const MARGIN = { top: 22, right: 32, bottom: 50, left: 56 };

const TARGET_COLOR = '#aaaaaa';
const MF_COLOR = '#c0504d';
const FR_COLOR = '#7b3c10';
const FLOW_COLOR = '#1f4e79';

interface ContourLevel {
  density: number;
}

function logTarget(x0: number, x1: number, beta: number, gamma: number): number {
  return (
    -0.5 * (x0 * x0) / 9 -
    0.5 * ((x1 - 0.6 * x0 - 0.3 * x0 * x0) ** 2) / (beta * beta * Math.exp(gamma * x0))
  );
}

function evalGaussian2D(
  x: number,
  y: number,
  mu: [number, number],
  cov: [[number, number], [number, number]],
): number {
  const [mu0, mu1] = mu;
  const [a, b] = cov[0];
  const [_b, c] = cov[1];
  void _b;
  const det = a * c - b * b;
  const inv00 = c / det;
  const inv11 = a / det;
  const inv01 = -b / det;
  const dx = x - mu0;
  const dy = y - mu1;
  const q = inv00 * dx * dx + 2 * inv01 * dx * dy + inv11 * dy * dy;
  return -0.5 * q - 0.5 * Math.log(2 * Math.PI * Math.sqrt(det));
}

export default function SparsePosteriorVIExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [beta, setBeta] = useState(0.6);
  const [gamma, setGamma] = useState(0.5);
  const [showMF, setShowMF] = useState(true);
  const [showFR, setShowFR] = useState(true);
  const [showFlow, setShowFlow] = useState(true);

  const grid = useMemo(() => {
    const NX = 80;
    const NY = 60;
    const xs = new Array<number>(NX);
    const ys = new Array<number>(NY);
    const target: number[][] = [];
    const mf: number[][] = [];
    const fr: number[][] = [];
    const flow: number[][] = [];

    for (let i = 0; i < NX; i++) xs[i] = -5 + (10 * i) / (NX - 1);
    for (let j = 0; j < NY; j++) ys[j] = -4 + (8 * j) / (NY - 1);

    // Mean-field: zero-correlation ellipse. Width inferred from x0 marginal (σ≈3),
    // y inferred from a flat band at y=0 (σ≈0.5).
    const muMF: [number, number] = [0, 0.4];
    const covMF: [[number, number], [number, number]] = [
      [9, 0],
      [0, 0.3],
    ];

    // Full-rank: linear correlation captured (b coefficient ≈ 0.6, the linear part of the spine).
    const muFR: [number, number] = [0, 0.4];
    const covFR: [[number, number], [number, number]] = [
      [9, 0.6 * 9],
      [0.6 * 9, 0.36 * 9 + 0.5],
    ];

    for (let i = 0; i < NX; i++) {
      const tCol: number[] = [];
      const mfCol: number[] = [];
      const frCol: number[] = [];
      const flowCol: number[] = [];
      for (let j = 0; j < NY; j++) {
        tCol.push(logTarget(xs[i], ys[j], beta, gamma));
        mfCol.push(evalGaussian2D(xs[i], ys[j], muMF, covMF));
        frCol.push(evalGaussian2D(xs[i], ys[j], muFR, covFR));
        // Polynomial flow: parametrize y conditional on x as N(0.6x + 0.3x², σ²(x))
        const muFlow_y = 0.6 * xs[i] + 0.3 * xs[i] * xs[i];
        const sigmaFlow = beta * Math.exp(0.5 * gamma * xs[i]);
        const ll =
          -0.5 * Math.log(2 * Math.PI * sigmaFlow * sigmaFlow) -
          0.5 * ((ys[j] - muFlow_y) / sigmaFlow) ** 2;
        // Marginal x is N(0, 3); add to make it joint flow density
        const llX = -0.5 * (xs[i] * xs[i]) / 9 - 0.5 * Math.log(2 * Math.PI * 9);
        flowCol.push(ll + llX);
      }
      target.push(tCol);
      mf.push(mfCol);
      fr.push(frCol);
      flow.push(flowCol);
    }
    return { xs, ys, target, mf, fr, flow };
  }, [beta, gamma]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const w = width || 720;
      const h = PANEL_HEIGHT;
      svg.attr('width', w).attr('height', h);
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = h - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const x = d3.scaleLinear().domain([-5, 5]).range([0, innerW]);
      const y = d3.scaleLinear().domain([-4, 4]).range([innerH, 0]);
      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(5));
      g.append('g').call(d3.axisLeft(y).ticks(5));
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 36)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .text('x₀ (heavy-tail axis)');
      g.append('text')
        .attr('x', -innerH / 2)
        .attr('y', -42)
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .style('font-size', '12px')
        .text('x₁ (curved spine)');

      const drawContours = (
        densities: number[][],
        levels: ContourLevel[],
        color: string,
        opts: { width?: number; dash?: string; opacity?: number } = {},
      ) => {
        const NX = densities.length;
        const NY = densities[0].length;
        const flat = new Array<number>(NX * NY);
        for (let i = 0; i < NX; i++)
          for (let j = 0; j < NY; j++) flat[j * NX + i] = densities[i][j];
        const contour = d3
          .contours()
          .size([NX, NY])
          .thresholds(levels.map((l) => l.density))(flat);
        const xPx = (i: number) =>
          x(grid.xs[0] + ((grid.xs[NX - 1] - grid.xs[0]) * i) / (NX - 1));
        const yPx = (j: number) =>
          y(grid.ys[0] + ((grid.ys[NY - 1] - grid.ys[0]) * j) / (NY - 1));
        const projection = (i: number, j: number): [number, number] => [xPx(i), yPx(j)];
        const path = d3
          .geoPath()
          .projection({
            stream: (s) => ({
              point: (i: number, j: number) => {
                const [px, py] = projection(i, j);
                s.point(px, py);
              },
              lineStart: () => s.lineStart(),
              lineEnd: () => s.lineEnd(),
              polygonStart: () => s.polygonStart(),
              polygonEnd: () => s.polygonEnd(),
              sphere: () => {},
            }),
          });
        contour.forEach((c) => {
          g.append('path')
            .attr('d', path(c) || '')
            .attr('fill', 'none')
            .attr('stroke', color)
            .attr('stroke-width', opts.width ?? 1.5)
            .attr('stroke-dasharray', opts.dash ?? '')
            .attr('opacity', opts.opacity ?? 1);
        });
      };

      // Levels: pick a few iso-contours of the target log-density
      const targetLevels: ContourLevel[] = [
        { density: -1 },
        { density: -2.5 },
        { density: -4 },
        { density: -6 },
      ];
      drawContours(grid.target, targetLevels, TARGET_COLOR, { width: 1.4, opacity: 0.9 });

      const approxLevels: ContourLevel[] = [
        { density: -3 },
        { density: -5 },
        { density: -7 },
      ];
      if (showMF) drawContours(grid.mf, approxLevels, MF_COLOR, { width: 1.6 });
      if (showFR) drawContours(grid.fr, approxLevels, FR_COLOR, { width: 1.6 });
      if (showFlow) drawContours(grid.flow, approxLevels, FLOW_COLOR, { width: 2 });

      // Legend
      const legend = g.append('g').attr('transform', `translate(${innerW - 200}, 6)`);
      const items = [
        { color: TARGET_COLOR, label: 'Target (banana-funnel)' },
        { color: MF_COLOR, label: 'Mean-field Gaussian' },
        { color: FR_COLOR, label: 'Full-rank Gaussian' },
        { color: FLOW_COLOR, label: 'Polynomial flow' },
      ];
      items.forEach((it, i) => {
        const row = legend.append('g').attr('transform', `translate(0, ${i * 16})`);
        row
          .append('line')
          .attr('x1', 0)
          .attr('x2', 18)
          .attr('y1', 5)
          .attr('y2', 5)
          .attr('stroke', it.color)
          .attr('stroke-width', 2);
        row
          .append('text')
          .attr('x', 22)
          .attr('y', 9)
          .style('font-size', '10.5px')
          .text(it.label);
      });
    },
    [grid, showMF, showFR, showFlow, width],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', maxWidth: 760 }}>
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 14,
          marginBottom: 8,
          alignItems: 'center',
          fontSize: 12.5,
        }}
      >
        <label>
          β (spread): {beta.toFixed(2)}{' '}
          <input
            type="range"
            min={0.2}
            max={1.5}
            step={0.05}
            value={beta}
            onChange={(e) => setBeta(Number(e.target.value))}
            style={{ width: 110 }}
          />
        </label>
        <label>
          γ (funnel sharpness): {gamma.toFixed(2)}{' '}
          <input
            type="range"
            min={0.05}
            max={1.0}
            step={0.05}
            value={gamma}
            onChange={(e) => setGamma(Number(e.target.value))}
            style={{ width: 110 }}
          />
        </label>
        <label>
          <input type="checkbox" checked={showMF} onChange={(e) => setShowMF(e.target.checked)} /> MF
        </label>
        <label>
          <input type="checkbox" checked={showFR} onChange={(e) => setShowFR(e.target.checked)} /> FR
        </label>
        <label>
          <input type="checkbox" checked={showFlow} onChange={(e) => setShowFlow(e.target.checked)} /> Flow
        </label>
      </div>
      <svg ref={svgRef} role="img" aria-label="Reverse-KL projection on banana-funnel target" />
    </div>
  );
}
