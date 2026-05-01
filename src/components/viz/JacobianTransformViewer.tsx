import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// JacobianTransformViewer — embedded after §3.2's logit subsection of the
// probabilistic-programming topic. Two-panel viz: left panel shows Beta(α, β)
// density on (0, 1); right panel shows the logit pushforward density on ℝ
// computed from eq. (3.2). When α, β > 1 both panels mark the modes — x* on
// the left, y* on the right. A faded dotted line on the right also marks T(x*),
// the image of the constrained mode under logit, which is generally NOT equal
// to y*. Reader sees that nonlinear reparameterization is not mode-preserving,
// and that the Jacobian factor log θ + log(1 − θ) is the conversion bridge.
// =============================================================================

const PANEL_HEIGHT = 320;
const N_GRID = 240;
const Y_DOMAIN: [number, number] = [-6, 6];
const PANEL_GAP = 36;

const COLORS = {
  constrained: '#0d9488',   // teal — constrained-side curve and T(x*) marker
  unconstrained: '#ea580c', // orange — unconstrained-side curve
  modeLine: '#374151',      // slate — true-mode dashed lines
};

// Lanczos approximation for log-gamma — accurate to ~1e-15 for x > 0. The
// reflection-formula branch handles x < 0.5 but is unused here since our
// sliders restrict α, β ≥ 1.
const LANCZOS_G = 7;
const LANCZOS_C = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

function logGamma(x: number): number {
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  x -= 1;
  let a = LANCZOS_C[0];
  const t = x + LANCZOS_G + 0.5;
  for (let i = 1; i < LANCZOS_C.length; i++) {
    a += LANCZOS_C[i] / (x + i);
  }
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function logBeta(a: number, b: number): number {
  return logGamma(a) + logGamma(b) - logGamma(a + b);
}

function betaDensity(x: number, alpha: number, beta: number): number {
  if (x <= 0 || x >= 1) return 0;
  const logD =
    (alpha - 1) * Math.log(x) + (beta - 1) * Math.log(1 - x) - logBeta(alpha, beta);
  return Math.exp(logD);
}

function sigmoid(y: number): number {
  return 1 / (1 + Math.exp(-y));
}

// Pushforward density of logit(X) when X ~ Beta(α, β). From eq. (3.2):
//   p_Y(y) = exp(α log σ(y) + β log σ(-y) − log B(α, β)).
function logitPushforward(y: number, alpha: number, beta: number): number {
  const sy = sigmoid(y);
  if (sy <= 0 || sy >= 1) return 0;
  const logD = alpha * Math.log(sy) + beta * Math.log(1 - sy) - logBeta(alpha, beta);
  return Math.exp(logD);
}

export default function JacobianTransformViewer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [alpha, setAlpha] = useState<number>(2);
  const [beta, setBeta] = useState<number>(2);
  const [showModes, setShowModes] = useState<boolean>(true);

  const xGrid = useMemo<Float64Array>(() => {
    const g = new Float64Array(N_GRID);
    for (let i = 0; i < N_GRID; i++) g[i] = (i + 0.5) / N_GRID;
    return g;
  }, []);

  const yGrid = useMemo<Float64Array>(() => {
    const g = new Float64Array(N_GRID);
    const span = Y_DOMAIN[1] - Y_DOMAIN[0];
    for (let i = 0; i < N_GRID; i++) g[i] = Y_DOMAIN[0] + (i / (N_GRID - 1)) * span;
    return g;
  }, []);

  const betaCurve = useMemo<Float64Array>(() => {
    const c = new Float64Array(N_GRID);
    for (let i = 0; i < N_GRID; i++) c[i] = betaDensity(xGrid[i], alpha, beta);
    return c;
  }, [xGrid, alpha, beta]);

  const pushforwardCurve = useMemo<Float64Array>(() => {
    const c = new Float64Array(N_GRID);
    for (let i = 0; i < N_GRID; i++) c[i] = logitPushforward(yGrid[i], alpha, beta);
    return c;
  }, [yGrid, alpha, beta]);

  const yMax = useMemo<number>(() => {
    let m = 0;
    for (let i = 0; i < N_GRID; i++) {
      if (betaCurve[i] > m) m = betaCurve[i];
      if (pushforwardCurve[i] > m) m = pushforwardCurve[i];
    }
    return m * 1.08;
  }, [betaCurve, pushforwardCurve]);

  // Beta has an interior mode only when α > 1 and β > 1.
  const xMode = useMemo<number | null>(() => {
    if (alpha <= 1 || beta <= 1) return null;
    return (alpha - 1) / (alpha + beta - 2);
  }, [alpha, beta]);

  const yMode = useMemo<number>(() => Math.log(alpha / beta), [alpha, beta]);

  const ref = useD3(
    (svg) => {
      const w = width || 720;
      const h = PANEL_HEIGHT;
      const margin = { top: 22, right: 18, bottom: 48, left: 48 };
      svg.attr('width', w).attr('height', h);
      svg.selectAll('*').remove();

      const innerW = w - margin.left - margin.right;
      const innerH = h - margin.top - margin.bottom;
      const panelW = (innerW - PANEL_GAP) / 2;

      const root = svg
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

      const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

      // ── Left panel: Beta density on (0, 1) ─────────────────────────────
      const leftG = root.append('g').attr('transform', 'translate(0,0)');
      const xScale = d3.scaleLinear().domain([0, 1]).range([0, panelW]);

      leftG
        .append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(xScale).ticks(5).tickSizeOuter(0));
      leftG.append('g').call(d3.axisLeft(yScale).ticks(5).tickSizeOuter(0));

      leftG
        .append('text')
        .attr('transform', `translate(${panelW / 2},${innerH + 34})`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('θ ∈ (0, 1)');

      leftG
        .append('text')
        .attr('transform', `translate(-36,${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('density');

      leftG
        .append('text')
        .attr('x', panelW / 2)
        .attr('y', -6)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .attr('font-weight', 600)
        .attr('fill', COLORS.constrained)
        .text('p_X(θ) = Beta(α, β)');

      const areaBeta = d3
        .area<number>()
        .x((_, i) => xScale(xGrid[i]))
        .y0(innerH)
        .y1((d) => yScale(d))
        .curve(d3.curveMonotoneX);

      leftG
        .append('path')
        .datum(Array.from(betaCurve))
        .attr('fill', COLORS.constrained)
        .attr('fill-opacity', 0.18)
        .attr('stroke', COLORS.constrained)
        .attr('stroke-width', 1.8)
        .attr('d', areaBeta);

      if (showModes && xMode !== null) {
        leftG
          .append('line')
          .attr('x1', xScale(xMode))
          .attr('x2', xScale(xMode))
          .attr('y1', 0)
          .attr('y2', innerH)
          .attr('stroke', COLORS.modeLine)
          .attr('stroke-dasharray', '4 3')
          .attr('stroke-width', 1.2);

        leftG
          .append('text')
          .attr('x', xScale(xMode))
          .attr('y', innerH + 14)
          .attr('text-anchor', 'middle')
          .attr('font-size', 10)
          .attr('fill', COLORS.modeLine)
          .text(`x* = ${xMode.toFixed(2)}`);
      }

      // ── Right panel: pushforward on ℝ ──────────────────────────────────
      const rightG = root
        .append('g')
        .attr('transform', `translate(${panelW + PANEL_GAP},0)`);
      const yAxisScale = d3.scaleLinear().domain(Y_DOMAIN).range([0, panelW]);

      rightG
        .append('g')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(yAxisScale).ticks(7).tickSizeOuter(0));
      rightG
        .append('g')
        .call(
          d3
            .axisLeft(yScale)
            .ticks(5)
            .tickSizeOuter(0)
            .tickFormat(() => ''),
        );

      rightG
        .append('text')
        .attr('transform', `translate(${panelW / 2},${innerH + 34})`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .text('y = logit(θ) ∈ ℝ');

      rightG
        .append('text')
        .attr('x', panelW / 2)
        .attr('y', -6)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .attr('font-weight', 600)
        .attr('fill', COLORS.unconstrained)
        .text('p_Y(y) = pushforward via logit');

      const areaPushforward = d3
        .area<number>()
        .x((_, i) => yAxisScale(yGrid[i]))
        .y0(innerH)
        .y1((d) => yScale(d))
        .curve(d3.curveMonotoneX);

      rightG
        .append('path')
        .datum(Array.from(pushforwardCurve))
        .attr('fill', COLORS.unconstrained)
        .attr('fill-opacity', 0.18)
        .attr('stroke', COLORS.unconstrained)
        .attr('stroke-width', 1.8)
        .attr('d', areaPushforward);

      if (showModes) {
        // y* — true mode of the pushforward density.
        rightG
          .append('line')
          .attr('x1', yAxisScale(yMode))
          .attr('x2', yAxisScale(yMode))
          .attr('y1', 0)
          .attr('y2', innerH)
          .attr('stroke', COLORS.modeLine)
          .attr('stroke-dasharray', '4 3')
          .attr('stroke-width', 1.2);

        rightG
          .append('text')
          .attr('x', yAxisScale(yMode))
          .attr('y', innerH + 14)
          .attr('text-anchor', 'middle')
          .attr('font-size', 10)
          .attr('fill', COLORS.modeLine)
          .text(`y* = ${yMode.toFixed(2)}`);

        // T(x*) — image of the constrained mode under logit. Generally ≠ y*.
        if (xMode !== null) {
          const tOfXMode = Math.log(xMode / (1 - xMode));
          if (tOfXMode >= Y_DOMAIN[0] && tOfXMode <= Y_DOMAIN[1]) {
            rightG
              .append('line')
              .attr('x1', yAxisScale(tOfXMode))
              .attr('x2', yAxisScale(tOfXMode))
              .attr('y1', 0)
              .attr('y2', innerH)
              .attr('stroke', COLORS.constrained)
              .attr('stroke-dasharray', '2 4')
              .attr('stroke-width', 1.0);

            rightG
              .append('text')
              .attr('x', yAxisScale(tOfXMode))
              .attr('y', 12)
              .attr('text-anchor', 'middle')
              .attr('font-size', 10)
              .attr('fill', COLORS.constrained)
              .text(`T(x*) = ${tOfXMode.toFixed(2)}`);
          }
        }
      }
    },
    [betaCurve, pushforwardCurve, yMax, xMode, yMode, showModes, width],
  );

  return (
    <div className="my-6 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-soft)] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2">
          α:
          <input
            type="range"
            min={1}
            max={10}
            step={0.5}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
            className="w-32"
            aria-label="alpha shape parameter"
          />
          <span className="tabular-nums w-8 text-right">{alpha.toFixed(1)}</span>
        </label>
        <label className="flex items-center gap-2">
          β:
          <input
            type="range"
            min={1}
            max={10}
            step={0.5}
            value={beta}
            onChange={(e) => setBeta(Number(e.target.value))}
            className="w-32"
            aria-label="beta shape parameter"
          />
          <span className="tabular-nums w-8 text-right">{beta.toFixed(1)}</span>
        </label>
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={showModes}
            onChange={(e) => setShowModes(e.target.checked)}
          />
          <span>Show modes</span>
        </label>
      </div>
      <div ref={containerRef} className="w-full">
        <svg ref={ref} />
      </div>
      <div className="mt-3 text-xs text-[var(--color-text-muted)] leading-relaxed">
        <strong>Jacobian correction:</strong>{' '}
        log p_Y(y) = log p_X(θ) + log θ + log(1 − θ), with θ = σ(y). Drag α or β —
        the constrained mode x* = (α − 1)/(α + β − 2) and the unconstrained mode
        y* = log(α/β) move independently. The faded teal line on the right marks
        T(x*), the image of x* under logit; it generally lands away from y*,
        confirming that modes are not reparameterization-invariant.
      </div>
    </div>
  );
}
