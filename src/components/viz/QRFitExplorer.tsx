import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  fitPredictMultipleQuantiles,
  fitPredictQuantile,
  mulberry32,
  rearrangedQuantilePredictions,
  synthHeteroscedastic,
} from './shared/nonparametric-ml';

// =============================================================================
// QRFitExplorer — embedded twice in the topic:
//   §2 (single-τ mode default): one QR fit at the user's chosen τ, with the
//        closed-form true conditional τ-quantile dashed for reference. Slider:
//        τ ∈ [0.05, 0.95], n ∈ {50, 100, 200, 500, 1000}, degree ∈ {1, 2, 3, 5}.
//   §6 (multi-τ + rearrange toggle activated): K = 11 simultaneous QR fits at
//        τ ∈ {0.05, 0.10, 0.20, ..., 0.90, 0.95}. With rearrangement OFF, the
//        marginal fits cross at some evaluation points (shown as red ×). With
//        rearrangement ON (CFV-G 2010, Definition 4), crossings disappear.
//
// Controls are debounced 200ms so that dragging the n or degree slider through
// {50 → 1000} or {1, 2, 3, 5} doesn't refit on every intermediate value.
// =============================================================================

const PANEL_HEIGHT = 360;
const SM_BREAKPOINT = 640;
const N_EVAL = 200;

const BLUE = '#2563EB';
const RED = '#DC2626';
const TEAL = '#0F6E56';
const SLATE = '#475569';

type Mode = 'single' | 'multi';

const N_OPTIONS = [50, 100, 200, 500, 1000] as const;
const DEGREE_OPTIONS = [1, 2, 3, 5] as const;

// 11-level τ grid for multi-τ mode (per brief §4b).
const TAUS_11 = new Float64Array([
  0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95,
]);

// Inline Acklam inverse Normal CDF (also in PinballLossExplorer; per repo
// convention, small distribution helpers are duplicated rather than shared).
function normalInverseCDF(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
    3.754408661907416,
  ];
  const pl = 0.02425;
  const ph = 1 - pl;
  let q: number;
  let r: number;
  if (p < pl) {
    q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  if (p <= ph) {
    q = p - 0.5;
    r = q * q;
    return (
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    );
  }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return (
    -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  );
}

// Closed-form true conditional τ-quantile for the synthHeteroscedastic DGP:
//   y = 0.5x + sin(1.5x) + σ(x)·ε,   σ(x) = 0.3 + 0.6|x|,   ε ~ N(0, 1)
// ⇒ ξ_τ(Y | X = x) = 0.5x + sin(1.5x) + σ(x) · Φ⁻¹(τ).
function trueConditionalQuantile(x: number, tau: number): number {
  const sigma = 0.3 + 0.6 * Math.abs(x);
  return 0.5 * x + Math.sin(1.5 * x) + sigma * normalInverseCDF(tau);
}

// Debounce hook — returns a value that updates `delay` ms after the last input.
function useDebounced<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return debounced;
}

const fmt = (x: number, digits = 2) => x.toFixed(digits);

export default function QRFitExplorer({ defaultMode = 'single' as Mode }: { defaultMode?: Mode } = {}) {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();
  const [tau, setTau] = useState(0.5);
  const [nIdx, setNIdx] = useState(2); // index into N_OPTIONS, default 200
  const [degIdx, setDegIdx] = useState(2); // index into DEGREE_OPTIONS, default 3
  const [mode, setMode] = useState<Mode>(defaultMode);
  const [rearrange, setRearrange] = useState(false);
  const [seed, setSeed] = useState(11);

  // Debounced settings — driving the heavy QR fits.
  const dTau = useDebounced(tau, 150);
  const dN = useDebounced(nIdx, 200);
  const dDeg = useDebounced(degIdx, 200);
  const dMode = useDebounced(mode, 50);
  const dRearrange = useDebounced(rearrange, 50);
  const dSeed = seed; // resample is button-driven; no debounce

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = containerWidth || 0;

  const n = N_OPTIONS[dN];
  const degree = DEGREE_OPTIONS[dDeg];

  // Synthesize data + fit QR. Heavy compute, memoised on debounced controls.
  const fit = useMemo(() => {
    const rng = mulberry32(dSeed);
    const { x, y } = synthHeteroscedastic(n, rng);
    const xEval = new Float64Array(N_EVAL);
    for (let i = 0; i < N_EVAL; i++) xEval[i] = -2 + (4 * i) / (N_EVAL - 1);

    if (dMode === 'single') {
      const preds = fitPredictQuantile(x, y, xEval, dTau);
      const truth = new Float64Array(N_EVAL);
      for (let i = 0; i < N_EVAL; i++) truth[i] = trueConditionalQuantile(xEval[i], dTau);
      return {
        kind: 'single' as const,
        x,
        y,
        xEval,
        preds,
        truth,
      };
    }

    const Q = fitPredictMultipleQuantiles(x, y, xEval, TAUS_11, degree);
    const Qfinal = dRearrange
      ? rearrangedQuantilePredictions(Q, TAUS_11.length, N_EVAL)
      : Q;

    // Detect crossings (only meaningful when rearrangement is OFF).
    const crossings: Array<{ x: number; y: number }> = [];
    if (!dRearrange) {
      for (let j = 0; j < N_EVAL; j++) {
        for (let k = 0; k < TAUS_11.length - 1; k++) {
          const a = Qfinal[k * N_EVAL + j];
          const b = Qfinal[(k + 1) * N_EVAL + j];
          if (a > b + 1e-9) {
            crossings.push({ x: xEval[j], y: 0.5 * (a + b) });
            break; // one × per eval-point is enough
          }
        }
      }
    }

    return {
      kind: 'multi' as const,
      x,
      y,
      xEval,
      Q: Qfinal,
      crossings,
    };
  }, [dSeed, dMode, dTau, dN, dDeg, dRearrange, n, degree]);

  // ── Single SVG render ──────────────────────────────────────────────────
  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;

      const margin = { top: 28, right: 16, bottom: 36, left: 44 };
      const w = panelWidth - margin.left - margin.right;
      const h = PANEL_HEIGHT - margin.top - margin.bottom;
      const g = svg
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

      // y-domain: pad the data range a bit so high/low quantiles aren't clipped.
      let ymin = Infinity;
      let ymax = -Infinity;
      for (let i = 0; i < fit.x.length; i++) {
        if (fit.y[i] < ymin) ymin = fit.y[i];
        if (fit.y[i] > ymax) ymax = fit.y[i];
      }
      const ypad = 0.1 * (ymax - ymin);
      const xScale = d3.scaleLinear().domain([-2.05, 2.05]).range([0, w]);
      const yScale = d3.scaleLinear().domain([ymin - ypad, ymax + ypad]).range([h, 0]);

      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text)')
        .style('font-size', '10px');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text)')
        .style('font-size', '10px');
      g.selectAll('path.domain, .tick line').style('stroke', 'var(--color-border)');

      // Scatter
      g.append('g')
        .selectAll('circle')
        .data(d3.range(fit.x.length))
        .join('circle')
        .attr('cx', (i) => xScale(fit.x[i]))
        .attr('cy', (i) => yScale(fit.y[i]))
        .attr('r', 2.2)
        .style('fill', SLATE)
        .style('opacity', 0.35);

      const xLine = (j: number) => xScale(fit.xEval[j]);

      if (fit.kind === 'single') {
        // True conditional quantile (dashed, black)
        const truthLine = d3
          .line<number>()
          .x(xLine)
          .y((j) => yScale(fit.truth[j]));
        g.append('path')
          .datum(d3.range(N_EVAL))
          .attr('d', truthLine)
          .style('fill', 'none')
          .style('stroke', 'var(--color-text)')
          .style('stroke-width', 1.4)
          .style('stroke-dasharray', '5,4')
          .style('opacity', 0.7);

        // QR fit (TEAL)
        const fitLine = d3
          .line<number>()
          .x(xLine)
          .y((j) => yScale(fit.preds[j]));
        g.append('path')
          .datum(d3.range(N_EVAL))
          .attr('d', fitLine)
          .style('fill', 'none')
          .style('stroke', TEAL)
          .style('stroke-width', 2.2);

        // Legend
        const legend = svg
          .append('g')
          .attr('transform', `translate(${margin.left + 10},${margin.top + 8})`)
          .style('font-family', 'var(--font-sans)')
          .style('font-size', '10.5px');
        legend
          .append('line')
          .attr('x1', 0)
          .attr('x2', 22)
          .attr('y1', 4)
          .attr('y2', 4)
          .style('stroke', TEAL)
          .style('stroke-width', 2.2);
        legend
          .append('text')
          .attr('x', 26)
          .attr('y', 7)
          .style('fill', 'var(--color-text)')
          .text(`QR fit (τ = ${fmt(dTau)})`);
        legend
          .append('line')
          .attr('x1', 0)
          .attr('x2', 22)
          .attr('y1', 22)
          .attr('y2', 22)
          .style('stroke', 'var(--color-text)')
          .style('stroke-width', 1.4)
          .style('stroke-dasharray', '5,4')
          .style('opacity', 0.7);
        legend
          .append('text')
          .attr('x', 26)
          .attr('y', 25)
          .style('fill', 'var(--color-text)')
          .text('true conditional τ-quantile');
      } else {
        // Multi-τ — colour by τ via viridis
        const cmap = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);
        for (let k = 0; k < TAUS_11.length; k++) {
          const line = d3
            .line<number>()
            .x(xLine)
            .y((j) => yScale(fit.Q[k * N_EVAL + j]));
          g.append('path')
            .datum(d3.range(N_EVAL))
            .attr('d', line)
            .style('fill', 'none')
            .style('stroke', cmap(0.05 + 0.85 * (k / (TAUS_11.length - 1))))
            .style('stroke-width', 1.4)
            .style('opacity', 0.95);
        }

        // Crossing markers (red ×)
        const crossG = g.append('g');
        for (const c of fit.crossings) {
          crossG
            .append('text')
            .attr('x', xScale(c.x))
            .attr('y', yScale(c.y) + 3)
            .attr('text-anchor', 'middle')
            .style('fill', RED)
            .style('font-size', '11px')
            .style('font-weight', 600)
            .text('×');
        }

        // Legend / status text
        const status = svg
          .append('text')
          .attr('x', margin.left + 6)
          .attr('y', 16)
          .style('font-family', 'var(--font-sans)')
          .style('font-size', '11px')
          .style('fill', 'var(--color-text)');
        status.append('tspan').text(
          `K = ${TAUS_11.length} QR fits  ·  τ ∈ {0.05, 0.10, 0.20, …, 0.90, 0.95}`,
        );
        svg
          .append('text')
          .attr('x', panelWidth - margin.right)
          .attr('y', 16)
          .attr('text-anchor', 'end')
          .style('font-family', 'var(--font-sans)')
          .style('font-size', '11px')
          .style('fill', dRearrange || fit.crossings.length === 0 ? TEAL : RED)
          .text(
            dRearrange
              ? `rearranged: 0 crossings`
              : `crossings: ${fit.crossings.length} / ${N_EVAL} eval points`,
          );
      }

      // x-axis label
      svg
        .append('text')
        .attr('x', margin.left + w / 2)
        .attr('y', PANEL_HEIGHT - 6)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '10.5px')
        .text('x');
      // y-axis label
      svg
        .append('text')
        .attr('transform', `translate(14,${margin.top + h / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '10.5px')
        .text('y');
    },
    [fit, dTau, dRearrange, panelWidth],
  );

  return (
    <div ref={containerRef} className="not-prose">
      <div
        style={{
          marginBottom: '0.75rem',
          display: 'flex',
          flexWrap: 'wrap',
          gap: isStacked ? '0.6rem 1rem' : '0.4rem 1.4rem',
          alignItems: 'center',
          fontFamily: 'var(--font-sans)',
          fontSize: '12px',
        }}
      >
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            flex: '1 1 220px',
            minWidth: 200,
            opacity: mode === 'single' ? 1 : 0.4,
          }}
        >
          <span style={{ color: 'var(--color-text-secondary)', minWidth: '1.2em' }}>τ</span>
          <input
            type="range"
            min={0.05}
            max={0.95}
            step={0.01}
            value={tau}
            disabled={mode !== 'single'}
            onChange={(e) => setTau(Number(e.target.value))}
            style={{ flex: 1 }}
            aria-label="QR τ slider"
          />
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              minWidth: '3em',
              textAlign: 'right',
              color: 'var(--color-text)',
            }}
          >
            {fmt(tau)}
          </span>
        </label>

        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ color: 'var(--color-text-secondary)' }}>n</span>
          <input
            type="range"
            min={0}
            max={N_OPTIONS.length - 1}
            step={1}
            value={nIdx}
            onChange={(e) => setNIdx(Number(e.target.value))}
            aria-label="Sample size n slider"
          />
          <span style={{ fontFamily: 'var(--font-mono)', minWidth: '3em', textAlign: 'right' }}>
            {N_OPTIONS[nIdx]}
          </span>
        </label>

        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ color: 'var(--color-text-secondary)' }}>degree</span>
          <input
            type="range"
            min={0}
            max={DEGREE_OPTIONS.length - 1}
            step={1}
            value={degIdx}
            onChange={(e) => setDegIdx(Number(e.target.value))}
            aria-label="Polynomial degree slider"
          />
          <span style={{ fontFamily: 'var(--font-mono)', minWidth: '1em', textAlign: 'right' }}>
            {DEGREE_OPTIONS[degIdx]}
          </span>
        </label>

        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <input
            type="checkbox"
            checked={mode === 'multi'}
            onChange={(e) => setMode(e.target.checked ? 'multi' : 'single')}
            aria-label="Multi-τ mode toggle"
          />
          <span style={{ color: 'var(--color-text)' }}>multi-τ (K = 11)</span>
        </label>

        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            opacity: mode === 'multi' ? 1 : 0.4,
          }}
        >
          <input
            type="checkbox"
            checked={rearrange}
            disabled={mode !== 'multi'}
            onChange={(e) => setRearrange(e.target.checked)}
            aria-label="Rearrangement toggle"
          />
          <span style={{ color: 'var(--color-text)' }}>rearrange (CFV-G)</span>
        </label>

        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          style={{
            padding: '4px 10px',
            background: 'var(--color-surface)',
            color: 'var(--color-text)',
            border: '1px solid var(--color-border)',
            borderRadius: 4,
            fontSize: '11.5px',
            cursor: 'pointer',
          }}
        >
          Resample
        </button>
      </div>
      <svg
        ref={svgRef}
        width={panelWidth}
        height={PANEL_HEIGHT}
        role="img"
        aria-label={
          mode === 'single'
            ? 'Single-τ QR fit on heteroscedastic data with the closed-form true conditional quantile'
            : 'Eleven simultaneous QR fits with optional CFV-G rearrangement; crossings marked'
        }
      />
    </div>
  );
}
