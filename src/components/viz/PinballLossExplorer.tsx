import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { gaussianSampler, mulberry32 } from './shared/nonparametric-ml';

// =============================================================================
// PinballLossExplorer — embedded in §1 after Theorem 1.
//
// Three-panel demonstration of Theorem 1 (pinball minimization recovers the
// population quantile):
//
//   Left panel   : the pinball loss V at the user's chosen τ. Right arm slope
//                  τ, left arm slope τ − 1; symmetric at τ = 0.5.
//
//   Middle panel : empirical expected pinball loss L(q) ≈ (1/N) Σ ρ_τ(Y_i − q)
//                  using N = 5000 pre-sampled draws from the user's choice of
//                  Y. Vertical dashed line at the empirical argmin; vertical
//                  solid line at the closed-form true τ-quantile. They should
//                  agree to ~0.02 at this N.
//
//   Right panel  : the gradient L'(q) = F_n(q) − τ via the empirical CDF.
//                  Crosses zero exactly at the τ-quantile (Theorem 1's
//                  first-order condition).
//
// Pre-samples 5000 Y values per distribution (cached on dist change) so τ
// changes are ~5ms — no throttling needed.
// =============================================================================

const PANEL_HEIGHT = 230;
// 640px breakpoint: at the topic-page max-w-3xl content width (~720px) we get
// the 3-column layout; below that (mobile + narrow tablets) panels stack.
const SM_BREAKPOINT = 640;
const N_MC = 5000;
const N_GRID = 200;

const BLUE = '#2563EB';
const RED = '#DC2626';
const TEAL = '#0F6E56';

type Distribution = 'normal' | 'exponential' | 'uniform';

// Acklam's algorithm for the inverse Normal CDF; relative error < 1e-9 across
// the support. Reference: P. J. Acklam, "An algorithm for computing the inverse
// normal cumulative distribution function" (2003).
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

function trueQuantile(dist: Distribution, tau: number): number {
  switch (dist) {
    case 'normal':
      return normalInverseCDF(tau);
    case 'exponential':
      return -Math.log(1 - tau);
    case 'uniform':
      return tau;
  }
}

function distRange(dist: Distribution): [number, number] {
  switch (dist) {
    case 'normal':
      return [-3, 3];
    case 'exponential':
      return [0, 5];
    case 'uniform':
      return [0, 1];
  }
}

function pinballLoss(u: number, tau: number): number {
  return u >= 0 ? tau * u : (tau - 1) * u;
}

const fmt = (x: number, digits = 2) => x.toFixed(digits);

export default function PinballLossExplorer() {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();
  const [tau, setTau] = useState(0.5);
  const [dist, setDist] = useState<Distribution>('normal');

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked
    ? containerWidth || 0
    : Math.floor((containerWidth || 700) / 3) - 4;

  // Pre-sample 5000 Y values once per distribution.
  const samples = useMemo(() => {
    const rng = mulberry32(42);
    const ys = new Float64Array(N_MC);
    if (dist === 'normal') {
      const gauss = gaussianSampler(rng);
      for (let i = 0; i < N_MC; i++) ys[i] = gauss();
    } else if (dist === 'exponential') {
      for (let i = 0; i < N_MC; i++) {
        let u = rng();
        if (u < 1e-12) u = 1e-12;
        ys[i] = -Math.log(u);
      }
    } else {
      for (let i = 0; i < N_MC; i++) ys[i] = rng();
    }
    const ysSorted = Float64Array.from(ys).sort();
    return { ys, ysSorted };
  }, [dist]);

  // Per-(tau, distribution) panel data.
  const panels = useMemo(() => {
    const [qLo, qHi] = distRange(dist);
    const qGrid = new Float64Array(N_GRID);
    for (let i = 0; i < N_GRID; i++) qGrid[i] = qLo + ((qHi - qLo) * i) / (N_GRID - 1);

    // Loss V over u ∈ [-2, 2]
    const uGrid = new Float64Array(N_GRID);
    const lossCurve = new Float64Array(N_GRID);
    for (let i = 0; i < N_GRID; i++) {
      uGrid[i] = -2 + (4 * i) / (N_GRID - 1);
      lossCurve[i] = pinballLoss(uGrid[i], tau);
    }

    // L(q) via Monte Carlo over the pre-sampled Y values.
    const Lcurve = new Float64Array(N_GRID);
    for (let j = 0; j < N_GRID; j++) {
      const q = qGrid[j];
      let s = 0;
      for (let i = 0; i < N_MC; i++) s += pinballLoss(samples.ys[i] - q, tau);
      Lcurve[j] = s / N_MC;
    }

    // Empirical argmin
    let argminIdx = 0;
    for (let j = 1; j < N_GRID; j++) if (Lcurve[j] < Lcurve[argminIdx]) argminIdx = j;
    const empiricalArgmin = qGrid[argminIdx];

    // Gradient L'(q) = F_n(q) - τ via binary search on the sorted samples.
    const grad = new Float64Array(N_GRID);
    for (let j = 0; j < N_GRID; j++) {
      const q = qGrid[j];
      let lo = 0;
      let hi = N_MC;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (samples.ysSorted[mid] <= q) lo = mid + 1;
        else hi = mid;
      }
      grad[j] = lo / N_MC - tau;
    }

    const trueQ = trueQuantile(dist, tau);

    // Y-axis scale for L(q): expand to fit the curve plus 5%.
    let Lmax = 0;
    for (let j = 0; j < N_GRID; j++) if (Lcurve[j] > Lmax) Lmax = Lcurve[j];

    return {
      uGrid,
      lossCurve,
      qGrid,
      Lcurve,
      grad,
      empiricalArgmin,
      trueQ,
      qLo,
      qHi,
      Lmax,
    };
  }, [tau, samples, dist]);

  // Y-axis maximum for the loss V (panel 1): max{τ·2, (1-τ)·2}.
  const lossVmax = Math.max(tau, 1 - tau) * 2;

  // ── Panel 1: pinball loss V ────────────────────────────────────────────
  const lossRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const margin = { top: 28, right: 12, bottom: 36, left: 38 };
      const w = panelWidth - margin.left - margin.right;
      const h = PANEL_HEIGHT - margin.top - margin.bottom;
      const g = svg
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([-2, 2]).range([0, w]);
      const yScale = d3.scaleLinear().domain([0, lossVmax]).range([h, 0]);

      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text)')
        .style('font-size', '10px');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(4))
        .selectAll('text')
        .style('fill', 'var(--color-text)')
        .style('font-size', '10px');
      g.selectAll('path.domain, .tick line').style('stroke', 'var(--color-border)');

      // Reference axes at 0
      g.append('line')
        .attr('x1', xScale(0))
        .attr('x2', xScale(0))
        .attr('y1', 0)
        .attr('y2', h)
        .style('stroke', 'var(--color-border)')
        .style('stroke-dasharray', '2,3');

      // Loss curve
      const line = d3
        .line<number>()
        .x((i) => xScale(panels.uGrid[i]))
        .y((i) => yScale(panels.lossCurve[i]));
      g.append('path')
        .datum(d3.range(N_GRID))
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', BLUE)
        .style('stroke-width', 2);

      // Title
      svg
        .append('text')
        .attr('x', margin.left + w / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '11.5px')
        .style('font-family', 'var(--font-sans)')
        .text(`Pinball loss  ρ_τ(u),  τ = ${fmt(tau)}`);

      // x-axis label
      svg
        .append('text')
        .attr('x', margin.left + w / 2)
        .attr('y', PANEL_HEIGHT - 6)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '10px')
        .text('u = y − q');
    },
    [panels, tau, panelWidth, lossVmax],
  );

  // ── Panel 2: expected loss L(q) with empirical argmin + true quantile ──
  const lqRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const margin = { top: 28, right: 12, bottom: 36, left: 38 };
      const w = panelWidth - margin.left - margin.right;
      const h = PANEL_HEIGHT - margin.top - margin.bottom;
      const g = svg
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([panels.qLo, panels.qHi]).range([0, w]);
      const yScale = d3
        .scaleLinear()
        .domain([0, Math.max(panels.Lmax * 1.05, 1e-6)])
        .range([h, 0]);

      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text)')
        .style('font-size', '10px');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(4))
        .selectAll('text')
        .style('fill', 'var(--color-text)')
        .style('font-size', '10px');
      g.selectAll('path.domain, .tick line').style('stroke', 'var(--color-border)');

      // Curve
      const line = d3
        .line<number>()
        .x((i) => xScale(panels.qGrid[i]))
        .y((i) => yScale(panels.Lcurve[i]));
      g.append('path')
        .datum(d3.range(N_GRID))
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', TEAL)
        .style('stroke-width', 2);

      // Empirical argmin (dashed blue)
      g.append('line')
        .attr('x1', xScale(panels.empiricalArgmin))
        .attr('x2', xScale(panels.empiricalArgmin))
        .attr('y1', 0)
        .attr('y2', h)
        .style('stroke', BLUE)
        .style('stroke-width', 1.6)
        .style('stroke-dasharray', '4,3');

      // True quantile (solid red), if it lies in range
      if (panels.trueQ >= panels.qLo && panels.trueQ <= panels.qHi) {
        g.append('line')
          .attr('x1', xScale(panels.trueQ))
          .attr('x2', xScale(panels.trueQ))
          .attr('y1', 0)
          .attr('y2', h)
          .style('stroke', RED)
          .style('stroke-width', 1.4);
      }

      svg
        .append('text')
        .attr('x', margin.left + w / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '11.5px')
        .style('font-family', 'var(--font-sans)')
        .text(`Expected loss  L(q) = E[ρ_τ(Y − q)]`);

      svg
        .append('text')
        .attr('x', margin.left + w / 2)
        .attr('y', PANEL_HEIGHT - 6)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '10px')
        .text(
          `q   (red = ξ_τ ≈ ${fmt(panels.trueQ)},  blue dashed = empirical argmin ${fmt(panels.empiricalArgmin)})`,
        );
    },
    [panels, tau, panelWidth],
  );

  // ── Panel 3: gradient L'(q) = F(q) − τ ────────────────────────────────
  const gradRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const margin = { top: 28, right: 12, bottom: 36, left: 38 };
      const w = panelWidth - margin.left - margin.right;
      const h = PANEL_HEIGHT - margin.top - margin.bottom;
      const g = svg
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([panels.qLo, panels.qHi]).range([0, w]);
      const yScale = d3.scaleLinear().domain([-1, 1]).range([h, 0]);

      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text)')
        .style('font-size', '10px');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text)')
        .style('font-size', '10px');
      g.selectAll('path.domain, .tick line').style('stroke', 'var(--color-border)');

      // Zero reference line
      g.append('line')
        .attr('x1', 0)
        .attr('x2', w)
        .attr('y1', yScale(0))
        .attr('y2', yScale(0))
        .style('stroke', 'var(--color-border)');

      // Gradient curve
      const line = d3
        .line<number>()
        .x((i) => xScale(panels.qGrid[i]))
        .y((i) => yScale(panels.grad[i]));
      g.append('path')
        .datum(d3.range(N_GRID))
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', BLUE)
        .style('stroke-width', 2);

      // True quantile (solid red)
      if (panels.trueQ >= panels.qLo && panels.trueQ <= panels.qHi) {
        g.append('line')
          .attr('x1', xScale(panels.trueQ))
          .attr('x2', xScale(panels.trueQ))
          .attr('y1', 0)
          .attr('y2', h)
          .style('stroke', RED)
          .style('stroke-width', 1.4);
      }

      svg
        .append('text')
        .attr('x', margin.left + w / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '11.5px')
        .style('font-family', 'var(--font-sans)')
        .text(`Gradient  L′(q) = F(q) − τ`);

      svg
        .append('text')
        .attr('x', margin.left + w / 2)
        .attr('y', PANEL_HEIGHT - 6)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '10px')
        .text('zero crossing at q = ξ_τ');
    },
    [panels, tau, panelWidth],
  );

  return (
    <div ref={containerRef} className="not-prose">
      <div
        style={{
          marginBottom: '0.75rem',
          display: 'flex',
          flexWrap: 'wrap',
          gap: '1rem',
          alignItems: 'center',
          fontFamily: 'var(--font-sans)',
        }}
      >
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            flex: '1 1 240px',
          }}
        >
          <span style={{ fontSize: '12px', color: 'var(--color-text-secondary)', minWidth: '1.5em' }}>τ</span>
          <input
            type="range"
            min={0.05}
            max={0.95}
            step={0.01}
            value={tau}
            onChange={(e) => setTau(Number(e.target.value))}
            style={{ flex: 1 }}
            aria-label="Pinball-loss τ slider"
          />
          <span
            style={{
              fontSize: '12px',
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
          <span style={{ fontSize: '12px', color: 'var(--color-text-secondary)' }}>Y ∼</span>
          <select
            value={dist}
            onChange={(e) => setDist(e.target.value as Distribution)}
            style={{
              fontSize: '12px',
              padding: '3px 6px',
              background: 'var(--color-surface)',
              color: 'var(--color-text)',
              border: '1px solid var(--color-border)',
              borderRadius: 4,
            }}
            aria-label="Distribution of Y"
          >
            <option value="normal">Normal(0, 1)</option>
            <option value="exponential">Exponential(1)</option>
            <option value="uniform">Uniform[0, 1]</option>
          </select>
        </label>
      </div>
      <div
        style={{
          display: 'flex',
          flexDirection: isStacked ? 'column' : 'row',
          gap: isStacked ? '0.75rem' : '0.25rem',
        }}
      >
        <svg ref={lossRef} width={panelWidth} height={PANEL_HEIGHT} role="img" aria-label="Pinball loss V at the chosen τ" />
        <svg ref={lqRef} width={panelWidth} height={PANEL_HEIGHT} role="img" aria-label="Empirical expected pinball loss with minimisers" />
        <svg ref={gradRef} width={panelWidth} height={PANEL_HEIGHT} role="img" aria-label="Gradient F(q) − τ with zero crossing at the τ-quantile" />
      </div>
    </div>
  );
}
