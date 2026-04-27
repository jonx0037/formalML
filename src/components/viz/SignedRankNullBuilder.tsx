import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// SignedRankNullBuilder — embedded in §1 after Theorem 2.
//
// Two-panel demonstration of how the Wilcoxon signed-rank null distribution
// emerges from finite-group enumeration:
//
//   Left panel  : number-line of paired differences D_i with circular handles.
//                 Drag any D_i to update its value; the histogram on the right
//                 redraws live. Vertical "0" reference and absolute-rank ticks
//                 anchor the geometry.
//
//   Right panel : histogram of W^+ over all 2^n sign vectors B ∈ {0,1}^n with
//                 W^+ = Σ i · B_i (Wilcoxon mode) or N^+ from Binomial(n, 1/2)
//                 (sign-test mode). The observed value is a red dashed line;
//                 the two-sided rejection region |W^+ − μ| ≥ |W^+_obs − μ| is
//                 shaded in light red.
//
// Numerical contract (verified against notebook Cell 4 at n = 8):
//   E[W^+] = n(n+1)/4              → 18.0
//   Var(W^+) = n(n+1)(2n+1)/24     → 51.0
//   exact two-sided p (W^+ = 28)   ≈ 0.195
//   exact two-sided sign-test p    ≈ 0.727  (5 of 8 positive)
//
// n is capped at 15 because 2^15 = 32 768 enumerations is the sweet spot
// between completeness and a sub-100 ms recompute on a mid-range laptop.
// =============================================================================

const PANEL_HEIGHT = 280;
const SM_BREAKPOINT = 640;

// Site palette — matches the notebook's plotting colors.
const BLUE = '#2563EB';
const RED = '#DC2626';
const GREEN = '#059669';
const SLATE = '#475569';
const LIGHT_RED = '#FEE2E2';
const LIGHT_BLUE = '#DBEAFE';

type TestMode = 'wilcoxon' | 'sign';

const fmt = (x: number, digits = 2) => x.toFixed(digits);

// Default data — the n = 8 paired-difference example from notebook Cell 4.
const DEFAULT_D = [-0.4, 1.2, 2.8, -0.1, 3.7, 0.6, -1.5, 2.0];

// Stable-sort rank without ties (ties not expected in interactive editing,
// but we use midrank as a safety against accidental coincidence).
function rankAbs(D: readonly number[]): Float64Array {
  const n = D.length;
  const indexed: { v: number; i: number }[] = Array.from({ length: n }, (_, i) => ({
    v: Math.abs(D[i]),
    i,
  }));
  indexed.sort((a, b) => a.v - b.v);
  const ranks = new Float64Array(n);
  let k = 0;
  while (k < n) {
    let kEnd = k + 1;
    while (kEnd < n && indexed[kEnd].v === indexed[k].v) kEnd++;
    const midrank = (k + kEnd + 1) / 2;
    for (let j = k; j < kEnd; j++) ranks[indexed[j].i] = midrank;
    k = kEnd;
  }
  return ranks;
}

// Enumerate the full Wilcoxon null over 2^n sign vectors, returning a
// histogram (counts indexed 0 .. n(n+1)/2). For n ≤ 15, this is < 33 k ops.
function enumerateWilcoxonNull(n: number): { counts: Int32Array; total: number } {
  const Mmax = (n * (n + 1)) / 2;
  const counts = new Int32Array(Mmax + 1);
  const total = 1 << n;
  for (let b = 0; b < total; b++) {
    let w = 0;
    for (let i = 0; i < n; i++) if ((b >> i) & 1) w += i + 1;
    counts[w]++;
  }
  return { counts, total };
}

// Binomial(n, 1/2) PMF as integer counts over 2^n sign vectors so the same
// histogram code paints both modes.
function enumerateSignNull(n: number): { counts: Int32Array; total: number } {
  const counts = new Int32Array(n + 1);
  // Pascal's triangle row n: C(n, k)
  let prev = new Int32Array(n + 1);
  prev[0] = 1;
  for (let row = 1; row <= n; row++) {
    const next = new Int32Array(n + 1);
    next[0] = 1;
    for (let j = 1; j <= row; j++) next[j] = prev[j - 1] + (prev[j] ?? 0);
    prev = next;
  }
  for (let k = 0; k <= n; k++) counts[k] = prev[k];
  return { counts, total: 1 << n };
}

export default function SignedRankNullBuilder() {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();

  const [n, setN] = useState(DEFAULT_D.length);
  const [data, setData] = useState<number[]>(DEFAULT_D.slice());
  const [mode, setMode] = useState<TestMode>('wilcoxon');
  const [alpha, setAlpha] = useState(0.05);

  // When n changes, grow or shrink data while preserving existing entries.
  useEffect(() => {
    setData((prev) => {
      if (n === prev.length) return prev;
      if (n < prev.length) return prev.slice(0, n);
      // Grow: add evenly spaced new values from a small Gaussian-ish ladder.
      const next = prev.slice();
      while (next.length < n) {
        const k = next.length;
        next.push(((k % 2 === 0 ? 1 : -1) * (1 + (k % 3) * 0.4)).valueOf());
      }
      return next;
    });
  }, [n]);

  // Observed statistics
  const stats = useMemo(() => {
    const ranks = rankAbs(data);
    let Wplus = 0;
    let Nplus = 0;
    for (let i = 0; i < data.length; i++) {
      if (data[i] > 0) {
        Wplus += ranks[i];
        Nplus += 1;
      }
    }
    return { ranks, Wplus, Nplus };
  }, [data]);

  // Null distribution (cached on n + mode)
  const nullDist = useMemo(() => {
    return mode === 'wilcoxon' ? enumerateWilcoxonNull(n) : enumerateSignNull(n);
  }, [n, mode]);

  // Two-sided p-value relative to the null mean.
  const pAndRegion = useMemo(() => {
    const { counts, total } = nullDist;
    const isWilcox = mode === 'wilcoxon';
    const obs = isWilcox ? stats.Wplus : stats.Nplus;
    const mu = isWilcox ? (n * (n + 1)) / 4 : n / 2;
    const dev = Math.abs(obs - mu);
    let pCount = 0;
    for (let k = 0; k < counts.length; k++) {
      if (Math.abs(k - mu) >= dev - 1e-9) pCount += counts[k];
    }
    const p = pCount / total;
    // Critical value: smallest deviation d such that P(|S - mu| >= d) <= alpha
    // (for shading the rejection region by alpha).
    let crit = Infinity;
    let cumExtreme = 0;
    const sortedDev = Array.from({ length: counts.length }, (_, k) => ({
      k,
      dev: Math.abs(k - mu),
      ct: counts[k],
    })).sort((a, b) => b.dev - a.dev);
    for (const e of sortedDev) {
      if ((cumExtreme + e.ct) / total > alpha) break;
      cumExtreme += e.ct;
      crit = e.dev;
    }
    return { p, mu, dev, crit, total };
  }, [nullDist, mode, stats, n, alpha]);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked
    ? containerWidth || 0
    : Math.floor((containerWidth || 700) / 2) - 4;

  // ── Left panel: editable D_i on a number line ─────────────────────────────
  const numLineRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = d3.select(numLineRef.current);
    if (!numLineRef.current || panelWidth <= 0) return;
    svg.selectAll('*').remove();
    const margin = { top: 28, right: 16, bottom: 50, left: 16 };
    const w = panelWidth - margin.left - margin.right;
    const h = PANEL_HEIGHT - margin.top - margin.bottom;
    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const maxAbs = Math.max(4, Math.max(...data.map((d) => Math.abs(d))) * 1.15);
    const xScale = d3.scaleLinear().domain([-maxAbs, maxAbs]).range([0, w]);

    // Background bands for negative / positive halves.
    g.append('rect')
      .attr('x', 0)
      .attr('y', h / 2 - 24)
      .attr('width', xScale(0))
      .attr('height', 48)
      .style('fill', LIGHT_RED)
      .style('opacity', 0.55);
    g.append('rect')
      .attr('x', xScale(0))
      .attr('y', h / 2 - 24)
      .attr('width', w - xScale(0))
      .attr('height', 48)
      .style('fill', LIGHT_BLUE)
      .style('opacity', 0.55);

    // Number-line axis at center
    g.append('line')
      .attr('x1', 0)
      .attr('x2', w)
      .attr('y1', h / 2)
      .attr('y2', h / 2)
      .style('stroke', 'var(--color-border)')
      .style('stroke-width', 1.5);

    // Tick marks at integers
    const tickAxis = d3.axisBottom(xScale).ticks(7).tickSize(6);
    g.append('g')
      .attr('transform', `translate(0,${h / 2})`)
      .call(tickAxis)
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px');
    g.selectAll('path.domain, .tick line').style('stroke', 'var(--color-border)');

    // Zero marker
    g.append('line')
      .attr('x1', xScale(0))
      .attr('x2', xScale(0))
      .attr('y1', h / 2 - 32)
      .attr('y2', h / 2 + 6)
      .style('stroke', SLATE)
      .style('stroke-width', 1.4);
    g.append('text')
      .attr('x', xScale(0))
      .attr('y', h / 2 - 38)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px')
      .text('0');

    // Drag handles
    const drag = d3
      .drag<SVGGElement, number>()
      .on('drag', function (event, d) {
        const newVal = xScale.invert(event.x);
        const i = Number(d3.select(this).attr('data-index'));
        if (Number.isFinite(newVal)) {
          setData((prev) => {
            const next = prev.slice();
            next[i] = Math.max(-maxAbs, Math.min(maxAbs, newVal));
            return next;
          });
        }
      });

    const handles = g
      .selectAll<SVGGElement, number>('.handle')
      .data(data)
      .enter()
      .append('g')
      .attr('class', 'handle')
      .attr('data-index', (_, i) => i)
      .attr('transform', (d) => `translate(${xScale(d)},${h / 2})`)
      .style('cursor', 'ew-resize')
      .call(drag);

    handles
      .append('circle')
      .attr('r', 9)
      .style('fill', (d) => (d > 0 ? BLUE : RED))
      .style('fill-opacity', 0.9)
      .style('stroke', 'var(--color-surface)')
      .style('stroke-width', 2);

    handles
      .append('text')
      .attr('y', -14)
      .attr('text-anchor', 'middle')
      .style('font-family', 'var(--font-mono)')
      .style('font-size', '9.5px')
      .style('fill', (d) => (d > 0 ? BLUE : RED))
      .text((_, i) => `D₍${i + 1}₎`);

    handles
      .append('text')
      .attr('y', 22)
      .attr('text-anchor', 'middle')
      .style('font-family', 'var(--font-mono)')
      .style('font-size', '9.5px')
      .style('fill', 'var(--color-text-secondary)')
      .text((_, i) => `R=${stats.ranks[i].toFixed(0)}`);

    // Title
    svg
      .append('text')
      .attr('x', margin.left + w / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '11.5px')
      .style('font-family', 'var(--font-sans)')
      .text(`Drag the D_i — sign and absolute rank update live (n = ${n})`);

    svg
      .append('text')
      .attr('x', margin.left + w / 2)
      .attr('y', PANEL_HEIGHT - 22)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px')
      .text(
        mode === 'wilcoxon'
          ? `W⁺ = sum of R at positive D = ${fmt(stats.Wplus, 1)}   ·   N⁺ = ${stats.Nplus} of ${n}`
          : `N⁺ = ${stats.Nplus} of ${n}   ·   W⁺ = ${fmt(stats.Wplus, 1)} (off mode)`,
      );
  }, [data, panelWidth, n, mode, stats]);

  // ── Right panel: histogram of the null with rejection shading ─────────────
  const histRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = d3.select(histRef.current);
    if (!histRef.current || panelWidth <= 0) return;
    svg.selectAll('*').remove();
    const margin = { top: 28, right: 14, bottom: 50, left: 40 };
    const w = panelWidth - margin.left - margin.right;
    const h = PANEL_HEIGHT - margin.top - margin.bottom;
    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const { counts, total } = nullDist;
    const isWilcox = mode === 'wilcoxon';
    const xMax = counts.length - 1;
    const probs = Array.from(counts, (c) => c / total);
    const yMax = Math.max(...probs) * 1.1;
    const obs = isWilcox ? stats.Wplus : stats.Nplus;
    const { mu, crit } = pAndRegion;

    const xScale = d3.scaleLinear().domain([0, xMax]).range([0, w]);
    const yScale = d3.scaleLinear().domain([0, yMax]).range([h, 0]);

    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .selectAll('text')
      .style('fill', 'var(--color-text)')
      .style('font-size', '10px');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(4).tickFormat((d) => (d as number).toFixed(2)))
      .selectAll('text')
      .style('fill', 'var(--color-text)')
      .style('font-size', '10px');
    g.selectAll('path.domain, .tick line').style('stroke', 'var(--color-border)');

    // Bars
    const barWidth = Math.max(1, w / counts.length - 1.5);
    g.selectAll('rect.bar')
      .data(probs)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', (_, i) => xScale(i) - barWidth / 2)
      .attr('y', (p) => yScale(p))
      .attr('width', barWidth)
      .attr('height', (p) => h - yScale(p))
      .style('fill', (_, i) => {
        const dev = Math.abs(i - mu);
        // Rejection region (using the alpha-tuned critical deviation): paint LIGHT_RED.
        if (Number.isFinite(crit) && dev >= crit - 1e-9) return LIGHT_RED;
        return LIGHT_BLUE;
      })
      .style('stroke', (_, i) => {
        const dev = Math.abs(i - mu);
        if (Number.isFinite(crit) && dev >= crit - 1e-9) return RED;
        return BLUE;
      })
      .style('stroke-width', 0.7);

    // Observed value
    g.append('line')
      .attr('x1', xScale(obs))
      .attr('x2', xScale(obs))
      .attr('y1', 0)
      .attr('y2', h)
      .style('stroke', RED)
      .style('stroke-width', 2)
      .style('stroke-dasharray', '4,3');

    g.append('text')
      .attr('x', xScale(obs))
      .attr('y', -6)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('font-family', 'var(--font-mono)')
      .style('fill', RED)
      .text(`${isWilcox ? 'W⁺' : 'N⁺'} = ${fmt(obs, isWilcox ? 1 : 0)}`);

    // Mean
    g.append('line')
      .attr('x1', xScale(mu))
      .attr('x2', xScale(mu))
      .attr('y1', 0)
      .attr('y2', h)
      .style('stroke', SLATE)
      .style('stroke-width', 1)
      .style('stroke-dasharray', '2,3');

    // Title
    svg
      .append('text')
      .attr('x', margin.left + w / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '11.5px')
      .style('font-family', 'var(--font-sans)')
      .text(
        isWilcox
          ? `Wilcoxon null over 2^${n} = ${total} sign vectors`
          : `Sign-test null  Binomial(${n}, ½) over 2^${n} = ${total}`,
      );

    svg
      .append('text')
      .attr('x', margin.left + w / 2)
      .attr('y', PANEL_HEIGHT - 22)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px')
      .text(
        `μ = ${fmt(mu, 1)}   |   two-sided p = ${pAndRegion.p.toFixed(4)}   |   reject at α = ${alpha.toFixed(2)}: ${pAndRegion.p <= alpha ? 'YES' : 'NO'}`,
      );
  }, [nullDist, stats, mode, panelWidth, alpha, n, pAndRegion]);

  // ── Controls ──────────────────────────────────────────────────────────────
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
            flex: '1 1 220px',
          }}
        >
          <span style={{ fontSize: '12px', color: 'var(--color-text-secondary)', minWidth: '1.5em' }}>n</span>
          <input
            type="range"
            min={4}
            max={15}
            step={1}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            style={{ flex: 1 }}
            aria-label="Sample size n"
          />
          <span
            style={{
              fontSize: '12px',
              fontFamily: 'var(--font-mono)',
              minWidth: '2em',
              textAlign: 'right',
              color: 'var(--color-text)',
            }}
          >
            {n}
          </span>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '12px', color: 'var(--color-text-secondary)' }}>statistic</span>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as TestMode)}
            style={{
              fontSize: '12px',
              padding: '3px 6px',
              background: 'var(--color-surface)',
              color: 'var(--color-text)',
              border: '1px solid var(--color-border)',
              borderRadius: 4,
            }}
            aria-label="Test statistic mode"
          >
            <option value="wilcoxon">Wilcoxon W⁺</option>
            <option value="sign">Sign test N⁺</option>
          </select>
        </label>
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            flex: '1 1 180px',
          }}
        >
          <span style={{ fontSize: '12px', color: 'var(--color-text-secondary)', minWidth: '1.5em' }}>α</span>
          <input
            type="range"
            min={0.01}
            max={0.20}
            step={0.01}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
            style={{ flex: 1 }}
            aria-label="Significance level α"
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
            {alpha.toFixed(2)}
          </span>
        </label>
      </div>

      <div
        style={{
          display: 'flex',
          flexDirection: isStacked ? 'column' : 'row',
          gap: isStacked ? '0.75rem' : '0.5rem',
        }}
      >
        <svg
          ref={numLineRef}
          width={panelWidth}
          height={PANEL_HEIGHT}
          role="img"
          aria-label="Number-line view of paired differences D_i with absolute-rank labels"
        />
        <svg
          ref={histRef}
          width={panelWidth}
          height={PANEL_HEIGHT}
          role="img"
          aria-label={`${mode === 'wilcoxon' ? 'Wilcoxon W+' : 'Sign-test N+'} null distribution histogram with rejection region and observed value`}
        />
      </div>

      <p
        style={{
          fontSize: '11px',
          color: 'var(--color-text-secondary)',
          fontFamily: 'var(--font-sans)',
          marginTop: '0.75rem',
          fontStyle: 'italic',
        }}
      >
        At n = 8 the Wilcoxon null is centred at μ = n(n+1)/4 = 18 with variance n(n+1)(2n+1)/24 = 51 (Theorem 2). Drag <em>D</em>₁ across zero to flip its sign and watch which bars enter the rejection region.
      </p>
    </div>
  );
}
