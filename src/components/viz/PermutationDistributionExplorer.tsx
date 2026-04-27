import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { gaussianSampler, mulberry32 } from './shared/nonparametric-ml';

// =============================================================================
// PermutationDistributionExplorer — embedded in §2 (after Theorem 3),
// §3 (after Definition 5), and §7 (after Remark 5) of rank-tests.
//
// One component, three call sites, distinguished by `defaultStatistic`:
//   §2 ⇒ "meanDiff"       (the canonical permutation framework introduction)
//   §3 ⇒ "mannWhitneyU"   (the rank-based two-sample setup)
//   §7 ⇒ "welchT"         (raw vs ranks: validity is the same, power differs)
//
// Single-panel layout: a histogram of the permutation distribution of T(X, Y)
// across B label-swap permutations, with the observed value marked and the
// two-sided rejection region shaded by user-selected α. Below the panel,
// a compact controls strip lets you switch group distribution, shift, n_1,
// n_2, the statistic, the resampling strategy, and B.
//
// Numerical contract: when both groups are drawn iid from the same density
// and shift = 0, the permutation p-value is uniform on the achievable grid
// (Theorem 3). Empirical rejection at nominal α tracks α to within Monte
// Carlo error of B samples (notebook Cell 6 reports 0.028 / 0.084 at
// α = 0.05 / 0.10 with 800 iterations of B = 2000 resamples).
// =============================================================================

const PANEL_HEIGHT = 290;
const SM_BREAKPOINT = 640;
const ENUMERATE_CAP = 100_000;

const BLUE = '#2563EB';
const RED = '#DC2626';
const GREEN = '#059669';
const SLATE = '#475569';
const LIGHT_RED = '#FEE2E2';
const LIGHT_BLUE = '#DBEAFE';

export type PermStatistic = 'meanDiff' | 'welchT' | 'mannWhitneyU';

const STATISTIC_OPTIONS: { value: PermStatistic; label: string }[] = [
  { value: 'meanDiff', label: 'Mean difference  X̄ − Ȳ' },
  { value: 'welchT', label: 'Welch t  (X̄−Ȳ)/√(s²ₓ/n₁ + s²ᵧ/n₂)' },
  { value: 'mannWhitneyU', label: 'Mann-Whitney  U' },
];

type Distribution = 'normal' | 'lognormal' | 'exponential' | 'uniform';

const DISTRIBUTION_OPTIONS: { value: Distribution; label: string }[] = [
  { value: 'normal', label: 'Normal(0, 1)' },
  { value: 'lognormal', label: 'Log-normal(0, 1)' },
  { value: 'exponential', label: 'Exponential(1)' },
  { value: 'uniform', label: 'Uniform[0, 1]' },
];

type Strategy = 'monte-carlo' | 'enumerate';

const fmt = (x: number, digits = 3) => x.toFixed(digits);

// Sample from the chosen distribution into a Float64Array.
function sampleFrom(
  rng: () => number,
  dist: Distribution,
  n: number,
  shift: number,
): Float64Array {
  const out = new Float64Array(n);
  if (dist === 'normal') {
    const g = gaussianSampler(rng);
    for (let i = 0; i < n; i++) out[i] = g() + shift;
    return out;
  }
  if (dist === 'lognormal') {
    const g = gaussianSampler(rng);
    for (let i = 0; i < n; i++) out[i] = Math.exp(g()) + shift;
    return out;
  }
  if (dist === 'exponential') {
    for (let i = 0; i < n; i++) {
      let u = rng();
      if (u < 1e-12) u = 1e-12;
      out[i] = -Math.log(u) + shift;
    }
    return out;
  }
  for (let i = 0; i < n; i++) out[i] = rng() + shift;
  return out;
}

function meanOf(arr: Float64Array, lo: number, hi: number): number {
  let s = 0;
  for (let i = lo; i < hi; i++) s += arr[i];
  return s / (hi - lo);
}

function varianceOf(arr: Float64Array, lo: number, hi: number, mu: number): number {
  let s = 0;
  for (let i = lo; i < hi; i++) {
    const d = arr[i] - mu;
    s += d * d;
  }
  return s / (hi - lo - 1);
}

// Compute statistic on (pool[0..n1], pool[n1..N]) where pool may be a label-permuted
// version of the original concatenation [X, Y].
function computeStat(
  pool: Float64Array,
  n1: number,
  n2: number,
  stat: PermStatistic,
  ranks?: Float64Array,
): number {
  const N = n1 + n2;
  if (stat === 'meanDiff') {
    return meanOf(pool, 0, n1) - meanOf(pool, n1, N);
  }
  if (stat === 'welchT') {
    const muX = meanOf(pool, 0, n1);
    const muY = meanOf(pool, n1, N);
    const vX = varianceOf(pool, 0, n1, muX);
    const vY = varianceOf(pool, n1, N, muY);
    const denom = Math.sqrt(vX / n1 + vY / n2);
    if (denom < 1e-12) return 0;
    return (muX - muY) / denom;
  }
  // mannWhitneyU: U = #{ (i, j) : pool[i] < pool[n1 + j], i < n1, j < n2 }
  // We use the rank-based formula via precomputed pooled ranks for the
  // permutation: U_X = sum(ranks of group-1 elements) − n1(n1+1)/2.
  if (!ranks) throw new Error('ranks required for mannWhitneyU');
  let rankSum = 0;
  for (let i = 0; i < n1; i++) rankSum += ranks[i];
  return rankSum - (n1 * (n1 + 1)) / 2;
}

// Stable midrank (handles ties).
function midRank(values: Float64Array): Float64Array {
  const n = values.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => values[a] - values[b]);
  const out = new Float64Array(n);
  let k = 0;
  while (k < n) {
    let kEnd = k + 1;
    while (kEnd < n && values[idx[kEnd]] === values[idx[k]]) kEnd++;
    const r = (k + kEnd + 1) / 2;
    for (let j = k; j < kEnd; j++) out[idx[j]] = r;
    k = kEnd;
  }
  return out;
}

// In-place Fisher-Yates shuffle of pool[0..N].
function shuffle(pool: Float64Array, rng: () => number) {
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = pool[i];
    pool[i] = pool[j];
    pool[j] = tmp;
  }
}

// Iterate over all C(N, n1) subsets directly via recursive backtracking.
// Cost: O(C(N, n1)) — independent of N — and avoids JS's 32-bit bitwise
// overflow for N ≥ 31. The brute-force `for b in 0..2^N` approach (with
// popcount filter) hangs the browser at N as small as 62 even when the
// combinatorial answer is tiny (e.g., n1=2, n2=60 → 1830 subsets but 2^62
// candidate masks). Used only when C(N, n1) ≤ ENUMERATE_CAP.
function enumerateSubsets(N: number, n1: number, cb: (subset: Uint8Array) => void) {
  const mask = new Uint8Array(N);
  let count = 0;
  // Choose `pick` more elements from positions [start..N).
  function recurse(start: number, pick: number): boolean {
    if (count >= ENUMERATE_CAP) return false;
    if (pick === 0) {
      cb(mask);
      count++;
      return count < ENUMERATE_CAP;
    }
    const remaining = N - start;
    if (remaining < pick) return true;
    // Take position `start`
    mask[start] = 1;
    if (!recurse(start + 1, pick - 1)) return false;
    mask[start] = 0;
    // Skip position `start`
    if (!recurse(start + 1, pick)) return false;
    return true;
  }
  recurse(0, n1);
}

// Binomial coefficient C(N, k), computed in floating point but exact for small N.
function binom(N: number, k: number): number {
  if (k < 0 || k > N) return 0;
  k = Math.min(k, N - k);
  let r = 1;
  for (let i = 1; i <= k; i++) r = (r * (N - k + i)) / i;
  return r;
}

interface Props {
  defaultStatistic?: PermStatistic;
}

export default function PermutationDistributionExplorer({
  defaultStatistic = 'meanDiff',
}: Props) {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();

  const [n1, setN1] = useState(20);
  const [n2, setN2] = useState(20);
  const [shift, setShift] = useState(0.5);
  const [distA, setDistA] = useState<Distribution>('normal');
  const [distB, setDistB] = useState<Distribution>('normal');
  const [statistic, setStatistic] = useState<PermStatistic>(defaultStatistic);
  const [strategy, setStrategy] = useState<Strategy>('monte-carlo');
  const [B, setB] = useState(2000);
  const [alpha, setAlpha] = useState(0.05);
  const [seed, setSeed] = useState(42);

  // Generate fresh data when knobs change.
  const data = useMemo(() => {
    const rng = mulberry32(seed);
    const X = sampleFrom(rng, distA, n1, 0);
    const Y = sampleFrom(rng, distB, n2, shift);
    return { X, Y };
  }, [seed, n1, n2, distA, distB, shift]);

  // Auto-disable enumerate when too many subsets.
  const totalSubsets = useMemo(() => binom(n1 + n2, n1), [n1, n2]);
  const canEnumerate = totalSubsets <= ENUMERATE_CAP;
  const effectiveStrategy: Strategy = !canEnumerate && strategy === 'enumerate' ? 'monte-carlo' : strategy;

  // Permutation distribution.
  const perm = useMemo(() => {
    const N = n1 + n2;
    const pool = new Float64Array(N);
    for (let i = 0; i < n1; i++) pool[i] = data.X[i];
    for (let j = 0; j < n2; j++) pool[n1 + j] = data.Y[j];

    const ranks = midRank(pool);

    // Observed
    const observed = computeStat(pool, n1, n2, statistic, ranks);

    // Permutation samples
    const numSamples = effectiveStrategy === 'enumerate' ? Math.min(totalSubsets, ENUMERATE_CAP) : B;
    const stats = new Float64Array(numSamples);

    if (effectiveStrategy === 'enumerate') {
      const work = new Float64Array(N);
      const workRanks = new Float64Array(N);
      let idx = 0;
      enumerateSubsets(N, n1, (mask) => {
        if (idx >= stats.length) return;
        let p1 = 0;
        let p2 = n1;
        for (let i = 0; i < N; i++) {
          if (mask[i]) {
            work[p1] = pool[i];
            workRanks[p1] = ranks[i];
            p1++;
          } else {
            work[p2] = pool[i];
            workRanks[p2] = ranks[i];
            p2++;
          }
        }
        stats[idx++] = computeStat(work, n1, n2, statistic, workRanks);
      });
    } else {
      const rng = mulberry32(seed + 1);
      const work = new Float64Array(N);
      for (let i = 0; i < N; i++) work[i] = pool[i];
      const workRanks = new Float64Array(N);
      for (let i = 0; i < N; i++) workRanks[i] = ranks[i];
      for (let s = 0; s < B; s++) {
        // Random permutation of indices [0..N): shuffle work and workRanks
        // in lockstep.
        for (let i = N - 1; i > 0; i--) {
          const j = Math.floor(rng() * (i + 1));
          const tmp = work[i]; work[i] = work[j]; work[j] = tmp;
          const tmp2 = workRanks[i]; workRanks[i] = workRanks[j]; workRanks[j] = tmp2;
        }
        stats[s] = computeStat(work, n1, n2, statistic, workRanks);
      }
    }

    // Two-sided p-value using Phipson-Smyth correction. Compute the
    // permutation-distribution mean once outside the loop — calling
    // `mean(stats)` per iteration would be O(B²).
    const muNull = mean(stats);
    const obsAbsDev = Math.abs(observed - muNull);
    let nExtreme = 0;
    for (let s = 0; s < stats.length; s++) {
      if (Math.abs(stats[s] - muNull) >= obsAbsDev - 1e-12) nExtreme++;
    }
    const p = (1 + nExtreme) / (1 + stats.length);

    return { stats, observed, p, mean: muNull, numSamples: stats.length };
  }, [data, n1, n2, statistic, effectiveStrategy, B, seed, totalSubsets]);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked ? containerWidth || 0 : Math.max(360, (containerWidth || 720) - 16);

  // ── Histogram panel ──────────────────────────────────────────────────────
  const histRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = d3.select(histRef.current);
    if (!histRef.current || panelWidth <= 0) return;
    svg.selectAll('*').remove();
    const margin = { top: 32, right: 16, bottom: 50, left: 50 };
    const w = panelWidth - margin.left - margin.right;
    const h = PANEL_HEIGHT - margin.top - margin.bottom;
    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const { stats, observed, p, mean: muNull } = perm;
    const sMin = Math.min(d3.min(stats) ?? 0, observed);
    const sMax = Math.max(d3.max(stats) ?? 0, observed);
    const span = sMax - sMin || 1;
    const xLo = sMin - 0.05 * span;
    const xHi = sMax + 0.05 * span;

    const xScale = d3.scaleLinear().domain([xLo, xHi]).range([0, w]);
    const nBins = Math.min(60, Math.max(20, Math.floor(Math.sqrt(stats.length))));
    const binner = d3.bin().domain([xLo, xHi]).thresholds(nBins);
    const bins = binner(Array.from(stats));
    const binMax = d3.max(bins, (b) => b.length) ?? 1;
    const yScale = d3.scaleLinear().domain([0, binMax]).range([h, 0]);

    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .selectAll('text')
      .style('fill', 'var(--color-text)')
      .style('font-size', '10px');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(4))
      .selectAll('text')
      .style('fill', 'var(--color-text)')
      .style('font-size', '10px');
    g.selectAll('path.domain, .tick line').style('stroke', 'var(--color-border)');

    // Compute critical-value cutoffs by quantile of |stat - μ_null|.
    const devs = Array.from(stats, (v) => Math.abs(v - muNull)).sort(d3.ascending);
    const critIdx = Math.floor((1 - alpha) * devs.length);
    const critDev = devs[Math.min(critIdx, devs.length - 1)];

    // Bars
    g.selectAll('rect.bar')
      .data(bins)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', (b) => xScale(b.x0 ?? 0))
      .attr('y', (b) => yScale(b.length))
      .attr('width', (b) => Math.max(0, xScale(b.x1 ?? 0) - xScale(b.x0 ?? 0) - 1))
      .attr('height', (b) => h - yScale(b.length))
      .style('fill', (b) => {
        const center = ((b.x0 ?? 0) + (b.x1 ?? 0)) / 2;
        return Math.abs(center - muNull) >= critDev ? LIGHT_RED : LIGHT_BLUE;
      })
      .style('stroke', (b) => {
        const center = ((b.x0 ?? 0) + (b.x1 ?? 0)) / 2;
        return Math.abs(center - muNull) >= critDev ? RED : BLUE;
      })
      .style('stroke-width', 0.6);

    // Mean line
    g.append('line')
      .attr('x1', xScale(muNull))
      .attr('x2', xScale(muNull))
      .attr('y1', 0)
      .attr('y2', h)
      .style('stroke', SLATE)
      .style('stroke-width', 1)
      .style('stroke-dasharray', '2,3');

    // Observed line
    g.append('line')
      .attr('x1', xScale(observed))
      .attr('x2', xScale(observed))
      .attr('y1', 0)
      .attr('y2', h)
      .style('stroke', RED)
      .style('stroke-width', 2)
      .style('stroke-dasharray', '4,3');

    g.append('text')
      .attr('x', xScale(observed))
      .attr('y', -8)
      .attr('text-anchor', 'middle')
      .style('font-size', '10.5px')
      .style('font-family', 'var(--font-mono)')
      .style('fill', RED)
      .text(`T_obs = ${fmt(observed, 3)}`);

    // Labels
    svg
      .append('text')
      .attr('x', margin.left + w / 2)
      .attr('y', 18)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '12px')
      .style('font-family', 'var(--font-sans)')
      .style('font-weight', 600)
      .text(
        `${STATISTIC_OPTIONS.find((s) => s.value === statistic)?.label} — permutation null over ${perm.numSamples.toLocaleString()} ${effectiveStrategy === 'enumerate' ? 'enumerated' : 'sampled'} permutations`,
      );

    svg
      .append('text')
      .attr('x', margin.left + w / 2)
      .attr('y', PANEL_HEIGHT - 22)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '11px')
      .style('font-family', 'var(--font-mono)')
      .text(
        `μ₀ = ${fmt(muNull, 3)}    p (Phipson-Smyth, two-sided) = ${fmt(p, 4)}    α = ${alpha.toFixed(2)} ⇒ reject: ${p <= alpha ? 'YES' : 'NO'}`,
      );
  }, [perm, panelWidth, alpha, statistic, effectiveStrategy]);

  return (
    <div ref={containerRef} className="not-prose">
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: isStacked ? '1fr' : 'repeat(2, minmax(0, 1fr))',
          gap: '0.5rem 1rem',
          marginBottom: '0.75rem',
          fontFamily: 'var(--font-sans)',
          fontSize: '12px',
        }}
      >
        <label style={controlRow()}>
          <span style={controlLabel()}>statistic</span>
          <select value={statistic} onChange={(e) => setStatistic(e.target.value as PermStatistic)} style={selectStyle()}>
            {STATISTIC_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <label style={controlRow()}>
          <span style={controlLabel()}>strategy</span>
          <select
            value={effectiveStrategy}
            onChange={(e) => setStrategy(e.target.value as Strategy)}
            style={selectStyle()}
          >
            <option value="monte-carlo">Monte Carlo (B samples)</option>
            <option value="enumerate" disabled={!canEnumerate}>
              Enumerate all{!canEnumerate ? ` (disabled — C(N, n₁) = ${totalSubsets.toExponential(1)} > ${ENUMERATE_CAP.toLocaleString()})` : ''}
            </option>
          </select>
        </label>
        <label style={controlRow()}>
          <span style={controlLabel()}>group A</span>
          <select value={distA} onChange={(e) => setDistA(e.target.value as Distribution)} style={selectStyle()}>
            {DISTRIBUTION_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </label>
        <label style={controlRow()}>
          <span style={controlLabel()}>group B</span>
          <select value={distB} onChange={(e) => setDistB(e.target.value as Distribution)} style={selectStyle()}>
            {DISTRIBUTION_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </label>
        <label style={controlRow()}>
          <span style={controlLabel()}>shift Δ</span>
          <input type="range" min={-2} max={2} step={0.1} value={shift} onChange={(e) => setShift(Number(e.target.value))} style={{ flex: 1 }} aria-label="treatment shift" />
          <span style={controlValue()}>{shift.toFixed(1)}</span>
        </label>
        <label style={controlRow()}>
          <span style={controlLabel()}>α</span>
          <input type="range" min={0.01} max={0.20} step={0.01} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} style={{ flex: 1 }} aria-label="significance level alpha" />
          <span style={controlValue()}>{alpha.toFixed(2)}</span>
        </label>
        <label style={controlRow()}>
          <span style={controlLabel()}>n₁</span>
          <input type="range" min={5} max={60} step={1} value={n1} onChange={(e) => setN1(Number(e.target.value))} style={{ flex: 1 }} aria-label="group A size" />
          <span style={controlValue()}>{n1}</span>
        </label>
        <label style={controlRow()}>
          <span style={controlLabel()}>n₂</span>
          <input type="range" min={5} max={60} step={1} value={n2} onChange={(e) => setN2(Number(e.target.value))} style={{ flex: 1 }} aria-label="group B size" />
          <span style={controlValue()}>{n2}</span>
        </label>
        <label style={controlRow()}>
          <span style={controlLabel()}>B</span>
          <input
            type="range"
            min={200}
            max={5000}
            step={100}
            value={B}
            disabled={effectiveStrategy === 'enumerate'}
            onChange={(e) => setB(Number(e.target.value))}
            style={{ flex: 1, opacity: effectiveStrategy === 'enumerate' ? 0.4 : 1 }}
            aria-label="number of permutation samples B"
          />
          <span style={controlValue()}>{B.toLocaleString()}</span>
        </label>
        <button
          type="button"
          onClick={() => setSeed((s) => (s + 1) % 999983)}
          style={{
            fontSize: '11px',
            padding: '4px 10px',
            background: 'var(--color-surface)',
            color: 'var(--color-text)',
            border: '1px solid var(--color-border)',
            borderRadius: 4,
            cursor: 'pointer',
            justifySelf: 'start',
          }}
        >
          ↻ resample data
        </button>
      </div>

      <svg
        ref={histRef}
        width={panelWidth}
        height={PANEL_HEIGHT}
        role="img"
        aria-label="Permutation distribution histogram with observed value, mean reference, and rejection region shaded"
      />

      <p
        style={{
          fontSize: '11px',
          color: 'var(--color-text-secondary)',
          fontFamily: 'var(--font-sans)',
          marginTop: '0.5rem',
          fontStyle: 'italic',
        }}
      >
        The permutation framework is statistic-agnostic — same procedure, four interchangeable test statistics, validity from Theorem 3 in every case. Switch the group distribution to log-normal or shrink Δ to feel the power profile shift.
      </p>
    </div>
  );
}

function mean(arr: Float64Array): number {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

// Inline style helpers — kept here to match the controls block density.
const controlRow = (): CSSProperties => ({
  display: 'flex',
  alignItems: 'center',
  gap: '0.5rem',
});
const controlLabel = (): CSSProperties => ({
  fontSize: '11px',
  color: 'var(--color-text-secondary)',
  minWidth: '4em',
});
const controlValue = (): CSSProperties => ({
  fontSize: '11px',
  fontFamily: 'var(--font-mono)',
  minWidth: '3em',
  textAlign: 'right',
  color: 'var(--color-text)',
});
const selectStyle = (): CSSProperties => ({
  flex: 1,
  fontSize: '11px',
  padding: '3px 6px',
  background: 'var(--color-surface)',
  color: 'var(--color-text)',
  border: '1px solid var(--color-border)',
  borderRadius: 4,
});
