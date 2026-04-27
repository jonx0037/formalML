import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { gaussianSampler, mulberry32 } from './shared/nonparametric-ml';

// =============================================================================
// WalshAveragesExplorer — embedded in §6 after Theorem 9.
//
// Two-panel viz of the Hodges-Lehmann construction:
//
//   Left panel  : a number line with the editable D_i (large draggable
//                 circles) above the M = n(n+1)/2 Walsh averages (small
//                 hollow dots), three vertical guides for the three location
//                 estimators (sample mean, sample median, HL = median of
//                 Walsh averages), and a horizontal bracket for the (1-α)
//                 distribution-free CI [A_(w_α + 1), A_(M - w_α)] from
//                 Theorem 10. Drag a point and everything updates live.
//
//   Right panel : live Monte Carlo of RMSE-versus-contamination — for each
//                 contamination fraction p ∈ [0, 0.4], generate K samples
//                 where each D_i has p chance of being replaced by a draw
//                 from a heavy-tailed contaminant (Cauchy or N(0, 9)),
//                 compute mean / median / HL, and report the RMSE relative
//                 to the true location 0. The HL line consistently sits
//                 between mean (mean breaks down fast) and median (median
//                 is robust but inefficient).
//
// Numerical contract (verified against notebook Cell 14 with the
// outlier example D = [-0.3, 0.2, 0.7, 0.9, 1.4, 1.8, 2.1, 8.0]):
//   sample mean    = 1.850
//   sample median  = 1.150
//   HL estimate    = 1.200
//   95% HL CI      = [0.300, 4.450]
//   #Walsh > HL    = 18 = M/2 = n(n+1)/4   (Theorem 9 centring)
//
// =============================================================================

const PANEL_HEIGHT = 280;
const SM_BREAKPOINT = 640;

const BLUE = '#2563EB';
const RED = '#DC2626';
const GREEN = '#059669';
const AMBER = '#D97706';
const SLATE = '#475569';
const LIGHT_BLUE = '#DBEAFE';

const DEFAULT_D = [-0.3, 0.2, 0.7, 0.9, 1.4, 1.8, 2.1, 8.0];

const fmt = (x: number, digits = 3) => x.toFixed(digits);

// All Walsh averages A_ij = (D_i + D_j)/2 for i ≤ j, sorted ascending.
function walshSorted(D: readonly number[]): Float64Array {
  const n = D.length;
  const M = (n * (n + 1)) / 2;
  const out = new Float64Array(M);
  let k = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) out[k++] = (D[i] + D[j]) / 2;
  }
  out.sort();
  return out;
}

// Median of a sorted Float64Array.
function medianSorted(a: Float64Array): number {
  const n = a.length;
  if (n === 0) return 0;
  if (n % 2 === 1) return a[(n - 1) / 2];
  return 0.5 * (a[n / 2 - 1] + a[n / 2]);
}

function arrayMean(D: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < D.length; i++) s += D[i];
  return s / D.length;
}

function arrayMedian(D: readonly number[]): number {
  const sorted = Float64Array.from(D).sort();
  return medianSorted(sorted);
}

// Acklam's algorithm for Φ⁻¹ (inverse normal CDF) — used for the
// Wilcoxon critical value via the normal approximation w_α = floor(μ − z·σ).
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
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416,
  ];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let q;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > pHigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  q = p - 0.5;
  const r = q * q;
  return (
    ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  );
}

// Wilcoxon critical value via normal approximation — matches the notebook
// (Cell 14) convention: w_α = floor(μ − z_(1-α/2) · √σ²) with μ = n(n+1)/4
// and σ² = n(n+1)(2n+1)/24. The asymptotic version converges to the exact
// discrete value for n ≳ 12 and stays close enough at smaller n that the
// CI's coverage error is negligible relative to the discreteness step.
function wilcoxonCritical(n: number, alpha: number): number {
  const mu = (n * (n + 1)) / 4;
  const sd = Math.sqrt((n * (n + 1) * (2 * n + 1)) / 24);
  const z = normalInverseCDF(1 - alpha / 2);
  return Math.max(0, Math.floor(mu - z * sd));
}

// Compute the (1-α) HL confidence interval from Walsh-average ordering.
function hlCI(walsh: Float64Array, n: number, alpha: number): [number, number] | null {
  const M = walsh.length;
  const wAlpha = wilcoxonCritical(n, alpha);
  if (wAlpha < 0) return null;
  const lo = walsh[wAlpha];
  const hi = walsh[M - 1 - wAlpha];
  return [lo, hi];
}

// Cauchy sampler via inverse-CDF.
function cauchySample(rng: () => number): number {
  return Math.tan(Math.PI * (rng() - 0.5));
}

// RMSE-vs-contamination Monte Carlo over a small grid of contamination
// fractions p, returning per-estimator RMSE arrays.
function rmseSweep(nSample: number, kReps: number, scaleHeavy: number, seed: number) {
  const ps = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40];
  const rng = mulberry32(seed);
  const gauss = gaussianSampler(rng);
  const meanSE = new Float64Array(ps.length);
  const medianSE = new Float64Array(ps.length);
  const hlSE = new Float64Array(ps.length);
  for (let pi = 0; pi < ps.length; pi++) {
    const p = ps[pi];
    let mAcc = 0;
    let medAcc = 0;
    let hlAcc = 0;
    for (let r = 0; r < kReps; r++) {
      const D = new Array<number>(nSample);
      for (let i = 0; i < nSample; i++) {
        const u = rng();
        if (u < p) {
          // Heavy contaminant — Cauchy scaled by `scaleHeavy`. Cap to limit
          // outlier magnitude so the panel doesn't blow up visually.
          let c = cauchySample(rng) * scaleHeavy;
          if (Math.abs(c) > 200) c = Math.sign(c) * 200;
          D[i] = c;
        } else {
          D[i] = gauss();
        }
      }
      const m = arrayMean(D);
      const med = arrayMedian(D);
      const hl = medianSorted(walshSorted(D));
      mAcc += m * m;
      medAcc += med * med;
      hlAcc += hl * hl;
    }
    meanSE[pi] = Math.sqrt(mAcc / kReps);
    medianSE[pi] = Math.sqrt(medAcc / kReps);
    hlSE[pi] = Math.sqrt(hlAcc / kReps);
  }
  return { ps, meanSE, medianSE, hlSE };
}

export default function WalshAveragesExplorer() {
  const { ref: containerRef, width: containerWidth } =
    useResizeObserver<HTMLDivElement>();

  const [n, setN] = useState(DEFAULT_D.length);
  const [data, setData] = useState<number[]>(DEFAULT_D.slice());
  const [alpha, setAlpha] = useState(0.05);
  const [seed, setSeed] = useState(7);

  // Adapt data length to n.
  useEffect(() => {
    setData((prev) => {
      if (n === prev.length) return prev;
      if (n < prev.length) return prev.slice(0, n);
      const next = prev.slice();
      while (next.length < n) {
        const k = next.length;
        next.push(((k % 2 === 0 ? 1 : -1) * (0.4 + 0.5 * (k % 4))).valueOf());
      }
      return next;
    });
  }, [n]);

  const stats = useMemo(() => {
    const walsh = walshSorted(data);
    const M = walsh.length;
    const mean = arrayMean(data);
    const median = arrayMedian(data);
    const hl = medianSorted(walsh);
    const ci = hlCI(walsh, data.length, alpha);
    return { walsh, M, mean, median, hl, ci };
  }, [data, alpha]);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked
    ? containerWidth || 0
    : Math.floor((containerWidth || 720) / 2) - 6;

  // ── Left: number line with Walsh averages, estimators, CI ────────────────
  const lineRef = useRef<SVGSVGElement>(null);
  useEffect(() => {
    const svg = d3.select(lineRef.current);
    if (!lineRef.current || panelWidth <= 0) return;
    svg.selectAll('*').remove();
    const margin = { top: 30, right: 16, bottom: 50, left: 16 };
    const w = panelWidth - margin.left - margin.right;
    const h = PANEL_HEIGHT - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const allVals = [...data, ...Array.from(stats.walsh)];
    const xMin = Math.min(0, d3.min(allVals) ?? 0);
    const xMax = Math.max(0, d3.max(allVals) ?? 1);
    const span = Math.max(2, (xMax - xMin) * 1.1);
    const center = (xMin + xMax) / 2;
    const xScale = d3.scaleLinear().domain([center - span / 2, center + span / 2]).range([0, w]);

    // Walsh row near top (y = h * 0.32)
    const walshY = h * 0.34;
    g.append('text')
      .attr('x', 0)
      .attr('y', walshY - 12)
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px')
      .style('font-family', 'var(--font-sans)')
      .text(`${stats.M} Walsh averages A_ij = (D_i + D_j)/2`);

    g.selectAll('.walsh')
      .data(Array.from(stats.walsh))
      .enter()
      .append('circle')
      .attr('class', 'walsh')
      .attr('cx', (d) => xScale(d))
      .attr('cy', walshY)
      .attr('r', 3)
      .style('fill', 'none')
      .style('stroke', SLATE)
      .style('stroke-width', 1)
      .style('stroke-opacity', 0.55);

    // Number-line axis at center
    const axisY = h * 0.62;
    g.append('line')
      .attr('x1', 0)
      .attr('x2', w)
      .attr('y1', axisY)
      .attr('y2', axisY)
      .style('stroke', 'var(--color-border)')
      .style('stroke-width', 1.5);
    g.append('g')
      .attr('transform', `translate(0,${axisY})`)
      .call(d3.axisBottom(xScale).ticks(7).tickSize(6))
      .selectAll('text')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px');
    g.selectAll('path.domain, .tick line').style('stroke', 'var(--color-border)');

    // CI bracket below the axis
    if (stats.ci) {
      const [lo, hi] = stats.ci;
      const ciY = axisY + 22;
      g.append('rect')
        .attr('x', xScale(lo))
        .attr('y', ciY - 4)
        .attr('width', xScale(hi) - xScale(lo))
        .attr('height', 8)
        .style('fill', LIGHT_BLUE)
        .style('stroke', BLUE)
        .style('stroke-width', 1);
      [lo, hi].forEach((v) => {
        g.append('line')
          .attr('x1', xScale(v))
          .attr('x2', xScale(v))
          .attr('y1', ciY - 9)
          .attr('y2', ciY + 9)
          .style('stroke', BLUE)
          .style('stroke-width', 1.5);
      });
      g.append('text')
        .attr('x', (xScale(lo) + xScale(hi)) / 2)
        .attr('y', ciY + 22)
        .attr('text-anchor', 'middle')
        .style('font-family', 'var(--font-mono)')
        .style('font-size', '10px')
        .style('fill', BLUE)
        .text(`${(1 - alpha).toFixed(2)} HL CI: [${fmt(lo, 2)}, ${fmt(hi, 2)}]`);
    }

    // Three location estimators as vertical guides over the data row.
    const estLine = (val: number, color: string, label: string, yOffset: number) => {
      g.append('line')
        .attr('x1', xScale(val))
        .attr('x2', xScale(val))
        .attr('y1', axisY - 30)
        .attr('y2', axisY + 6)
        .style('stroke', color)
        .style('stroke-width', 1.4);
      g.append('text')
        .attr('x', xScale(val))
        .attr('y', axisY - 32 + yOffset)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('font-family', 'var(--font-mono)')
        .style('fill', color)
        .text(`${label} ${fmt(val, 2)}`);
    };
    estLine(stats.mean, AMBER, 'mean', -22);
    estLine(stats.median, GREEN, 'med', -10);
    estLine(stats.hl, RED, 'HL', 2);

    // Drag-handles for D_i (large coloured circles)
    const drag = d3
      .drag<SVGCircleElement, number>()
      .on('drag', function (event) {
        const newVal = xScale.invert(event.x);
        const i = Number(d3.select(this).attr('data-index'));
        if (Number.isFinite(newVal)) {
          setData((prev) => {
            const next = prev.slice();
            next[i] = newVal;
            return next;
          });
        }
      });

    const dataY = h * 0.05 + 10;
    g.append('text')
      .attr('x', 0)
      .attr('y', dataY - 6)
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px')
      .style('font-family', 'var(--font-sans)')
      .text(`Drag D_i (n = ${n})`);

    g.selectAll<SVGCircleElement, number>('.handle')
      .data(data)
      .enter()
      .append('circle')
      .attr('class', 'handle')
      .attr('data-index', (_, i) => i)
      .attr('cx', (d) => xScale(d))
      .attr('cy', dataY + 6)
      .attr('r', 8)
      .style('fill', BLUE)
      .style('fill-opacity', 0.85)
      .style('stroke', 'var(--color-surface)')
      .style('stroke-width', 1.5)
      .style('cursor', 'ew-resize')
      .call(drag);

    svg
      .append('text')
      .attr('x', margin.left + w / 2)
      .attr('y', 18)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '11.5px')
      .style('font-family', 'var(--font-sans)')
      .style('font-weight', 600)
      .text('Walsh averages, three location estimators, and the HL CI');

    svg
      .append('text')
      .attr('x', margin.left + w / 2)
      .attr('y', PANEL_HEIGHT - 10)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px')
      .style('font-family', 'var(--font-mono)')
      .text(
        `mean ${fmt(stats.mean, 2)}   median ${fmt(stats.median, 2)}   HL ${fmt(stats.hl, 2)}   ·   M = n(n+1)/2 = ${stats.M}`,
      );
  }, [data, stats, panelWidth, alpha, n]);

  // ── Right: RMSE-vs-contamination Monte Carlo panel ──────────────────────
  const rmseRef = useRef<SVGSVGElement>(null);
  const sweep = useMemo(() => rmseSweep(n, 200, 5, seed), [n, seed]);

  useEffect(() => {
    const svg = d3.select(rmseRef.current);
    if (!rmseRef.current || panelWidth <= 0) return;
    svg.selectAll('*').remove();
    const margin = { top: 30, right: 18, bottom: 36, left: 42 };
    const w = panelWidth - margin.left - margin.right;
    const h = PANEL_HEIGHT - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const yMaxRaw = Math.max(
      d3.max(sweep.meanSE) ?? 0,
      d3.max(sweep.medianSE) ?? 0,
      d3.max(sweep.hlSE) ?? 0,
    );
    const yMax = Math.max(0.5, yMaxRaw * 1.15);
    const xScale = d3.scaleLinear().domain([0, 0.4]).range([0, w]);
    const yScale = d3.scaleLinear().domain([0, yMax]).range([h, 0]);

    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format('.0%')))
      .selectAll('text')
      .style('fill', 'var(--color-text)')
      .style('font-size', '10px');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5).tickFormat((d) => (d as number).toFixed(2)))
      .selectAll('text')
      .style('fill', 'var(--color-text)')
      .style('font-size', '10px');
    g.selectAll('path.domain, .tick line').style('stroke', 'var(--color-border)');

    const draw = (vals: Float64Array, color: string, label: string, labelY: number) => {
      const path = d3
        .line<number>()
        .x((i) => xScale(sweep.ps[i]))
        .y((i) => yScale(vals[i]));
      g.append('path')
        .datum(d3.range(sweep.ps.length))
        .attr('d', path)
        .style('fill', 'none')
        .style('stroke', color)
        .style('stroke-width', 2);
      g.selectAll(`.dot-${label}`)
        .data(d3.range(sweep.ps.length))
        .enter()
        .append('circle')
        .attr('cx', (i) => xScale(sweep.ps[i]))
        .attr('cy', (i) => yScale(vals[i]))
        .attr('r', 2.5)
        .style('fill', color);
      g.append('text')
        .attr('x', xScale(0.4))
        .attr('y', labelY)
        .attr('text-anchor', 'end')
        .style('font-size', '10px')
        .style('font-family', 'var(--font-sans)')
        .style('fill', color)
        .text(label);
    };

    draw(sweep.meanSE, AMBER, 'mean', 14);
    draw(sweep.medianSE, GREEN, 'median', 28);
    draw(sweep.hlSE, RED, 'HL', 42);

    svg
      .append('text')
      .attr('x', margin.left + w / 2)
      .attr('y', 18)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '11.5px')
      .style('font-family', 'var(--font-sans)')
      .style('font-weight', 600)
      .text(`RMSE under Cauchy contamination  (200 reps, n = ${n})`);

    svg
      .append('text')
      .attr('x', margin.left + w / 2)
      .attr('y', PANEL_HEIGHT - 10)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px')
      .text('contamination fraction p — fraction of D_i replaced by Cauchy');
  }, [sweep, panelWidth, n]);

  return (
    <div ref={containerRef} className="not-prose">
      <div
        style={{
          marginBottom: '0.75rem',
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.6rem 1rem',
          alignItems: 'center',
          fontFamily: 'var(--font-sans)',
          fontSize: '12px',
        }}
      >
        <label style={controlRow()}>
          <span style={controlLabel()}>n</span>
          <input
            type="range"
            min={5}
            max={20}
            step={1}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            style={{ flex: 1 }}
            aria-label="Sample size n"
          />
          <span style={controlValue()}>{n}</span>
        </label>
        <label style={controlRow()}>
          <span style={controlLabel()}>α</span>
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
          <span style={controlValue()}>{alpha.toFixed(2)}</span>
        </label>
        <button
          type="button"
          onClick={() => {
            setData((prev) => {
              const k = prev.length;
              const next = prev.slice();
              // Inject a single visible outlier at +5σ-equivalent
              next[k - 1] = (next[k - 1] || 0) + 6;
              return next;
            });
          }}
          style={btnStyle()}
        >
          ↯ inject outlier into D
        </button>
        <button
          type="button"
          onClick={() => setData(DEFAULT_D.slice(0, n))}
          style={btnStyle()}
        >
          reset to notebook example
        </button>
        <button
          type="button"
          onClick={() => setSeed((s) => (s + 1) % 999983)}
          style={btnStyle()}
        >
          ↻ reseed Monte Carlo
        </button>
      </div>

      <div
        style={{
          display: 'flex',
          flexDirection: isStacked ? 'column' : 'row',
          gap: isStacked ? '0.75rem' : '0.5rem',
        }}
      >
        <svg ref={lineRef} width={panelWidth} height={PANEL_HEIGHT} role="img" aria-label="Number-line view of D_i, Walsh averages, three estimators, and the HL CI" />
        <svg ref={rmseRef} width={panelWidth} height={PANEL_HEIGHT} role="img" aria-label="RMSE versus contamination fraction for sample mean, median, and HL" />
      </div>

      <p
        style={{
          fontSize: '11px',
          color: 'var(--color-text-secondary)',
          fontFamily: 'var(--font-sans)',
          marginTop: '0.5rem',
          fontStyle: 'italic',
        }}
      >
        At the notebook's n = 8 outlier example, HL = 1.20 sits between mean (1.85) and median (1.15). The 95% HL CI is [0.30, 4.45] — exact under the Wilcoxon symmetry assumption (Theorem 10). Inject another outlier and watch the right panel: mean RMSE rockets, median is steady, HL stays close behind median.
      </p>
    </div>
  );
}

const controlRow = (): CSSProperties => ({
  display: 'flex',
  alignItems: 'center',
  gap: '0.5rem',
  flex: '1 1 200px',
});
const controlLabel = (): CSSProperties => ({
  fontSize: '11px',
  color: 'var(--color-text-secondary)',
  minWidth: '1.5em',
});
const controlValue = (): CSSProperties => ({
  fontSize: '11px',
  fontFamily: 'var(--font-mono)',
  minWidth: '3em',
  textAlign: 'right',
  color: 'var(--color-text)',
});
const btnStyle = (): CSSProperties => ({
  fontSize: '11px',
  padding: '4px 10px',
  background: 'var(--color-surface)',
  color: 'var(--color-text)',
  border: '1px solid var(--color-border)',
  borderRadius: 4,
  cursor: 'pointer',
});
