import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// AREExplorer — embedded in §5 after Theorem 8.
//
// Two-panel demonstration of the Pitman ARE landscape:
//
//   Left panel  : the chosen symmetric density f(x), with σ² and
//                 I(f) = ∫ f² annotated. The density curve, plus shaded ±σ
//                 spread, makes the relationship "narrow-tailed = small σ,
//                 large I; heavy-tailed = large σ, smaller-but-not-tiny I"
//                 visually concrete.
//
//   Right panel : asymptotic local power curves Φ(c · θ · √n − z₁₋α/2) for
//                 both Wilcoxon (c_W = √12 · I(f)) and the t-test
//                 (c_T = 1/σ), plotted against effect size θ. The two
//                 curves cross at θ = 0 (both have α size); the gap to the
//                 right encodes the ARE numerically.
//
// Numerical contract (verified against notebook Cell 12):
//
//     Density       σ²     I(f)     ARE = 12 σ² I(f)²
//     Normal       1.000   0.2821   0.9549   ≈ 3/π
//     Logistic     3.290   0.1667   1.0966   ≈ π²/9
//     Uniform      0.333   0.5000   1.0000   exact
//     Laplace      2.000   0.2500   1.5000   exact
//     t₅           1.667   0.2491   1.2412
//     HL min       1.000   0.2683   0.864    Hodges-Lehmann floor
//
// =============================================================================

const PANEL_HEIGHT = 270;
const SM_BREAKPOINT = 640;

const BLUE = '#2563EB';
const RED = '#DC2626';
const GREEN = '#059669';
const AMBER = '#D97706';
const SLATE = '#475569';
const LIGHT_BLUE = '#DBEAFE';
const LIGHT_RED = '#FEE2E2';

type Family = 'normal' | 'laplace' | 'logistic' | 'uniform' | 'student-t' | 'contaminated' | 'hl-min';

const FAMILIES: { value: Family; label: string; needsShape: boolean; shapeLabel?: string; shapeMin?: number; shapeMax?: number; shapeStep?: number; defaultShape?: number }[] = [
  { value: 'normal', label: 'Normal(0, 1)', needsShape: false },
  { value: 'laplace', label: 'Laplace (rate 1)', needsShape: false },
  { value: 'logistic', label: 'Logistic (scale 1)', needsShape: false },
  { value: 'uniform', label: 'Uniform [−1, 1]', needsShape: false },
  { value: 'student-t', label: 'Student t (df = ν)', needsShape: true, shapeLabel: 'ν', shapeMin: 3, shapeMax: 30, shapeStep: 1, defaultShape: 5 },
  { value: 'contaminated', label: 'Contaminated normal', needsShape: true, shapeLabel: 'ε', shapeMin: 0, shapeMax: 0.5, shapeStep: 0.01, defaultShape: 0.1 },
  { value: 'hl-min', label: 'Hodges-Lehmann minimum (parabolic)', needsShape: false },
];

const fmt = (x: number, digits = 4) => x.toFixed(digits);

// Erf via Abramowitz & Stegun rational approximation; relative error ≤ 1.5e-7.
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-ax * ax);
  return sign * y;
}

function normalCDF(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

// Acklam's algorithm for Φ⁻¹ (inverse normal CDF).
function normalInverseCDF(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [
    -3.969683028665376e1,
    2.209460984245205e2,
    -2.759285104469687e2,
    1.38357751867269e2,
    -3.066479806614716e1,
    2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1,
    1.615858368580409e2,
    -1.556989798598866e2,
    6.680131188771972e1,
    -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3,
    -3.223964580411365e-1,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
  ];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
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

// Density-family closed forms.
interface DensitySpec {
  pdf: (x: number) => number;
  range: [number, number];
  sigma2: number;
  Iclosed?: number;
  ARE: number;
  label: string;
}

function logGamma(z: number): number {
  // Stirling's via Lanczos coefficients.
  const g = [
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  }
  z -= 1;
  let x = 0.99999999999980993;
  for (let i = 0; i < g.length; i++) x += g[i] / (z + i + 1);
  const t = z + g.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

function gammaFn(z: number): number {
  return Math.exp(logGamma(z));
}

// Numerical integration of g over [a, b] via Simpson's rule with n intervals
// (n must be even). Stable for the bounded densities we use here.
function simpson(g: (x: number) => number, a: number, b: number, n: number): number {
  const h = (b - a) / n;
  let s = g(a) + g(b);
  for (let i = 1; i < n; i++) {
    s += (i % 2 === 0 ? 2 : 4) * g(a + i * h);
  }
  return (s * h) / 3;
}

function buildDensity(family: Family, shape: number): DensitySpec {
  if (family === 'normal') {
    const Iclosed = 1 / (2 * Math.sqrt(Math.PI));
    return {
      pdf: (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI),
      range: [-4, 4],
      sigma2: 1,
      Iclosed,
      ARE: 12 * 1 * Iclosed * Iclosed,
      label: 'Normal(0, 1)',
    };
  }
  if (family === 'laplace') {
    const Iclosed = 1 / 4;
    return {
      pdf: (x) => 0.5 * Math.exp(-Math.abs(x)),
      range: [-6, 6],
      sigma2: 2,
      Iclosed,
      ARE: 12 * 2 * Iclosed * Iclosed,
      label: 'Laplace (rate 1)',
    };
  }
  if (family === 'logistic') {
    const Iclosed = 1 / 6;
    const sigma2 = Math.PI * Math.PI / 3;
    return {
      pdf: (x) => {
        const e = Math.exp(-x);
        const denom = (1 + e);
        return e / (denom * denom);
      },
      range: [-7, 7],
      sigma2,
      Iclosed,
      ARE: 12 * sigma2 * Iclosed * Iclosed,
      label: 'Logistic (scale 1)',
    };
  }
  if (family === 'uniform') {
    const a = 1; // Uniform on [-a, a]
    const Iclosed = 1 / (2 * a);
    const sigma2 = (a * a) / 3;
    return {
      pdf: (x) => (Math.abs(x) <= a ? 1 / (2 * a) : 0),
      range: [-1.5, 1.5],
      sigma2,
      Iclosed,
      ARE: 12 * sigma2 * Iclosed * Iclosed,
      label: 'Uniform on [−1, 1]',
    };
  }
  if (family === 'student-t') {
    const nu = shape;
    const C = Math.exp(logGamma((nu + 1) / 2) - logGamma(nu / 2)) / Math.sqrt(nu * Math.PI);
    const pdf = (x: number) => C * Math.pow(1 + (x * x) / nu, -(nu + 1) / 2);
    const range: [number, number] = [-6, 6];
    const sigma2 = nu > 2 ? nu / (nu - 2) : 5; // fallback for ν=2 edge
    const I = simpson((x) => pdf(x) * pdf(x), -10, 10, 800);
    return {
      pdf,
      range,
      sigma2,
      ARE: 12 * sigma2 * I * I,
      label: `Student t (df = ${nu})`,
    };
  }
  if (family === 'contaminated') {
    const eps = shape;
    const sigO = 4; // outlier scale: contamination from N(0, σ_O²)
    const pdf = (x: number) => {
      const f1 = Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
      const f2 = Math.exp(-0.5 * (x / sigO) ** 2) / (sigO * Math.sqrt(2 * Math.PI));
      return (1 - eps) * f1 + eps * f2;
    };
    const sigma2 = (1 - eps) * 1 + eps * sigO * sigO;
    const I = simpson((x) => pdf(x) * pdf(x), -25, 25, 1200);
    return {
      pdf,
      range: [-8, 8],
      sigma2,
      ARE: 12 * sigma2 * I * I,
      label: `Contaminated normal (ε = ${eps.toFixed(2)})`,
    };
  }
  // hl-min: parabolic density f*(x) = 3/(4 sqrt(5)) (1 - x²/5) on |x| ≤ sqrt(5)
  const a = Math.sqrt(5);
  const c = 3 / (4 * Math.sqrt(5));
  const pdf = (x: number) => (Math.abs(x) <= a ? c * (1 - (x * x) / 5) : 0);
  // Closed form: σ² = 1, I(f*) = 3/(5 sqrt(5)) ≈ 0.2683 (numerical for safety)
  const I = simpson((x) => pdf(x) * pdf(x), -a, a, 800);
  const sigma2 = 1;
  return {
    pdf,
    range: [-3, 3],
    sigma2,
    ARE: 12 * sigma2 * I * I,
    label: 'Hodges-Lehmann minimum density',
  };
}

export default function AREExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [family, setFamily] = useState<Family>('normal');
  const [shape, setShape] = useState(5);
  const [n, setN] = useState(30);
  const [alpha, setAlpha] = useState(0.05);

  const familyMeta = FAMILIES.find((f) => f.value === family)!;

  const spec = useMemo(() => buildDensity(family, shape), [family, shape]);

  // I(f) used for power curves — recompute from pdf so the contaminated /
  // student-t numerical integrals are always in sync.
  const Iexact = useMemo(() => {
    if (spec.Iclosed != null) return spec.Iclosed;
    return simpson((x) => spec.pdf(x) * spec.pdf(x), -25, 25, 1200);
  }, [spec]);

  const cT = 1 / Math.sqrt(spec.sigma2);
  const cW = Math.sqrt(12) * Iexact;
  const ARE = (cW / cT) ** 2;

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked
    ? containerWidth || 0
    : Math.floor((containerWidth || 720) / 2) - 6;

  // ── Left: density panel ──────────────────────────────────────────────────
  const densityRef = useRef<SVGSVGElement>(null);
  useEffect(() => {
    const svg = d3.select(densityRef.current);
    if (!densityRef.current || panelWidth <= 0) return;
    svg.selectAll('*').remove();
    const margin = { top: 30, right: 14, bottom: 36, left: 40 };
    const w = panelWidth - margin.left - margin.right;
    const h = PANEL_HEIGHT - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const [xLo, xHi] = spec.range;
    const xScale = d3.scaleLinear().domain([xLo, xHi]).range([0, w]);
    const N_GRID = 200;
    const xs = d3.range(N_GRID).map((i) => xLo + ((xHi - xLo) * i) / (N_GRID - 1));
    const ys = xs.map(spec.pdf);
    const yMax = (d3.max(ys) ?? 1) * 1.1;
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

    const sigma = Math.sqrt(spec.sigma2);
    g.append('rect')
      .attr('x', xScale(-sigma))
      .attr('y', 0)
      .attr('width', xScale(sigma) - xScale(-sigma))
      .attr('height', h)
      .style('fill', LIGHT_BLUE)
      .style('opacity', 0.5);

    const line = d3
      .line<number>()
      .x((i) => xScale(xs[i]))
      .y((i) => yScale(ys[i]));
    g.append('path')
      .datum(d3.range(N_GRID))
      .attr('d', line)
      .style('fill', 'none')
      .style('stroke', BLUE)
      .style('stroke-width', 2);

    // Annotations for σ², I(f)
    g.append('text')
      .attr('x', 6)
      .attr('y', 12)
      .style('fill', 'var(--color-text)')
      .style('font-size', '11px')
      .style('font-family', 'var(--font-mono)')
      .text(`σ² = ${fmt(spec.sigma2, 3)}`);
    g.append('text')
      .attr('x', 6)
      .attr('y', 28)
      .style('fill', 'var(--color-text)')
      .style('font-size', '11px')
      .style('font-family', 'var(--font-mono)')
      .text(`I(f) = ∫ f² = ${fmt(Iexact, 4)}`);

    svg
      .append('text')
      .attr('x', margin.left + w / 2)
      .attr('y', 18)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '11.5px')
      .style('font-family', 'var(--font-sans)')
      .style('font-weight', 600)
      .text(`Density   ${spec.label}`);

    svg
      .append('text')
      .attr('x', margin.left + w / 2)
      .attr('y', PANEL_HEIGHT - 10)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text-secondary)')
      .style('font-size', '10px')
      .text('shaded band: ±σ around 0');
  }, [spec, Iexact, panelWidth]);

  // ── Right: power curves panel ────────────────────────────────────────────
  const powerRef = useRef<SVGSVGElement>(null);
  useEffect(() => {
    const svg = d3.select(powerRef.current);
    if (!powerRef.current || panelWidth <= 0) return;
    svg.selectAll('*').remove();
    const margin = { top: 30, right: 14, bottom: 36, left: 42 };
    const w = panelWidth - margin.left - margin.right;
    const h = PANEL_HEIGHT - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const N_GRID = 120;
    const thetaMax = 1.4 / Math.sqrt(n) * 6; // sweep over a useful effect range
    const thetas = d3.range(N_GRID).map((i) => -thetaMax + (2 * thetaMax * i) / (N_GRID - 1));
    const z = normalInverseCDF(1 - alpha / 2);
    const powerT = thetas.map((th) => normalCDF(cT * th * Math.sqrt(n) - z) + (1 - normalCDF(cT * th * Math.sqrt(n) + z)));
    const powerW = thetas.map((th) => normalCDF(cW * th * Math.sqrt(n) - z) + (1 - normalCDF(cW * th * Math.sqrt(n) + z)));

    const xScale = d3.scaleLinear().domain([-thetaMax, thetaMax]).range([0, w]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([h, 0]);

    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text')
      .style('fill', 'var(--color-text)')
      .style('font-size', '10px');
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.0%')))
      .selectAll('text')
      .style('fill', 'var(--color-text)')
      .style('font-size', '10px');
    g.selectAll('path.domain, .tick line').style('stroke', 'var(--color-border)');

    // Reference horizontal line at α
    g.append('line')
      .attr('x1', 0)
      .attr('x2', w)
      .attr('y1', yScale(alpha))
      .attr('y2', yScale(alpha))
      .style('stroke', SLATE)
      .style('stroke-dasharray', '2,3');
    g.append('text')
      .attr('x', w - 4)
      .attr('y', yScale(alpha) - 4)
      .attr('text-anchor', 'end')
      .style('font-size', '10px')
      .style('fill', SLATE)
      .text(`α = ${alpha.toFixed(2)}`);

    const linePath = (vals: number[]) =>
      d3
        .line<number>()
        .x((i) => xScale(thetas[i]))
        .y((i) => yScale(vals[i]))(d3.range(N_GRID)) as string;

    g.append('path')
      .attr('d', linePath(powerT))
      .style('fill', 'none')
      .style('stroke', RED)
      .style('stroke-width', 2);
    g.append('path')
      .attr('d', linePath(powerW))
      .style('fill', 'none')
      .style('stroke', BLUE)
      .style('stroke-width', 2);

    // Legend
    const legend = g.append('g').attr('transform', `translate(${w - 100}, 8)`);
    legend
      .append('line')
      .attr('x1', 0)
      .attr('x2', 14)
      .attr('y1', 0)
      .attr('y2', 0)
      .style('stroke', RED)
      .style('stroke-width', 2);
    legend
      .append('text')
      .attr('x', 18)
      .attr('y', 3)
      .style('font-size', '10.5px')
      .style('fill', 'var(--color-text)')
      .text('t-test  (c_T = 1/σ)');
    legend
      .append('line')
      .attr('x1', 0)
      .attr('x2', 14)
      .attr('y1', 14)
      .attr('y2', 14)
      .style('stroke', BLUE)
      .style('stroke-width', 2);
    legend
      .append('text')
      .attr('x', 18)
      .attr('y', 17)
      .style('font-size', '10.5px')
      .style('fill', 'var(--color-text)')
      .text('Wilcoxon  (c_W = √12·I(f))');

    svg
      .append('text')
      .attr('x', margin.left + w / 2)
      .attr('y', 18)
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--color-text)')
      .style('font-size', '11.5px')
      .style('font-family', 'var(--font-sans)')
      .style('font-weight', 600)
      .text(`Asymptotic local power  (n = ${n})`);

    const areCol = ARE >= 1 ? GREEN : AMBER;
    svg
      .append('text')
      .attr('x', margin.left + w / 2)
      .attr('y', PANEL_HEIGHT - 10)
      .attr('text-anchor', 'middle')
      .style('fill', areCol)
      .style('font-size', '11.5px')
      .style('font-family', 'var(--font-mono)')
      .style('font-weight', 600)
      .text(`ARE(W, T) = 12 σ² (∫ f²)² = ${fmt(ARE, 4)}`);
  }, [spec, Iexact, n, alpha, cT, cW, ARE, panelWidth]);

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
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flex: '1 1 240px' }}>
          <span style={{ fontSize: '11px', color: 'var(--color-text-secondary)' }}>density f</span>
          <select
            value={family}
            onChange={(e) => {
              const f = e.target.value as Family;
              setFamily(f);
              const meta = FAMILIES.find((m) => m.value === f);
              if (meta?.needsShape && meta.defaultShape != null) setShape(meta.defaultShape);
            }}
            style={selectStyle()}
          >
            {FAMILIES.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </label>
        {familyMeta.needsShape && (
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flex: '1 1 200px' }}>
            <span style={{ fontSize: '11px', color: 'var(--color-text-secondary)', minWidth: '1.5em' }}>{familyMeta.shapeLabel}</span>
            <input
              type="range"
              min={familyMeta.shapeMin}
              max={familyMeta.shapeMax}
              step={familyMeta.shapeStep}
              value={shape}
              onChange={(e) => setShape(Number(e.target.value))}
              style={{ flex: 1 }}
              aria-label="density shape parameter"
            />
            <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', minWidth: '3em', textAlign: 'right', color: 'var(--color-text)' }}>
              {familyMeta.value === 'student-t' ? shape.toFixed(0) : shape.toFixed(2)}
            </span>
          </label>
        )}
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flex: '1 1 180px' }}>
          <span style={{ fontSize: '11px', color: 'var(--color-text-secondary)', minWidth: '1.5em' }}>n</span>
          <input
            type="range"
            min={5}
            max={200}
            step={5}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            style={{ flex: 1 }}
            aria-label="sample size n"
          />
          <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', minWidth: '3em', textAlign: 'right', color: 'var(--color-text)' }}>{n}</span>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flex: '1 1 180px' }}>
          <span style={{ fontSize: '11px', color: 'var(--color-text-secondary)', minWidth: '1.5em' }}>α</span>
          <input
            type="range"
            min={0.01}
            max={0.20}
            step={0.01}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
            style={{ flex: 1 }}
            aria-label="significance level alpha"
          />
          <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', minWidth: '3em', textAlign: 'right', color: 'var(--color-text)' }}>{alpha.toFixed(2)}</span>
        </label>
      </div>

      <div
        style={{
          display: 'flex',
          flexDirection: isStacked ? 'column' : 'row',
          gap: isStacked ? '0.75rem' : '0.5rem',
        }}
      >
        <svg ref={densityRef} width={panelWidth} height={PANEL_HEIGHT} role="img" aria-label="density plot with σ² and I(f) annotations" />
        <svg ref={powerRef} width={panelWidth} height={PANEL_HEIGHT} role="img" aria-label="asymptotic local power curves for Wilcoxon and t-test, with ARE readout" />
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
        Try Student-t with df = 5 (ARE ≈ 1.24) or contaminated normal with ε = 0.1 (ARE rises sharply with the contamination weight). The Hodges-Lehmann minimum is the parabolic density at which ARE(W, T) bottoms out at 0.864.
      </p>
    </div>
  );
}

const selectStyle = (): CSSProperties => ({
  flex: 1,
  fontSize: '11.5px',
  padding: '3px 6px',
  background: 'var(--color-surface)',
  color: 'var(--color-text)',
  border: '1px solid var(--color-border)',
  borderRadius: 4,
});
