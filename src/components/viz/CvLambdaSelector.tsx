import { useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  generateDgp1,
  lassoIsta,
  operatorNorm,
  xMul,
} from './shared/high-dim-regression';

// =============================================================================
// CvLambdaSelector — interactive companion to §7.2 Figure 7.1 + §7.4 Figure 7.2.
// Combines two views into one panel:
//   - 10-fold CV-MSE curve over a log-spaced λ-grid (with ±1 SE shaded band)
//   - Vertical markers at five selectors: LassoCV(λ_min), LassoCV(λ_1SE),
//     LassoLarsIC-AIC, LassoLarsIC-BIC, theory-guided RIC.
//
// AIC / BIC are approximated on the CV-grid (rather than knots from the LARS
// path) for in-browser tractability. The substantive ordering λ_min < AIC <
// λ_1SE < BIC < RIC is preserved.
//
// DGP-1 scale: n = 200, p = 200, s = 10, σ = 0.5 (matches LassoUCurve and
// LassoSolutionPath for visual consistency).
//
// Compute: 10 folds × 25 λ × ISTA-with-warm-starts ≈ 1-2 s.
//
// Static fallbacks: fig_07_01_cv_curve.png + fig_07_02_selector_comparison.png
// =============================================================================

const HEIGHT = 460;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 28, right: 16, bottom: 50, left: 60 };

const N = 200;
const P = 200;
const S = 10;
const SIGMA = 0.5;
const RHO = 0.5;
const SEED = 42;
const K_FOLDS = 10;
const N_LAMBDA = 25;
const LAMBDA_MIN_RATIO = 1e-3;
const ISTA_ITERS_FIRST = 200;
const ISTA_ITERS_WARM = 50;
const NONZERO_THRESH = 0.01;

const TEAL = '#0F6E56';
const PURPLE = '#534AB7';
const AMBER = '#D97706';
const RED = '#B91C1C';
const SLATE = '#6B6B6B';
const ROSE = '#DB2777';

interface CVPoint {
  lambda: number;
  mean: number;
  sd: number;
  se: number;
  aic: number;
  bic: number;
}

interface SelectorMarker {
  name: string;
  lambda: number;
  color: string;
}

// -----------------------------------------------------------------------------
// Compute CV curve + AIC + BIC at each λ.
// -----------------------------------------------------------------------------

function computeCVCurve(): { curve: CVPoint[]; selectors: SelectorMarker[] } {
  const opts = { n: N, p: P, s: S, sigma: SIGMA, rho: RHO, seed: SEED };
  const sample = generateDgp1(opts);
  // Lambda max: ‖Xᵀy/n‖_∞ on the full sample.
  const Xty = new Float64Array(P);
  for (let j = 0; j < P; j++) {
    let s2 = 0;
    for (let i = 0; i < N; i++) s2 += sample.X[i][j] * sample.y[i];
    Xty[j] = s2 / N;
  }
  let lMax = 0;
  for (let j = 0; j < P; j++) lMax = Math.max(lMax, Math.abs(Xty[j]));

  // Lambda grid (descending for warm starts).
  const lambdas: number[] = [];
  const logHi = Math.log(lMax);
  const logLo = Math.log(lMax * LAMBDA_MIN_RATIO);
  for (let i = 0; i < N_LAMBDA; i++) {
    lambdas.push(Math.exp(logHi - ((logHi - logLo) * i) / (N_LAMBDA - 1)));
  }

  // K-fold CV: for each fold, compute the path on the leave-fold-out training
  // set + the held-out MSE at each λ.
  const foldSize = Math.floor(N / K_FOLDS);
  const foldMse: number[][] = Array.from({ length: K_FOLDS }, () => new Array<number>(N_LAMBDA).fill(0));
  for (let k = 0; k < K_FOLDS; k++) {
    const foldStart = k * foldSize;
    const foldEnd = k === K_FOLDS - 1 ? N : foldStart + foldSize;
    const trainX: Float64Array[] = [];
    const trainY: number[] = [];
    const valX: Float64Array[] = [];
    const valY: number[] = [];
    for (let i = 0; i < N; i++) {
      if (i >= foldStart && i < foldEnd) {
        valX.push(sample.X[i]);
        valY.push(sample.y[i]);
      } else {
        trainX.push(sample.X[i]);
        trainY.push(sample.y[i]);
      }
    }
    const trainXArr = trainX;
    const trainYArr = new Float64Array(trainY);
    const Lk = operatorNorm(trainXArr);
    let prevBeta: Float64Array | undefined = undefined;
    for (let li = 0; li < lambdas.length; li++) {
      const iters = li === 0 ? ISTA_ITERS_FIRST : ISTA_ITERS_WARM;
      const beta = lassoIsta(trainXArr, trainYArr, lambdas[li], Lk, iters, prevBeta);
      // Held-out MSE.
      let sse = 0;
      for (let i = 0; i < valX.length; i++) {
        let pred = 0;
        for (let j = 0; j < P; j++) pred += valX[i][j] * beta[j];
        const r = valY[i] - pred;
        sse += r * r;
      }
      foldMse[k][li] = sse / valX.length;
      prevBeta = beta;
    }
  }

  // Aggregate: mean + sd across folds + SE.
  const curve: CVPoint[] = [];
  for (let li = 0; li < lambdas.length; li++) {
    let mean = 0;
    for (let k = 0; k < K_FOLDS; k++) mean += foldMse[k][li];
    mean /= K_FOLDS;
    let v = 0;
    for (let k = 0; k < K_FOLDS; k++) {
      const d = foldMse[k][li] - mean;
      v += d * d;
    }
    v /= K_FOLDS - 1;
    const sd = Math.sqrt(v);
    const se = sd / Math.sqrt(K_FOLDS);
    curve.push({ lambda: lambdas[li], mean, sd, se, aic: 0, bic: 0 });
  }

  // Compute AIC + BIC on the full sample (one fit per λ for the AIC/BIC values).
  const Lfull = operatorNorm(sample.X);
  let prevBeta: Float64Array | undefined = undefined;
  for (let li = 0; li < lambdas.length; li++) {
    const iters = li === 0 ? ISTA_ITERS_FIRST : ISTA_ITERS_WARM;
    const beta = lassoIsta(sample.X, sample.y, lambdas[li], Lfull, iters, prevBeta);
    // Compute RSS and active-set size.
    const Xb = xMul(sample.X, beta);
    let rss = 0;
    for (let i = 0; i < N; i++) {
      const r = sample.y[i] - Xb[i];
      rss += r * r;
    }
    let nz = 0;
    for (let j = 0; j < P; j++) if (Math.abs(beta[j]) > NONZERO_THRESH) nz++;
    curve[li].aic = N * Math.log(rss / N) + 2 * nz;
    curve[li].bic = N * Math.log(rss / N) + nz * Math.log(N);
    prevBeta = beta;
  }

  // Selectors.
  let argminCV = 0;
  for (let li = 1; li < curve.length; li++) {
    if (curve[li].mean < curve[argminCV].mean) argminCV = li;
  }
  const lambdaMinCV = curve[argminCV].lambda;
  const seThreshold = curve[argminCV].mean + curve[argminCV].se;
  // 1-SE rule: largest λ (smallest λ-index since we reversed) within seThreshold.
  // Curve is in descending λ order, so smaller li means larger λ; pick smallest li with mean ≤ threshold.
  let lambda1SE = lambdaMinCV;
  for (let li = 0; li < curve.length; li++) {
    if (curve[li].mean <= seThreshold && curve[li].lambda > lambda1SE) {
      lambda1SE = curve[li].lambda;
    }
  }
  // AIC / BIC argmins.
  let argminAIC = 0;
  let argminBIC = 0;
  for (let li = 1; li < curve.length; li++) {
    if (curve[li].aic < curve[argminAIC].aic) argminAIC = li;
    if (curve[li].bic < curve[argminBIC].bic) argminBIC = li;
  }
  // RIC: theory-guided.
  const lambdaRIC = 2 * SIGMA * Math.sqrt((2 * Math.log(P)) / N);

  const selectors: SelectorMarker[] = [
    { name: 'CV-min', lambda: lambdaMinCV, color: TEAL },
    { name: 'CV-1SE', lambda: lambda1SE, color: PURPLE },
    { name: 'AIC', lambda: curve[argminAIC].lambda, color: AMBER },
    { name: 'BIC', lambda: curve[argminBIC].lambda, color: ROSE },
    { name: 'RIC', lambda: lambdaRIC, color: SLATE },
  ];

  return { curve, selectors };
}

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export default function CvLambdaSelector() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const w = containerWidth || 720;
  const isMobile = w < SM_BREAKPOINT;

  const { curve, selectors } = useMemo(() => computeCVCurve(), []);

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLog().domain([curve[curve.length - 1].lambda, curve[0].lambda]).range([0, innerW]);
      const yMin = Math.min(...curve.map((c) => c.mean - c.se)) * 0.95;
      const yMax = Math.max(...curve.map((c) => c.mean + c.se)) * 1.05;
      const yScale = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]);

      const xAxis = d3.axisBottom(xScale).ticks(isMobile ? 4 : 6, '~g').tickSize(-innerH);
      const yAxis = d3.axisLeft(yScale).ticks(isMobile ? 4 : 6).tickSize(-innerW);
      g.append('g').attr('transform', `translate(0,${innerH})`).call(xAxis).call((sel) => {
        sel.selectAll('line').style('stroke', 'var(--color-border)');
        sel.selectAll('path').style('stroke', 'var(--color-text-secondary)');
        sel.selectAll('text').style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)');
      });
      g.append('g').call(yAxis).call((sel) => {
        sel.selectAll('line').style('stroke', 'var(--color-border)');
        sel.selectAll('path').style('stroke', 'var(--color-text-secondary)');
        sel.selectAll('text').style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)');
      });

      g.append('text').attr('text-anchor', 'middle').attr('x', innerW / 2).attr('y', innerH + 38).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '13px').text('λ (log scale)');
      g.append('text').attr('text-anchor', 'middle').attr('transform', `translate(-44,${innerH / 2}) rotate(-90)`).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '13px').text('CV-MSE (10-fold)');

      // ±1 SE band.
      const areaGen = d3
        .area<CVPoint>()
        .x((c) => xScale(c.lambda))
        .y0((c) => yScale(c.mean - c.se))
        .y1((c) => yScale(c.mean + c.se));
      g.append('path').datum(curve).attr('d', areaGen).style('fill', TEAL).style('fill-opacity', 0.15);

      // CV curve.
      const lineGen = d3.line<CVPoint>().x((c) => xScale(c.lambda)).y((c) => yScale(c.mean));
      g.append('path').datum(curve).attr('d', lineGen).style('fill', 'none').style('stroke', TEAL).style('stroke-width', 2);
      g.selectAll('.dot').data(curve).enter().append('circle').attr('class', 'dot').attr('cx', (c) => xScale(c.lambda)).attr('cy', (c) => yScale(c.mean)).attr('r', 2.5).style('fill', TEAL);

      // Selector vertical markers.
      selectors.forEach((sel, i) => {
        if (sel.lambda < curve[curve.length - 1].lambda || sel.lambda > curve[0].lambda) return;
        g.append('line').attr('x1', xScale(sel.lambda)).attr('x2', xScale(sel.lambda)).attr('y1', 0).attr('y2', innerH).style('stroke', sel.color).style('stroke-dasharray', '4 3').style('stroke-width', 1.5).style('opacity', 0.8);
        g.append('text').attr('x', xScale(sel.lambda) + 4).attr('y', 14 + i * 14).style('fill', sel.color).style('font-family', 'var(--font-mono)').style('font-size', '10.5px').text(`${sel.name}: ${sel.lambda.toFixed(3)}`);
      });
    },
    [curve, selectors, w, isMobile],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={renderRef} width={w} height={HEIGHT} role="img" aria-label="LassoCV curve with five selector markers" />
      <p
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '13px',
          color: 'var(--color-text-secondary)',
          marginTop: '8px',
        }}
      >
        10-fold LassoCV on smaller-scale DGP-1 (n = {N}, p = {P}, s = {S}, σ = {SIGMA}, AR(1) ρ = {RHO}). The teal curve is mean CV-MSE across folds; shaded band is ±1 SE. Five selector markers (vertical dashed lines) ordered from smallest to largest λ: CV-min (largest active set, smallest test MSE), CV-1SE (parsimony-favoring within 1 SE of CV-min), AIC and BIC (information-criterion picks on the lasso path; BIC selects sparser models than AIC), and theory-guided RIC = 2σ√(2 log p / n) from Theorem 1 (largest, conservative). Computed live in-browser via 10-fold warm-started ISTA across {N_LAMBDA} log-spaced λ values (~1-2 s).
      </p>
    </div>
  );
}
