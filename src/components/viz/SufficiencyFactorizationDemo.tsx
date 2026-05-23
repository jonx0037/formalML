import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { centerColumns, pcaTopK, pcaProject, paletteRL, mulberry32, gaussianPair } from './shared/representation-learning';

// =============================================================================
// SufficiencyFactorizationDemo — §2.3 companion. The reader sets the
// nuisance-axis variance σ_n² of the §1.2 fixture (20-D two-class Gaussian
// with discriminative axis in e_1 and high-variance nuisance along e_20).
// The viz computes the saturation curve I(top-d PCA; Y) and I(top-d LDA; Y)
// vs d, demonstrating that LDA captures the class information in a single
// dimension while PCA needs many dimensions when σ_n² ≫ 1 — the soft-
// sufficiency picture of Definition 2.3.
//
// MI is estimated by a discretized plug-in: for a binary Y and a d-dimensional
// representation Z, fit class-conditional Gaussians and use the closed-form
// MI of a Gaussian-mixture / scalar-Gaussian binary classifier as a proxy.
// We use a simple 1D / 2D / k-NN proxy: estimate p(Y=1|z) by logistic on top-d
// projection, then compute I = E_z[ KL(p(·|z) || p(·)) ] empirically.
// =============================================================================

const D_MAX = 20;
const N = 800;
const NUISANCE_INDEX = D_MAX - 1; // e_20
const DISCRIM_INDEX = 0;

// Logistic regression via Newton-Raphson on a small dataset — used for the MI
// estimator. d ≤ 20, n = 800, max 20 iterations: trivially fast in browser.
function logisticFitMI(X: Float64Array, y: Int32Array, n: number, d: number): number {
  // Append intercept.
  const p = d + 1;
  const Xa = new Float64Array(n * p);
  for (let i = 0; i < n; i++) {
    Xa[i * p] = 1;
    for (let j = 0; j < d; j++) Xa[i * p + 1 + j] = X[i * d + j];
  }
  const beta = new Float64Array(p);
  const pi = new Float64Array(n);
  const w = new Float64Array(n);
  for (let iter = 0; iter < 25; iter++) {
    // pi_i = σ(x_i^T β).
    for (let i = 0; i < n; i++) {
      let z = 0;
      for (let j = 0; j < p; j++) z += Xa[i * p + j] * beta[j];
      const e = Math.exp(-Math.max(Math.min(z, 30), -30));
      pi[i] = 1 / (1 + e);
      w[i] = pi[i] * (1 - pi[i]);
    }
    // Build H = X^T W X (p × p) and g = X^T (y − pi).
    const H = new Float64Array(p * p);
    const g = new Float64Array(p);
    for (let i = 0; i < n; i++) {
      const wi = w[i];
      const ri = y[i] - pi[i];
      for (let a = 0; a < p; a++) {
        g[a] += Xa[i * p + a] * ri;
        for (let b = a; b < p; b++) {
          H[a * p + b] += Xa[i * p + a] * Xa[i * p + b] * wi;
        }
      }
    }
    for (let a = 0; a < p; a++) for (let b = a + 1; b < p; b++) H[b * p + a] = H[a * p + b];
    // Solve H δ = g via Gauss-Jordan with partial pivoting.
    const M = new Float64Array(p * (p + 1));
    for (let a = 0; a < p; a++) {
      for (let b = 0; b < p; b++) M[a * (p + 1) + b] = H[a * p + b];
      M[a * (p + 1) + p] = g[a];
    }
    for (let k = 0; k < p; k++) {
      let piv = k;
      for (let r = k + 1; r < p; r++) if (Math.abs(M[r * (p + 1) + k]) > Math.abs(M[piv * (p + 1) + k])) piv = r;
      if (piv !== k) {
        for (let c = 0; c < p + 1; c++) {
          const tmp = M[k * (p + 1) + c];
          M[k * (p + 1) + c] = M[piv * (p + 1) + c];
          M[piv * (p + 1) + c] = tmp;
        }
      }
      const dk = M[k * (p + 1) + k];
      if (Math.abs(dk) < 1e-12) continue;
      for (let c = k; c < p + 1; c++) M[k * (p + 1) + c] /= dk;
      for (let r = 0; r < p; r++) {
        if (r === k) continue;
        const f = M[r * (p + 1) + k];
        if (f === 0) continue;
        for (let c = k; c < p + 1; c++) M[r * (p + 1) + c] -= f * M[k * (p + 1) + c];
      }
    }
    let step = 0;
    for (let a = 0; a < p; a++) {
      const delta = M[a * (p + 1) + p];
      beta[a] += delta;
      step += delta * delta;
    }
    if (step < 1e-10) break;
  }
  // I(Z; Y) estimate: H(Y) − H(Y | Z).
  let h0 = 0,
    h1 = 0;
  for (let i = 0; i < n; i++) (y[i] === 1 ? h1++ : h0++);
  const pY1 = h1 / n;
  const HY = pY1 === 0 || pY1 === 1 ? 0 : -pY1 * Math.log(pY1) - (1 - pY1) * Math.log(1 - pY1);
  let HYgivenZ = 0;
  for (let i = 0; i < n; i++) {
    let z = 0;
    for (let j = 0; j < p; j++) z += Xa[i * p + j] * beta[j];
    const e = Math.exp(-Math.max(Math.min(z, 30), -30));
    const p1 = 1 / (1 + e);
    if (p1 > 1e-9 && p1 < 1 - 1e-9) HYgivenZ += -p1 * Math.log(p1) - (1 - p1) * Math.log(1 - p1);
  }
  HYgivenZ /= n;
  return Math.max(0, HY - HYgivenZ);
}

interface Fixture {
  X: Float64Array;
  y: Int32Array;
  lda1d: Float64Array; // length-N projection onto Fisher direction
}

function buildFixture(sigmaN: number, seed: number): Fixture {
  const rng = mulberry32(seed);
  const d = D_MAX;
  const X = new Float64Array(N * d);
  const y = new Int32Array(N);
  const lda1d = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    y[i] = i < N / 2 ? 0 : 1;
  }
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < d; j += 2) {
      const [a, b] = gaussianPair(rng);
      const scale = j === NUISANCE_INDEX ? sigmaN : 1;
      X[i * d + j] = scale * a;
      if (j + 1 < d) {
        const scale2 = j + 1 === NUISANCE_INDEX ? sigmaN : 1;
        X[i * d + j + 1] = scale2 * b;
      }
    }
    // Shift class 1 by Δ = 2 along e_1.
    if (y[i] === 1) X[i * d + DISCRIM_INDEX] += 2;
  }
  // LDA 1-D projection along the discriminative direction.
  // For Gaussian classes with shared covariance, the Fisher direction is
  // Σ⁻¹ (μ_1 − μ_0). With Σ diagonal and the discriminative axis at variance 1,
  // the direction reduces to e_1, so the projection is X[:, 0].
  for (let i = 0; i < N; i++) lda1d[i] = X[i * d + DISCRIM_INDEX];
  return { X, y, lda1d };
}

export default function SufficiencyFactorizationDemo() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const [sigmaN, setSigmaN] = useState(5);
  const [dSelected, setDSelected] = useState(1);

  const W = Math.max(360, Math.min(width || 720, 720));
  const H = 360;
  const margin = { top: 22, right: 24, bottom: 42, left: 56 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  const fixture = useMemo(() => buildFixture(sigmaN, 17), [sigmaN]);

  const pcaCurve = useMemo(() => {
    const X = new Float64Array(fixture.X);
    centerColumns(X, N, D_MAX);
    const { components } = pcaTopK(X, N, D_MAX, D_MAX);
    const Z = pcaProject(X, components, N, D_MAX, D_MAX);
    const mi = new Float64Array(D_MAX);
    for (let d = 1; d <= D_MAX; d++) {
      const slice = new Float64Array(N * d);
      for (let i = 0; i < N; i++) for (let j = 0; j < d; j++) slice[i * d + j] = Z[i * D_MAX + j];
      mi[d - 1] = logisticFitMI(slice, fixture.y, N, d);
    }
    return mi;
  }, [fixture]);

  const ldaMI = useMemo(() => {
    const lda = new Float64Array(fixture.lda1d);
    return logisticFitMI(lda, fixture.y, N, 1);
  }, [fixture]);

  const fullMI = useMemo(() => {
    return logisticFitMI(fixture.X, fixture.y, N, D_MAX);
  }, [fixture]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.attr('viewBox', `0 0 ${W} ${H}`).selectAll('*').remove();
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLinear().domain([1, D_MAX]).range([0, innerW]);
      const yMax = Math.max(0.05, Math.max(ldaMI, fullMI, ...pcaCurve) * 1.15);
      const y = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(10).tickFormat(d3.format('d')));
      g.append('g').call(d3.axisLeft(y).ticks(6));
      g.append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 32)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .style('fill', 'currentColor')
        .text('Representation dimension d');
      g.append('text')
        .attr('x', -innerH / 2)
        .attr('y', -42)
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .attr('font-size', 11)
        .style('fill', 'currentColor')
        .text('Î(Z_d; Y)  (nats)');

      // Reference lines.
      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', y(fullMI)).attr('y2', y(fullMI)).attr('stroke-width', 1).attr('stroke-dasharray', '4 3').attr('opacity', 0.6).style('stroke', paletteRL.data);
      g.append('text').attr('x', innerW - 6).attr('y', y(fullMI) - 4).attr('text-anchor', 'end').attr('font-size', 10).style('fill', paletteRL.data).text(`Î(X; Y) = ${fullMI.toFixed(3)}`);
      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', y(ldaMI)).attr('y2', y(ldaMI)).attr('stroke-width', 1).attr('stroke-dasharray', '2 3').attr('opacity', 0.7).style('stroke', paletteRL.optimum);
      g.append('text').attr('x', 6).attr('y', y(ldaMI) - 4).attr('text-anchor', 'start').attr('font-size', 10).style('fill', paletteRL.optimum).text(`LDA 1-D: Î = ${ldaMI.toFixed(3)}`);

      // PCA curve.
      const line = d3
        .line<number>()
        .x((_d, i) => x(i + 1))
        .y((d) => y(d));
      g.append('path')
        .datum(Array.from(pcaCurve))
        .attr('fill', 'none')
        .attr('stroke-width', 2.4)
        .attr('d', line)
        .style('stroke', paletteRL.signal);
      // PCA markers.
      g.selectAll('circle.pca')
        .data(Array.from(pcaCurve))
        .join('circle')
        .attr('class', 'pca')
        .attr('cx', (_d, i) => x(i + 1))
        .attr('cy', (d) => y(d))
        .attr('r', (_d, i) => (i + 1 === dSelected ? 6 : 3))
        .attr('stroke-width', 1.2)
        .style('fill', (_d, i) => (i + 1 === dSelected ? paletteRL.highlight : paletteRL.signal))
        .style('stroke', paletteRL.signal);
    },
    [pcaCurve, ldaMI, fullMI, dSelected, W, H, innerW, innerH],
  );

  return (
    <div ref={containerRef} className="not-prose my-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-2 text-sm">
        <label className="flex flex-col">
          <span>Nuisance-axis std-dev σ_n = {sigmaN.toFixed(1)}</span>
          <input
            type="range"
            min={1}
            max={8}
            step={0.5}
            value={sigmaN}
            onChange={(e) => setSigmaN(Number(e.target.value))}
            aria-label="Nuisance-axis standard deviation"
          />
        </label>
        <label className="flex flex-col">
          <span>PCA dimensions kept d = {dSelected}</span>
          <input
            type="range"
            min={1}
            max={D_MAX}
            step={1}
            value={dSelected}
            onChange={(e) => setDSelected(Number(e.target.value))}
            aria-label="PCA dimensions"
          />
        </label>
      </div>
      <svg ref={svgRef} className="w-full" style={{ maxHeight: H }} role="img" aria-label="Soft sufficiency saturation curve"/>
      <div className="text-xs opacity-75 mt-1 text-center">
        At σ_n = {sigmaN.toFixed(1)}: PCA-top-{dSelected} captures Î(Z_d; Y) = <strong>{pcaCurve[dSelected - 1].toFixed(3)}</strong> nats, LDA-1D captures <strong>{ldaMI.toFixed(3)}</strong> nats. The full-data ceiling is Î(X; Y) = <strong>{fullMI.toFixed(3)}</strong>. Raise σ_n to push the nuisance axis up the PCA ranking; LDA's 1-D estimate stays constant.
      </div>
    </div>
  );
}
