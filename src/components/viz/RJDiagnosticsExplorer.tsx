// =============================================================================
// RJDiagnosticsExplorer.tsx
//
// §11 of reversible-jump-mcmc. Diagnostic dashboard for the §10 Galaxy chain.
// Three panels: running posterior P_t(K | D) with ±2σ multinomial CLT bands;
// per-move acceptance bar with the 5%-50% Goldilocks zone shaded; within-K
// autocorrelation function for the smallest-mean component at the dominant K.
//
// Re-runs a Galaxy chain internally (same RNG seed → same chain as §10 at the
// default settings). Display-vs-committed slider for chain length.
//
// Static fallback: /images/topics/reversible-jump-mcmc/fig11_diagnostics.png
// =============================================================================
import { useCallback, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';

const GALAXY: number[] = [
  9.172, 9.35, 9.483, 9.558, 9.775, 10.227, 10.406, 16.084, 16.17, 18.419, 18.552, 18.6, 18.927,
  19.052, 19.07, 19.33, 19.343, 19.349, 19.44, 19.473, 19.529, 19.541, 19.547, 19.663, 19.846,
  19.856, 19.863, 19.914, 19.918, 19.973, 19.989, 20.166, 20.175, 20.179, 20.196, 20.215, 20.221,
  20.415, 20.629, 20.795, 20.821, 20.846, 20.875, 20.986, 21.137, 21.492, 21.701, 21.814, 21.921,
  21.96, 22.185, 22.209, 22.242, 22.249, 22.314, 22.374, 22.495, 22.746, 22.747, 22.888, 22.914,
  23.206, 23.241, 23.263, 23.484, 23.538, 23.542, 23.666, 23.706, 23.711, 24.129, 24.285, 24.289,
  24.366, 24.717, 24.99, 25.633, 26.96, 26.995, 32.065, 32.789, 34.279,
];
const N = GALAXY.length;

function mulberry32(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
type RNG = () => number;
function randn(rng: RNG): number {
  const u = 1 - rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function gammaSample(rng: RNG, shape: number, scale: number): number {
  if (shape < 1) return gammaSample(rng, shape + 1, scale) * Math.pow(rng(), 1 / shape);
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x = 0;
    let v = 0;
    do {
      x = randn(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (u < 1 - 0.0331 * x ** 4) return d * v * scale;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * scale;
  }
}
function invGammaSample(rng: RNG, alpha: number, beta: number): number {
  return 1 / gammaSample(rng, alpha, 1 / beta);
}
function dirichletSample(rng: RNG, alpha: number[], out: Float64Array): void {
  let s = 0;
  for (let i = 0; i < alpha.length; i++) {
    out[i] = gammaSample(rng, alpha[i], 1);
    s += out[i];
  }
  for (let i = 0; i < alpha.length; i++) out[i] /= s;
}
function lgamma(x: number): number {
  const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (const c of cof) ser += c / ++y;
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}
function logBetaPdf(x: number, a: number, b: number): number {
  if (x <= 0 || x >= 1) return -Infinity;
  return (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - (lgamma(a) + lgamma(b) - lgamma(a + b));
}
function logNormPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return -0.5 * Math.log(2 * Math.PI) - Math.log(sigma) - 0.5 * z * z;
}
function logInvGammaPdf(x: number, alpha: number, beta: number): number {
  if (x <= 0) return -Infinity;
  return alpha * Math.log(beta) - lgamma(alpha) - (alpha + 1) * Math.log(x) - beta / x;
}

interface DiagResult {
  Ktrace: Int8Array;
  // Snapshots at the dominant K for ACF
  muTraceSmallest: number[]; // smallest-mean component μ over snapshots at dominant K
  accepts: { split: number; merge: number; birth: number; death: number; gibbs: number };
  tries: { split: number; merge: number; birth: number; death: number; gibbs: number };
}

const SNAPSHOT_THIN = 4;

function runDiagChain(T: number, burn: number, Kmax: number, seed: number): DiagResult {
  const rng = mulberry32(seed);
  const yBar = GALAXY.reduce((s, x) => s + x, 0) / N;
  const yVar = GALAXY.reduce((s, x) => s + (x - yBar) ** 2, 0) / N;
  const R = Math.max(...GALAXY) - Math.min(...GALAXY);
  const xi = yBar;
  const tau2 = (R * R) / 16;
  const alphaSigma = 2;
  const betaSigma = yVar / 4;
  const alphaW = 1;

  const muBuf = new Float64Array(Kmax);
  const s2Buf = new Float64Array(Kmax);
  const wBuf = new Float64Array(Kmax);
  const propMu = new Float64Array(Kmax);
  const propS2 = new Float64Array(Kmax);
  const propW = new Float64Array(Kmax);
  const z = new Int32Array(N);
  let K_cur = 2;
  muBuf[0] = 20;
  muBuf[1] = 25;
  s2Buf[0] = 4;
  s2Buf[1] = 4;
  wBuf[0] = 0.5;
  wBuf[1] = 0.5;

  const Ktrace = new Int8Array(T);
  const accepts = { split: 0, merge: 0, birth: 0, death: 0, gibbs: 0 };
  const tries = { split: 0, merge: 0, birth: 0, death: 0, gibbs: 0 };
  // Snapshots at the most-visited K (we don't know yet; collect all and filter later)
  const snapK: number[] = [];
  const snapMuSorted: number[][] = [];

  function logTarget(K: number, mu: Float64Array, s2: Float64Array, w: Float64Array): number {
    let ll = 0;
    for (let i = 0; i < N; i++) {
      let p = 0;
      for (let k = 0; k < K; k++) p += w[k] * Math.exp(logNormPdf(GALAXY[i], mu[k], Math.sqrt(s2[k])));
      ll += Math.log(p + 1e-300);
    }
    let lp = 0;
    for (let k = 0; k < K; k++) {
      lp += logNormPdf(mu[k], xi, Math.sqrt(tau2));
      lp += logInvGammaPdf(s2[k], alphaSigma, betaSigma);
    }
    let lpw = lgamma(alphaW * K);
    for (let k = 0; k < K; k++) {
      lpw -= lgamma(alphaW);
      lpw += (alphaW - 1) * Math.log(Math.max(w[k], 1e-300));
    }
    return ll + lp + lpw - Math.log(Kmax);
  }

  function copyTo(K: number) {
    for (let k = 0; k < K; k++) {
      propMu[k] = muBuf[k];
      propS2[k] = s2Buf[k];
      propW[k] = wBuf[k];
    }
  }

  function attemptSplit() {
    tries.split++;
    if (K_cur >= Kmax) return;
    const j = Math.floor(rng() * K_cur);
    const u1 = 0.05 + rng() * 0.9;
    const u2 = 0.05 + rng() * 0.9;
    const u3 = 0.05 + rng() * 0.9;
    const wJ = wBuf[j];
    const muJ = muBuf[j];
    const sJ = Math.sqrt(s2Buf[j]);
    const w1 = u1 * wJ;
    const w2 = (1 - u1) * wJ;
    const mu1 = muJ - u2 * sJ * Math.sqrt(w2 / w1);
    const mu2 = muJ + u2 * sJ * Math.sqrt(w1 / w2);
    const s2_1 = u3 * (1 - u2 * u2) * s2Buf[j] * (wJ / w1);
    const s2_2 = (1 - u3) * (1 - u2 * u2) * s2Buf[j] * (wJ / w2);
    copyTo(K_cur);
    propMu[j] = mu1;
    propS2[j] = s2_1;
    propW[j] = w1;
    propMu[K_cur] = mu2;
    propS2[K_cur] = s2_2;
    propW[K_cur] = w2;
    const logDetJ = Math.log(wJ) + Math.log(Math.abs(mu2 - mu1)) + Math.log(s2_1) + Math.log(s2_2) - Math.log(s2Buf[j]) - Math.log(u2) - Math.log(1 - u2 * u2) - Math.log(u3) - Math.log(1 - u3);
    const logQu = logBetaPdf(u1, 2, 2) + logBetaPdf(u2, 2, 2);
    const logAlpha = logTarget(K_cur + 1, propMu, propS2, propW) - logTarget(K_cur, muBuf, s2Buf, wBuf) - logQu + logDetJ;
    if (Math.log(rng()) < logAlpha) {
      K_cur++;
      muBuf.set(propMu.subarray(0, K_cur));
      s2Buf.set(propS2.subarray(0, K_cur));
      wBuf.set(propW.subarray(0, K_cur));
      accepts.split++;
    }
  }
  function attemptMerge() {
    tries.merge++;
    if (K_cur <= 1) return;
    const order = Array.from({ length: K_cur }, (_, i) => i).sort((a, b) => muBuf[a] - muBuf[b]);
    const idx = Math.floor(rng() * (K_cur - 1));
    const j1 = order[idx];
    const j2 = order[idx + 1];
    const w1 = wBuf[j1];
    const w2 = wBuf[j2];
    const wJ = w1 + w2;
    const muJ = (w1 * muBuf[j1] + w2 * muBuf[j2]) / wJ;
    const s2_J = (w1 * (s2Buf[j1] + (muBuf[j1] - muJ) ** 2) + w2 * (s2Buf[j2] + (muBuf[j2] - muJ) ** 2)) / wJ;
    const sJ = Math.sqrt(s2_J);
    const u1 = w1 / wJ;
    const u2 = (muBuf[j2] - muJ) / (sJ * Math.sqrt(w1 / w2));
    const u3 = (s2Buf[j1] * (w1 / wJ)) / ((1 - u2 * u2) * s2_J);
    if (u1 <= 0 || u1 >= 1 || u2 <= 0 || u2 >= 1 || u3 <= 0 || u3 >= 1) return;
    const jKeep = Math.min(j1, j2);
    const jDrop = Math.max(j1, j2);
    let widx = 0;
    for (let k = 0; k < K_cur; k++) {
      if (k === jDrop) continue;
      if (k === jKeep) {
        propMu[widx] = muJ;
        propS2[widx] = s2_J;
        propW[widx] = wJ;
      } else {
        propMu[widx] = muBuf[k];
        propS2[widx] = s2Buf[k];
        propW[widx] = wBuf[k];
      }
      widx++;
    }
    const logDetJfwd = Math.log(wJ) + Math.log(Math.abs(muBuf[j2] - muBuf[j1])) + Math.log(s2Buf[j1]) + Math.log(s2Buf[j2]) - Math.log(s2_J) - Math.log(u2) - Math.log(1 - u2 * u2) - Math.log(u3) - Math.log(1 - u3);
    const logQu = logBetaPdf(u1, 2, 2) + logBetaPdf(u2, 2, 2);
    const logAlpha = logTarget(K_cur - 1, propMu, propS2, propW) - logTarget(K_cur, muBuf, s2Buf, wBuf) + logQu - logDetJfwd;
    if (Math.log(rng()) < logAlpha) {
      K_cur--;
      muBuf.set(propMu.subarray(0, K_cur));
      s2Buf.set(propS2.subarray(0, K_cur));
      wBuf.set(propW.subarray(0, K_cur));
      accepts.merge++;
    }
  }
  function attemptBirth() {
    tries.birth++;
    if (K_cur >= Kmax) return;
    const muNew = xi + Math.sqrt(tau2) * randn(rng);
    const s2New = invGammaSample(rng, alphaSigma, betaSigma);
    const g1 = gammaSample(rng, 1, 1);
    const g2 = gammaSample(rng, K_cur, 1);
    const uw = g1 / (g1 + g2);
    copyTo(K_cur);
    for (let k = 0; k < K_cur; k++) propW[k] = (1 - uw) * wBuf[k];
    propMu[K_cur] = muNew;
    propS2[K_cur] = s2New;
    propW[K_cur] = uw;
    const logQmu = logNormPdf(muNew, xi, Math.sqrt(tau2));
    const logQs2 = logInvGammaPdf(s2New, alphaSigma, betaSigma);
    const logAlpha = logTarget(K_cur + 1, propMu, propS2, propW) - logTarget(K_cur, muBuf, s2Buf, wBuf) - logQmu - logQs2 - Math.log(K_cur) - Math.log(K_cur + 1);
    if (Math.log(rng()) < logAlpha) {
      K_cur++;
      muBuf.set(propMu.subarray(0, K_cur));
      s2Buf.set(propS2.subarray(0, K_cur));
      wBuf.set(propW.subarray(0, K_cur));
      accepts.birth++;
    }
  }
  function attemptDeath() {
    tries.death++;
    if (K_cur <= 1) return;
    const j = Math.floor(rng() * K_cur);
    if (wBuf[j] >= 1) return;
    let idx = 0;
    for (let k = 0; k < K_cur; k++) {
      if (k === j) continue;
      propMu[idx] = muBuf[k];
      propS2[idx] = s2Buf[k];
      propW[idx] = wBuf[k] / (1 - wBuf[j]);
      idx++;
    }
    const logQmu = logNormPdf(muBuf[j], xi, Math.sqrt(tau2));
    const logQs2 = logInvGammaPdf(s2Buf[j], alphaSigma, betaSigma);
    const logAlpha = logTarget(K_cur - 1, propMu, propS2, propW) - logTarget(K_cur, muBuf, s2Buf, wBuf) + logQmu + logQs2 + Math.log(K_cur - 1) + Math.log(K_cur);
    if (Math.log(rng()) < logAlpha) {
      K_cur--;
      muBuf.set(propMu.subarray(0, K_cur));
      s2Buf.set(propS2.subarray(0, K_cur));
      wBuf.set(propW.subarray(0, K_cur));
      accepts.death++;
    }
  }
  function gibbsSweep() {
    tries.gibbs++;
    for (let i = 0; i < N; i++) {
      let maxLog = -Infinity;
      const lp = new Float64Array(K_cur);
      for (let k = 0; k < K_cur; k++) {
        lp[k] = Math.log(wBuf[k] + 1e-300) + logNormPdf(GALAXY[i], muBuf[k], Math.sqrt(s2Buf[k]));
        if (lp[k] > maxLog) maxLog = lp[k];
      }
      let sumExp = 0;
      for (let k = 0; k < K_cur; k++) sumExp += Math.exp(lp[k] - maxLog);
      const u = rng() * sumExp;
      let cum = 0;
      let chosen = K_cur - 1;
      for (let k = 0; k < K_cur; k++) {
        cum += Math.exp(lp[k] - maxLog);
        if (u < cum) { chosen = k; break; }
      }
      z[i] = chosen;
    }
    const ns = new Int32Array(K_cur);
    const sums = new Float64Array(K_cur);
    for (let i = 0; i < N; i++) {
      ns[z[i]]++;
      sums[z[i]] += GALAXY[i];
    }
    for (let k = 0; k < K_cur; k++) {
      const n_k = ns[k];
      const sig2 = s2Buf[k];
      const post_var = (tau2 * sig2) / (tau2 * n_k + sig2);
      const post_mean = post_var * (sums[k] / sig2 + xi / tau2);
      muBuf[k] = post_mean + Math.sqrt(post_var) * randn(rng);
      let ssq = 0;
      for (let i = 0; i < N; i++) if (z[i] === k) ssq += (GALAXY[i] - muBuf[k]) ** 2;
      s2Buf[k] = invGammaSample(rng, alphaSigma + n_k / 2, betaSigma + 0.5 * ssq);
    }
    const dirAlpha = Array.from({ length: K_cur }, (_, k) => alphaW + ns[k]);
    dirichletSample(rng, dirAlpha, wBuf);
    accepts.gibbs++;
  }

  for (let t = 0; t < T; t++) {
    const move = Math.floor(rng() * 5);
    if (move === 0) attemptSplit();
    else if (move === 1) attemptMerge();
    else if (move === 2) attemptBirth();
    else if (move === 3) attemptDeath();
    else gibbsSweep();
    Ktrace[t] = K_cur;
    if (t >= burn && (t - burn) % SNAPSHOT_THIN === 0) {
      snapK.push(K_cur);
      const m = new Array(K_cur);
      for (let k = 0; k < K_cur; k++) m[k] = muBuf[k];
      m.sort((a, b) => a - b);
      snapMuSorted.push(m);
    }
  }
  // Pick dominant K and extract smallest-mean trace
  const Kcounts = new Map<number, number>();
  for (const K of snapK) Kcounts.set(K, (Kcounts.get(K) ?? 0) + 1);
  let dominant = 3;
  let best = 0;
  for (const [K, c] of Kcounts) if (c > best) { best = c; dominant = K; }
  const muTraceSmallest: number[] = [];
  for (let i = 0; i < snapK.length; i++) if (snapK[i] === dominant) muTraceSmallest.push(snapMuSorted[i][0]);

  return { Ktrace, muTraceSmallest, accepts, tries };
}

// Initial-positive-sequence integrated autocorrelation time estimator (Geyer 1992)
function autocorrTimeIPS(x: number[]): { acf: number[]; tauIat: number } {
  const n = x.length;
  if (n < 4) return { acf: [1], tauIat: 1 };
  const mean = x.reduce((a, b) => a + b, 0) / n;
  const c0 = x.reduce((s, v) => s + (v - mean) * (v - mean), 0) / n;
  const maxLag = Math.min(100, Math.floor(n / 2));
  const acf: number[] = new Array(maxLag).fill(0);
  for (let lag = 0; lag < maxLag; lag++) {
    let s = 0;
    for (let i = 0; i < n - lag; i++) s += (x[i] - mean) * (x[i + lag] - mean);
    acf[lag] = c0 > 0 ? s / ((n - lag) * c0) : 0;
  }
  // IPS: sum over pairs (Γ_m = ρ_{2m} + ρ_{2m+1}) while positive
  let tau = -1; // -1 for the lag-0 double-count
  for (let m = 0; 2 * m + 1 < maxLag; m++) {
    const gamma = acf[2 * m] + acf[2 * m + 1];
    if (gamma <= 0) break;
    tau += 2 * gamma;
  }
  return { acf, tauIat: Math.max(1, tau) };
}

interface Committed {
  T: number;
  Kmax: number;
}
const DEFAULTS: Committed = { T: 5000, Kmax: 6 };

export default function RJDiagnosticsExplorer() {
  const [dT, setDT] = useState(DEFAULTS.T);
  const [dKmax, setDKmax] = useState(DEFAULTS.Kmax);
  const [committed, setCommitted] = useState<Committed>(DEFAULTS);

  const commit = useCallback((next: Partial<Committed>) => setCommitted((c) => ({ ...c, ...next })), []);

  const result = useMemo(() => {
    const burn = Math.floor(committed.T * 0.2);
    const ch = runDiagChain(committed.T, burn, committed.Kmax, 7);
    // Running fraction p_t(K) for K = 1..Kmax
    const T = ch.Ktrace.length;
    const Kvals = d3.range(1, committed.Kmax + 1);
    const running: { K: number; pts: number[] }[] = Kvals.map((K) => ({ K, pts: [] }));
    const totals = new Array(committed.Kmax + 1).fill(0);
    let nPost = 0;
    for (let t = 0; t < T; t++) {
      if (t < burn) continue;
      totals[ch.Ktrace[t]]++;
      nPost++;
      // Record every ~50 iterations for plotting
      if (nPost % 25 === 0 || t === T - 1) {
        for (const r of running) r.pts.push(totals[r.K] / nPost);
      }
    }
    // Final P̂
    const PKfinal = totals.map((c, K) => ({ K, p: c / Math.max(1, nPost) }));
    // ACF on within-dominant-K sub-series
    const { acf, tauIat } = autocorrTimeIPS(ch.muTraceSmallest);
    const essK = ch.muTraceSmallest.length / Math.max(tauIat, 1);
    return { ch, running, PKfinal, acf, tauIat, essK, nPost };
  }, [committed]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const W = containerWidth || 800;
  const isMobile = W < 640;
  const panelW = isMobile ? W : (W - 24) / 3;
  const panelH = 280;

  const runRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 16, right: 12, bottom: 36, left: 36 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const tMax = result.nPost;
      const x = d3.scaleLinear().domain([1, tMax]).range([0, w]);
      const y = d3.scaleLinear().domain([0, 0.6]).range([h, 0]);
      const colors = d3.schemeTableau10;

      // CLT band for each visible curve (only K with > 5% mass)
      result.running.forEach((r, idx) => {
        const pK = result.PKfinal[r.K]?.p ?? 0;
        if (pK < 0.02) return;
        const ts = r.pts.map((_, i) => Math.floor(((i + 1) * tMax) / r.pts.length));
        const sigmaAt = (t: number) => 2 * Math.sqrt((pK * (1 - pK)) / Math.max(t, 1));
        // Band
        const area = d3
          .area<number>()
          .x((_, i) => x(ts[i]))
          .y0((_, i) => y(Math.max(0, pK - sigmaAt(ts[i]))))
          .y1((_, i) => y(Math.min(1, pK + sigmaAt(ts[i]))))
          .curve(d3.curveMonotoneX);
        g.append('path').datum(r.pts).attr('d', area).attr('fill', colors[idx % colors.length]).attr('opacity', 0.15);
        // Curve
        const line = d3.line<number>().x((_, i) => x(ts[i])).y((d) => y(d)).curve(d3.curveMonotoneX);
        g.append('path').datum(r.pts).attr('d', line).attr('fill', 'none').attr('stroke', colors[idx % colors.length]).attr('stroke-width', 1.5);
        // Label at right
        if (r.pts.length > 0) {
          g.append('text')
            .attr('x', x(tMax) + 4)
            .attr('y', y(r.pts[r.pts.length - 1]))
            .style('font-size', '10px')
            .style('font-family', 'var(--font-mono)')
            .style('fill', colors[idx % colors.length])
            .text(`K=${r.K}`);
        }
      });

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(4, '~s')).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(y).ticks(5, '.2f')).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle').style('font-size', '11px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('post-burn iterations');
      g.append('text').attr('x', -h / 2).attr('y', -28).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('font-size', '11px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('P̂_t(K | D)');
    },
    [result, panelW, panelH],
  );

  const accRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 16, right: 12, bottom: 36, left: 36 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const moves = ['split', 'merge', 'birth', 'death', 'gibbs'] as const;
      const rates = moves.map((m) => ({
        m,
        rate: result.ch.tries[m] > 0 ? result.ch.accepts[m] / result.ch.tries[m] : 0,
      }));
      const x = d3.scaleBand().domain(moves as unknown as string[]).range([0, w]).padding(0.2);
      const y = d3.scaleLinear().domain([0, 1.05]).range([h, 0]);
      // Goldilocks zone 0.05 - 0.50
      g.append('rect')
        .attr('x', 0)
        .attr('y', y(0.5))
        .attr('width', w)
        .attr('height', y(0.05) - y(0.5))
        .style('fill', 'var(--color-accent)')
        .attr('opacity', 0.1);
      g.append('text').attr('x', 4).attr('y', y(0.5) + 12).style('font-size', '10px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text-secondary)').text('Goldilocks 5-50%');

      g.selectAll('rect.bar')
        .data(rates)
        .join('rect')
        .attr('class', 'bar')
        .attr('x', (d) => x(d.m) ?? 0)
        .attr('y', (d) => y(d.rate))
        .attr('width', x.bandwidth())
        .attr('height', (d) => h - y(d.rate))
        .style('fill', (d) =>
          d.m === 'gibbs' ? 'var(--color-text-secondary)' : d.rate >= 0.05 && d.rate <= 0.5 ? 'var(--color-accent)' : '#ef4444',
        );
      g.selectAll('text.lbl')
        .data(rates)
        .join('text')
        .attr('class', 'lbl')
        .attr('x', (d) => (x(d.m) ?? 0) + x.bandwidth() / 2)
        .attr('y', (d) => y(d.rate) - 4)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('font-family', 'var(--font-mono)')
        .style('fill', 'var(--color-text)')
        .text((d) => `${(d.rate * 100).toFixed(1)}%`);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(y).ticks(5, '.1f')).selectAll('text').style('fill', 'var(--color-text-secondary)');
    },
    [result, panelW, panelH],
  );

  const acfRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 16, right: 12, bottom: 36, left: 36 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const lags = d3.range(0, Math.min(40, result.acf.length));
      const x = d3.scaleBand().domain(lags.map((l) => l.toString())).range([0, w]).padding(0.1);
      const y = d3.scaleLinear().domain([-0.2, 1.05]).range([h, 0]);
      // Zero line
      g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0)).attr('y2', y(0)).style('stroke', 'var(--color-border)');
      g.selectAll('line.bar')
        .data(lags)
        .join('line')
        .attr('class', 'bar')
        .attr('x1', (l) => (x(l.toString()) ?? 0) + x.bandwidth() / 2)
        .attr('x2', (l) => (x(l.toString()) ?? 0) + x.bandwidth() / 2)
        .attr('y1', y(0))
        .attr('y2', (l) => y(result.acf[l]))
        .style('stroke', 'var(--color-accent)')
        .attr('stroke-width', 2);
      // Two-axis labels
      const tickLags = [0, 10, 20, 30];
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).tickValues(tickLags.map((l) => l.toString())).tickFormat((d) => d)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(y).ticks(5, '.1f')).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle').style('font-size', '11px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('lag');
      g.append('text').attr('x', w - 4).attr('y', 12).attr('text-anchor', 'end').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text)').text(`τ_iat ≈ ${result.tauIat.toFixed(1)}, ESS ≈ ${result.essK.toFixed(0)}`);
    },
    [result, panelW, panelH],
  );

  const sliderRow = (
    label: string,
    value: number,
    setVal: (n: number) => void,
    commitFn: (v: number) => void,
    min: number,
    max: number,
    step: number,
    fmt: (n: number) => string,
    aria: string,
  ) => (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        marginBottom: 6,
        fontFamily: 'var(--font-sans)',
        fontSize: 12,
        color: 'var(--color-text)',
      }}
    >
      <label style={{ minWidth: 110 }}>
        {label} = <strong>{fmt(value)}</strong>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => setVal(parseFloat(e.target.value))}
        onMouseUp={() => commitFn(value)}
        onTouchEnd={() => commitFn(value)}
        onKeyUp={() => commitFn(value)}
        style={{ flex: 1 }}
        aria-label={aria}
      />
    </div>
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ marginBottom: 8 }}>
        {sliderRow('chain T', dT, setDT, (v) => commit({ T: Math.round(v) }), 1000, 10000, 500, (n) => n.toFixed(0), 'Chain length T')}
        {sliderRow('K_max', dKmax, setDKmax, (v) => commit({ Kmax: Math.round(v) }), 4, 10, 1, (n) => n.toFixed(0), 'Maximum mixture components')}
      </div>
      <div
        style={{
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          gap: 16,
          alignItems: 'flex-start',
        }}
      >
        <div style={{ width: isMobile ? '100%' : panelW }}>
          <div style={{ marginBottom: 4, fontSize: 11, fontFamily: 'var(--font-sans)', color: 'var(--color-text-secondary)', textAlign: 'center' }}>running P̂_t(K | D) ± 2σ CLT</div>
          <svg ref={runRef} width={panelW} height={panelH} />
        </div>
        <div style={{ width: isMobile ? '100%' : panelW }}>
          <div style={{ marginBottom: 4, fontSize: 11, fontFamily: 'var(--font-sans)', color: 'var(--color-text-secondary)', textAlign: 'center' }}>per-move acceptance rates</div>
          <svg ref={accRef} width={panelW} height={panelH} />
        </div>
        <div style={{ width: isMobile ? '100%' : panelW }}>
          <div style={{ marginBottom: 4, fontSize: 11, fontFamily: 'var(--font-sans)', color: 'var(--color-text-secondary)', textAlign: 'center' }}>within-K ACF — smallest mean at dominant K</div>
          <svg ref={acfRef} width={panelW} height={panelH} />
        </div>
      </div>
    </div>
  );
}
