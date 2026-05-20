// =============================================================================
// GalaxyMixtureRJMCMC.tsx
//
// §10 of reversible-jump-mcmc. Full Galaxy RJ-MCMC chain — Richardson–Green
// split/merge + birth/death + Gibbs — on n = 82 velocity observations. Three
// panels: K-trace, posterior P(K|D), posterior predictive density overlay.
//
// Float64Array buffer hoisting per CLAUDE.md — pre-allocated mu/s2/w buffers
// of length K_MAX with in-place updates per move. Notebook uses np.concatenate
// for clarity; on-site viz hoists.
//
// Display-vs-committed: chain reruns only on slider release.
// Static fallback: /images/topics/reversible-jump-mcmc/fig10_galaxy_mixture.png
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

// RNG and sampling helpers
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
  // Marsaglia & Tsang for shape >= 1; boost otherwise.
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
function logBetaPdf(x: number, a: number, b: number): number {
  // a, b > 0; x in (0, 1)
  if (x <= 0 || x >= 1) return -Infinity;
  // log B(a,b) = lgamma(a) + lgamma(b) - lgamma(a+b)
  const logBeta = lgamma(a) + lgamma(b) - lgamma(a + b);
  return (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logBeta;
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
function logNormPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return -0.5 * Math.log(2 * Math.PI) - Math.log(sigma) - 0.5 * z * z;
}
function logInvGammaPdf(x: number, alpha: number, beta: number): number {
  if (x <= 0) return -Infinity;
  return alpha * Math.log(beta) - lgamma(alpha) - (alpha + 1) * Math.log(x) - beta / x;
}

interface Hyp {
  xi: number;
  tau2: number;
  alphaSigma: number;
  betaSigma: number;
  alphaW: number;
  Kmax: number;
}

function defaultHyp(Kmax: number, tau2Scale: number): Hyp {
  const yBar = GALAXY.reduce((s, x) => s + x, 0) / N;
  const yVar = GALAXY.reduce((s, x) => s + (x - yBar) ** 2, 0) / N;
  const R = Math.max(...GALAXY) - Math.min(...GALAXY);
  return {
    xi: yBar,
    tau2: (R * R / 16) * tau2Scale,
    alphaSigma: 2,
    betaSigma: yVar / 4,
    alphaW: 1,
    Kmax,
  };
}

// Log target density of (K, theta_K) ∝ likelihood * priors * p(K).
function logTargetActive(K_cur: number, mu: Float64Array, s2: Float64Array, w: Float64Array, hyp: Hyp): number {
  // Likelihood — sum over data of log of mixture density
  let ll = 0;
  for (let i = 0; i < N; i++) {
    const yi = GALAXY[i];
    let p = 0;
    for (let k = 0; k < K_cur; k++) p += w[k] * Math.exp(logNormPdf(yi, mu[k], Math.sqrt(s2[k])));
    ll += Math.log(p + 1e-300);
  }
  let lp = 0;
  for (let k = 0; k < K_cur; k++) {
    lp += logNormPdf(mu[k], hyp.xi, Math.sqrt(hyp.tau2));
    lp += logInvGammaPdf(s2[k], hyp.alphaSigma, hyp.betaSigma);
  }
  // Dirichlet log prior on w
  let lpw = lgamma(hyp.alphaW * K_cur);
  for (let k = 0; k < K_cur; k++) {
    lpw -= lgamma(hyp.alphaW);
    lpw += (hyp.alphaW - 1) * Math.log(Math.max(w[k], 1e-300));
  }
  return ll + lp + lpw - Math.log(hyp.Kmax); // uniform K prior
}

interface ChainState {
  Ktrace: Int8Array;
  // For posterior predictive: snapshots of (K, mu, s2, w)
  snapMu: number[][];
  snapS2: number[][];
  snapW: number[][];
  snapK: number[];
  accepts: { split: number; merge: number; birth: number; death: number; gibbs: number };
  tries: { split: number; merge: number; birth: number; death: number; gibbs: number };
  betweenKJumps: number;
}

const SNAPSHOT_THIN = 4;

function runGalaxyChain(T: number, burn: number, hyp: Hyp, seed: number): ChainState {
  const rng = mulberry32(seed);
  const Kmax = hyp.Kmax;
  const muBuf = new Float64Array(Kmax);
  const s2Buf = new Float64Array(Kmax);
  const wBuf = new Float64Array(Kmax);
  let K_cur = 2;
  // Init: two components at the 25th / 75th quantiles of the data.
  const galaxySorted = GALAXY.slice().sort((a, b) => a - b);
  muBuf[0] = d3.quantile(galaxySorted, 0.25) ?? 20;
  muBuf[1] = d3.quantile(galaxySorted, 0.75) ?? 24;
  s2Buf[0] = 4;
  s2Buf[1] = 4;
  wBuf[0] = 0.5;
  wBuf[1] = 0.5;

  const Ktrace = new Int8Array(T);
  const accepts = { split: 0, merge: 0, birth: 0, death: 0, gibbs: 0 };
  const tries = { split: 0, merge: 0, birth: 0, death: 0, gibbs: 0 };
  let betweenKJumps = 0;
  let lastK = K_cur;
  const snapMu: number[][] = [];
  const snapS2: number[][] = [];
  const snapW: number[][] = [];
  const snapK: number[] = [];

  // Scratch buffers for move proposals (hoisted)
  const propMu = new Float64Array(Kmax);
  const propS2 = new Float64Array(Kmax);
  const propW = new Float64Array(Kmax);

  function copyTo(srcMu: Float64Array, srcS2: Float64Array, srcW: Float64Array, dstMu: Float64Array, dstS2: Float64Array, dstW: Float64Array, K: number): void {
    for (let k = 0; k < K; k++) {
      dstMu[k] = srcMu[k];
      dstS2[k] = srcS2[k];
      dstW[k] = srcW[k];
    }
  }

  function logProb(K: number, mu: Float64Array, s2: Float64Array, w: Float64Array): number {
    return logTargetActive(K, mu, s2, w, hyp);
  }

  function attemptSplit(): void {
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
    // Order j1 < j2 by mean
    // Copy non-j components into propMu (length K_cur+1) and insert mu1 < mu2.
    // Place j_1 at j (replace) and j_2 appended at K_cur.
    copyTo(muBuf, s2Buf, wBuf, propMu, propS2, propW, K_cur);
    propMu[j] = mu1;
    propS2[j] = s2_1;
    propW[j] = w1;
    propMu[K_cur] = mu2;
    propS2[K_cur] = s2_2;
    propW[K_cur] = w2;

    const logDetJ =
      Math.log(wJ) +
      Math.log(Math.abs(mu2 - mu1)) +
      Math.log(s2_1) +
      Math.log(s2_2) -
      Math.log(s2Buf[j]) -
      Math.log(u2) -
      Math.log(1 - u2 * u2) -
      Math.log(u3) -
      Math.log(1 - u3);

    const logQu = logBetaPdf(u1, 2, 2) + logBetaPdf(u2, 2, 2) + 0; // u3 ~ Uniform(0,1) → log = 0
    const logAlpha = logProb(K_cur + 1, propMu, propS2, propW) - logProb(K_cur, muBuf, s2Buf, wBuf) - logQu + logDetJ;

    if (Math.log(rng()) < logAlpha) {
      K_cur++;
      muBuf.set(propMu.subarray(0, K_cur));
      s2Buf.set(propS2.subarray(0, K_cur));
      wBuf.set(propW.subarray(0, K_cur));
      accepts.split++;
    }
  }

  function attemptMerge(): void {
    tries.merge++;
    if (K_cur <= 1) return;
    // Pick an adjacent pair in mean-sorted order
    const order = Array.from({ length: K_cur }, (_, i) => i).sort((a, b) => muBuf[a] - muBuf[b]);
    const pairIdx = Math.floor(rng() * (K_cur - 1));
    const j1 = order[pairIdx];
    const j2 = order[pairIdx + 1];
    const w1 = wBuf[j1];
    const w2 = wBuf[j2];
    const mu1 = muBuf[j1];
    const mu2 = muBuf[j2];
    const s2_1 = s2Buf[j1];
    const s2_2 = s2Buf[j2];
    const wJ = w1 + w2;
    const muJ = (w1 * mu1 + w2 * mu2) / wJ;
    // Recover u_1, u_2, u_3 by inversion
    const u1 = w1 / wJ;
    // From mu definitions: μ_2 - μ_1 = u_2 σ_J (sqrt(w1/w2) + sqrt(w2/w1)) = u_2 σ_J / sqrt(w1*w2) * (w1+w2)/sqrt(w1w2)... simpler: invert via s2_J = w1*(s2_1 + (mu1-muJ)^2)/wJ + w2*(s2_2 + (mu2-muJ)^2)/wJ
    const s2_J = (w1 * (s2_1 + (mu1 - muJ) ** 2) + w2 * (s2_2 + (mu2 - muJ) ** 2)) / wJ;
    const sJ = Math.sqrt(s2_J);
    const u2 = (mu2 - muJ) / (sJ * Math.sqrt(w1 / w2));
    const u3 = (s2_1 * (w1 / wJ)) / ((1 - u2 * u2) * s2_J);
    if (u1 <= 0 || u1 >= 1 || u2 <= 0 || u2 >= 1 || u3 <= 0 || u3 >= 1) return; // out of support → reject silently

    // Build proposal of K_cur - 1 components: keep all except j1, j2; put merged at min(j1, j2)
    const jKeep = Math.min(j1, j2);
    const jDrop = Math.max(j1, j2);
    let idx = 0;
    for (let k = 0; k < K_cur; k++) {
      if (k === jDrop) continue;
      if (k === jKeep) {
        propMu[idx] = muJ;
        propS2[idx] = s2_J;
        propW[idx] = wJ;
      } else {
        propMu[idx] = muBuf[k];
        propS2[idx] = s2Buf[k];
        propW[idx] = wBuf[k];
      }
      idx++;
    }
    const logDetJfwd =
      Math.log(wJ) +
      Math.log(Math.abs(mu2 - mu1)) +
      Math.log(s2_1) +
      Math.log(s2_2) -
      Math.log(s2_J) -
      Math.log(u2) -
      Math.log(1 - u2 * u2) -
      Math.log(u3) -
      Math.log(1 - u3);
    const logQu = logBetaPdf(u1, 2, 2) + logBetaPdf(u2, 2, 2) + 0;
    // For merge, acceptance is inverse direction: logAlpha = logπ(merged) - logπ(current) - log|det J_fwd| + log q_u
    const logAlpha = logProb(K_cur - 1, propMu, propS2, propW) - logProb(K_cur, muBuf, s2Buf, wBuf) + logQu - logDetJfwd;

    if (Math.log(rng()) < logAlpha) {
      K_cur--;
      muBuf.set(propMu.subarray(0, K_cur));
      s2Buf.set(propS2.subarray(0, K_cur));
      wBuf.set(propW.subarray(0, K_cur));
      accepts.merge++;
    }
  }

  function attemptBirth(): void {
    tries.birth++;
    if (K_cur >= Kmax) return;
    // Sample new component from prior
    const muNew = hyp.xi + Math.sqrt(hyp.tau2) * randn(rng);
    const s2New = invGammaSample(rng, hyp.alphaSigma, hyp.betaSigma);
    // u_w ~ Beta(1, K)
    const g1 = gammaSample(rng, 1, 1);
    const g2 = gammaSample(rng, K_cur, 1);
    const uw = g1 / (g1 + g2);

    copyTo(muBuf, s2Buf, wBuf, propMu, propS2, propW, K_cur);
    for (let k = 0; k < K_cur; k++) propW[k] = (1 - uw) * wBuf[k];
    propMu[K_cur] = muNew;
    propS2[K_cur] = s2New;
    propW[K_cur] = uw;

    // After Jacobian cancellation: log A = logπ(K+1) - logπ(K) - log q(μ_new) - log q(σ²_new) - log K - log(K+1)
    const logQmu = logNormPdf(muNew, hyp.xi, Math.sqrt(hyp.tau2));
    const logQs2 = logInvGammaPdf(s2New, hyp.alphaSigma, hyp.betaSigma);
    const logAlpha = logProb(K_cur + 1, propMu, propS2, propW) - logProb(K_cur, muBuf, s2Buf, wBuf) - logQmu - logQs2 - Math.log(K_cur) - Math.log(K_cur + 1);

    if (Math.log(rng()) < logAlpha) {
      K_cur++;
      muBuf.set(propMu.subarray(0, K_cur));
      s2Buf.set(propS2.subarray(0, K_cur));
      wBuf.set(propW.subarray(0, K_cur));
      accepts.birth++;
    }
  }

  function attemptDeath(): void {
    tries.death++;
    if (K_cur <= 1) return;
    const j = Math.floor(rng() * K_cur);
    const muJ = muBuf[j];
    const s2J = s2Buf[j];
    const wJ = wBuf[j];
    if (wJ >= 1) return; // degenerate
    let idx = 0;
    for (let k = 0; k < K_cur; k++) {
      if (k === j) continue;
      propMu[idx] = muBuf[k];
      propS2[idx] = s2Buf[k];
      propW[idx] = wBuf[k] / (1 - wJ);
      idx++;
    }
    const logQmu = logNormPdf(muJ, hyp.xi, Math.sqrt(hyp.tau2));
    const logQs2 = logInvGammaPdf(s2J, hyp.alphaSigma, hyp.betaSigma);
    const logAlpha = logProb(K_cur - 1, propMu, propS2, propW) - logProb(K_cur, muBuf, s2Buf, wBuf) + logQmu + logQs2 + Math.log(K_cur - 1) + Math.log(K_cur);

    if (Math.log(rng()) < logAlpha) {
      K_cur--;
      muBuf.set(propMu.subarray(0, K_cur));
      s2Buf.set(propS2.subarray(0, K_cur));
      wBuf.set(propW.subarray(0, K_cur));
      accepts.death++;
    }
  }

  // Z assignment buffer
  const z = new Int32Array(N);
  function gibbsSweep(): void {
    tries.gibbs++;
    // E-step on z
    for (let i = 0; i < N; i++) {
      let totalLog = -Infinity;
      const logProbs = new Float64Array(K_cur);
      for (let k = 0; k < K_cur; k++) {
        logProbs[k] = Math.log(wBuf[k] + 1e-300) + logNormPdf(GALAXY[i], muBuf[k], Math.sqrt(s2Buf[k]));
        if (logProbs[k] > totalLog) totalLog = logProbs[k];
      }
      let sumExp = 0;
      for (let k = 0; k < K_cur; k++) sumExp += Math.exp(logProbs[k] - totalLog);
      const u = rng() * sumExp;
      let cum = 0;
      let chosen = K_cur - 1;
      for (let k = 0; k < K_cur; k++) {
        cum += Math.exp(logProbs[k] - totalLog);
        if (u < cum) {
          chosen = k;
          break;
        }
      }
      z[i] = chosen;
    }
    // Component counts and sums
    const ns = new Int32Array(K_cur);
    const sums = new Float64Array(K_cur);
    const sumSq = new Float64Array(K_cur);
    for (let i = 0; i < N; i++) {
      const k = z[i];
      ns[k]++;
      sums[k] += GALAXY[i];
      sumSq[k] += GALAXY[i] * GALAXY[i];
    }
    // Update mu, sigma2, w
    for (let k = 0; k < K_cur; k++) {
      const n_k = ns[k];
      // mu | rest ~ N((τ²·sum + σ²·ξ)/(τ²·n_k + σ²), τ²σ²/(τ²·n_k + σ²))
      const sig2 = s2Buf[k];
      const post_var_mu = (hyp.tau2 * sig2) / (hyp.tau2 * n_k + sig2);
      const post_mean_mu = post_var_mu * (sums[k] / sig2 + hyp.xi / hyp.tau2);
      muBuf[k] = post_mean_mu + Math.sqrt(post_var_mu) * randn(rng);
      // sigma² | rest ~ InvGamma(alphaSigma + n_k/2, betaSigma + 0.5 Σ(yi - μ_k)²)
      let ssq = 0;
      for (let i = 0; i < N; i++) if (z[i] === k) ssq += (GALAXY[i] - muBuf[k]) ** 2;
      s2Buf[k] = invGammaSample(rng, hyp.alphaSigma + n_k / 2, hyp.betaSigma + 0.5 * ssq);
    }
    // w | rest ~ Dir(alphaW + n_1, …, alphaW + n_K)
    const dirAlpha = Array.from({ length: K_cur }, (_, k) => hyp.alphaW + ns[k]);
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
    if (t > 0 && Ktrace[t] !== lastK) betweenKJumps++;
    lastK = K_cur;
    if (t >= burn && (t - burn) % SNAPSHOT_THIN === 0) {
      snapK.push(K_cur);
      const m = new Array(K_cur);
      const s = new Array(K_cur);
      const w = new Array(K_cur);
      for (let k = 0; k < K_cur; k++) {
        m[k] = muBuf[k];
        s[k] = s2Buf[k];
        w[k] = wBuf[k];
      }
      snapMu.push(m);
      snapS2.push(s);
      snapW.push(w);
    }
  }

  return { Ktrace, snapMu, snapS2, snapW, snapK, accepts, tries, betweenKJumps };
}

interface Committed {
  Kmax: number;
  tau2Scale: number;
  T: number;
}
const DEFAULTS: Committed = { Kmax: 6, tau2Scale: 1, T: 5000 };

export default function GalaxyMixtureRJMCMC() {
  const [dKmax, setDKmax] = useState(DEFAULTS.Kmax);
  const [dTau, setDTau] = useState(DEFAULTS.tau2Scale);
  const [dT, setDT] = useState(DEFAULTS.T);
  const [committed, setCommitted] = useState<Committed>(DEFAULTS);

  const commit = useCallback((next: Partial<Committed>) => {
    setCommitted((c) => ({ ...c, ...next }));
  }, []);

  const result = useMemo(() => {
    const hyp = defaultHyp(committed.Kmax, committed.tau2Scale);
    const burn = Math.floor(committed.T * 0.2);
    const ch = runGalaxyChain(committed.T, burn, hyp, 7);
    // P(K | D) over post-burn
    const Kcounts = new Array(committed.Kmax + 1).fill(0);
    for (let t = burn; t < committed.T; t++) Kcounts[ch.Ktrace[t]]++;
    const total = committed.T - burn;
    const PK = Kcounts.map((c) => c / total);
    let EK = 0;
    for (let k = 0; k <= committed.Kmax; k++) EK += k * PK[k];
    const modeK = PK.indexOf(Math.max(...PK));
    // Posterior predictive density on grid
    const xs = d3.range(8, 36, 0.1);
    const dens = new Float64Array(xs.length);
    const NSnap = ch.snapMu.length;
    for (let s = 0; s < NSnap; s++) {
      const mu = ch.snapMu[s];
      const s2 = ch.snapS2[s];
      const w = ch.snapW[s];
      for (let i = 0; i < xs.length; i++) {
        let p = 0;
        for (let k = 0; k < mu.length; k++) p += w[k] * Math.exp(logNormPdf(xs[i], mu[k], Math.sqrt(s2[k])));
        dens[i] += p;
      }
    }
    if (NSnap > 0) for (let i = 0; i < xs.length; i++) dens[i] /= NSnap;
    return { ch, PK, EK, modeK, xs, dens, burn };
  }, [committed]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const W = containerWidth || 800;
  const isMobile = W < 640;
  const panelW = isMobile ? W : (W - 24) / 3;
  const panelH = 280;

  const traceRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 16, right: 12, bottom: 36, left: 32 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const T = result.ch.Ktrace.length;
      const x = d3.scaleLinear().domain([0, T]).range([0, w]);
      const y = d3.scaleLinear().domain([0, committed.Kmax + 1]).range([h, 0]);
      const downsample = Math.max(1, Math.floor(T / 600));
      const pts: { t: number; K: number }[] = [];
      for (let t = 0; t < T; t += downsample) pts.push({ t, K: result.ch.Ktrace[t] });
      const line = d3.line<{ t: number; K: number }>().x((d) => x(d.t)).y((d) => y(d.K)).curve(d3.curveStepAfter);
      g.append('path').datum(pts).attr('d', line).style('fill', 'none').style('stroke', 'var(--color-text)').attr('stroke-width', 1);
      g.append('line').attr('x1', x(result.burn)).attr('x2', x(result.burn)).attr('y1', 0).attr('y2', h).style('stroke', 'var(--color-text-secondary)').attr('stroke-dasharray', '3 3');
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5, '~s')).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(y).ticks(committed.Kmax)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle').style('font-size', '11px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('iteration t');
      g.append('text').attr('x', -h / 2).attr('y', -22).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('font-size', '11px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('K^(t)');
    },
    [result, committed, panelW, panelH],
  );

  const pkRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 16, right: 12, bottom: 36, left: 32 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const ks = d3.range(1, committed.Kmax + 1);
      const x = d3.scaleBand().domain(ks.map((k) => k.toString())).range([0, w]).padding(0.2);
      const yMax = Math.max(...result.PK, 0.1) * 1.2;
      const y = d3.scaleLinear().domain([0, yMax]).range([h, 0]);
      g.selectAll('rect')
        .data(ks)
        .join('rect')
        .attr('x', (k) => x(k.toString()) ?? 0)
        .attr('y', (k) => y(result.PK[k]))
        .attr('width', x.bandwidth())
        .attr('height', (k) => h - y(result.PK[k]))
        .style('fill', (k) => (k === result.modeK ? 'var(--color-accent)' : 'var(--color-text-secondary)'));
      g.selectAll('text.val')
        .data(ks)
        .join('text')
        .attr('class', 'val')
        .attr('x', (k) => (x(k.toString()) ?? 0) + x.bandwidth() / 2)
        .attr('y', (k) => y(result.PK[k]) - 4)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('font-family', 'var(--font-mono)')
        .style('fill', 'var(--color-text)')
        .text((k) => result.PK[k].toFixed(2));
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(y).ticks(5, '.2f')).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle').style('font-size', '11px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('K');
      g.append('text').attr('x', -h / 2).attr('y', -22).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('font-size', '11px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('P(K | D)');
    },
    [result, committed, panelW, panelH],
  );

  const predRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 16, right: 12, bottom: 36, left: 32 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const xDomain: [number, number] = [8, 36];
      const x = d3.scaleLinear().domain(xDomain).range([0, w]);
      const bins = d3.bin().domain(xDomain).thresholds(d3.range(xDomain[0], xDomain[1] + 0.5, 1))(GALAXY);
      const histDens = bins.map((b) => b.length / N);
      const densArr = Array.from(result.dens);
      const yMax = Math.max(Math.max(...histDens), Math.max(...densArr)) * 1.15;
      const y = d3.scaleLinear().domain([0, yMax]).range([h, 0]);
      g.selectAll('rect.hist')
        .data(bins)
        .join('rect')
        .attr('class', 'hist')
        .attr('x', (b) => x(b.x0!))
        .attr('y', (b) => y(b.length / N))
        .attr('width', (b) => Math.max(0, x(b.x1!) - x(b.x0!) - 1))
        .attr('height', (b) => h - y(b.length / N))
        .style('fill', 'var(--color-border)')
        .attr('opacity', 0.7);

      const densLine = d3.line<number>().x((_, i) => x(result.xs[i])).y((d) => y(d)).curve(d3.curveMonotoneX);
      g.append('path').datum(densArr).attr('d', densLine).style('fill', 'none').style('stroke', 'var(--color-accent)').attr('stroke-width', 2);
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(6)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(y).ticks(5, '.2f')).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle').style('font-size', '11px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('velocity y');
    },
    [result, panelW, panelH],
  );

  const totalTrans = result.ch.tries.split + result.ch.tries.merge + result.ch.tries.birth + result.ch.tries.death;
  const acceptedTrans = result.ch.accepts.split + result.ch.accepts.merge + result.ch.accepts.birth + result.ch.accepts.death;

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
        {sliderRow('K_max', dKmax, setDKmax, (v) => commit({ Kmax: Math.round(v) }), 4, 10, 1, (n) => n.toFixed(0), 'Maximum mixture components K_max')}
        {sliderRow('τ² scale', dTau, setDTau, (v) => commit({ tau2Scale: v }), 0.25, 4, 0.25, (n) => n.toFixed(2), 'Prior variance τ² scale')}
        {sliderRow('chain length T', dT, setDT, (v) => commit({ T: Math.round(v) }), 1000, 10000, 500, (n) => n.toFixed(0), 'Chain length T')}
      </div>
      <div
        style={{
          marginBottom: 8,
          fontFamily: 'var(--font-mono)',
          fontSize: 11,
          color: 'var(--color-text-secondary)',
        }}
      >
        E[K|D] = {result.EK.toFixed(2)} · mode K = {result.modeK} · between-K jump rate {((result.ch.betweenKJumps / committed.T) * 100).toFixed(1)}% ·
        trans-dim accept {totalTrans > 0 ? ((acceptedTrans / totalTrans) * 100).toFixed(1) : '0'}% ·
        split {(result.ch.tries.split > 0 ? (result.ch.accepts.split / result.ch.tries.split * 100).toFixed(1) : '0')}%, merge {(result.ch.tries.merge > 0 ? (result.ch.accepts.merge / result.ch.tries.merge * 100).toFixed(1) : '0')}%, birth {(result.ch.tries.birth > 0 ? (result.ch.accepts.birth / result.ch.tries.birth * 100).toFixed(1) : '0')}%, death {(result.ch.tries.death > 0 ? (result.ch.accepts.death / result.ch.tries.death * 100).toFixed(1) : '0')}%
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
          <div style={{ marginBottom: 4, fontSize: 11, fontFamily: 'var(--font-sans)', color: 'var(--color-text-secondary)', textAlign: 'center' }}>K-trace</div>
          <svg ref={traceRef} width={panelW} height={panelH} />
        </div>
        <div style={{ width: isMobile ? '100%' : panelW }}>
          <div style={{ marginBottom: 4, fontSize: 11, fontFamily: 'var(--font-sans)', color: 'var(--color-text-secondary)', textAlign: 'center' }}>P(K | D)</div>
          <svg ref={pkRef} width={panelW} height={panelH} />
        </div>
        <div style={{ width: isMobile ? '100%' : panelW }}>
          <div style={{ marginBottom: 4, fontSize: 11, fontFamily: 'var(--font-sans)', color: 'var(--color-text-secondary)', textAlign: 'center' }}>posterior predictive density</div>
          <svg ref={predRef} width={panelW} height={panelH} />
        </div>
      </div>
    </div>
  );
}
