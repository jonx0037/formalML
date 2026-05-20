// =============================================================================
// MasterAcceptanceRatioChain.tsx
//
// §7 of reversible-jump-mcmc. Numerical verification that Theorem 3's master
// acceptance ratio gives correct posterior model probabilities on a toy
// two-model setup:
//   M_1: y ~ N(0, 1)              (null)
//   M_2: y ~ N(β, 1), β ~ N(0, σ_β²)  (one-parameter)
// Equal model priors. Closed-form via Gaussian completion-of-squares.
//
// Display-vs-committed: chain reruns only on slider release. Notebook default:
// β* = 0.3, n = 50, σ_β² = 4, T = 2000 → P(M_2|D) = 0.253, chain 0.223.
//
// Static fallback: /images/topics/reversible-jump-mcmc/fig07_master_acceptance.png
// =============================================================================
import { useCallback, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';

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
function randn(rng: () => number): number {
  const u = 1 - rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Closed-form log marginal likelihoods for the two models.
// y_i ~ N(μ, 1) with μ either 0 (M_1) or β (M_2) and β ~ N(0, σ_β²).
function logMarg(y: number[], model: 1 | 2, sigmaB2: number): number {
  const n = y.length;
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumY2 = y.reduce((a, b) => a + b * b, 0);
  if (model === 1) {
    // log N(y | 0, I) marginal — no parameter to integrate.
    return -0.5 * n * Math.log(2 * Math.PI) - 0.5 * sumY2;
  }
  // M_2: ∫ p(y | β) p(β) dβ, both Gaussian — closed-form completion of squares.
  // Posterior variance: σ_post² = 1 / (1/σ_β² + n);
  // Posterior mean: μ_post = σ_post² · sumY.
  const sigmaPost2 = 1 / (1 / sigmaB2 + n);
  const muPost = sigmaPost2 * sumY;
  // Marginal log density:
  //  -0.5 n log(2π) - 0.5 log(σ_β²) - 0.5 sumY² + 0.5 log(σ_post²) + 0.5 μ_post² / σ_post²
  return (
    -0.5 * n * Math.log(2 * Math.PI) -
    0.5 * Math.log(sigmaB2) -
    0.5 * sumY2 +
    0.5 * Math.log(sigmaPost2) +
    0.5 * (muPost * muPost) / sigmaPost2
  );
}

interface ChainResult {
  Mtrace: Int8Array;
  betaTrace: Float64Array;
  pHat: number;
  acceptDim: number;
  acceptWith: number;
}

function runChain(
  y: number[],
  sigmaB2: number,
  T: number,
  burn: number,
  seed: number,
): ChainResult {
  const rng = mulberry32(seed);
  // State: (M, β); β meaningful only when M=2
  let M: 1 | 2 = 1;
  let beta = 0;
  const Mtrace = new Int8Array(T);
  const betaTrace = new Float64Array(T);
  const sigmaU = Math.sqrt(sigmaB2); // proposal scale matches prior — reasonable default
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumY2 = y.reduce((a, b) => a + b * b, 0);
  const n = y.length;

  // log target π(M, β | D) ∝ p(y|M,β) · p(β|M) · p(M)
  const logTarget = (M: 1 | 2, b: number): number => {
    const pM = -Math.log(2); // equal model priors
    if (M === 1) {
      return -0.5 * n * Math.log(2 * Math.PI) - 0.5 * sumY2 + pM;
    }
    // M=2: log p(y|β) + log p(β)
    const ll = -0.5 * n * Math.log(2 * Math.PI) - 0.5 * (sumY2 - 2 * b * sumY + n * b * b);
    const lp = -0.5 * Math.log(2 * Math.PI * sigmaB2) - 0.5 * (b * b) / sigmaB2;
    return ll + lp + pM;
  };

  let tries = { dim: 0, within: 0 };
  let accepts = { dim: 0, within: 0 };

  for (let t = 0; t < T; t++) {
    // Random scan: 0.5 dim-changing, 0.5 within-model (only meaningful in M=2; else burn an iter)
    if (rng() < 0.5) {
      tries.dim++;
      if (M === 1) {
        // Add move: propose β ~ N(0, σ_u²)
        const uProp = sigmaU * randn(rng);
        const logQu = -0.5 * Math.log(2 * Math.PI * sigmaU * sigmaU) - 0.5 * (uProp * uProp) / (sigmaU * sigmaU);
        const logAlpha = logTarget(2, uProp) - logTarget(1, 0) - logQu;
        if (Math.log(rng()) < logAlpha) {
          M = 2;
          beta = uProp;
          accepts.dim++;
        }
      } else {
        // Drop move: deterministic, u = beta_current
        const logQu = -0.5 * Math.log(2 * Math.PI * sigmaU * sigmaU) - 0.5 * (beta * beta) / (sigmaU * sigmaU);
        const logAlpha = logTarget(1, 0) - logTarget(2, beta) + logQu;
        if (Math.log(rng()) < logAlpha) {
          M = 1;
          beta = 0;
          accepts.dim++;
        }
      }
    } else if (M === 2) {
      // Within-model RWM update on β
      tries.within++;
      const prop = beta + 0.5 * randn(rng);
      const logAlpha = logTarget(2, prop) - logTarget(2, beta);
      if (Math.log(rng()) < logAlpha) {
        beta = prop;
        accepts.within++;
      }
    }
    Mtrace[t] = M;
    betaTrace[t] = beta;
  }

  let count2 = 0;
  for (let t = burn; t < T; t++) if (Mtrace[t] === 2) count2++;
  const pHat = count2 / (T - burn);
  return {
    Mtrace,
    betaTrace,
    pHat,
    acceptDim: tries.dim > 0 ? accepts.dim / tries.dim : 0,
    acceptWith: tries.within > 0 ? accepts.within / tries.within : 0,
  };
}

function generateData(betaStar: number, n: number, seed: number): number[] {
  const rng = mulberry32(seed);
  return Array.from({ length: n }, () => betaStar + randn(rng));
}

interface Committed {
  betaStar: number;
  sigmaB2: number;
  n: number;
  T: number;
}
const DEFAULTS: Committed = { betaStar: 0.3, sigmaB2: 4.0, n: 50, T: 2000 };

export default function MasterAcceptanceRatioChain() {
  const [dBeta, setDBeta] = useState(DEFAULTS.betaStar);
  const [dSig2, setDSig2] = useState(DEFAULTS.sigmaB2);
  const [dN, setDN] = useState(DEFAULTS.n);
  const [dT, setDT] = useState(DEFAULTS.T);
  const [committed, setCommitted] = useState<Committed>(DEFAULTS);

  const commit = useCallback((next: Partial<Committed>) => {
    setCommitted((c) => ({ ...c, ...next }));
  }, []);

  const result = useMemo(() => {
    const y = generateData(committed.betaStar, committed.n, 11);
    // Closed-form
    const lm1 = logMarg(y, 1, committed.sigmaB2);
    const lm2 = logMarg(y, 2, committed.sigmaB2);
    // Equal model priors cancel
    const maxL = Math.max(lm1, lm2);
    const pM1 = Math.exp(lm1 - maxL) / (Math.exp(lm1 - maxL) + Math.exp(lm2 - maxL));
    const pM2 = 1 - pM1;
    // Chain
    const chain = runChain(y, committed.sigmaB2, committed.T, Math.floor(committed.T * 0.25), 42);
    return { pM1, pM2, ...chain };
  }, [committed]);

  const stderr = Math.sqrt((result.pM2 * (1 - result.pM2)) / Math.floor(committed.T * 0.75));
  const sigmaCount = Math.abs(result.pHat - result.pM2) / Math.max(stderr, 1e-9);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const W = containerWidth || 800;
  const isMobile = W < 640;
  const panelW = isMobile ? W : (W - 24) / 2;
  const panelH = 300;

  const barRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 20, right: 18, bottom: 32, left: 40 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const data = [
        { model: 'M₁', closed: result.pM1, chain: 1 - result.pHat },
        { model: 'M₂', closed: result.pM2, chain: result.pHat },
      ];
      const x = d3.scaleBand().domain(data.map((d) => d.model)).range([0, w]).padding(0.3);
      const xSub = d3.scaleBand().domain(['closed', 'chain']).range([0, x.bandwidth()]).padding(0.08);
      const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);

      for (const d of data) {
        g.append('rect')
          .attr('x', (x(d.model) ?? 0) + (xSub('closed') ?? 0))
          .attr('y', y(d.closed))
          .attr('width', xSub.bandwidth())
          .attr('height', h - y(d.closed))
          .style('fill', 'var(--color-text-secondary)');
        g.append('rect')
          .attr('x', (x(d.model) ?? 0) + (xSub('chain') ?? 0))
          .attr('y', y(d.chain))
          .attr('width', xSub.bandwidth())
          .attr('height', h - y(d.chain))
          .style('fill', 'var(--color-accent)');
        g.append('text')
          .attr('x', (x(d.model) ?? 0) + (xSub('closed') ?? 0) + xSub.bandwidth() / 2)
          .attr('y', y(d.closed) - 4)
          .attr('text-anchor', 'middle')
          .style('font-size', '10px')
          .style('font-family', 'var(--font-mono)')
          .style('fill', 'var(--color-text)')
          .text(d.closed.toFixed(3));
        g.append('text')
          .attr('x', (x(d.model) ?? 0) + (xSub('chain') ?? 0) + xSub.bandwidth() / 2)
          .attr('y', y(d.chain) - 4)
          .attr('text-anchor', 'middle')
          .style('font-size', '10px')
          .style('font-family', 'var(--font-mono)')
          .style('fill', 'var(--color-text)')
          .text(d.chain.toFixed(3));
      }
      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(y).ticks(5, '.1f')).selectAll('text').style('fill', 'var(--color-text-secondary)');

      // Legend
      const lg = g.append('g').attr('transform', `translate(${w - 110},${0})`);
      lg.append('rect').attr('width', 12).attr('height', 12).attr('y', 0).style('fill', 'var(--color-text-secondary)');
      lg.append('text').attr('x', 16).attr('y', 10).style('font-size', '11px').style('fill', 'var(--color-text)').text('closed-form');
      lg.append('rect').attr('width', 12).attr('height', 12).attr('y', 16).style('fill', 'var(--color-accent)');
      lg.append('text').attr('x', 16).attr('y', 26).style('font-size', '11px').style('fill', 'var(--color-text)').text('chain (Th3)');
    },
    [result, panelW, panelH],
  );

  const traceRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 20, right: 18, bottom: 32, left: 40 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const T = result.Mtrace.length;
      const x = d3.scaleLinear().domain([0, T]).range([0, w]);
      const y = d3.scaleLinear().domain([0.5, 2.5]).range([h, 0]);

      // Shade in-M2 region as horizontal bar
      g.append('rect')
        .attr('x', 0)
        .attr('y', y(2.05))
        .attr('width', w)
        .attr('height', y(1.95) - y(2.05))
        .style('fill', 'var(--color-accent)')
        .style('opacity', 0.08);

      // Trace as steps
      const burn = Math.floor(T * 0.25);
      const downsample = Math.max(1, Math.floor(T / 600));
      const pts: { t: number; M: number }[] = [];
      for (let t = 0; t < T; t += downsample) pts.push({ t, M: result.Mtrace[t] });
      const line = d3
        .line<{ t: number; M: number }>()
        .x((d) => x(d.t))
        .y((d) => y(d.M))
        .curve(d3.curveStepAfter);
      g.append('path').datum(pts).attr('d', line).style('fill', 'none').style('stroke', 'var(--color-text)').style('stroke-width', 1);

      // Burn-in line
      g.append('line')
        .attr('x1', x(burn))
        .attr('x2', x(burn))
        .attr('y1', 0)
        .attr('y2', h)
        .style('stroke', 'var(--color-text-secondary)')
        .style('stroke-dasharray', '3 3');
      g.append('text')
        .attr('x', x(burn) + 4)
        .attr('y', 12)
        .style('font-size', '10px')
        .style('font-family', 'var(--font-mono)')
        .style('fill', 'var(--color-text-secondary)')
        .text('burn-in');

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(6)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(y).tickValues([1, 2]).tickFormat((d) => `M${d}`)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle').style('font-size', '12px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('iteration t');
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
    ariaLabel: string,
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
        aria-label={ariaLabel}
      />
    </div>
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ marginBottom: 12 }}>
        {sliderRow('β★', dBeta, setDBeta, (v) => commit({ betaStar: v }), -2, 2, 0.05, (n) => n.toFixed(2), 'True effect size β-star')}
        {sliderRow('σ_β²', dSig2, setDSig2, (v) => commit({ sigmaB2: v }), 0.5, 16, 0.25, (n) => n.toFixed(2), 'Prior variance σ_β²')}
        {sliderRow('n', dN, setDN, (v) => commit({ n: Math.round(v) }), 20, 200, 5, (n) => n.toFixed(0), 'Sample size n')}
        {sliderRow('chain length T', dT, setDT, (v) => commit({ T: Math.round(v) }), 500, 5000, 250, (n) => n.toFixed(0), 'Chain length T')}
      </div>
      <div
        style={{
          marginBottom: 8,
          fontFamily: 'var(--font-mono)',
          fontSize: 12,
          color: 'var(--color-text-secondary)',
        }}
      >
        Closed-form P(M₂|D) = {result.pM2.toFixed(4)} · chain P̂(M₂|D) = {result.pHat.toFixed(4)} ·
        |error| = {Math.abs(result.pHat - result.pM2).toFixed(4)} ({sigmaCount.toFixed(1)}σ) ·
        dim-acc {(result.acceptDim * 100).toFixed(1)}% · within-acc {(result.acceptWith * 100).toFixed(1)}%
      </div>
      <div
        style={{
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          gap: 24,
          alignItems: 'flex-start',
        }}
      >
        <div style={{ width: isMobile ? '100%' : panelW }}>
          <svg ref={barRef} width={panelW} height={panelH} />
        </div>
        <div style={{ width: isMobile ? '100%' : panelW }}>
          <svg ref={traceRef} width={panelW} height={panelH} />
        </div>
      </div>
    </div>
  );
}
