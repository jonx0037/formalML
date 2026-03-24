import { useState, useMemo, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

const MARGIN = { top: 30, right: 16, bottom: 40, left: 50 };
const N_MC = 100000;

// Seeded LCG PRNG (same as ConvergenceModesDemo)
function lcg(index: number, seed: number): number {
  let s = (seed * 2147483647 + index * 16807 + 12345) & 0x7fffffff;
  s = (s * 16807 + 12345) & 0x7fffffff;
  s = (s * 16807 + 12345) & 0x7fffffff;
  return (s & 0x7fffffff) / 0x7fffffff;
}

function seededNormal(i: number, seed: number): number {
  const u1 = Math.max(lcg(i * 2, seed), 1e-10);
  const u2 = lcg(i * 2 + 1, seed);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ─── Distribution definitions ───

type DistKey = 'gaussian' | 'rademacher' | 'uniform' | 'exponential' | 't3' | 'pareto';

interface DistDef {
  label: string;
  /** Analytical log-MGF if available, null otherwise */
  logMGF: ((lambda: number) => number) | null;
  /** Generate a sample from this distribution */
  sample: (i: number, seed: number) => number;
  classification: 'sub-gaussian' | 'sub-exponential' | 'heavy-tailed';
  sigmaParam?: number; // sub-Gaussian parameter
}

const DIST_DEFS: Record<DistKey, DistDef> = {
  gaussian: {
    label: 'N(0,1)',
    logMGF: (lam) => lam * lam / 2,
    sample: (i, seed) => seededNormal(i, seed),
    classification: 'sub-gaussian',
    sigmaParam: 1,
  },
  rademacher: {
    label: 'Rademacher (±1)',
    logMGF: (lam) => Math.log(Math.cosh(lam)),
    sample: (i, seed) => (lcg(i, seed) < 0.5 ? -1 : 1),
    classification: 'sub-gaussian',
    sigmaParam: 1,
  },
  uniform: {
    label: 'Uniform[-1,1]',
    logMGF: (lam) => (Math.abs(lam) < 1e-8 ? 0 : Math.log(Math.sinh(lam) / lam)),
    sample: (i, seed) => lcg(i, seed) * 2 - 1,
    classification: 'sub-gaussian',
    sigmaParam: 1,
  },
  exponential: {
    label: 'Exp(1) − 1',
    logMGF: (lam) => (lam < 0.99 ? -lam - Math.log(1 - lam) : 20), // log(e^{-λ}/(1-λ))
    sample: (i, seed) => -Math.log(Math.max(lcg(i, seed), 1e-10)) - 1,
    classification: 'sub-exponential',
  },
  t3: {
    label: 't(3)',
    logMGF: null, // MGF does not exist for all λ
    sample: (i, seed) => {
      const z = seededNormal(i, seed);
      // Chi-squared(3) via sum of 3 squared normals
      const chi2 = seededNormal(i + N_MC, seed) ** 2 + seededNormal(i + 2 * N_MC, seed) ** 2 + seededNormal(i + 3 * N_MC, seed) ** 2;
      return z / Math.sqrt(chi2 / 3);
    },
    classification: 'sub-exponential', // technically not even sub-exponential, but tails decay polynomially
  },
  pareto: {
    label: 'Pareto(2) − 2',
    logMGF: null, // MGF does not exist for λ > 0
    sample: (i, seed) => {
      const u = Math.max(lcg(i, seed), 1e-10);
      return 1 / Math.sqrt(u) - 2; // Pareto(α=2, x_m=1) shifted by -E[X] = -2
    },
    classification: 'heavy-tailed',
  },
};

const DIST_KEYS: DistKey[] = ['gaussian', 'rademacher', 'uniform', 'exponential', 't3', 'pareto'];

const BADGE_STYLES: Record<string, { bg: string; border: string; text: string; label: string }> = {
  'sub-gaussian': { bg: '#ECFDF5', border: '#059669', text: '#059669', label: 'Sub-Gaussian ✓' },
  'sub-exponential': { bg: '#FEF9C3', border: '#D97706', text: '#D97706', label: 'Sub-Exponential (not sub-G)' },
  'heavy-tailed': { bg: '#FEF2F2', border: '#DC2626', text: '#DC2626', label: 'Heavy-tailed' },
};

export default function SubGaussianClassifier() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const leftSvgRef = useRef<SVGSVGElement>(null);
  const rightSvgRef = useRef<SVGSVGElement>(null);

  const [dist, setDist] = useState<DistKey>('gaussian');

  const isDesktop = (containerWidth || 0) > 640;
  const panelW = isDesktop ? Math.floor((containerWidth - 12) / 2) : containerWidth;
  const panelH = Math.min(280, Math.max(200, panelW * 0.65));

  const cfg = DIST_DEFS[dist];
  const badge = BADGE_STYLES[cfg.classification];

  // ─── Monte Carlo samples (generated once per distribution) ───

  const samples = useMemo(() => {
    const arr = new Float64Array(N_MC);
    for (let i = 0; i < N_MC; i++) {
      arr[i] = cfg.sample(i, 42);
    }
    return arr;
  }, [cfg]);

  // ─── Sub-Gaussian parameter estimation ───

  const sigmaEst = useMemo(() => {
    if (cfg.sigmaParam !== undefined) return cfg.sigmaParam;
    // Binary search for tightest sigma such that log MGF <= sigma^2 * lambda^2 / 2 for lambda in [-3, 3]
    const lambdas = Array.from({ length: 61 }, (_, i) => -3 + i * 0.1);
    let lo = 0.1, hi = 10;
    for (let iter = 0; iter < 30; iter++) {
      const mid = (lo + hi) / 2;
      let ok = true;
      for (const lam of lambdas) {
        // Empirical log-MGF
        let logMGFVal: number;
        if (cfg.logMGF) {
          logMGFVal = cfg.logMGF(lam);
        } else {
          // Monte Carlo estimate
          let maxExp = -Infinity;
          for (let i = 0; i < Math.min(N_MC, 10000); i++) {
            maxExp = Math.max(maxExp, lam * samples[i]);
          }
          let sum = 0;
          for (let i = 0; i < Math.min(N_MC, 10000); i++) {
            sum += Math.exp(lam * samples[i] - maxExp);
          }
          logMGFVal = Math.log(sum / Math.min(N_MC, 10000)) + maxExp;
        }
        if (logMGFVal > mid * mid * lam * lam / 2 + 0.01) {
          ok = false;
          break;
        }
      }
      if (ok) hi = mid; else lo = mid;
    }
    return hi;
  }, [cfg, samples]);

  // ─── Left panel: log MGF vs λ ───

  useEffect(() => {
    const svg = leftSvgRef.current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    const nPts = 120;
    const lambdaRange = 3;
    const lambdas = Array.from({ length: nPts }, (_, i) => -lambdaRange + 2 * lambdaRange * i / (nPts - 1));

    // Compute log-MGF values
    const logMGFVals = lambdas.map(lam => {
      if (cfg.logMGF) return cfg.logMGF(lam);
      // Empirical estimate
      let maxExp = -Infinity;
      for (let i = 0; i < Math.min(N_MC, 20000); i++) {
        maxExp = Math.max(maxExp, lam * samples[i]);
      }
      let sum = 0;
      for (let i = 0; i < Math.min(N_MC, 20000); i++) {
        sum += Math.exp(lam * samples[i] - maxExp);
      }
      return Math.log(sum / Math.min(N_MC, 20000)) + maxExp;
    });

    // Sub-Gaussian envelope: σ²λ²/2
    const envelope = lambdas.map(lam => sigmaEst * sigmaEst * lam * lam / 2);

    const yMax = Math.min(Math.max(d3.max(logMGFVals) || 5, d3.max(envelope) || 5) * 1.1, 20);

    const xScale = d3.scaleLinear().domain([-lambdaRange, lambdaRange]).range([MARGIN.left, panelW - MARGIN.right]);
    const yScale = d3.scaleLinear().domain([Math.min(d3.min(logMGFVals) || 0, 0), yMax]).range([panelH - MARGIN.bottom, MARGIN.top]);

    // Axes
    sel.append('g').attr('transform', `translate(0,${panelH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(5).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    sel.append('g').attr('transform', `translate(${MARGIN.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    sel.append('text').attr('x', panelW / 2).attr('y', panelH - 4)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('λ');
    sel.append('text').attr('transform', 'rotate(-90)').attr('x', -(panelH / 2)).attr('y', 14)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('log M_X(λ)');

    // Envelope (parabola)
    const envLine = d3.line<number>()
      .x((_, i) => xScale(lambdas[i]))
      .y(d => yScale(Math.min(d, yMax)));
    sel.append('path').datum(envelope).attr('d', envLine)
      .style('fill', 'none').style('stroke', '#059669').style('stroke-width', 1.5).style('stroke-dasharray', '6 3').style('opacity', 0.7);

    // Log-MGF curve
    const mgfLine = d3.line<number>()
      .defined(d => isFinite(d) && d <= yMax)
      .x((_, i) => xScale(lambdas[i]))
      .y(d => yScale(d));
    sel.append('path').datum(logMGFVals).attr('d', mgfLine)
      .style('fill', 'none').style('stroke', '#2563EB').style('stroke-width', 2.5);

    // Title
    sel.append('text').attr('x', MARGIN.left + 4).attr('y', MARGIN.top - 8)
      .style('font-size', '11px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', 'var(--color-text)')
      .text('log-MGF vs λ');

    // Legend
    const ly = MARGIN.top + 10;
    const lx = panelW - MARGIN.right - 90;
    [{ label: 'log M_X(λ)', color: '#2563EB', dash: false },
     { label: 'σ²λ²/2', color: '#059669', dash: true }].forEach((item, i) => {
      sel.append('line').attr('x1', lx).attr('x2', lx + 16).attr('y1', ly + i * 14).attr('y2', ly + i * 14)
        .style('stroke', item.color).style('stroke-width', 2).style('stroke-dasharray', item.dash ? '6 3' : 'none');
      sel.append('text').attr('x', lx + 20).attr('y', ly + i * 14 + 3)
        .style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
        .text(item.label);
    });
  }, [cfg, samples, sigmaEst, panelW, panelH]);

  // ─── Right panel: log P(|X| >= t) vs t ───

  useEffect(() => {
    const svg = rightSvgRef.current;
    if (!svg || panelW <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    // Sort absolute values for empirical tail
    const absSorted = Array.from(samples).map(Math.abs).sort((a, b) => a - b);
    const nPts = 100;
    const tMax = absSorted[Math.floor(N_MC * 0.999)] || 5;
    const ts = Array.from({ length: nPts }, (_, i) => 0.1 + (tMax - 0.1) * i / (nPts - 1));

    // Empirical tail: P(|X| >= t)
    const empiricalTail = ts.map(t => {
      // Binary search for index
      let lo = 0, hi = absSorted.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (absSorted[mid] < t) lo = mid + 1; else hi = mid;
      }
      return Math.max((absSorted.length - lo) / absSorted.length, 1e-6);
    });

    // Sub-Gaussian envelope: 2 exp(-t²/(2σ²))
    const subGEnv = ts.map(t => 2 * Math.exp(-t * t / (2 * sigmaEst * sigmaEst)));

    // Sub-exponential envelope: 2 exp(-t/b)  where b ≈ sigmaEst
    const subExpEnv = ts.map(t => 2 * Math.exp(-t / sigmaEst));

    const xScale = d3.scaleLinear().domain([0, tMax]).range([MARGIN.left, panelW - MARGIN.right]);
    const yScale = d3.scaleLog().domain([1e-5, 2]).range([panelH - MARGIN.bottom, MARGIN.top]).clamp(true);

    // Axes
    sel.append('g').attr('transform', `translate(0,${panelH - MARGIN.bottom})`)
      .call(d3.axisBottom(xScale).ticks(5).tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    sel.append('g').attr('transform', `translate(${MARGIN.left},0)`)
      .call(d3.axisLeft(yScale).ticks(4, '.0e').tickSize(3))
      .call(g => { g.selectAll('text').style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)'); g.select('.domain').style('stroke', 'var(--color-border)'); });

    sel.append('text').attr('x', panelW / 2).attr('y', panelH - 4)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('t');
    sel.append('text').attr('transform', 'rotate(-90)').attr('x', -(panelH / 2)).attr('y', 14)
      .attr('text-anchor', 'middle').style('font-size', '10px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
      .text('P(|X| ≥ t)');

    // Envelopes
    const lineGen = (vals: number[]) => d3.line<number>()
      .defined(d => d > 1e-6 && isFinite(d))
      .x((_, i) => xScale(ts[i]))
      .y(d => yScale(d));

    sel.append('path').datum(subGEnv).attr('d', lineGen(subGEnv))
      .style('fill', 'none').style('stroke', '#059669').style('stroke-width', 1.5).style('stroke-dasharray', '6 3').style('opacity', 0.7);
    sel.append('path').datum(subExpEnv).attr('d', lineGen(subExpEnv))
      .style('fill', 'none').style('stroke', '#D97706').style('stroke-width', 1.5).style('stroke-dasharray', '4 2').style('opacity', 0.6);

    // Empirical tail
    sel.append('path').datum(empiricalTail).attr('d', lineGen(empiricalTail))
      .style('fill', 'none').style('stroke', '#2563EB').style('stroke-width', 2.5);

    // Title
    sel.append('text').attr('x', MARGIN.left + 4).attr('y', MARGIN.top - 8)
      .style('font-size', '11px').style('font-family', 'var(--font-sans)').style('font-weight', '600').style('fill', 'var(--color-text)')
      .text('Tail Decay');

    // Legend
    const ly = MARGIN.top + 10;
    const lx = panelW - MARGIN.right - 100;
    [{ label: 'Empirical', color: '#2563EB', dash: false },
     { label: 'Sub-G env.', color: '#059669', dash: true },
     { label: 'Sub-Exp env.', color: '#D97706', dash: true }].forEach((item, i) => {
      sel.append('line').attr('x1', lx).attr('x2', lx + 16).attr('y1', ly + i * 14).attr('y2', ly + i * 14)
        .style('stroke', item.color).style('stroke-width', 2).style('stroke-dasharray', item.dash ? '6 3' : 'none');
      sel.append('text').attr('x', lx + 20).attr('y', ly + i * 14 + 3)
        .style('font-size', '9px').style('font-family', 'var(--font-mono)').style('fill', 'var(--color-text-secondary)')
        .text(item.label);
    });
  }, [samples, sigmaEst, panelW, panelH]);

  // ─── Render ───

  return (
    <div
      ref={containerRef}
      style={{
        border: '1px solid var(--color-border)',
        borderRadius: '8px',
        padding: '16px',
        background: 'var(--color-surface)',
        marginTop: '1.5rem',
        marginBottom: '1.5rem',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ fontFamily: 'var(--font-sans)', fontSize: '13px', fontWeight: 600, color: 'var(--color-text)' }}>
          Sub-Gaussian Classifier
        </div>
        {/* Classification badge */}
        <div style={{
          padding: '4px 12px', borderRadius: '12px', fontSize: '11px', fontWeight: 600,
          fontFamily: 'var(--font-mono)',
          background: badge.bg, border: `1px solid ${badge.border}`, color: badge.text,
        }}>
          {badge.label}
        </div>
      </div>

      {/* Two-panel layout */}
      <div style={{ display: 'grid', gridTemplateColumns: isDesktop ? '1fr 1fr' : '1fr', gap: '8px' }}>
        <svg ref={leftSvgRef} width={panelW} height={panelH} style={{
          border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)',
        }} />
        <svg ref={rightSvgRef} width={panelW} height={panelH} style={{
          border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)',
        }} />
      </div>

      {/* Controls */}
      <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' }}>
        {DIST_KEYS.map(k => (
          <button
            key={k}
            onClick={() => setDist(k)}
            style={{
              padding: '4px 10px', fontSize: '11px', fontFamily: 'var(--font-mono)',
              border: `1px solid ${dist === k ? BADGE_STYLES[DIST_DEFS[k].classification].border : 'var(--color-border)'}`,
              borderRadius: '4px', cursor: 'pointer',
              background: dist === k ? BADGE_STYLES[DIST_DEFS[k].classification].bg : 'transparent',
              color: dist === k ? BADGE_STYLES[DIST_DEFS[k].classification].text : 'var(--color-text-secondary)',
              fontWeight: dist === k ? 600 : 400,
            }}
          >
            {DIST_DEFS[k].label}
          </button>
        ))}
      </div>
    </div>
  );
}
