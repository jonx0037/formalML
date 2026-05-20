// =============================================================================
// EnumerationWallExplorer.tsx
//
// §2 of reversible-jump-mcmc. Two-panel illustration of "why we sample":
// (A) 2^p log curve with feasibility frontiers; (B) toy p=5 posterior over
// all 32 subset-models computed via closed-form g-prior marginal likelihood.
//
// Both panels are cheap closed-form — slider redraws are instant.
// Static fallback: /images/topics/reversible-jump-mcmc/fig02_model_selection_target.png
// =============================================================================
import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';

// Lightweight deterministic RNG for the toy regression simulation.
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

function olsR2(X: number[][], y: number[]): number {
  // X is n×k, y is n-vector; assume centered columns. Returns R² = 1 - RSS/TSS.
  const n = y.length;
  const k = X[0]?.length ?? 0;
  if (k === 0) return 0;
  // Normal equations: β = (XᵀX)⁻¹ Xᵀy via Cholesky-free Gaussian elimination.
  const XtX: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  const Xty = new Array(k).fill(0);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < k; a++) {
      Xty[a] += X[i][a] * y[i];
      for (let b = 0; b < k; b++) XtX[a][b] += X[i][a] * X[i][b];
    }
  }
  // Solve XtX β = Xty by partial pivoting.
  const m = k;
  const A = XtX.map((row, i) => [...row, Xty[i]]);
  for (let i = 0; i < m; i++) {
    let piv = i;
    for (let r = i + 1; r < m; r++) if (Math.abs(A[r][i]) > Math.abs(A[piv][i])) piv = r;
    if (piv !== i) [A[i], A[piv]] = [A[piv], A[i]];
    if (Math.abs(A[i][i]) < 1e-12) return 0; // singular
    for (let r = i + 1; r < m; r++) {
      const f = A[r][i] / A[i][i];
      for (let c = i; c <= m; c++) A[r][c] -= f * A[i][c];
    }
  }
  const beta = new Array(m).fill(0);
  for (let i = m - 1; i >= 0; i--) {
    let s = A[i][m];
    for (let j = i + 1; j < m; j++) s -= A[i][j] * beta[j];
    beta[i] = s / A[i][i];
  }
  // RSS
  let rss = 0;
  let tss = 0;
  const yMean = y.reduce((a, b) => a + b, 0) / n;
  for (let i = 0; i < n; i++) {
    let yhat = 0;
    for (let a = 0; a < k; a++) yhat += X[i][a] * beta[a];
    rss += (y[i] - yhat) ** 2;
    tss += (y[i] - yMean) ** 2;
  }
  return tss > 1e-12 ? 1 - rss / tss : 0;
}

function logMarglikGPrior(R2: number, kCard: number, n: number, g: number): number {
  // Liang et al. 2008 eq. (5) closed-form g-prior log marginal likelihood.
  // Constants dropped (cancel in ratios).
  return ((n - kCard - 1) / 2) * Math.log(1 + g) - ((n - 1) / 2) * Math.log(1 + g * (1 - R2));
}
function logChoose(n: number, k: number): number {
  // log binomial coefficient via lgamma surrogates.
  function lgamma(x: number): number {
    // Stirling-like, good enough for n ≤ 100.
    const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x;
    let tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    let ser = 1.000000000190015;
    for (const c of cof) ser += c / ++y;
    return -tmp + Math.log((2.5066282746310005 * ser) / x);
  }
  return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
}

interface ToyResult {
  bits: string;
  card: number;
  logPost: number;
}

function simulateToyPosterior(p: number, seed: number): ToyResult[] {
  const n = 80;
  const rng = mulberry32(seed);
  // True active covariates at indices 0 and 2 (regardless of p)
  const trueActive = [0, 2].filter((j) => j < p);
  // Generate design matrix (centered) and response.
  const X: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array(p);
    for (let j = 0; j < p; j++) row[j] = randn(rng);
    X.push(row);
  }
  // Center
  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += X[i][j];
    const m = s / n;
    for (let i = 0; i < n; i++) X[i][j] -= m;
  }
  // y = 2*X[0] - 1.5*X[2] + noise
  const beta = new Array(p).fill(0);
  if (trueActive.includes(0)) beta[0] = 2;
  if (trueActive.includes(2)) beta[2] = -1.5;
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let yi = 0;
    for (let j = 0; j < p; j++) yi += X[i][j] * beta[j];
    y[i] = yi + 0.5 * randn(rng);
  }
  // Center y
  const yMean = y.reduce((a, b) => a + b, 0) / n;
  for (let i = 0; i < n; i++) y[i] -= yMean;

  const g = n;
  const results: ToyResult[] = [];
  for (let id = 0; id < 1 << p; id++) {
    const cols: number[] = [];
    let bitsStr = '';
    for (let j = 0; j < p; j++) {
      if ((id >> j) & 1) cols.push(j);
      bitsStr += (id >> j) & 1 ? '1' : '0';
    }
    const card = cols.length;
    let logMarg = 0;
    if (card === 0) {
      logMarg = 0; // baseline; constants cancel
    } else {
      const X_S: number[][] = X.map((row) => cols.map((c) => row[c]));
      const R2 = olsR2(X_S, y);
      logMarg = logMarglikGPrior(R2, card, n, g);
    }
    const logPrior = -logChoose(p, card) - Math.log(p + 1);
    results.push({ bits: bitsStr, card, logPost: logMarg + logPrior });
  }
  // Normalize via logsumexp
  const maxLP = Math.max(...results.map((r) => r.logPost));
  const sumExp = results.reduce((s, r) => s + Math.exp(r.logPost - maxLP), 0);
  const logZ = maxLP + Math.log(sumExp);
  for (const r of results) r.logPost = Math.exp(r.logPost - logZ);
  return results.sort((a, b) => b.logPost - a.logPost);
}

export default function EnumerationWallExplorer() {
  const [p, setP] = useState(5);
  const toy = useMemo(() => simulateToyPosterior(p, 1234), [p]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const W = containerWidth || 800;
  const isMobile = W < 640;
  const panelW = isMobile ? W : (W - 24) / 2;
  const panelH = 300;

  const wallRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 16, right: 18, bottom: 32, left: 52 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const pRange = d3.range(1, 61);
      const x = d3.scaleLinear().domain([1, 60]).range([0, w]);
      const y = d3.scaleLog().domain([1, 1e18]).range([h, 0]);

      // Feasibility frontier horizontal lines
      const frontiers = [
        { y: 1e4, label: 'trivial enumeration' },
        { y: 1e7, label: 'feasible with vectorized OLS' },
        { y: 1e10, label: 'RJ-MCMC territory' },
      ];
      for (const fr of frontiers) {
        g.append('line')
          .attr('x1', 0)
          .attr('x2', w)
          .attr('y1', y(fr.y))
          .attr('y2', y(fr.y))
          .style('stroke', 'var(--color-border)')
          .style('stroke-dasharray', '4 4');
        g.append('text')
          .attr('x', w - 4)
          .attr('y', y(fr.y) - 4)
          .attr('text-anchor', 'end')
          .style('font-size', '10px')
          .style('font-family', 'var(--font-sans)')
          .style('fill', 'var(--color-text-secondary)')
          .text(fr.label);
      }
      const line = d3
        .line<number>()
        .x((d) => x(d))
        .y((d) => y(Math.pow(2, d)))
        .defined((d) => Math.pow(2, d) >= 1 && Math.pow(2, d) <= 1e18);
      g.append('path')
        .datum(pRange)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', 'var(--color-accent)')
        .style('stroke-width', 2);

      // Highlight current p
      g.append('circle')
        .attr('cx', x(p))
        .attr('cy', y(Math.pow(2, p)))
        .attr('r', 5)
        .style('fill', 'var(--color-accent)');
      g.append('text')
        .attr('x', x(p) + 8)
        .attr('y', y(Math.pow(2, p)) - 8)
        .style('font-size', '12px')
        .style('font-family', 'var(--font-mono)')
        .style('fill', 'var(--color-text)')
        .text(`p=${p}: ${(1 << p).toLocaleString()} models`);

      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(x).ticks(6))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(y).ticks(6, '~s'))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 28)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-family', 'var(--font-sans)')
        .style('fill', 'var(--color-text)')
        .text('p (number of covariates)');
      g.append('text')
        .attr('x', -h / 2)
        .attr('y', -38)
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .style('font-size', '12px')
        .style('font-family', 'var(--font-sans)')
        .style('fill', 'var(--color-text)')
        .text('|M| = 2^p');
    },
    [p, panelW, panelH],
  );

  const postRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 16, right: 18, bottom: 32, left: 40 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const x = d3.scaleBand().domain(toy.map((_, i) => i.toString())).range([0, w]).padding(0.15);
      const y = d3.scaleLinear().domain([0, Math.max(0.05, (toy[0]?.logPost ?? 0) * 1.1)]).range([h, 0]);
      g.selectAll('rect')
        .data(toy)
        .join('rect')
        .attr('x', (_, i) => x(i.toString()) ?? 0)
        .attr('y', (d) => y(d.logPost))
        .attr('width', x.bandwidth())
        .attr('height', (d) => h - y(d.logPost))
        .style('fill', (_, i) => (i < 5 ? 'var(--color-accent)' : 'var(--color-border)'))
        .style('opacity', (_, i) => (i < 5 ? 1 : 0.7));
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(x).tickValues([]))
        .selectAll('text')
        .remove();
      g.append('g')
        .call(d3.axisLeft(y).ticks(5, '.2f'))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.append('text')
        .attr('x', w / 2)
        .attr('y', h + 28)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-family', 'var(--font-sans)')
        .style('fill', 'var(--color-text)')
        .text(`models sorted by P(M | D), ${toy.length} total`);
      // Annotation
      g.append('text')
        .attr('x', w - 8)
        .attr('y', 14)
        .attr('text-anchor', 'end')
        .style('font-size', '11px')
        .style('font-family', 'var(--font-mono)')
        .style('fill', 'var(--color-text)')
        .text(`top-5 carry ${(toy.slice(0, 5).reduce((s, r) => s + r.logPost, 0) * 100).toFixed(1)}% of mass`);
    },
    [toy, panelW, panelH],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div
        style={{
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          gap: 24,
          alignItems: 'flex-start',
        }}
      >
        <div style={{ width: isMobile ? '100%' : panelW }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              marginBottom: 6,
              fontFamily: 'var(--font-sans)',
              fontSize: 13,
              color: 'var(--color-text)',
            }}
          >
            <label htmlFor="p2-slider" style={{ minWidth: 70 }}>
              p = <strong>{p}</strong>
            </label>
            <input
              id="p2-slider"
              type="range"
              min={3}
              max={6}
              step={1}
              value={p}
              onChange={(e) => setP(parseInt(e.target.value, 10))}
              style={{ flex: 1 }}
              aria-label="Number of covariates p for toy posterior"
            />
          </div>
          <svg ref={wallRef} width={panelW} height={panelH} />
        </div>
        <div style={{ width: isMobile ? '100%' : panelW }}>
          <div
            style={{
              marginBottom: 6,
              fontFamily: 'var(--font-sans)',
              fontSize: 13,
              color: 'var(--color-text-secondary)',
            }}
          >
            Toy regression at p = {p}, n = 80, σ = 0.5 — enumerated under g-prior + Scott–Berger.
          </div>
          <svg ref={postRef} width={panelW} height={panelH} />
        </div>
      </div>
    </div>
  );
}
