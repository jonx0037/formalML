import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { modeFinder } from './shared/unsupervised';
import { ari, BLOB3, makeMoons, mulberry32, CLUSTER_COLORS, type Dataset, type Pt } from '../../data/clustering-data';

// =============================================================================
// GMMSoftAssignments — §10.5. 2×2 grid: top row 3-blob, bottom row moons.
// Left column GMM soft responsibilities (RGB blend), right column mean-shift basins.
// =============================================================================

const HEIGHT = 280;
const SM_BP = 640;

type CovType = 'spherical' | 'diagonal' | 'full' | 'tied';

interface GmmFit {
  pi: number[];
  mu: Pt[];
  cov: number[][][]; // K × 2 × 2 (full); diagonal/spherical/tied represented as full
  responsibilities: number[][]; // n × K
  labels: number[]; // argmax of responsibilities
}

function mvnPdf(x: Pt, mu: Pt, cov: number[][]): number {
  // 2x2 only.
  const dx0 = x[0] - mu[0], dx1 = x[1] - mu[1];
  const a = cov[0][0], b = cov[0][1], c = cov[1][1];
  const det = a * c - b * b;
  if (det <= 0) return 1e-300;
  const inv00 = c / det, inv01 = -b / det, inv11 = a / det;
  const q = dx0 * dx0 * inv00 + 2 * dx0 * dx1 * inv01 + dx1 * dx1 * inv11;
  return Math.exp(-0.5 * q) / (2 * Math.PI * Math.sqrt(det));
}

function gmmFit(X: Pt[], K: number, covType: CovType, seed: number, maxIter = 60): GmmFit {
  const n = X.length;
  const rng = mulberry32(seed);
  // Initialize means via k-means++-like seeding.
  const mu: Pt[] = [[X[Math.floor(rng() * n)][0], X[Math.floor(rng() * n)][1]]];
  const dSq = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const dx = X[j][0] - mu[0][0], dy = X[j][1] - mu[0][1];
    dSq[j] = dx * dx + dy * dy;
  }
  while (mu.length < K) {
    let total = 0; for (let j = 0; j < n; j++) total += dSq[j];
    let target = rng() * total; let pick = n - 1;
    for (let j = 0; j < n; j++) { target -= dSq[j]; if (target <= 0) { pick = j; break; } }
    mu.push([X[pick][0], X[pick][1]]);
    for (let j = 0; j < n; j++) {
      const dx = X[j][0] - mu[mu.length - 1][0], dy = X[j][1] - mu[mu.length - 1][1];
      const s = dx * dx + dy * dy;
      if (s < dSq[j]) dSq[j] = s;
    }
  }
  const cov: number[][][] = mu.map(() => [[1, 0], [0, 1]]);
  const pi = new Array<number>(K).fill(1 / K);
  const resp: number[][] = Array.from({ length: n }, () => new Array<number>(K).fill(0));

  for (let it = 0; it < maxIter; it++) {
    // E-step.
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        const p = pi[k] * mvnPdf(X[i], mu[k], cov[k]);
        resp[i][k] = p;
        sum += p;
      }
      if (sum > 0) for (let k = 0; k < K; k++) resp[i][k] /= sum;
      else for (let k = 0; k < K; k++) resp[i][k] = 1 / K;
    }
    // M-step.
    const Nk = new Array<number>(K).fill(0);
    for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) Nk[k] += resp[i][k];
    for (let k = 0; k < K; k++) {
      pi[k] = Math.max(Nk[k] / n, 1e-6);
      mu[k] = [0, 0];
      if (Nk[k] > 0) {
        for (let i = 0; i < n; i++) {
          const r = resp[i][k];
          mu[k][0] += r * X[i][0];
          mu[k][1] += r * X[i][1];
        }
        mu[k][0] /= Nk[k]; mu[k][1] /= Nk[k];
      }
    }
    // Covariances.
    for (let k = 0; k < K; k++) {
      if (Nk[k] < 1e-6) { cov[k] = [[1, 0], [0, 1]]; continue; }
      let s00 = 0, s01 = 0, s11 = 0;
      for (let i = 0; i < n; i++) {
        const r = resp[i][k];
        const dx = X[i][0] - mu[k][0], dy = X[i][1] - mu[k][1];
        s00 += r * dx * dx; s01 += r * dx * dy; s11 += r * dy * dy;
      }
      let c00 = s00 / Nk[k], c01 = s01 / Nk[k], c11 = s11 / Nk[k];
      if (covType === 'spherical') {
        const s = (c00 + c11) / 2; c00 = s; c11 = s; c01 = 0;
      } else if (covType === 'diagonal') {
        c01 = 0;
      }
      // Regularize.
      c00 = Math.max(c00, 1e-4);
      c11 = Math.max(c11, 1e-4);
      cov[k] = [[c00, c01], [c01, c11]];
    }
    if (covType === 'tied') {
      // Pool across components.
      const Ntot = Nk.reduce((s, v) => s + v, 0);
      let p00 = 0, p01 = 0, p11 = 0;
      for (let k = 0; k < K; k++) {
        p00 += (Nk[k] / Ntot) * cov[k][0][0];
        p01 += (Nk[k] / Ntot) * cov[k][0][1];
        p11 += (Nk[k] / Ntot) * cov[k][1][1];
      }
      for (let k = 0; k < K; k++) cov[k] = [[p00, p01], [p01, p11]];
    }
  }
  const labels = resp.map((r) => {
    let best = 0, bestV = -Infinity;
    for (let k = 0; k < r.length; k++) if (r[k] > bestV) { bestV = r[k]; best = k; }
    return best;
  });
  return { pi, mu, cov, responsibilities: resp, labels };
}

function rgbBlend(resp: number[]): string {
  // K=2: red → cluster 0, blue → cluster 1
  // K=3: R/G/B
  const K = resp.length;
  let r = 0, g = 0, b = 0;
  if (K === 2) {
    r = Math.round(resp[0] * 220 + 30);
    g = 30;
    b = Math.round(resp[1] * 220 + 30);
  } else {
    r = Math.round(resp[0] * 220 + 30);
    g = Math.round((resp[1] || 0) * 220 + 30);
    b = Math.round((resp[2] || 0) * 220 + 30);
  }
  return `rgb(${r},${g},${b})`;
}

function drawEllipse(g: d3.Selection<SVGGElement, unknown, null, undefined>, mu: Pt, cov: number[][], scaleX: (x: number) => number, scaleY: (y: number) => number, color: string): void {
  // Eigendecomp of 2x2 cov.
  const a = cov[0][0], b = cov[0][1], c = cov[1][1];
  const tr = a + c;
  const det = a * c - b * b;
  const disc = Math.sqrt(Math.max(0, (tr / 2) ** 2 - det));
  const l1 = tr / 2 + disc;
  const l2 = tr / 2 - disc;
  // Eigenvector for l1.
  let vx = 1, vy = 0;
  if (Math.abs(b) > 1e-9) { vx = l1 - c; vy = b; }
  else if (a < c) { vx = 0; vy = 1; }
  const theta = Math.atan2(vy, vx);
  const rx = Math.sqrt(Math.max(0, l1)) * Math.abs(scaleX(1) - scaleX(0));
  const ry = Math.sqrt(Math.max(0, l2)) * Math.abs(scaleY(0) - scaleY(1));
  g.append('ellipse')
    .attr('cx', scaleX(mu[0])).attr('cy', scaleY(mu[1]))
    .attr('rx', rx).attr('ry', ry)
    .attr('transform', `rotate(${(theta * 180) / Math.PI}, ${scaleX(mu[0])}, ${scaleY(mu[1])})`)
    .style('fill', 'none').style('stroke', color).style('stroke-width', 1.5);
}

export default function GMMSoftAssignments() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [K, setK] = useState(3);
  const [h, setH] = useState(0.65);
  const [committedH, setCommittedH] = useState(0.65);
  const [covType, setCovType] = useState<CovType>('full');
  const [showEllipses, setShowEllipses] = useState(true);

  const blob3 = useMemo(() => BLOB3(), []);
  const moons = useMemo(() => makeMoons(200, 0.1, 42), []);

  const fitBlob3 = useMemo(() => gmmFit(blob3.X, K, covType, 17), [blob3, K, covType]);
  const fitMoons = useMemo(() => gmmFit(moons.X, Math.min(K, 2), covType, 17), [moons, K, covType]);
  const msBlob3 = useMemo(() => modeFinder(blob3.X.map((p) => [p[0], p[1]]), committedH), [blob3, committedH]);
  const msMoons = useMemo(() => modeFinder(moons.X.map((p) => [p[0], p[1]]), committedH * 0.6), [moons, committedH]);

  const ariBlob3Gmm = useMemo(() => ari(blob3.y, fitBlob3.labels), [blob3.y, fitBlob3.labels]);
  const ariMoonsGmm = useMemo(() => ari(moons.y, fitMoons.labels), [moons.y, fitMoons.labels]);
  const ariBlob3Ms = useMemo(() => ari(blob3.y, msBlob3.labels), [blob3.y, msBlob3.labels]);
  const ariMoonsMs = useMemo(() => ari(moons.y, msMoons.labels), [moons.y, msMoons.labels]);

  const w = containerWidth;
  const isMobile = w > 0 && w < SM_BP;
  const panelW = (isMobile ? w : w / 2) - 6;

  const drawSoft = (dataset: Dataset, fit: GmmFit, ariV: number, title: string) =>
    useD3<SVGSVGElement>((svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const m = { top: 30, right: 12, bottom: 28, left: 32 };
      const innerW = panelW - m.left - m.right;
      const innerH = HEIGHT - m.top - m.bottom;
      const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

      svg.append('text').attr('x', panelW / 2).attr('y', 16).attr('text-anchor', 'middle')
        .style('font', '600 11px var(--font-sans)').style('fill', 'var(--color-text)')
        .text(`${title} (ARI=${ariV.toFixed(3)})`);

      const x = d3.scaleLinear().domain(dataset.xDomain).range([0, innerW]);
      const y = d3.scaleLinear().domain(dataset.yDomain).range([innerH, 0]);
      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(4)).style('color', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(y).ticks(4)).style('color', 'var(--color-text-secondary)');

      g.selectAll('.pt').data(dataset.X).enter().append('circle')
        .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1]))
        .attr('r', 2.5)
        .style('fill', (_, i) => rgbBlend(fit.responsibilities[i]));

      if (showEllipses) {
        for (let k = 0; k < fit.mu.length; k++) drawEllipse(g, fit.mu[k], fit.cov[k], x, y, 'var(--color-text)');
        const cross = d3.symbol().type(d3.symbolCross).size(80);
        g.selectAll('.center').data(fit.mu).enter().append('path')
          .attr('d', cross)
          .attr('transform', (d) => `translate(${x(d[0])},${y(d[1])}) rotate(45)`)
          .style('fill', 'var(--color-text)').style('stroke', 'white').style('stroke-width', 1);
      }
    }, [dataset, fit, ariV, panelW, showEllipses]);

  const drawHard = (dataset: Dataset, ms: { labels: number[]; modes: number[][] }, ariV: number, title: string) =>
    useD3<SVGSVGElement>((svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const m = { top: 30, right: 12, bottom: 28, left: 32 };
      const innerW = panelW - m.left - m.right;
      const innerH = HEIGHT - m.top - m.bottom;
      const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

      svg.append('text').attr('x', panelW / 2).attr('y', 16).attr('text-anchor', 'middle')
        .style('font', '600 11px var(--font-sans)').style('fill', 'var(--color-text)')
        .text(`${title} (ARI=${ariV.toFixed(3)})`);

      const x = d3.scaleLinear().domain(dataset.xDomain).range([0, innerW]);
      const y = d3.scaleLinear().domain(dataset.yDomain).range([innerH, 0]);
      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(4)).style('color', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(y).ticks(4)).style('color', 'var(--color-text-secondary)');

      g.selectAll('.pt').data(dataset.X).enter().append('circle')
        .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1]))
        .attr('r', 2.5)
        .style('fill', (_, i) => CLUSTER_COLORS[ms.labels[i] % CLUSTER_COLORS.length]);

      const cross = d3.symbol().type(d3.symbolCross).size(100);
      g.selectAll('.mode').data(ms.modes).enter().append('path')
        .attr('d', cross)
        .attr('transform', (d) => `translate(${x(d[0])},${y(d[1])}) rotate(45)`)
        .style('fill', 'var(--color-text)').style('stroke', 'white').style('stroke-width', 1);
    }, [dataset, ms, ariV, panelW]);

  const r1 = drawSoft(blob3, fitBlob3, ariBlob3Gmm, `GMM K=${K} ${covType} — 3-blob`);
  const r2 = drawHard(blob3, msBlob3, ariBlob3Ms, `mean-shift — 3-blob`);
  const r3 = drawSoft(moons, fitMoons, ariMoonsGmm, `GMM K=2 ${covType} — moons`);
  const r4 = drawHard(moons, msMoons, ariMoonsMs, `mean-shift — moons`);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap', marginBottom: 12 }}>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 140 }}>
          K (GMM, 3-blob): <strong style={{ color: 'var(--color-text)' }}>{K}</strong>
          <input type="range" min={2} max={5} step={1} value={K} onChange={(e) => setK(+e.target.value)} aria-label="K for GMM" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 180 }}>
          h (mean-shift, 3-blob; moons uses 0.6h): <strong style={{ color: 'var(--color-text)' }}>{h.toFixed(2)}</strong> {h !== committedH && <em>(release)</em>}
          <input type="range" min={0.20} max={1.2} step={0.01}
            value={h}
            onChange={(e) => setH(+e.target.value)}
            onMouseUp={() => setCommittedH(h)}
            onTouchEnd={() => setCommittedH(h)}
            onKeyUp={() => setCommittedH(h)}
            aria-label="bandwidth h" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)' }}>
          covariance:
          <select value={covType} onChange={(e) => setCovType(e.target.value as CovType)} aria-label="GMM covariance type">
            <option value="full">full</option>
            <option value="diagonal">diagonal</option>
            <option value="spherical">spherical</option>
            <option value="tied">tied</option>
          </select>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--color-text-secondary)', marginTop: 12 }}>
          <input type="checkbox" checked={showEllipses} onChange={(e) => setShowEllipses(e.target.checked)} />
          show ellipses
        </label>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(2, 1fr)`, gap: 8 }}>
        <svg ref={r1} width={panelW} height={HEIGHT} style={{ background: 'var(--color-surface)', borderRadius: 8 }} />
        <svg ref={r2} width={panelW} height={HEIGHT} style={{ background: 'var(--color-surface)', borderRadius: 8 }} />
        <svg ref={r3} width={panelW} height={HEIGHT} style={{ background: 'var(--color-surface)', borderRadius: 8 }} />
        <svg ref={r4} width={panelW} height={HEIGHT} style={{ background: 'var(--color-surface)', borderRadius: 8 }} />
      </div>
    </div>
  );
}
