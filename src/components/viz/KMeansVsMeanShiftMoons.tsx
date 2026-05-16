import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { modeFinder } from './shared/unsupervised';

// =============================================================================
// KMeansVsMeanShiftMoons — §1.3 signature opening figure.
// Side-by-side k-means (left) vs mean-shift (right) on make_moons.
// =============================================================================

const HEIGHT = 360;
const SM_BP = 640;

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussian(rng: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  };
}
function makeMoons(n: number, noise: number, seed: number) {
  const rng = mulberry32(seed);
  const g = gaussian(rng);
  const X: [number, number][] = [];
  const y: number[] = [];
  const nU = Math.floor(n / 2);
  for (let i = 0; i < nU; i++) {
    const th = (Math.PI * i) / Math.max(1, nU - 1);
    X.push([Math.cos(th) + noise * g(), Math.sin(th) + noise * g()]);
    y.push(0);
  }
  for (let i = 0; i < n - nU; i++) {
    const th = (Math.PI * i) / Math.max(1, n - nU - 1);
    X.push([1 - Math.cos(th) + noise * g(), -Math.sin(th) + 0.5 + noise * g()]);
    y.push(1);
  }
  return { X, y };
}

function ari(yTrue: number[], yPred: number[]): number {
  const n = yTrue.length;
  const rows = new Map<number, Map<number, number>>();
  const rowSums = new Map<number, number>();
  const colSums = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const r = yTrue[i], c = yPred[i];
    let inner = rows.get(r);
    if (!inner) { inner = new Map(); rows.set(r, inner); }
    inner.set(c, (inner.get(c) || 0) + 1);
    rowSums.set(r, (rowSums.get(r) || 0) + 1);
    colSums.set(c, (colSums.get(c) || 0) + 1);
  }
  const c2 = (k: number) => (k * (k - 1)) / 2;
  let sIJ = 0;
  for (const m of rows.values()) for (const v of m.values()) sIJ += c2(v);
  let sR = 0; for (const v of rowSums.values()) sR += c2(v);
  let sC = 0; for (const v of colSums.values()) sC += c2(v);
  const t = c2(n);
  const exp = (sR * sC) / t;
  const max = (sR + sC) / 2;
  if (max - exp === 0) return 1;
  return (sIJ - exp) / (max - exp);
}

function silhouette(X: [number, number][], labels: number[]): number {
  const n = X.length;
  let total = 0;
  let scored = 0;
  for (let i = 0; i < n; i++) {
    const li = labels[i];
    const byCluster = new Map<number, number[]>();
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const dx = X[i][0] - X[j][0];
      const dy = X[i][1] - X[j][1];
      const d = Math.sqrt(dx * dx + dy * dy);
      let arr = byCluster.get(labels[j]);
      if (!arr) { arr = []; byCluster.set(labels[j], arr); }
      arr.push(d);
    }
    const aArr = byCluster.get(li);
    if (!aArr || aArr.length === 0) continue;
    const a = aArr.reduce((s, x) => s + x, 0) / aArr.length;
    let bMin = Infinity;
    for (const [lab, arr] of byCluster) {
      if (lab === li) continue;
      const mean = arr.reduce((s, x) => s + x, 0) / arr.length;
      if (mean < bMin) bMin = mean;
    }
    if (!Number.isFinite(bMin)) continue;
    total += (bMin - a) / Math.max(a, bMin);
    scored++;
  }
  return scored > 0 ? total / scored : 0;
}

function kmeans(X: [number, number][], K: number, seed: number, maxIter = 100): { labels: number[]; centers: [number, number][] } {
  const rng = mulberry32(seed);
  const n = X.length;
  // k-means++ init.
  const centers: [number, number][] = [X[Math.floor(rng() * n)].slice() as [number, number]];
  const dSq = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const dx = X[j][0] - centers[0][0];
    const dy = X[j][1] - centers[0][1];
    dSq[j] = dx * dx + dy * dy;
  }
  while (centers.length < K) {
    let total = 0;
    for (let j = 0; j < n; j++) total += dSq[j];
    let target = rng() * total;
    let pick = n - 1;
    for (let j = 0; j < n; j++) {
      target -= dSq[j];
      if (target <= 0) { pick = j; break; }
    }
    const c: [number, number] = [X[pick][0], X[pick][1]];
    centers.push(c);
    for (let j = 0; j < n; j++) {
      const dx = X[j][0] - c[0];
      const dy = X[j][1] - c[1];
      const s = dx * dx + dy * dy;
      if (s < dSq[j]) dSq[j] = s;
    }
  }
  const labels = new Array<number>(n).fill(0);
  for (let it = 0; it < maxIter; it++) {
    let changed = false;
    for (let j = 0; j < n; j++) {
      let best = 0;
      let bestD = Infinity;
      for (let m = 0; m < K; m++) {
        const dx = X[j][0] - centers[m][0];
        const dy = X[j][1] - centers[m][1];
        const d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = m; }
      }
      if (labels[j] !== best) { labels[j] = best; changed = true; }
    }
    if (!changed) break;
    for (let m = 0; m < K; m++) centers[m] = [0, 0];
    const counts = new Array<number>(K).fill(0);
    for (let j = 0; j < n; j++) {
      const lab = labels[j];
      counts[lab]++;
      centers[lab][0] += X[j][0];
      centers[lab][1] += X[j][1];
    }
    for (let m = 0; m < K; m++) if (counts[m] > 0) {
      centers[m][0] /= counts[m];
      centers[m][1] /= counts[m];
    }
  }
  return { labels, centers };
}

const COLORS = ['#0F6E56', '#534AB7', '#D97706', '#0EA5E9', '#DC2626', '#7C3AED'];

export default function KMeansVsMeanShiftMoons() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [K, setK] = useState(2);
  const [h, setH] = useState(0.30);
  const [noise, setNoise] = useState(0.1);
  const [showGT, setShowGT] = useState(false);
  const isMobile = containerWidth > 0 && containerWidth < SM_BP;
  const w = containerWidth;

  const data = useMemo(() => makeMoons(200, noise, 42), [noise]);

  const km = useMemo(() => kmeans(data.X, K, 17), [data, K]);
  const ms = useMemo(() => {
    const X2 = data.X.map((row) => [row[0], row[1]] as number[]);
    return modeFinder(X2, h, { dedupTol: 1e-3 });
  }, [data, h]);

  const kmAri = useMemo(() => ari(data.y, km.labels), [data.y, km.labels]);
  const msAri = useMemo(() => ari(data.y, ms.labels), [data.y, ms.labels]);
  const kmSil = useMemo(() => silhouette(data.X, km.labels), [data.X, km.labels]);
  const msSil = useMemo(() => silhouette(data.X, ms.labels), [data.X, ms.labels]);

  const xDom: [number, number] = [-1.5, 2.6];
  const yDom: [number, number] = [-0.8, 1.4];

  const panelW = (isMobile ? w : w / 2) - 8;
  const panelH = HEIGHT;

  const renderLeft = useD3<SVGSVGElement>((svg) => {
    svg.selectAll('*').remove();
    if (panelW <= 0) return;
    const m = { top: 36, right: 12, bottom: 36, left: 36 };
    const innerW = panelW - m.left - m.right;
    const innerH = panelH - m.top - m.bottom;
    const x = d3.scaleLinear().domain(xDom).range([0, innerW]);
    const y = d3.scaleLinear().domain(yDom).range([innerH, 0]);
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    // Title and metrics.
    svg.append('text')
      .attr('x', panelW / 2).attr('y', 18)
      .attr('text-anchor', 'middle')
      .style('font', '600 13px var(--font-sans)')
      .style('fill', 'var(--color-text)')
      .text('k-means');
    svg.append('text')
      .attr('x', panelW / 2).attr('y', 32)
      .attr('text-anchor', 'middle')
      .style('font', '11px var(--font-sans)')
      .style('fill', 'var(--color-text-secondary)')
      .text(`ARI=${kmAri.toFixed(3)}  silhouette=${kmSil.toFixed(3)}`);

    // Axes.
    g.append('g').attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(5)).style('color', 'var(--color-text-secondary)');
    g.append('g').call(d3.axisLeft(y).ticks(5)).style('color', 'var(--color-text-secondary)');

    // Ground truth overlay (faint circles underneath).
    if (showGT) {
      g.selectAll('.gt').data(data.X).enter().append('circle')
        .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1]))
        .attr('r', 5)
        .style('fill', (_, i) => COLORS[data.y[i] % COLORS.length])
        .style('fill-opacity', 0.18);
    }

    // Voronoi boundary (perpendicular bisectors between consecutive centers).
    if (K === 2) {
      const [c0, c1] = km.centers;
      const mid: [number, number] = [(c0[0] + c1[0]) / 2, (c0[1] + c1[1]) / 2];
      const dirx = c1[0] - c0[0], diry = c1[1] - c0[1];
      // Perpendicular vector.
      const px = -diry, py = dirx;
      const norm = Math.sqrt(px * px + py * py);
      const ux = px / norm, uy = py / norm;
      const L = 5; // long enough to span panel
      g.append('line')
        .attr('x1', x(mid[0] - L * ux))
        .attr('y1', y(mid[1] - L * uy))
        .attr('x2', x(mid[0] + L * ux))
        .attr('y2', y(mid[1] + L * uy))
        .style('stroke', 'var(--color-text)')
        .style('stroke-width', 1.5)
        .style('stroke-dasharray', '4 4')
        .style('opacity', 0.5);
    }

    // Data points colored by k-means label.
    g.selectAll('.pt').data(data.X).enter().append('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1]))
      .attr('r', 3)
      .style('fill', (_, i) => COLORS[km.labels[i] % COLORS.length])
      .style('stroke', 'var(--color-bg)')
      .style('stroke-width', 0.5);

    // Centroids as crosses.
    const cross = d3.symbol().type(d3.symbolCross).size(120);
    g.selectAll('.center').data(km.centers).enter().append('path')
      .attr('d', cross)
      .attr('transform', (d) => `translate(${x(d[0])},${y(d[1])}) rotate(45)`)
      .style('fill', 'var(--color-text)')
      .style('stroke', 'var(--color-bg)')
      .style('stroke-width', 1.5);
  }, [data, km, kmAri, kmSil, panelW, panelH, K, showGT]);

  const renderRight = useD3<SVGSVGElement>((svg) => {
    svg.selectAll('*').remove();
    if (panelW <= 0) return;
    const m = { top: 36, right: 12, bottom: 36, left: 36 };
    const innerW = panelW - m.left - m.right;
    const innerH = panelH - m.top - m.bottom;
    const x = d3.scaleLinear().domain(xDom).range([0, innerW]);
    const y = d3.scaleLinear().domain(yDom).range([innerH, 0]);
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    svg.append('text')
      .attr('x', panelW / 2).attr('y', 18)
      .attr('text-anchor', 'middle')
      .style('font', '600 13px var(--font-sans)')
      .style('fill', 'var(--color-text)')
      .text(`mean-shift (h=${h.toFixed(2)}, M=${ms.modes.length})`);
    svg.append('text')
      .attr('x', panelW / 2).attr('y', 32)
      .attr('text-anchor', 'middle')
      .style('font', '11px var(--font-sans)')
      .style('fill', 'var(--color-text-secondary)')
      .text(`ARI=${msAri.toFixed(3)}  silhouette=${msSil.toFixed(3)}`);

    g.append('g').attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(5)).style('color', 'var(--color-text-secondary)');
    g.append('g').call(d3.axisLeft(y).ticks(5)).style('color', 'var(--color-text-secondary)');

    if (showGT) {
      g.selectAll('.gt').data(data.X).enter().append('circle')
        .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1]))
        .attr('r', 5)
        .style('fill', (_, i) => COLORS[data.y[i] % COLORS.length])
        .style('fill-opacity', 0.18);
    }

    g.selectAll('.pt').data(data.X).enter().append('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1]))
      .attr('r', 3)
      .style('fill', (_, i) => COLORS[ms.labels[i] % COLORS.length])
      .style('stroke', 'var(--color-bg)')
      .style('stroke-width', 0.5);

    const cross = d3.symbol().type(d3.symbolCross).size(120);
    g.selectAll('.mode').data(ms.modes).enter().append('path')
      .attr('d', cross)
      .attr('transform', (d) => `translate(${x(d[0])},${y(d[1])}) rotate(45)`)
      .style('fill', 'var(--color-text)')
      .style('stroke', 'var(--color-bg)')
      .style('stroke-width', 1.5);
  }, [data, ms, msAri, msSil, panelW, panelH, h, showGT]);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap', marginBottom: 12 }}>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 140 }}>
          K (k-means): <strong style={{ color: 'var(--color-text)' }}>{K}</strong>
          <input type="range" min={2} max={6} step={1} value={K} onChange={(e) => setK(+e.target.value)} aria-label="K for k-means" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 180 }}>
          h (mean-shift bandwidth): <strong style={{ color: 'var(--color-text)' }}>{h.toFixed(2)}</strong>
          <input type="range" min={0.05} max={1.0} step={0.01} value={h} onChange={(e) => setH(+e.target.value)} aria-label="mean-shift bandwidth h" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 160 }}>
          noise: <strong style={{ color: 'var(--color-text)' }}>{noise.toFixed(2)}</strong>
          <input type="range" min={0} max={0.3} step={0.01} value={noise} onChange={(e) => setNoise(+e.target.value)} aria-label="moons noise level" />
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--color-text-secondary)', marginTop: 12 }}>
          <input type="checkbox" checked={showGT} onChange={(e) => setShowGT(e.target.checked)} />
          ground-truth overlay
        </label>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16 }}>
        <svg ref={renderLeft} width={panelW} height={panelH} style={{ background: 'var(--color-surface)', borderRadius: 8 }} />
        <svg ref={renderRight} width={panelW} height={panelH} style={{ background: 'var(--color-surface)', borderRadius: 8 }} />
      </div>
    </div>
  );
}
