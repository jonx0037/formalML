import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { modeFinder } from './shared/unsupervised';
import { symEigJacobi } from './shared/structural-risk-minimization';
import { ari, makeCircles, CLUSTER_COLORS, type Dataset, type Pt } from '../../data/clustering-data';

// =============================================================================
// SpectralClusteringTeaser — §11.4. Four-panel comparison on make_circles:
//   1) mean-shift fails (fragmented)
//   2) spectral with kNN affinity succeeds
//   3) DBSCAN succeeds
//   4) spectral embedding shows linear separability
// Light pointer per strategic doc §3.3 — proper theory lives in graph-laplacians.
// =============================================================================

const HEIGHT = 280;
const SM_BP = 640;

function kNNAffinity(X: Pt[], k: number): number[][] {
  const n = X.length;
  const W: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    const dists: { j: number; d: number }[] = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dx = X[i][0] - X[j][0], dy = X[i][1] - X[j][1];
      dists.push({ j, d: Math.sqrt(dx * dx + dy * dy) });
    }
    dists.sort((a, b) => a.d - b.d);
    for (let m = 0; m < Math.min(k, dists.length); m++) {
      const { j, d } = dists[m];
      const wij = Math.exp(-(d * d) / (2 * 0.5 * 0.5));
      W[i][j] = wij;
      W[j][i] = Math.max(W[j][i], wij); // symmetrize.
    }
  }
  return W;
}

// Symmetric normalized Laplacian and bottom-K eigenvectors.
function spectralEmbedding(W: number[][], K: number): { embedding: number[][]; labels: number[] } {
  const n = W.length;
  const Dinv = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += W[i][j];
    Dinv[i] = s > 0 ? 1 / Math.sqrt(s) : 0;
  }
  // Flatten symmetric L = I - D^{-1/2} W D^{-1/2} to row-major Float64Array.
  const Lflat = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    Lflat[i * n + i] = 1;
    for (let j = 0; j < n; j++) {
      if (i !== j) Lflat[i * n + j] = -Dinv[i] * W[i][j] * Dinv[j];
    }
  }
  // Re-symmetrize numerically.
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const avg = (Lflat[i * n + j] + Lflat[j * n + i]) * 0.5;
      Lflat[i * n + j] = avg;
      Lflat[j * n + i] = avg;
    }
  }
  const { values, vectors } = symEigJacobi(Lflat, n);
  // Sort eigenvalues ascending. (Per CLAUDE.md: symEigJacobi output is unsorted.)
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => values[a] - values[b]);
  const start = K === 1 ? 0 : 1; // Skip the constant eigenvector for K ≥ 2.
  const cols = order.slice(start, start + K);
  const embedding: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (const c of cols) row.push(vectors[i * n + c]); // i-th component of c-th eigenvector
    // Row-normalize (Ng-Jordan-Weiss).
    let norm = 0;
    for (const v of row) norm += v * v;
    norm = Math.sqrt(norm);
    if (norm > 1e-9) for (let j = 0; j < row.length; j++) row[j] /= norm;
    embedding.push(row);
  }
  // k-means on the row-normalized embedding for the labels.
  const labels = kmeansLite(embedding, K, 17);
  return { embedding, labels };
}

function kmeansLite(X: number[][], K: number, seed: number): number[] {
  const n = X.length;
  let s = seed >>> 0;
  const rng = () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const centers: number[][] = [X[Math.floor(rng() * n)].slice()];
  while (centers.length < K) centers.push(X[Math.floor(rng() * n)].slice());
  const labels = new Array<number>(n).fill(0);
  for (let it = 0; it < 50; it++) {
    let changed = false;
    for (let j = 0; j < n; j++) {
      let best = 0, bestD = Infinity;
      for (let m = 0; m < K; m++) {
        let d = 0;
        for (let k = 0; k < X[j].length; k++) {
          const dv = X[j][k] - centers[m][k];
          d += dv * dv;
        }
        if (d < bestD) { bestD = d; best = m; }
      }
      if (labels[j] !== best) { labels[j] = best; changed = true; }
    }
    if (!changed) break;
    for (let m = 0; m < K; m++) centers[m] = new Array(X[0].length).fill(0);
    const counts = new Array<number>(K).fill(0);
    for (let j = 0; j < n; j++) {
      counts[labels[j]]++;
      for (let k = 0; k < X[j].length; k++) centers[labels[j]][k] += X[j][k];
    }
    for (let m = 0; m < K; m++) if (counts[m] > 0) for (let k = 0; k < centers[m].length; k++) centers[m][k] /= counts[m];
  }
  return labels;
}

// DBSCAN — classical implementation.
function dbscan(X: Pt[], eps: number, minPts: number): number[] {
  const n = X.length;
  const labels = new Array<number>(n).fill(-1);
  const visited = new Array<boolean>(n).fill(false);
  let cluster = 0;
  const epsSq = eps * eps;
  const neighbors = (i: number): number[] => {
    const out: number[] = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dx = X[i][0] - X[j][0], dy = X[i][1] - X[j][1];
      if (dx * dx + dy * dy <= epsSq) out.push(j);
    }
    return out;
  };
  for (let i = 0; i < n; i++) {
    if (visited[i]) continue;
    visited[i] = true;
    const nb = neighbors(i);
    if (nb.length < minPts) continue;
    labels[i] = cluster;
    const queue = nb.slice();
    while (queue.length) {
      const j = queue.shift()!;
      if (!visited[j]) {
        visited[j] = true;
        const nb2 = neighbors(j);
        if (nb2.length >= minPts) for (const m of nb2) if (!queue.includes(m)) queue.push(m);
      }
      if (labels[j] === -1) labels[j] = cluster;
    }
    cluster++;
  }
  return labels;
}

export default function SpectralClusteringTeaser() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [h, setH] = useState(0.20);
  const [committedH, setCommittedH] = useState(0.20);
  const [K, setK] = useState(2);
  const [kNN, setKNN] = useState(10);
  const [eps, setEps] = useState(0.20);
  const [committedEps, setCommittedEps] = useState(0.20);

  const dataset = useMemo(() => makeCircles(200, 0.5, 0.05, 42), []);

  const msLabels = useMemo(() => modeFinder(dataset.X.map((p) => [p[0], p[1]]), committedH).labels, [dataset, committedH]);
  const spectral = useMemo(() => {
    const W = kNNAffinity(dataset.X, kNN);
    return spectralEmbedding(W, K);
  }, [dataset, K, kNN]);
  const dbLabels = useMemo(() => dbscan(dataset.X, committedEps, 5), [dataset, committedEps]);

  const ariMs = useMemo(() => ari(dataset.y, msLabels), [dataset.y, msLabels]);
  const ariSp = useMemo(() => ari(dataset.y, spectral.labels), [dataset.y, spectral.labels]);
  const ariDb = useMemo(() => ari(dataset.y, dbLabels), [dataset.y, dbLabels]);

  const w = containerWidth;
  const isMobile = w > 0 && w < SM_BP;
  const cols = isMobile ? 2 : 4;
  const panelW = w / cols - 8;

  const drawScatter = (title: string, labels: number[], ariV: number, points?: Pt[]) =>
    useD3<SVGSVGElement>((svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const m = { top: 32, right: 8, bottom: 24, left: 28 };
      const innerW = panelW - m.left - m.right;
      const innerH = HEIGHT - m.top - m.bottom;
      const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

      svg.append('text').attr('x', panelW / 2).attr('y', 14).attr('text-anchor', 'middle')
        .style('font', '600 11px var(--font-sans)').style('fill', 'var(--color-text)')
        .text(title);
      svg.append('text').attr('x', panelW / 2).attr('y', 28).attr('text-anchor', 'middle')
        .style('font', '10px var(--font-sans)').style('fill', 'var(--color-text-secondary)')
        .text(`ARI=${ariV.toFixed(3)}`);

      const pts = points ?? dataset.X;
      const xDom: [number, number] = points
        ? [d3.min(pts, (d) => d[0])! - 0.05, d3.max(pts, (d) => d[0])! + 0.05]
        : dataset.xDomain;
      const yDom: [number, number] = points
        ? [d3.min(pts, (d) => d[1])! - 0.05, d3.max(pts, (d) => d[1])! + 0.05]
        : dataset.yDomain;
      const x = d3.scaleLinear().domain(xDom).range([0, innerW]);
      const y = d3.scaleLinear().domain(yDom).range([innerH, 0]);
      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(4)).style('color', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(y).ticks(4)).style('color', 'var(--color-text-secondary)');

      g.selectAll('.pt').data(pts).enter().append('circle')
        .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1]))
        .attr('r', 2.5)
        .style('fill', (_, i) => labels[i] < 0 ? '#999' : CLUSTER_COLORS[labels[i] % CLUSTER_COLORS.length])
        .style('fill-opacity', (_, i) => labels[i] < 0 ? 0.4 : 0.9);
    }, [labels, ariV, points, panelW, dataset]);

  const r1 = drawScatter(`mean-shift h=${committedH.toFixed(2)}`, msLabels, ariMs);
  const r2 = drawScatter(`spectral K=${K} kNN=${kNN}`, spectral.labels, ariSp);
  const r3 = drawScatter(`DBSCAN ε=${committedEps.toFixed(2)}`, dbLabels, ariDb);
  // Spectral embedding panel — use eigenvector coordinates as the "points" with ground-truth coloring.
  const embedPts: Pt[] = spectral.embedding.map((row) => [row[0] ?? 0, row[1] ?? 0]);
  const r4 = drawScatter(`spectral embedding`, dataset.y, ariSp, embedPts);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <p style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginBottom: 8 }}>
        Light pointer per §11. Full spectral-clustering theory lives in <a href="/topics/graph-laplacians" style={{ color: 'var(--color-accent)' }}>graph-laplacians</a> and <a href="/topics/random-walks" style={{ color: 'var(--color-accent)' }}>random-walks</a> in the Graph Theory track.
      </p>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap', marginBottom: 12 }}>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 160 }}>
          h (mean-shift): <strong style={{ color: 'var(--color-text)' }}>{h.toFixed(2)}</strong> {h !== committedH && <em>(release)</em>}
          <input type="range" min={0.05} max={0.5} step={0.01} value={h}
            onChange={(e) => setH(+e.target.value)}
            onMouseUp={() => setCommittedH(h)}
            onTouchEnd={() => setCommittedH(h)}
            onKeyUp={() => setCommittedH(h)}
            aria-label="bandwidth h" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 120 }}>
          K (spectral): <strong style={{ color: 'var(--color-text)' }}>{K}</strong>
          <input type="range" min={2} max={4} step={1} value={K} onChange={(e) => setK(+e.target.value)} aria-label="K for spectral" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 120 }}>
          kNN (affinity): <strong style={{ color: 'var(--color-text)' }}>{kNN}</strong>
          <input type="range" min={5} max={30} step={1} value={kNN} onChange={(e) => setKNN(+e.target.value)} aria-label="kNN for spectral" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', fontSize: 12, color: 'var(--color-text-secondary)', minWidth: 160 }}>
          ε (DBSCAN): <strong style={{ color: 'var(--color-text)' }}>{eps.toFixed(2)}</strong> {eps !== committedEps && <em>(release)</em>}
          <input type="range" min={0.05} max={0.5} step={0.01} value={eps}
            onChange={(e) => setEps(+e.target.value)}
            onMouseUp={() => setCommittedEps(eps)}
            onTouchEnd={() => setCommittedEps(eps)}
            onKeyUp={() => setCommittedEps(eps)}
            aria-label="DBSCAN epsilon" />
        </label>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: 8 }}>
        <svg ref={r1} width={panelW} height={HEIGHT} style={{ background: 'var(--color-surface)', borderRadius: 8 }} />
        <svg ref={r2} width={panelW} height={HEIGHT} style={{ background: 'var(--color-surface)', borderRadius: 8 }} />
        <svg ref={r3} width={panelW} height={HEIGHT} style={{ background: 'var(--color-surface)', borderRadius: 8 }} />
        <svg ref={r4} width={panelW} height={HEIGHT} style={{ background: 'var(--color-surface)', borderRadius: 8 }} />
      </div>
    </div>
  );
}
