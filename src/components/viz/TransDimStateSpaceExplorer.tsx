// =============================================================================
// TransDimStateSpaceExplorer.tsx
//
// §1 of reversible-jump-mcmc. Two-panel motivating figure: the variable-selection
// Boolean lattice (left, p∈{3,4,5,6}) and the Galaxy dataset with GMM fits at
// K∈{1..6} (right). Both panels are cheap (lattice is closed-form; GMMs are
// precomputed lazily on first render), so no display-vs-committed slider is needed.
//
// Static fallback: /images/topics/reversible-jump-mcmc/fig01_state_space.png
// =============================================================================
import { useMemo, useState } from 'react';
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

function gaussianPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

interface GMMFit {
  mu: number[];
  sigma2: number[];
  w: number[];
}

function emGMM(data: number[], K: number, maxIter = 200, tol = 1e-6, seed = 42): GMMFit {
  const n = data.length;
  let rngState = seed;
  const rand = () => {
    rngState = (rngState * 1664525 + 1013904223) >>> 0;
    return rngState / 4294967296;
  };
  // Init with K quantiles for means; pooled var for variances; uniform weights.
  const sorted = data.slice().sort((a, b) => a - b);
  const mu = Array.from({ length: K }, (_, k) => sorted[Math.floor(((k + 0.5) / K) * n)]);
  const pooledVar = d3.variance(data) ?? 1;
  const sigma2 = Array.from({ length: K }, () => pooledVar);
  const w = Array.from({ length: K }, () => 1 / K);
  // Small RNG noise on means to break degeneracy on repeated points.
  for (let k = 0; k < K; k++) mu[k] += (rand() - 0.5) * 0.01;

  let prevLL = -Infinity;
  for (let iter = 0; iter < maxIter; iter++) {
    // E-step: responsibilities
    const r: number[][] = [];
    let ll = 0;
    for (let i = 0; i < n; i++) {
      const row = new Array(K);
      let rowSum = 0;
      for (let k = 0; k < K; k++) {
        row[k] = w[k] * gaussianPdf(data[i], mu[k], Math.sqrt(sigma2[k]));
        rowSum += row[k];
      }
      ll += Math.log(rowSum + 1e-300);
      for (let k = 0; k < K; k++) row[k] /= rowSum + 1e-300;
      r.push(row);
    }
    // M-step
    for (let k = 0; k < K; k++) {
      let nk = 0;
      let muk = 0;
      for (let i = 0; i < n; i++) {
        nk += r[i][k];
        muk += r[i][k] * data[i];
      }
      mu[k] = muk / nk;
      let s2k = 0;
      for (let i = 0; i < n; i++) {
        const d = data[i] - mu[k];
        s2k += r[i][k] * d * d;
      }
      sigma2[k] = Math.max(s2k / nk, 1e-6);
      w[k] = nk / n;
    }
    if (Math.abs(ll - prevLL) < tol) break;
    prevLL = ll;
  }
  return { mu, sigma2, w };
}

function mixturePdf(x: number, fit: GMMFit): number {
  let p = 0;
  for (let k = 0; k < fit.mu.length; k++) {
    p += fit.w[k] * gaussianPdf(x, fit.mu[k], Math.sqrt(fit.sigma2[k]));
  }
  return p;
}

interface LatticeNode {
  id: string;
  bits: number[];
  cardinality: number;
  x: number;
  y: number;
}
interface LatticeEdge {
  source: string;
  target: string;
}
function buildLattice(p: number): { nodes: LatticeNode[]; edges: LatticeEdge[] } {
  const total = 1 << p;
  const nodes: LatticeNode[] = [];
  for (let id = 0; id < total; id++) {
    const bits: number[] = [];
    for (let j = 0; j < p; j++) bits.push((id >> j) & 1);
    const card = bits.reduce((s, b) => s + b, 0);
    nodes.push({ id: bits.join(''), bits, cardinality: card, x: 0, y: 0 });
  }
  // Layout: layer k at vertical position card/p; horizontal position by binomial index
  const layerCounts = new Array(p + 1).fill(0);
  const layerIndices: number[] = new Array(total).fill(0);
  for (let i = 0; i < total; i++) {
    layerIndices[i] = layerCounts[nodes[i].cardinality];
    layerCounts[nodes[i].cardinality]++;
  }
  for (let i = 0; i < total; i++) {
    const k = nodes[i].cardinality;
    const cnt = layerCounts[k];
    nodes[i].x = (layerIndices[i] + 0.5) / cnt;
    nodes[i].y = k / p;
  }
  // Edges: Hamming-distance-one neighbors with target.card = source.card + 1
  const edges: LatticeEdge[] = [];
  for (let i = 0; i < total; i++) {
    for (let j = 0; j < p; j++) {
      if (nodes[i].bits[j] === 0) {
        const targetId = i | (1 << j);
        edges.push({ source: nodes[i].id, target: nodes[targetId].id });
      }
    }
  }
  return { nodes, edges };
}

function hammingDistanceOne(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) diff++;
    if (diff > 1) return false;
  }
  return diff === 1;
}

export default function TransDimStateSpaceExplorer() {
  const [p, setP] = useState(4);
  const [K, setK] = useState(3);
  const [selected, setSelected] = useState<string | null>(null);

  const lattice = useMemo(() => buildLattice(p), [p]);

  const gmmFits = useMemo(() => {
    const fits: Record<number, GMMFit> = {};
    for (let k = 1; k <= 6; k++) fits[k] = emGMM(GALAXY, k);
    return fits;
  }, []);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const totalWidth = containerWidth || 800;
  const isMobile = totalWidth < 640;
  const panelWidth = isMobile ? totalWidth : (totalWidth - 24) / 2;
  const panelHeight = 320;

  const latticeRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const M = { top: 20, right: 20, bottom: 28, left: 30 };
      const W = panelWidth - M.left - M.right;
      const H = panelHeight - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const xScale = d3.scaleLinear().domain([0, 1]).range([0, W]);
      const yScale = d3.scaleLinear().domain([0, 1]).range([H, 0]);

      const isNeighbor = (id: string) =>
        selected !== null && (id === selected || hammingDistanceOne(id, selected));

      g.selectAll('line.edge')
        .data(lattice.edges)
        .join('line')
        .attr('class', 'edge')
        .attr('x1', (e) => xScale(lattice.nodes.find((n) => n.id === e.source)!.x))
        .attr('y1', (e) => yScale(lattice.nodes.find((n) => n.id === e.source)!.y))
        .attr('x2', (e) => xScale(lattice.nodes.find((n) => n.id === e.target)!.x))
        .attr('y2', (e) => yScale(lattice.nodes.find((n) => n.id === e.target)!.y))
        .style('stroke', 'var(--color-border)')
        .style('stroke-width', 1);

      const nodeR = Math.max(4, Math.min(10, 18 - 2 * p));
      g.selectAll('circle.node')
        .data(lattice.nodes)
        .join('circle')
        .attr('class', 'node')
        .attr('cx', (d) => xScale(d.x))
        .attr('cy', (d) => yScale(d.y))
        .attr('r', nodeR)
        .style('fill', (d) =>
          selected === d.id
            ? 'var(--color-accent)'
            : isNeighbor(d.id)
              ? 'var(--color-accent)'
              : 'var(--color-surface)',
        )
        .style('stroke', 'var(--color-text)')
        .style('stroke-width', 1)
        .style('cursor', 'pointer')
        .on('click', (_, d) => setSelected(d.id === selected ? null : d.id));

      // Layer labels
      for (let k = 0; k <= p; k++) {
        g.append('text')
          .attr('x', -8)
          .attr('y', yScale(k / p))
          .attr('dy', '0.32em')
          .attr('text-anchor', 'end')
          .style('font-size', '11px')
          .style('font-family', 'var(--font-mono)')
          .style('fill', 'var(--color-text-secondary)')
          .text(`|S|=${k}`);
      }

      // Selected-node info
      if (selected !== null) {
        const sn = lattice.nodes.find((n) => n.id === selected)!;
        const neighborCount = lattice.nodes.filter((n) => hammingDistanceOne(n.id, selected)).length;
        g.append('text')
          .attr('x', W / 2)
          .attr('y', H + 22)
          .attr('text-anchor', 'middle')
          .style('font-size', '12px')
          .style('font-family', 'var(--font-mono)')
          .style('fill', 'var(--color-text)')
          .text(`γ = ${sn.id} · |S| = ${sn.cardinality} · ${neighborCount} neighbors`);
      } else {
        g.append('text')
          .attr('x', W / 2)
          .attr('y', H + 22)
          .attr('text-anchor', 'middle')
          .style('font-size', '12px')
          .style('font-family', 'var(--font-sans)')
          .style('fill', 'var(--color-text-secondary)')
          .text('Click a node to highlight its neighbors');
      }
    },
    [lattice, selected, p, panelWidth, panelHeight],
  );

  const galaxyRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth <= 0) return;
      const M = { top: 20, right: 20, bottom: 36, left: 36 };
      const W = panelWidth - M.left - M.right;
      const H = panelHeight - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);

      const xDomain: [number, number] = [8, 36];
      const xScale = d3.scaleLinear().domain(xDomain).range([0, W]);

      const bins = d3
        .bin()
        .domain(xDomain)
        .thresholds(d3.range(xDomain[0], xDomain[1] + 0.5, 1))(GALAXY);
      const histDensity = bins.map((b) => b.length / GALAXY.length / 1.0); // bin width = 1

      const fit = gmmFits[K];
      const xGrid = d3.range(xDomain[0], xDomain[1], 0.05);
      const pdfVals = xGrid.map((x) => mixturePdf(x, fit));

      const yMax = Math.max(d3.max(histDensity) ?? 0.1, d3.max(pdfVals) ?? 0.1) * 1.15;
      const yScale = d3.scaleLinear().domain([0, yMax]).range([H, 0]);

      g.append('g')
        .attr('transform', `translate(0,${H})`)
        .call(d3.axisBottom(xScale).ticks(7))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');
      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)');

      // Histogram bars
      g.selectAll('rect.hist')
        .data(bins)
        .join('rect')
        .attr('class', 'hist')
        .attr('x', (b) => xScale(b.x0!))
        .attr('y', (b) => yScale(b.length / GALAXY.length))
        .attr('width', (b) => Math.max(0, xScale(b.x1!) - xScale(b.x0!) - 1))
        .attr('height', (b) => H - yScale(b.length / GALAXY.length))
        .style('fill', 'var(--color-border)')
        .style('opacity', 0.7);

      // PDF curve
      const line = d3
        .line<number>()
        .x((_, i) => xScale(xGrid[i]))
        .y((y) => yScale(y))
        .curve(d3.curveMonotoneX);
      g.append('path')
        .datum(pdfVals)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', 'var(--color-accent)')
        .style('stroke-width', 2);

      // Component means as ticks
      g.selectAll('line.mu')
        .data(fit.mu)
        .join('line')
        .attr('class', 'mu')
        .attr('x1', (m) => xScale(m))
        .attr('x2', (m) => xScale(m))
        .attr('y1', H)
        .attr('y2', H - 8)
        .style('stroke', 'var(--color-accent)')
        .style('stroke-width', 2);

      // x-label
      g.append('text')
        .attr('x', W / 2)
        .attr('y', H + 32)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-family', 'var(--font-sans)')
        .style('fill', 'var(--color-text)')
        .text('Galaxy velocity (km/s × 10³)');
    },
    [K, gmmFits, panelWidth, panelHeight],
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
        <div style={{ width: isMobile ? '100%' : panelWidth }}>
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
            <label htmlFor="p-slider" style={{ minWidth: 90 }}>
              p = <strong>{p}</strong>
            </label>
            <input
              id="p-slider"
              type="range"
              min={3}
              max={6}
              step={1}
              value={p}
              onChange={(e) => {
                setP(parseInt(e.target.value, 10));
                setSelected(null);
              }}
              style={{ flex: 1 }}
              aria-label="Number of covariates p"
            />
            <span style={{ minWidth: 80, color: 'var(--color-text-secondary)' }}>
              2<sup>{p}</sup> = {1 << p} models
            </span>
          </div>
          <svg ref={latticeRef} width={panelWidth} height={panelHeight} />
        </div>
        <div style={{ width: isMobile ? '100%' : panelWidth }}>
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
            <label htmlFor="K-slider" style={{ minWidth: 90 }}>
              K = <strong>{K}</strong>
            </label>
            <input
              id="K-slider"
              type="range"
              min={1}
              max={6}
              step={1}
              value={K}
              onChange={(e) => setK(parseInt(e.target.value, 10))}
              style={{ flex: 1 }}
              aria-label="Number of mixture components K"
            />
            <span style={{ minWidth: 80, color: 'var(--color-text-secondary)' }}>
              n = {GALAXY.length}
            </span>
          </div>
          <svg ref={galaxyRef} width={panelWidth} height={panelHeight} />
        </div>
      </div>
    </div>
  );
}
