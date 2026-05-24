import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// Visualize swap-exchangeability of the 2p × 2p Gram matrix [X, X̃]^T [X, X̃].
// Build Σ from AR(1) ρ = 0.3 with p = 6, set s = min(2 λ_min(Σ), 1), and verify
// that swapping X_j ↔ X̃_j leaves the Gram unchanged.

const P = 6;
const RHO = 0.3;

function ar1Sigma(): number[][] {
  const S: number[][] = Array.from({ length: P }, () => new Array(P).fill(0));
  for (let i = 0; i < P; i++) for (let j = 0; j < P; j++) S[i][j] = Math.pow(RHO, Math.abs(i - j));
  return S;
}

function eigMin(S: number[][]): number {
  // Power iteration on (cI - S) for smallest eigenvalue. For small P, use a
  // tridiagonal-friendly approach: AR(1) has known eigs but we compute numerically.
  // Inverse power iteration: solve (S - μ I) x = b iteratively. Simpler: compute
  // all eigenvalues via Jacobi rotations.
  const n = S.length;
  const A: number[][] = S.map((row) => row.slice());
  // Symmetric Jacobi
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
    if (off < 1e-14) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p][q];
        if (Math.abs(apq) < 1e-14) continue;
        const app = A[p][p];
        const aqq = A[q][q];
        const theta = (aqq - app) / (2 * apq);
        const t = theta >= 0 ? 1 / (theta + Math.sqrt(1 + theta * theta)) : 1 / (theta - Math.sqrt(1 + theta * theta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;
        A[p][p] = app - t * apq;
        A[q][q] = aqq + t * apq;
        A[p][q] = 0;
        A[q][p] = 0;
        for (let r = 0; r < n; r++) {
          if (r === p || r === q) continue;
          const arp = A[r][p];
          const arq = A[r][q];
          A[r][p] = c * arp - s * arq;
          A[r][q] = s * arp + c * arq;
          A[p][r] = A[r][p];
          A[q][r] = A[r][q];
        }
      }
    }
  }
  let m = Infinity;
  for (let i = 0; i < n; i++) if (A[i][i] < m) m = A[i][i];
  return m;
}

function buildGram(swapSet: Set<number>): { pre: number[][]; post: number[][]; s: number; absDiff: number } {
  const Sigma = ar1Sigma();
  const lambdaMin = eigMin(Sigma);
  const s = Math.min(2 * lambdaMin, 1);
  // Build 2p × 2p block matrix: [[Σ, Σ-diag(s)], [Σ-diag(s), Σ]]
  const Q = 2 * P;
  const G: number[][] = Array.from({ length: Q }, () => new Array(Q).fill(0));
  for (let i = 0; i < P; i++) {
    for (let j = 0; j < P; j++) {
      G[i][j] = Sigma[i][j];
      G[P + i][P + j] = Sigma[i][j];
      G[i][P + j] = Sigma[i][j] - (i === j ? s : 0);
      G[P + i][j] = G[i][P + j];
    }
  }
  // Apply swap: for j in swapSet, swap row j ↔ row P+j and column j ↔ column P+j.
  const post: number[][] = G.map((row) => row.slice());
  for (const j of swapSet) {
    const rj = j;
    const tj = P + j;
    for (let k = 0; k < Q; k++) {
      const tmpRow = post[rj][k];
      post[rj][k] = post[tj][k];
      post[tj][k] = tmpRow;
    }
    for (let k = 0; k < Q; k++) {
      const tmpCol = post[k][rj];
      post[k][rj] = post[k][tj];
      post[k][tj] = tmpCol;
    }
  }
  let absDiff = 0;
  for (let i = 0; i < Q; i++)
    for (let j = 0; j < Q; j++) absDiff = Math.max(absDiff, Math.abs(G[i][j] - post[i][j]));
  return { pre: G, post, s, absDiff };
}

export default function KnockoffSwapExchangeability() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [swapSet, setSwapSet] = useState<Set<number>>(new Set([1, 4]));

  const { pre, post, s, absDiff } = useMemo(() => buildGram(swapSet), [swapSet]);
  const Q = 2 * P;
  const W = Math.max(280, Math.min(containerWidth || 720, 720));
  const isMobile = W < 640;
  const PANEL_W = isMobile ? W : (W - 16) / 2;
  const PANEL_H = PANEL_W;
  const margin = 28;
  const cell = (PANEL_W - 2 * margin) / Q;

  function renderHeatmap(matrix: number[][], title: string) {
    return (svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) => {
      svg.selectAll('*').remove();
      const g = svg.append('g').attr('transform', `translate(${margin},${margin})`);
      const color = d3.scaleLinear<string>().domain([-1, 0, 1]).range(['#1d4ed8', '#f5f5f5', '#dc2626']);
      for (let i = 0; i < Q; i++) {
        for (let j = 0; j < Q; j++) {
          g.append('rect')
            .attr('x', j * cell)
            .attr('y', i * cell)
            .attr('width', cell - 0.5)
            .attr('height', cell - 0.5)
            .attr('fill', color(matrix[i][j]));
          if (Math.abs(matrix[i][j]) > 0.15) {
            g.append('text')
              .attr('x', j * cell + cell / 2)
              .attr('y', i * cell + cell / 2 + 3)
              .attr('text-anchor', 'middle')
              .style('fill', Math.abs(matrix[i][j]) > 0.6 ? 'white' : 'var(--color-text)')
              .style('font-family', 'var(--font-mono)')
              .style('font-size', `${Math.min(cell * 0.32, 11)}px`)
              .text(matrix[i][j].toFixed(2));
          }
        }
      }
      // block separators
      g.append('line').attr('x1', P * cell).attr('x2', P * cell).attr('y1', 0).attr('y2', Q * cell).style('stroke', 'var(--color-text)').attr('stroke-width', 1.4);
      g.append('line').attr('x1', 0).attr('x2', Q * cell).attr('y1', P * cell).attr('y2', P * cell).style('stroke', 'var(--color-text)').attr('stroke-width', 1.4);
      // labels
      svg.append('text').attr('x', margin + (P * cell) / 2).attr('y', margin - 8).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '0.72rem').text('X');
      svg.append('text').attr('x', margin + P * cell + (P * cell) / 2).attr('y', margin - 8).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '0.72rem').text('X̃');
      svg.append('text').attr('x', 8).attr('y', PANEL_H - 6).style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '0.72rem').text(title);
    };
  }

  const preRef = useD3<SVGSVGElement>(renderHeatmap(pre, 'before swap'), [pre, cell]);
  const postRef = useD3<SVGSVGElement>(renderHeatmap(post, 'after swap'), [post, cell]);

  function toggle(j: number) {
    const next = new Set(swapSet);
    if (next.has(j)) next.delete(j);
    else next.add(j);
    setSwapSet(next);
  }

  return (
    <div ref={containerRef} className="not-prose my-6 w-full">
      <div className="flex flex-wrap items-center gap-3 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
        <span>swap subset S:</span>
        {Array.from({ length: P }, (_, j) => (
          <button
            key={j}
            type="button"
            onClick={() => toggle(j)}
            className="rounded border px-2 py-1 tabular-nums"
            style={{
              borderColor: 'var(--color-border)',
              background: swapSet.has(j) ? 'var(--color-accent)' : 'var(--color-surface)',
              color: swapSet.has(j) ? 'white' : 'var(--color-text)',
            }}
            aria-pressed={swapSet.has(j)}
          >
            X_{j}
          </button>
        ))}
        <span style={{ color: 'var(--color-text)' }}>
          equicorrelation s = <span className="tabular-nums">{s.toFixed(3)}</span>
        </span>
        <span style={{ color: 'var(--color-text)' }}>
          max | pre − post | = <span className="tabular-nums">{absDiff.toFixed(4)}</span>
        </span>
      </div>
      <div className={isMobile ? 'mt-2' : 'mt-2 flex gap-4'}>
        <svg ref={preRef} width={PANEL_W} height={PANEL_H} role="img" aria-label="Gram matrix before swap" />
        <svg ref={postRef} width={PANEL_W} height={PANEL_H} role="img" aria-label="Gram matrix after swap" />
      </div>
      <p className="mt-1 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
        AR(1) Σ with ρ = 0.3, p = 6 features. The 2p × 2p Gram matrix [X, X̃]ᵀ [X, X̃] has block structure
        [[Σ, Σ − diag(s)], [Σ − diag(s), Σ]]. Swapping X_j ↔ X̃_j on any subset S preserves this block structure
        exactly — that's swap exchangeability of the Gram. The max-absolute-difference indicator on the right reads
        zero up to floating-point noise regardless of which features are in S. Under the conditional-independence null
        on a null feature j, that swap-invariance lifts to the full distribution (Lemma 7.1).
      </p>
    </div>
  );
}
