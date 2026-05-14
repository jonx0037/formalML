import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  betaFromSVD,
  betaRidgeFromSVD,
  excessRiskMisspecified,
  fillIsotropicGaussian,
  gaussianRng,
  matVec,
  mulberry32,
  sampleSphereBetaStar,
  thinSVD,
  type ThinSVD,
} from './shared/double-descent';

// =============================================================================
// RidgelessVsRidge — §4.4 (Figure 4)
//
// Test risk curves for five λ values overlaid. Caches per-(b, p) SVDs across
// λ values to reuse via the shrinkage formula. λ slider updates live.
// Static fallback: 04_ridgeless_vs_ridge.png
// =============================================================================

const HEIGHT = 460;
const P_GRID = Array.from({ length: 41 }, (_, i) => Math.max(1, Math.round((i / 40) * 200)));
const LAMBDAS_DEFAULT = [0, 0.01, 0.1, 1, 10];

interface SvdCache {
  // cache[bi][gi] = SVD of the b-th replicate's design matrix at p = pGrid[gi]
  svds: ThinSVD[][];
  ys: Float64Array[];
  pGrid: number[];
  betaStar: Float64Array;
}

export default function RidgelessVsRidge() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nCommitted, setNCommitted] = useState(50);
  const [nDisplay, setNDisplay] = useState(50);
  const [snrLogCommitted, setSnrLogCommitted] = useState(0);
  const [snrLogDisplay, setSnrLogDisplay] = useState(0);
  const [BCommitted, setBCommitted] = useState(20);
  const [BDisplay, setBDisplay] = useState(20);

  // Recompute SVD cache when n, SNR, B change. λ values are not in deps — they
  // reuse the same cache through the shrinkage formula.
  const cache: SvdCache = useMemo(() => {
    const n = nCommitted;
    const snr = Math.pow(10, snrLogCommitted);
    const sigma = 1;
    const r = Math.sqrt(snr);
    const pMax = 200;
    const pGrid = P_GRID.filter((p) => p <= pMax);
    const rng = mulberry32(2024);
    const gauss = gaussianRng(rng);
    const betaStar = sampleSphereBetaStar(pMax, r, gauss);
    const replicateRng = mulberry32(7919);
    const replicateGauss = gaussianRng(replicateRng);
    const svds: ThinSVD[][] = [];
    const ys: Float64Array[] = [];
    const Xfull = new Float64Array(n * pMax);
    const Xslice = new Float64Array(n * pMax);
    for (let b = 0; b < BCommitted; b++) {
      fillIsotropicGaussian(Xfull, n, pMax, replicateGauss);
      const y = new Float64Array(n);
      matVec(Xfull, betaStar, n, pMax, y);
      for (let i = 0; i < n; i++) y[i] += sigma * replicateGauss();
      ys.push(y);
      const perP: ThinSVD[] = [];
      for (const p of pGrid) {
        for (let i = 0; i < n; i++) {
          const sb = i * pMax;
          const db = i * p;
          for (let j = 0; j < p; j++) Xslice[db + j] = Xfull[sb + j];
        }
        const svd = thinSVD(Xslice, n, p);
        perP.push(svd);
      }
      svds.push(perP);
    }
    return { svds, ys, pGrid, betaStar };
  }, [nCommitted, snrLogCommitted, BCommitted]);

  // Compute curves for the five default λ values from the cached SVDs.
  const curves = useMemo(() => {
    const { svds, ys, pGrid, betaStar } = cache;
    const result: { lambda: number; means: number[] }[] = [];
    for (const lam of LAMBDAS_DEFAULT) {
      const means = new Array(pGrid.length).fill(0);
      for (let bi = 0; bi < svds.length; bi++) {
        const y = ys[bi];
        for (let gi = 0; gi < pGrid.length; gi++) {
          const svd = svds[bi][gi];
          const beta = lam === 0 ? betaFromSVD(svd, y) : betaRidgeFromSVD(svd, y, lam);
          means[gi] += excessRiskMisspecified(beta, betaStar, pGrid[gi]) / svds.length;
        }
      }
      result.push({ lambda: lam, means });
    }
    return result;
  }, [cache]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 24, right: 36, bottom: 56, left: 64 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const allVals = curves.flatMap((c) => c.means).filter((v) => v > 0);
      const yMin = Math.max(1e-2, d3.min(allVals) ?? 1);
      const yMax = d3.max(allVals) ?? 100;

      const xScale = d3.scaleLinear().domain([1, 200]).range([0, W]);
      const yScale = d3.scaleLog().domain([yMin * 0.7, yMax * 1.4]).range([H, 0]);

      g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xScale).ticks(8))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6, '~g'))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', W / 2).attr('y', H + 38).attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)').style('font-size', '12px').text('model size p');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', -48)
        .attr('text-anchor', 'middle').style('fill', 'var(--color-text)').style('font-size', '12px')
        .text('excess test risk (log scale)');

      // Threshold marker
      g.append('line').attr('x1', xScale(nCommitted)).attr('x2', xScale(nCommitted))
        .attr('y1', 0).attr('y2', H).style('stroke', 'var(--color-accent)')
        .attr('stroke-dasharray', '4 4').attr('opacity', 0.5);

      const palette = ['#0F6E56', '#5BA289', '#D97706', '#534AB7', '#9B1D20'];
      const lineGen = d3.line<[number, number]>().x((d) => xScale(d[0]))
        .y((d) => yScale(Math.max(yMin * 0.7, d[1])));
      curves.forEach((c, idx) => {
        const data: [number, number][] = cache.pGrid.map((p, i) => [p, c.means[i]]);
        g.append('path').datum(data).attr('d', lineGen).attr('fill', 'none')
          .style('stroke', palette[idx]).attr('stroke-width', idx === 0 ? 2.4 : 1.6);
        // Legend entry
        const ly = 14 + idx * 16;
        g.append('line').attr('x1', W - 110).attr('x2', W - 90).attr('y1', ly).attr('y2', ly)
          .style('stroke', palette[idx]).attr('stroke-width', 2);
        g.append('text').attr('x', W - 84).attr('y', ly + 4)
          .style('fill', 'var(--color-text)').style('font-size', '11px')
          .text(c.lambda === 0 ? 'λ = 0 (ridgeless)' : `λ = ${c.lambda}`);
      });
    },
    [containerWidth, curves, cache, nCommitted],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
        aria-label="Ridgeless vs ridge test-risk curves at five lambda values." />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>n: <strong>{nDisplay}</strong></span>
          <input type="range" min={20} max={100} step={5} value={nDisplay}
            onChange={(e) => setNDisplay(parseInt(e.target.value, 10))}
            onMouseUp={() => setNCommitted(nDisplay)} onTouchEnd={() => setNCommitted(nDisplay)}
            onKeyUp={() => setNCommitted(nDisplay)} aria-label="Training set size" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>SNR: <strong>{Math.pow(10, snrLogDisplay).toFixed(2)}</strong></span>
          <input type="range" min={-1} max={1} step={0.05} value={snrLogDisplay}
            onChange={(e) => setSnrLogDisplay(parseFloat(e.target.value))}
            onMouseUp={() => setSnrLogCommitted(snrLogDisplay)}
            onTouchEnd={() => setSnrLogCommitted(snrLogDisplay)}
            onKeyUp={() => setSnrLogCommitted(snrLogDisplay)} aria-label="SNR" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 180 }}>
          <span>B (replicates): <strong>{BDisplay}</strong></span>
          <input type="range" min={5} max={50} step={5} value={BDisplay}
            onChange={(e) => setBDisplay(parseInt(e.target.value, 10))}
            onMouseUp={() => setBCommitted(BDisplay)} onTouchEnd={() => setBCommitted(BDisplay)}
            onKeyUp={() => setBCommitted(BDisplay)} aria-label="Monte Carlo replicates" />
        </label>
      </div>
    </div>
  );
}
