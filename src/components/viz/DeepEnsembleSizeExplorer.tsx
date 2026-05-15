import { useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  ensembleMixtureDensity,
  ensemblePredict,
  fTrue,
  linspace,
  type MlpCoefs,
} from './shared/uncertainty-quantification';

// =============================================================================
// DeepEnsembleSizeExplorer — §9
//
// Slider for M ∈ [2, 50] (commit-on-release). Plots the mixture-of-Gaussians
// predictive density at a user-selected diagnostic test point x*. Shows the
// per-member component densities faintly and the bold mixture density bold.
// Verifies the §9.4 LTV identity numerically: mixture variance ≈ σ²_ale + σ²_ens.
// Static fallback: public/images/topics/uncertainty-quantification/fig_09_mixture_density.png
// =============================================================================

const HEIGHT = 440;
const Y_GRID = linspace(-3.5, 3.5, 400);

type EnsemblePayload = {
  X: number[];
  y: number[];
  members: { coefs: number[][][]; intercepts: number[][] }[];
  aleatoric: { centers: number[]; vals: number[] };
};

function interpAleatoric(payload: EnsemblePayload, X: number[]): number[] {
  const { centers, vals } = payload.aleatoric;
  return X.map((x) => {
    if (x <= centers[0]) return vals[0];
    if (x >= centers[centers.length - 1]) return vals[centers.length - 1];
    let lo = 0;
    let hi = centers.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (centers[mid] <= x) lo = mid;
      else hi = mid;
    }
    const t = (x - centers[lo]) / (centers[hi] - centers[lo]);
    return vals[lo] + t * (vals[hi] - vals[lo]);
  });
}

export default function DeepEnsembleSizeExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [payload, setPayload] = useState<EnsemblePayload | null>(null);
  const [mDisplay, setMDisplay] = useState(20);
  const [mCommitted, setMCommitted] = useState(20);
  const [xStar, setXStar] = useState(1.5);

  useEffect(() => {
    let cancelled = false;
    fetch('/sample-data/uncertainty-quantification/ensemble.json')
      .then((r) => r.json())
      .then((data) => { if (!cancelled) setPayload(data); })
      .catch((err) => console.error('[DeepEnsembleSizeExplorer] fetch failed:', err));
    return () => { cancelled = true; };
  }, []);

  const result = useMemo(() => {
    if (!payload) return null;
    const members = payload.members.slice(0, mCommitted) as unknown as MlpCoefs[];
    members.forEach((m) => { m.activation = 'tanh'; });
    const pred = ensemblePredict([xStar], members);
    const muMembers = pred.members.map((row) => row[0]);
    const aleVar = interpAleatoric(payload, [xStar])[0];
    const aleSd = Math.sqrt(aleVar);
    const density = ensembleMixtureDensity(Y_GRID, muMembers, aleSd);
    const epsVar = pred.variance[0];
    const ltvTotal = aleVar + epsVar;
    return { muMembers, aleSd, density, epsVar, aleVar, ltvTotal, ensembleMean: pred.mean[0] };
  }, [payload, mCommitted, xStar]);

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const margin = { top: 28, right: 24, bottom: 48, left: 56 };
      const W = w - margin.left - margin.right;
      const H = HEIGHT - margin.top - margin.bottom;
      svg.selectAll('*').remove();
      if (W <= 0 || !result) return;
      svg.attr('viewBox', `0 0 ${w} ${HEIGHT}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const xScale = d3.scaleLinear().domain([-3, 3]).range([0, W]);
      const yMax = Math.max(...result.density) * 1.1;
      const yScale = d3.scaleLinear().domain([0, yMax]).range([H, 0]).nice();

      // Per-member component densities (faint).
      for (const mu of result.muMembers) {
        const comp = Y_GRID.map((y) => {
          const z = (y - mu) / result.aleSd;
          return Math.exp(-0.5 * z * z) / (result.aleSd * Math.sqrt(2 * Math.PI)) / result.muMembers.length;
        });
        const lineComp = d3.line<number>()
          .x((_, i) => xScale(Y_GRID[i])).y((d) => yScale(d));
        g.append('path').datum(comp).attr('d', lineComp).attr('fill', 'none')
          .style('stroke', '#f59e0b').style('opacity', 0.16).attr('stroke-width', 1);
      }

      // Bold mixture density.
      const lineGen = d3.line<number>()
        .x((_, i) => xScale(Y_GRID[i])).y((d) => yScale(d));
      g.append('path').datum(result.density).attr('d', lineGen).attr('fill', 'none')
        .style('stroke', '#f59e0b').attr('stroke-width', 2.4);

      // Vertical lines: ensemble mean and true f(x*).
      const drawV = (x: number, color: string, label: string, dash: string) => {
        g.append('line').attr('x1', xScale(x)).attr('x2', xScale(x))
          .attr('y1', 0).attr('y2', H).style('stroke', color)
          .attr('stroke-dasharray', dash).attr('stroke-width', 1.4);
        g.append('text').attr('x', xScale(x) + 4).attr('y', 12)
          .style('font-size', '11px').style('fill', color).text(label);
      };
      drawV(result.ensembleMean, '#000', `ensemble mean = ${result.ensembleMean.toFixed(2)}`, '');
      drawV(fTrue(xStar), '#dc2626', `true f(x*) = ${fTrue(xStar).toFixed(2)}`, '4 3');

      g.append('g').attr('transform', `translate(0,${H})`)
        .call(d3.axisBottom(xScale).ticks(7))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.selectAll('.domain, .tick line').style('stroke', 'var(--color-border)');
      g.append('text').attr('x', W / 2).attr('y', H + 36).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text)').text('y');
      g.append('text').attr('x', W / 2).attr('y', -10).attr('text-anchor', 'middle')
        .style('font-size', '12px').style('fill', 'var(--color-text)')
        .text(`Mixture density at x* = ${xStar.toFixed(2)}, M = ${mCommitted} — `
          + `σ²_ens = ${result.epsVar.toFixed(3)}, σ²_ale = ${result.aleVar.toFixed(3)}, total = ${result.ltvTotal.toFixed(3)} (LTV)`);
    },
    [containerWidth, result, xStar, mCommitted],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      {payload ? (
        <svg ref={svgRef} width="100%" height={HEIGHT} role="img"
          aria-label="Mixture-of-Gaussians predictive density from a deep ensemble at a user-selected x*, with the LTV decomposition annotated." />
      ) : (
        <div style={{ height: HEIGHT, display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: 'var(--color-text-secondary)' }}>
          Loading pre-trained ensemble …
        </div>
      )}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem',
        fontSize: '13px', color: 'var(--color-text)' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 220 }}>
          <span>ensemble size M: <strong>{mDisplay}</strong></span>
          <input type="range" min={2} max={50} step={1} value={mDisplay}
            onChange={(e) => setMDisplay(parseInt(e.target.value, 10))}
            onMouseUp={() => setMCommitted(mDisplay)} onTouchEnd={() => setMCommitted(mDisplay)}
            onKeyUp={() => setMCommitted(mDisplay)} aria-label="Ensemble size M" />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', minWidth: 220 }}>
          <span>test point x*: <strong>{xStar.toFixed(2)}</strong></span>
          <input type="range" min={-2.8} max={2.8} step={0.1} value={xStar}
            onChange={(e) => setXStar(parseFloat(e.target.value))}
            aria-label="Diagnostic test point x*" />
        </label>
      </div>
    </div>
  );
}
