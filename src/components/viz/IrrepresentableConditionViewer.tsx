import { useMemo } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  choleskyFactor,
  solveLowerTriangular,
  solveUpperTriangularT,
} from './shared/gaussian-processes';

// =============================================================================
// IrrepresentableConditionViewer — interactive companion to §6.2 Figure 6.1.
// Plots the population irrepresentable quantity
//   IC(ρ) := ‖Σ_{Sᶜ S} Σ_{S S}⁻¹ sign(β*_S)‖_∞
// for DGP-1-style AR(1) Toeplitz designs Σ_jk = ρ^|j-k|, contiguous active
// set S = {0, …, s-1}, sign(β*_S) = (1, 1, …, 1), as ρ varies in [0, 0.95].
//
// The horizontal line at IC = 1 is the sign-consistency threshold (Zhao-Yu
// 2006 / Wainwright 2009). At ρ small, IC is far below 1 (lasso reliably
// recovers the support). As ρ → 1 the IC quantity grows; at the crossover
// point, the lasso provably fails sign-consistency and elastic net or
// adaptive lasso become necessary.
//
// Compute: 50 ρ values × Cholesky on a 10×10 Σ_{S,S} ≈ <10 ms. Pre-compute
// once with useMemo.
//
// Static fallback: public/images/topics/high-dimensional-regression/fig_06_01_irrepresentable_condition.png
// =============================================================================

const HEIGHT = 460;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 28, right: 16, bottom: 50, left: 60 };

const P = 200;
const S = 10;
const N_RHO = 50;
const RHO_MIN = 0.0;
const RHO_MAX = 0.95;

const TEAL = '#0F6E56';
const RED = '#B91C1C';
const SLATE = '#6B6B6B';

interface ICPoint {
  rho: number;
  ic: number;
}

// -----------------------------------------------------------------------------
// AR(1) Toeplitz population covariance: Σ_jk = ρ^|j-k|.
// -----------------------------------------------------------------------------

function ar1Sigma(p: number, rho: number): number[][] {
  const Sigma: number[][] = [];
  for (let i = 0; i < p; i++) {
    const row = new Array<number>(p).fill(0);
    for (let j = 0; j < p; j++) row[j] = Math.pow(rho, Math.abs(i - j));
    Sigma.push(row);
  }
  return Sigma;
}

// -----------------------------------------------------------------------------
// Compute IC quantity for given p, s, ρ.
// -----------------------------------------------------------------------------

function computeIC(p: number, s: number, rho: number): number {
  // Σ_{S,S} is the top-left s×s AR(1) Toeplitz block.
  const Sigma_SS: number[][] = [];
  for (let i = 0; i < s; i++) {
    const row = new Array<number>(s).fill(0);
    for (let j = 0; j < s; j++) row[j] = Math.pow(rho, Math.abs(i - j));
    Sigma_SS.push(row);
  }
  // Add tiny jitter for ρ very close to 1 (otherwise Cholesky pivot may underflow).
  for (let i = 0; i < s; i++) Sigma_SS[i][i] += 1e-10;
  // sign(β*_S) = all-ones vector.
  const signBeta = new Array<number>(s).fill(1);
  // Solve Σ_{S,S} v = sign(β*_S) for v.
  const L = choleskyFactor(Sigma_SS);
  const yt = solveLowerTriangular(L, signBeta);
  const v = solveUpperTriangularT(L, yt);
  // For each j ∈ S^c (i.e., j ≥ s), compute Σ_{j, S} · v = sum_{k=0..s-1} ρ^|j-k| · v_k.
  // Take the max absolute value.
  let maxIC = 0;
  for (let j = s; j < p; j++) {
    let dot = 0;
    for (let k = 0; k < s; k++) dot += Math.pow(rho, Math.abs(j - k)) * v[k];
    maxIC = Math.max(maxIC, Math.abs(dot));
  }
  return maxIC;
}

function computeICCurve(): ICPoint[] {
  const points: ICPoint[] = [];
  for (let i = 0; i < N_RHO; i++) {
    const rho = RHO_MIN + ((RHO_MAX - RHO_MIN) * i) / (N_RHO - 1);
    points.push({ rho, ic: computeIC(P, S, rho) });
  }
  return points;
}

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export default function IrrepresentableConditionViewer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const w = containerWidth || 720;
  const isMobile = w < SM_BREAKPOINT;

  const curve = useMemo(() => computeICCurve(), []);

  // Find the crossover ρ where IC first exceeds 1.
  const crossoverRho = useMemo(() => {
    for (let i = 1; i < curve.length; i++) {
      if (curve[i - 1].ic <= 1 && curve[i].ic > 1) {
        const t = (1 - curve[i - 1].ic) / (curve[i].ic - curve[i - 1].ic);
        return curve[i - 1].rho + t * (curve[i].rho - curve[i - 1].rho);
      }
    }
    return null; // never crosses (unlikely with rho up to 0.95)
  }, [curve]);

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLinear().domain([0, RHO_MAX]).range([0, innerW]);
      const yMax = Math.max(2.5, Math.max(...curve.map((c) => c.ic)) * 1.1);
      const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

      const xAxis = d3.axisBottom(xScale).ticks(isMobile ? 5 : 8).tickSize(-innerH);
      const yAxis = d3.axisLeft(yScale).ticks(isMobile ? 5 : 8).tickSize(-innerW);
      g.append('g').attr('transform', `translate(0,${innerH})`).call(xAxis).call((sel) => {
        sel.selectAll('line').style('stroke', 'var(--color-border)');
        sel.selectAll('path').style('stroke', 'var(--color-text-secondary)');
        sel.selectAll('text').style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)');
      });
      g.append('g').call(yAxis).call((sel) => {
        sel.selectAll('line').style('stroke', 'var(--color-border)');
        sel.selectAll('path').style('stroke', 'var(--color-text-secondary)');
        sel.selectAll('text').style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)');
      });

      g.append('text').attr('text-anchor', 'middle').attr('x', innerW / 2).attr('y', innerH + 38).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '13px').text('AR(1) correlation ρ');
      g.append('text').attr('text-anchor', 'middle').attr('transform', `translate(-44,${innerH / 2}) rotate(-90)`).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '13px').text('IC quantity ‖Σ_{Sᶜ S} Σ_{S S}⁻¹ sign(β*_S)‖_∞');

      // IC = 1 threshold line.
      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yScale(1)).attr('y2', yScale(1)).style('stroke', RED).style('stroke-dasharray', '5 3').style('stroke-width', 1.5).style('opacity', 0.7);
      g.append('text').attr('x', innerW - 4).attr('y', yScale(1) - 6).attr('text-anchor', 'end').style('fill', RED).style('font-family', 'var(--font-mono)').style('font-size', '11px').text('IC = 1 (sign-consistency threshold)');

      // ρ = 0.5 marker (DGP-1 default).
      g.append('line').attr('x1', xScale(0.5)).attr('x2', xScale(0.5)).attr('y1', 0).attr('y2', innerH).style('stroke', SLATE).style('stroke-dasharray', '3 3').style('stroke-width', 1).style('opacity', 0.5);
      g.append('text').attr('x', xScale(0.5) + 4).attr('y', innerH - 8).style('fill', SLATE).style('font-family', 'var(--font-mono)').style('font-size', '11px').text('ρ = 0.5 (DGP-1)');

      // Crossover marker.
      if (crossoverRho !== null) {
        g.append('line').attr('x1', xScale(crossoverRho)).attr('x2', xScale(crossoverRho)).attr('y1', 0).attr('y2', innerH).style('stroke', RED).style('stroke-dasharray', '3 3').style('stroke-width', 1).style('opacity', 0.7);
        g.append('text').attr('x', xScale(crossoverRho) + 4).attr('y', 12).style('fill', RED).style('font-family', 'var(--font-mono)').style('font-size', '11px').text(`crossover ρ ≈ ${crossoverRho.toFixed(3)}`);
      }

      // IC curve.
      const lineGen = d3.line<ICPoint>().x((c) => xScale(c.rho)).y((c) => yScale(c.ic));
      g.append('path').datum(curve).attr('d', lineGen).style('fill', 'none').style('stroke', TEAL).style('stroke-width', 2);

      // Dots at each computed point.
      g.selectAll('.dot').data(curve).enter().append('circle').attr('class', 'dot').attr('cx', (c) => xScale(c.rho)).attr('cy', (c) => yScale(c.ic)).attr('r', 2).style('fill', TEAL);
    },
    [curve, crossoverRho, w, isMobile],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={renderRef} width={w} height={HEIGHT} role="img" aria-label="Population irrepresentable condition versus correlation strength" />
      <p
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '13px',
          color: 'var(--color-text-secondary)',
          marginTop: '8px',
        }}
      >
        Population IC quantity for AR(1) Toeplitz designs Σⱼₖ = ρ^|j−k| with contiguous active set S = {`{0, …, ${S - 1}}`} and sign(β*_S) = (1, …, 1). Below the IC = 1 threshold, the lasso is sign-consistent (Wainwright 2009 Theorem 1); above the threshold, the lasso provably fails sign-consistency (Wainwright 2009 Theorem 3) regardless of how λ is chosen — elastic net (§8.2) or adaptive lasso (§8.3) become necessary. At ρ = 0.5 (DGP-1 default) the IC sits comfortably below 1 and the §1 viz showed clean recovery. {crossoverRho !== null ? `Crossover at ρ ≈ ${crossoverRho.toFixed(3)}.` : 'No crossover in this ρ range.'} Computed live in-browser via Cholesky on the s × s = {S} × {S} active-set Gram block.
      </p>
    </div>
  );
}
