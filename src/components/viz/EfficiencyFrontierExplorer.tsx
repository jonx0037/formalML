import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  RUNNING_EXAMPLE_A,
  RUNNING_EXAMPLE_THETA0,
  confidenceEllipse2D,
  det2x2,
  efficientWeight,
  hansenBound,
  identityMat,
  mulberry32,
  sampleWishart,
  sandwichVariance,
  trace2x2,
  type Mat,
} from './shared/generalized-method-of-moments';

// -----------------------------------------------------------------------------
// Viz 6.1 — Efficiency frontier explorer.
//
// Two panels:
//   Panel A — Scatter of (log det V_W, tr V_W) over M=200 Wishart-sampled
//     random positive-definite weighting matrices. V* sits at the
//     southwest extreme (minimum both axes by Loewner — det and tr are both
//     order-preserving on positive-definite matrices in the Loewner sense).
//     A few highlighted markers: V_I (identity weighting), V*, V_diag.
//   Panel B — Nested ellipses for the interpolation W_α = (1-α) I + α Ω⁻¹.
//     As α slides from 0 (identity) to 1 (efficient), the ellipse shrinks
//     monotonically by Hansen's theorem. Reader controls α.
//
// Closed-form throughout — no MC except the Wishart draws on Panel A.
// -----------------------------------------------------------------------------

const M_WISHART = 200;
const N_FIXED = 200;
const PANEL_H = 320;
const ELLIPSE_H = 320;
const MARGIN = { top: 22, right: 18, bottom: 42, left: 56 };

const COLOR_RANDOM = '#9aa3ad';
const COLOR_IDENT = '#5b6b78';
const COLOR_EFF = 'var(--color-accent)';
const COLOR_DIAG = '#d09a3e';

function buildOmega(sigma3: number): Mat {
  const sigma = [1, 1, sigma3, 1];
  const Omega: Mat = identityMat(4);
  for (let i = 0; i < 4; i++) Omega[i][i] = sigma[i] * sigma[i];
  return Omega;
}

interface FrontierPoint {
  logDet: number;
  trace: number;
}

interface FrontierState {
  random: FrontierPoint[];
  Videnti: FrontierPoint;
  Vdiag: FrontierPoint;
  Vstar: FrontierPoint;
  alphaEllipses: { alpha: number; ellipse: { rx: number; ry: number; angle: number } }[];
}

function computeFrontier(sigma3: number, seed: number, alpha: number): FrontierState {
  const A = RUNNING_EXAMPLE_A;
  const G: Mat = A.map((row) => row.map((v) => -v));
  const Omega = buildOmega(sigma3);
  const Wstar = efficientWeight(Omega);
  const Vstar = hansenBound(G, Omega);

  // Random PD W's: Wishart with df=8, scale=identity.
  const rng = mulberry32(seed);
  const random: FrontierPoint[] = [];
  for (let m = 0; m < M_WISHART; m++) {
    const W = sampleWishart(rng, 8, identityMat(4));
    const V = sandwichVariance(G, W, Omega);
    const d = det2x2(V);
    if (!isFinite(d) || d <= 0) continue;
    random.push({ logDet: Math.log(d), trace: trace2x2(V) });
  }
  const Videnti = (() => {
    const V = sandwichVariance(G, identityMat(4), Omega);
    return { logDet: Math.log(det2x2(V)), trace: trace2x2(V) };
  })();
  const Vdiag = (() => {
    const W: Mat = identityMat(4).map((row, i) => row.map((v, j) => (i === j ? 1 / Omega[i][i] : 0)));
    const V = sandwichVariance(G, W, Omega);
    return { logDet: Math.log(det2x2(V)), trace: trace2x2(V) };
  })();
  const VstarPt = { logDet: Math.log(det2x2(Vstar)), trace: trace2x2(Vstar) };

  // α interpolation ellipses (independent of seed).
  const alphaGrid = [0, 0.25, 0.5, 0.75, 1.0];
  const alphaEllipses = alphaGrid.map((a) => {
    const W: Mat = identityMat(4).map((row, i) => row.map((v, j) => (1 - a) * (i === j ? 1 : 0) + a * Wstar[i][j]));
    const V = sandwichVariance(G, W, Omega);
    const Vscaled: Mat = V.map((r) => r.map((v) => v / N_FIXED));
    return { alpha: a, ellipse: confidenceEllipse2D(Vscaled, 0.05) };
  });
  // Also compute the user-controlled ellipse.
  const Wa: Mat = identityMat(4).map((row, i) => row.map((v, j) => (1 - alpha) * (i === j ? 1 : 0) + alpha * Wstar[i][j]));
  const Va = sandwichVariance(G, Wa, Omega);
  const VaScaled: Mat = Va.map((r) => r.map((v) => v / N_FIXED));
  alphaEllipses.push({ alpha, ellipse: confidenceEllipse2D(VaScaled, 0.05) });

  return { random, Videnti, Vdiag, Vstar: VstarPt, alphaEllipses };
}

export default function EfficiencyFrontierExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [sigma3, setSigma3] = useState(2.0);
  const [alpha, setAlpha] = useState(0.5);
  const [seed, setSeed] = useState(7);

  const state = useMemo(() => computeFrontier(sigma3, seed, alpha), [sigma3, seed, alpha]);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const scatterRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xs = state.random.map((p) => p.logDet).concat([state.Videnti.logDet, state.Vdiag.logDet, state.Vstar.logDet]);
      const ys = state.random.map((p) => p.trace).concat([state.Videnti.trace, state.Vdiag.trace, state.Vstar.trace]);
      const xExt = d3.extent(xs) as [number, number];
      const yExt = d3.extent(ys) as [number, number];
      const padX = (xExt[1] - xExt[0]) * 0.08;
      const padY = (yExt[1] - yExt[0]) * 0.08;
      const xScale = d3.scaleLinear().domain([xExt[0] - padX, xExt[1] + padX]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([yExt[0] - padY, yExt[1] + padY]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0, ${innerH})`).call(d3.axisBottom(xScale).ticks(6)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');

      state.random.forEach((p) => g.append('circle').attr('cx', xScale(p.logDet)).attr('cy', yScale(p.trace)).attr('r', 2.2).style('fill', COLOR_RANDOM).attr('opacity', 0.5));

      // Highlighted points.
      const marks: [FrontierPoint, string, string][] = [
        [state.Videnti, COLOR_IDENT, 'V_I'],
        [state.Vdiag, COLOR_DIAG, 'V_diag'],
        [state.Vstar, COLOR_EFF, 'V* (Hansen bound)'],
      ];
      marks.forEach(([pt, color, label]) => {
        g.append('circle').attr('cx', xScale(pt.logDet)).attr('cy', yScale(pt.trace)).attr('r', 6).style('fill', color).style('stroke', 'var(--color-bg)').style('stroke-width', 1.5);
        g.append('text').attr('x', xScale(pt.logDet) + 9).attr('y', yScale(pt.trace) + 4).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', color).text(label);
      });

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('log det V_W (uncertainty volume)');
      g.append('text').attr('x', -innerH / 2).attr('y', -42).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('tr V_W (total uncertainty)');
    },
    [state, w],
  );

  const ellipseRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = ELLIPSE_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const allRx = state.alphaEllipses.map((a) => a.ellipse.rx);
      const allRy = state.alphaEllipses.map((a) => a.ellipse.ry);
      const span = Math.max(Math.max(...allRx), Math.max(...allRy)) * 1.3;
      const xScale = d3.scaleLinear().domain([RUNNING_EXAMPLE_THETA0[0] - span, RUNNING_EXAMPLE_THETA0[0] + span]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([RUNNING_EXAMPLE_THETA0[1] - span, RUNNING_EXAMPLE_THETA0[1] + span]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format('.3~f'))).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.3~f'))).selectAll('text').style('fill', 'var(--color-text)');

      const x0 = xScale(RUNNING_EXAMPLE_THETA0[0]);
      const y0 = yScale(RUNNING_EXAMPLE_THETA0[1]);

      // Draw nested ellipses for the grid first (fading).
      const grid = state.alphaEllipses.filter((e) => e.alpha !== alpha);
      const color = d3.scaleLinear<string>().domain([0, 1]).range([COLOR_IDENT, COLOR_EFF as string]);
      grid.forEach((entry) => {
        const rx = Math.abs(xScale(RUNNING_EXAMPLE_THETA0[0] + entry.ellipse.rx) - x0);
        const ry = Math.abs(yScale(RUNNING_EXAMPLE_THETA0[1] + entry.ellipse.ry) - y0);
        const angleDeg = (-entry.ellipse.angle * 180) / Math.PI;
        g.append('ellipse').attr('cx', x0).attr('cy', y0).attr('rx', rx).attr('ry', ry).attr('transform', `rotate(${angleDeg}, ${x0}, ${y0})`).attr('fill', 'none').style('stroke', color(entry.alpha)).attr('stroke-width', 1).attr('opacity', 0.45);
      });
      // User's ellipse (last entry) drawn boldly on top.
      const user = state.alphaEllipses[state.alphaEllipses.length - 1];
      const rxU = Math.abs(xScale(RUNNING_EXAMPLE_THETA0[0] + user.ellipse.rx) - x0);
      const ryU = Math.abs(yScale(RUNNING_EXAMPLE_THETA0[1] + user.ellipse.ry) - y0);
      const angleDegU = (-user.ellipse.angle * 180) / Math.PI;
      g.append('ellipse').attr('cx', x0).attr('cy', y0).attr('rx', rxU).attr('ry', ryU).attr('transform', `rotate(${angleDegU}, ${x0}, ${y0})`).attr('fill', 'none').style('stroke', color(alpha)).attr('stroke-width', 2.5);

      // θ₀ cross.
      g.append('line').attr('x1', x0 - 7).attr('x2', x0 + 7).attr('y1', y0).attr('y2', y0).style('stroke', 'var(--color-text)').style('stroke-width', 1.4);
      g.append('line').attr('x1', x0).attr('x2', x0).attr('y1', y0 - 7).attr('y2', y0 + 7).style('stroke', 'var(--color-text)').style('stroke-width', 1.4);

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('W_α = (1−α) I + α Ω⁻¹ — 95% ellipses nest as α → 1');
      g.append('text').attr('x', -innerH / 2).attr('y', -42).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('θ_2');
    },
    [state, w, alpha],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>σ_3 = <strong>{sigma3.toFixed(2)}</strong></span>
          <input type="range" min={0.3} max={4.0} step={0.1} value={sigma3} aria-label="Sensor 3 noise" onChange={(e) => setSigma3(Number(e.target.value))} />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '220px' }}>
          <span>α = <strong>{alpha.toFixed(2)}</strong><span style={{ color: 'var(--color-text-secondary)', marginLeft: 6 }}>(0 = identity, 1 = efficient)</span></span>
          <input type="range" min={0} max={1} step={0.05} value={alpha} aria-label="Interpolation weight" onChange={(e) => setAlpha(Number(e.target.value))} />
        </label>
        <button type="button" onClick={() => setSeed((s) => s + 1)} style={{ padding: '6px 10px', fontFamily: 'var(--font-mono)', fontSize: '12px', border: '1px solid var(--color-border)', background: 'var(--color-bg)', color: 'var(--color-text)', borderRadius: '4px', cursor: 'pointer', alignSelf: 'flex-end' }}>resample W's</button>
      </div>
      <svg ref={scatterRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} aria-label="Scatter of random W's vs Hansen bound" />
      <svg ref={ellipseRef} width={w} height={ELLIPSE_H} style={{ maxWidth: '100%', display: 'block', marginTop: 8 }} aria-label="Nested ellipses across W_α" />
    </div>
  );
}
