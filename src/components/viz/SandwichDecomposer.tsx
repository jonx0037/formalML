import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  RUNNING_EXAMPLE_A,
  RUNNING_EXAMPLE_THETA0,
  confidenceEllipse2D,
  efficientWeight,
  hansenBound,
  identityMat,
  matMul,
  matTranspose,
  operatorNorm,
  sandwichVariance,
  type Mat,
} from './shared/generalized-method-of-moments';

// -----------------------------------------------------------------------------
// Viz 5.1 — Sandwich variance decomposer.
//
// Two panels:
//   Panel A — 95% confidence ellipses for three weighting matrices (identity,
//     diagonal-inverse-variance, efficient Ω⁻¹) overlaid in (θ_1, θ_2) space.
//     The ellipses nest by the Loewner ordering: efficient ⪯ diagonal ⪯
//     identity. Reader controls per-sensor heteroskedasticity (σ_3).
//   Panel B — bar chart of the operator norms ‖G' W G‖, ‖G' WΩW G‖, ‖V_W‖
//     for each weighting. Reader sees how the "bread × meat × bread" pieces
//     trade off against each other across W choices.
//
// Closed-form math throughout — no MC, no commit-on-release. n is fixed at
// 200 to scale the ellipse sizes; the user's experiment is about the *shape*
// of the ellipses across weightings, not the scale.
// -----------------------------------------------------------------------------

const N_FIXED = 200;
const PANEL_H = 320;
const BAR_H = 200;
const MARGIN = { top: 22, right: 18, bottom: 42, left: 56 };

const COLOR_IDENT = '#9aa3ad';
const COLOR_DIAG = '#d09a3e';
const COLOR_EFF = 'var(--color-accent)';

function buildOmega(sigma3: number): Mat {
  const sigma = [1, 1, sigma3, 1];
  const Omega: Mat = identityMat(4);
  for (let i = 0; i < 4; i++) Omega[i][i] = sigma[i] * sigma[i];
  return Omega;
}

interface Decomposition {
  V: Mat;
  bread: Mat;
  meat: Mat;
  breadNorm: number;
  meatNorm: number;
  VNorm: number;
  ellipse: { rx: number; ry: number; angle: number };
}

function decompose(W: Mat, G: Mat, Omega: Mat): Decomposition {
  const Gt = matTranspose(G);
  const GtW = matMul(Gt, W);
  const bread = matMul(GtW, G);
  const meatRight = matMul(W, G);
  const meat = matMul(GtW, matMul(Omega, meatRight));
  const V = sandwichVariance(G, W, Omega);
  const VScaled: Mat = V.map((r) => r.map((v) => v / N_FIXED));
  return {
    V,
    bread,
    meat,
    breadNorm: operatorNorm(bread),
    meatNorm: operatorNorm(meat),
    VNorm: operatorNorm(V),
    ellipse: confidenceEllipse2D(VScaled, 0.05),
  };
}

export default function SandwichDecomposer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [sigma3, setSigma3] = useState(2.0);

  const state = useMemo(() => {
    const A = RUNNING_EXAMPLE_A;
    const G: Mat = A.map((row) => row.map((v) => -v));
    const Omega = buildOmega(sigma3);
    const Wident = identityMat(4);
    const Wdiag: Mat = identityMat(4).map((row, i) => row.map((v, j) => (i === j ? 1 / Omega[i][i] : 0)));
    const Weff = efficientWeight(Omega);
    return {
      ident: decompose(Wident, G, Omega),
      diag: decompose(Wdiag, G, Omega),
      eff: decompose(Weff, G, Omega),
    };
  }, [sigma3]);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const ellipseRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const maxRx = Math.max(state.ident.ellipse.rx, state.diag.ellipse.rx, state.eff.ellipse.rx);
      const maxRy = Math.max(state.ident.ellipse.ry, state.diag.ellipse.ry, state.eff.ellipse.ry);
      const span = Math.max(maxRx, maxRy) * 1.3;

      const xScale = d3.scaleLinear().domain([RUNNING_EXAMPLE_THETA0[0] - span, RUNNING_EXAMPLE_THETA0[0] + span]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([RUNNING_EXAMPLE_THETA0[1] - span, RUNNING_EXAMPLE_THETA0[1] + span]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format('.3~f'))).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.3~f'))).selectAll('text').style('fill', 'var(--color-text)');

      const x0 = xScale(RUNNING_EXAMPLE_THETA0[0]);
      const y0 = yScale(RUNNING_EXAMPLE_THETA0[1]);
      const drawEllipse = (ell: { rx: number; ry: number; angle: number }, color: string, label: string, dy: number) => {
        const rx = Math.abs(xScale(RUNNING_EXAMPLE_THETA0[0] + ell.rx) - x0);
        const ry = Math.abs(yScale(RUNNING_EXAMPLE_THETA0[1] + ell.ry) - y0);
        const angleDeg = (-ell.angle * 180) / Math.PI;
        g.append('ellipse').attr('cx', x0).attr('cy', y0).attr('rx', rx).attr('ry', ry).attr('transform', `rotate(${angleDeg}, ${x0}, ${y0})`).attr('fill', 'none').style('stroke', color).attr('stroke-width', 2);
        g.append('text').attr('x', x0 + rx + 4).attr('y', y0 + dy).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', color).text(label);
      };
      drawEllipse(state.ident.ellipse, COLOR_IDENT, 'W = I', -6);
      drawEllipse(state.diag.ellipse, COLOR_DIAG, 'W = diag(1/σ²)', 10);
      drawEllipse(state.eff.ellipse, COLOR_EFF, 'W = Ω⁻¹', 26);

      // θ₀ cross.
      g.append('line').attr('x1', x0 - 7).attr('x2', x0 + 7).attr('y1', y0).attr('y2', y0).style('stroke', 'var(--color-text)').style('stroke-width', 1.4);
      g.append('line').attr('x1', x0).attr('x2', x0).attr('y1', y0 - 7).attr('y2', y0 + 7).style('stroke', 'var(--color-text)').style('stroke-width', 1.4);

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text(`95% confidence ellipses for θ̂_W at n = ${N_FIXED}`);
      g.append('text').attr('x', -innerH / 2).attr('y', -42).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('θ_2');
    },
    [state, w],
  );

  const barRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = BAR_H - 30;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},10)`);

      const groups: [string, string, Decomposition][] = [
        ['identity', COLOR_IDENT, state.ident],
        ['diagonal', COLOR_DIAG, state.diag],
        ['efficient', COLOR_EFF, state.eff],
      ];
      const metrics: [string, (d: Decomposition) => number][] = [
        ['‖bread‖', (d) => d.breadNorm],
        ['‖meat‖', (d) => d.meatNorm],
        ['‖V_W‖', (d) => d.VNorm],
      ];
      const allValues = groups.flatMap(([, , d]) => metrics.map(([, fn]) => fn(d)));
      const yMax = Math.max(...allValues) * 1.15;
      const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

      const groupScale = d3.scaleBand<string>().domain(metrics.map(([m]) => m)).range([0, innerW]).padding(0.3);
      const subScale = d3.scaleBand<string>().domain(groups.map(([n]) => n)).range([0, groupScale.bandwidth()]).padding(0.1);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(groupScale)).selectAll('text').style('fill', 'var(--color-text)').style('font-family', 'var(--font-mono)').style('font-size', '11px');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');

      metrics.forEach(([m, fn]) => {
        const gx = groupScale(m) ?? 0;
        groups.forEach(([name, color, dec]) => {
          const sx = subScale(name) ?? 0;
          const val = fn(dec);
          g.append('rect').attr('x', gx + sx).attr('y', yScale(val)).attr('width', subScale.bandwidth()).attr('height', innerH - yScale(val)).style('fill', color).style('opacity', 0.85);
        });
      });

      // Legend.
      const legend = g.append('g').attr('transform', `translate(${innerW - 200}, 4)`);
      groups.forEach(([name, color], i) => {
        legend.append('rect').attr('x', i * 70).attr('width', 12).attr('height', 12).style('fill', color);
        legend.append('text').attr('x', i * 70 + 18).attr('y', 10).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text(name);
      });
    },
    [state, w],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '240px' }}>
          <span>
            σ_3 = <strong>{sigma3.toFixed(2)}</strong>
            <span style={{ color: 'var(--color-text-secondary)', marginLeft: 6 }}>(heteroskedasticity on sensor 3)</span>
          </span>
          <input type="range" min={0.3} max={4.0} step={0.1} value={sigma3} aria-label="Sensor 3 noise" onChange={(e) => setSigma3(Number(e.target.value))} />
        </label>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px', alignSelf: 'flex-end' }}>
          ‖V_I‖/‖V*‖ = <strong style={{ color: 'var(--color-text)' }}>{(state.ident.VNorm / state.eff.VNorm).toFixed(2)}×</strong>
          {' '}— efficient sandwich tightens by this factor on the largest direction.
        </div>
      </div>
      <svg ref={ellipseRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} aria-label="Nested 95% confidence ellipses" />
      <svg ref={barRef} width={w} height={BAR_H} style={{ maxWidth: '100%', display: 'block', marginTop: 8 }} aria-label="Bread, meat, sandwich operator norms" />
    </div>
  );
}
