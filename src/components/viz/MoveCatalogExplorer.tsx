// =============================================================================
// MoveCatalogExplorer.tsx
//
// §8 of reversible-jump-mcmc. Two panels:
// (A) Richardson-Green split visualization: single N(0,1) source component
//     split into two via sliders u_1, u_2, u_3. Source PDF overlaid with the
//     sum of two new component PDFs (matches by moment preservation).
// (B) Birth/death Jacobian (1 - u_w)^(K-1) on log scale for K=1,3,5,10,30.
//
// Both panels closed-form; sliders are instant.
// Static fallback: /images/topics/reversible-jump-mcmc/fig08_move_catalog.png
// =============================================================================
import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';

function gaussianPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

interface SplitOutput {
  mu1: number;
  mu2: number;
  s2_1: number;
  s2_2: number;
  w1: number;
  w2: number;
  logDetJ: number;
}
function rgSplit(muJ: number, s2J: number, wJ: number, u1: number, u2: number, u3: number): SplitOutput {
  const sigmaJ = Math.sqrt(s2J);
  const w1 = u1 * wJ;
  const w2 = (1 - u1) * wJ;
  const mu1 = muJ - u2 * sigmaJ * Math.sqrt(w2 / w1);
  const mu2 = muJ + u2 * sigmaJ * Math.sqrt(w1 / w2);
  const s2_1 = u3 * (1 - u2 * u2) * s2J * (wJ / w1);
  const s2_2 = (1 - u3) * (1 - u2 * u2) * s2J * (wJ / w2);
  // log |det J| = log w_j + log|μ_2 − μ_1| + log s2_1 + log s2_2 - log s2_J - log u_2 - log(1−u_2²) - log u_3 - log(1−u_3)
  const logDetJ =
    Math.log(wJ) +
    Math.log(Math.abs(mu2 - mu1)) +
    Math.log(s2_1) +
    Math.log(s2_2) -
    Math.log(s2J) -
    Math.log(u2) -
    Math.log(1 - u2 * u2) -
    Math.log(u3) -
    Math.log(1 - u3);
  return { mu1, mu2, s2_1, s2_2, w1, w2, logDetJ };
}

export default function MoveCatalogExplorer() {
  const [u1, setU1] = useState(0.5);
  const [u2, setU2] = useState(0.5);
  const [u3, setU3] = useState(0.5);
  const [K, setK] = useState(10);

  const split = useMemo(() => rgSplit(0, 1, 1, u1, u2, u3), [u1, u2, u3]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const W = containerWidth || 800;
  const isMobile = W < 640;
  const panelW = isMobile ? W : (W - 24) / 2;
  const panelH = 300;

  const splitRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 20, right: 18, bottom: 32, left: 40 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const xRange: [number, number] = [-5, 5];
      const xS = d3.scaleLinear().domain(xRange).range([0, w]);
      const xs = d3.range(xRange[0], xRange[1], 0.05);
      const srcPdf = xs.map((x) => gaussianPdf(x, 0, 1));
      const c1Pdf = xs.map((x) => split.w1 * gaussianPdf(x, split.mu1, Math.sqrt(split.s2_1)));
      const c2Pdf = xs.map((x) => split.w2 * gaussianPdf(x, split.mu2, Math.sqrt(split.s2_2)));
      const sumPdf = xs.map((_, i) => c1Pdf[i] + c2Pdf[i]);

      const yMax = Math.max(...srcPdf, ...sumPdf, ...c1Pdf, ...c2Pdf) * 1.1;
      const yS = d3.scaleLinear().domain([0, yMax]).range([h, 0]);

      const lineGen = (ys: number[]) =>
        d3
          .line<number>()
          .x((_, i) => xS(xs[i]))
          .y((y) => yS(y))
          .curve(d3.curveMonotoneX)(ys) ?? '';

      // Source: dashed
      g.append('path')
        .attr('d', lineGen(srcPdf))
        .style('fill', 'none')
        .style('stroke', 'var(--color-text-secondary)')
        .style('stroke-width', 1.5)
        .style('stroke-dasharray', '4 3');
      // Children
      g.append('path').attr('d', lineGen(c1Pdf)).style('fill', 'var(--color-accent)').style('opacity', 0.25).style('stroke', 'var(--color-accent)').style('stroke-width', 1);
      g.append('path').attr('d', lineGen(c2Pdf)).style('fill', 'var(--color-accent)').style('opacity', 0.25).style('stroke', 'var(--color-accent)').style('stroke-width', 1);
      // Sum: solid
      g.append('path').attr('d', lineGen(sumPdf)).style('fill', 'none').style('stroke', 'var(--color-accent)').style('stroke-width', 2);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xS).ticks(6)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(yS).ticks(5, '.2f')).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('text').attr('x', w / 2).attr('y', h + 28).attr('text-anchor', 'middle').style('font-size', '12px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text(`split of N(0,1): w₁=${split.w1.toFixed(2)}, w₂=${split.w2.toFixed(2)}`);
    },
    [split, panelW, panelH],
  );

  const jacRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelW <= 0) return;
      const M = { top: 20, right: 18, bottom: 36, left: 52 };
      const w = panelW - M.left - M.right;
      const h = panelH - M.top - M.bottom;
      const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);
      const us = d3.range(0.01, 1.0, 0.01);
      const Ks = [1, 3, 5, 10, 30, 50];
      const xS = d3.scaleLinear().domain([0, 1]).range([0, w]);
      const yS = d3.scaleLog().domain([1e-17, 2]).range([h, 0]).clamp(true);

      // Underflow horizon
      g.append('line')
        .attr('x1', 0)
        .attr('x2', w)
        .attr('y1', yS(1e-15))
        .attr('y2', yS(1e-15))
        .style('stroke', 'var(--color-text-secondary)')
        .style('stroke-dasharray', '3 3');
      g.append('text')
        .attr('x', w - 4)
        .attr('y', yS(1e-15) - 4)
        .attr('text-anchor', 'end')
        .style('font-size', '10px')
        .style('font-family', 'var(--font-sans)')
        .style('fill', 'var(--color-text-secondary)')
        .text('float64 underflow ≈ 10⁻¹⁵');

      const colors = d3.schemeTableau10;
      Ks.forEach((Kk, idx) => {
        const pts = us.map((u) => Math.max(Math.pow(1 - u, Kk - 1), 1e-300));
        const line = d3
          .line<number>()
          .x((_, i) => xS(us[i]))
          .y((y) => yS(y));
        g.append('path').datum(pts).attr('d', line).style('fill', 'none').style('stroke', colors[idx % colors.length]).style('stroke-width', Kk === K ? 2.5 : 1.2);
        g.append('text')
          .attr('x', xS(0.85))
          .attr('y', yS(Math.pow(1 - 0.85, Kk - 1) || 1e-300))
          .style('font-size', '10px')
          .style('font-family', 'var(--font-mono)')
          .style('fill', colors[idx % colors.length])
          .text(`K=${Kk}`);
      });
      // Current u_w marker (use u_1 as proxy demonstration)
      const valHere = Math.pow(1 - 0.5, K - 1);
      g.append('circle').attr('cx', xS(0.5)).attr('cy', yS(Math.max(valHere, 1e-17))).attr('r', 4).style('fill', 'var(--color-accent)');
      g.append('text')
        .attr('x', xS(0.5) + 6)
        .attr('y', yS(Math.max(valHere, 1e-17)) - 8)
        .style('font-size', '11px')
        .style('font-family', 'var(--font-mono)')
        .style('fill', 'var(--color-accent)')
        .text(`K=${K}, u_w=0.5: ${valHere.toExponential(2)}`);

      g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xS).ticks(6)).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('g').call(d3.axisLeft(yS).ticks(7, '~e')).selectAll('text').style('fill', 'var(--color-text-secondary)');
      g.append('text').attr('x', w / 2).attr('y', h + 30).attr('text-anchor', 'middle').style('font-size', '12px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('u_w  (Beta(1, K) auxiliary)');
      g.append('text').attr('x', -h / 2).attr('y', -42).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('font-size', '12px').style('font-family', 'var(--font-sans)').style('fill', 'var(--color-text)').text('|det J_birth| = (1 - u_w)^(K-1)');
    },
    [K, panelW, panelH],
  );

  const sliderRow = (
    label: string,
    value: number,
    setVal: (n: number) => void,
    min: number,
    max: number,
    step: number,
    fmt: (n: number) => string,
    aria: string,
  ) => (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        marginBottom: 6,
        fontFamily: 'var(--font-sans)',
        fontSize: 12,
        color: 'var(--color-text)',
      }}
    >
      <label style={{ minWidth: 80 }}>
        {label} = <strong>{fmt(value)}</strong>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => setVal(parseFloat(e.target.value))}
        style={{ flex: 1 }}
        aria-label={aria}
      />
    </div>
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
        <div style={{ width: isMobile ? '100%' : panelW }}>
          {sliderRow('u₁', u1, setU1, 0.05, 0.95, 0.01, (n) => n.toFixed(2), 'Split auxiliary u_1')}
          {sliderRow('u₂', u2, setU2, 0.05, 0.95, 0.01, (n) => n.toFixed(2), 'Split auxiliary u_2')}
          {sliderRow('u₃', u3, setU3, 0.05, 0.95, 0.01, (n) => n.toFixed(2), 'Split auxiliary u_3')}
          <div style={{ marginBottom: 6, fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--color-text-secondary)' }}>
            μ₁={split.mu1.toFixed(2)}, μ₂={split.mu2.toFixed(2)}, σ₁²={split.s2_1.toFixed(2)}, σ₂²={split.s2_2.toFixed(2)} · log|det J_split|={split.logDetJ.toFixed(2)}
          </div>
          <svg ref={splitRef} width={panelW} height={panelH} />
        </div>
        <div style={{ width: isMobile ? '100%' : panelW }}>
          {sliderRow('K', K, (v) => setK(Math.round(v)), 1, 50, 1, (n) => n.toFixed(0), 'Current mixture order K')}
          <svg ref={jacRef} width={panelW} height={panelH} />
        </div>
      </div>
    </div>
  );
}
