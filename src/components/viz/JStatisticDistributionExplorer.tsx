import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  RUNNING_EXAMPLE_A,
  RUNNING_EXAMPLE_THETA0,
  chi2Pdf,
  chi2Quantile,
  gaussianFrom,
  mulberry32,
  sampleSensorObservations,
  twoStepAffineGMM,
  type Mat,
  type Vec,
} from './shared/generalized-method-of-moments';

// -----------------------------------------------------------------------------
// Viz 8.1 + 8.2 (merged) — Hansen J-statistic distribution and power.
//
// Two panels:
//   Panel A — Empirical histogram of the J-statistic under H₀ over B
//     two-step replicates, overlaid with the χ²_{L-k} = χ²_2 theoretical
//     density. Reader controls n.
//   Panel B — Power curve: rejection rate at α = 0.05 vs misspecification
//     amplitude δ on the third moment (sensor 3's mean is shifted by δ
//     from its θ₀-implied value). Reader controls n and the δ grid.
//
// MC cost: B × (sample n × 4 + twoStep linear-algebra). At B=400, n=300,
// per refresh ≈ 400 × 300 × 4 + 400 × ~200 linalg flops ≈ 600k. Commit-on-
// release for n; the δ slider redraws the power curve cheaply.
// -----------------------------------------------------------------------------

const N_OPTIONS = [50, 100, 200, 500, 1000];
const B_H0 = 400;
const B_POWER = 250;
const DELTA_GRID = [0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0];
const PANEL_H = 300;
const POWER_H = 280;
const MARGIN = { top: 22, right: 18, bottom: 42, left: 56 };

const COLOR_HIST = '#9aa3ad';
const COLOR_DENS = 'var(--color-accent)';
const COLOR_REJ = 'var(--color-accent)';
const COLOR_ALPHA = '#d09a3e';

interface H0Result {
  Js: Vec; // length B_H0
  meanJ: number;
  pctileJ: number; // empirical 95th percentile
  rejRate: number; // empirical fraction > chi²₂(0.95)
}

interface PowerPoint { delta: number; rej: number; }

function runH0(n: number): H0Result {
  const A = RUNNING_EXAMPLE_A;
  const theta0 = RUNNING_EXAMPLE_THETA0;
  const sigma = [1, 1, 1, 1];
  const Js: Vec = [];
  for (let b = 0; b < B_H0; b++) {
    const rng = mulberry32(40000 + b * 11);
    const X = sampleSensorObservations(rng, theta0, A, n, sigma);
    const res = twoStepAffineGMM(X, A);
    Js.push(res.Jhat);
  }
  const sorted = Js.slice().sort((a, b) => a - b);
  const meanJ = Js.reduce((s, v) => s + v, 0) / Js.length;
  const pctileJ = sorted[Math.floor(0.95 * sorted.length)];
  const crit = chi2Quantile(0.95, 2);
  const rejRate = Js.filter((v) => v > crit).length / Js.length;
  return { Js, meanJ, pctileJ, rejRate };
}

function runPower(n: number): PowerPoint[] {
  const A = RUNNING_EXAMPLE_A;
  const theta0 = RUNNING_EXAMPLE_THETA0;
  const sigma = [1, 1, 1, 1];
  const baseMu = [
    1 * theta0[0] + 0 * theta0[1],
    0 * theta0[0] + 1 * theta0[1],
    1 * theta0[0] + 1 * theta0[1],
    2 * theta0[0] - 1 * theta0[1],
  ];
  const crit = chi2Quantile(0.95, 2);
  return DELTA_GRID.map((delta) => {
    let rejCount = 0;
    for (let b = 0; b < B_POWER; b++) {
      const rng = mulberry32(80000 + b * 7 + Math.floor(delta * 1000));
      const gauss = gaussianFrom(rng);
      // Synthesize X directly with sensor-3 shifted by delta.
      const X: Mat = [];
      for (let i = 0; i < n; i++) {
        X.push([
          baseMu[0] + sigma[0] * gauss(),
          baseMu[1] + sigma[1] * gauss(),
          baseMu[2] + delta + sigma[2] * gauss(),
          baseMu[3] + sigma[3] * gauss(),
        ]);
      }
      const res = twoStepAffineGMM(X, A);
      if (res.Jhat > crit) rejCount += 1;
    }
    return { delta, rej: rejCount / B_POWER };
  });
}

export default function JStatisticDistributionExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [nIdx, setNIdx] = useState(2); // n = 200
  const [nIdxDisplay, setNIdxDisplay] = useState(2);

  const n = N_OPTIONS[nIdx];
  const h0 = useMemo(() => runH0(n), [n]);
  const power = useMemo(() => runPower(n), [n]);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const histRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const ext = d3.extent(h0.Js) as [number, number];
      const xMax = Math.max(ext[1], chi2Quantile(0.995, 2));
      const xScale = d3.scaleLinear().domain([0, xMax]).range([0, innerW]);
      const binner = d3.bin().domain([0, xMax]).thresholds(30);
      const bins = binner(h0.Js);
      const counts = bins.map((b) => b.length);
      const widths = bins.map((b) => (b.x1 ?? 0) - (b.x0 ?? 0));
      const densities = counts.map((c, i) => c / (h0.Js.length * widths[i]));
      const maxDens = d3.max(densities) ?? 0;
      const maxPdf = Math.max(maxDens, chi2Pdf(1, 2));
      const yScale = d3.scaleLinear().domain([0, maxPdf * 1.1]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(6)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.2~f'))).selectAll('text').style('fill', 'var(--color-text)');

      // Histogram (density-scaled).
      bins.forEach((b, i) => {
        if (b.x0 === undefined || b.x1 === undefined) return;
        g.append('rect').attr('x', xScale(b.x0)).attr('y', yScale(densities[i])).attr('width', Math.max(0, xScale(b.x1) - xScale(b.x0) - 1)).attr('height', innerH - yScale(densities[i])).style('fill', COLOR_HIST).style('opacity', 0.7);
      });

      // χ²_2 density overlay.
      const dens: [number, number][] = [];
      const M = 200;
      for (let i = 1; i <= M; i++) {
        const x = (i / M) * xMax;
        dens.push([x, chi2Pdf(x, 2)]);
      }
      const lineGen = d3.line<[number, number]>().x((d) => xScale(d[0])).y((d) => yScale(d[1])).curve(d3.curveBasis);
      g.append('path').attr('d', lineGen(dens)).attr('fill', 'none').style('stroke', COLOR_DENS).attr('stroke-width', 2);

      // 95% critical line.
      const crit = chi2Quantile(0.95, 2);
      g.append('line').attr('x1', xScale(crit)).attr('x2', xScale(crit)).attr('y1', 0).attr('y2', innerH).style('stroke', COLOR_ALPHA).attr('stroke-dasharray', '4 3').style('stroke-width', 1.2);
      g.append('text').attr('x', xScale(crit) + 4).attr('y', 12).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_ALPHA).text(`χ²₂(0.95) = ${crit.toFixed(2)}`);

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text(`Ĵ (empirical density across ${B_H0} replicates at n = ${n})`);
      g.append('text').attr('x', -innerH / 2).attr('y', -44).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('density');
    },
    [h0, w, n],
  );

  const powerRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = POWER_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLinear().domain([0, DELTA_GRID[DELTA_GRID.length - 1]]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0, 1.05]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(6).tickFormat(d3.format('.2~f'))).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.0%'))).selectAll('text').style('fill', 'var(--color-text)');

      // α reference line.
      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yScale(0.05)).attr('y2', yScale(0.05)).style('stroke', COLOR_ALPHA).attr('stroke-dasharray', '4 3').style('stroke-width', 1);
      g.append('text').attr('x', innerW - 4).attr('y', yScale(0.05) - 4).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_ALPHA).text('α = 0.05');

      const pts: [number, number][] = power.map((p) => [p.delta, p.rej]);
      const lineGen = d3.line<[number, number]>().x((d) => xScale(d[0])).y((d) => yScale(d[1])).curve(d3.curveCatmullRom);
      g.append('path').attr('d', lineGen(pts)).attr('fill', 'none').style('stroke', COLOR_REJ).attr('stroke-width', 2.2);
      pts.forEach((p) => g.append('circle').attr('cx', xScale(p[0])).attr('cy', yScale(p[1])).attr('r', 3.5).style('fill', COLOR_REJ));

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 32).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text(`misspecification δ on g₃ at n = ${n}`);
      g.append('text').attr('x', -innerH / 2).attr('y', -44).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('rejection rate at α = 0.05');
    },
    [power, w, n],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '220px' }}>
          <span>n = <strong>{N_OPTIONS[nIdxDisplay]}</strong></span>
          <input type="range" min={0} max={N_OPTIONS.length - 1} step={1} value={nIdxDisplay} aria-label="Sample size" onChange={(e) => setNIdxDisplay(Number(e.target.value))} onMouseUp={(e) => setNIdx(Number((e.target as HTMLInputElement).value))} onTouchEnd={(e) => setNIdx(Number((e.target as HTMLInputElement).value))} onKeyUp={(e) => setNIdx(Number((e.target as HTMLInputElement).value))} />
        </label>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px', alignSelf: 'flex-end' }}>
          mean Ĵ = <strong style={{ color: 'var(--color-text)' }}>{h0.meanJ.toFixed(2)}</strong>
          {' '}(theory L−k = 2) · empirical Type-I = <strong style={{ color: 'var(--color-text)' }}>{(h0.rejRate * 100).toFixed(1)}%</strong>
        </div>
      </div>
      <svg ref={histRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} aria-label="Empirical J distribution vs χ²_{L-k}" />
      <svg ref={powerRef} width={w} height={POWER_H} style={{ maxWidth: '100%', display: 'block', marginTop: 8 }} aria-label="J-test power curve" />
    </div>
  );
}
