import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32, gaussianFrom, inverseNormalCDF } from './shared/always-valid-inference';

// -----------------------------------------------------------------------------
// Viz 3.1 — SPRT trajectory explorer.
//
// Top panel: log Λ_n for 30 H_0 (blue) and 30 H_1 (red) trajectories, with
//            horizontal boundary lines at log A and log B. Stopped paths
//            truncated at τ and marked with a dot (red dot = reject, blue dot
//            = accept).
// Bottom panel: histogram of τ under each hypothesis, with vertical Wald-
//               prediction lines for E[τ | H_0] and E[τ | H_1].
//
// Sliders (commit-on-release): μ_1, α, β. Boundaries A = (1−β)/α, B = β/(1−α).
// -----------------------------------------------------------------------------

const SIGMA = 1;
const MU0 = 0;
const N_MAX_INTERNAL = 5000;
const B_REPLICATES = 1000;
const N_PATHS_SHOWN = 30;
const ERROR_OPTIONS = [0.01, 0.025, 0.05, 0.10];

const TOP_H = 280;
const BOT_H = 220;
const MARGIN = { top: 28, right: 16, bottom: 36, left: 56 };

const COLOR_H0 = '#356fa3';
const COLOR_H1 = '#c44545';
const COLOR_BOUND = 'var(--color-accent)';

interface SPRTRun {
  pathsH0: Float64Array[];
  pathsH1: Float64Array[];
  stopH0: Int32Array; // -1 = censored at N_MAX; else τ
  stopH1: Int32Array;
  decisionH0: Int8Array; // 1 = rejected H_0, 0 = accepted, -1 = unstopped
  decisionH1: Int8Array;
  meanTauH0: number;
  meanTauH1: number;
  waldTau: number;
  empRejectH0: number;
  empAcceptH1: number;
  nNP: number;
  logA: number;
  logB: number;
}

function runSPRT(mu1: number, alpha: number, beta: number): SPRTRun {
  const A = (1 - beta) / alpha;
  const B = beta / (1 - alpha);
  const logA = Math.log(A);
  const logB = Math.log(B);
  // Per-step log-LR increment: Z_i = (μ_1 - μ_0)/σ² · (X_i − (μ_0 + μ_1)/2).
  const slope = (mu1 - MU0) / (SIGMA * SIGMA);
  const center = (MU0 + mu1) / 2;

  const pathsH0: Float64Array[] = [];
  const pathsH1: Float64Array[] = [];
  const stopH0 = new Int32Array(B_REPLICATES);
  const stopH1 = new Int32Array(B_REPLICATES);
  const decisionH0 = new Int8Array(B_REPLICATES);
  const decisionH1 = new Int8Array(B_REPLICATES);

  const rng0 = mulberry32(20260701);
  const rng1 = mulberry32(20260702);
  const g0 = gaussianFrom(rng0);
  const g1 = gaussianFrom(rng1);

  let sumTauH0 = 0;
  let sumTauH1 = 0;
  let countTauH0 = 0;
  let countTauH1 = 0;
  let rejectH0 = 0;
  let acceptH1 = 0;

  for (let b = 0; b < B_REPLICATES; b++) {
    const keep = b < N_PATHS_SHOWN;
    const p0 = keep ? new Float64Array(N_MAX_INTERNAL) : null;
    const p1 = keep ? new Float64Array(N_MAX_INTERNAL) : null;
    let log0 = 0;
    let log1 = 0;
    let tau0 = -1;
    let tau1 = -1;
    let dec0: 0 | 1 = 0;
    let dec1: 0 | 1 = 0;
    for (let n = 1; n <= N_MAX_INTERNAL; n++) {
      const x0 = MU0 + SIGMA * g0();
      const x1 = mu1 + SIGMA * g1();
      log0 += slope * (x0 - center);
      log1 += slope * (x1 - center);
      if (p0) p0[n - 1] = log0;
      if (p1) p1[n - 1] = log1;
      if (tau0 === -1) {
        if (log0 >= logA) {
          tau0 = n;
          dec0 = 1;
        } else if (log0 <= logB) {
          tau0 = n;
          dec0 = 0;
        }
      }
      if (tau1 === -1) {
        if (log1 >= logA) {
          tau1 = n;
          dec1 = 1;
        } else if (log1 <= logB) {
          tau1 = n;
          dec1 = 0;
        }
      }
      if (tau0 !== -1 && tau1 !== -1 && !keep) break;
    }
    stopH0[b] = tau0;
    stopH1[b] = tau1;
    decisionH0[b] = tau0 === -1 ? -1 : dec0;
    decisionH1[b] = tau1 === -1 ? -1 : dec1;
    if (p0) pathsH0.push(p0);
    if (p1) pathsH1.push(p1);
    if (tau0 > 0) {
      sumTauH0 += tau0;
      countTauH0++;
      if (dec0 === 1) rejectH0++;
    }
    if (tau1 > 0) {
      sumTauH1 += tau1;
      countTauH1++;
      if (dec1 === 0) acceptH1++;
    }
  }

  // Wald's identity for E[τ | H_0]
  const KL = ((mu1 - MU0) * (mu1 - MU0)) / (2 * SIGMA * SIGMA); // symmetric KL for equal-variance Gaussians
  const waldTau = (alpha * logA + (1 - alpha) * logB) / -KL;

  // Fixed-n NP sample size at (α, β) with effect μ_1 - μ_0:
  const z_alpha = inverseNormalCDF(1 - alpha);
  const z_beta = inverseNormalCDF(1 - beta);
  const nNP = Math.pow((z_alpha + z_beta) * SIGMA / (mu1 - MU0), 2);

  return {
    pathsH0,
    pathsH1,
    stopH0,
    stopH1,
    decisionH0,
    decisionH1,
    meanTauH0: countTauH0 > 0 ? sumTauH0 / countTauH0 : NaN,
    meanTauH1: countTauH1 > 0 ? sumTauH1 / countTauH1 : NaN,
    waldTau,
    empRejectH0: rejectH0 / B_REPLICATES,
    empAcceptH1: acceptH1 / B_REPLICATES,
    nNP,
    logA,
    logB,
  };
}

export default function SPRTTrajectoryExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayMu, setDisplayMu] = useState(0.3);
  const [committedMu, setCommittedMu] = useState(0.3);
  const [alphaIdx, setAlphaIdx] = useState(2); // 0.05
  const [betaIdx, setBetaIdx] = useState(2); // 0.05

  const alpha = ERROR_OPTIONS[alphaIdx];
  const beta = ERROR_OPTIONS[betaIdx];

  const sim = useMemo(() => runSPRT(committedMu, alpha, beta), [committedMu, alpha, beta]);

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  // Top panel
  const topRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = TOP_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      // x: τ_max = max stop over visible paths, capped at N_MAX_INTERNAL
      let maxN = 50;
      for (let i = 0; i < sim.pathsH0.length; i++) maxN = Math.max(maxN, sim.stopH0[i] > 0 ? sim.stopH0[i] : 0);
      for (let i = 0; i < sim.pathsH1.length; i++) maxN = Math.max(maxN, sim.stopH1[i] > 0 ? sim.stopH1[i] : 0);
      maxN = Math.min(maxN + 10, N_MAX_INTERNAL);

      const xScale = d3.scaleLinear().domain([1, maxN]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([sim.logB - 1, sim.logA + 1]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');

      // boundary lines
      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yScale(sim.logA)).attr('y2', yScale(sim.logA)).style('stroke', COLOR_BOUND).attr('stroke-width', 1.4).attr('stroke-dasharray', '6 3');
      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yScale(sim.logB)).attr('y2', yScale(sim.logB)).style('stroke', COLOR_BOUND).attr('stroke-width', 1.4).attr('stroke-dasharray', '6 3');
      g.append('text').attr('x', innerW - 4).attr('y', yScale(sim.logA) - 4).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_BOUND).text(`log A = ${sim.logA.toFixed(2)}`);
      g.append('text').attr('x', innerW - 4).attr('y', yScale(sim.logB) + 14).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_BOUND).text(`log B = ${sim.logB.toFixed(2)}`);

      // trajectories
      for (let h = 0; h < 2; h++) {
        const paths = h === 0 ? sim.pathsH0 : sim.pathsH1;
        const stops = h === 0 ? sim.stopH0 : sim.stopH1;
        const decisions = h === 0 ? sim.decisionH0 : sim.decisionH1;
        const color = h === 0 ? COLOR_H0 : COLOR_H1;
        for (let i = 0; i < paths.length; i++) {
          const tau = stops[i];
          const end = tau > 0 ? Math.min(tau, maxN) : maxN;
          const pts: [number, number][] = [];
          for (let n = 1; n <= end; n++) pts.push([n, paths[i][n - 1]]);
          const line = d3.line<[number, number]>().x((d) => xScale(d[0])).y((d) => yScale(d[1]));
          g.append('path').datum(pts).attr('d', line as never).style('fill', 'none').style('stroke', color).attr('stroke-width', 0.7).attr('opacity', 0.55);
          if (tau > 0) {
            g.append('circle')
              .attr('cx', xScale(tau))
              .attr('cy', yScale(paths[i][tau - 1]))
              .attr('r', 3)
              .style('fill', decisions[i] === 1 ? COLOR_H1 : COLOR_H0)
              .attr('stroke', 'var(--color-bg)')
              .attr('stroke-width', 1);
          }
        }
      }

      g.append('text').attr('x', 0).attr('y', -10).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '12px').style('font-weight', 600).text('log Λ_n trajectories — 30 H_0 (blue), 30 H_1 (red), dots mark stopping decisions');
      g.append('text').attr('x', innerW / 2).attr('y', innerH + 28).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('sample size n');
    },
    [sim, w],
  );

  // Bottom panel: stopping-time histogram
  const botRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = BOT_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const tau0 = Array.from(sim.stopH0).filter((t) => t > 0);
      const tau1 = Array.from(sim.stopH1).filter((t) => t > 0);
      const allTau = [...tau0, ...tau1];
      if (allTau.length === 0) return;
      const tauMax = d3.quantile(allTau, 0.98) ?? 200;

      const xScale = d3.scaleLinear().domain([0, tauMax]).range([0, innerW]).nice();
      const bins0 = d3.bin().domain(xScale.domain() as [number, number]).thresholds(28)(tau0);
      const bins1 = d3.bin().domain(xScale.domain() as [number, number]).thresholds(28)(tau1);
      const yMax = Math.max(...bins0.map((b) => b.length), ...bins1.map((b) => b.length));
      const yScale = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]).nice();

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(4)).selectAll('text').style('fill', 'var(--color-text)');

      for (const [bins, color] of [[bins0, COLOR_H0], [bins1, COLOR_H1]] as const) {
        g.append('g')
          .selectAll('rect')
          .data(bins)
          .join('rect')
          .attr('x', (d) => xScale(d.x0 ?? 0) + 1)
          .attr('y', (d) => yScale(d.length))
          .attr('width', (d) => Math.max(xScale(d.x1 ?? 0) - xScale(d.x0 ?? 0) - 2, 0))
          .attr('height', (d) => innerH - yScale(d.length))
          .style('fill', color)
          .style('opacity', 0.55);
      }

      // Wald-prediction line
      if (sim.waldTau > 0 && sim.waldTau < tauMax) {
        g.append('line').attr('x1', xScale(sim.waldTau)).attr('x2', xScale(sim.waldTau)).attr('y1', 0).attr('y2', innerH).style('stroke', COLOR_BOUND).attr('stroke-width', 1.5).attr('stroke-dasharray', '5 3');
        g.append('text').attr('x', xScale(sim.waldTau) + 4).attr('y', 12).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', COLOR_BOUND).text(`E[τ|Wald] = ${sim.waldTau.toFixed(1)}`);
      }
      // Fixed-n NP line
      if (sim.nNP > 0 && sim.nNP < tauMax) {
        g.append('line').attr('x1', xScale(sim.nNP)).attr('x2', xScale(sim.nNP)).attr('y1', 0).attr('y2', innerH).style('stroke', 'var(--color-text-secondary)').attr('stroke-width', 1.2).attr('stroke-dasharray', '3 3');
        g.append('text').attr('x', xScale(sim.nNP) + 4).attr('y', innerH - 6).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text(`n_NP = ${sim.nNP.toFixed(0)}`);
      }

      g.append('text').attr('x', 0).attr('y', -10).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '12px').style('font-weight', 600).text(`Stopping-time histogram — emp. E[τ|H_0] = ${sim.meanTauH0.toFixed(1)}, E[τ|H_1] = ${sim.meanTauH1.toFixed(1)}`);
      g.append('text').attr('x', innerW / 2).attr('y', innerH + 28).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('τ (sample size at stop)');
    },
    [sim, w],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '14px', padding: '10px 0', fontSize: '13px' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '200px' }}>
          <span>
            Alt mean μ₁ = <strong>{displayMu.toFixed(2)}</strong> (μ₀ = 0, σ = 1)
          </span>
          <input
            type="range"
            min={0.1}
            max={1.0}
            step={0.05}
            value={displayMu}
            aria-label="Alternative mean"
            onChange={(e) => setDisplayMu(Number(e.target.value))}
            onMouseUp={(e) => setCommittedMu(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedMu(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedMu(Number((e.target as HTMLInputElement).value))}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '160px' }}>
          <span>
            α target = <strong>{alpha.toFixed(3)}</strong>
          </span>
          <input
            type="range"
            min={0}
            max={ERROR_OPTIONS.length - 1}
            step={1}
            value={alphaIdx}
            aria-label="Type-I error target"
            onChange={(e) => setAlphaIdx(Number(e.target.value))}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '160px' }}>
          <span>
            β target = <strong>{beta.toFixed(3)}</strong>
          </span>
          <input
            type="range"
            min={0}
            max={ERROR_OPTIONS.length - 1}
            step={1}
            value={betaIdx}
            aria-label="Type-II error target"
            onChange={(e) => setBetaIdx(Number(e.target.value))}
          />
        </label>
        <div
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '12px',
            color: 'var(--color-text-secondary)',
            padding: '6px 10px',
            border: '1px solid var(--color-border)',
            borderRadius: '4px',
            minWidth: '300px',
          }}
        >
          A = {Math.exp(sim.logA).toFixed(2)}, B = {Math.exp(sim.logB).toFixed(4)}.
          {' '}Empirical α = {sim.empRejectH0.toFixed(3)} (target {alpha.toFixed(3)}),
          β = {sim.empAcceptH1.toFixed(3)} (target {beta.toFixed(3)}).
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <svg ref={topRef} width={w} height={TOP_H} style={{ maxWidth: '100%', display: 'block' }} />
        <svg ref={botRef} width={w} height={BOT_H} style={{ maxWidth: '100%', display: 'block' }} />
      </div>
    </div>
  );
}
