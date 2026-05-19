// =============================================================================
// SMCSkeletonExplorer.tsx
//
// §3.4 of sequential-monte-carlo. Three-panel SMC sweep on a Gaussian bridge
// π_0 = N(0, 4) → π_T = N(5, 1). The reader can change N, T, the resampling
// threshold τ, and the RWM step size, then watch the cloud trajectory, the
// ESS profile, and the terminal histogram update.
//
// Reader-discoverable behaviours:
//   - τ = 0 (never resample): ESS collapses, log Ẑ_T variance blows up.
//   - τ = 1 (resample every step): more variance injection than necessary.
//   - T = 2: the cloud can't track the target; terminal mean misses 5.
//   - T = 20: smooth tracking; log Ẑ_T near 0.
//   - rwmStep too small: weights stay degenerate post-resample.
//   - rwmStep too large: low acceptance, cloud explores poorly.
//
// Sliders use the display-vs-committed pattern from FiniteSampleBiasExplorer
// and SGLDBatchSizeExplorer: live label feedback on drag, heavy compute
// only on release (mouseup / touchend / keyup).
//
// Pure TS computation via shared/sequential-monte-carlo.ts → smcGaussianBridge.
// At default (N = 200, T = 10, 3 RWM sweeps) the run takes ≈ 15-30 ms; the
// upper bound N = 500, T = 20, 5 sweeps takes ≈ 250 ms.
//
// Static fallback: /images/topics/sequential-monte-carlo/03_smc_skeleton.png
// =============================================================================

import { useCallback, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  smcGaussianBridge,
  paletteSMC,
  SMC_SEED,
  type SmcSkeletonResult,
} from './shared/sequential-monte-carlo';

const DEFAULT_N = 200;
const DEFAULT_T = 10;
const DEFAULT_TAU = 0.5;
const DEFAULT_RWM_STEP = 0.5;
const DEFAULT_SWEEPS = 3;

const MU0 = 0.0;
const SIGMA0 = 2.0;
const MUT = 5.0;
const SIGMAT = 1.0;

function gaussianPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

interface Committed {
  N: number;
  T: number;
  tau: number;
  rwmStep: number;
}

export default function SMCSkeletonExplorer() {
  // Display state (slider position, updates on every change).
  const [dN, setDN] = useState(DEFAULT_N);
  const [dT, setDT] = useState(DEFAULT_T);
  const [dTau, setDTau] = useState(DEFAULT_TAU);
  const [dRwm, setDRwm] = useState(DEFAULT_RWM_STEP);
  // Committed state (drives the heavy useMemo).
  const [committed, setCommitted] = useState<Committed>({
    N: DEFAULT_N,
    T: DEFAULT_T,
    tau: DEFAULT_TAU,
    rwmStep: DEFAULT_RWM_STEP,
  });

  const commitN = useCallback((v: number) => setCommitted((c) => ({ ...c, N: v })), []);
  const commitT = useCallback((v: number) => setCommitted((c) => ({ ...c, T: v })), []);
  const commitTau = useCallback((v: number) => setCommitted((c) => ({ ...c, tau: v })), []);
  const commitRwm = useCallback((v: number) => setCommitted((c) => ({ ...c, rwmStep: v })), []);

  const result: SmcSkeletonResult = useMemo(
    () =>
      smcGaussianBridge({
        N: committed.N,
        T: committed.T,
        mu0: MU0,
        sigma0: SIGMA0,
        muT: MUT,
        sigmaT: SIGMAT,
        resampleThreshold: committed.tau,
        rwmStep: committed.rwmStep,
        rwmSweeps: DEFAULT_SWEEPS,
        scheme: 'systematic',
        seed: SMC_SEED + 3,
      }),
    [committed],
  );

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const width = containerWidth || 800;
  const panelGap = 16;
  const panelH = 220;
  const panelW = Math.max(180, (width - panelGap * 2) / 3);
  const margin = { top: 20, right: 14, bottom: 34, left: 46 };
  const innerW = panelW - margin.left - margin.right;
  const innerH = panelH - margin.top - margin.bottom;

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      const N = committed.N;
      const T = committed.T;

      // ---- Panel A: cloud trajectory across β
      const gA = svg.append('g').attr('transform', 'translate(0, 0)');
      const innerA = gA.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xA = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
      // y range: capture both initial cloud spread and terminal target
      const allTheta = result.thetaHistory.flat();
      const yMin = Math.min(...allTheta, MU0 - 2 * SIGMA0);
      const yMax = Math.max(...allTheta, MUT + 2 * SIGMAT);
      const yA = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]);
      innerA
        .append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xA).ticks(4))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerA
        .append('g')
        .call(d3.axisLeft(yA).ticks(5))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerA
        .append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 26)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('β_t');
      innerA
        .append('text')
        .attr('transform', `translate(${-32}, ${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('θ');
      // particle scatter per step (subsample if N is large for rendering perf)
      const stride = Math.max(1, Math.floor(N / 200));
      for (let t = 0; t <= T; t++) {
        const theta = result.thetaHistory[t];
        const logW = result.logWHistory[t];
        const lwMax = Math.max(...logW);
        const wNorm = logW.map((lw) => Math.exp(lw - lwMax));
        const wSum = wNorm.reduce((s, v) => s + v, 0);
        const maxW = Math.max(...wNorm) / wSum;
        for (let i = 0; i < N; i += stride) {
          const w = wNorm[i] / wSum;
          const alpha = Math.min(0.85, 0.15 + 0.7 * (w / Math.max(maxW, 1e-12)));
          innerA
            .append('circle')
            .attr('cx', xA(result.betaGrid[t]))
            .attr('cy', yA(theta[i]))
            .attr('r', 1.6)
            .style('fill', paletteSMC.cloud)
            .style('opacity', alpha);
        }
      }
      // target mean lines
      innerA
        .append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yA(MU0))
        .attr('y2', yA(MU0))
        .style('stroke', paletteSMC.muted)
        .style('stroke-width', 1)
        .style('stroke-dasharray', '2 3');
      innerA
        .append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yA(MUT))
        .attr('y2', yA(MUT))
        .style('stroke', paletteSMC.target)
        .style('stroke-width', 1)
        .style('stroke-dasharray', '4 3');
      // resample marks
      result.resampleMarks.forEach((mark, t) => {
        if (mark) {
          innerA
            .append('text')
            .attr('x', xA(result.betaGrid[t]))
            .attr('y', 10)
            .attr('text-anchor', 'middle')
            .style('font-size', '9px')
            .style('font-weight', '700')
            .style('fill', paletteSMC.accent)
            .text('R');
        }
      });
      gA.append('text')
        .attr('x', margin.left)
        .attr('y', 12)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('(A) particle cloud across the path');

      // ---- Panel B: ESS over time
      const gB = svg.append('g').attr('transform', `translate(${panelW + panelGap}, 0)`);
      const innerB = gB.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xB = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
      const yB = d3.scaleLinear().domain([0, N * 1.05]).range([innerH, 0]);
      innerB
        .append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xB).ticks(4))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerB
        .append('g')
        .call(d3.axisLeft(yB).ticks(5))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerB
        .append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 26)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('β_t');
      innerB
        .append('text')
        .attr('transform', `translate(${-32}, ${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('ESS');
      // ESS threshold line
      innerB
        .append('line')
        .attr('x1', 0)
        .attr('x2', innerW)
        .attr('y1', yB(committed.tau * N))
        .attr('y2', yB(committed.tau * N))
        .style('stroke', paletteSMC.accent)
        .style('stroke-width', 1)
        .style('stroke-dasharray', '4 3');
      innerB
        .append('text')
        .attr('x', innerW - 4)
        .attr('y', yB(committed.tau * N) - 4)
        .attr('text-anchor', 'end')
        .style('font-size', '9px')
        .style('fill', paletteSMC.accent)
        .text(`τ N = ${(committed.tau * N).toFixed(0)}`);
      // ESS line
      const essLine = d3
        .line<number>()
        .x((_, i) => xB(result.betaGrid[i]))
        .y((e) => yB(e));
      innerB
        .append('path')
        .datum(result.essHistory)
        .attr('d', essLine)
        .style('fill', 'none')
        .style('stroke', paletteSMC.cloud)
        .style('stroke-width', 1.6);
      innerB
        .selectAll('.ess-dot')
        .data(result.essHistory)
        .enter()
        .append('circle')
        .attr('cx', (_, i) => xB(result.betaGrid[i]))
        .attr('cy', (e) => yB(e))
        .attr('r', 2.5)
        .style('fill', paletteSMC.cloud);
      gB.append('text')
        .attr('x', margin.left)
        .attr('y', 12)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('(B) ESS through the path');

      // ---- Panel C: terminal histogram vs target
      const gC = svg.append('g').attr('transform', `translate(${(panelW + panelGap) * 2}, 0)`);
      const innerC = gC.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xC = d3.scaleLinear().domain([MUT - 4, MUT + 4]).range([0, innerW]);
      // Build weighted histogram of terminal cloud
      const nBins = 24;
      const binEdges = d3.range(nBins + 1).map((k) => xC.domain()[0] + (k / nBins) * (xC.domain()[1] - xC.domain()[0]));
      const binCounts = new Array<number>(nBins).fill(0);
      const lwT = result.logWHistory[T];
      const lwMax = Math.max(...lwT);
      const wT = lwT.map((lw) => Math.exp(lw - lwMax));
      const wTSum = wT.reduce((s, v) => s + v, 0);
      const wTNorm = wT.map((v) => v / wTSum);
      const thetaT = result.thetaHistory[T];
      for (let i = 0; i < N; i++) {
        const idx = Math.min(nBins - 1, Math.max(0, Math.floor((thetaT[i] - binEdges[0]) / (binEdges[1] - binEdges[0]))));
        binCounts[idx] += wTNorm[i];
      }
      const binWidth = binEdges[1] - binEdges[0];
      const densities = binCounts.map((c) => c / binWidth);
      const targetMax = gaussianPdf(MUT, MUT, SIGMAT);
      const yMaxC = Math.max(targetMax * 1.2, ...densities) * 1.05;
      const yC = d3.scaleLinear().domain([0, yMaxC]).range([innerH, 0]);
      innerC
        .append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xC).ticks(4))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerC
        .append('g')
        .call(d3.axisLeft(yC).ticks(4))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerC
        .append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 26)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('θ');
      innerC
        .append('text')
        .attr('transform', `translate(${-32}, ${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('density');
      // SMC weighted histogram bars
      for (let k = 0; k < nBins; k++) {
        const x0 = xC(binEdges[k]);
        const x1 = xC(binEdges[k + 1]);
        innerC
          .append('rect')
          .attr('x', x0)
          .attr('y', yC(densities[k]))
          .attr('width', Math.max(1, x1 - x0 - 1))
          .attr('height', innerH - yC(densities[k]))
          .style('fill', paletteSMC.cloud)
          .style('opacity', 0.65);
      }
      // target density curve
      const targetX = d3.range(0, 201).map((k) => xC.domain()[0] + (k / 200) * (xC.domain()[1] - xC.domain()[0]));
      const targetLine = d3
        .line<number>()
        .x((x) => xC(x))
        .y((x) => yC(gaussianPdf(x, MUT, SIGMAT)));
      innerC
        .append('path')
        .datum(targetX)
        .attr('d', targetLine)
        .style('fill', 'none')
        .style('stroke', paletteSMC.target)
        .style('stroke-width', 1.6)
        .style('stroke-dasharray', '4 3');
      gC.append('text')
        .attr('x', margin.left)
        .attr('y', 12)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text(`(C) terminal cloud vs π_T = N(${MUT}, ${SIGMAT}²)`);
    },
    [width, panelW, innerW, innerH, result, committed.tau, committed.N, committed.T],
  );

  const isStale = committed.N !== dN || committed.T !== dT || committed.tau !== dTau || committed.rwmStep !== dRwm;

  return (
    <div ref={containerRef} className="w-full">
      <div
        className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs"
        style={{ color: 'var(--color-text)' }}
      >
        <label className="inline-flex items-center gap-2">
          <span>
            N = <span className="font-mono">{dN}</span>
          </span>
          <input
            type="range"
            min={50}
            max={500}
            step={10}
            value={dN}
            onChange={(e) => setDN(parseInt(e.target.value, 10))}
            onMouseUp={(e) => commitN(parseInt((e.target as HTMLInputElement).value, 10))}
            onTouchEnd={(e) => commitN(parseInt((e.target as HTMLInputElement).value, 10))}
            onKeyUp={(e) => commitN(parseInt((e.target as HTMLInputElement).value, 10))}
            className="w-32"
            aria-label="Particle count N"
          />
        </label>
        <label className="inline-flex items-center gap-2">
          <span>
            T = <span className="font-mono">{dT}</span>
          </span>
          <input
            type="range"
            min={2}
            max={20}
            step={1}
            value={dT}
            onChange={(e) => setDT(parseInt(e.target.value, 10))}
            onMouseUp={(e) => commitT(parseInt((e.target as HTMLInputElement).value, 10))}
            onTouchEnd={(e) => commitT(parseInt((e.target as HTMLInputElement).value, 10))}
            onKeyUp={(e) => commitT(parseInt((e.target as HTMLInputElement).value, 10))}
            className="w-28"
            aria-label="Path length T"
          />
        </label>
        <label className="inline-flex items-center gap-2">
          <span>
            τ = <span className="font-mono">{dTau.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={dTau}
            onChange={(e) => setDTau(parseFloat(e.target.value))}
            onMouseUp={(e) => commitTau(parseFloat((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => commitTau(parseFloat((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => commitTau(parseFloat((e.target as HTMLInputElement).value))}
            className="w-28"
            aria-label="ESS threshold τ"
          />
        </label>
        <label className="inline-flex items-center gap-2">
          <span>
            RWM step = <span className="font-mono">{dRwm.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.1}
            max={2.0}
            step={0.05}
            value={dRwm}
            onChange={(e) => setDRwm(parseFloat(e.target.value))}
            onMouseUp={(e) => commitRwm(parseFloat((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => commitRwm(parseFloat((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => commitRwm(parseFloat((e.target as HTMLInputElement).value))}
            className="w-28"
            aria-label="RWM proposal scale"
          />
        </label>
        {isStale && (
          <span className="text-[10px] italic" style={{ color: 'var(--color-text-secondary)' }}>
            release the slider to recompute
          </span>
        )}
      </div>
      <svg
        ref={svgRef}
        width={width}
        height={panelH}
        viewBox={`0 0 ${width} ${panelH}`}
        role="img"
        aria-label="Three-panel SMC skeleton on a Gaussian bridge: cloud trajectory, ESS, terminal histogram."
      />
      <div
        className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] sm:grid-cols-4"
        style={{ color: 'var(--color-text)' }}
      >
        <div className="font-mono">log Ẑ_T = <span style={{ color: paletteSMC.target }}>{result.logZHat.toFixed(3)}</span> <span style={{ color: 'var(--color-text-secondary)' }}>(truth 0)</span></div>
        <div className="font-mono">terminal μ = {result.terminalMean.toFixed(3)} <span style={{ color: 'var(--color-text-secondary)' }}>(truth {MUT})</span></div>
        <div className="font-mono">terminal σ = {result.terminalStd.toFixed(3)} <span style={{ color: 'var(--color-text-secondary)' }}>(truth {SIGMAT})</span></div>
        <div className="font-mono">resamples = {result.resampleMarks.filter(Boolean).length} / {committed.T}</div>
      </div>
    </div>
  );
}
