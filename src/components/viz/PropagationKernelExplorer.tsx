// =============================================================================
// PropagationKernelExplorer.tsx
//
// §5.4 of sequential-monte-carlo. Three-kernel head-to-head on an
// anisotropic 2D Gaussian bridge from π_0 = N(0, I) to π_T = N([3, 3], diag(0.3, 1.5)).
// Compares RWM-k=1, RWM-k=5, and IMH cloud-fit propagation kernels under
// identical reweight + systematic-resample machinery.
//
// Panels:
//   (A) ESS over the path, one line per kernel.
//   (B) Terminal cloud scatter (2D), one panel per kernel side-by-side, with
//       the target 95% level set as a dashed ellipse.
//   (C) Per-step acceptance rate, one line per kernel.
//
// Reader-discoverable behaviours:
//   - RWM k=1: ESS dips, terminal scatter is loose around the long axis.
//   - RWM k=5: ESS healthy, terminal scatter tighter — 5 sweeps amortize the
//     mixing cost.
//   - IMH cloud-fit: highest ESS and tightest terminal scatter; acceptance
//     rises through the path as the cloud concentrates on π_T.
//
// Sliders use the display-vs-committed pattern. At default (N=200, T=8, σ_RWM=0.4)
// the three-kernel sweep takes ≈ 100-200 ms.
//
// Static fallback: /images/topics/sequential-monte-carlo/05_kernel_comparison.png
// =============================================================================

import { useCallback, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  smcAnisotropic2D,
  paletteSMC,
  SMC_SEED,
  type SmcAnisotropicResult,
} from './shared/sequential-monte-carlo';

type KernelKey = 'rwm-k1' | 'rwm-k5' | 'imh';
const KERNELS: KernelKey[] = ['rwm-k1', 'rwm-k5', 'imh'];
const KERNEL_LABELS: Record<KernelKey, string> = {
  'rwm-k1': 'RWM (k=1)',
  'rwm-k5': 'RWM (k=5)',
  imh: 'IMH cloud-fit',
};
const KERNEL_COLORS: Record<KernelKey, string> = {
  'rwm-k1': paletteSMC.muted,
  'rwm-k5': paletteSMC.cloud,
  imh: paletteSMC.target,
};

const MU_T: [number, number] = [3, 3];
const S1 = Math.sqrt(0.3);
const S2 = Math.sqrt(1.5);

interface Committed {
  N: number;
  T: number;
  sigmaRwm: number;
}

function runAll(c: Committed): Record<KernelKey, SmcAnisotropicResult> {
  const base = {
    N: c.N,
    T: c.T,
    muT: MU_T,
    s1: S1,
    s2: S2,
    resampleThreshold: 0.5,
    scheme: 'systematic' as const,
    seed: SMC_SEED + 5,
  };
  return {
    'rwm-k1': smcAnisotropic2D({ ...base, kernel: { kind: 'rwm', sigma: c.sigmaRwm, sweeps: 1 } }),
    'rwm-k5': smcAnisotropic2D({ ...base, kernel: { kind: 'rwm', sigma: c.sigmaRwm, sweeps: 5 } }),
    imh: smcAnisotropic2D({ ...base, kernel: { kind: 'imh', ridge: 1e-6 } }),
  };
}

export default function PropagationKernelExplorer() {
  const [dN, setDN] = useState(200);
  const [dT, setDT] = useState(8);
  const [dSigma, setDSigma] = useState(0.4);
  const [committed, setCommitted] = useState<Committed>({ N: 200, T: 8, sigmaRwm: 0.4 });

  const commit = useCallback(
    (patch: Partial<Committed>) => setCommitted((c) => ({ ...c, ...patch })),
    [],
  );

  const results = useMemo(() => runAll(committed), [committed]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const width = containerWidth || 760;
  const panelGap = 14;
  const topH = 200;
  const bottomH = 200;
  const margin = { top: 22, right: 12, bottom: 32, left: 44 };
  const topPanelW = width;
  const topInnerW = topPanelW - margin.left - margin.right;
  const topInnerH = topH - margin.top - margin.bottom;
  const scatterW = (width - panelGap * 2) / 3;
  const scatterInnerW = scatterW - margin.left - margin.right;
  const scatterInnerH = bottomH - margin.top - margin.bottom;

  const topSvgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;

      // Panel A: ESS lines per kernel
      const xA = d3.scaleLinear().domain([0, 1]).range([0, topInnerW / 2 - panelGap / 2]);
      const yA = d3.scaleLinear().domain([0, committed.N * 1.05]).range([topInnerH, 0]);
      const gA = svg
        .append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);
      gA.append('g')
        .attr('transform', `translate(0, ${topInnerH})`)
        .call(d3.axisBottom(xA).ticks(4))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      gA.append('g')
        .call(d3.axisLeft(yA).ticks(4))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      gA.append('text')
        .attr('x', (topInnerW / 2 - panelGap / 2) / 2)
        .attr('y', topInnerH + 24)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('β_t');
      gA.append('text')
        .attr('transform', `translate(${-32}, ${topInnerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('ESS');
      // threshold line
      gA.append('line')
        .attr('x1', 0)
        .attr('x2', topInnerW / 2 - panelGap / 2)
        .attr('y1', yA(0.5 * committed.N))
        .attr('y2', yA(0.5 * committed.N))
        .style('stroke', paletteSMC.muted)
        .style('stroke-width', 1)
        .style('stroke-dasharray', '4 3');
      for (const k of KERNELS) {
        const res = results[k];
        const ln = d3
          .line<number>()
          .x((_, i) => xA(res.betaGrid[i]))
          .y((e) => yA(e));
        gA.append('path')
          .datum(res.essHistory)
          .attr('d', ln)
          .style('fill', 'none')
          .style('stroke', KERNEL_COLORS[k])
          .style('stroke-width', 1.6)
          .style('opacity', 0.9);
      }
      svg
        .append('text')
        .attr('x', margin.left)
        .attr('y', 13)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('(A) ESS through the path');

      // Panel C: per-step acceptance rate (uses right half of top row)
      const gC = svg
        .append('g')
        .attr(
          'transform',
          `translate(${margin.left + topInnerW / 2 + panelGap / 2}, ${margin.top})`,
        );
      const innerWC = topInnerW / 2 - panelGap / 2;
      const xC = d3.scaleLinear().domain([0, 1]).range([0, innerWC]);
      const yC = d3.scaleLinear().domain([0, 1]).range([topInnerH, 0]);
      gC.append('g')
        .attr('transform', `translate(0, ${topInnerH})`)
        .call(d3.axisBottom(xC).ticks(4))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      gC.append('g')
        .call(d3.axisLeft(yC).ticks(4).tickFormat(d3.format('.0%')))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      gC.append('text')
        .attr('x', innerWC / 2)
        .attr('y', topInnerH + 24)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('β_t');
      gC.append('text')
        .attr('transform', `translate(${-32}, ${topInnerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('acceptance');
      // Roberts-Gelman-Gilks 0.234 reference for RWM
      gC.append('line')
        .attr('x1', 0)
        .attr('x2', innerWC)
        .attr('y1', yC(0.234))
        .attr('y2', yC(0.234))
        .style('stroke', paletteSMC.muted)
        .style('stroke-width', 1)
        .style('stroke-dasharray', '2 3')
        .style('opacity', 0.7);
      gC.append('text')
        .attr('x', innerWC - 4)
        .attr('y', yC(0.234) - 3)
        .attr('text-anchor', 'end')
        .style('font-size', '9px')
        .style('fill', paletteSMC.muted)
        .text('RWM optimal 0.234');
      for (const k of KERNELS) {
        const res = results[k];
        // drop first sentinel (rate 1 at t=0)
        const data = res.acceptRateHistory.slice(1);
        const ln = d3
          .line<number>()
          .x((_, i) => xC(res.betaGrid[i + 1]))
          .y((r) => yC(r));
        gC.append('path')
          .datum(data)
          .attr('d', ln)
          .style('fill', 'none')
          .style('stroke', KERNEL_COLORS[k])
          .style('stroke-width', 1.6)
          .style('opacity', 0.9);
      }
      svg
        .append('text')
        .attr('x', margin.left + topInnerW / 2 + panelGap / 2)
        .attr('y', 13)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('(C) per-step acceptance rate');
    },
    [width, topInnerW, topInnerH, panelGap, results, committed.N],
  );

  const bottomSvgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;
      // x and y range across all three terminal clouds
      const allTheta = KERNELS.flatMap((k) => results[k].terminalCloud);
      const xExt = d3.extent(allTheta, (d) => d[0]) as [number, number];
      const yExt = d3.extent(allTheta, (d) => d[1]) as [number, number];
      const xMin = Math.min(xExt[0], MU_T[0] - 3 * S1);
      const xMax = Math.max(xExt[1], MU_T[0] + 3 * S1);
      const yMin = Math.min(yExt[0], MU_T[1] - 3 * S2);
      const yMax = Math.max(yExt[1], MU_T[1] + 3 * S2);

      KERNELS.forEach((k, ki) => {
        const gK = svg
          .append('g')
          .attr('transform', `translate(${ki * (scatterW + panelGap)}, 0)`);
        const inner = gK
          .append('g')
          .attr('transform', `translate(${margin.left}, ${margin.top})`);
        const x = d3.scaleLinear().domain([xMin, xMax]).range([0, scatterInnerW]);
        const y = d3.scaleLinear().domain([yMin, yMax]).range([scatterInnerH, 0]);
        inner
          .append('g')
          .attr('transform', `translate(0, ${scatterInnerH})`)
          .call(d3.axisBottom(x).ticks(4))
          .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '9px'))
          .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
        inner
          .append('g')
          .call(d3.axisLeft(y).ticks(4))
          .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '9px'))
          .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
        inner
          .append('text')
          .attr('x', scatterInnerW / 2)
          .attr('y', scatterInnerH + 22)
          .attr('text-anchor', 'middle')
          .style('font-size', '9px')
          .style('fill', 'var(--color-text-secondary)')
          .text('θ_1');
        inner
          .append('text')
          .attr('transform', `translate(${-30}, ${scatterInnerH / 2}) rotate(-90)`)
          .attr('text-anchor', 'middle')
          .style('font-size', '9px')
          .style('fill', 'var(--color-text-secondary)')
          .text('θ_2');
        // target 95% ellipse: axes at ~2σ for diag Σ
        inner
          .append('ellipse')
          .attr('cx', x(MU_T[0]))
          .attr('cy', y(MU_T[1]))
          .attr('rx', Math.abs(x(MU_T[0] + 2 * S1) - x(MU_T[0])))
          .attr('ry', Math.abs(y(MU_T[1] + 2 * S2) - y(MU_T[1])))
          .style('fill', 'none')
          .style('stroke', paletteSMC.target)
          .style('stroke-width', 1)
          .style('stroke-dasharray', '4 3');
        // weighted scatter
        const cloud = results[k].terminalCloud;
        const w = results[k].terminalWeights;
        const wMax = Math.max(...w);
        for (let i = 0; i < cloud.length; i++) {
          inner
            .append('circle')
            .attr('cx', x(cloud[i][0]))
            .attr('cy', y(cloud[i][1]))
            .attr('r', 1.8)
            .style('fill', KERNEL_COLORS[k])
            .style('opacity', Math.min(0.75, 0.2 + 0.55 * (w[i] / Math.max(wMax, 1e-12))));
        }
        gK.append('text')
          .attr('x', margin.left)
          .attr('y', 13)
          .style('font-size', '10px')
          .style('font-weight', '600')
          .style('fill', 'var(--color-text)')
          .text(`(B${ki + 1}) ${KERNEL_LABELS[k]}`);
      });
    },
    [width, scatterW, scatterInnerW, scatterInnerH, panelGap, results],
  );

  const isStale =
    committed.N !== dN || committed.T !== dT || committed.sigmaRwm !== dSigma;

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
            max={300}
            step={10}
            value={dN}
            onChange={(e) => setDN(parseInt(e.target.value, 10))}
            onMouseUp={(e) => commit({ N: parseInt((e.target as HTMLInputElement).value, 10) })}
            onTouchEnd={(e) => commit({ N: parseInt((e.target as HTMLInputElement).value, 10) })}
            onKeyUp={(e) => commit({ N: parseInt((e.target as HTMLInputElement).value, 10) })}
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
            min={4}
            max={15}
            step={1}
            value={dT}
            onChange={(e) => setDT(parseInt(e.target.value, 10))}
            onMouseUp={(e) => commit({ T: parseInt((e.target as HTMLInputElement).value, 10) })}
            onTouchEnd={(e) => commit({ T: parseInt((e.target as HTMLInputElement).value, 10) })}
            onKeyUp={(e) => commit({ T: parseInt((e.target as HTMLInputElement).value, 10) })}
            className="w-28"
            aria-label="Path length T"
          />
        </label>
        <label className="inline-flex items-center gap-2">
          <span>
            RWM σ = <span className="font-mono">{dSigma.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.1}
            max={1.0}
            step={0.05}
            value={dSigma}
            onChange={(e) => setDSigma(parseFloat(e.target.value))}
            onMouseUp={(e) => commit({ sigmaRwm: parseFloat((e.target as HTMLInputElement).value) })}
            onTouchEnd={(e) => commit({ sigmaRwm: parseFloat((e.target as HTMLInputElement).value) })}
            onKeyUp={(e) => commit({ sigmaRwm: parseFloat((e.target as HTMLInputElement).value) })}
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
        ref={topSvgRef}
        width={width}
        height={topH}
        viewBox={`0 0 ${width} ${topH}`}
        role="img"
        aria-label="ESS over the path and per-step acceptance rate for three propagation kernels."
      />
      <svg
        ref={bottomSvgRef}
        width={width}
        height={bottomH}
        viewBox={`0 0 ${width} ${bottomH}`}
        role="img"
        aria-label="Terminal cloud scatter for each of three propagation kernels with the target 95% ellipse overlaid."
      />
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
        {KERNELS.map((k) => (
          <span key={k} className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-2.5 rounded-sm"
              style={{ background: KERNEL_COLORS[k] }}
            />
            <span className="font-mono text-[10px]" style={{ color: 'var(--color-text)' }}>
              {KERNEL_LABELS[k]}
            </span>
          </span>
        ))}
        <span className="text-[10px]" style={{ color: 'var(--color-text-secondary)' }}>
          target π_T = N([3, 3], diag(0.3, 1.5))
        </span>
      </div>
    </div>
  );
}
