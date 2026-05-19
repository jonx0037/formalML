// =============================================================================
// EvidenceUnbiasednessExplorer.tsx
//
// §8.2 of sequential-monte-carlo. Pedagogical demonstration of Theorem 4
// (Del Moral-Doucet-Jasra 2006 Proposition 1): the SMC estimator Ẑ_T is
// unbiased for Z_T, while log Ẑ_T is biased downward by ≈ σ²/2 (Proposition 5
// Jensen-gap correction).
//
// Two panels, M independent SMC replicates per setting:
//   (A) Ẑ_T / Z_T histogram with truth = 1 marked. The sample mean is the
//       empirical-MSE-style point estimate of E[Ẑ_T / Z_T].
//   (B) log Ẑ_T histogram with truth = 0 marked, plus the Jensen-gap
//       prediction -σ̂²/2 from Proposition 5.
//
// The target is the §3.4 Gaussian bridge (truth Z_T = 1, log Z_T = 0), so
// the truth values are known closed-form and serve as ground-truth markers.
//
// Pure TS via shared/sequential-monte-carlo.ts → smcGaussianBridge. M = 150
// replicates at (N up to 500, T up to 15) keeps the run under 8 s. Sliders
// use commit-on-release.
//
// Static fallback: /images/topics/sequential-monte-carlo/08_gmm_evidence.png
// =============================================================================

import { useCallback, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import { smcGaussianBridge, paletteSMC, SMC_SEED } from './shared/sequential-monte-carlo';

const M_REPS = 150;

interface Committed {
  N: number;
  T: number;
}

interface ResultRow {
  Zhat: number; // Ẑ_T (Z_truth = 1)
  logZhat: number;
}

function runReps(c: Committed): ResultRow[] {
  const rows: ResultRow[] = [];
  for (let m = 0; m < M_REPS; m++) {
    const r = smcGaussianBridge({
      N: c.N,
      T: c.T,
      mu0: 0,
      sigma0: 2,
      muT: 5,
      sigmaT: 1,
      resampleThreshold: 0.5,
      rwmStep: 0.5,
      rwmSweeps: 3,
      scheme: 'systematic',
      seed: SMC_SEED + 800 + m * 13,
    });
    rows.push({ Zhat: Math.exp(r.logZHat), logZhat: r.logZHat });
  }
  return rows;
}

export default function EvidenceUnbiasednessExplorer() {
  const [dN, setDN] = useState(200);
  const [dT, setDT] = useState(10);
  const [committed, setCommitted] = useState<Committed>({ N: 200, T: 10 });

  const commit = useCallback(
    (patch: Partial<Committed>) => setCommitted((c) => ({ ...c, ...patch })),
    [],
  );

  const rows = useMemo(() => runReps(committed), [committed]);

  // Empirical summaries
  const zHatMean = rows.reduce((s, r) => s + r.Zhat, 0) / rows.length;
  const logZHatMean = rows.reduce((s, r) => s + r.logZhat, 0) / rows.length;
  // sigma^2 ≈ Var(Ẑ_T) / Z_T^2 — Proposition 5 Jensen-gap uses σ̂² / 2
  const zHatVar = (() => {
    let v = 0;
    for (const r of rows) v += (r.Zhat - zHatMean) * (r.Zhat - zHatMean);
    return v / (rows.length - 1);
  })();
  const jensenGapPred = -zHatVar / 2; // since Z_truth = 1, σ̂² / Z² = σ̂²

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const width = containerWidth || 760;
  const panelGap = 16;
  const panelH = 240;
  const panelW = (width - panelGap) / 2;
  const margin = { top: 24, right: 14, bottom: 36, left: 48 };
  const innerW = panelW - margin.left - margin.right;
  const innerH = panelH - margin.top - margin.bottom;

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;

      // ---- Panel A: histogram of Ẑ_T centered at truth Z = 1
      const gA = svg.append('g').attr('transform', 'translate(0, 0)');
      const innerA = gA.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const zExt = d3.extent(rows, (r) => r.Zhat) as [number, number];
      const zSpan = Math.max(Math.abs(zExt[0] - 1), Math.abs(zExt[1] - 1)) * 1.15;
      const xA = d3.scaleLinear().domain([1 - zSpan, 1 + zSpan]).range([0, innerW]);
      const nBinsA = 18;
      const binEdgesA = d3
        .range(nBinsA + 1)
        .map((k) => xA.domain()[0] + (k / nBinsA) * (xA.domain()[1] - xA.domain()[0]));
      const binWidthA = binEdgesA[1] - binEdgesA[0];
      const binsA = new Array<number>(nBinsA).fill(0);
      for (const r of rows) {
        const idx = Math.min(nBinsA - 1, Math.max(0, Math.floor((r.Zhat - binEdgesA[0]) / binWidthA)));
        binsA[idx]++;
      }
      const yMaxA = Math.max(...binsA);
      const yA = d3.scaleLinear().domain([0, yMaxA * 1.1]).range([innerH, 0]);
      innerA
        .append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xA).ticks(4))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerA
        .append('g')
        .call(d3.axisLeft(yA).ticks(4))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      for (let k = 0; k < nBinsA; k++) {
        innerA
          .append('rect')
          .attr('x', xA(binEdgesA[k]))
          .attr('y', yA(binsA[k]))
          .attr('width', Math.max(1, xA(binEdgesA[k + 1]) - xA(binEdgesA[k]) - 1))
          .attr('height', innerH - yA(binsA[k]))
          .style('fill', paletteSMC.cloud)
          .style('opacity', 0.6);
      }
      // truth marker Z = 1
      innerA
        .append('line')
        .attr('x1', xA(1))
        .attr('x2', xA(1))
        .attr('y1', 0)
        .attr('y2', innerH)
        .style('stroke', paletteSMC.target)
        .style('stroke-width', 1.4)
        .style('stroke-dasharray', '4 3');
      innerA
        .append('text')
        .attr('x', xA(1) + 4)
        .attr('y', 12)
        .style('font-size', '10px')
        .style('fill', paletteSMC.target)
        .text(`truth Z = 1`);
      // sample mean marker
      innerA
        .append('line')
        .attr('x1', xA(zHatMean))
        .attr('x2', xA(zHatMean))
        .attr('y1', 0)
        .attr('y2', innerH)
        .style('stroke', paletteSMC.accent)
        .style('stroke-width', 1.4);
      innerA
        .append('text')
        .attr('x', xA(zHatMean) + 4)
        .attr('y', 26)
        .style('font-size', '10px')
        .style('fill', paletteSMC.accent)
        .text(`mean = ${zHatMean.toFixed(3)}`);
      innerA
        .append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 26)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('Ẑ_T');
      innerA
        .append('text')
        .attr('transform', `translate(${-34}, ${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text(`count (M = ${M_REPS})`);
      gA.append('text')
        .attr('x', margin.left)
        .attr('y', 14)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('(A) Ẑ_T histogram — unbiased for Z = 1 (Theorem 4)');

      // ---- Panel B: histogram of log Ẑ_T, Jensen-gap shift
      const gB = svg.append('g').attr('transform', `translate(${panelW + panelGap}, 0)`);
      const innerB = gB.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const lzExt = d3.extent(rows, (r) => r.logZhat) as [number, number];
      const lzSpan = Math.max(Math.abs(lzExt[0]), Math.abs(lzExt[1])) * 1.15;
      const xB = d3.scaleLinear().domain([-lzSpan, lzSpan]).range([0, innerW]);
      const nBinsB = 18;
      const binEdgesB = d3
        .range(nBinsB + 1)
        .map((k) => xB.domain()[0] + (k / nBinsB) * (xB.domain()[1] - xB.domain()[0]));
      const binWidthB = binEdgesB[1] - binEdgesB[0];
      const binsB = new Array<number>(nBinsB).fill(0);
      for (const r of rows) {
        const idx = Math.min(nBinsB - 1, Math.max(0, Math.floor((r.logZhat - binEdgesB[0]) / binWidthB)));
        binsB[idx]++;
      }
      const yMaxB = Math.max(...binsB);
      const yB = d3.scaleLinear().domain([0, yMaxB * 1.1]).range([innerH, 0]);
      innerB
        .append('g')
        .attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(xB).ticks(4))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      innerB
        .append('g')
        .call(d3.axisLeft(yB).ticks(4))
        .call((g) => g.selectAll('text').style('fill', 'var(--color-text)').style('font-size', '10px'))
        .call((g) => g.selectAll('line, path').style('stroke', 'var(--color-muted)'));
      for (let k = 0; k < nBinsB; k++) {
        innerB
          .append('rect')
          .attr('x', xB(binEdgesB[k]))
          .attr('y', yB(binsB[k]))
          .attr('width', Math.max(1, xB(binEdgesB[k + 1]) - xB(binEdgesB[k]) - 1))
          .attr('height', innerH - yB(binsB[k]))
          .style('fill', paletteSMC.cloud)
          .style('opacity', 0.6);
      }
      // truth marker log Z = 0
      innerB
        .append('line')
        .attr('x1', xB(0))
        .attr('x2', xB(0))
        .attr('y1', 0)
        .attr('y2', innerH)
        .style('stroke', paletteSMC.target)
        .style('stroke-width', 1.4)
        .style('stroke-dasharray', '4 3');
      innerB
        .append('text')
        .attr('x', xB(0) + 4)
        .attr('y', 12)
        .style('font-size', '10px')
        .style('fill', paletteSMC.target)
        .text('truth log Z = 0');
      // sample mean marker
      innerB
        .append('line')
        .attr('x1', xB(logZHatMean))
        .attr('x2', xB(logZHatMean))
        .attr('y1', 0)
        .attr('y2', innerH)
        .style('stroke', paletteSMC.accent)
        .style('stroke-width', 1.4);
      innerB
        .append('text')
        .attr('x', xB(logZHatMean) + 4)
        .attr('y', 26)
        .style('font-size', '10px')
        .style('fill', paletteSMC.accent)
        .text(`mean = ${logZHatMean.toFixed(3)}`);
      // Jensen-gap prediction
      innerB
        .append('line')
        .attr('x1', xB(jensenGapPred))
        .attr('x2', xB(jensenGapPred))
        .attr('y1', 0)
        .attr('y2', innerH)
        .style('stroke', paletteSMC.proposal)
        .style('stroke-width', 1.2)
        .style('stroke-dasharray', '2 2');
      innerB
        .append('text')
        .attr('x', xB(jensenGapPred) - 4)
        .attr('y', 40)
        .attr('text-anchor', 'end')
        .style('font-size', '10px')
        .style('fill', paletteSMC.proposal)
        .text(`Jensen ≈ ${jensenGapPred.toFixed(3)}`);
      innerB
        .append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 26)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('log Ẑ_T');
      gB.append('text')
        .attr('x', margin.left)
        .attr('y', 14)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('(B) log Ẑ_T — Jensen-gap bias (Proposition 5)');
    },
    [width, panelW, innerW, innerH, rows, zHatMean, logZHatMean, jensenGapPred],
  );

  const isStale = committed.N !== dN || committed.T !== dT;

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
            step={50}
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
            min={5}
            max={20}
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
        <span className="font-mono text-[10px]" style={{ color: 'var(--color-text-secondary)' }}>
          Var(Ẑ) ≈ {zHatVar.toExponential(2)}
        </span>
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
        aria-label="Two-panel evidence-unbiasedness demonstration: Ẑ histogram and log Ẑ histogram with Jensen-gap marker."
      />
    </div>
  );
}
