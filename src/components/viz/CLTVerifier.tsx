// =============================================================================
// CLTVerifier.tsx
//
// §10.3 of sequential-monte-carlo. Numerical verification of Theorem 7
// (CLT for SMC): rescaled deviations √N(π_T^N(f) - π_T(f)) converge to
// N(0, σ²(f)) as N → ∞.
//
// Three panels:
//   (A) Example terminal cloud at the chosen N — histogram with target overlay.
//   (B) Histogram of rescaled deviations √N (π_T^N(f) - π_T(f)) over M
//       replicates, with the fitted normal density overlaid.
//   (C) Q-Q plot of those rescaled deviations vs standard-normal quantiles.
//
// Test function defaults to f(θ) = θ (mean estimator). The reader can pick
// f(θ) = θ², a more variance-sensitive choice, via the dropdown.
//
// Pure TS via shared/sequential-monte-carlo.ts → smcGaussianBridge. At N
// up to 1000 with M = 200 replicates the run takes ≈ 4-8 s, so we use
// commit-on-release for N.
//
// Static fallback: /images/topics/sequential-monte-carlo/10_clt_verification.png
// =============================================================================

import { useCallback, useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { useD3 } from './shared/useD3';
import {
  smcGaussianBridge,
  paletteSMC,
  SMC_SEED,
} from './shared/sequential-monte-carlo';

const MU_T = 5.0;
const SIGMA_T = 1.0;
const M_DEFAULT = 200;
const T_PATH = 10;

type TestFn = 'mean' | 'second-moment' | 'indicator';

const TEST_FN_LABELS: Record<TestFn, string> = {
  mean: 'f(θ) = θ',
  'second-moment': 'f(θ) = θ²',
  indicator: 'f(θ) = 𝟙[θ > 5]',
};

function evalTestFn(kind: TestFn, theta: number): number {
  if (kind === 'mean') return theta;
  if (kind === 'second-moment') return theta * theta;
  return theta > 5 ? 1 : 0;
}

function trueExpectation(kind: TestFn): number {
  // π_T = N(5, 1)
  if (kind === 'mean') return MU_T;
  if (kind === 'second-moment') return MU_T * MU_T + SIGMA_T * SIGMA_T;
  // P(θ > 5) under N(5, 1) is 0.5
  return 0.5;
}

interface Committed {
  N: number;
  M: number;
  fn: TestFn;
}

interface CltResult {
  rescaled: number[]; // √N (π_T^N(f) - π_T(f))
  sampleSd: number;
  exampleTheta: number[];
  exampleWeights: number[]; // normalized
}

function runCltSweep(c: Committed): CltResult {
  const rescaled = new Array<number>(c.M);
  const truth = trueExpectation(c.fn);
  let exampleTheta: number[] = [];
  let exampleWeights: number[] = [];
  for (let m = 0; m < c.M; m++) {
    const r = smcGaussianBridge({
      N: c.N,
      T: T_PATH,
      mu0: 0,
      sigma0: 2,
      muT: MU_T,
      sigmaT: SIGMA_T,
      resampleThreshold: 0.5,
      rwmStep: 0.5,
      rwmSweeps: 3,
      scheme: 'systematic',
      seed: SMC_SEED + 1000 + m * 7,
    });
    // weighted f-estimate at terminal
    const lwT = r.logWHistory[T_PATH];
    const lwMax = Math.max(...lwT);
    const w = lwT.map((lw) => Math.exp(lw - lwMax));
    const wSum = w.reduce((s, v) => s + v, 0);
    let est = 0;
    for (let i = 0; i < c.N; i++) est += (w[i] / wSum) * evalTestFn(c.fn, r.thetaHistory[T_PATH][i]);
    rescaled[m] = Math.sqrt(c.N) * (est - truth);
    if (m === 0) {
      exampleTheta = r.thetaHistory[T_PATH];
      exampleWeights = w.map((v) => v / wSum);
    }
  }
  const mean = rescaled.reduce((s, v) => s + v, 0) / c.M;
  let v = 0;
  for (let i = 0; i < c.M; i++) {
    const d = rescaled[i] - mean;
    v += d * d;
  }
  const sd = Math.sqrt(v / (c.M - 1));
  return { rescaled, sampleSd: sd, exampleTheta, exampleWeights };
}

function normalQuantile(p: number): number {
  // Beasley-Springer-Moro inverse normal approximation
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let q;
  let r;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p <= pHigh) {
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

function normalPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

export default function CLTVerifier() {
  const [dN, setDN] = useState(200);
  const [dFn, setDFn] = useState<TestFn>('mean');
  const [committed, setCommitted] = useState<Committed>({ N: 200, M: M_DEFAULT, fn: 'mean' });

  const commit = useCallback(
    (patch: Partial<Committed>) => setCommitted((c) => ({ ...c, ...patch })),
    [],
  );

  const result = useMemo(() => runCltSweep(committed), [committed]);

  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const width = containerWidth || 760;
  const panelGap = 14;
  const panelH = 220;
  const panelW = (width - panelGap * 2) / 3;
  const margin = { top: 22, right: 12, bottom: 32, left: 46 };
  const innerW = panelW - margin.left - margin.right;
  const innerH = panelH - margin.top - margin.bottom;

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (width <= 0) return;

      // ---- Panel A: example terminal cloud at this N
      const gA = svg.append('g').attr('transform', 'translate(0, 0)');
      const innerA = gA.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const xA = d3.scaleLinear().domain([MU_T - 4, MU_T + 4]).range([0, innerW]);
      const nBins = 22;
      const binEdges = d3
        .range(nBins + 1)
        .map((k) => xA.domain()[0] + (k / nBins) * (xA.domain()[1] - xA.domain()[0]));
      const binCounts = new Array<number>(nBins).fill(0);
      const binWidth = binEdges[1] - binEdges[0];
      for (let i = 0; i < result.exampleTheta.length; i++) {
        const idx = Math.min(
          nBins - 1,
          Math.max(0, Math.floor((result.exampleTheta[i] - binEdges[0]) / binWidth)),
        );
        binCounts[idx] += result.exampleWeights[i];
      }
      const dens = binCounts.map((c) => c / binWidth);
      const yMaxA = Math.max(normalPdf(MU_T, MU_T, SIGMA_T) * 1.2, ...dens) * 1.05;
      const yA = d3.scaleLinear().domain([0, yMaxA]).range([innerH, 0]);
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
      for (let k = 0; k < nBins; k++) {
        innerA
          .append('rect')
          .attr('x', xA(binEdges[k]))
          .attr('y', yA(dens[k]))
          .attr('width', Math.max(1, xA(binEdges[k + 1]) - xA(binEdges[k]) - 1))
          .attr('height', innerH - yA(dens[k]))
          .style('fill', paletteSMC.cloud)
          .style('opacity', 0.65);
      }
      // target curve
      const xs = d3.range(0, 201).map((k) => xA.domain()[0] + (k / 200) * (xA.domain()[1] - xA.domain()[0]));
      const lnTarget = d3
        .line<number>()
        .x((x) => xA(x))
        .y((x) => yA(normalPdf(x, MU_T, SIGMA_T)));
      innerA
        .append('path')
        .datum(xs)
        .attr('d', lnTarget)
        .style('fill', 'none')
        .style('stroke', paletteSMC.target)
        .style('stroke-width', 1.4)
        .style('stroke-dasharray', '4 3');
      gA.append('text')
        .attr('x', margin.left)
        .attr('y', 13)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text(`(A) one terminal cloud at N=${committed.N}`);

      // ---- Panel B: histogram of rescaled deviations + fitted normal
      const gB = svg.append('g').attr('transform', `translate(${panelW + panelGap}, 0)`);
      const innerB = gB.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const rExt = d3.extent(result.rescaled) as [number, number];
      const span = Math.max(Math.abs(rExt[0]), Math.abs(rExt[1])) * 1.15;
      const xB = d3.scaleLinear().domain([-span, span]).range([0, innerW]);
      const nBinsB = 22;
      const binEdgesB = d3
        .range(nBinsB + 1)
        .map((k) => xB.domain()[0] + (k / nBinsB) * (xB.domain()[1] - xB.domain()[0]));
      const binWidthB = binEdgesB[1] - binEdgesB[0];
      const binCountsB = new Array<number>(nBinsB).fill(0);
      for (const r of result.rescaled) {
        const idx = Math.min(nBinsB - 1, Math.max(0, Math.floor((r - binEdgesB[0]) / binWidthB)));
        binCountsB[idx]++;
      }
      const densB = binCountsB.map((c) => c / (result.rescaled.length * binWidthB));
      const fitMaxB = normalPdf(0, 0, result.sampleSd) * 1.1;
      const yMaxB = Math.max(fitMaxB, ...densB) * 1.05;
      const yB = d3.scaleLinear().domain([0, yMaxB]).range([innerH, 0]);
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
          .attr('y', yB(densB[k]))
          .attr('width', Math.max(1, xB(binEdgesB[k + 1]) - xB(binEdgesB[k]) - 1))
          .attr('height', innerH - yB(densB[k]))
          .style('fill', paletteSMC.cloud)
          .style('opacity', 0.65);
      }
      const fitXs = d3.range(0, 201).map((k) => -span + (k / 200) * 2 * span);
      const fitLn = d3
        .line<number>()
        .x((x) => xB(x))
        .y((x) => yB(normalPdf(x, 0, result.sampleSd)));
      innerB
        .append('path')
        .datum(fitXs)
        .attr('d', fitLn)
        .style('fill', 'none')
        .style('stroke', paletteSMC.target)
        .style('stroke-width', 1.5)
        .style('stroke-dasharray', '4 3');
      innerB
        .append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 24)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('√N (π_T^N(f) − π_T(f))');
      gB.append('text')
        .attr('x', margin.left)
        .attr('y', 13)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text(`(B) rescaled deviations (M=${committed.M})`);

      // ---- Panel C: Q-Q plot
      const gC = svg.append('g').attr('transform', `translate(${(panelW + panelGap) * 2}, 0)`);
      const innerC = gC.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);
      const sortedR = result.rescaled.slice().sort((a, b) => a - b);
      const qqN = sortedR.length;
      const theo = sortedR.map((_, k) => result.sampleSd * normalQuantile((k + 0.5) / qqN));
      const qqExt = d3.extent([...theo, ...sortedR]) as [number, number];
      const qqSpan = Math.max(Math.abs(qqExt[0]), Math.abs(qqExt[1])) * 1.1;
      const xC = d3.scaleLinear().domain([-qqSpan, qqSpan]).range([0, innerW]);
      const yC = d3.scaleLinear().domain([-qqSpan, qqSpan]).range([innerH, 0]);
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
      // y = x reference
      innerC
        .append('line')
        .attr('x1', xC(-qqSpan))
        .attr('x2', xC(qqSpan))
        .attr('y1', yC(-qqSpan))
        .attr('y2', yC(qqSpan))
        .style('stroke', paletteSMC.target)
        .style('stroke-width', 1)
        .style('stroke-dasharray', '4 3');
      // points
      for (let k = 0; k < qqN; k++) {
        innerC
          .append('circle')
          .attr('cx', xC(theo[k]))
          .attr('cy', yC(sortedR[k]))
          .attr('r', 1.8)
          .style('fill', paletteSMC.cloud)
          .style('opacity', 0.75);
      }
      innerC
        .append('text')
        .attr('x', innerW / 2)
        .attr('y', innerH + 24)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('theoretical quantile');
      innerC
        .append('text')
        .attr('transform', `translate(${-32}, ${innerH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)')
        .text('sample quantile');
      gC.append('text')
        .attr('x', margin.left)
        .attr('y', 13)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .style('fill', 'var(--color-text)')
        .text('(C) Q–Q plot of rescaled deviations');
    },
    [width, panelW, innerW, innerH, result, committed.N, committed.M],
  );

  const isStale = committed.N !== dN || committed.fn !== dFn;

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
            max={1000}
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
          <span>test function:</span>
          <select
            value={dFn}
            onChange={(e) => {
              const v = e.target.value as TestFn;
              setDFn(v);
              commit({ fn: v });
            }}
            className="rounded border px-1 py-0.5 text-xs"
            aria-label="Test function f"
          >
            <option value="mean">{TEST_FN_LABELS.mean}</option>
            <option value="second-moment">{TEST_FN_LABELS['second-moment']}</option>
            <option value="indicator">{TEST_FN_LABELS.indicator}</option>
          </select>
        </label>
        <span className="font-mono text-[10px]" style={{ color: 'var(--color-text)' }}>
          sample SD = {result.sampleSd.toFixed(3)}
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
        aria-label="CLT verification: example cloud, rescaled-deviation histogram, Q-Q plot."
      />
    </div>
  );
}
