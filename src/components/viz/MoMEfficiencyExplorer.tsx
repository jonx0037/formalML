import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  gammaFrom,
  gammaMLE,
  gammaMoM,
  gaussianFrom,
  gaussianMLE,
  gaussianMoM,
  mulberry32,
  type Vec,
} from './shared/generalized-method-of-moments';

// -----------------------------------------------------------------------------
// Viz 2.1 — MoM vs MLE asymptotic efficiency.
//
// Reader picks a distribution family (Gaussian, Gamma). For Gaussian, MoM and
// MLE coincide (the first two moments are sufficient statistics). For Gamma,
// MoM is consistent but inefficient — the gap is visible at moderate n.
//
// Three sliders: family selector, sample size n, family-specific parameter
// (μ for Gaussian, α for Gamma). For each (family, n, param), we run B=400
// replicates of MoM and MLE, plot their sampling distributions as overlaid
// histograms, and report the empirical variance ratio Var(θ̂_MoM) / Var(θ̂_MLE).
//
// MC compute is light (~B · n · 2 estimators) but enough to warrant
// commit-on-release for the n slider per CLAUDE.md.
// -----------------------------------------------------------------------------

type Family = 'gaussian' | 'gamma';
const N_OPTIONS = [50, 100, 200, 500, 1000];
const B_REPLICATES = 400;

const COLOR_MOM = '#9aa3ad';
const COLOR_MLE = 'var(--color-accent)';
const PANEL_H = 280;
const BAR_H = 110;
const MARGIN = { top: 24, right: 18, bottom: 38, left: 60 };

interface ReplicateResult {
  momParam: Vec;
  mleParam: Vec;
  momVar: number;
  mleVar: number;
  paramLabel: string;
  trueValue: number;
}

function runReplicates(family: Family, n: number, param: number): ReplicateResult {
  const momParam: Vec = [];
  const mleParam: Vec = [];
  for (let b = 0; b < B_REPLICATES; b++) {
    const rng = mulberry32(2026 + b * 31);
    const samples: Vec = [];
    if (family === 'gaussian') {
      const gauss = gaussianFrom(rng);
      // μ = param, σ = 1.0 (held fixed).
      for (let i = 0; i < n; i++) samples.push(param + gauss());
      const mom = gaussianMoM(samples);
      const mle = gaussianMLE(samples);
      momParam.push(mom.mu);
      mleParam.push(mle.mu);
    } else {
      // Gamma(α = param, β = 1).
      const gamma = gammaFrom(rng, param, 1);
      for (let i = 0; i < n; i++) samples.push(gamma());
      const mom = gammaMoM(samples);
      const mle = gammaMLE(samples);
      momParam.push(mom.alpha);
      mleParam.push(mle.alpha);
    }
  }
  return {
    momParam,
    mleParam,
    momVar: sampleVariance(momParam),
    mleVar: sampleVariance(mleParam),
    paramLabel: family === 'gaussian' ? 'μ̂' : 'α̂',
    trueValue: param,
  };
}

function sampleVariance(x: Vec): number {
  const n = x.length;
  if (n < 2) return 0;
  let sum = 0;
  for (const v of x) sum += v;
  const mean = sum / n;
  let sq = 0;
  for (const v of x) sq += (v - mean) * (v - mean);
  return sq / (n - 1);
}

export default function MoMEfficiencyExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [family, setFamily] = useState<Family>('gamma'); // default Gamma — the interesting case
  const [nIdx, setNIdx] = useState(2); // n = 200
  const [nIdxDisplay, setNIdxDisplay] = useState(2);
  const [param, setParam] = useState(2.5);

  const n = N_OPTIONS[nIdx];
  const nDisplay = N_OPTIONS[nIdxDisplay];

  const result = useMemo(() => runReplicates(family, n, param), [family, n, param]);
  const ratio = result.momVar > 0 ? result.momVar / Math.max(result.mleVar, 1e-12) : 1;

  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const histRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const combined = result.momParam.concat(result.mleParam);
      const ext = d3.extent(combined) as [number, number];
      const pad = (ext[1] - ext[0]) * 0.05;
      const xScale = d3.scaleLinear().domain([ext[0] - pad, ext[1] + pad]).nice().range([0, innerW]);

      const binner = d3.bin().domain(xScale.domain() as [number, number]).thresholds(28);
      const momBins = binner(result.momParam);
      const mleBins = binner(result.mleParam);
      const yMax = Math.max(d3.max(momBins, (b) => b.length) ?? 0, d3.max(mleBins, (b) => b.length) ?? 0);
      const yScale = d3.scaleLinear().domain([0, yMax * 1.1]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(6).tickFormat(d3.format('.3~f')))
        .selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5)).selectAll('text').style('fill', 'var(--color-text)');

      const renderBins = (bins: d3.Bin<number, number>[], color: string, opacity: number) => {
        for (const b of bins) {
          if (b.x0 === undefined || b.x1 === undefined) continue;
          g.append('rect')
            .attr('x', xScale(b.x0))
            .attr('y', yScale(b.length))
            .attr('width', Math.max(0, xScale(b.x1) - xScale(b.x0) - 1))
            .attr('height', innerH - yScale(b.length))
            .style('fill', color)
            .style('opacity', opacity);
        }
      };
      renderBins(momBins, COLOR_MOM, 0.7);
      renderBins(mleBins, COLOR_MLE, 0.55);

      // True value vertical line.
      g.append('line')
        .attr('x1', xScale(result.trueValue))
        .attr('x2', xScale(result.trueValue))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke-dasharray', '4 3')
        .style('stroke', 'var(--color-text-secondary)')
        .style('stroke-width', 1.2);
      g.append('text')
        .attr('x', xScale(result.trueValue) + 4)
        .attr('y', 12)
        .style('font-family', 'var(--font-mono)')
        .style('font-size', '11px')
        .style('fill', 'var(--color-text-secondary)')
        .text(`truth ${family === 'gaussian' ? 'μ' : 'α'} = ${result.trueValue.toFixed(2)}`);

      // Axis labels.
      g.append('text').attr('x', innerW / 2).attr('y', innerH + 28).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text(`${result.paramLabel} across ${B_REPLICATES} replicates`);
      g.append('text').attr('x', -innerH / 2).attr('y', -44).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('count');

      // Legend.
      const legend = g.append('g').attr('transform', `translate(${innerW - 110}, 4)`);
      legend.append('rect').attr('width', 14).attr('height', 14).style('fill', COLOR_MOM).style('opacity', 0.7);
      legend.append('text').attr('x', 20).attr('y', 12).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text('MoM');
      legend.append('rect').attr('x', 0).attr('y', 22).attr('width', 14).attr('height', 14).style('fill', COLOR_MLE).style('opacity', 0.55);
      legend.append('text').attr('x', 20).attr('y', 34).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text('MLE');
    },
    [result, w, family],
  );

  const barRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = BAR_H - 14;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left}, 6)`);

      const maxVar = Math.max(result.momVar, result.mleVar, 1e-12);
      const xScale = d3.scaleLinear().domain([0, maxVar * 1.1]).range([0, innerW * 0.8]);

      const labelX = 0;
      const barX = innerW * 0.15;
      const labels = ['MoM', 'MLE'];
      const vars = [result.momVar, result.mleVar];
      const colors = [COLOR_MOM, COLOR_MLE];

      const rowH = innerH / 2 - 4;
      labels.forEach((label, i) => {
        const y = i * (innerH / 2);
        g.append('text').attr('x', labelX).attr('y', y + rowH / 2 + 4).style('font-family', 'var(--font-mono)').style('font-size', '12px').style('fill', 'var(--color-text)').text(label);
        g.append('rect').attr('x', barX).attr('y', y).attr('width', xScale(vars[i])).attr('height', rowH).style('fill', colors[i]).style('opacity', 0.8);
        g.append('text').attr('x', barX + xScale(vars[i]) + 6).attr('y', y + rowH / 2 + 4).style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text-secondary)').text(`Var = ${vars[i].toExponential(2)}`);
      });
    },
    [result, w],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column' }}>
          <span>Family</span>
          <select
            value={family}
            onChange={(e) => setFamily(e.target.value as Family)}
            style={{ padding: '4px 8px', fontFamily: 'var(--font-mono)', fontSize: '12px' }}
            aria-label="Distribution family"
          >
            <option value="gaussian">Gaussian (μ, σ=1)</option>
            <option value="gamma">Gamma (α, β=1)</option>
          </select>
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>
            n = <strong>{nDisplay}</strong>
          </span>
          <input
            type="range"
            min={0}
            max={N_OPTIONS.length - 1}
            step={1}
            value={nIdxDisplay}
            aria-label="Sample size"
            onChange={(e) => setNIdxDisplay(Number(e.target.value))}
            onMouseUp={(e) => setNIdx(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setNIdx(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setNIdx(Number((e.target as HTMLInputElement).value))}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '180px' }}>
          <span>
            {family === 'gaussian' ? 'μ' : 'α'} = <strong>{param.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min={family === 'gaussian' ? -2 : 0.5}
            max={family === 'gaussian' ? 2 : 6}
            step={0.1}
            value={param}
            aria-label="True parameter value"
            onChange={(e) => setParam(Number(e.target.value))}
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
            alignSelf: 'flex-end',
          }}
        >
          Var(MoM) / Var(MLE) = <strong style={{ color: 'var(--color-text)' }}>{ratio.toFixed(2)}×</strong>
          {family === 'gaussian'
            ? ' — MoM = MLE for the Gaussian (sufficient statistics).'
            : ` — Gamma MoM is ${ratio > 1.05 ? 'inefficient' : 'nearly efficient'} relative to MLE.`}
        </div>
      </div>
      <svg ref={histRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} aria-label="Sampling distributions of MoM and MLE" />
      <svg ref={barRef} width={w} height={BAR_H} style={{ maxWidth: '100%', display: 'block', marginTop: 8 }} aria-label="Empirical variance bars" />
    </div>
  );
}
