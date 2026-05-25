import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32, binaryKL } from './shared/always-valid-inference';

// -----------------------------------------------------------------------------
// Viz 9.1 — GRO + Kelly growth-rate race.
//
// Under H_1: Bern(μ_1) testing H_0: μ = 0.5, plot four e-processes' log E_n
// trajectories:
//   (1) GRO = Kelly λ★ likelihood ratio   (steepest upward drift under H_1)
//   (2) Empirical Kelly                    (slightly less steep)
//   (3) Mis-tuned bet at λ = 1.0           (DRIFTS DOWN — over-betting fails)
//   (4) Universal mixture                  (Kelly minus O(log n) regret)
//
// Theoretical drift lines n · G_i overlaid.
// -----------------------------------------------------------------------------

const MU0 = 0.5;
const MARGIN = { top: 28, right: 16, bottom: 38, left: 56 };
const PANEL_H = 380;
const N_PATHS = 30;
const N_MAX_OPTIONS = [500, 1000, 2000];

const COLOR_KELLY = '#0f6e56';
const COLOR_EMP = '#356fa3';
const COLOR_MISTUNED = '#c44545';
const COLOR_UNIV = '#7c3aed';

function clip(x: number, m: number): number {
  return Math.max(-m, Math.min(m, x));
}

interface Sim {
  ng: number[];
  kelly: number[];
  emp: number[];
  mistuned: number[];
  univ: number[];
  kellyDrift: number[];
  empGrowthMistuned: number;
  empGrowthKelly: number;
}

function simulate(mu1: number, nmax: number): Sim {
  const lambdaKelly = clip((mu1 - MU0) / (MU0 * (1 - MU0)), 1.0);
  const lambdaMistuned = 1.0;
  const rng = mulberry32(20261401);
  // mean-trajectory ensemble: simulate N_PATHS streams; report average log E_n.
  const sumKelly = new Float64Array(nmax);
  const sumEmp = new Float64Array(nmax);
  const sumMistuned = new Float64Array(nmax);
  const sumUniv = new Float64Array(nmax);

  for (let b = 0; b < N_PATHS; b++) {
    let logK = 0;
    let logE = 0;
    let logM = 0;
    let logU = 0;
    let running = 0;
    let runningSq = 0;
    for (let n = 1; n <= nmax; n++) {
      const x = rng() < mu1 ? 1 : 0;
      running += x;
      runningSq += x * x;
      // Kelly
      logK += Math.log1p(lambdaKelly * (x - MU0));
      // Empirical Kelly
      const lambdaEmp = n > 1 ? clip((running - x) / (n - 1) - MU0, 0.5) * 4 : 0; // crude empirical
      logE += Math.log1p(clip(lambdaEmp, 1.0) * (x - MU0));
      // Mis-tuned
      logM += Math.log1p(lambdaMistuned * (x - MU0));
      // Universal: average over 21-point grid of λ in [-1, 1]; we just use
      // logsumexp over λ-grid of log K_n(λ) − log 21.
      // For demo, approximate via the average of three sample λs at
      // {0.3 λ_kelly, λ_kelly, 0.7 λ_kelly}
      const samples = [0.3 * lambdaKelly, lambdaKelly, 0.7 * lambdaKelly];
      const incs = samples.map((s) => logU + Math.log1p(s * (x - MU0)));
      const m = Math.max(...incs);
      logU = m + Math.log(incs.reduce((s, v) => s + Math.exp(v - m), 0) / 3);
      sumKelly[n - 1] += logK;
      sumEmp[n - 1] += logE;
      sumMistuned[n - 1] += logM;
      sumUniv[n - 1] += logU;
    }
  }
  const ng = Array.from({ length: nmax }, (_, i) => i + 1);
  const kelly = Array.from(sumKelly).map((v) => v / N_PATHS);
  const emp = Array.from(sumEmp).map((v) => v / N_PATHS);
  const mistuned = Array.from(sumMistuned).map((v) => v / N_PATHS);
  const univ = Array.from(sumUniv).map((v) => v / N_PATHS);

  // Theoretical drift lines: kelly drift = binary KL
  const kellyG = binaryKL(mu1, MU0);
  const kellyDrift = ng.map((n) => n * kellyG);

  const empGrowthMistuned = mistuned[mistuned.length - 1] / nmax;
  const empGrowthKelly = kelly[kelly.length - 1] / nmax;

  return { ng, kelly, emp, mistuned, univ, kellyDrift, empGrowthMistuned, empGrowthKelly };
}

export default function GROAndKellyExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayMu, setDisplayMu] = useState(0.6);
  const [committedMu, setCommittedMu] = useState(0.6);
  const [nmaxIdx, setNmaxIdx] = useState(1); // 1000

  const nmax = N_MAX_OPTIONS[nmaxIdx];
  const sim = useMemo(() => simulate(committedMu, nmax), [committedMu, nmax]);
  const w = Math.max(280, Math.min(containerWidth || 720, 820));

  const svgRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleLog().domain([1, nmax]).range([0, innerW]);
      const allY = [...sim.kelly, ...sim.emp, ...sim.mistuned, ...sim.univ];
      const yMin = Math.min(...allY) - 1;
      const yMax = Math.max(...allY) + 1;
      const yScale = d3.scaleLinear().domain([yMin, yMax]).range([innerH, 0]);

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(5, '~s')).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(6)).selectAll('text').style('fill', 'var(--color-text)');

      // zero line
      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yScale(0)).attr('y2', yScale(0)).style('stroke', 'var(--color-text-secondary)').attr('stroke-width', 0.6).attr('opacity', 0.5);

      const series = [
        { values: sim.kelly, color: COLOR_KELLY, label: 'GRO Kelly λ★' },
        { values: sim.emp, color: COLOR_EMP, label: 'empirical Kelly' },
        { values: sim.univ, color: COLOR_UNIV, label: 'universal mixture' },
        { values: sim.mistuned, color: COLOR_MISTUNED, label: 'mis-tuned λ = 1.0' },
      ];
      const line = d3.line<number>().x((_, i) => xScale(i + 1)).y((v) => yScale(v));
      series.forEach((s, i) => {
        g.append('path').datum(s.values).attr('d', line as never).style('fill', 'none').style('stroke', s.color).attr('stroke-width', 2);
        g.append('text').attr('x', innerW - 4).attr('y', 12 + i * 14).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', s.color).text(s.label);
      });

      // Theoretical Kelly drift line (dashed)
      const driftLine = d3.line<number>().x((_, i) => xScale(i + 1)).y((v) => yScale(v));
      g.append('path').datum(sim.kellyDrift).attr('d', driftLine as never).style('fill', 'none').style('stroke', COLOR_KELLY).attr('stroke-width', 1).attr('stroke-dasharray', '4 3').attr('opacity', 0.5);

      g.append('text').attr('x', innerW / 2).attr('y', innerH + 30).attr('text-anchor', 'middle').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('n (log scale)');
      g.append('text').attr('x', -innerH / 2).attr('y', -44).attr('text-anchor', 'middle').attr('transform', 'rotate(-90)').style('fill', 'var(--color-text-secondary)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('mean log E_n (across 30 streams)');
    },
    [sim, w, nmax],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '200px' }}>
          <span>Alt μ₁ = <strong>{displayMu.toFixed(2)}</strong> (μ₀ = 0.5)</span>
          <input
            type="range"
            min={0.52}
            max={0.95}
            step={0.01}
            value={displayMu}
            aria-label="Alternative mean"
            onChange={(e) => setDisplayMu(Number(e.target.value))}
            onMouseUp={(e) => setCommittedMu(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedMu(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedMu(Number((e.target as HTMLInputElement).value))}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '140px' }}>
          <span>n_max = <strong>{nmax}</strong></span>
          <input type="range" min={0} max={N_MAX_OPTIONS.length - 1} step={1} value={nmaxIdx} aria-label="Horizon" onChange={(e) => setNmaxIdx(Number(e.target.value))} />
        </label>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px', maxWidth: '380px' }}>
          Kelly G★ = kl({committedMu.toFixed(2)} ‖ 0.5) = <strong>{binaryKL(committedMu, MU0).toFixed(4)}</strong> per step.
          {' '}Mis-tuned (λ=1) Ĝ ≈ <strong style={{ color: sim.empGrowthMistuned < 0 ? COLOR_MISTUNED : COLOR_KELLY }}>{sim.empGrowthMistuned.toFixed(4)}</strong>.
          {sim.empGrowthMistuned < 0 && ' Wealth shrinks under H_1 — over-bet.'}
        </div>
      </div>
      <svg ref={svgRef} width={w} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
    </div>
  );
}
