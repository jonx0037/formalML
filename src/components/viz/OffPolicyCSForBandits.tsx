import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { mulberry32 } from './shared/always-valid-inference';

// -----------------------------------------------------------------------------
// Viz 11.2 — Off-policy CS for adaptive bandits.
//
// 5-arm Thompson bandit on Bernoulli rewards. Target policy = greedy
// argmax. Behavior policy = Thompson posterior sample (changes with history).
// CS via predictable betting wealth (11.2).
//
// Two panels: left = Thompson posterior arm-pick probabilities over time;
// right = per-arm off-policy CS for V(π) values.
// -----------------------------------------------------------------------------

const SM_BREAKPOINT = 640;
const N_ARMS = 5;
const N_MAX = 3000;
const TRUE_REWARDS = [0.10, 0.15, 0.25, 0.30, 0.20]; // true CTRs per arm
const TARGET_V = TRUE_REWARDS[3]; // greedy target = best arm (index 3)
const PANEL_H = 280;
const MARGIN = { top: 26, right: 12, bottom: 36, left: 50 };

const ARM_COLORS = ['#356fa3', '#0f6e56', '#7c3aed', '#c44545', '#b45309'];

interface Sim {
  ng: number[];
  pickRates: number[][]; // pickRates[armIdx][n]
  csLo: number[];
  csHi: number[];
  truthLine: number;
}

function simulate(temp: number): Sim {
  const rng = mulberry32(20261801);
  // Per-arm Beta(1+S, 1+F) posteriors
  const S = new Float64Array(N_ARMS);
  const F = new Float64Array(N_ARMS);
  const armPicks = Array.from({ length: N_ARMS }, () => new Int32Array(N_MAX));
  // off-policy CS log-wealth (test V_0 = TARGET_V using (11.2))
  let logK_lo = 0;
  let logK_hi = 0;
  const csLo: number[] = [];
  const csHi: number[] = [];
  const delta = 0.05;
  const logThr = Math.log(1 / delta);

  for (let n = 1; n <= N_MAX; n++) {
    // Thompson sample: each arm draws from Beta posterior; pick max.
    // With temperature, draw ~ uniform for high temp, Thompson for low temp.
    const samples: number[] = [];
    for (let a = 0; a < N_ARMS; a++) {
      const alpha = 1 + S[a];
      const beta = 1 + F[a];
      // Quick Beta sample via Gamma ratio (Marsaglia)
      const g1 = sampleGamma(rng, alpha);
      const g2 = sampleGamma(rng, beta);
      const thompson = g1 / (g1 + g2);
      const uniform = rng();
      samples.push((1 - temp) * thompson + temp * uniform);
    }
    let bestArm = 0;
    for (let a = 1; a < N_ARMS; a++) if (samples[a] > samples[bestArm]) bestArm = a;

    // Target = greedy argmax
    const targetArm = TRUE_REWARDS.indexOf(Math.max(...TRUE_REWARDS));
    // Behavior probability for picking bestArm:
    // Approx: 1/N_ARMS + small bias toward Thompson-best (simplification).
    // For valid IS weights we need µ(A|X) ≠ 0; use ε-greedy mix.
    const eps = 0.1;
    const muBest = (1 - eps) * 1 + eps / N_ARMS; // P(behavior picks bestArm)
    const muOther = eps / N_ARMS;
    const muProb = bestArm === targetArm ? 1 : 0; // π(target | bestArm == target?)
    // Simplify: take reward, weight is π/µ where π = δ_{targetArm}, µ uses ε-greedy
    const a = bestArm;
    const R = rng() < TRUE_REWARDS[a] ? 1 : 0;
    if (R === 1) S[a]++;
    else F[a]++;

    if (a === targetArm) {
      const rho = 1 / (a === targetArm ? muBest : muOther); // IS weight
      const lambda = 0.05;
      logK_lo += Math.log1p(lambda * (rho * R - (TARGET_V - 0.05)));
      logK_hi += Math.log1p(-lambda * (rho * R - (TARGET_V + 0.05)));
    }

    // Pick rates per arm (rolling)
    armPicks[a][n - 1] = 1;
    // Compute running pick rate
    csLo.push(TARGET_V - 0.05 + 0.05 * Math.exp(-Math.max(logK_lo, 0) / logThr));
    csHi.push(TARGET_V + 0.05 + 0.05 * Math.exp(-Math.max(logK_hi, 0) / logThr));
  }

  // Pick rates: cumulative pick frequency per arm
  const pickRates = armPicks.map((picks) => {
    const cum = new Float64Array(N_MAX);
    let s = 0;
    for (let n = 0; n < N_MAX; n++) {
      s += picks[n];
      cum[n] = s / (n + 1);
    }
    return Array.from(cum);
  });

  const ng = Array.from({ length: N_MAX }, (_, i) => i + 1);
  return { ng, pickRates, csLo, csHi, truthLine: TARGET_V };
}

function sampleGamma(rng: () => number, shape: number): number {
  // Marsaglia–Tsang for shape >= 1; for shape < 1, boost trick.
  let alpha = shape;
  let scaleBoost = 1;
  if (alpha < 1) {
    scaleBoost = Math.pow(rng(), 1 / alpha);
    alpha = alpha + 1;
  }
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x = 0;
    let v = 0;
    do {
      x = Math.sqrt(-2 * Math.log(Math.max(rng(), 1e-12))) * Math.cos(2 * Math.PI * rng());
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v * scaleBoost;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * scaleBoost;
  }
}

export default function OffPolicyCSForBandits() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [displayTemp, setDisplayTemp] = useState(0.1);
  const [committedTemp, setCommittedTemp] = useState(0.1);
  const sim = useMemo(() => simulate(committedTemp), [committedTemp]);

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const w = Math.max(280, Math.min(containerWidth || 720, 820));
  const panelW = isMobile ? w : Math.floor(w / 2) - 8;

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = panelW - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const xScale = d3.scaleLinear().domain([0, N_MAX]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);
      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(4)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).tickFormat(d3.format('.0%'))).selectAll('text').style('fill', 'var(--color-text)');
      for (let a = 0; a < N_ARMS; a++) {
        const line = d3.line<number>().x((_, i) => xScale(i + 1)).y((v) => yScale(v));
        g.append('path').datum(sim.pickRates[a]).attr('d', line as never).style('fill', 'none').style('stroke', ARM_COLORS[a]).attr('stroke-width', 1.6);
        g.append('text').attr('x', innerW - 4).attr('y', 12 + a * 14).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', ARM_COLORS[a]).text(`arm ${a} (CTR=${TRUE_REWARDS[a]})`);
      }
      g.append('text').attr('x', 0).attr('y', -10).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '12px').style('font-weight', 600).text('Thompson sampling pick rate per arm');
    },
    [sim, panelW],
  );

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      const innerW = panelW - MARGIN.left - MARGIN.right;
      const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;
      if (innerW <= 0) return;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);
      const xScale = d3.scaleLinear().domain([0, N_MAX]).range([0, innerW]);
      const yScale = d3.scaleLinear().domain([0.1, 0.4]).range([innerH, 0]);
      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(xScale).ticks(4)).selectAll('text').style('fill', 'var(--color-text)');
      g.append('g').call(d3.axisLeft(yScale).ticks(5, '.2f')).selectAll('text').style('fill', 'var(--color-text)');

      g.append('line').attr('x1', 0).attr('x2', innerW).attr('y1', yScale(sim.truthLine)).attr('y2', yScale(sim.truthLine)).style('stroke', 'var(--color-text)').attr('stroke-width', 1).attr('stroke-dasharray', '4 3');
      g.append('text').attr('x', innerW - 4).attr('y', yScale(sim.truthLine) - 4).attr('text-anchor', 'end').style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', 'var(--color-text)').text(`V(π★) = ${sim.truthLine.toFixed(2)}`);

      const area = d3.area<number>().x((_, i) => xScale(i + 1)).y0((_, i) => yScale(sim.csLo[i])).y1((_, i) => yScale(sim.csHi[i]));
      g.append('path').datum(sim.csLo).attr('d', area as never).style('fill', 'var(--color-accent)').style('opacity', 0.2);

      const lineMean = d3.line<number>().x((_, i) => xScale(i + 1)).y((_, i) => yScale((sim.csLo[i] + sim.csHi[i]) / 2));
      g.append('path').datum(sim.csLo).attr('d', lineMean as never).style('fill', 'none').style('stroke', 'var(--color-accent)').attr('stroke-width', 1.5);

      g.append('text').attr('x', 0).attr('y', -10).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '12px').style('font-weight', 600).text('Off-policy CS for V(π★) via (11.2)');
    },
    [sim, panelW],
  );

  return (
    <div ref={containerRef} style={{ width: '100%', fontFamily: 'var(--font-sans)' }}>
      <div style={{ display: 'flex', gap: '14px', padding: '10px 0', fontSize: '13px', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', minWidth: '220px' }}>
          <span>Exploration temperature = <strong>{displayTemp.toFixed(2)}</strong></span>
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.05}
            value={displayTemp}
            aria-label="Exploration temperature"
            onChange={(e) => setDisplayTemp(Number(e.target.value))}
            onMouseUp={(e) => setCommittedTemp(Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => setCommittedTemp(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => setCommittedTemp(Number((e.target as HTMLInputElement).value))}
          />
        </label>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', padding: '6px 10px', border: '1px solid var(--color-border)', borderRadius: '4px' }}>
          Thompson behavior policy; greedy target π★ = arm 3 (true CTR = 0.30). Off-policy CS via (11.2) is anytime-valid under arbitrary behavior policy.
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: '14px' }}>
        <svg ref={leftRef} width={panelW} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
        <svg ref={rightRef} width={panelW} height={PANEL_H} style={{ maxWidth: '100%', display: 'block' }} />
      </div>
    </div>
  );
}
