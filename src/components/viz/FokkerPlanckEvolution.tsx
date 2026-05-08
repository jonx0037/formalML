// =============================================================================
// FokkerPlanckEvolution.tsx
//
// §4 Fokker–Planck and stationary distributions, demonstrated on the OU
// process — the only nontrivial Langevin SDE with a closed-form transition
// density. Starting from x₀ = 0:
//   dX_t = -α X_t dt + σ dW_t
//   ⇒ p(x, t) = N(x; 0, σ²(1 - e^{-2αt})/(2α))
// As t → ∞, p(x, t) approaches the stationary N(0, σ²/(2α)) — the
// continuous-time analog of "the chain converges to π."
//
// Two panels:
//   (a) Density p(x, t) at the current scrub-bar t, with the stationary
//       limit overlaid as a ghost reference.
//   (b) Variance σ²(t) vs t (analytic), with the asymptote σ²/(2α).
//
// Controls:
//   - t slider (scrubbable, 0.05..5.0)
//   - α slider (drift rate, 0.5..3.0)
//   - σ slider (diffusion scale, 0.5..2.0)
//
// Computation: closed-form, no chain. Instant updates on slider drag.
// Static fallback: /images/topics/stochastic-gradient-mcmc/04_fokker_planck.png.
// =============================================================================

import { useEffect, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { ouDensity, paletteSGMCMC } from './shared/sgmcmc';

const DEFAULT_T = 0.5;
const DEFAULT_ALPHA = 1.0;
const DEFAULT_SIGMA = 1.0;
const X_LO = -4;
const X_HI = 4;
const T_MAX = 5.0;

const FIGURE_PATH = '/images/topics/stochastic-gradient-mcmc/04_fokker_planck.png';
const ALT =
  'Two-panel figure of the OU Fokker–Planck evolution. (a) Density p(x, t) at the current t with the stationary limit overlaid. (b) Variance σ²(t) vs t with the asymptote σ²/(2α).';
const CAPTION =
  'Figure 4. Fokker–Planck evolution for the Ornstein–Uhlenbeck SDE — the canonical case where the time-evolution of the density has a closed form. The variance grows from 0 as σ²(1 - e^{-2αt})/(2α) and saturates at σ²/(2α). Slide t to scrub through the evolution; α and σ control the convergence rate and stationary spread.';
const ARIA = 'Figure 4: Interactive Fokker–Planck (OU) density evolution';

export default function FokkerPlanckEvolution() {
  const { ref, width } = useResizeObserver<HTMLElement>();
  const [t, setT] = useState(DEFAULT_T);
  const [alpha, setAlpha] = useState(DEFAULT_ALPHA);
  const [sigma, setSigma] = useState(DEFAULT_SIGMA);
  // Derived
  const sigma2 = sigma * sigma;
  const stationaryVar = sigma2 / (2 * alpha);
  const currentVar = (sigma2 * (1 - Math.exp(-2 * alpha * t))) / (2 * alpha);

  return (
    <figure
      ref={ref as React.RefObject<HTMLElement>}
      aria-label={ARIA}
      style={{ width: '100%', margin: '2rem auto', display: 'flex', flexDirection: 'column', alignItems: 'center' }}
    >
      <div style={{ width: '100%', maxWidth: '880px' }}>
        <Controls t={t} alpha={alpha} sigma={sigma} onTChange={setT} onAlphaChange={setAlpha} onSigmaChange={setSigma} />
        <Panels t={t} alpha={alpha} sigma2={sigma2} stationaryVar={stationaryVar} currentVar={currentVar} width={width} />
      </div>
      <figcaption style={{ fontSize: '0.875rem', lineHeight: 1.5, color: 'var(--color-text-muted, #666)', marginTop: '0.75rem', textAlign: 'left', maxWidth: '880px', padding: '0 0.5rem' }}>{CAPTION}</figcaption>
    </figure>
  );
}

function Controls({
  t,
  alpha,
  sigma,
  onTChange,
  onAlphaChange,
  onSigmaChange,
}: {
  t: number;
  alpha: number;
  sigma: number;
  onTChange: (n: number) => void;
  onAlphaChange: (n: number) => void;
  onSigmaChange: (n: number) => void;
}) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem 1.25rem', alignItems: 'center', fontSize: '0.875rem', color: 'var(--color-text, #333)', marginBottom: '0.75rem' }}>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>t:</span>
        <input type="range" min={0.05} max={T_MAX} step={0.05} value={t}
          onChange={(e) => onTChange(Number(e.target.value))}
          style={{ width: '160px' }} aria-label="Time t" />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '3em' }}>{t.toFixed(2)}</span>
      </label>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>α:</span>
        <input type="range" min={0.5} max={3.0} step={0.1} value={alpha}
          onChange={(e) => onAlphaChange(Number(e.target.value))}
          style={{ width: '120px' }} aria-label="Drift rate alpha" />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '2.5em' }}>{alpha.toFixed(1)}</span>
      </label>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>σ:</span>
        <input type="range" min={0.5} max={2.0} step={0.1} value={sigma}
          onChange={(e) => onSigmaChange(Number(e.target.value))}
          style={{ width: '120px' }} aria-label="Diffusion scale sigma" />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '2.5em' }}>{sigma.toFixed(1)}</span>
      </label>
    </div>
  );
}

function Panels({ t, alpha, sigma2, stationaryVar, currentVar, width }: { t: number; alpha: number; sigma2: number; stationaryVar: number; currentVar: number; width: number }) {
  const layoutWidth = Math.max(width, 320);
  const stack = layoutWidth < 720;
  const panelW = stack ? layoutWidth : (layoutWidth - 16) / 2;
  const panelH = 240;
  return (
    <div style={{ display: 'flex', flexDirection: stack ? 'column' : 'row', gap: stack ? '12px' : '16px', alignItems: stack ? 'center' : 'flex-start' }}>
      <Panel title={`(a) p(x, t = ${t.toFixed(2)})`} width={panelW} height={panelH}>
        <DensityPanel t={t} alpha={alpha} sigma2={sigma2} stationaryVar={stationaryVar} width={panelW} height={panelH} />
      </Panel>
      <Panel title={`(b) σ²(t) → σ²/(2α) = ${stationaryVar.toFixed(3)}`} width={panelW} height={panelH}>
        <VariancePanel t={t} currentVar={currentVar} stationaryVar={stationaryVar} alpha={alpha} sigma2={sigma2} width={panelW} height={panelH} />
      </Panel>
    </div>
  );
}

function Panel({ title, width, height, children }: { title: string; width: number; height: number; children: React.ReactNode }) {
  return (
    <div style={{ width, display: 'flex', flexDirection: 'column' }}>
      <div style={{ fontSize: '0.825rem', fontWeight: 600, color: 'var(--color-text, #333)', marginBottom: '0.25rem' }}>{title}</div>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img">{children}</svg>
    </div>
  );
}

function DensityPanel({ t, alpha, sigma2, stationaryVar, width, height }: { t: number; alpha: number; sigma2: number; stationaryVar: number; width: number; height: number }) {
  const margin = { top: 8, right: 12, bottom: 28, left: 36 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const xS = d3.scaleLinear().domain([X_LO, X_HI]).range([0, innerW]);
  const refPts = Array.from({ length: 200 }, (_, i) => X_LO + ((X_HI - X_LO) * i) / 199);
  const currentDensity = refPts.map((x) => ouDensity(x, t, alpha, sigma2));
  const stationaryDensity = refPts.map((x) => Math.exp(-(x * x) / (2 * stationaryVar)) / Math.sqrt(2 * Math.PI * stationaryVar));
  const yMax = Math.max(...currentDensity, ...stationaryDensity) * 1.1;
  const yS = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);
  const linePath = (vals: number[]) => refPts.map((x, i) => `${i === 0 ? 'M' : 'L'}${xS(x)},${yS(vals[i])}`).join(' ');
  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      <path d={linePath(stationaryDensity)} fill="none" stroke={paletteSGMCMC.target} strokeWidth={1.4} strokeDasharray="4 3" opacity={0.7} />
      <path d={linePath(currentDensity)} fill="none" stroke={paletteSGMCMC.langevin} strokeWidth={2} />
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      {[-3, 0, 3].map((tk) => (
        <text key={tk} x={xS(tk)} y={innerH + 16} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">{tk}</text>
      ))}
      <text x={innerW / 2} y={innerH + 24} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">x</text>
      <text transform={`translate(${-26}, ${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">density</text>
      <text x={innerW - 6} y={14} textAnchor="end" fontSize={10} fill={paletteSGMCMC.langevin}>p(x, t)</text>
      <text x={innerW - 6} y={28} textAnchor="end" fontSize={10} fill={paletteSGMCMC.target}>π = lim_{'{t→∞}'}</text>
    </g>
  );
}

function VariancePanel({ t, currentVar, stationaryVar, alpha, sigma2, width, height }: { t: number; currentVar: number; stationaryVar: number; alpha: number; sigma2: number; width: number; height: number }) {
  const margin = { top: 8, right: 12, bottom: 28, left: 40 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const tPts = Array.from({ length: 200 }, (_, i) => (i / 199) * T_MAX);
  const variance = tPts.map((tt) => (sigma2 * (1 - Math.exp(-2 * alpha * tt))) / (2 * alpha));
  const yMax = stationaryVar * 1.15;
  const xS = d3.scaleLinear().domain([0, T_MAX]).range([0, innerW]);
  const yS = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);
  const linePath = tPts.map((tt, i) => `${i === 0 ? 'M' : 'L'}${xS(tt)},${yS(variance[i])}`).join(' ');
  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      <line x1={0} x2={innerW} y1={yS(stationaryVar)} y2={yS(stationaryVar)} stroke={paletteSGMCMC.target} strokeWidth={1.4} strokeDasharray="4 3" opacity={0.8} />
      <path d={linePath} fill="none" stroke={paletteSGMCMC.langevin} strokeWidth={2} />
      <circle cx={xS(t)} cy={yS(currentVar)} r={5} fill={paletteSGMCMC.biasBound} stroke="var(--color-surface, #fff)" strokeWidth={1.5} />
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      {[0, 1, 2, 3, 4, 5].map((tk) => (
        <text key={tk} x={xS(tk)} y={innerH + 16} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">{tk}</text>
      ))}
      <text x={innerW / 2} y={innerH + 24} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">t</text>
      <text transform={`translate(${-30}, ${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">σ²(t)</text>
      <text x={innerW - 6} y={yS(stationaryVar) - 4} textAnchor="end" fontSize={10} fill={paletteSGMCMC.target}>asymptote σ²/(2α)</text>
    </g>
  );
}
