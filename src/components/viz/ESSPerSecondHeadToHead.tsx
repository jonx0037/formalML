// =============================================================================
// ESSPerSecondHeadToHead.tsx
//
// §12 Head-to-head between SG-MCMC (SGLD, SGHMC) and NUTS on Bayesian
// logistic regression with D = 10 features and N ∈ {500, 2k, 10k, 50k}
// synthetic observations. Fetches precomputed JSON because NUTS via PyMC
// at N = 50,000 is infeasible to run in-browser (genuinely non-conjugate).
//
// Two panels:
//   (a) ESS per second of wall-clock time vs N (log-log). SG-MCMC step sizes
//       are scaled to the posterior curvature, which grows linearly in N
//       (SGLD η ∝ 1/N; SGHMC η ∝ 1/√N with C ∝ √N). NUTS leads at N = 500;
//       SG-MCMC leads from N = 2,000 on and the gap widens with N, because
//       NUTS pays O(N) per gradient.
//   (b) Posterior mean ± 1 sd at N_max for the first 5 components, against
//       the true β. All three methods recover the means; at fixed B = 32 the
//       SG-MCMC spread is 5–10× NUTS's — the gradient-noise bias of (8.3),
//       which ESS/sec does not see. With a FIXED η instead, SGLD crosses its
//       stability limit ηλ_max < 2 near N = 10,000 and both chains diverge.
//
// Data: public/sample-data/stochastic-gradient-mcmc/head_to_head.json, written
// by notebooks/stochastic-gradient-mcmc/precompute_head_to_head.py (same code
// as notebook cell 26). ess_per_sec and wall_clock_sec are machine-dependent.
// Controls: N selector. Other parameters fixed (this is a precomputed sweep).
// Static fallback: /images/topics/stochastic-gradient-mcmc/12_head_to_head.png.
// =============================================================================

import { useEffect, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { paletteSGMCMC } from './shared/sgmcmc';

interface HeadToHeadPayload {
  true_beta_first5: number[];
  N_grid: number[];
  methods: string[];
  ess_per_sec: { SGLD: number[]; SGHMC: number[]; NUTS: number[] };
  wall_clock_sec: { SGLD: number[]; SGHMC: number[]; NUTS: number[] };
  posterior_means_at_N_max: { SGLD: number[]; SGHMC: number[]; NUTS: number[] };
  posterior_sd_at_N_max?: { SGLD: number[]; SGHMC: number[]; NUTS: number[] };
}

const FIGURE_PATH = '/images/topics/stochastic-gradient-mcmc/12_head_to_head.png';
const ALT =
  'Three-panel head-to-head benchmark on Bayesian logistic regression with step sizes scaled to N. (a) ESS per second vs dataset size N: NUTS leads at N = 500, SGLD and SGHMC lead from N = 2,000 on and the gap widens with N. (b) Posterior mean plus or minus one standard deviation at N = 50,000 for the first five coefficients: all methods recover the true means, but the SG-MCMC intervals are several times wider than NUTS\'s. (c) Wall-clock time vs N: NUTS grows roughly linearly in N while SG-MCMC stays flat.';
const CAPTION =
  'Figure 12. ESS per second vs N for SGLD, SGHMC and NUTS on Bayesian logistic regression, with the SG-MCMC step sizes scaled to the posterior curvature (∝ N). NUTS leads at N = 500 by about 1.5×; SG-MCMC leads from N = 2,000 on, by about 90× (SGLD) and 56× (SGHMC) at N = 50,000, because NUTS pays O(N) per gradient. Panel (b): all three recover the posterior means, but at fixed B = 32 the SG-MCMC posteriors are 5–10× too wide at N = 50,000 — the gradient-noise bias of eq. (8.3), invisible to ESS per second. With a fixed step size instead, SGLD crosses its stability limit near N = 10,000 and the chains diverge. Timings come from one machine.';
const ARIA = 'Figure 12: SG-MCMC vs NUTS head-to-head on Bayesian logistic regression';

export default function ESSPerSecondHeadToHead() {
  const { ref, width } = useResizeObserver<HTMLElement>();
  const [payload, setPayload] = useState<HeadToHeadPayload | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [nIndex, setNIndex] = useState(3); // default to N_max

  useEffect(() => {
    let alive = true;
    fetch('/sample-data/stochastic-gradient-mcmc/head_to_head.json')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: HeadToHeadPayload) => {
        if (alive) setPayload(data);
      })
      .catch((e) => {
        if (alive) setLoadError(String(e));
      });
    return () => {
      alive = false;
    };
  }, []);

  return (
    <figure
      ref={ref as React.RefObject<HTMLElement>}
      aria-label={ARIA}
      style={{ width: '100%', margin: '2rem auto', display: 'flex', flexDirection: 'column', alignItems: 'center' }}
    >
      <div style={{ width: '100%', maxWidth: '880px' }}>
        {!payload ? (
          <Fallback label={loadError ? `Could not load benchmark data: ${loadError}` : 'Loading head-to-head benchmark JSON…'} />
        ) : (
          <>
            <Controls payload={payload} nIndex={nIndex} onNIndexChange={setNIndex} />
            <Panels payload={payload} nIndex={nIndex} width={width} />
          </>
        )}
      </div>
      <figcaption style={{ fontSize: '0.875rem', lineHeight: 1.5, color: 'var(--color-text-muted, #666)', marginTop: '0.75rem', textAlign: 'left', maxWidth: '880px', padding: '0 0.5rem' }}>{CAPTION}</figcaption>
    </figure>
  );
}

function Fallback({ label }: { label: string }) {
  return (
    <div style={{ position: 'relative', width: '100%' }}>
      <img src={FIGURE_PATH} alt={ALT} loading="lazy" style={{ width: '100%', height: 'auto', backgroundColor: 'var(--color-surface, transparent)', borderRadius: '0.25rem', boxShadow: '0 1px 2px var(--color-shadow, rgba(0, 0, 0, 0.04))', opacity: 0.55 }} />
      <div role="status" aria-live="polite" style={{ position: 'absolute', left: '50%', top: '50%', transform: 'translate(-50%, -50%)', padding: '0.5rem 0.875rem', background: 'var(--color-surface, #fff)', border: '1px solid var(--color-border, #ccc)', borderRadius: '4px', fontSize: '0.875rem', color: 'var(--color-text, #333)', boxShadow: '0 2px 6px rgba(0,0,0,0.08)' }}>{label}</div>
    </div>
  );
}

function Controls({ payload, nIndex, onNIndexChange }: { payload: HeadToHeadPayload; nIndex: number; onNIndexChange: (i: number) => void }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem 1.25rem', alignItems: 'center', fontSize: '0.875rem', color: 'var(--color-text, #333)', marginBottom: '0.75rem' }}>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
        <span>N (highlight):</span>
        <input
          type="range"
          min={0}
          max={payload.N_grid.length - 1}
          step={1}
          value={nIndex}
          onChange={(e) => onNIndexChange(Number(e.target.value))}
          style={{ width: '160px' }}
          aria-label="Highlight N value"
        />
        <span style={{ fontVariantNumeric: 'tabular-nums', minWidth: '5em' }}>{payload.N_grid[nIndex].toLocaleString()}</span>
      </label>
    </div>
  );
}

function Panels({ payload, nIndex, width }: { payload: HeadToHeadPayload; nIndex: number; width: number }) {
  const layoutWidth = Math.max(width, 320);
  const stack = layoutWidth < 720;
  const panelW = stack ? layoutWidth : (layoutWidth - 16) / 2;
  const panelH = 280;
  return (
    <div style={{ display: 'flex', flexDirection: stack ? 'column' : 'row', gap: stack ? '12px' : '16px', alignItems: stack ? 'center' : 'flex-start' }}>
      <Panel title="(a) ESS / sec vs N (log-log)" width={panelW} height={panelH}>
        <ESSPanel payload={payload} nIndex={nIndex} width={panelW} height={panelH} />
      </Panel>
      <Panel title={`(b) Posterior mean ± 1 sd at N = ${payload.N_grid[payload.N_grid.length - 1].toLocaleString()} (first 5 components)`} width={panelW} height={panelH}>
        <PosteriorMeansPanel payload={payload} width={panelW} height={panelH} />
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

function ESSPanel({ payload, nIndex, width, height }: { payload: HeadToHeadPayload; nIndex: number; width: number; height: number }) {
  const margin = { top: 12, right: 100, bottom: 28, left: 50 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const allValues = [
    ...payload.ess_per_sec.SGLD,
    ...payload.ess_per_sec.SGHMC,
    ...payload.ess_per_sec.NUTS,
  ].filter((v) => v > 0);
  const yMin = Math.min(...allValues) * 0.6;
  const yMax = Math.max(...allValues) * 1.4;

  const xS = d3.scaleLog().domain([payload.N_grid[0] * 0.7, payload.N_grid[payload.N_grid.length - 1] * 1.3]).range([0, innerW]);
  const yS = d3.scaleLog().domain([yMin, yMax]).range([innerH, 0]);

  const seriesPath = (vals: number[]) =>
    payload.N_grid.map((n, i) => `${i === 0 ? 'M' : 'L'}${xS(n)},${yS(Math.max(vals[i], yMin))}`).join(' ');

  const fmtPow = (v: number) => `10^${Math.round(Math.log10(v))}`;
  const xTicks = [1e3, 1e4, 1e5].filter((t) => t >= xS.domain()[0] && t <= xS.domain()[1]);
  const yTicks = [1e1, 1e2, 1e3, 1e4, 1e5].filter((t) => t >= yS.domain()[0] && t <= yS.domain()[1]);

  const series: { name: 'SGLD' | 'SGHMC' | 'NUTS'; color: string }[] = [
    { name: 'SGLD', color: paletteSGMCMC.sgld },
    { name: 'SGHMC', color: paletteSGMCMC.sghmc },
    { name: 'NUTS', color: paletteSGMCMC.nuts },
  ];

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      {xTicks.map((t) => (
        <g key={`xg-${t}`}>
          <line x1={xS(t)} x2={xS(t)} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" strokeOpacity={0.15} />
          <text x={xS(t)} y={innerH + 16} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">{fmtPow(t)}</text>
        </g>
      ))}
      {yTicks.map((t) => (
        <g key={`yg-${t}`}>
          <line x1={0} x2={innerW} y1={yS(t)} y2={yS(t)} stroke="var(--color-text-muted, #888)" strokeOpacity={0.15} />
          <text x={-6} y={yS(t) + 3} textAnchor="end" fontSize={10} fill="var(--color-text, #333)">{fmtPow(t)}</text>
        </g>
      ))}
      {/* Highlighted N */}
      <line
        x1={xS(payload.N_grid[nIndex])}
        x2={xS(payload.N_grid[nIndex])}
        y1={0}
        y2={innerH}
        stroke={paletteSGMCMC.biasBound}
        strokeOpacity={0.6}
        strokeWidth={1.5}
      />
      {series.map(({ name, color }) => (
        <g key={name}>
          <path d={seriesPath(payload.ess_per_sec[name])} fill="none" stroke={color} strokeWidth={1.6} />
          {payload.N_grid.map((n, i) => (
            <circle
              key={i}
              cx={xS(n)}
              cy={yS(Math.max(payload.ess_per_sec[name][i], yMin))}
              r={i === nIndex ? 6 : 3.5}
              fill={i === nIndex ? color : color}
              stroke={i === nIndex ? 'var(--color-surface, #fff)' : undefined}
              strokeWidth={i === nIndex ? 1.5 : 0}
            />
          ))}
        </g>
      ))}
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <text x={innerW / 2} y={innerH + 24} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">N (log)</text>
      <text transform={`translate(${-38}, ${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">ESS / sec (log)</text>
      <g transform={`translate(${innerW + 8}, 12)`}>
        <rect x={0} y={-6} width={92} height={56} fill="var(--color-surface, #fff)" stroke="var(--color-border, #ccc)" />
        {series.map(({ name, color }, i) => (
          <g key={name} transform={`translate(0, ${i * 18})`}>
            <line x1={4} x2={20} y1={4} y2={4} stroke={color} strokeWidth={2} />
            <text x={24} y={7} fontSize={10} fill="var(--color-text, #333)">{name}</text>
          </g>
        ))}
      </g>
    </g>
  );
}

function PosteriorMeansPanel({ payload, width, height }: { payload: HeadToHeadPayload; width: number; height: number }) {
  const margin = { top: 12, right: 100, bottom: 28, left: 50 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const D = payload.true_beta_first5.length;
  const truth = payload.true_beta_first5;
  const means = payload.posterior_means_at_N_max;
  const sds = payload.posterior_sd_at_N_max;
  const sdOf = (name: 'SGLD' | 'SGHMC' | 'NUTS', i: number) => (sds ? sds[name][i] : 0);

  const series: { name: 'SGLD' | 'SGHMC' | 'NUTS'; color: string; offset: number }[] = [
    { name: 'SGLD', color: paletteSGMCMC.sgld, offset: -0.27 },
    { name: 'SGHMC', color: paletteSGMCMC.sghmc, offset: 0.0 },
    { name: 'NUTS', color: paletteSGMCMC.nuts, offset: 0.27 },
  ];

  // Linear axis spanning every mean ± 1 sd and the truth: the point of the panel is that the
  // SG-MCMC means agree with NUTS while their spread (posterior sd) is several times wider.
  const ends = series.flatMap(({ name }) =>
    means[name].flatMap((m, i) => [m - sdOf(name, i), m + sdOf(name, i)]),
  );
  const lo = Math.min(...ends, ...truth, 0);
  const hi = Math.max(...ends, ...truth, 0);
  const pad = 0.08 * (hi - lo || 1);
  const yS = d3.scaleLinear().domain([lo - pad, hi + pad]).range([innerH, 0]).nice();
  const xS = d3
    .scaleBand<number>()
    .domain(Array.from({ length: D }, (_, i) => i))
    .range([0, innerW])
    .padding(0.25);
  const bw = xS.bandwidth() ?? 30;
  const center = (i: number) => (xS(i) ?? 0) + bw / 2;
  const yTicks = yS.ticks(5);

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      <rect x={0} y={0} width={innerW} height={innerH} fill="var(--color-surface, #fafafa)" />
      {yTicks.map((t) => (
        <g key={`yt-${t}`}>
          <line x1={0} x2={innerW} y1={yS(t)} y2={yS(t)} stroke="var(--color-text-muted, #888)" strokeOpacity={t === 0 ? 0.5 : 0.15} />
          <text x={-6} y={yS(t) + 3} textAnchor="end" fontSize={10} fill="var(--color-text, #333)">{t}</text>
        </g>
      ))}
      {/* Truth markers (black diamonds) */}
      {truth.map((v, i) => (
        <polygon
          key={`truth-${i}`}
          points={`${center(i)},${yS(v) - 5} ${center(i) + 5},${yS(v)} ${center(i)},${yS(v) + 5} ${center(i) - 5},${yS(v)}`}
          fill="black"
        />
      ))}
      {series.map(({ name, color, offset }) =>
        means[name].map((m, i) => {
          const cx = center(i) + offset * bw;
          const s = sdOf(name, i);
          return (
            <g key={`${name}-${i}`}>
              {s > 0 && (
                <>
                  <line x1={cx} x2={cx} y1={yS(m - s)} y2={yS(m + s)} stroke={color} strokeWidth={1.6} />
                  <line x1={cx - 3} x2={cx + 3} y1={yS(m - s)} y2={yS(m - s)} stroke={color} strokeWidth={1.6} />
                  <line x1={cx - 3} x2={cx + 3} y1={yS(m + s)} y2={yS(m + s)} stroke={color} strokeWidth={1.6} />
                </>
              )}
              <circle cx={cx} cy={yS(m)} r={3.5} fill={color} />
            </g>
          );
        }),
      )}
      <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--color-text-muted, #888)" />
      <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--color-text-muted, #888)" />
      {Array.from({ length: D }, (_, i) => (
        <text key={i} x={center(i)} y={innerH + 16} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
          β_{i + 1}
        </text>
      ))}
      <text transform={`translate(${-38}, ${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="var(--color-text, #333)">
        {sds ? 'posterior mean ± 1 sd' : 'posterior mean'}
      </text>
      <g transform={`translate(${innerW + 8}, 12)`}>
        <rect x={0} y={-6} width={92} height={70} fill="var(--color-surface, #fff)" stroke="var(--color-border, #ccc)" />
        {series.map(({ name, color }, i) => (
          <g key={name} transform={`translate(0, ${i * 18})`}>
            <line x1={10} x2={10} y1={-2} y2={10} stroke={color} strokeWidth={1.6} />
            <circle cx={10} cy={4} r={3} fill={color} />
            <text x={20} y={7} fontSize={10} fill="var(--color-text, #333)">{name}</text>
          </g>
        ))}
        <g transform={`translate(0, 56)`}>
          <polygon points={`10,-3 14,0 10,3 6,0`} fill="black" />
          <text x={20} y={3} fontSize={10} fill="var(--color-text, #333)">truth</text>
        </g>
      </g>
    </g>
  );
}
