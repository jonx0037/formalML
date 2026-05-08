// =============================================================================
// BNNCalibrationComparisonViz.tsx
//
// Interactive head-to-head calibration of six BNN methods on Two Moons. Two
// panels:
//   1. Reliability diagram (predicted-confidence x-axis, empirical-accuracy
//      y-axis) with one line per toggled-on method plus the y=x reference.
//   2. Bar chart of ECE / Brier × 10 / NLL × 10 grouped by method.
//
// Data: /sample-data/bayesian-neural-networks/calibration.json — emitted by
// notebooks/bayesian-neural-networks/precompute_viz_data.py. The JSON ships
// per-method test predictions; ECE/Brier/NLL are recomputed in-browser via
// bnnCalibrationDiagnostic on every bin-count or temperature-scaling change
// so the user sees the calibration metric respond to their slider.
// =============================================================================

import { useMemo } from 'react';
import * as d3 from 'd3';
import BNNInteractiveFigure from './BNNInteractiveFigure';
import {
  BCE_EPSILON,
  bnnCalibrationDiagnostic,
  paletteBNN,
  type BNNColorKey,
} from './shared/bayesian-ml';

interface CalibrationPayload {
  test_x: number[][];
  test_y: number[];
  n_test: number;
  probs: Record<string, number[]>;
  metrics: Record<
    string,
    {
      ece: number;
      brier: number;
      nll: number;
      accuracy: number;
      reliability_bins: { binConf: number; binAcc: number; binCount: number }[];
    }
  >;
}

interface ControlState {
  enabled: Record<string, boolean>;
  nBins: number;
  temperatureT: number;
  temperatureOn: boolean;
}

const METHOD_ORDER: { key: string; label: string; color: BNNColorKey }[] = [
  { key: 'point', label: 'Point estimate', color: 'point' },
  { key: 'laplace', label: 'Laplace', color: 'laplace' },
  { key: 'dropout', label: 'MC-dropout', color: 'dropout' },
  { key: 'ensemble', label: 'Deep ensemble', color: 'ensemble' },
  { key: 'sgld', label: 'SGLD', color: 'sgld' },
  { key: 'sghmc', label: 'SGHMC', color: 'sghmc' },
];

const initialState: ControlState = {
  enabled: METHOD_ORDER.reduce(
    (acc, m) => ({ ...acc, [m.key]: true }),
    {} as Record<string, boolean>,
  ),
  nBins: 15,
  temperatureT: 1.0,
  temperatureOn: false,
};

function applyTemperature(probs: number[], T: number): Float32Array {
  const out = new Float32Array(probs.length);
  for (let i = 0; i < probs.length; i++) {
    const p = Math.min(Math.max(probs[i], BCE_EPSILON), 1 - BCE_EPSILON);
    const logit = Math.log(p / (1 - p));
    const scaled = logit / T;
    out[i] = 1 / (1 + Math.exp(-scaled));
  }
  return out;
}

export default function BNNCalibrationComparisonViz() {
  return (
    <BNNInteractiveFigure<CalibrationPayload, ControlState>
      dataPath="/sample-data/bayesian-neural-networks/calibration.json"
      initialState={initialState}
      figurePath="/images/topics/bayesian-neural-networks/08_calibration_comparison.png"
      alt="Two panels: panel (a) reliability diagram with predicted confidence on the x-axis and empirical accuracy on the y-axis, with the diagonal y=x as the reference line and one connected line per method (point estimate, Laplace, MC-dropout, deep ensemble, SGLD, SGHMC); the point estimate's curve sits below the diagonal in mid-confidence bins (over-confidence) and the BNN methods' curves sit closer to the diagonal; panel (b) grouped bar chart of ECE, Brier × 10, and NLL × 10 for the six methods, with deep ensemble and SGHMC having the lowest values."
      caption="Figure 8. Head-to-head calibration on Two Moons. (a) Reliability diagram — distance from the y=x diagonal measures miscalibration. (b) Bar chart of ECE / Brier×10 / NLL×10 across methods. Toggle methods to compare; slide bin count to see binning sensitivity. The optional manual-temperature slider rescales logits as a sanity check; full temperature scaling fits T on a held-out set rather than letting the reader pick — that step is left for a v3 enhancement."
      ariaLabel="Figure 8: Interactive head-to-head calibration comparison across six methods"
      controls={(s, setS) => <CalibrationControls state={s} setState={setS} />}
      chart={(data, s, w) => <CalibrationChart data={data} state={s} width={w} />}
      // CalibrationChart doesn't drive state from inside the chart; it ignores
      // the fourth setState argument the substrate now passes.
      maxWidth={760}
    />
  );
}

function CalibrationControls({
  state,
  setState,
}: {
  state: ControlState;
  setState: (s: ControlState) => void;
}) {
  return (
    <div
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '0.75rem',
        fontSize: '0.875rem',
        color: 'var(--color-text, #333)',
        alignItems: 'center',
      }}
    >
      <span style={{ fontWeight: 600 }}>Methods:</span>
      {METHOD_ORDER.map((m) => (
        <label
          key={m.key}
          style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}
        >
          <input
            type="checkbox"
            checked={state.enabled[m.key]}
            onChange={(e) =>
              setState({
                ...state,
                enabled: { ...state.enabled, [m.key]: e.target.checked },
              })
            }
          />
          <span style={{ color: paletteBNN[m.color] }}>{m.label}</span>
        </label>
      ))}
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
        <span>Bins:</span>
        <input
          type="range"
          min={5}
          max={20}
          value={state.nBins}
          onChange={(e) => setState({ ...state, nBins: Number(e.target.value) })}
        />
        <span style={{ minWidth: '1.5em' }}>{state.nBins}</span>
      </label>
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
        <input
          type="checkbox"
          checked={state.temperatureOn}
          onChange={(e) => setState({ ...state, temperatureOn: e.target.checked })}
        />
        <span>Manual temperature</span>
      </label>
      {state.temperatureOn && (
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
          <span>T:</span>
          <input
            type="range"
            min={0.5}
            max={5.0}
            step={0.1}
            value={state.temperatureT}
            onChange={(e) =>
              setState({ ...state, temperatureT: Number(e.target.value) })
            }
          />
          <span style={{ minWidth: '2em' }}>{state.temperatureT.toFixed(1)}</span>
        </label>
      )}
    </div>
  );
}

function CalibrationChart({
  data,
  state,
  width,
}: {
  data: CalibrationPayload;
  state: ControlState;
  width: number;
}) {
  const recomputed = useMemo(() => {
    const labels = new Float32Array(data.test_y);
    const out: Record<
      string,
      {
        ece: number;
        brier: number;
        nll: number;
        accuracy: number;
        reliability: { binConf: number; binAcc: number; binCount: number }[];
      }
    > = {};
    for (const m of METHOD_ORDER) {
      const raw = data.probs[m.key];
      if (!raw) continue;
      const probs = state.temperatureOn
        ? applyTemperature(raw, state.temperatureT)
        : new Float32Array(raw);
      const mtr = bnnCalibrationDiagnostic(probs, labels, state.nBins);
      out[m.key] = {
        ece: mtr.ece,
        brier: mtr.brier,
        nll: mtr.nll,
        accuracy: mtr.accuracy,
        reliability: mtr.reliabilityBins,
      };
    }
    return out;
  }, [data, state.nBins, state.temperatureT, state.temperatureOn]);

  const layoutWidth = Math.max(width, 320);
  const halfWidth = (layoutWidth - 32) / 2;
  const panelHeight = 280;
  const margin = { top: 24, right: 14, bottom: 40, left: 44 };
  const innerW = halfWidth - margin.left - margin.right;
  const innerH = panelHeight - margin.top - margin.bottom;

  const xScale = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
  const yScale = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

  const enabledMethods = METHOD_ORDER.filter(
    (m) => state.enabled[m.key] && recomputed[m.key],
  );

  const metrics: { key: 'ece' | 'brier' | 'nll'; label: string; scale: number }[] = [
    { key: 'ece', label: 'ECE', scale: 1 },
    { key: 'brier', label: 'Brier×10', scale: 10 },
    { key: 'nll', label: 'NLL×10', scale: 10 },
  ];
  const barX0 = d3
    .scaleBand<string>()
    .domain(metrics.map((m) => m.label))
    .range([0, innerW])
    .padding(0.2);
  const barX1 = d3
    .scaleBand<string>()
    .domain(enabledMethods.map((m) => m.key))
    .range([0, barX0.bandwidth()])
    .padding(0.05);
  const barMaxValue =
    Math.max(
      ...enabledMethods.flatMap((m) => [
        recomputed[m.key].ece,
        recomputed[m.key].brier * 10,
        recomputed[m.key].nll * 10,
      ]),
      0.1,
    ) * 1.15;
  const barY = d3.scaleLinear().domain([0, barMaxValue]).range([innerH, 0]);

  return (
    <div style={{ display: 'flex', gap: '32px', flexWrap: 'wrap' }}>
      {/* Reliability diagram */}
      <svg
        width={halfWidth}
        height={panelHeight}
        viewBox={`0 0 ${halfWidth} ${panelHeight}`}
        style={{ background: 'var(--color-surface, transparent)' }}
      >
        <g transform={`translate(${margin.left},${margin.top})`}>
          <line
            x1={xScale(0)}
            y1={yScale(0)}
            x2={xScale(1)}
            y2={yScale(1)}
            stroke="var(--color-text-muted, #999)"
            strokeDasharray="3 3"
          />
          {enabledMethods.map((m) => {
            const bins = recomputed[m.key].reliability;
            const filtered = bins.filter((b) => b.binCount > 0);
            if (filtered.length === 0) return null;
            const line = d3
              .line<{ binConf: number; binAcc: number }>()
              .x((d) => xScale(d.binConf))
              .y((d) => yScale(d.binAcc));
            return (
              <g key={m.key}>
                <path
                  d={line(filtered) ?? ''}
                  fill="none"
                  stroke={paletteBNN[m.color]}
                  strokeWidth={2}
                />
                {filtered.map((b, i) => (
                  <circle
                    key={i}
                    cx={xScale(b.binConf)}
                    cy={yScale(b.binAcc)}
                    r={3}
                    fill={paletteBNN[m.color]}
                  />
                ))}
              </g>
            );
          })}
          <g transform={`translate(0, ${innerH})`}>
            {[0, 0.25, 0.5, 0.75, 1].map((t) => (
              <g key={t} transform={`translate(${xScale(t)}, 0)`}>
                <line y2={4} stroke="var(--color-text, #333)" />
                <text y={16} fontSize={10} textAnchor="middle" fill="var(--color-text, #333)">
                  {t.toFixed(2)}
                </text>
              </g>
            ))}
          </g>
          <g>
            {[0, 0.25, 0.5, 0.75, 1].map((t) => (
              <g key={t} transform={`translate(0, ${yScale(t)})`}>
                <line x2={-4} stroke="var(--color-text, #333)" />
                <text
                  x={-8}
                  dy="0.32em"
                  fontSize={10}
                  textAnchor="end"
                  fill="var(--color-text, #333)"
                >
                  {t.toFixed(2)}
                </text>
              </g>
            ))}
          </g>
          <text
            x={innerW / 2}
            y={innerH + 32}
            textAnchor="middle"
            fontSize={11}
            fill="var(--color-text, #333)"
          >
            Predicted confidence
          </text>
          <text
            transform={`translate(${-32}, ${innerH / 2}) rotate(-90)`}
            textAnchor="middle"
            fontSize={11}
            fill="var(--color-text, #333)"
          >
            Empirical accuracy
          </text>
          <text
            x={innerW / 2}
            y={-8}
            textAnchor="middle"
            fontSize={12}
            fontWeight={600}
            fill="var(--color-text, #333)"
          >
            Reliability diagram
          </text>
        </g>
      </svg>
      {/* Bar chart */}
      <svg
        width={halfWidth}
        height={panelHeight}
        viewBox={`0 0 ${halfWidth} ${panelHeight}`}
        style={{ background: 'var(--color-surface, transparent)' }}
      >
        <g transform={`translate(${margin.left},${margin.top})`}>
          {metrics.map((mt) => (
            <g key={mt.label} transform={`translate(${barX0(mt.label) ?? 0}, 0)`}>
              {enabledMethods.map((m) => {
                const v = recomputed[m.key][mt.key] * mt.scale;
                return (
                  <rect
                    key={m.key}
                    x={barX1(m.key) ?? 0}
                    y={barY(v)}
                    width={barX1.bandwidth()}
                    height={Math.max(innerH - barY(v), 0)}
                    fill={paletteBNN[m.color]}
                    opacity={0.85}
                  />
                );
              })}
              <text
                x={barX0.bandwidth() / 2}
                y={innerH + 16}
                fontSize={10}
                textAnchor="middle"
                fill="var(--color-text, #333)"
              >
                {mt.label}
              </text>
            </g>
          ))}
          <g>
            {[0, barMaxValue / 4, barMaxValue / 2, (3 * barMaxValue) / 4, barMaxValue].map((t) => (
              <g key={t} transform={`translate(0, ${barY(t)})`}>
                <line x2={-4} stroke="var(--color-text, #333)" />
                <text
                  x={-8}
                  dy="0.32em"
                  fontSize={10}
                  textAnchor="end"
                  fill="var(--color-text, #333)"
                >
                  {t.toFixed(2)}
                </text>
              </g>
            ))}
          </g>
          <text
            x={innerW / 2}
            y={-8}
            textAnchor="middle"
            fontSize={12}
            fontWeight={600}
            fill="var(--color-text, #333)"
          >
            Calibration metrics
          </text>
        </g>
      </svg>
    </div>
  );
}
