// =============================================================================
// NNGPSidebarViz.tsx
//
// Interactive function-space view of the BNN topic. Two panels:
//   1. Width-convergence — bar chart of empirical Var f(x_0) at six finite
//      widths h ∈ {50, 100, 200, 500, 1000, 2000}, with a horizontal reference
//      line for the closed-form NNGP arc-cosine K(x_0, x_0). The reader sees
//      the empirical variance converge to the closed-form as h grows.
//   2. NNGP-kernel GP regression — closed-form posterior mean ± 2σ band on a
//      small 1D synthetic dataset (8 training points), demonstrating that the
//      arc-cosine kernel is a usable GP kernel without any training.
//
// Data: /sample-data/bayesian-neural-networks/nngp.json — emitted by
// notebooks/bayesian-neural-networks/precompute_viz_data.py.
// =============================================================================

import * as d3 from 'd3';
import BNNInteractiveFigure from './BNNInteractiveFigure';
import { paletteBNN } from './shared/bayesian-ml';

// Layout constants — kept here rather than as magic numbers in the chart body.
const Y_AXIS_PADDING = 0.2;
const BAR_HEAD_ROOM = 1.2;

interface NNGPPayload {
  width_convergence: { width: number; empirical: number; closed_form_K00: number }[];
  regression: {
    x_train: number[];
    y_train: number[];
    x_grid: number[];
    mean: number[];
    std: number[];
  };
  kernel_params: { sigma_w2: number; sigma_b2: number };
}

interface ControlState {
  highlightedWidth: number;
  showSamples: boolean;
}

const initialState: ControlState = {
  highlightedWidth: 200,
  showSamples: true,
};

export default function NNGPSidebarViz() {
  return (
    <BNNInteractiveFigure<NNGPPayload, ControlState>
      dataPath="/sample-data/bayesian-neural-networks/nngp.json"
      initialState={initialState}
      figurePath="/images/topics/bayesian-neural-networks/09_nngp_sidebar.png"
      alt="Two panels: panel (a) bar chart of empirical Var f(x_0) at six widths h ∈ {50, 100, 200, 500, 1000, 2000} with a horizontal reference line for the closed-form NNGP arc-cosine kernel value at x_0; panel (b) NNGP-kernel GP regression posterior on a small synthetic regression dataset showing posterior mean and ±2σ band, with uncertainty growing between training points."
      caption="Figure 9. The function-space view. (a) Empirical Var f(x_0) of finite-width MLPs converges to the closed-form arc-cosine NNGP kernel value as h grows. (b) NNGP-kernel GP regression — closed-form, no-training, with uncertainty growing between training points. Toggle 95% bands; click a bar to highlight that width's deviation from the limit."
      ariaLabel="Figure 9: Interactive NNGP convergence and arc-cosine kernel regression"
      controls={(s, setS) => <NNGPControls state={s} setState={setS} />}
      chart={(data, s, w, setS) => (
        <NNGPChart data={data} state={s} width={w} setState={setS} />
      )}
      maxWidth={760}
    />
  );
}

function NNGPControls({
  state,
  setState,
}: {
  state: ControlState;
  setState: (s: ControlState) => void;
}) {
  const widths = [50, 100, 200, 500, 1000, 2000];
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
      <span style={{ fontWeight: 600 }}>Highlight width:</span>
      {widths.map((w) => (
        <button
          key={w}
          type="button"
          onClick={() => setState({ ...state, highlightedWidth: w })}
          style={{
            padding: '0.25rem 0.5rem',
            border: '1px solid var(--color-border, #ccc)',
            borderRadius: '4px',
            background:
              state.highlightedWidth === w ? paletteBNN.nngp : 'var(--color-surface, #fff)',
            color: state.highlightedWidth === w ? 'white' : 'var(--color-text, #333)',
            cursor: 'pointer',
            fontSize: '0.875rem',
          }}
        >
          h={w}
        </button>
      ))}
      <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
        <input
          type="checkbox"
          checked={state.showSamples}
          onChange={(e) => setState({ ...state, showSamples: e.target.checked })}
        />
        <span>Show ±2σ band</span>
      </label>
    </div>
  );
}

function NNGPChart({
  data,
  state,
  width,
  setState,
}: {
  data: NNGPPayload;
  state: ControlState;
  width: number;
  setState: (s: ControlState) => void;
}) {
  const layoutWidth = Math.max(width, 320);
  const halfWidth = (layoutWidth - 32) / 2;
  const panelHeight = 280;
  const margin = { top: 24, right: 14, bottom: 40, left: 50 };
  const innerW = halfWidth - margin.left - margin.right;
  const innerH = panelHeight - margin.top - margin.bottom;

  // Panel 1: width-convergence bars
  const widths = data.width_convergence.map((d) => d.width);
  const closedForm = data.width_convergence[0]?.closed_form_K00 ?? 1;
  const xBar = d3
    .scaleBand<number>()
    .domain(widths)
    .range([0, innerW])
    .padding(0.18);
  const yMaxBar =
    Math.max(...data.width_convergence.map((d) => d.empirical), closedForm) * BAR_HEAD_ROOM;
  const yBar = d3.scaleLinear().domain([0, yMaxBar]).range([innerH, 0]);

  // Panel 2: GP regression
  const xMin = Math.min(...data.regression.x_grid);
  const xMax = Math.max(...data.regression.x_grid);
  const yAll = [
    ...data.regression.y_train,
    ...data.regression.mean.map((m, i) => m + 2 * data.regression.std[i]),
    ...data.regression.mean.map((m, i) => m - 2 * data.regression.std[i]),
  ];
  const yMinR = Math.min(...yAll) - Y_AXIS_PADDING;
  const yMaxR = Math.max(...yAll) + Y_AXIS_PADDING;
  const xR = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]);
  const yR = d3.scaleLinear().domain([yMinR, yMaxR]).range([innerH, 0]);
  const meanLine = d3
    .line<number>()
    .x((_, i) => xR(data.regression.x_grid[i]))
    .y((m) => yR(m));
  const bandArea = d3
    .area<number>()
    .x((_, i) => xR(data.regression.x_grid[i]))
    .y0((_, i) => yR(data.regression.mean[i] - 2 * data.regression.std[i]))
    .y1((_, i) => yR(data.regression.mean[i] + 2 * data.regression.std[i]));

  return (
    <div style={{ display: 'flex', gap: '32px', flexWrap: 'wrap' }}>
      {/* Width convergence */}
      <svg
        width={halfWidth}
        height={panelHeight}
        viewBox={`0 0 ${halfWidth} ${panelHeight}`}
        style={{ background: 'var(--color-surface, transparent)' }}
      >
        <g transform={`translate(${margin.left},${margin.top})`}>
          {data.width_convergence.map((d) => {
            const isHighlight = d.width === state.highlightedWidth;
            return (
              <g key={d.width}>
                <rect
                  x={xBar(d.width) ?? 0}
                  y={yBar(d.empirical)}
                  width={xBar.bandwidth()}
                  height={Math.max(innerH - yBar(d.empirical), 0)}
                  fill={paletteBNN.nngp}
                  opacity={isHighlight ? 1 : 0.5}
                  stroke={isHighlight ? 'var(--color-text, #333)' : 'none'}
                  strokeWidth={isHighlight ? 1.5 : 0}
                  style={{ cursor: 'pointer' }}
                  role="button"
                  tabIndex={0}
                  aria-label={`Highlight width ${d.width}`}
                  onClick={() => setState({ ...state, highlightedWidth: d.width })}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      setState({ ...state, highlightedWidth: d.width });
                    }
                  }}
                />
                <text
                  x={(xBar(d.width) ?? 0) + xBar.bandwidth() / 2}
                  y={innerH + 14}
                  fontSize={10}
                  textAnchor="middle"
                  fill="var(--color-text, #333)"
                >
                  {d.width}
                </text>
                {isHighlight && (
                  <text
                    x={(xBar(d.width) ?? 0) + xBar.bandwidth() / 2}
                    y={yBar(d.empirical) - 4}
                    fontSize={10}
                    textAnchor="middle"
                    fontWeight={600}
                    fill="var(--color-text, #333)"
                  >
                    {d.empirical.toFixed(3)}
                  </text>
                )}
              </g>
            );
          })}
          {/* Closed-form K(x_0,x_0) reference line */}
          <line
            x1={0}
            x2={innerW}
            y1={yBar(closedForm)}
            y2={yBar(closedForm)}
            stroke={paletteBNN.posterior}
            strokeWidth={2}
            strokeDasharray="4 3"
          />
          <text
            x={innerW - 4}
            y={yBar(closedForm) - 4}
            textAnchor="end"
            fontSize={10}
            fill="var(--color-text-muted, #666)"
          >
            K(x₀,x₀)={closedForm.toFixed(3)}
          </text>
          {/* Y axis */}
          <g>
            {[0, yMaxBar / 2, yMaxBar].map((t) => (
              <g key={t} transform={`translate(0, ${yBar(t)})`}>
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
            Hidden width h
          </text>
          <text
            transform={`translate(${-36}, ${innerH / 2}) rotate(-90)`}
            textAnchor="middle"
            fontSize={11}
            fill="var(--color-text, #333)"
          >
            Empirical Var f(x₀)
          </text>
          <text
            x={innerW / 2}
            y={-8}
            textAnchor="middle"
            fontSize={12}
            fontWeight={600}
            fill="var(--color-text, #333)"
          >
            Width convergence to NNGP
          </text>
        </g>
      </svg>
      {/* GP regression */}
      <svg
        width={halfWidth}
        height={panelHeight}
        viewBox={`0 0 ${halfWidth} ${panelHeight}`}
        style={{ background: 'var(--color-surface, transparent)' }}
      >
        <g transform={`translate(${margin.left},${margin.top})`}>
          {state.showSamples && (
            <path
              d={bandArea(data.regression.mean) ?? ''}
              fill={paletteBNN.nngp}
              opacity={0.18}
            />
          )}
          <path
            d={meanLine(data.regression.mean) ?? ''}
            fill="none"
            stroke={paletteBNN.nngp}
            strokeWidth={2}
          />
          {data.regression.x_train.map((x, i) => (
            <circle
              key={i}
              cx={xR(x)}
              cy={yR(data.regression.y_train[i])}
              r={4}
              fill={paletteBNN.data}
              stroke="white"
              strokeWidth={1.5}
            />
          ))}
          {/* Axes */}
          <g transform={`translate(0, ${innerH})`}>
            {[xMin, (xMin + xMax) / 2, xMax].map((t) => (
              <g key={t} transform={`translate(${xR(t)}, 0)`}>
                <line y2={4} stroke="var(--color-text, #333)" />
                <text
                  y={16}
                  fontSize={10}
                  textAnchor="middle"
                  fill="var(--color-text, #333)"
                >
                  {t.toFixed(1)}
                </text>
              </g>
            ))}
          </g>
          <g>
            {[yMinR, (yMinR + yMaxR) / 2, yMaxR].map((t) => (
              <g key={t} transform={`translate(0, ${yR(t)})`}>
                <line x2={-4} stroke="var(--color-text, #333)" />
                <text
                  x={-8}
                  dy="0.32em"
                  fontSize={10}
                  textAnchor="end"
                  fill="var(--color-text, #333)"
                >
                  {t.toFixed(1)}
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
            x
          </text>
          <text
            transform={`translate(${-36}, ${innerH / 2}) rotate(-90)`}
            textAnchor="middle"
            fontSize={11}
            fill="var(--color-text, #333)"
          >
            f(x)
          </text>
          <text
            x={innerW / 2}
            y={-8}
            textAnchor="middle"
            fontSize={12}
            fontWeight={600}
            fill="var(--color-text, #333)"
          >
            Arc-cosine NNGP regression
          </text>
        </g>
      </svg>
    </div>
  );
}
