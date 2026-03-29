import { useState, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  circlePoints,
  stabilitySigmas,
  stabilityBottleneckDistances,
  perturbationOffsetsX,
  perturbationOffsetsY,
} from '../../data/statistical-tda-data';

const MARGIN = { top: 20, right: 16, bottom: 32, left: 40 };
const SCATTER_HEIGHT = 300;
const CHART_HEIGHT = 160;

export default function StabilityExplorer() {
  const { ref, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [sigma, setSigma] = useState(0.15);

  const handleSigmaChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setSigma(parseFloat(e.target.value));
  }, []);

  // Compute perturbed points
  const perturbedPoints = useMemo(() => {
    return circlePoints.map((p, i) => ({
      x: Math.round((p.x + perturbationOffsetsX[i] * sigma) * 100) / 100,
      y: Math.round((p.y + perturbationOffsetsY[i] * sigma) * 100) / 100,
    }));
  }, [sigma]);

  // Interpolate bottleneck distance from pre-computed data
  const bottleneckDistance = useMemo(() => {
    const sigmas = stabilitySigmas;
    const dists = stabilityBottleneckDistances;

    if (sigma <= sigmas[0]) return dists[0];
    if (sigma >= sigmas[sigmas.length - 1]) return dists[dists.length - 1];

    for (let i = 0; i < sigmas.length - 1; i++) {
      if (sigma >= sigmas[i] && sigma <= sigmas[i + 1]) {
        const t = (sigma - sigmas[i]) / (sigmas[i + 1] - sigmas[i]);
        return Math.round((dists[i] + t * (dists[i + 1] - dists[i])) * 1000) / 1000;
      }
    }
    return 0;
  }, [sigma]);

  // Layout
  const panelWidth = useMemo(() => {
    if (!containerWidth) return 400;
    return Math.min(containerWidth, 560);
  }, [containerWidth]);

  const innerW = panelWidth - MARGIN.left - MARGIN.right;
  const scatterInnerH = SCATTER_HEIGHT - MARGIN.top - MARGIN.bottom;
  const chartInnerH = CHART_HEIGHT - MARGIN.top - MARGIN.bottom;

  // Scales for scatter plot
  const scatterScales = useMemo(() => {
    const pad = 0.3;
    const xScale = d3.scaleLinear().domain([-1 - pad, 1 + pad]).range([MARGIN.left, MARGIN.left + innerW]);
    const yScale = d3.scaleLinear().domain([-1 - pad, 1 + pad]).range([MARGIN.top + scatterInnerH, MARGIN.top]);
    return { xScale, yScale };
  }, [innerW, scatterInnerH]);

  // Scales for line chart
  const chartScales = useMemo(() => {
    const xScale = d3.scaleLinear().domain([0, 0.5]).range([MARGIN.left, MARGIN.left + innerW]);
    const yScale = d3.scaleLinear().domain([0, 0.4]).range([MARGIN.top + chartInnerH, MARGIN.top]);
    return { xScale, yScale };
  }, [innerW, chartInnerH]);

  // Tick arrays
  const scatterXTicks = scatterScales.xScale.ticks(5);
  const scatterYTicks = scatterScales.yScale.ticks(5);
  const chartXTicks = chartScales.xScale.ticks(5);
  const chartYTicks = chartScales.yScale.ticks(4);

  // Line path for bottleneck chart
  const linePath = useMemo(() => {
    const { xScale, yScale } = chartScales;
    const lineGen = d3.line<number>()
      .x((_, i) => xScale(stabilitySigmas[i]))
      .y((d) => yScale(d));
    return lineGen(stabilityBottleneckDistances) ?? '';
  }, [chartScales]);

  return (
    <div ref={ref} className="w-full space-y-3">
      {/* Slider */}
      <div className="flex items-center gap-3">
        <label
          className="text-xs font-medium whitespace-nowrap min-w-[120px]"
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          Perturbation σ = {sigma.toFixed(2)}
        </label>
        <input
          type="range"
          min={0}
          max={0.5}
          step={0.01}
          value={sigma}
          onChange={handleSigmaChange}
          className="w-full accent-[var(--color-accent)]"
        />
      </div>

      {/* Scatter plot */}
      <svg role="img" aria-label="Stability explorer visualization (panel 1 of 2)"
        width={panelWidth}
        height={SCATTER_HEIGHT}
        className="rounded-lg border border-[var(--color-border)]"
      >
        {/* Axes */}
        {scatterXTicks.map((t) => (
          <g key={`sx-${t}`}>
            <line
              x1={Math.round(scatterScales.xScale(t) * 100) / 100}
              y1={MARGIN.top}
              x2={Math.round(scatterScales.xScale(t) * 100) / 100}
              y2={MARGIN.top + scatterInnerH}
              stroke="var(--color-border)"
              strokeOpacity={0.3}
            />
            <text
              x={Math.round(scatterScales.xScale(t) * 100) / 100}
              y={MARGIN.top + scatterInnerH + 14}
              textAnchor="middle"
              fontSize={9}
              fill="var(--color-text)"
              opacity={0.5}
              style={{ fontFamily: 'var(--font-sans)' }}
            >
              {t}
            </text>
          </g>
        ))}
        {scatterYTicks.map((t) => (
          <g key={`sy-${t}`}>
            <line
              x1={MARGIN.left}
              y1={Math.round(scatterScales.yScale(t) * 100) / 100}
              x2={MARGIN.left + innerW}
              y2={Math.round(scatterScales.yScale(t) * 100) / 100}
              stroke="var(--color-border)"
              strokeOpacity={0.3}
            />
            <text
              x={MARGIN.left - 6}
              y={Math.round(scatterScales.yScale(t) * 100) / 100 + 3}
              textAnchor="end"
              fontSize={9}
              fill="var(--color-text)"
              opacity={0.5}
              style={{ fontFamily: 'var(--font-sans)' }}
            >
              {t}
            </text>
          </g>
        ))}

        {/* Points */}
        {perturbedPoints.map((p, i) => (
          <circle
            key={i}
            cx={Math.round(scatterScales.xScale(p.x) * 100) / 100}
            cy={Math.round(scatterScales.yScale(p.y) * 100) / 100}
            r={3.5}
            fill="steelblue"
            fillOpacity={0.75}
            stroke="var(--color-surface)"
            strokeWidth={0.5}
          />
        ))}
      </svg>

      {/* Bottleneck distance chart */}
      <div className="space-y-1">
        <p
          className="text-xs font-medium opacity-70"
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          Bottleneck distance d<sub>B</sub> = {bottleneckDistance.toFixed(3)}
        </p>
        <svg role="img" aria-label="Stability explorer visualization (panel 2 of 2)"
          width={panelWidth}
          height={CHART_HEIGHT}
          className="rounded-lg border border-[var(--color-border)]"
        >
          {/* Grid */}
          {chartXTicks.map((t) => (
            <g key={`cx-${t}`}>
              <line
                x1={Math.round(chartScales.xScale(t) * 100) / 100}
                y1={MARGIN.top}
                x2={Math.round(chartScales.xScale(t) * 100) / 100}
                y2={MARGIN.top + chartInnerH}
                stroke="var(--color-border)"
                strokeOpacity={0.3}
              />
              <text
                x={Math.round(chartScales.xScale(t) * 100) / 100}
                y={MARGIN.top + chartInnerH + 14}
                textAnchor="middle"
                fontSize={9}
                fill="var(--color-text)"
                opacity={0.5}
                style={{ fontFamily: 'var(--font-sans)' }}
              >
                {t}
              </text>
            </g>
          ))}
          {chartYTicks.map((t) => (
            <g key={`cy-${t}`}>
              <line
                x1={MARGIN.left}
                y1={Math.round(chartScales.yScale(t) * 100) / 100}
                x2={MARGIN.left + innerW}
                y2={Math.round(chartScales.yScale(t) * 100) / 100}
                stroke="var(--color-border)"
                strokeOpacity={0.3}
              />
              <text
                x={MARGIN.left - 6}
                y={Math.round(chartScales.yScale(t) * 100) / 100 + 3}
                textAnchor="end"
                fontSize={9}
                fill="var(--color-text)"
                opacity={0.5}
                style={{ fontFamily: 'var(--font-sans)' }}
              >
                {t}
              </text>
            </g>
          ))}

          {/* Axis labels */}
          <text
            x={MARGIN.left + innerW / 2}
            y={MARGIN.top + chartInnerH + 26}
            textAnchor="middle"
            fontSize={10}
            fill="var(--color-text)"
            opacity={0.6}
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            σ (perturbation)
          </text>

          {/* Line */}
          <path
            d={linePath}
            fill="none"
            stroke="coral"
            strokeWidth={2}
          />

          {/* Current position dot */}
          <circle
            cx={Math.round(chartScales.xScale(sigma) * 100) / 100}
            cy={Math.round(chartScales.yScale(bottleneckDistance) * 100) / 100}
            r={5}
            fill="coral"
            stroke="var(--color-surface)"
            strokeWidth={2}
          />
        </svg>
      </div>
    </div>
  );
}
