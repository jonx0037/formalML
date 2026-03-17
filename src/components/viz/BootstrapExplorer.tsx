import { useState, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  diagramPoints,
  confidenceThresholds,
} from '../../data/statistical-tda-data';

const MARGIN = { top: 20, right: 20, bottom: 40, left: 44 };
const SVG_HEIGHT = 360;

export default function BootstrapExplorer() {
  const { ref, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [alpha, setAlpha] = useState(0.05);

  const handleAlphaChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setAlpha(parseFloat(e.target.value));
  }, []);

  const panelWidth = useMemo(() => {
    if (!containerWidth) return 420;
    return Math.min(containerWidth, 560);
  }, [containerWidth]);

  const innerW = panelWidth - MARGIN.left - MARGIN.right;
  const innerH = SVG_HEIGHT - MARGIN.top - MARGIN.bottom;

  const cAlpha = useMemo(() => {
    const key = alpha.toFixed(2);
    return confidenceThresholds[key] ?? 0.06;
  }, [alpha]);

  const bandWidth = 2 * cAlpha;

  // Classify points
  const classifiedPoints = useMemo(() => {
    return diagramPoints.map((p) => {
      const persistence = p.death - p.birth;
      const significant = persistence > bandWidth;
      return { ...p, persistence, significant };
    });
  }, [bandWidth]);

  const significantCount = classifiedPoints.filter((p) => p.significant).length;
  const noiseCount = classifiedPoints.filter((p) => !p.significant).length;

  // Scales
  const axisMax = 1.0;
  const xScale = useMemo(
    () => d3.scaleLinear().domain([0, axisMax]).range([MARGIN.left, MARGIN.left + innerW]),
    [innerW],
  );
  const yScale = useMemo(
    () => d3.scaleLinear().domain([0, axisMax]).range([MARGIN.top + innerH, MARGIN.top]),
    [innerH],
  );

  const xTicks = xScale.ticks(5);
  const yTicks = yScale.ticks(5);

  // Confidence band polygon: a strip of width bandWidth above the diagonal
  // The diagonal goes from (0,0) to (axisMax, axisMax).
  // The band is the region where death - birth <= bandWidth, i.e., death <= birth + bandWidth.
  // We draw it as a polygon: bottom edge = diagonal, top edge = diagonal shifted up.
  const bandPath = useMemo(() => {
    const steps = 50;
    const pts: string[] = [];
    // Bottom edge (the diagonal, left to right)
    for (let i = 0; i <= steps; i++) {
      const b = (i / steps) * axisMax;
      pts.push(`${Math.round(xScale(b) * 100) / 100},${Math.round(yScale(b) * 100) / 100}`);
    }
    // Top edge (diagonal + bandWidth, right to left)
    for (let i = steps; i >= 0; i--) {
      const b = (i / steps) * axisMax;
      const d = Math.min(b + bandWidth, axisMax);
      pts.push(`${Math.round(xScale(b) * 100) / 100},${Math.round(yScale(d) * 100) / 100}`);
    }
    return pts.join(' ');
  }, [xScale, yScale, bandWidth]);

  return (
    <div ref={ref} className="w-full space-y-3">
      {/* Slider */}
      <div className="flex items-center gap-3">
        <label
          className="text-xs font-medium whitespace-nowrap min-w-[140px]"
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          Significance level α = {alpha.toFixed(2)}
        </label>
        <input
          type="range"
          min={0.01}
          max={0.20}
          step={0.01}
          value={alpha}
          onChange={handleAlphaChange}
          className="w-full accent-[var(--color-accent)]"
        />
      </div>

      {/* Persistence diagram */}
      <svg
        width={panelWidth}
        height={SVG_HEIGHT}
        className="rounded-lg border border-[var(--color-border)]"
      >
        {/* Grid lines */}
        {xTicks.map((t) => (
          <line
            key={`gx-${t}`}
            x1={Math.round(xScale(t) * 100) / 100}
            y1={MARGIN.top}
            x2={Math.round(xScale(t) * 100) / 100}
            y2={MARGIN.top + innerH}
            stroke="var(--color-border)"
            strokeOpacity={0.3}
          />
        ))}
        {yTicks.map((t) => (
          <line
            key={`gy-${t}`}
            x1={MARGIN.left}
            y1={Math.round(yScale(t) * 100) / 100}
            x2={MARGIN.left + innerW}
            y2={Math.round(yScale(t) * 100) / 100}
            stroke="var(--color-border)"
            strokeOpacity={0.3}
          />
        ))}

        {/* Axis tick labels */}
        {xTicks.map((t) => (
          <text
            key={`xt-${t}`}
            x={Math.round(xScale(t) * 100) / 100}
            y={MARGIN.top + innerH + 16}
            textAnchor="middle"
            fontSize={9}
            fill="var(--color-text)"
            opacity={0.5}
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            {t}
          </text>
        ))}
        {yTicks.map((t) => (
          <text
            key={`yt-${t}`}
            x={MARGIN.left - 6}
            y={Math.round(yScale(t) * 100) / 100 + 3}
            textAnchor="end"
            fontSize={9}
            fill="var(--color-text)"
            opacity={0.5}
            style={{ fontFamily: 'var(--font-sans)' }}
          >
            {t}
          </text>
        ))}

        {/* Axis labels */}
        <text
          x={MARGIN.left + innerW / 2}
          y={MARGIN.top + innerH + 32}
          textAnchor="middle"
          fontSize={11}
          fill="var(--color-text)"
          opacity={0.7}
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          Birth
        </text>
        <text
          x={14}
          y={MARGIN.top + innerH / 2}
          textAnchor="middle"
          fontSize={11}
          fill="var(--color-text)"
          opacity={0.7}
          transform={`rotate(-90, 14, ${MARGIN.top + innerH / 2})`}
          style={{ fontFamily: 'var(--font-sans)' }}
        >
          Death
        </text>

        {/* Diagonal line (birth = death) */}
        <line
          x1={Math.round(xScale(0) * 100) / 100}
          y1={Math.round(yScale(0) * 100) / 100}
          x2={Math.round(xScale(axisMax) * 100) / 100}
          y2={Math.round(yScale(axisMax) * 100) / 100}
          stroke="var(--color-text)"
          strokeOpacity={0.3}
          strokeWidth={1}
          strokeDasharray="4,3"
        />

        {/* Confidence band */}
        <polygon
          points={bandPath}
          fill="salmon"
          fillOpacity={0.25}
          stroke="salmon"
          strokeOpacity={0.4}
          strokeWidth={1}
        />

        {/* Diagram points */}
        {classifiedPoints.map((p, i) => (
          <circle
            key={i}
            cx={Math.round(xScale(p.birth) * 100) / 100}
            cy={Math.round(yScale(p.death) * 100) / 100}
            r={p.significant ? 6 : 3.5}
            fill={p.significant ? 'steelblue' : '#999'}
            fillOpacity={p.significant ? 0.85 : 0.6}
            stroke="var(--color-surface)"
            strokeWidth={p.significant ? 1.5 : 0.5}
          />
        ))}
      </svg>

      {/* Readout */}
      <p
        className="text-xs opacity-70"
        style={{ fontFamily: 'var(--font-sans)' }}
      >
        c<sub>α</sub> = {cAlpha.toFixed(3)} &nbsp;|&nbsp; Significant features: {significantCount} &nbsp;|&nbsp; Noise features: {noiseCount}
      </p>
    </div>
  );
}
