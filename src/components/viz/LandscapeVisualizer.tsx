import { useState, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  diagramPoints,
  landscapeT,
  landscapeLayers,
} from '../../data/statistical-tda-data';

const MARGIN = { top: 20, right: 20, bottom: 40, left: 44 };
const SVG_HEIGHT = 360;

const LAYER_COLORS = ['#1e3a5f', '#2a5f8f', '#3a80b8', '#5ea0d0', '#8ec2e6'];
const LAYER_LABELS = ['k = 1', 'k = 2', 'k = 3', 'k = 4', 'k = 5'];

export default function LandscapeVisualizer() {
  const { ref, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [mode, setMode] = useState<'diagram' | 'landscape'>('diagram');
  const [visibleLayers, setVisibleLayers] = useState([true, true, true, false, false]);

  const toggleMode = useCallback(() => {
    setMode((m) => (m === 'diagram' ? 'landscape' : 'diagram'));
  }, []);

  const toggleLayer = useCallback((idx: number) => {
    setVisibleLayers((prev) => {
      const next = [...prev];
      next[idx] = !next[idx];
      return next;
    });
  }, []);

  const panelWidth = useMemo(() => {
    if (!containerWidth) return 420;
    return Math.min(containerWidth, 560);
  }, [containerWidth]);

  const innerW = panelWidth - MARGIN.left - MARGIN.right;
  const innerH = SVG_HEIGHT - MARGIN.top - MARGIN.bottom;

  // ─── Diagram mode scales ───
  const axisMax = 1.0;
  const diagXScale = useMemo(
    () => d3.scaleLinear().domain([0, axisMax]).range([MARGIN.left, MARGIN.left + innerW]),
    [innerW],
  );
  const diagYScale = useMemo(
    () => d3.scaleLinear().domain([0, axisMax]).range([MARGIN.top + innerH, MARGIN.top]),
    [innerH],
  );

  // ─── Landscape mode scales ───
  const landscapeMaxY = useMemo(() => {
    let max = 0;
    for (let k = 0; k < landscapeLayers.length; k++) {
      if (visibleLayers[k]) {
        for (const v of landscapeLayers[k]) {
          if (v > max) max = v;
        }
      }
    }
    return Math.max(max * 1.1, 0.05);
  }, [visibleLayers]);

  const landXScale = useMemo(
    () => d3.scaleLinear().domain([0, 1.0]).range([MARGIN.left, MARGIN.left + innerW]),
    [innerW],
  );
  const landYScale = useMemo(
    () => d3.scaleLinear().domain([0, landscapeMaxY]).range([MARGIN.top + innerH, MARGIN.top]),
    [innerH, landscapeMaxY],
  );

  // Tick arrays
  const diagXTicks = diagXScale.ticks(5);
  const diagYTicks = diagYScale.ticks(5);
  const landXTicks = landXScale.ticks(5);
  const landYTicks = landYScale.ticks(4);

  // Area paths for landscape layers (rendered bottom-up: highest k first so k=1 is on top)
  const areaPaths = useMemo(() => {
    return landscapeLayers.map((layer, k) => {
      if (!visibleLayers[k]) return '';
      const points: string[] = [];
      // Top edge
      for (let i = 0; i < landscapeT.length; i++) {
        const x = Math.round(landXScale(landscapeT[i]) * 100) / 100;
        const y = Math.round(landYScale(layer[i]) * 100) / 100;
        points.push(`${x},${y}`);
      }
      // Bottom edge (baseline at y=0)
      const baseline = Math.round(landYScale(0) * 100) / 100;
      const lastX = Math.round(landXScale(landscapeT[landscapeT.length - 1]) * 100) / 100;
      const firstX = Math.round(landXScale(landscapeT[0]) * 100) / 100;
      points.push(`${lastX},${baseline}`);
      points.push(`${firstX},${baseline}`);

      return points.join(' ');
    });
  }, [visibleLayers, landXScale, landYScale]);

  return (
    <div ref={ref} className="w-full space-y-3">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={toggleMode}
          className="rounded-md px-3 py-1.5 text-xs font-medium transition-colors"
          style={{
            fontFamily: 'var(--font-sans)',
            background: 'var(--color-accent)',
            color: '#fff',
            border: '1px solid var(--color-accent)',
          }}
        >
          {mode === 'diagram' ? 'Persistence Diagram' : 'Persistence Landscape'}
        </button>

        {mode === 'landscape' && (
          <div className="flex flex-wrap gap-2">
            {LAYER_LABELS.map((label, i) => (
              <label
                key={i}
                className="flex items-center gap-1 text-xs cursor-pointer"
                style={{ fontFamily: 'var(--font-sans)' }}
              >
                <input
                  type="checkbox"
                  checked={visibleLayers[i]}
                  onChange={() => toggleLayer(i)}
                  className="accent-[var(--color-accent)]"
                />
                <span
                  className="inline-block w-2.5 h-2.5 rounded-sm"
                  style={{ background: LAYER_COLORS[i] }}
                />
                {label}
              </label>
            ))}
          </div>
        )}
      </div>

      {/* SVG container with transition */}
      <div className="relative" style={{ height: SVG_HEIGHT }}>
        {/* Diagram mode */}
        <svg
          width={panelWidth}
          height={SVG_HEIGHT}
          className="rounded-lg border border-[var(--color-border)] absolute top-0 left-0 transition-opacity duration-300"
          style={{ opacity: mode === 'diagram' ? 1 : 0, pointerEvents: mode === 'diagram' ? 'auto' : 'none' }}
        >
          {/* Grid */}
          {diagXTicks.map((t) => (
            <line
              key={`dgx-${t}`}
              x1={Math.round(diagXScale(t) * 100) / 100}
              y1={MARGIN.top}
              x2={Math.round(diagXScale(t) * 100) / 100}
              y2={MARGIN.top + innerH}
              stroke="var(--color-border)"
              strokeOpacity={0.3}
            />
          ))}
          {diagYTicks.map((t) => (
            <line
              key={`dgy-${t}`}
              x1={MARGIN.left}
              y1={Math.round(diagYScale(t) * 100) / 100}
              x2={MARGIN.left + innerW}
              y2={Math.round(diagYScale(t) * 100) / 100}
              stroke="var(--color-border)"
              strokeOpacity={0.3}
            />
          ))}

          {/* Tick labels */}
          {diagXTicks.map((t) => (
            <text
              key={`dxt-${t}`}
              x={Math.round(diagXScale(t) * 100) / 100}
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
          {diagYTicks.map((t) => (
            <text
              key={`dyt-${t}`}
              x={MARGIN.left - 6}
              y={Math.round(diagYScale(t) * 100) / 100 + 3}
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

          {/* Diagonal */}
          <line
            x1={Math.round(diagXScale(0) * 100) / 100}
            y1={Math.round(diagYScale(0) * 100) / 100}
            x2={Math.round(diagXScale(axisMax) * 100) / 100}
            y2={Math.round(diagYScale(axisMax) * 100) / 100}
            stroke="var(--color-text)"
            strokeOpacity={0.3}
            strokeWidth={1}
            strokeDasharray="4,3"
          />

          {/* Diagram points */}
          {diagramPoints.map((p, i) => (
            <circle
              key={i}
              cx={Math.round(diagXScale(p.birth) * 100) / 100}
              cy={Math.round(diagYScale(p.death) * 100) / 100}
              r={p.death - p.birth > 0.3 ? 6 : 3.5}
              fill="#534AB7"
              fillOpacity={p.death - p.birth > 0.3 ? 0.85 : 0.5}
              stroke="var(--color-surface)"
              strokeWidth={1}
            />
          ))}
        </svg>

        {/* Landscape mode */}
        <svg
          width={panelWidth}
          height={SVG_HEIGHT}
          className="rounded-lg border border-[var(--color-border)] absolute top-0 left-0 transition-opacity duration-300"
          style={{ opacity: mode === 'landscape' ? 1 : 0, pointerEvents: mode === 'landscape' ? 'auto' : 'none' }}
        >
          {/* Grid */}
          {landXTicks.map((t) => (
            <line
              key={`lgx-${t}`}
              x1={Math.round(landXScale(t) * 100) / 100}
              y1={MARGIN.top}
              x2={Math.round(landXScale(t) * 100) / 100}
              y2={MARGIN.top + innerH}
              stroke="var(--color-border)"
              strokeOpacity={0.3}
            />
          ))}
          {landYTicks.map((t) => (
            <line
              key={`lgy-${t}`}
              x1={MARGIN.left}
              y1={Math.round(landYScale(t) * 100) / 100}
              x2={MARGIN.left + innerW}
              y2={Math.round(landYScale(t) * 100) / 100}
              stroke="var(--color-border)"
              strokeOpacity={0.3}
            />
          ))}

          {/* Tick labels */}
          {landXTicks.map((t) => (
            <text
              key={`lxt-${t}`}
              x={Math.round(landXScale(t) * 100) / 100}
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
          {landYTicks.map((t) => (
            <text
              key={`lyt-${t}`}
              x={MARGIN.left - 6}
              y={Math.round(landYScale(t) * 100) / 100 + 3}
              textAnchor="end"
              fontSize={9}
              fill="var(--color-text)"
              opacity={0.5}
              style={{ fontFamily: 'var(--font-sans)' }}
            >
              {t.toFixed(2)}
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
            t
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
            λ(t)
          </text>

          {/* Area fills — render in reverse order so k=1 paints on top */}
          {[...Array(5).keys()].reverse().map((k) =>
            visibleLayers[k] && areaPaths[k] ? (
              <polygon
                key={`area-${k}`}
                points={areaPaths[k]}
                fill={LAYER_COLORS[k]}
                fillOpacity={0.35}
                stroke={LAYER_COLORS[k]}
                strokeWidth={1.5}
                strokeOpacity={0.8}
              />
            ) : null,
          )}
        </svg>
      </div>
    </div>
  );
}
