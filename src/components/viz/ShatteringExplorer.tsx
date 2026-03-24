import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

// ─── Types ───

interface PlacedPoint {
  x: number; // Normalized 0–1
  y: number;
  id: number;
}

type HypothesisClass = 'thresholds' | 'intervals' | 'linear' | 'rectangles';

const HYPOTHESIS_INFO: Record<HypothesisClass, { label: string; vcDim: number }> = {
  thresholds: { label: 'Thresholds (d = 1)', vcDim: 1 },
  intervals: { label: 'Intervals (d = 2)', vcDim: 2 },
  linear: { label: 'Linear classifiers (d = 3)', vcDim: 3 },
  rectangles: { label: 'Rectangles (d = 4)', vcDim: 4 },
};

const MAX_POINTS = 6;

const COLORS = {
  positive: '#0F6E56',
  negative: '#DC2626',
} as const;

// ─── Realizability checks ───

/** Thresholds on x-axis: positive if x <= threshold */
function canRealizeThreshold(points: PlacedPoint[], labeling: boolean[]): boolean {
  if (points.length === 0) return true;
  // Sort by x, check if labeling is consistent with some threshold
  const indexed = points.map((p, i) => ({ x: p.x, label: labeling[i] })).sort((a, b) => a.x - b.x);
  // All positive then all negative, or all negative then all positive
  // Threshold at t: label = (x <= t) or label = (x > t)
  // Convention: positive = true = 1
  // Try "positive if x <= t": find if there's a cut where all left are positive, all right are negative
  for (let cut = -1; cut < indexed.length; cut++) {
    let valid = true;
    for (let i = 0; i < indexed.length; i++) {
      const expected = i <= cut;
      if (indexed[i].label !== expected) { valid = false; break; }
    }
    if (valid) return true;
  }
  // Try "positive if x > t"
  for (let cut = -1; cut < indexed.length; cut++) {
    let valid = true;
    for (let i = 0; i < indexed.length; i++) {
      const expected = i > cut;
      if (indexed[i].label !== expected) { valid = false; break; }
    }
    if (valid) return true;
  }
  return false;
}

/** Intervals on x-axis: positive if a <= x <= b */
function canRealizeInterval(points: PlacedPoint[], labeling: boolean[]): boolean {
  if (points.length === 0) return true;
  const indexed = points.map((p, i) => ({ x: p.x, label: labeling[i] })).sort((a, b) => a.x - b.x);
  // Check if positive labels form a contiguous block in sorted order
  // Also allow empty positive set and full positive set
  const posIndices = indexed.map((d, i) => d.label ? i : -1).filter(i => i >= 0);
  if (posIndices.length === 0) return true; // all negative — interval with a > b
  const minPos = posIndices[0];
  const maxPos = posIndices[posIndices.length - 1];
  // All points between minPos and maxPos must be positive
  for (let i = minPos; i <= maxPos; i++) {
    if (!indexed[i].label) return false;
  }
  return true;
}

/** Linear classifiers in 2D: positive if w·x + b >= 0 */
function canRealizeLinear(points: PlacedPoint[], labeling: boolean[]): boolean {
  if (points.length === 0) return true;
  if (points.length === 1) return true; // Any single point is separable

  const pos = points.filter((_, i) => labeling[i]);
  const neg = points.filter((_, i) => !labeling[i]);

  if (pos.length === 0 || neg.length === 0) return true; // Trivially separable

  // For small point sets (≤6 points), brute-force check linear separability
  // Try all candidate separating lines defined by pairs and offset directions
  // A line through two points divides the plane; we check all orientations
  const allPts = points.map((p, i) => ({ ...p, label: labeling[i] }));

  // Generate candidate normals from pairs of points
  const candidates: Array<{ wx: number; wy: number }> = [];
  for (let i = 0; i < allPts.length; i++) {
    for (let j = i + 1; j < allPts.length; j++) {
      const dx = allPts[j].x - allPts[i].x;
      const dy = allPts[j].y - allPts[i].y;
      candidates.push({ wx: -dy, wy: dx }); // perpendicular to line through i,j
      candidates.push({ wx: dy, wy: -dx });
    }
    // Also try axis-aligned
    candidates.push({ wx: 1, wy: 0 });
    candidates.push({ wx: 0, wy: 1 });
    candidates.push({ wx: -1, wy: 0 });
    candidates.push({ wx: 0, wy: -1 });
    candidates.push({ wx: 1, wy: 1 });
    candidates.push({ wx: 1, wy: -1 });
  }

  // For each candidate normal, try all possible bias values
  for (const { wx, wy } of candidates) {
    // Project all points onto the normal direction
    const projections = allPts.map(p => wx * p.x + wy * p.y);

    // Try thresholds between consecutive projections
    const sortedProj = [...new Set(projections)].sort((a, b) => a - b);
    const thresholds = [-Infinity];
    for (let i = 0; i < sortedProj.length - 1; i++) {
      thresholds.push((sortedProj[i] + sortedProj[i + 1]) / 2);
    }
    thresholds.push(Infinity);

    for (const b of thresholds) {
      let valid = true;
      for (let k = 0; k < allPts.length; k++) {
        const predicted = projections[k] >= b;
        if (predicted !== allPts[k].label) { valid = false; break; }
      }
      if (valid) return true;
    }
  }
  return false;
}

/** Axis-aligned rectangles in 2D: positive if a1 <= x <= b1 and a2 <= y <= b2 */
function canRealizeRectangle(points: PlacedPoint[], labeling: boolean[]): boolean {
  if (points.length === 0) return true;
  const pos = points.filter((_, i) => labeling[i]);
  const neg = points.filter((_, i) => !labeling[i]);

  if (pos.length === 0) return true; // Empty rectangle
  // The tightest rectangle encloses all positive points
  const xMin = Math.min(...pos.map(p => p.x));
  const xMax = Math.max(...pos.map(p => p.x));
  const yMin = Math.min(...pos.map(p => p.y));
  const yMax = Math.max(...pos.map(p => p.y));

  // Check that no negative point falls inside
  for (const p of neg) {
    if (p.x >= xMin && p.x <= xMax && p.y >= yMin && p.y <= yMax) {
      return false;
    }
  }
  return true;
}

function canRealize(points: PlacedPoint[], labeling: boolean[], hClass: HypothesisClass): boolean {
  switch (hClass) {
    case 'thresholds': return canRealizeThreshold(points, labeling);
    case 'intervals': return canRealizeInterval(points, labeling);
    case 'linear': return canRealizeLinear(points, labeling);
    case 'rectangles': return canRealizeRectangle(points, labeling);
  }
}

// ─── Component ───

export default function ShatteringExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const nextId = useRef(0);

  const [points, setPoints] = useState<PlacedPoint[]>([]);
  const [hClass, setHClass] = useState<HypothesisClass>('thresholds');

  const isDesktop = (containerWidth || 0) > 640;
  const canvasSize = Math.max(
    100,
    isDesktop ? Math.min(Math.floor(containerWidth * 0.5), 360) : Math.min(containerWidth - 32, 360),
  );

  // Generate all 2^m labelings and check realizability
  const labelings = useMemo(() => {
    const m = points.length;
    if (m === 0) return [];
    const total = 1 << m;
    const results: Array<{ bits: boolean[]; realizable: boolean }> = [];
    for (let mask = 0; mask < total; mask++) {
      const bits = Array.from({ length: m }, (_, i) => ((mask >> i) & 1) === 1);
      results.push({ bits, realizable: canRealize(points, bits, hClass) });
    }
    return results;
  }, [points, hClass]);

  const realizedCount = labelings.filter(l => l.realizable).length;
  const totalLabelings = labelings.length;
  const isShattered = totalLabelings > 0 && realizedCount === totalLabelings;

  // Handle click on canvas to add/remove points
  const handleCanvasClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const x = (e.clientX - rect.left) / canvasSize;
    const y = (e.clientY - rect.top) / canvasSize;

    // Check if clicking near existing point (remove it)
    const threshold = 0.05;
    const nearIdx = points.findIndex(p =>
      Math.abs(p.x - x) < threshold && Math.abs(p.y - y) < threshold
    );
    if (nearIdx >= 0) {
      setPoints(prev => prev.filter((_, i) => i !== nearIdx));
      return;
    }

    // Add new point if under limit
    if (points.length >= MAX_POINTS) return;
    const id = nextId.current++;
    setPoints(prev => [...prev, { x, y, id }]);
  }, [points, canvasSize]);

  // D3 canvas rendering
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || canvasSize <= 0) return;
    const sel = d3.select(svg);
    sel.selectAll('*').remove();

    // Background grid
    const gridLines = [0.25, 0.5, 0.75];
    gridLines.forEach(v => {
      sel.append('line').attr('x1', v * canvasSize).attr('x2', v * canvasSize).attr('y1', 0).attr('y2', canvasSize)
        .style('stroke', 'var(--color-border)').style('opacity', 0.3);
      sel.append('line').attr('x1', 0).attr('x2', canvasSize).attr('y1', v * canvasSize).attr('y2', v * canvasSize)
        .style('stroke', 'var(--color-border)').style('opacity', 0.3);
    });

    // Points
    points.forEach(p => {
      sel.append('circle')
        .attr('cx', p.x * canvasSize).attr('cy', p.y * canvasSize)
        .attr('r', 8).style('fill', COLORS.positive).style('stroke', '#fff').style('stroke-width', 2)
        .style('cursor', 'pointer');
    });

    // Instructions
    if (points.length === 0) {
      sel.append('text').attr('x', canvasSize / 2).attr('y', canvasSize / 2)
        .attr('text-anchor', 'middle').style('font-size', '12px').style('font-family', 'var(--font-sans)')
        .style('fill', 'var(--color-text-secondary)').style('opacity', 0.7)
        .text('Click to place points (max 6)');
    } else if (points.length < MAX_POINTS) {
      sel.append('text').attr('x', canvasSize / 2).attr('y', canvasSize - 8)
        .attr('text-anchor', 'middle').style('font-size', '9px').style('font-family', 'var(--font-mono)')
        .style('fill', 'var(--color-text-secondary)').style('opacity', 0.5)
        .text(`${points.length}/${MAX_POINTS} points — click to add, click point to remove`);
    }
  }, [points, canvasSize]);

  return (
    <div
      ref={containerRef}
      style={{
        border: '1px solid var(--color-border)',
        borderRadius: '8px',
        padding: '16px',
        background: 'var(--color-surface)',
        marginTop: '1.5rem',
        marginBottom: '1.5rem',
      }}
    >
      <div style={{ fontFamily: 'var(--font-sans)', fontSize: '13px', fontWeight: 600, marginBottom: '12px', color: 'var(--color-text)' }}>
        Shattering Explorer
      </div>

      {/* Hypothesis class selector */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '12px' }}>
        {(Object.keys(HYPOTHESIS_INFO) as HypothesisClass[]).map(key => (
          <label key={key} style={{
            display: 'flex', alignItems: 'center', gap: '4px',
            fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', cursor: 'pointer',
            padding: '4px 8px', borderRadius: '4px',
            background: hClass === key ? 'var(--color-definition-bg)' : 'transparent',
            border: `1px solid ${hClass === key ? 'var(--color-definition-border)' : 'transparent'}`,
          }}>
            <input type="radio" name="hclass" checked={hClass === key} onChange={() => setHClass(key)}
              style={{ accentColor: 'var(--color-accent)' }} />
            {HYPOTHESIS_INFO[key].label}
          </label>
        ))}
        <button
          onClick={() => { setPoints([]); nextId.current = 0; }}
          style={{
            fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)',
            background: 'var(--color-muted-bg)', border: '1px solid var(--color-border)',
            borderRadius: '4px', padding: '4px 10px', cursor: 'pointer',
          }}
        >
          Reset
        </button>
      </div>

      <div style={{ display: 'flex', flexDirection: isDesktop ? 'row' : 'column', gap: '16px' }}>
        {/* Canvas */}
        <svg
          ref={svgRef}
          width={canvasSize}
          height={canvasSize}
          onClick={handleCanvasClick}
          style={{
            border: '1px solid var(--color-border)', borderRadius: '6px',
            background: 'var(--color-muted-bg)', cursor: 'crosshair', flexShrink: 0,
          }}
        />

        {/* Right panel: labeling grid + summary */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {/* Summary stats */}
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', marginBottom: '12px',
          }}>
            <div style={{ padding: '8px', borderRadius: '4px', background: 'var(--color-muted-bg)', textAlign: 'center' }}>
              <div style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>Total</div>
              <div style={{ fontSize: '18px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--color-text)' }}>
                {totalLabelings || '—'}
              </div>
              <div style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
                {points.length > 0 ? `2^${points.length}` : ''}
              </div>
            </div>
            <div style={{ padding: '8px', borderRadius: '4px', background: 'var(--color-muted-bg)', textAlign: 'center' }}>
              <div style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>Realized</div>
              <div style={{ fontSize: '18px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: COLORS.positive }}>
                {totalLabelings > 0 ? realizedCount : '—'}
              </div>
              <div style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>|H_C|</div>
            </div>
            <div style={{ padding: '8px', borderRadius: '4px', background: isShattered ? 'rgba(15,110,86,0.1)' : 'var(--color-muted-bg)', textAlign: 'center', border: isShattered ? `1px solid ${COLORS.positive}` : '1px solid transparent' }}>
              <div style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>Shattered?</div>
              <div style={{ fontSize: '18px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: isShattered ? COLORS.positive : COLORS.negative }}>
                {totalLabelings > 0 ? (isShattered ? '✓' : '✗') : '—'}
              </div>
              <div style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
                VCdim = {HYPOTHESIS_INFO[hClass].vcDim}
              </div>
            </div>
          </div>

          {/* Labeling grid */}
          {points.length > 0 && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: `repeat(auto-fill, minmax(60px, 1fr))`,
              gap: '4px',
              maxHeight: '300px',
              overflowY: 'auto',
            }}>
              {labelings.map((l, idx) => (
                <LabelingThumbnail
                  key={idx}
                  points={points}
                  labeling={l.bits}
                  realizable={l.realizable}
                  size={60}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Labeling Thumbnail ───

function LabelingThumbnail({
  points,
  labeling,
  realizable,
  size,
}: {
  points: PlacedPoint[];
  labeling: boolean[];
  realizable: boolean;
  size: number;
}) {
  return (
    <svg
      width={size}
      height={size}
      style={{
        border: `1px solid ${realizable ? COLORS.positive : 'var(--color-border)'}`,
        borderRadius: '4px',
        background: realizable ? 'rgba(15,110,86,0.05)' : 'var(--color-muted-bg)',
        opacity: realizable ? 1 : 0.4,
      }}
    >
      {points.map((p, i) => (
        <circle
          key={p.id}
          cx={p.x * (size - 12) + 6}
          cy={p.y * (size - 12) + 6}
          r={4}
          fill={labeling[i] ? COLORS.positive : COLORS.negative}
          stroke="#fff"
          strokeWidth={0.5}
        />
      ))}
      {/* Realizable/not indicator */}
      <text
        x={size - 8}
        y={10}
        textAnchor="end"
        style={{
          fontSize: '9px',
          fontFamily: 'var(--font-mono)',
          fill: realizable ? COLORS.positive : COLORS.negative,
          fontWeight: 700,
        }}
      >
        {realizable ? '✓' : '✗'}
      </text>
    </svg>
  );
}
