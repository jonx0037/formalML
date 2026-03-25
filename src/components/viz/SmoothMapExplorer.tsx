import { useState, useMemo, useId } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import { type Vec2 } from './shared/manifoldGeometry';

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 320;
const SM_BREAKPOINT = 640;
const TEAL = dimensionColors[0];
const PURPLE = dimensionColors[1];
const AMBER = '#D97706';

type MapType = 'stereo' | 'double-cover' | 'inclusion';

interface MapConfig {
  label: string;
  sourceLabel: string;
  targetLabel: string;
  /** Compute F(theta) -> target position */
  map: (theta: number) => Vec2;
  /** Compute |dF| scale factor at theta (tangent vector multiplier) */
  jacobianScale: (theta: number) => number;
  /** Jacobian matrix entries for display */
  jacobianMatrix: (theta: number) => number[][];
  sourceType: 'circle' | 'sphere-proj';
  targetType: 'circle' | 'plane';
}

const MAPS: Record<MapType, MapConfig> = {
  'stereo': {
    label: 'Stereographic: S¹ → ℝ',
    sourceLabel: 'S¹',
    targetLabel: 'ℝ',
    map: (theta) => {
      // Stereographic from north pole of S^1: (cos θ, sin θ) -> sin θ / (1 - cos θ)
      // Use: t = tan(θ/2) = sin θ/(1 + cos θ)
      const t = Math.tan(theta / 2);
      return { x: Math.max(-8, Math.min(8, t)), y: 0 };
    },
    jacobianScale: (theta) => {
      // |dt/dθ| = 1/(2 cos²(θ/2)) = (1 + tan²(θ/2))/2
      const c = Math.cos(theta / 2);
      return Math.min(8, 1 / (2 * c * c));
    },
    jacobianMatrix: (theta) => {
      const c = Math.cos(theta / 2);
      return [[Number((1 / (2 * c * c)).toFixed(3))]];
    },
    sourceType: 'circle',
    targetType: 'plane',
  },
  'double-cover': {
    label: 'Double cover: S¹ → S¹, θ ↦ 2θ',
    sourceLabel: 'S¹ (source)',
    targetLabel: 'S¹ (target)',
    map: (theta) => ({ x: Math.cos(2 * theta), y: Math.sin(2 * theta) }),
    jacobianScale: () => 2, // constant
    jacobianMatrix: () => [[2]],
    sourceType: 'circle',
    targetType: 'circle',
  },
  'inclusion': {
    label: 'Inclusion: S¹ ↪ ℝ²',
    sourceLabel: 'S¹',
    targetLabel: 'ℝ²',
    map: (theta) => ({ x: Math.cos(theta), y: Math.sin(theta) }),
    jacobianScale: () => 1,
    jacobianMatrix: (theta) => [
      [Number((-Math.sin(theta)).toFixed(3))],
      [Number(Math.cos(theta).toFixed(3))],
    ],
    sourceType: 'circle',
    targetType: 'plane',
  },
};

// ── Component ────────────────────────────────────────────────────────

export default function SmoothMapExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const instanceId = useId().replace(/:/g, '');

  const [mapType, setMapType] = useState<MapType>('double-cover');
  const [theta, setTheta] = useState(Math.PI / 4);
  const [magnitude, setMagnitude] = useState(1.0);
  const [showJacobian, setShowJacobian] = useState(false);

  const cfg = MAPS[mapType];
  const isStacked = containerWidth < SM_BREAKPOINT;
  const panelWidth = isStacked ? containerWidth : Math.floor((containerWidth - 12) / 2);

  const mapData = useMemo(() => {
    const sourcePos: Vec2 = { x: Math.cos(theta), y: Math.sin(theta) };
    const targetPos = cfg.map(theta);
    const jScale = cfg.jacobianScale(theta);
    const jacobian = cfg.jacobianMatrix(theta);

    // Source tangent vector (perpendicular to radius on S^1)
    const sourceTangent: Vec2 = { x: -Math.sin(theta), y: Math.cos(theta) };

    // Target tangent vector (pushforward)
    let targetTangent: Vec2;
    if (cfg.targetType === 'circle') {
      // On target S^1, tangent is perpendicular to radius at F(theta)
      targetTangent = { x: -targetPos.y * jScale, y: targetPos.x * jScale };
    } else if (mapType === 'stereo') {
      // On ℝ, tangent is horizontal
      targetTangent = { x: jScale, y: 0 };
    } else {
      // Inclusion: tangent in ℝ² is the derivative of (cos θ, sin θ)
      targetTangent = { x: -Math.sin(theta), y: Math.cos(theta) };
    }

    return { sourcePos, targetPos, sourceTangent, targetTangent, jScale, jacobian };
  }, [theta, cfg, mapType]);

  // ── Left panel: source manifold ────────────────────────────────────

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth < 80) return;

      const w = panelWidth;
      const h = HEIGHT;
      svg.attr('width', w).attr('height', h);

      const cx = w / 2;
      const cy = h / 2;
      const r = Math.min(w, h) / 2 - 50;

      // Draw S^1
      svg.append('circle')
        .attr('cx', cx).attr('cy', cy).attr('r', r)
        .style('fill', 'none')
        .style('stroke', 'var(--color-text-muted)')
        .style('stroke-width', 2);

      const { sourcePos, sourceTangent } = mapData;

      // Point p
      const px = cx + sourcePos.x * r;
      const py = cy - sourcePos.y * r;

      svg.append('circle')
        .attr('cx', px).attr('cy', py).attr('r', 6)
        .style('fill', TEAL)
        .style('stroke', 'var(--color-surface)')
        .style('stroke-width', 2)
        .style('cursor', 'grab');

      // Tangent vector at p
      const arrowLen = magnitude * r * 0.4;
      const tx = px + sourceTangent.x * arrowLen;
      const ty = py - sourceTangent.y * arrowLen;

      const markerId = `arr-src-${instanceId}`;
      svg.append('defs')
        .append('marker')
        .attr('id', markerId)
        .attr('viewBox', '0 0 10 6')
        .attr('refX', 8).attr('refY', 3)
        .attr('markerWidth', 7).attr('markerHeight', 5)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,0 L10,3 L0,6 Z')
        .style('fill', TEAL);

      svg.append('line')
        .attr('x1', px).attr('y1', py)
        .attr('x2', tx).attr('y2', ty)
        .style('stroke', TEAL).style('stroke-width', 2.5)
        .attr('marker-end', `url(#${markerId})`);

      // Labels
      svg.append('text')
        .attr('x', px + 10).attr('y', py - 10)
        .style('fill', TEAL).style('font-size', '12px').style('font-family', 'var(--font-sans)').style('font-weight', '600')
        .text('p');
      svg.append('text')
        .attr('x', tx + 6).attr('y', ty - 6)
        .style('fill', TEAL).style('font-size', '11px').style('font-family', 'var(--font-sans)')
        .text('v');

      // Title
      svg.append('text')
        .attr('x', cx).attr('y', 18)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '13px')
        .style('font-family', 'var(--font-sans)')
        .style('font-weight', '600')
        .text(cfg.sourceLabel);

      // Theta annotation
      svg.append('text')
        .attr('x', cx).attr('y', h - 8)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text-muted)')
        .style('font-size', '11px')
        .style('font-family', 'var(--font-mono)')
        .text(`θ = ${(theta * 180 / Math.PI).toFixed(0)}°`);

      // Drag
      const drag = d3.drag<SVGCircleElement, unknown>()
        .on('drag', function (event) {
          const dx = event.x - cx;
          const dy = -(event.y - cy);
          let angle = Math.atan2(dy, dx);
          if (angle < 0) angle += 2 * Math.PI;
          // Clamp away from poles for stereographic
          if (mapType === 'stereo') {
            angle = Math.max(0.15, Math.min(Math.PI * 2 - 0.15, angle));
          }
          setTheta(angle);
        });

      svg.select<SVGCircleElement>(`circle[cx="${px}"]`).call(drag);
    },
    [panelWidth, mapData, theta, magnitude, cfg, instanceId, mapType],
  );

  // ── Right panel: target manifold ───────────────────────────────────

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (panelWidth < 80) return;

      const w = panelWidth;
      const h = HEIGHT;
      svg.attr('width', w).attr('height', h);

      const cx = w / 2;
      const cy = h / 2;
      const r = Math.min(w, h) / 2 - 50;

      const { targetPos, targetTangent, jScale, jacobian } = mapData;

      if (cfg.targetType === 'circle') {
        // Draw target S^1
        svg.append('circle')
          .attr('cx', cx).attr('cy', cy).attr('r', r)
          .style('fill', 'none')
          .style('stroke', 'var(--color-text-muted)')
          .style('stroke-width', 2);

        const fpx = cx + targetPos.x * r;
        const fpy = cy - targetPos.y * r;

        svg.append('circle')
          .attr('cx', fpx).attr('cy', fpy).attr('r', 6)
          .style('fill', PURPLE)
          .style('stroke', 'var(--color-surface)')
          .style('stroke-width', 2);

        // Pushed-forward tangent vector
        const arrowLen = magnitude * r * 0.4;
        const normFactor = Math.sqrt(targetTangent.x * targetTangent.x + targetTangent.y * targetTangent.y);
        const nTx = normFactor > 1e-8 ? targetTangent.x / normFactor : 0;
        const nTy = normFactor > 1e-8 ? targetTangent.y / normFactor : 0;
        const scaledLen = arrowLen * jScale;
        const tx = fpx + nTx * Math.min(scaledLen, r * 1.5);
        const ty = fpy - nTy * Math.min(scaledLen, r * 1.5);

        const markerId = `arr-tgt-${instanceId}`;
        svg.append('defs')
          .append('marker')
          .attr('id', markerId)
          .attr('viewBox', '0 0 10 6')
          .attr('refX', 8).attr('refY', 3)
          .attr('markerWidth', 7).attr('markerHeight', 5)
          .attr('orient', 'auto')
          .append('path')
          .attr('d', 'M0,0 L10,3 L0,6 Z')
          .style('fill', PURPLE);

        svg.append('line')
          .attr('x1', fpx).attr('y1', fpy)
          .attr('x2', tx).attr('y2', ty)
          .style('stroke', PURPLE).style('stroke-width', 2.5)
          .attr('marker-end', `url(#${markerId})`);

        // Labels
        svg.append('text')
          .attr('x', fpx + 10).attr('y', fpy - 10)
          .style('fill', PURPLE).style('font-size', '12px').style('font-family', 'var(--font-sans)').style('font-weight', '600')
          .text('F(p)');
        svg.append('text')
          .attr('x', tx + 6).attr('y', ty - 6)
          .style('fill', PURPLE).style('font-size', '11px').style('font-family', 'var(--font-sans)')
          .text('dF(v)');

      } else {
        // Plane target (ℝ or ℝ²)
        // Draw axes
        svg.append('line')
          .attr('x1', 20).attr('x2', w - 20)
          .attr('y1', cy).attr('y2', cy)
          .style('stroke', 'var(--color-text-muted)').style('stroke-width', 1);

        if (mapType === 'inclusion') {
          svg.append('line')
            .attr('x1', cx).attr('x2', cx)
            .attr('y1', 30).attr('y2', h - 30)
            .style('stroke', 'var(--color-text-muted)').style('stroke-width', 1);

          // Unit circle outline for reference
          svg.append('circle')
            .attr('cx', cx).attr('cy', cy).attr('r', r * 0.7)
            .style('fill', 'none')
            .style('stroke', 'var(--color-muted-border)')
            .style('stroke-width', 0.5)
            .style('stroke-dasharray', '3,3');
        }

        const scale = mapType === 'stereo' ? r * 0.15 : r * 0.7;
        const fpx = cx + targetPos.x * scale;
        const fpy = cy - targetPos.y * scale;

        svg.append('circle')
          .attr('cx', fpx).attr('cy', fpy).attr('r', 6)
          .style('fill', PURPLE)
          .style('stroke', 'var(--color-surface)')
          .style('stroke-width', 2);

        // Pushed-forward tangent
        const arrowScale = magnitude * r * 0.3;
        const normFactor = Math.sqrt(targetTangent.x * targetTangent.x + targetTangent.y * targetTangent.y);
        const dispScale = Math.min(normFactor, 5) / (normFactor || 1);
        const tx = fpx + targetTangent.x * dispScale * arrowScale;
        const ty = fpy - targetTangent.y * dispScale * arrowScale;

        const markerId = `arr-tgt-${instanceId}`;
        svg.append('defs')
          .append('marker')
          .attr('id', markerId)
          .attr('viewBox', '0 0 10 6')
          .attr('refX', 8).attr('refY', 3)
          .attr('markerWidth', 7).attr('markerHeight', 5)
          .attr('orient', 'auto')
          .append('path')
          .attr('d', 'M0,0 L10,3 L0,6 Z')
          .style('fill', PURPLE);

        svg.append('line')
          .attr('x1', fpx).attr('y1', fpy)
          .attr('x2', tx).attr('y2', ty)
          .style('stroke', PURPLE).style('stroke-width', 2.5)
          .attr('marker-end', `url(#${markerId})`);

        svg.append('text')
          .attr('x', fpx + 10).attr('y', fpy - 10)
          .style('fill', PURPLE).style('font-size', '12px').style('font-family', 'var(--font-sans)').style('font-weight', '600')
          .text('F(p)');
        svg.append('text')
          .attr('x', tx + 6).attr('y', ty - 6)
          .style('fill', PURPLE).style('font-size', '11px').style('font-family', 'var(--font-sans)')
          .text('dF(v)');
      }

      // Title
      svg.append('text')
        .attr('x', cx).attr('y', 18)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '13px')
        .style('font-family', 'var(--font-sans)')
        .style('font-weight', '600')
        .text(cfg.targetLabel);

      // Jacobian display
      if (showJacobian) {
        const { jacobian } = mapData;
        const jx = 12;
        const jy = h - 45;

        svg.append('rect')
          .attr('x', jx).attr('y', jy - 2)
          .attr('width', 120).attr('height', jacobian.length * 18 + 20)
          .attr('rx', 4)
          .style('fill', 'var(--color-surface)')
          .style('stroke', 'var(--color-muted-border)')
          .style('stroke-width', 0.5);

        svg.append('text')
          .attr('x', jx + 6).attr('y', jy + 14)
          .style('fill', AMBER).style('font-size', '11px').style('font-family', 'var(--font-sans)').style('font-weight', '600')
          .text('Jacobian:');

        jacobian.forEach((row, i) => {
          svg.append('text')
            .attr('x', jx + 10).attr('y', jy + 30 + i * 16)
            .style('fill', 'var(--color-text)').style('font-size', '11px').style('font-family', 'var(--font-mono)')
            .text(`[ ${row.map((v) => v.toFixed(3)).join('  ')} ]`);
        });

        svg.append('text')
          .attr('x', jx + 6).attr('y', jy + 30 + jacobian.length * 16)
          .style('fill', 'var(--color-text-muted)').style('font-size', '10px').style('font-family', 'var(--font-mono)')
          .text(`|dF| = ${mapData.jScale.toFixed(3)}`);
      }
    },
    [panelWidth, mapData, cfg, showJacobian, instanceId, magnitude, mapType],
  );

  // ── Render ─────────────────────────────────────────────────────────

  return (
    <div ref={containerRef} className="w-full my-8">
      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-3`}>
        <svg ref={leftRef} />
        <svg ref={rightRef} />
      </div>

      {/* Controls */}
      <div className="mt-4 flex flex-wrap gap-4 items-center text-sm" style={{ fontFamily: 'var(--font-sans)' }}>
        <label className="flex items-center gap-2" style={{ color: 'var(--color-text)' }}>
          <span className="font-medium">Map:</span>
          <select
            value={mapType}
            onChange={(e) => setMapType(e.target.value as MapType)}
            className="px-2 py-1 rounded"
            style={{
              background: 'var(--color-surface)',
              border: '1px solid var(--color-muted-border)',
              color: 'var(--color-text)',
            }}
          >
            {Object.entries(MAPS).map(([key, val]) => (
              <option key={key} value={key}>{val.label}</option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-2" style={{ color: 'var(--color-text)' }}>
          <span className="font-medium">|v|:</span>
          <input
            type="range" min="0.2" max="2" step="0.1" value={magnitude}
            onChange={(e) => setMagnitude(Number(e.target.value))}
            className="w-20"
          />
          <span style={{ fontFamily: 'var(--font-mono)' }}>{magnitude.toFixed(1)}</span>
        </label>

        <label className="flex items-center gap-2 cursor-pointer" style={{ color: 'var(--color-text)' }}>
          <input
            type="checkbox"
            checked={showJacobian}
            onChange={(e) => setShowJacobian(e.target.checked)}
            className="accent-[#D97706]"
          />
          <span>Show Jacobian</span>
        </label>
      </div>
    </div>
  );
}
