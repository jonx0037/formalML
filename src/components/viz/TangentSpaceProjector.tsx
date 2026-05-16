import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { paletteSemi, saddleSurface, type Vec3 } from './shared/semiparametric-inference';

// =============================================================================
// TangentSpaceProjector — §2.4 finite-dimensional visual anchor.
// A 2D saddle surface in 3D, the tangent plane at the origin, a nuisance
// direction (red) and the orthogonal complement (green) within that plane,
// and an ambient vector decomposed across them.
// =============================================================================

const HEIGHT = 460;
const SM_BREAKPOINT = 640;

// Project a 3D point to 2D screen coordinates via an isometric-ish camera.
function project3D(p: Vec3, az: number, el: number, scale: number, cx: number, cy: number): [number, number] {
  // Rotate around z-axis by az, then tilt by el.
  const ca = Math.cos(az), sa = Math.sin(az);
  const ce = Math.cos(el), se = Math.sin(el);
  const x1 = ca * p[0] - sa * p[1];
  const y1 = sa * p[0] + ca * p[1];
  const z1 = p[2];
  const x2 = x1;
  const y2 = ce * y1 - se * z1;
  return [cx + scale * x2, cy - scale * y2];
}

export default function TangentSpaceProjector() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [alphaDeg, setAlphaDeg] = useState(30); // nuisance-direction angle
  const [vxDeg, setVxDeg] = useState(60); // ambient vector direction within tangent plane

  const isMobile = containerWidth > 0 && containerWidth < SM_BREAKPOINT;

  const data = useMemo(() => {
    const uGrid = d3.range(-1.2, 1.21, 0.12);
    const vGrid = d3.range(-1.2, 1.21, 0.12);
    const surface = saddleSurface(uGrid, vGrid, 0.4);
    const alpha = (alphaDeg * Math.PI) / 180;
    const vxRad = (vxDeg * Math.PI) / 180;
    // Ambient vector lives in the tangent plane at the origin: cos(vx)·b1 + sin(vx)·b2
    // At origin b1 = (1,0,0), b2 = (0,1,0), so vector = (cos vx, sin vx, 0).
    const vector: Vec3 = [Math.cos(vxRad), Math.sin(vxRad), 0];
    const dec = surface.decomposeVector(0, 0, alpha, vector);
    return { uGrid, vGrid, surface, alpha, vector, dec };
  }, [alphaDeg, vxDeg]);

  const ref = useD3<SVGSVGElement>(
    (svg) => {
      const w = containerWidth || 720;
      const W = isMobile ? w : Math.min(w, 720);
      const H = HEIGHT;
      svg.selectAll('*').remove();
      svg.attr('viewBox', `0 0 ${W} ${H}`);
      const scale = isMobile ? 90 : 120;
      const cx = W / 2;
      const cy = H / 2 + 30;
      const azim = -Math.PI / 4;
      const elev = Math.PI / 6;
      const proj = (p: Vec3) => project3D(p, azim, elev, scale, cx, cy);

      const { uGrid, vGrid, surface, vector, dec } = data;

      // Draw the saddle surface as a wireframe mesh (every other line for clarity).
      const meshColor = 'var(--color-border)';
      for (let i = 0; i < uGrid.length; i += 2) {
        const linePts: string[] = [];
        for (let j = 0; j < vGrid.length; j++) {
          const p: Vec3 = [uGrid[i], vGrid[j], surface.z[i][j]];
          const [sx, sy] = proj(p);
          linePts.push(`${sx},${sy}`);
        }
        svg.append('polyline')
          .attr('points', linePts.join(' '))
          .style('fill', 'none')
          .style('stroke', meshColor)
          .style('stroke-width', 0.7)
          .style('opacity', 0.5);
      }
      for (let j = 0; j < vGrid.length; j += 2) {
        const linePts: string[] = [];
        for (let i = 0; i < uGrid.length; i++) {
          const p: Vec3 = [uGrid[i], vGrid[j], surface.z[i][j]];
          const [sx, sy] = proj(p);
          linePts.push(`${sx},${sy}`);
        }
        svg.append('polyline')
          .attr('points', linePts.join(' '))
          .style('fill', 'none')
          .style('stroke', meshColor)
          .style('stroke-width', 0.7)
          .style('opacity', 0.5);
      }

      // Draw the tangent plane at the origin: a translucent quad.
      const planeSize = 1.0;
      const planeCorners: Vec3[] = [
        [-planeSize, -planeSize, 0],
        [planeSize, -planeSize, 0],
        [planeSize, planeSize, 0],
        [-planeSize, planeSize, 0],
      ];
      svg.append('polygon')
        .attr('points', planeCorners.map((p) => proj(p).join(',')).join(' '))
        .style('fill', paletteSemi.tangent)
        .style('opacity', 0.13)
        .style('stroke', paletteSemi.tangent)
        .style('stroke-width', 0.8)
        .style('stroke-dasharray', '4,3');

      // Origin point.
      const [ox, oy] = proj([0, 0, 0]);
      svg.append('circle').attr('cx', ox).attr('cy', oy).attr('r', 3.5)
        .style('fill', 'var(--color-text)');

      // Helper to draw a vector arrow from origin.
      const drawArrow = (
        tip: Vec3,
        color: string,
        labelText: string,
        labelOffsetX = 0,
        labelOffsetY = 0,
        strokeWidth = 2,
        dasharray: string | null = null,
      ) => {
        const [tx, ty] = proj(tip);
        const arrowId = `arrow-${labelText.replace(/[^a-zA-Z0-9]/g, '')}`;
        svg.append('defs').append('marker')
          .attr('id', arrowId)
          .attr('viewBox', '0 -4 8 8')
          .attr('refX', 7)
          .attr('refY', 0)
          .attr('markerWidth', 7)
          .attr('markerHeight', 7)
          .attr('orient', 'auto')
          .append('path')
          .attr('d', 'M0,-4L8,0L0,4Z')
          .style('fill', color);
        const lineSel = svg.append('line')
          .attr('x1', ox).attr('y1', oy)
          .attr('x2', tx).attr('y2', ty)
          .style('stroke', color)
          .style('stroke-width', strokeWidth)
          .attr('marker-end', `url(#${arrowId})`);
        if (dasharray) lineSel.style('stroke-dasharray', dasharray);
        svg.append('text')
          .attr('x', tx + labelOffsetX)
          .attr('y', ty + labelOffsetY)
          .style('fill', color)
          .style('font-family', 'var(--font-mono)')
          .style('font-size', '12px')
          .style('font-weight', '600')
          .text(labelText);
      };

      // Nuisance direction Λ_η (red).
      const dirN = dec.nuisanceDir;
      const tipN: Vec3 = [dirN[0] * 0.9, dirN[1] * 0.9, dirN[2] * 0.9];
      drawArrow(tipN, paletteSemi.ambient, 'Λη', 6, -4);

      // Orthogonal complement Λ_η⊥ (green).
      const dirP = dec.orthogonalDir;
      const tipP: Vec3 = [dirP[0] * 0.9, dirP[1] * 0.9, dirP[2] * 0.9];
      drawArrow(tipP, paletteSemi.orthogonal, 'Λη⊥', 8, -4);

      // Ambient vector v (blue).
      drawArrow(vector, paletteSemi.tangent, 'v', 8, -4, 2.5);

      // Decomposition projections (dashed parallelogram).
      const cN = dec.coordNuisance;
      const cP = dec.coordOrthogonal;
      const projN: Vec3 = [dirN[0] * cN, dirN[1] * cN, dirN[2] * cN];
      const projP: Vec3 = [dirP[0] * cP, dirP[1] * cP, dirP[2] * cP];
      // Line from tip of v parallel to b2 (orthogonal direction).
      svg.append('line')
        .attr('x1', proj(vector)[0]).attr('y1', proj(vector)[1])
        .attr('x2', proj(projN)[0]).attr('y2', proj(projN)[1])
        .style('stroke', paletteSemi.orthogonal)
        .style('stroke-width', 1.3)
        .style('stroke-dasharray', '3,3')
        .style('opacity', 0.7);
      svg.append('line')
        .attr('x1', proj(vector)[0]).attr('y1', proj(vector)[1])
        .attr('x2', proj(projP)[0]).attr('y2', proj(projP)[1])
        .style('stroke', paletteSemi.ambient)
        .style('stroke-width', 1.3)
        .style('stroke-dasharray', '3,3')
        .style('opacity', 0.7);

      // Coordinate readout.
      svg.append('text')
        .attr('x', 12).attr('y', 22)
        .style('fill', 'var(--color-text)')
        .style('font-family', 'var(--font-mono)')
        .style('font-size', '12px')
        .text(`v = ${cN.toFixed(3)} · Λη + ${cP.toFixed(3)} · Λη⊥`);
    },
    [containerWidth, isMobile, data],
  );

  return (
    <div ref={containerRef} className="viz-container" style={{ width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 12 }}>
        <label style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>
          Nuisance angle Λη (within tangent plane): <strong>{alphaDeg}°</strong>
          <input
            type="range"
            min={0}
            max={180}
            value={alphaDeg}
            onChange={(e) => setAlphaDeg(+e.target.value)}
            aria-label="Nuisance direction angle"
            style={{ width: '100%', marginTop: 4 }}
          />
        </label>
        <label style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>
          Ambient vector direction (within tangent plane): <strong>{vxDeg}°</strong>
          <input
            type="range"
            min={0}
            max={360}
            value={vxDeg}
            onChange={(e) => setVxDeg(+e.target.value)}
            aria-label="Ambient vector direction"
            style={{ width: '100%', marginTop: 4 }}
          />
        </label>
      </div>
      <svg ref={ref} style={{ width: '100%', height: HEIGHT, display: 'block' }} />
    </div>
  );
}
