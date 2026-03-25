import { useState, useMemo, useId } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import {
  type Vec2,
  type Vec3,
  stereoNorth,
  stereoSouth,
  invStereoNorth,
  invStereoSouth,
  transitionNS,
  spherePoint,
  geoToSpherical,
  vec2Norm,
} from './shared/manifoldGeometry';

// ── Constants ────────────────────────────────────────────────────────

const HEIGHT = 360;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 24, right: 16, bottom: 32, left: 16 };
const CHART_CLAMP = 8; // clamp stereographic coords to avoid blowup

const TEAL = dimensionColors[0];  // chart 1 (north pole)
const PURPLE = dimensionColors[1]; // chart 2 (south pole)

type AtlasType = 'stereographic' | 'cylindrical';

interface ChartInfo {
  name: string;
  color: string;
  coords: Vec2;
  valid: boolean;
}

// ── Helpers ──────────────────────────────────────────────────────────

function clampVec2(v: Vec2, max: number): Vec2 {
  return {
    x: Math.max(-max, Math.min(max, v.x)),
    y: Math.max(-max, Math.min(max, v.y)),
  };
}

/** Cylindrical chart (equator band): (x,y,z) -> (atan2(y,x), z) */
function cylChart1(p: Vec3): Vec2 {
  return { x: Math.atan2(p.y, p.x), y: p.z };
}

/** Cylindrical chart 2 (rotated 180deg): (x,y,z) -> (atan2(-y,-x), z) */
function cylChart2(p: Vec3): Vec2 {
  return { x: Math.atan2(-p.y, -p.x), y: p.z };
}

// ── Component ────────────────────────────────────────────────────────

export default function ChartAtlasExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const instanceId = useId().replace(/:/g, '');

  // State
  const [atlas, setAtlas] = useState<AtlasType>('stereographic');
  const [showOverlap, setShowOverlap] = useState(true);
  const [showTransition, setShowTransition] = useState(false);
  // Point on the sphere stored as geographic coords [longitude, latitude] in degrees
  const [pointGeo, setPointGeo] = useState<[number, number]>([30, 20]);

  const isStacked = containerWidth < SM_BREAKPOINT;
  const leftWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.48);
  const rightWidth = isStacked ? containerWidth : containerWidth - leftWidth - 12;

  // Compute sphere point and chart coords
  const sphereData = useMemo(() => {
    const lonRad = (pointGeo[0] * Math.PI) / 180;
    const latRad = (pointGeo[1] * Math.PI) / 180;
    const { theta, phi } = geoToSpherical(lonRad, latRad);
    const p: Vec3 = spherePoint(theta, phi);

    let chart1: ChartInfo;
    let chart2: ChartInfo;

    if (atlas === 'stereographic') {
      const c1 = clampVec2(stereoNorth(p), CHART_CLAMP);
      const c2 = clampVec2(stereoSouth(p), CHART_CLAMP);
      chart1 = { name: 'φ_N (North)', color: TEAL, coords: c1, valid: p.z < 0.95 };
      chart2 = { name: 'φ_S (South)', color: PURPLE, coords: c2, valid: p.z > -0.95 };
    } else {
      const c1 = cylChart1(p);
      const c2 = cylChart2(p);
      // Cylindrical charts valid when not near the "seam"
      chart1 = { name: 'ψ₁ (Front)', color: TEAL, coords: c1, valid: Math.abs(c1.x) < 2.8 };
      chart2 = { name: 'ψ₂ (Back)', color: PURPLE, coords: c2, valid: Math.abs(c2.x) < 2.8 };
    }

    const inOverlap = chart1.valid && chart2.valid;

    // Transition map value
    let transitionCoords: Vec2 | null = null;
    if (inOverlap && atlas === 'stereographic') {
      transitionCoords = transitionNS(chart1.coords);
    }

    return { p, chart1, chart2, inOverlap, transitionCoords };
  }, [pointGeo, atlas]);

  // ── Left panel: sphere with chart domains ──────────────────────────

  const leftRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (leftWidth < 80) return;

      const w = leftWidth;
      const h = HEIGHT;
      svg.attr('width', w).attr('height', h);

      const cx = w / 2;
      const cy = h / 2;
      const radius = Math.min(w, h) / 2 - 30;

      // Orthographic projection
      const projection = d3.geoOrthographic()
        .translate([cx, cy])
        .scale(radius)
        .rotate([-20, -20, 0])
        .clipAngle(90);

      const path = d3.geoPath(projection);

      // Globe outline
      svg.append('circle')
        .attr('cx', cx)
        .attr('cy', cy)
        .attr('r', radius)
        .style('fill', 'var(--color-surface)')
        .style('stroke', 'var(--color-muted-border)')
        .style('stroke-width', 1.5);

      // Graticule
      const graticule = d3.geoGraticule().step([30, 30]);
      svg.append('path')
        .datum(graticule())
        .attr('d', path as any)
        .style('fill', 'none')
        .style('stroke', 'var(--color-muted-border)')
        .style('stroke-width', 0.5)
        .style('opacity', 0.5);

      // Chart domain regions
      if (showOverlap) {
        if (atlas === 'stereographic') {
          // North chart covers everything except a cap around south pole
          const northCap: GeoJSON.Feature = {
            type: 'Feature',
            properties: {},
            geometry: {
              type: 'Polygon',
              coordinates: [
                Array.from({ length: 61 }, (_, i) => {
                  const lon = (i / 60) * 360 - 180;
                  return [lon, -70]; // south cap boundary at -70deg latitude
                }).concat([[-180, -70]]),
              ],
            },
          };
          // South chart covers everything except a cap around north pole
          const southCap: GeoJSON.Feature = {
            type: 'Feature',
            properties: {},
            geometry: {
              type: 'Polygon',
              coordinates: [
                Array.from({ length: 61 }, (_, i) => {
                  const lon = (i / 60) * 360 - 180;
                  return [lon, 70]; // north cap boundary at 70deg latitude
                }).concat([[-180, 70]]),
              ],
            },
          };

          // Draw overlap band (between -70 and 70 latitude)
          const overlapBand: GeoJSON.Feature = {
            type: 'Feature',
            properties: {},
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  ...Array.from({ length: 61 }, (_, i) => {
                    const lon = -180 + (i / 60) * 360;
                    return [lon, 70] as [number, number];
                  }),
                  ...Array.from({ length: 61 }, (_, i) => {
                    const lon = 180 - (i / 60) * 360;
                    return [lon, -70] as [number, number];
                  }),
                  [-180, 70],
                ],
              ],
            },
          };

          svg.append('path')
            .datum(overlapBand)
            .attr('d', path as any)
            .style('fill', TEAL)
            .style('opacity', 0.08);

          svg.append('path')
            .datum(overlapBand)
            .attr('d', path as any)
            .style('fill', PURPLE)
            .style('opacity', 0.08);

          // Draw excluded caps
          svg.append('path')
            .datum(northCap)
            .attr('d', path as any)
            .style('fill', PURPLE)
            .style('opacity', 0.12);

          svg.append('path')
            .datum(southCap)
            .attr('d', path as any)
            .style('fill', TEAL)
            .style('opacity', 0.12);
        } else {
          // Cylindrical: front/back hemispheres
          const frontHemi: GeoJSON.Feature = {
            type: 'Feature',
            properties: {},
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  ...Array.from({ length: 61 }, (_, i) => {
                    const lat = -90 + (i / 60) * 180;
                    return [-90, lat] as [number, number];
                  }),
                  ...Array.from({ length: 61 }, (_, i) => {
                    const lat = 90 - (i / 60) * 180;
                    return [90, lat] as [number, number];
                  }),
                  [-90, -90],
                ],
              ],
            },
          };

          svg.append('path')
            .datum(frontHemi)
            .attr('d', path as any)
            .style('fill', TEAL)
            .style('opacity', 0.12);
        }
      }

      // Point on sphere
      const projected = projection([pointGeo[0], pointGeo[1]]);
      if (projected) {
        // Active chart highlight
        const activeColor = sphereData.chart1.valid ? TEAL : PURPLE;

        svg.append('circle')
          .attr('cx', projected[0])
          .attr('cy', projected[1])
          .attr('r', 14)
          .style('fill', activeColor)
          .style('opacity', 0.15);

        svg.append('circle')
          .attr('cx', projected[0])
          .attr('cy', projected[1])
          .attr('r', 6)
          .style('fill', activeColor)
          .style('stroke', 'var(--color-surface)')
          .style('stroke-width', 2)
          .style('cursor', 'grab');

        // Drag behavior
        const drag = d3.drag<SVGCircleElement, unknown>()
          .on('drag', function (event) {
            const inv = projection.invert?.([event.x, event.y]);
            if (inv) {
              setPointGeo([inv[0], inv[1]]);
            }
          });

        svg.select<SVGCircleElement>('circle:last-of-type').call(drag);
      }

      // Labels
      svg.append('text')
        .attr('x', cx)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '13px')
        .style('font-family', 'var(--font-sans)')
        .style('font-weight', '600')
        .text('S² with chart domains');

      // Pole markers
      const northProj = projection([0, 90]);
      const southProj = projection([0, -90]);
      if (northProj) {
        svg.append('text')
          .attr('x', northProj[0] + 10).attr('y', northProj[1])
          .style('fill', 'var(--color-text-muted)')
          .style('font-size', '11px')
          .style('font-family', 'var(--font-sans)')
          .text('N');
      }
      if (southProj) {
        svg.append('text')
          .attr('x', southProj[0] + 10).attr('y', southProj[1])
          .style('fill', 'var(--color-text-muted)')
          .style('font-size', '11px')
          .style('font-family', 'var(--font-sans)')
          .text('S');
      }
    },
    [leftWidth, pointGeo, atlas, showOverlap, sphereData],
  );

  // ── Right panel: chart coordinates in R² ────────────────────────────

  const rightRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (rightWidth < 80) return;

      const w = rightWidth;
      const h = HEIGHT;
      svg.attr('width', w).attr('height', h);

      const plotW = w - MARGIN.left - MARGIN.right;
      const plotH = h - MARGIN.top - MARGIN.bottom;

      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      // Determine domain based on atlas
      const chartDomain = atlas === 'stereographic' ? [-4, 4] : [-Math.PI, Math.PI];
      const xScale = d3.scaleLinear().domain(chartDomain).range([0, plotW]);
      const yScale = d3.scaleLinear()
        .domain(atlas === 'stereographic' ? [-4, 4] : [-1.2, 1.2])
        .range([plotH, 0]);

      // Background
      g.append('rect')
        .attr('width', plotW)
        .attr('height', plotH)
        .style('fill', 'var(--color-surface)')
        .style('stroke', 'var(--color-muted-border)')
        .style('stroke-width', 0.5);

      // Grid
      const xTicks = atlas === 'stereographic' ? [-3, -2, -1, 0, 1, 2, 3] : [-2, -1, 0, 1, 2];
      const yTicks = atlas === 'stereographic' ? [-3, -2, -1, 0, 1, 2, 3] : [-1, 0, 1];
      xTicks.forEach((t) => {
        g.append('line')
          .attr('x1', xScale(t)).attr('x2', xScale(t))
          .attr('y1', 0).attr('y2', plotH)
          .style('stroke', 'var(--color-muted-border)').style('stroke-width', 0.3);
      });
      yTicks.forEach((t) => {
        g.append('line')
          .attr('x1', 0).attr('x2', plotW)
          .attr('y1', yScale(t)).attr('y2', yScale(t))
          .style('stroke', 'var(--color-muted-border)').style('stroke-width', 0.3);
      });

      // Axes
      g.append('line')
        .attr('x1', xScale(0)).attr('x2', xScale(0))
        .attr('y1', 0).attr('y2', plotH)
        .style('stroke', 'var(--color-text-muted)').style('stroke-width', 1);
      g.append('line')
        .attr('x1', 0).attr('x2', plotW)
        .attr('y1', yScale(0)).attr('y2', yScale(0))
        .style('stroke', 'var(--color-text-muted)').style('stroke-width', 1);

      // Chart 1 point
      const { chart1, chart2, inOverlap, transitionCoords } = sphereData;
      if (chart1.valid) {
        const px = xScale(chart1.coords.x);
        const py = yScale(chart1.coords.y);
        if (px >= 0 && px <= plotW && py >= 0 && py <= plotH) {
          g.append('circle')
            .attr('cx', px).attr('cy', py).attr('r', 6)
            .style('fill', chart1.color).style('stroke', 'var(--color-surface)').style('stroke-width', 2);
          g.append('text')
            .attr('x', px + 10).attr('y', py - 8)
            .style('fill', chart1.color).style('font-size', '11px').style('font-family', 'var(--font-sans)')
            .text(`(${chart1.coords.x.toFixed(2)}, ${chart1.coords.y.toFixed(2)})`);
        }
      }

      // Chart 2 point (only if in overlap)
      if (showOverlap && chart2.valid && inOverlap) {
        const px = xScale(chart2.coords.x);
        const py = yScale(chart2.coords.y);
        if (px >= 0 && px <= plotW && py >= 0 && py <= plotH) {
          g.append('circle')
            .attr('cx', px).attr('cy', py).attr('r', 6)
            .style('fill', chart2.color).style('stroke', 'var(--color-surface)').style('stroke-width', 2);
          g.append('text')
            .attr('x', px + 10).attr('y', py + 16)
            .style('fill', chart2.color).style('font-size', '11px').style('font-family', 'var(--font-sans)')
            .text(`(${chart2.coords.x.toFixed(2)}, ${chart2.coords.y.toFixed(2)})`);
        }
      }

      // Transition map arrow
      if (showTransition && inOverlap && chart1.valid && chart2.valid) {
        const x1 = xScale(chart1.coords.x);
        const y1 = yScale(chart1.coords.y);
        const x2 = xScale(chart2.coords.x);
        const y2 = yScale(chart2.coords.y);

        if (x1 >= 0 && x1 <= plotW && x2 >= 0 && x2 <= plotW) {
          // Arrow marker
          const markerId = `arr-trans-${instanceId}`;
          g.append('defs')
            .append('marker')
            .attr('id', markerId)
            .attr('viewBox', '0 0 10 6')
            .attr('refX', 8).attr('refY', 3)
            .attr('markerWidth', 8).attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,0 L10,3 L0,6 Z')
            .style('fill', '#D97706');

          // Curved arrow
          const mx = (x1 + x2) / 2;
          const my = Math.min(y1, y2) - 30;
          g.append('path')
            .attr('d', `M${x1},${y1} Q${mx},${my} ${x2},${y2}`)
            .style('fill', 'none')
            .style('stroke', '#D97706')
            .style('stroke-width', 1.5)
            .style('stroke-dasharray', '4,3')
            .attr('marker-end', `url(#${markerId})`);

          // Label
          g.append('text')
            .attr('x', mx).attr('y', my - 6)
            .attr('text-anchor', 'middle')
            .style('fill', '#D97706').style('font-size', '10px').style('font-family', 'var(--font-sans)')
            .text(atlas === 'stereographic' ? 'φ_S ∘ φ_N⁻¹: inversion' : 'transition');
        }
      }

      // Title
      g.append('text')
        .attr('x', plotW / 2).attr('y', -8)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '13px')
        .style('font-family', 'var(--font-sans)')
        .style('font-weight', '600')
        .text('Chart coordinates (ℝ²)');

      // Axis labels
      g.append('text')
        .attr('x', plotW).attr('y', yScale(0) - 6)
        .attr('text-anchor', 'end')
        .style('fill', 'var(--color-text-muted)').style('font-size', '11px').style('font-family', 'var(--font-sans)')
        .text('u');
      g.append('text')
        .attr('x', xScale(0) + 8).attr('y', 10)
        .style('fill', 'var(--color-text-muted)').style('font-size', '11px').style('font-family', 'var(--font-sans)')
        .text('v');
    },
    [rightWidth, sphereData, showOverlap, showTransition, atlas, instanceId],
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
          <span className="font-medium">Atlas:</span>
          <select
            value={atlas}
            onChange={(e) => setAtlas(e.target.value as AtlasType)}
            className="px-2 py-1 rounded"
            style={{
              background: 'var(--color-surface)',
              border: '1px solid var(--color-muted-border)',
              color: 'var(--color-text)',
            }}
          >
            <option value="stereographic">Stereographic (N/S poles)</option>
            <option value="cylindrical">Cylindrical (front/back)</option>
          </select>
        </label>

        <label className="flex items-center gap-2 cursor-pointer" style={{ color: 'var(--color-text)' }}>
          <input
            type="checkbox"
            checked={showOverlap}
            onChange={(e) => setShowOverlap(e.target.checked)}
            className="accent-[#0F6E56]"
          />
          <span>Show overlap</span>
        </label>

        <label className="flex items-center gap-2 cursor-pointer" style={{ color: 'var(--color-text)' }}>
          <input
            type="checkbox"
            checked={showTransition}
            onChange={(e) => setShowTransition(e.target.checked)}
            className="accent-[#D97706]"
          />
          <span>Show transition map</span>
        </label>
      </div>

      {/* Info readout */}
      <div
        className="mt-3 px-4 py-3 rounded-lg text-sm grid grid-cols-2 gap-x-6 gap-y-1"
        style={{
          background: 'var(--color-muted-bg)',
          color: 'var(--color-text)',
          fontFamily: 'var(--font-mono)',
        }}
      >
        <div>
          <span style={{ color: TEAL, fontWeight: 600 }}>{sphereData.chart1.name}:</span>{' '}
          {sphereData.chart1.valid
            ? `(${sphereData.chart1.coords.x.toFixed(3)}, ${sphereData.chart1.coords.y.toFixed(3)})`
            : '— (pole excluded)'}
        </div>
        <div>
          <span style={{ color: PURPLE, fontWeight: 600 }}>{sphereData.chart2.name}:</span>{' '}
          {sphereData.chart2.valid
            ? `(${sphereData.chart2.coords.x.toFixed(3)}, ${sphereData.chart2.coords.y.toFixed(3)})`
            : '— (pole excluded)'}
        </div>
        {sphereData.inOverlap && (
          <div className="col-span-2" style={{ color: '#D97706' }}>
            Overlap region — both charts valid
            {showTransition && atlas === 'stereographic' && sphereData.transitionCoords && (
              <span>
                {' '}· φ_S ∘ φ_N⁻¹ = ({sphereData.transitionCoords.x.toFixed(3)},{' '}
                {sphereData.transitionCoords.y.toFixed(3)})
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
