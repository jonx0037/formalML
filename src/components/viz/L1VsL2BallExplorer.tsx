import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';

// =============================================================================
// L1VsL2BallExplorer — interactive companion to §2.2 Figure 2.1.
// 2D pedagogical picture of the lasso vs ridge constraint geometry:
//   - OLS centred at β̂_OLS = (0.4, 1.6) (red dot, fixed per brief)
//   - Loss contour: an ellipse with quadratic form Q(β) = (β - β̂_OLS)ᵀ H (β - β̂_OLS)
//     where H = [[1, 0.4], [0.4, 1]] (per brief)
//   - L1 diamond: ‖β‖₁ ≤ t
//   - L2 disk: ‖β‖₂ ≤ t
//   - L1 contact (sparse, generically at a vertex)
//   - L2 contact (dense, smooth point on disk)
// User slides budget t ∈ [0.3, 2.5]; both contact points + the just-tangent
// loss contour redraw live.
//
// Contact points computed by grid search (401 × 401 over [-3, 3]²; ~5 ms per call).
// The tangent loss-contour level is the loss value at the L1 contact (so the
// drawn ellipse just touches the diamond at the L1 vertex).
//
// Static fallback: public/images/topics/high-dimensional-regression/fig_02_01_l1_vs_l2_geometry.png
// =============================================================================

const HEIGHT = 460;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 16, right: 16, bottom: 50, left: 50 };

const OLS: [number, number] = [0.4, 1.6];
// Hessian H = XᵀX/n (positive definite, mild correlation).
const H: [[number, number], [number, number]] = [
  [1, 0.4],
  [0.4, 1],
];

const T_MIN = 0.3;
const T_MAX = 2.5;
const T_STEP = 0.05;
const T_DEFAULT = 1.0;

const DOMAIN_MIN = -1.5;
const DOMAIN_MAX = 2.5;
const GRID_RES = 301;

const TEAL = '#0F6E56';
const PURPLE = '#534AB7';
const AMBER = '#D97706';
const RED = '#B91C1C';
const SLATE = '#6B6B6B';

// -----------------------------------------------------------------------------
// Quadratic loss + grid-search optimal contact within a constraint.
// -----------------------------------------------------------------------------

function quadLoss(b: [number, number]): number {
  const dx = b[0] - OLS[0];
  const dy = b[1] - OLS[1];
  return dx * (H[0][0] * dx + H[0][1] * dy) + dy * (H[1][0] * dx + H[1][1] * dy);
}

interface Contact {
  beta: [number, number];
  loss: number;
}

function bestInConstraint(t: number, inside: (b: [number, number]) => boolean): Contact {
  let best: Contact = { beta: [0, 0], loss: Infinity };
  const step = (DOMAIN_MAX - DOMAIN_MIN) / (GRID_RES - 1);
  for (let i = 0; i < GRID_RES; i++) {
    const x = DOMAIN_MIN + i * step;
    for (let j = 0; j < GRID_RES; j++) {
      const y = DOMAIN_MIN + j * step;
      if (!inside([x, y])) continue;
      const l = quadLoss([x, y]);
      if (l < best.loss) {
        best = { beta: [x, y], loss: l };
      }
    }
  }
  return best;
}

function l1Contact(t: number): Contact {
  return bestInConstraint(t, (b) => Math.abs(b[0]) + Math.abs(b[1]) <= t + 1e-6);
}

function l2Contact(t: number): Contact {
  return bestInConstraint(t, (b) => b[0] * b[0] + b[1] * b[1] <= t * t + 1e-6);
}

// Generate ellipse points at loss level c: points satisfying quadLoss(β) = c.
// Use parametric form via H = R diag(λ₁, λ₂) Rᵀ. For H = [[1,a],[a,1]],
// eigenvalues are 1 ± a (for a < 1) with eigenvectors (1, ±1)/√2.
function ellipsePoints(level: number, npts = 100): [number, number][] {
  const a = H[0][1];
  const lam1 = 1 + a;
  const lam2 = 1 - a;
  const r1 = Math.sqrt(level / lam1);
  const r2 = Math.sqrt(level / lam2);
  const points: [number, number][] = [];
  for (let i = 0; i < npts; i++) {
    const theta = (2 * Math.PI * i) / (npts - 1);
    // Eigenvector basis: e1 = (1,1)/√2, e2 = (1,-1)/√2.
    const u1 = r1 * Math.cos(theta);
    const u2 = r2 * Math.sin(theta);
    const x = OLS[0] + (u1 + u2) / Math.sqrt(2);
    const y = OLS[1] + (u1 - u2) / Math.sqrt(2);
    points.push([x, y]);
  }
  return points;
}

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export default function L1VsL2BallExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const w = containerWidth || 720;
  const isMobile = w < SM_BREAKPOINT;
  const [tDisplay, setTDisplay] = useState(T_DEFAULT);
  const [tCommitted, setTCommitted] = useState(T_DEFAULT);

  // Contacts driven by committed slider value (avoids 5 ms grid search per
  // mouse-move tick — heavy enough to feel laggy on mobile if not deferred).
  const { l1, l2 } = useMemo(() => {
    return { l1: l1Contact(tCommitted), l2: l2Contact(tCommitted) };
  }, [tCommitted]);

  // Tangent loss level: the just-touching ellipse at the L1 contact.
  const lossLevel = l1.loss;
  const tangentEllipse = useMemo(() => ellipsePoints(lossLevel, 120), [lossLevel]);

  const renderRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (w <= 0) return;
      const innerW = w - MARGIN.left - MARGIN.right;
      const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;
      // Use the smaller dimension to keep the picture square (constraints are
      // symmetric in 2D, distortion would mislead).
      const side = Math.min(innerW, innerH);
      const xOff = (innerW - side) / 2;
      const yOff = (innerH - side) / 2;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left + xOff},${MARGIN.top + yOff})`);

      const xScale = d3.scaleLinear().domain([DOMAIN_MIN, DOMAIN_MAX]).range([0, side]);
      const yScale = d3.scaleLinear().domain([DOMAIN_MIN, DOMAIN_MAX]).range([side, 0]);

      // Background grid (subtle).
      const gridLines = d3.range(Math.floor(DOMAIN_MIN), Math.ceil(DOMAIN_MAX) + 1);
      gridLines.forEach((v) => {
        g.append('line').attr('x1', xScale(v)).attr('x2', xScale(v)).attr('y1', 0).attr('y2', side).style('stroke', 'var(--color-border)').style('stroke-width', 0.5).style('opacity', 0.5);
        g.append('line').attr('y1', yScale(v)).attr('y2', yScale(v)).attr('x1', 0).attr('x2', side).style('stroke', 'var(--color-border)').style('stroke-width', 0.5).style('opacity', 0.5);
      });

      // Axes through origin.
      g.append('line').attr('x1', xScale(DOMAIN_MIN)).attr('x2', xScale(DOMAIN_MAX)).attr('y1', yScale(0)).attr('y2', yScale(0)).style('stroke', 'var(--color-text-secondary)').style('stroke-width', 1);
      g.append('line').attr('x1', xScale(0)).attr('x2', xScale(0)).attr('y1', yScale(DOMAIN_MIN)).attr('y2', yScale(DOMAIN_MAX)).style('stroke', 'var(--color-text-secondary)').style('stroke-width', 1);

      // Axis labels.
      g.append('text').attr('x', xScale(DOMAIN_MAX) - 4).attr('y', yScale(0) - 6).attr('text-anchor', 'end').style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('β₁');
      g.append('text').attr('x', xScale(0) + 6).attr('y', yScale(DOMAIN_MAX) + 12).style('fill', 'var(--color-text)').style('font-family', 'var(--font-sans)').style('font-size', '11px').text('β₂');

      // L1 diamond: vertices (±t, 0), (0, ±t).
      const diamondPoints: [number, number][] = [
        [tCommitted, 0],
        [0, tCommitted],
        [-tCommitted, 0],
        [0, -tCommitted],
        [tCommitted, 0],
      ];
      const diamondPath = d3
        .line<[number, number]>()
        .x((p) => xScale(p[0]))
        .y((p) => yScale(p[1]));
      g.append('path').attr('d', diamondPath(diamondPoints)).style('fill', TEAL).style('fill-opacity', 0.07).style('stroke', TEAL).style('stroke-width', 2);

      // L2 disk: circle of radius t centered at origin.
      g.append('circle').attr('cx', xScale(0)).attr('cy', yScale(0)).attr('r', xScale(tCommitted) - xScale(0)).style('fill', PURPLE).style('fill-opacity', 0.07).style('stroke', PURPLE).style('stroke-width', 2);

      // Tangent ellipse (loss contour at L1-contact level).
      const ellipsePath = d3
        .line<[number, number]>()
        .x((p) => xScale(p[0]))
        .y((p) => yScale(p[1]));
      g.append('path').attr('d', ellipsePath(tangentEllipse)).style('fill', 'none').style('stroke', AMBER).style('stroke-width', 1.5).style('stroke-dasharray', '5 3');

      // OLS marker.
      g.append('circle').attr('cx', xScale(OLS[0])).attr('cy', yScale(OLS[1])).attr('r', 5).style('fill', RED).style('stroke', 'var(--color-bg)').style('stroke-width', 1.5);
      g.append('text').attr('x', xScale(OLS[0]) + 8).attr('y', yScale(OLS[1]) - 6).style('fill', RED).style('font-family', 'var(--font-mono)').style('font-size', '11px').text(`β̂_OLS (${OLS[0]}, ${OLS[1]})`);

      // L1 contact marker.
      g.append('circle').attr('cx', xScale(l1.beta[0])).attr('cy', yScale(l1.beta[1])).attr('r', 5).style('fill', TEAL).style('stroke', 'var(--color-bg)').style('stroke-width', 1.5);
      g.append('text').attr('x', xScale(l1.beta[0]) - 6).attr('y', yScale(l1.beta[1]) + 16).attr('text-anchor', 'end').style('fill', TEAL).style('font-family', 'var(--font-mono)').style('font-size', '11px').text(`L1 (${l1.beta[0].toFixed(2)}, ${l1.beta[1].toFixed(2)})`);

      // L2 contact marker.
      g.append('circle').attr('cx', xScale(l2.beta[0])).attr('cy', yScale(l2.beta[1])).attr('r', 5).style('fill', PURPLE).style('stroke', 'var(--color-bg)').style('stroke-width', 1.5);
      g.append('text').attr('x', xScale(l2.beta[0]) + 8).attr('y', yScale(l2.beta[1]) + 4).style('fill', PURPLE).style('font-family', 'var(--font-mono)').style('font-size', '11px').text(`L2 (${l2.beta[0].toFixed(2)}, ${l2.beta[1].toFixed(2)})`);

      // Sparsity annotation: highlight when L1 lands on an axis.
      const l1OnAxis = Math.abs(l1.beta[0]) < 0.05 || Math.abs(l1.beta[1]) < 0.05;
      if (l1OnAxis) {
        g.append('text').attr('x', side / 2).attr('y', side - 6).attr('text-anchor', 'middle').style('fill', TEAL).style('font-family', 'var(--font-sans)').style('font-size', '12px').style('font-weight', 600).text('L1 contact at vertex → one coordinate exactly zero (sparse)');
      }
    },
    [tCommitted, l1, l2, tangentEllipse, w],
  );

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <div style={{ marginBottom: 12, display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <label style={{ fontFamily: 'var(--font-sans)', fontSize: '14px', color: 'var(--color-text)' }}>
          budget t = <strong>{tDisplay.toFixed(2)}</strong>
        </label>
        <input
          type="range"
          min={T_MIN}
          max={T_MAX}
          step={T_STEP}
          value={tDisplay}
          aria-label="budget t"
          onChange={(e) => setTDisplay(parseFloat(e.target.value))}
          onMouseUp={(e) => setTCommitted(parseFloat((e.target as HTMLInputElement).value))}
          onTouchEnd={(e) => setTCommitted(parseFloat((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => setTCommitted(parseFloat((e.target as HTMLInputElement).value))}
          style={{ flex: 1, minWidth: 200 }}
        />
        <span style={{ fontFamily: 'var(--font-sans)', fontSize: '12px', color: 'var(--color-text-secondary)' }}>
          (release to recompute contacts)
        </span>
      </div>
      <svg ref={renderRef} width={w} height={HEIGHT} role="img" aria-label="L1 diamond versus L2 disk geometric picture" />
      <p
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '13px',
          color: 'var(--color-text-secondary)',
          marginTop: '8px',
        }}
      >
        β̂_OLS = ({OLS[0]}, {OLS[1]}); Hessian H = [[1, {H[0][1]}], [{H[1][0]}, 1]]; loss Q(β) = (β − β̂_OLS)ᵀ H (β − β̂_OLS). The amber dashed ellipse is the just-tangent loss contour at the L1 contact. As you drag t, watch the L1 contact stay pinned to the diamond's nearest vertex (one coordinate exactly zero) while the L2 contact slides smoothly around the disk. The L1 vertex generically achieves a lower loss than any edge interior — that's the geometric source of sparsity.
      </p>
    </div>
  );
}
