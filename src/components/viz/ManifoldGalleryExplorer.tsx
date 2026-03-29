import { useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import {
  type Vec2,
  torusPoint,
  mobiusPoint,
  orthoProject,
} from './shared/manifoldGeometry';

// ── Manifold Data ────────────────────────────────────────────────────

interface ManifoldInfo {
  name: string;
  symbol: string;
  dimension: number;
  compact: boolean;
  orientable: boolean;
  fundamentalGroup: string;
  minCharts: number;
  connection: string;
  wireframe: (w: number, h: number) => string;
}

function circleSVG(w: number, h: number): string {
  const r = Math.min(w, h) / 2 - 8;
  return `M${w / 2 + r},${h / 2} A${r},${r} 0 1,1 ${w / 2 - r},${h / 2} A${r},${r} 0 1,1 ${w / 2 + r},${h / 2}`;
}

function sphereSVG(w: number, h: number): string {
  const cx = w / 2;
  const cy = h / 2;
  const r = Math.min(w, h) / 2 - 8;
  const paths: string[] = [];

  // Outline
  paths.push(`M${cx + r},${cy} A${r},${r} 0 1,1 ${cx - r},${cy} A${r},${r} 0 1,1 ${cx + r},${cy}`);

  // Latitude lines (3 ellipses)
  [-0.5, 0, 0.5].forEach((f) => {
    const y = cy - f * r;
    const rx = r * Math.sqrt(1 - f * f);
    const ry = rx * 0.3;
    paths.push(`M${cx - rx},${y} A${rx},${ry} 0 1,0 ${cx + rx},${y}`);
  });

  // Longitude line (vertical ellipse)
  paths.push(`M${cx},${cy - r} A${r * 0.3},${r} 0 1,0 ${cx},${cy + r}`);
  paths.push(`M${cx},${cy - r} A${r * 0.3},${r} 0 1,1 ${cx},${cy + r}`);

  return paths.join(' ');
}

function torusSVG(w: number, h: number): string {
  const cx = w / 2;
  const cy = h / 2;
  const sc = Math.min(w, h) * 0.35;
  const pts: Vec2[][] = [];

  // Generate torus wireframe lines
  const nU = 12;
  const nV = 8;

  // Constant-u lines (around the tube)
  for (let i = 0; i < nU; i++) {
    const u = (i / nU) * 2 * Math.PI;
    const line: Vec2[] = [];
    for (let j = 0; j <= nV * 4; j++) {
      const v = (j / (nV * 4)) * 2 * Math.PI;
      const p = torusPoint(u, v, 2, 0.8);
      const proj = orthoProject(p, -0.3, -0.5);
      line.push({ x: cx + proj.x * sc, y: cy - proj.y * sc });
    }
    pts.push(line);
  }

  // Constant-v lines (around the ring)
  for (let j = 0; j < nV; j++) {
    const v = (j / nV) * 2 * Math.PI;
    const line: Vec2[] = [];
    for (let i = 0; i <= nU * 4; i++) {
      const u = (i / (nU * 4)) * 2 * Math.PI;
      const p = torusPoint(u, v, 2, 0.8);
      const proj = orthoProject(p, -0.3, -0.5);
      line.push({ x: cx + proj.x * sc, y: cy - proj.y * sc });
    }
    pts.push(line);
  }

  return pts.map((line) => {
    const d = line.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
    return d;
  }).join(' ');
}

function crossCapSVG(w: number, h: number): string {
  // Simplified RP^2 cross-cap visualization
  const cx = w / 2;
  const cy = h / 2;
  const r = Math.min(w, h) / 2 - 10;

  // Draw a figure-8 like shape representing the self-intersection
  const paths: string[] = [];
  // Outer loop
  paths.push(`M${cx + r},${cy} A${r},${r} 0 1,1 ${cx - r},${cy} A${r},${r} 0 1,1 ${cx + r},${cy}`);
  // Cross-cap twist (figure 8 inside)
  const r2 = r * 0.6;
  paths.push(`M${cx},${cy - r * 0.8} Q${cx + r2},${cy} ${cx},${cy + r * 0.8} Q${cx - r2},${cy} ${cx},${cy - r * 0.8}`);

  return paths.join(' ');
}

function glMatrixSVG(w: number, h: number): string {
  // Abstract matrix grid icon
  const cx = w / 2;
  const cy = h / 2;
  const size = Math.min(w, h) * 0.6;
  const half = size / 2;
  const paths: string[] = [];

  // 3x3 grid
  for (let i = 0; i <= 3; i++) {
    const offset = -half + (i / 3) * size;
    paths.push(`M${cx - half},${cy + offset} L${cx + half},${cy + offset}`);
    paths.push(`M${cx + offset},${cy - half} L${cx + offset},${cy + half}`);
  }

  return paths.join(' ');
}

function mobiusSVG(w: number, h: number): string {
  const cx = w / 2;
  const cy = h / 2;
  const sc = Math.min(w, h) * 0.3;
  const pts: Vec2[][] = [];

  // Generate Mobius strip wireframe
  const nU = 20;
  for (const vVal of [-0.8, 0, 0.8]) {
    const line: Vec2[] = [];
    for (let i = 0; i <= nU * 3; i++) {
      const u = (i / (nU * 3)) * 2 * Math.PI;
      const p = mobiusPoint(u, vVal, 1.5);
      const proj = orthoProject(p, -0.2, -0.5);
      line.push({ x: cx + proj.x * sc, y: cy - proj.y * sc });
    }
    pts.push(line);
  }

  // Cross-sections
  for (let i = 0; i < 10; i++) {
    const u = (i / 10) * 2 * Math.PI;
    const line: Vec2[] = [];
    for (let j = 0; j <= 8; j++) {
      const v = -1 + (j / 8) * 2;
      const p = mobiusPoint(u, v, 1.5);
      const proj = orthoProject(p, -0.2, -0.5);
      line.push({ x: cx + proj.x * sc, y: cy - proj.y * sc });
    }
    pts.push(line);
  }

  return pts.map((line) => {
    return line.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
  }).join(' ');
}

const MANIFOLDS: ManifoldInfo[] = [
  {
    name: 'Circle',
    symbol: 'S¹',
    dimension: 1,
    compact: true,
    orientable: true,
    fundamentalGroup: 'ℤ',
    minCharts: 2,
    connection: 'The simplest closed manifold. Two arc charts suffice. The stereographic atlas in §3 generalizes this.',
    wireframe: circleSVG,
  },
  {
    name: 'Sphere',
    symbol: 'S²',
    dimension: 2,
    compact: true,
    orientable: true,
    fundamentalGroup: '{e} (trivial)',
    minCharts: 2,
    connection: 'The main running example: stereographic charts from §3, tangent spaces from §7.',
    wireframe: sphereSVG,
  },
  {
    name: 'Torus',
    symbol: 'T²',
    dimension: 2,
    compact: true,
    orientable: true,
    fundamentalGroup: 'ℤ × ℤ',
    minCharts: 4,
    connection: 'Product manifold S¹ × S¹. Charts are products of arc charts. Genus 1.',
    wireframe: torusSVG,
  },
  {
    name: 'Real Projective Plane',
    symbol: 'ℝP²',
    dimension: 2,
    compact: true,
    orientable: false,
    fundamentalGroup: 'ℤ/2ℤ',
    minCharts: 3,
    connection: 'Quotient S² / {x ~ -x}. Non-orientable — cannot embed in ℝ³ without self-intersection.',
    wireframe: crossCapSVG,
  },
  {
    name: 'General Linear Group',
    symbol: 'GL(n,ℝ)',
    dimension: -1, // n²
    compact: false,
    orientable: true,
    fundamentalGroup: 'ℤ for n = 2, ℤ/2ℤ for n ≥ 3',
    minCharts: 1,
    connection: 'Open subset of ℝⁿ² (det ≠ 0). A Lie group — smooth manifold with smooth group operations.',
    wireframe: glMatrixSVG,
  },
  {
    name: 'Mobius Band',
    symbol: 'Möbius',
    dimension: 2,
    compact: true, // compact manifold with boundary
    orientable: false,
    fundamentalGroup: 'ℤ',
    minCharts: 2,
    connection: 'Compact non-orientable surface with boundary. Shows why orientability matters for integration.',
    wireframe: mobiusSVG,
  },
];

// ── Component ────────────────────────────────────────────────────────

export default function ManifoldGalleryExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [expanded, setExpanded] = useState<number | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);

  const rawCardSize =
    containerWidth < 400
      ? Math.floor(containerWidth / 2) - 8
      : Math.min(180, Math.floor(containerWidth / 3) - 12);
  const cardSize = Math.max(32, rawCardSize);

  // Don't render grid until we have a valid container width
  if (containerWidth === 0 && expanded === null) {
    return <div ref={containerRef} className="w-full my-8" />;
  }

  if (expanded !== null) {
    const m = MANIFOLDS[expanded];
    return (
      <div ref={containerRef} className="w-full my-8">
        <div
          className="rounded-lg p-5"
          style={{
            background: 'var(--color-muted-bg)',
            border: '1px solid var(--color-muted-border)',
          }}
        >
          {/* Header */}
          <div className="flex justify-between items-start mb-4">
            <div>
              <h4
                className="text-lg font-semibold m-0"
                style={{ color: 'var(--color-text)', fontFamily: 'var(--font-sans)' }}
              >
                {m.symbol} — {m.name}
              </h4>
            </div>
            <button
              onClick={() => setExpanded(null)}
              className="px-3 py-1 rounded text-sm cursor-pointer"
              style={{
                background: 'var(--color-surface)',
                border: '1px solid var(--color-muted-border)',
                color: 'var(--color-text)',
                fontFamily: 'var(--font-sans)',
              }}
            >
              ✕ Close
            </button>
          </div>

          <div className={`flex ${containerWidth < 500 ? 'flex-col' : 'flex-row'} gap-5`}>
            {/* Wireframe */}
            <svg role="img" aria-label="Manifold gallery explorer visualization (panel 1 of 2)" width={160} height={140}>
              <path
                d={m.wireframe(160, 140)}
                fill="none"
                stroke="var(--color-text-muted)"
                strokeWidth={1.2}
              />
            </svg>

            {/* Properties */}
            <div className="flex-1 text-sm" style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-text)' }}>
              <table className="w-full" style={{ borderCollapse: 'collapse' }}>
                <tbody>
                  {[
                    ['Dimension', m.dimension === -1 ? 'n²' : String(m.dimension)],
                    ['Compact', m.compact ? 'Yes' : 'No'],
                    ['Orientable', m.orientable ? 'Yes' : 'No'],
                    ['π₁', m.fundamentalGroup],
                    ['Min. charts', String(m.minCharts)],
                  ].map(([label, value]) => (
                    <tr key={label} style={{ borderBottom: '1px solid var(--color-muted-border)' }}>
                      <td className="py-1.5 pr-4 font-medium" style={{ color: 'var(--color-text-muted)' }}>{label}</td>
                      <td className="py-1.5" style={{ fontFamily: 'var(--font-mono)' }}>{value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <p className="mt-3 text-sm leading-relaxed" style={{ color: 'var(--color-text-muted)' }}>
                {m.connection}
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full my-8">
      <div
        className="grid gap-3"
        style={{
          gridTemplateColumns: `repeat(${containerWidth < 400 ? 2 : 3}, 1fr)`,
        }}
      >
        {MANIFOLDS.map((m, i) => (
          <button
            key={m.symbol}
            onClick={() => setExpanded(i)}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
            className="rounded-lg p-3 cursor-pointer transition-all"
            style={{
              background: 'var(--color-surface)',
              border: '1px solid',
              borderColor: hovered === i ? dimensionColors[0] : 'var(--color-muted-border)',
              textAlign: 'center',
            }}
          >
            <svg role="img" aria-label="Manifold gallery explorer visualization (panel 2 of 2)" width={cardSize - 24} height={Math.min(cardSize - 24, 100)} style={{ margin: '0 auto', display: 'block' }}>
              <path
                d={m.wireframe(cardSize - 24, Math.min(cardSize - 24, 100))}
                fill="none"
                stroke="var(--color-text-muted)"
                strokeWidth={1}
              />
            </svg>
            <div
              className="mt-2 text-sm font-medium"
              style={{ color: 'var(--color-text)', fontFamily: 'var(--font-sans)' }}
            >
              {m.symbol}
            </div>
            <div
              className="text-xs mt-0.5"
              style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-sans)' }}
            >
              {m.name}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
