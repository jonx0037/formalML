import { useState, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

const MARGIN = { top: 20, right: 20, bottom: 40, left: 44 };
const SVG_HEIGHT = 400;
const NUM_POINTS = 200;

// Simple LCG pseudo-random number generator (seeded)
function lcg(index: number, seed: number): number {
  let s = (seed * 2147483647 + index * 16807 + 12345) & 0x7fffffff;
  s = (s * 16807 + 12345) & 0x7fffffff;
  s = (s * 16807 + 12345) & 0x7fffffff;
  return (s & 0x7fffffff) / 0x7fffffff;
}

function seededNormal(i: number, seed: number): number {
  const u1 = Math.max(lcg(i * 2, seed), 1e-10);
  const u2 = lcg(i * 2 + 1, seed);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

interface EigenResult {
  lambda1: number;
  lambda2: number;
  v1: [number, number];
  v2: [number, number];
}

function eigenDecomposition2x2(
  a: number,
  b: number,
  c: number,
): EigenResult {
  // Matrix [[a, b], [b, c]]
  const trace = a + c;
  const diff = a - c;
  const disc = Math.sqrt(diff * diff + 4 * b * b);
  const lambda1 = (trace + disc) / 2;
  const lambda2 = (trace - disc) / 2;

  let v1: [number, number];
  let v2: [number, number];

  if (Math.abs(b) < 1e-12) {
    // Diagonal matrix — eigenvectors are axis-aligned
    if (a >= c) {
      v1 = [1, 0];
      v2 = [0, 1];
    } else {
      v1 = [0, 1];
      v2 = [1, 0];
    }
  } else {
    const raw1x = lambda1 - c;
    const raw1y = b;
    const len1 = Math.sqrt(raw1x * raw1x + raw1y * raw1y);
    v1 = [raw1x / len1, raw1y / len1];

    const raw2x = lambda2 - c;
    const raw2y = b;
    const len2 = Math.sqrt(raw2x * raw2x + raw2y * raw2y);
    v2 = [raw2x / len2, raw2y / len2];
  }

  return { lambda1, lambda2, v1, v2 };
}

const PC1_COLOR = '#e74c3c';
const PC2_COLOR = '#2ecc71';
const CANDIDATE_COLOR = '#f39c12';

export default function PCAGeometryExplorer() {
  const { ref, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [sigma1sq, setSigma1sq] = useState(3.0);
  const [sigma2sq, setSigma2sq] = useState(1.5);
  const [rho, setRho] = useState(0.6);
  const [angle, setAngle] = useState(0);
  const [showProjection, setShowProjection] = useState(false);
  const [seed, setSeed] = useState(42);

  const isDesktop = (containerWidth || 0) > 640;

  // Covariance matrix entries
  const sigma1 = Math.sqrt(sigma1sq);
  const sigma2 = Math.sqrt(sigma2sq);
  const covXY = rho * sigma1 * sigma2;

  // Eigen decomposition
  const eigen = useMemo(
    () => eigenDecomposition2x2(sigma1sq, covXY, sigma2sq),
    [sigma1sq, covXY, sigma2sq],
  );

  // Generate data via Cholesky factorization
  const points = useMemo(() => {
    // Cholesky of [[sigma1sq, covXY], [covXY, sigma2sq]]
    const L11 = sigma1;
    const L21 = covXY / sigma1;
    const L22inner = sigma2sq - (covXY * covXY) / sigma1sq;
    const L22 = Math.sqrt(Math.max(L22inner, 1e-10));

    const pts: [number, number][] = [];
    for (let i = 0; i < NUM_POINTS; i++) {
      const z1 = seededNormal(i, seed);
      const z2 = seededNormal(i + NUM_POINTS, seed);
      const x = L11 * z1;
      const y = L21 * z1 + L22 * z2;
      pts.push([x, y]);
    }
    return pts;
  }, [sigma1, sigma2sq, covXY, sigma1sq, seed]);

  // Rayleigh quotient at candidate angle
  const rayleigh = useMemo(() => {
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);
    return sigma1sq * cosA * cosA + 2 * covXY * cosA * sinA + sigma2sq * sinA * sinA;
  }, [sigma1sq, sigma2sq, covXY, angle]);

  // SVG dimensions
  const svgContainerWidth = useMemo(() => {
    if (!containerWidth) return 400;
    return isDesktop ? Math.floor(containerWidth * 0.5) : containerWidth;
  }, [containerWidth, isDesktop]);

  const innerW = svgContainerWidth - MARGIN.left - MARGIN.right;
  const innerH = SVG_HEIGHT - MARGIN.top - MARGIN.bottom;

  // Data extent for scales (symmetric around origin)
  const dataExtent = useMemo(() => {
    let maxAbs = 1;
    for (const [x, y] of points) {
      maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y));
    }
    return Math.ceil(maxAbs * 1.15);
  }, [points]);

  const xScale = useMemo(
    () => d3.scaleLinear().domain([-dataExtent, dataExtent]).range([MARGIN.left, MARGIN.left + innerW]),
    [dataExtent, innerW],
  );
  const yScale = useMemo(
    () => d3.scaleLinear().domain([-dataExtent, dataExtent]).range([MARGIN.top + innerH, MARGIN.top]),
    [dataExtent, innerH],
  );

  const xTicks = xScale.ticks(5);
  const yTicks = yScale.ticks(5);

  // Arrow length proportional to sqrt(eigenvalue), scaled to data extent
  const arrowScale = useMemo(() => {
    const maxLen = Math.sqrt(eigen.lambda1);
    return (dataExtent * 0.6) / Math.max(maxLen, 1e-6);
  }, [eigen.lambda1, dataExtent]);

  // Projection lines (project onto PC1)
  const projections = useMemo(() => {
    if (!showProjection) return null;
    const [vx, vy] = eigen.v1;
    return points.map(([px, py]) => {
      const dot = px * vx + py * vy;
      const projX = dot * vx;
      const projY = dot * vy;
      return { origX: px, origY: py, projX, projY };
    });
  }, [showProjection, eigen.v1, points]);

  // Explained variance ratio
  const evr1 = eigen.lambda1 / (eigen.lambda1 + eigen.lambda2);
  const evr2 = eigen.lambda2 / (eigen.lambda1 + eigen.lambda2);
  const conditionNumber = eigen.lambda2 > 1e-10 ? eigen.lambda1 / eigen.lambda2 : Infinity;

  // Eigenvector angle (for highlighting when candidate aligns)
  const pc1Angle = Math.atan2(eigen.v1[1], eigen.v1[0]);
  // Normalize angles to [0, pi) for comparison
  const normAngle = ((angle % Math.PI) + Math.PI) % Math.PI;
  const normPC1 = ((pc1Angle % Math.PI) + Math.PI) % Math.PI;
  const angleMatch = Math.abs(normAngle - normPC1) < 0.05;

  const handleSigma1Change = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setSigma1sq(parseFloat(e.target.value));
  }, []);
  const handleSigma2Change = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setSigma2sq(parseFloat(e.target.value));
  }, []);
  const handleRhoChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setRho(parseFloat(e.target.value));
  }, []);
  const handleAngleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setAngle(parseFloat(e.target.value));
  }, []);
  const handleToggleProjection = useCallback(() => {
    setShowProjection((prev) => !prev);
  }, []);
  const handleRegenerate = useCallback(() => {
    setSeed((prev) => prev + 1);
  }, []);

  const fmt = (v: number) => v.toFixed(3);
  const fmt2 = (v: number) => v.toFixed(2);

  const cx0 = xScale(0);
  const cy0 = yScale(0);

  return (
    <div ref={ref} className="w-full space-y-3">
      {/* Main panels */}
      <div
        style={{
          display: 'flex',
          flexDirection: isDesktop ? 'row' : 'column',
          gap: '1rem',
        }}
      >
        {/* Left panel: SVG data cloud */}
        <div style={{ flex: isDesktop ? '0 0 50%' : '1 1 auto' }}>
          <svg role="img" aria-label="PCAGeometry explorer visualization"
            width={svgContainerWidth}
            height={SVG_HEIGHT}
            viewBox={`0 0 ${svgContainerWidth} ${SVG_HEIGHT}`}
            style={{ fontFamily: 'var(--font-sans)', overflow: 'visible' }}
          >
            {/* Grid lines */}
            {xTicks.map((t) => (
              <line
                key={`gx-${t}`}
                x1={xScale(t)}
                x2={xScale(t)}
                y1={MARGIN.top}
                y2={MARGIN.top + innerH}
                stroke="var(--color-border)"
                strokeOpacity={0.3}
                strokeDasharray="2,4"
              />
            ))}
            {yTicks.map((t) => (
              <line
                key={`gy-${t}`}
                x1={MARGIN.left}
                x2={MARGIN.left + innerW}
                y1={yScale(t)}
                y2={yScale(t)}
                stroke="var(--color-border)"
                strokeOpacity={0.3}
                strokeDasharray="2,4"
              />
            ))}

            {/* Axes */}
            <line
              x1={MARGIN.left}
              x2={MARGIN.left + innerW}
              y1={cy0}
              y2={cy0}
              stroke="var(--color-muted)"
              strokeWidth={1}
            />
            <line
              x1={cx0}
              x2={cx0}
              y1={MARGIN.top}
              y2={MARGIN.top + innerH}
              stroke="var(--color-muted)"
              strokeWidth={1}
            />

            {/* Axis labels */}
            {xTicks.filter((t) => t !== 0).map((t) => (
              <text
                key={`xl-${t}`}
                x={xScale(t)}
                y={MARGIN.top + innerH + 16}
                textAnchor="middle"
                fontSize={10}
                fill="var(--color-muted)"
              >
                {t}
              </text>
            ))}
            {yTicks.filter((t) => t !== 0).map((t) => (
              <text
                key={`yl-${t}`}
                x={MARGIN.left - 8}
                y={yScale(t) + 3}
                textAnchor="end"
                fontSize={10}
                fill="var(--color-muted)"
              >
                {t}
              </text>
            ))}
            <text
              x={MARGIN.left + innerW / 2}
              y={MARGIN.top + innerH + 32}
              textAnchor="middle"
              fontSize={12}
              fill="var(--color-text)"
            >
              x₁
            </text>
            <text
              x={MARGIN.left - 30}
              y={MARGIN.top + innerH / 2}
              textAnchor="middle"
              fontSize={12}
              fill="var(--color-text)"
              transform={`rotate(-90, ${MARGIN.left - 30}, ${MARGIN.top + innerH / 2})`}
            >
              x₂
            </text>

            {/* Projection lines */}
            {projections &&
              projections.map((p, i) => (
                <line
                  key={`proj-${i}`}
                  x1={xScale(p.origX)}
                  y1={yScale(p.origY)}
                  x2={xScale(p.projX)}
                  y2={yScale(p.projY)}
                  stroke="var(--color-muted)"
                  strokeOpacity={0.25}
                  strokeWidth={0.5}
                />
              ))}

            {/* Data points */}
            {points.map(([px, py], i) => (
              <circle
                key={`pt-${i}`}
                cx={xScale(px)}
                cy={yScale(py)}
                r={3}
                fill="steelblue"
                fillOpacity={0.3}
              />
            ))}

            {/* Projected points on PC1 */}
            {projections &&
              projections.map((p, i) => (
                <circle
                  key={`ppt-${i}`}
                  cx={xScale(p.projX)}
                  cy={yScale(p.projY)}
                  r={2}
                  fill={PC1_COLOR}
                  fillOpacity={0.5}
                />
              ))}

            {/* PC1 arrow */}
            <line
              x1={cx0}
              y1={cy0}
              x2={xScale(eigen.v1[0] * Math.sqrt(eigen.lambda1) * arrowScale)}
              y2={yScale(eigen.v1[1] * Math.sqrt(eigen.lambda1) * arrowScale)}
              stroke={PC1_COLOR}
              strokeWidth={2.5}
              markerEnd="url(#arrowPC1)"
            />
            {/* PC2 arrow */}
            <line
              x1={cx0}
              y1={cy0}
              x2={xScale(eigen.v2[0] * Math.sqrt(eigen.lambda2) * arrowScale)}
              y2={yScale(eigen.v2[1] * Math.sqrt(eigen.lambda2) * arrowScale)}
              stroke={PC2_COLOR}
              strokeWidth={2.5}
              markerEnd="url(#arrowPC2)"
            />

            {/* Candidate direction (dashed) */}
            <line
              x1={xScale(-dataExtent * 0.85 * Math.cos(angle))}
              y1={yScale(-dataExtent * 0.85 * Math.sin(angle))}
              x2={xScale(dataExtent * 0.85 * Math.cos(angle))}
              y2={yScale(dataExtent * 0.85 * Math.sin(angle))}
              stroke={CANDIDATE_COLOR}
              strokeWidth={1.5}
              strokeDasharray="6,4"
              strokeOpacity={0.8}
            />

            {/* Rayleigh quotient label near candidate direction */}
            <text
              x={xScale(dataExtent * 0.65 * Math.cos(angle)) + 8}
              y={yScale(dataExtent * 0.65 * Math.sin(angle)) - 8}
              fontSize={11}
              fontWeight={600}
              fill={CANDIDATE_COLOR}
              style={{ fontFamily: 'var(--font-sans)' }}
            >
              R(θ) = {fmt2(rayleigh)}
            </text>

            {/* PC labels */}
            <text
              x={xScale(eigen.v1[0] * Math.sqrt(eigen.lambda1) * arrowScale) + 6}
              y={yScale(eigen.v1[1] * Math.sqrt(eigen.lambda1) * arrowScale) - 6}
              fontSize={11}
              fontWeight={600}
              fill={PC1_COLOR}
            >
              PC1
            </text>
            <text
              x={xScale(eigen.v2[0] * Math.sqrt(eigen.lambda2) * arrowScale) + 6}
              y={yScale(eigen.v2[1] * Math.sqrt(eigen.lambda2) * arrowScale) - 6}
              fontSize={11}
              fontWeight={600}
              fill={PC2_COLOR}
            >
              PC2
            </text>

            {/* Arrow markers */}
            <defs>
              <marker
                id="arrowPC1"
                markerWidth={8}
                markerHeight={6}
                refX={8}
                refY={3}
                orient="auto"
              >
                <polygon points="0 0, 8 3, 0 6" fill={PC1_COLOR} />
              </marker>
              <marker
                id="arrowPC2"
                markerWidth={8}
                markerHeight={6}
                refX={8}
                refY={3}
                orient="auto"
              >
                <polygon points="0 0, 8 3, 0 6" fill={PC2_COLOR} />
              </marker>
            </defs>
          </svg>
        </div>

        {/* Right panel: readout */}
        <div
          style={{
            flex: isDesktop ? '0 0 50%' : '1 1 auto',
            fontFamily: 'var(--font-sans)',
            fontSize: '13px',
            color: 'var(--color-text)',
            padding: '0.5rem',
          }}
        >
          {/* Covariance matrix */}
          <div style={{ marginBottom: '1rem' }}>
            <div
              className="text-xs font-medium"
              style={{ color: 'var(--color-muted)', marginBottom: '0.25rem' }}
            >
              Covariance Matrix Σ
            </div>
            <div
              style={{
                display: 'inline-grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '2px 12px',
                padding: '6px 10px',
                borderRadius: '6px',
                border: '1px solid var(--color-border)',
                background: 'var(--color-surface)',
                fontFamily: 'monospace',
                fontSize: '13px',
              }}
            >
              <span>{fmt(sigma1sq)}</span>
              <span>{fmt(covXY)}</span>
              <span>{fmt(covXY)}</span>
              <span>{fmt(sigma2sq)}</span>
            </div>
          </div>

          {/* Eigenvalues */}
          <div style={{ marginBottom: '0.75rem' }}>
            <div
              className="text-xs font-medium"
              style={{ color: 'var(--color-muted)', marginBottom: '0.25rem' }}
            >
              Eigenvalues
            </div>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <span>
                <span
                  style={{
                    display: 'inline-block',
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: PC1_COLOR,
                    marginRight: 4,
                  }}
                />
                λ₁ = {fmt(eigen.lambda1)}
              </span>
              <span>
                <span
                  style={{
                    display: 'inline-block',
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: PC2_COLOR,
                    marginRight: 4,
                  }}
                />
                λ₂ = {fmt(eigen.lambda2)}
              </span>
            </div>
          </div>

          {/* Eigenvectors */}
          <div style={{ marginBottom: '0.75rem' }}>
            <div
              className="text-xs font-medium"
              style={{ color: 'var(--color-muted)', marginBottom: '0.25rem' }}
            >
              Eigenvectors
            </div>
            <div style={{ fontFamily: 'monospace', fontSize: '12px', lineHeight: 1.6 }}>
              <div>
                <span style={{ color: PC1_COLOR, fontWeight: 600 }}>q₁</span> = [{fmt(eigen.v1[0])},{' '}
                {fmt(eigen.v1[1])}]
              </div>
              <div>
                <span style={{ color: PC2_COLOR, fontWeight: 600 }}>q₂</span> = [{fmt(eigen.v2[0])},{' '}
                {fmt(eigen.v2[1])}]
              </div>
            </div>
          </div>

          {/* Explained variance ratio */}
          <div style={{ marginBottom: '0.75rem' }}>
            <div
              className="text-xs font-medium"
              style={{ color: 'var(--color-muted)', marginBottom: '0.25rem' }}
            >
              Explained Variance Ratio
            </div>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <div
                style={{
                  flex: 1,
                  height: 14,
                  borderRadius: 4,
                  overflow: 'hidden',
                  display: 'flex',
                  border: '1px solid var(--color-border)',
                }}
              >
                <div
                  style={{
                    width: `${evr1 * 100}%`,
                    background: PC1_COLOR,
                    transition: 'width 0.2s',
                  }}
                />
                <div
                  style={{
                    width: `${evr2 * 100}%`,
                    background: PC2_COLOR,
                    transition: 'width 0.2s',
                  }}
                />
              </div>
            </div>
            <div style={{ fontSize: '11px', marginTop: '2px' }}>
              <span style={{ color: PC1_COLOR }}>PC1: {(evr1 * 100).toFixed(1)}%</span>
              {' · '}
              <span style={{ color: PC2_COLOR }}>PC2: {(evr2 * 100).toFixed(1)}%</span>
            </div>
          </div>

          {/* Condition number */}
          <div style={{ marginBottom: '0.75rem' }}>
            <div
              className="text-xs font-medium"
              style={{ color: 'var(--color-muted)', marginBottom: '0.25rem' }}
            >
              Condition Number
            </div>
            <div style={{ fontFamily: 'monospace' }}>
              κ = λ₁/λ₂ = {conditionNumber === Infinity ? '∞' : fmt2(conditionNumber)}
            </div>
          </div>

          {/* Current Rayleigh quotient */}
          <div
            style={{
              padding: '8px 10px',
              borderRadius: '6px',
              border: `1px solid ${angleMatch ? PC1_COLOR : 'var(--color-border)'}`,
              background: angleMatch ? `${PC1_COLOR}11` : 'var(--color-surface)',
              transition: 'all 0.2s',
            }}
          >
            <div
              className="text-xs font-medium"
              style={{ color: 'var(--color-muted)', marginBottom: '0.25rem' }}
            >
              Rayleigh Quotient R(θ) = w⊤Σw
            </div>
            <div style={{ fontFamily: 'monospace', fontSize: '14px' }}>
              <span style={{ color: CANDIDATE_COLOR, fontWeight: 700 }}>{fmt2(rayleigh)}</span>
              <span style={{ color: 'var(--color-muted)', fontSize: '12px', marginLeft: 8 }}>
                max = λ₁ = {fmt2(eigen.lambda1)}
              </span>
            </div>
            {angleMatch && (
              <div style={{ fontSize: '11px', color: PC1_COLOR, marginTop: 2 }}>
                Maximized — θ aligns with the first eigenvector
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom controls */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: isDesktop ? '1fr 1fr' : '1fr',
          gap: '0.5rem 1.5rem',
          padding: '0.75rem',
          borderRadius: '8px',
          border: '1px solid var(--color-border)',
          background: 'var(--color-surface)',
          fontFamily: 'var(--font-sans)',
        }}
      >
        {/* sigma1sq slider */}
        <label style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
          <span className="text-xs font-medium" style={{ fontFamily: 'var(--font-sans)' }}>
            σ₁² = {fmt2(sigma1sq)}
          </span>
          <input
            type="range"
            min={0.5}
            max={5.0}
            step={0.1}
            value={sigma1sq}
            onChange={handleSigma1Change}
            style={{ accentColor: 'var(--color-accent)' }}
          />
        </label>

        {/* sigma2sq slider */}
        <label style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
          <span className="text-xs font-medium" style={{ fontFamily: 'var(--font-sans)' }}>
            σ₂² = {fmt2(sigma2sq)}
          </span>
          <input
            type="range"
            min={0.5}
            max={5.0}
            step={0.1}
            value={sigma2sq}
            onChange={handleSigma2Change}
            style={{ accentColor: 'var(--color-accent)' }}
          />
        </label>

        {/* rho slider */}
        <label style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
          <span className="text-xs font-medium" style={{ fontFamily: 'var(--font-sans)' }}>
            ρ = {fmt2(rho)}
          </span>
          <input
            type="range"
            min={-0.99}
            max={0.99}
            step={0.01}
            value={rho}
            onChange={handleRhoChange}
            style={{ accentColor: 'var(--color-accent)' }}
          />
        </label>

        {/* angle slider */}
        <label style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
          <span className="text-xs font-medium" style={{ fontFamily: 'var(--font-sans)' }}>
            θ = {fmt2(angle)} rad ({fmt2((angle * 180) / Math.PI)}°)
          </span>
          <input
            type="range"
            min={0}
            max={Math.PI}
            step={0.01}
            value={angle}
            onChange={handleAngleChange}
            style={{ accentColor: CANDIDATE_COLOR }}
          />
        </label>

        {/* Buttons row */}
        <div
          style={{
            display: 'flex',
            gap: '0.75rem',
            alignItems: 'center',
            gridColumn: isDesktop ? 'span 2' : 'span 1',
          }}
        >
          <button
            onClick={handleRegenerate}
            className="text-xs font-medium"
            style={{
              padding: '6px 14px',
              borderRadius: '6px',
              border: '1px solid var(--color-border)',
              background: 'var(--color-surface)',
              color: 'var(--color-text)',
              cursor: 'pointer',
              fontFamily: 'var(--font-sans)',
            }}
          >
            Regenerate Data
          </button>
          <button
            onClick={handleToggleProjection}
            className="text-xs font-medium"
            style={{
              padding: '6px 14px',
              borderRadius: '6px',
              border: `1px solid ${showProjection ? PC1_COLOR : 'var(--color-border)'}`,
              background: showProjection ? `${PC1_COLOR}18` : 'var(--color-surface)',
              color: showProjection ? PC1_COLOR : 'var(--color-text)',
              cursor: 'pointer',
              fontFamily: 'var(--font-sans)',
              transition: 'all 0.15s',
            }}
          >
            {showProjection ? 'Hide Projection' : 'Show Projection'}
          </button>
        </div>
      </div>
    </div>
  );
}
