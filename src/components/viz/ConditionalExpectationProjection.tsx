import { useState, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

const NUM_POINTS = 200;
const MARGIN = { top: 16, right: 16, bottom: 36, left: 40 };

// Seeded LCG PRNG
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

export default function ConditionalExpectationProjection() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();
  const [rho, setRho] = useState(0.7);
  const [showResiduals, setShowResiduals] = useState(false);
  const seed = 42;

  const isDesktop = (containerWidth || 0) > 640;
  const panelW = isDesktop ? Math.floor((containerWidth - 24) / 2) : containerWidth;
  const panelH = Math.min(340, panelW * 0.85);

  // Generate bivariate normal data: X = Z1, Y = rho*Z1 + sqrt(1-rho^2)*Z2
  const points = useMemo(() => {
    const pts: { x: number; y: number }[] = [];
    for (let i = 0; i < NUM_POINTS; i++) {
      const z1 = seededNormal(i, seed);
      const z2 = seededNormal(i + NUM_POINTS, seed + 1);
      pts.push({
        x: z1,
        y: rho * z1 + Math.sqrt(Math.max(0, 1 - rho * rho)) * z2,
      });
    }
    return pts;
  }, [rho]);

  // Scales for scatter plot
  const extent = 3.5;
  const scatterScaleX = useMemo(
    () => d3.scaleLinear().domain([-extent, extent]).range([MARGIN.left, panelW - MARGIN.right]),
    [panelW],
  );
  const scatterScaleY = useMemo(
    () => d3.scaleLinear().domain([-extent, extent]).range([panelH - MARGIN.bottom, MARGIN.top]),
    [panelH],
  );

  // MSE curve: MSE(b) = E[(Y - bX)^2] = 1 - 2*rho*b + b^2 for standard normals
  const mseBSlope = useMemo(() => {
    const bValues: number[] = [];
    const mseValues: number[] = [];
    for (let b = -1.5; b <= 1.5; b += 0.02) {
      bValues.push(b);
      mseValues.push(1 - 2 * rho * b + b * b);
    }
    return { bValues, mseValues };
  }, [rho]);

  const mseScaleX = useMemo(
    () => d3.scaleLinear().domain([-1.5, 1.5]).range([MARGIN.left, panelW - MARGIN.right]),
    [panelW],
  );
  const mseMax = Math.max(1 - 2 * rho * (-1.5) + (-1.5) * (-1.5), 1 - 2 * rho * 1.5 + 1.5 * 1.5, 1);
  const mseScaleY = useMemo(
    () => d3.scaleLinear().domain([0, mseMax + 0.2]).range([panelH - MARGIN.bottom, MARGIN.top]),
    [panelH, mseMax],
  );

  const msePath = useMemo(() => {
    const lineGen = d3.line<number>()
      .x((_, i) => mseScaleX(mseBSlope.bValues[i]))
      .y((_, i) => mseScaleY(mseBSlope.mseValues[i]));
    return lineGen(mseBSlope.bValues) || '';
  }, [mseBSlope, mseScaleX, mseScaleY]);

  const mseMin = 1 - rho * rho;

  // Axis ticks
  const xTicks = [-3, -2, -1, 0, 1, 2, 3];
  const yTicks = [-3, -2, -1, 0, 1, 2, 3];
  const bTicks = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5];

  const handleRho = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setRho(parseFloat(e.target.value));
  }, []);

  if (!containerWidth) {
    return <div ref={containerRef} style={{ minHeight: 300 }} />;
  }

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
        Conditional Expectation as L² Projection
      </div>

      <div style={{ display: 'flex', flexDirection: isDesktop ? 'row' : 'column', gap: '12px' }}>
        {/* Left: Scatter plot */}
        <svg width={panelW} height={panelH} style={{ border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)' }}>
          {/* Grid lines */}
          {xTicks.map(t => (
            <line key={`gx-${t}`} x1={scatterScaleX(t)} x2={scatterScaleX(t)} y1={MARGIN.top} y2={panelH - MARGIN.bottom}
              stroke="var(--color-border)" strokeWidth={0.5} strokeDasharray={t === 0 ? 'none' : '3 3'} />
          ))}
          {yTicks.map(t => (
            <line key={`gy-${t}`} y1={scatterScaleY(t)} y2={scatterScaleY(t)} x1={MARGIN.left} x2={panelW - MARGIN.right}
              stroke="var(--color-border)" strokeWidth={0.5} strokeDasharray={t === 0 ? 'none' : '3 3'} />
          ))}
          {/* Axis labels */}
          {xTicks.filter(t => t !== 0).map(t => (
            <text key={`xl-${t}`} x={scatterScaleX(t)} y={panelH - MARGIN.bottom + 14} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--color-text-secondary)">{t}</text>
          ))}
          {yTicks.filter(t => t !== 0).map(t => (
            <text key={`yl-${t}`} x={MARGIN.left - 6} y={scatterScaleY(t) + 3} textAnchor="end" fontSize={10} fontFamily="var(--font-mono)" fill="var(--color-text-secondary)">{t}</text>
          ))}
          <text x={panelW / 2} y={panelH - 2} textAnchor="middle" fontSize={11} fontFamily="var(--font-serif)" fill="var(--color-text-secondary)">X</text>
          <text x={8} y={panelH / 2} textAnchor="middle" fontSize={11} fontFamily="var(--font-serif)" fill="var(--color-text-secondary)" transform={`rotate(-90, 8, ${panelH / 2})`}>Y</text>

          {/* Residual lines (if toggled) */}
          {showResiduals && points.map((p, i) => {
            const yHat = rho * p.x;
            return (
              <line key={`r-${i}`}
                x1={scatterScaleX(p.x)} y1={scatterScaleY(p.y)}
                x2={scatterScaleX(p.x)} y2={scatterScaleY(yHat)}
                stroke="var(--color-text-secondary)" strokeWidth={0.4} opacity={0.3} />
            );
          })}

          {/* Data points */}
          {points.map((p, i) => (
            <circle key={`p-${i}`}
              cx={scatterScaleX(p.x)} cy={scatterScaleY(p.y)}
              r={2.5} fill="steelblue" opacity={0.45} />
          ))}

          {/* Regression line: E[Y|X=x] = rho * x */}
          <line
            x1={scatterScaleX(-extent)} y1={scatterScaleY(-rho * extent)}
            x2={scatterScaleX(extent)} y2={scatterScaleY(rho * extent)}
            stroke="#DC2626" strokeWidth={2.5} opacity={0.85}
          />
          <text x={scatterScaleX(2.2)} y={scatterScaleY(rho * 2.2) - 8} fontSize={11} fontFamily="var(--font-serif)" fill="#DC2626">
            E[Y|X] = ρx
          </text>
        </svg>

        {/* Right: MSE curve */}
        <svg width={panelW} height={panelH} style={{ border: '1px solid var(--color-border)', borderRadius: '6px', background: 'var(--color-muted-bg)' }}>
          {/* Grid */}
          {bTicks.map(t => (
            <line key={`gb-${t}`} x1={mseScaleX(t)} x2={mseScaleX(t)} y1={MARGIN.top} y2={panelH - MARGIN.bottom}
              stroke="var(--color-border)" strokeWidth={0.5} strokeDasharray={t === 0 ? 'none' : '3 3'} />
          ))}
          {/* Axis labels */}
          {bTicks.map(t => (
            <text key={`bl-${t}`} x={mseScaleX(t)} y={panelH - MARGIN.bottom + 14} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--color-text-secondary)">{t}</text>
          ))}
          <text x={panelW / 2} y={panelH - 2} textAnchor="middle" fontSize={11} fontFamily="var(--font-serif)" fill="var(--color-text-secondary)">slope b</text>
          <text x={8} y={panelH / 2} textAnchor="middle" fontSize={11} fontFamily="var(--font-serif)" fill="var(--color-text-secondary)" transform={`rotate(-90, 8, ${panelH / 2})`}>MSE(b)</text>

          {/* MSE parabola */}
          <path d={msePath} fill="none" stroke="var(--color-accent)" strokeWidth={2.5} />

          {/* Minimum marker */}
          <line
            x1={mseScaleX(rho)} x2={mseScaleX(rho)}
            y1={mseScaleY(mseMin)} y2={panelH - MARGIN.bottom}
            stroke="#DC2626" strokeWidth={1.5} strokeDasharray="4 3" />
          <circle cx={mseScaleX(rho)} cy={mseScaleY(mseMin)} r={5} fill="#DC2626" opacity={0.9} />
          <text x={mseScaleX(rho) + 8} y={mseScaleY(mseMin) - 6} fontSize={11} fontFamily="var(--font-mono)" fill="#DC2626">
            b* = ρ = {rho.toFixed(2)}
          </text>
          <text x={mseScaleX(rho) + 8} y={mseScaleY(mseMin) + 10} fontSize={10} fontFamily="var(--font-mono)" fill="var(--color-text-secondary)">
            MSE = 1−ρ² = {mseMin.toFixed(3)}
          </text>
        </svg>
      </div>

      {/* Controls */}
      <div style={{ marginTop: '14px', display: 'flex', flexWrap: 'wrap', gap: '16px', alignItems: 'center' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)' }}>
          ρ = <strong style={{ color: 'var(--color-text)', minWidth: '40px' }}>{rho.toFixed(2)}</strong>
          <input type="range" min={-0.95} max={0.95} step={0.05} value={rho} onChange={handleRho}
            style={{ width: '180px', accentColor: 'var(--color-accent)' }} />
        </label>
        <button
          onClick={() => setShowResiduals(prev => !prev)}
          style={{
            padding: '4px 12px',
            fontSize: '12px',
            fontFamily: 'var(--font-mono)',
            border: `1px solid ${showResiduals ? 'var(--color-accent)' : 'var(--color-border)'}`,
            borderRadius: '4px',
            background: showResiduals ? 'var(--color-definition-bg)' : 'transparent',
            color: showResiduals ? 'var(--color-accent)' : 'var(--color-text-secondary)',
            cursor: 'pointer',
          }}
        >
          {showResiduals ? 'Hide' : 'Show'} residuals
        </button>
        <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}>
          Variance explained: <strong style={{ color: 'var(--color-accent)' }}>{(rho * rho * 100).toFixed(1)}%</strong>
        </span>
      </div>
    </div>
  );
}
