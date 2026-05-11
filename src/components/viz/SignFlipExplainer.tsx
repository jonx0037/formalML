// =============================================================================
// SignFlipExplainer.tsx
//
// §9.3 — Schematic showing the same flow T_phi used two ways:
//   (a) density evaluation: log p(x) = log p_Z(T^-1(x)) − log|det dT/dz|
//   (b) entropy:            H[q]     = H[p_eps]      + E[log|det dT/dε|]
// The same log-det term highlighted in both, with the sign flip annotated.
//
// Static SVG, no interactivity. Educational target: the sign flip is bookkeeping
// (entropy is −log p), not a separate mathematical fact.
// =============================================================================

export default function SignFlipExplainer() {
  const W = 760;
  const H = 320;

  return (
    <figure className="my-8 not-prose">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        role="img"
        aria-label="Two-panel schematic comparing the log-det sign in the density formula (minus) and the entropy formula (plus) for the same flow."
        style={{ maxWidth: '100%', height: 'auto' }}
      >
        {/* Panel A — density */}
        <rect
          x={20}
          y={20}
          width={350}
          height={280}
          rx={8}
          fill="none"
          style={{ stroke: 'var(--color-border)', strokeWidth: 1 }}
        />
        <text x={195} y={50} textAnchor="middle"
              style={{ fill: 'var(--color-text)', fontSize: 14, fontWeight: 600, fontFamily: 'var(--font-sans, sans-serif)' }}>
          (a) Density evaluation
        </text>
        <text x={195} y={75} textAnchor="middle"
              style={{ fill: 'var(--color-text-secondary)', fontSize: 12, fontFamily: 'var(--font-sans, sans-serif)' }}>
          given x, push back to z = T⁻¹(x)
        </text>

        {/* The density formula, broken into pieces with sign-flip highlighting */}
        <text x={195} y={130} textAnchor="middle"
              style={{ fill: 'var(--color-text)', fontSize: 15, fontFamily: 'var(--font-serif, serif)', fontStyle: 'italic' }}>
          log p(x)
        </text>
        <text x={195} y={155} textAnchor="middle"
              style={{ fill: 'var(--color-text)', fontSize: 13, fontFamily: 'var(--font-serif, serif)' }}>
          = log p_Z(T⁻¹(x))
        </text>
        <g>
          <text x={195} y={195} textAnchor="middle"
                style={{ fill: 'var(--color-accent)', fontSize: 18, fontWeight: 700, fontFamily: 'var(--font-serif, serif)' }}>
            −
          </text>
          <text x={235} y={195} textAnchor="middle"
                style={{ fill: 'var(--color-text)', fontSize: 13, fontFamily: 'var(--font-serif, serif)' }}>
            log |det ∂T/∂z|
          </text>
        </g>

        {/* Stretch interpretation */}
        <text x={195} y={245} textAnchor="middle"
              style={{ fill: 'var(--color-text-secondary)', fontSize: 11, fontStyle: 'italic', fontFamily: 'var(--font-sans, sans-serif)' }}>
          stretching T → reduces density
        </text>
        <text x={195} y={263} textAnchor="middle"
              style={{ fill: 'var(--color-text-secondary)', fontSize: 11, fontStyle: 'italic', fontFamily: 'var(--font-sans, sans-serif)' }}>
          (same mass / larger region)
        </text>

        {/* Panel B — entropy */}
        <rect
          x={390}
          y={20}
          width={350}
          height={280}
          rx={8}
          fill="none"
          style={{ stroke: 'var(--color-border)', strokeWidth: 1 }}
        />
        <text x={565} y={50} textAnchor="middle"
              style={{ fill: 'var(--color-text)', fontSize: 14, fontWeight: 600, fontFamily: 'var(--font-sans, sans-serif)' }}>
          (b) Entropy
        </text>
        <text x={565} y={75} textAnchor="middle"
              style={{ fill: 'var(--color-text-secondary)', fontSize: 12, fontFamily: 'var(--font-sans, sans-serif)' }}>
          q is the pushforward of p_ε under T
        </text>

        <text x={565} y={130} textAnchor="middle"
              style={{ fill: 'var(--color-text)', fontSize: 15, fontFamily: 'var(--font-serif, serif)', fontStyle: 'italic' }}>
          H[q]
        </text>
        <text x={565} y={155} textAnchor="middle"
              style={{ fill: 'var(--color-text)', fontSize: 13, fontFamily: 'var(--font-serif, serif)' }}>
          = H[p_ε]
        </text>
        <g>
          <text x={565} y={195} textAnchor="middle"
                style={{ fill: 'var(--color-accent)', fontSize: 18, fontWeight: 700, fontFamily: 'var(--font-serif, serif)' }}>
            +
          </text>
          <text x={605} y={195} textAnchor="middle"
                style={{ fill: 'var(--color-text)', fontSize: 13, fontFamily: 'var(--font-serif, serif)' }}>
            E[log |det ∂T/∂ε|]
          </text>
        </g>

        <text x={565} y={245} textAnchor="middle"
              style={{ fill: 'var(--color-text-secondary)', fontSize: 11, fontStyle: 'italic', fontFamily: 'var(--font-sans, sans-serif)' }}>
          stretching T → spreads distribution
        </text>
        <text x={565} y={263} textAnchor="middle"
              style={{ fill: 'var(--color-text-secondary)', fontSize: 11, fontStyle: 'italic', fontFamily: 'var(--font-sans, sans-serif)' }}>
          (more uncertainty / higher entropy)
        </text>

        {/* Title bar across both: the sign flip */}
        <text x={W / 2} y={H - 8} textAnchor="middle"
              style={{ fill: 'var(--color-text)', fontSize: 12, fontFamily: 'var(--font-sans, sans-serif)' }}>
          Same log-det. Opposite sign. Density (−) vs entropy (+).
        </text>
      </svg>
      <figcaption className="text-sm mt-2 text-center" style={{ color: 'var(--color-text-secondary)' }}>
        The sign flip of §9.3: density carries a minus sign in front of log|det ∂T/∂z|; entropy carries a plus. Once you
        see why (entropy is −log p), every flow-VI paper reads without sign-tracking gymnastics.
      </figcaption>
    </figure>
  );
}
