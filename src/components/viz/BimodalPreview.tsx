// =============================================================================
// BimodalPreview.tsx
//
// §1.3 — Static-figure mirror of the notebook's Figure 1.1: base Gaussian on
// the left, bimodal target on the right, with an arrow labeled T between
// them. The visual hook, not a manipulable widget. The interactive build-up
// of T happens in §2's ChangeOfVariablesAnimator.
//
// Pure-SVG, no D3, no math module — content is fixed at compile time.
// =============================================================================

export default function BimodalPreview() {
  // Pre-computed standard-Gaussian curve and 0.5 N(-2, 0.5²) + 0.5 N(2, 0.5²)
  // bimodal curve sampled at 81 grid points x ∈ [-4, 4].
  const xs: number[] = [];
  for (let i = 0; i <= 80; i++) xs.push(-4 + (i * 8) / 80);

  const gauss = (x: number, mu: number, sigma: number) =>
    Math.exp(-((x - mu) ** 2) / (2 * sigma * sigma)) / (sigma * Math.sqrt(2 * Math.PI));

  const baseValues = xs.map((x) => gauss(x, 0, 1));
  const targetValues = xs.map((x) => 0.5 * gauss(x, -2, 0.5) + 0.5 * gauss(x, 2, 0.5));
  const maxY = Math.max(...baseValues, ...targetValues);

  const W = 760;
  const H = 280;
  const panelW = 280;
  const padX = 24;
  const padBottom = 36;
  const padTop = 16;
  const innerH = H - padTop - padBottom;

  const scaleX = (x: number) => padX + ((x + 4) / 8) * panelW;
  const scaleY = (y: number) => padTop + innerH - (y / maxY) * innerH;

  const pathFromValues = (vals: number[], offsetX = 0) => {
    let d = '';
    for (let i = 0; i < xs.length; i++) {
      const px = scaleX(xs[i]) + offsetX;
      const py = scaleY(vals[i]);
      d += i === 0 ? `M ${px} ${py}` : ` L ${px} ${py}`;
    }
    return d;
  };

  const baseOffset = 0;
  const targetOffset = panelW + 220; // panel + arrow gap

  return (
    <figure className="my-8 not-prose">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        role="img"
        aria-label="Two density panels connected by a labeled arrow T: standard Gaussian on the left, bimodal mixture on the right."
        style={{ maxWidth: '100%', height: 'auto' }}
      >
        {/* Axes */}
        <line
          x1={padX + baseOffset}
          y1={padTop + innerH}
          x2={padX + baseOffset + panelW}
          y2={padTop + innerH}
          style={{ stroke: 'var(--color-border)', strokeWidth: 1 }}
        />
        <line
          x1={padX + targetOffset}
          y1={padTop + innerH}
          x2={padX + targetOffset + panelW}
          y2={padTop + innerH}
          style={{ stroke: 'var(--color-border)', strokeWidth: 1 }}
        />

        {/* Base density curve */}
        <path
          d={pathFromValues(baseValues, baseOffset)}
          fill="none"
          style={{ stroke: 'var(--color-accent)', strokeWidth: 2 }}
        />
        {/* Target density curve */}
        <path
          d={pathFromValues(targetValues, targetOffset)}
          fill="none"
          style={{ stroke: 'var(--color-accent)', strokeWidth: 2 }}
        />

        {/* Panel labels */}
        <text
          x={padX + baseOffset + panelW / 2}
          y={padTop + innerH + 24}
          textAnchor="middle"
          style={{ fill: 'var(--color-text)', fontSize: 13, fontFamily: 'var(--font-sans, sans-serif)' }}
        >
          base: Z ~ N(0, 1)
        </text>
        <text
          x={padX + targetOffset + panelW / 2}
          y={padTop + innerH + 24}
          textAnchor="middle"
          style={{ fill: 'var(--color-text)', fontSize: 13, fontFamily: 'var(--font-sans, sans-serif)' }}
        >
          target: 0.5 N(-2, 0.25) + 0.5 N(2, 0.25)
        </text>

        {/* Arrow */}
        <defs>
          <marker
            id="bp-arrowhead"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="8"
            markerHeight="8"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" style={{ fill: 'var(--color-text-secondary)' }} />
          </marker>
        </defs>
        <line
          x1={padX + baseOffset + panelW + 22}
          y1={padTop + innerH / 2}
          x2={padX + targetOffset - 22}
          y2={padTop + innerH / 2}
          style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1.5 }}
          markerEnd="url(#bp-arrowhead)"
        />
        <text
          x={(padX + baseOffset + panelW + padX + targetOffset) / 2}
          y={padTop + innerH / 2 - 10}
          textAnchor="middle"
          style={{
            fill: 'var(--color-text)',
            fontSize: 18,
            fontFamily: 'var(--font-serif, serif)',
            fontStyle: 'italic',
          }}
        >
          T
        </text>
        <text
          x={(padX + baseOffset + panelW + padX + targetOffset) / 2}
          y={padTop + innerH / 2 + 22}
          textAnchor="middle"
          style={{ fill: 'var(--color-text-secondary)', fontSize: 11, fontFamily: 'var(--font-sans, sans-serif)' }}
        >
          (unbuilt — §3 and §4)
        </text>
      </svg>
      <figcaption className="text-sm mt-2 text-center" style={{ color: 'var(--color-text-secondary)' }}>
        Figure 1.1 — the two endpoints of the flow: a unimodal base on the left, a bimodal target on the right. §2 derives
        the formula relating their densities under a diffeomorphism $T$; §4 builds the family of $T$'s that flows
        parameterize.
      </figcaption>
    </figure>
  );
}
