// =============================================================================
// BNNStaticFigure.tsx
//
// Shared layout component for the bayesian-neural-networks topic's nine viz
// components. v1 ships static-figure-only per the handoff brief's
// "MDX/React-layer enhancement deferred to v2" framing; v2 will wrap each
// of these in a useD3-driven interactive layer that consumes the precomputed
// JSON payloads emitted by notebooks/bayesian-neural-networks/precompute_*.py
// (not yet written — tracked as follow-up workstream).
//
// In v1 this component is a responsive, dark-mode-safe `<figure>` rendering
// the corresponding notebook PNG with a caption and accessibility metadata.
// useResizeObserver is wired in so future v2 enhancements can pivot from
// width-aware static layout to width-aware D3 layout without restructuring
// the consumer components.
// =============================================================================

import { useResizeObserver } from './shared/useResizeObserver';

export interface BNNStaticFigureProps {
  /** Public-served figure URL — typically /images/topics/bayesian-neural-networks/0X_*.png. */
  figurePath: string;
  /** Detailed alt text for screen readers describing what the figure shows. */
  alt: string;
  /** Caption rendered below the figure. */
  caption: string;
  /** ARIA label for the wrapping figure element. */
  ariaLabel: string;
  /** Optional max-width override (default 720px). */
  maxWidth?: number;
}

export default function BNNStaticFigure({
  figurePath,
  alt,
  caption,
  ariaLabel,
  maxWidth = 720,
}: BNNStaticFigureProps) {
  // ResizeObserver wires width tracking now so v2 enhancements can use it
  // without restructuring; in v1 we only consume the ref for layout.
  const { ref } = useResizeObserver<HTMLDivElement>();

  return (
    <div
      ref={ref}
      role="figure"
      aria-label={ariaLabel}
      style={{
        width: '100%',
        margin: '2rem auto',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}
    >
      <img
        src={figurePath}
        alt={alt}
        loading="lazy"
        style={{
          width: '100%',
          maxWidth: `${maxWidth}px`,
          height: 'auto',
          backgroundColor: 'var(--color-surface, transparent)',
          borderRadius: '0.25rem',
          boxShadow: '0 1px 2px var(--color-shadow, rgba(0, 0, 0, 0.04))',
        }}
      />
      <figcaption
        style={{
          fontSize: '0.875rem',
          lineHeight: 1.5,
          color: 'var(--color-text-muted, #666)',
          marginTop: '0.75rem',
          textAlign: 'left',
          maxWidth: `${maxWidth}px`,
          padding: '0 0.5rem',
        }}
      >
        {caption}
      </figcaption>
    </div>
  );
}
