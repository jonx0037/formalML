// =============================================================================
// BNNInteractiveFigure.tsx
//
// Shared substrate for v2 interactive BNN viz components. Handles the boilerplate
// every interactive viz needs:
//   - useResizeObserver for width-aware layout
//   - JSON fetch from /sample-data/bayesian-neural-networks/<file>.json
//   - loading / error / ready state machine
//   - graceful degradation: inline static-figure fallback before payload arrives
//     and on fetch failure (we render the same <img>+<figcaption> directly here
//     rather than nesting BNNStaticFigure, to avoid invalid nested <figure>
//     elements)
//   - container <figure> always mounted (per CLAUDE.md "Loading-state JSX rule")
//
// Per-section components specialize this helper by passing a `dataPath` and
// a render function that consumes the parsed JSON plus current control state.
// =============================================================================

import { useEffect, useState, useRef, type ReactNode } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { type BNNStaticFigureProps } from './BNNStaticFigure';

export interface BNNInteractiveFigureProps<TData, TState>
  extends Omit<BNNStaticFigureProps, 'maxWidth'> {
  /** Path to JSON payload, e.g., '/sample-data/bayesian-neural-networks/calibration.json'. */
  dataPath: string;
  /** Initial control-state seed. */
  initialState: TState;
  /** Render the controls (top section above the chart). */
  controls: (state: TState, setState: (s: TState) => void) => ReactNode;
  /** Render the interactive chart given parsed data, current state, width, and setState (so clicks inside the chart can drive state). */
  chart: (
    data: TData,
    state: TState,
    width: number,
    setState: (s: TState) => void,
  ) => ReactNode;
  /** Optional max-width override (default 720px). */
  maxWidth?: number;
}

type FetchState<TData> =
  | { kind: 'loading' }
  | { kind: 'error'; message: string }
  | { kind: 'ready'; data: TData };

export default function BNNInteractiveFigure<TData, TState>({
  dataPath,
  initialState,
  controls,
  chart,
  figurePath,
  alt,
  caption,
  ariaLabel,
  maxWidth = 720,
}: BNNInteractiveFigureProps<TData, TState>) {
  const { ref, width } = useResizeObserver<HTMLElement>();
  const [fetchState, setFetchState] = useState<FetchState<TData>>({ kind: 'loading' });
  const [controlState, setControlState] = useState<TState>(initialState);
  const initialStateRef = useRef(initialState);

  useEffect(() => {
    let cancelled = false;
    fetch(dataPath)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: TData) => {
        if (!cancelled) setFetchState({ kind: 'ready', data });
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          const msg = e instanceof Error ? e.message : String(e);
          setFetchState({ kind: 'error', message: msg });
          // eslint-disable-next-line no-console
          console.error('[BNNInteractiveFigure] fetch failed:', msg);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [dataPath]);

  // Reset controls if initialState identity changes (rare).
  useEffect(() => {
    if (initialStateRef.current !== initialState) {
      initialStateRef.current = initialState;
      setControlState(initialState);
    }
  }, [initialState]);

  return (
    <figure
      ref={ref as React.RefObject<HTMLElement>}
      aria-label={ariaLabel}
      style={{
        width: '100%',
        margin: '2rem auto',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}
    >
      <div style={{ width: '100%', maxWidth: `${maxWidth}px` }}>
        {fetchState.kind === 'ready' ? (
          <>
            <div style={{ marginBottom: '0.75rem' }}>
              {controls(controlState, setControlState)}
            </div>
            {chart(fetchState.data, controlState, width || maxWidth, setControlState)}
          </>
        ) : (
          // Loading / error: render the static fallback img inline (same parent
          // <figure>) rather than nesting <BNNStaticFigure>, which would produce
          // invalid nested <figure> elements.
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
        )}
      </div>
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
    </figure>
  );
}
