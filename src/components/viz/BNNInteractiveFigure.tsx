// =============================================================================
// BNNInteractiveFigure.tsx
//
// Shared substrate for v2 interactive BNN viz components. Handles the boilerplate
// every interactive viz needs:
//   - useResizeObserver for width-aware layout
//   - JSON fetch from /sample-data/bayesian-neural-networks/<file>.json
//   - loading / error / ready state machine
//   - graceful degradation: static-figure fallback before payload arrives
//     and on fetch failure
//   - container <div> always mounted (per CLAUDE.md "Loading-state JSX rule")
//
// Per-section components specialize this helper by passing a `dataPath` and
// a render function that consumes the parsed JSON plus current control state.
// =============================================================================

import { useEffect, useState, useRef, type ReactNode } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import BNNStaticFigure, { type BNNStaticFigureProps } from './BNNStaticFigure';

export interface BNNInteractiveFigureProps<TData, TState>
  extends Omit<BNNStaticFigureProps, 'maxWidth'> {
  /** Path to JSON payload, e.g., '/sample-data/bayesian-neural-networks/calibration.json'. */
  dataPath: string;
  /** Initial control-state seed. */
  initialState: TState;
  /** Render the controls (top section above the chart). */
  controls: (state: TState, setState: (s: TState) => void) => ReactNode;
  /** Render the interactive chart given parsed data, current state, and width. */
  chart: (data: TData, state: TState, width: number) => ReactNode;
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
  const { ref, width } = useResizeObserver<HTMLDivElement>();
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
      <div style={{ width: '100%', maxWidth: `${maxWidth}px` }}>
        {fetchState.kind === 'ready' ? (
          <>
            <div style={{ marginBottom: '0.75rem' }}>
              {controls(controlState, setControlState)}
            </div>
            {chart(fetchState.data, controlState, width || maxWidth)}
          </>
        ) : (
          // Loading / error: degrade to static figure so the reader still gets
          // the pedagogical content. The error message is logged to console.
          <BNNStaticFigure
            figurePath={figurePath}
            alt={alt}
            caption={caption}
            ariaLabel={ariaLabel}
            maxWidth={maxWidth}
          />
        )}
      </div>
      {fetchState.kind === 'ready' && (
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
      )}
    </div>
  );
}
