import { useResizeObserver } from './shared/useResizeObserver';

/**
 * BettiNumberTracker — placeholder component.
 *
 * Full implementation will display β₀ and β₁ of a Vietoris-Rips complex
 * updating in real time as the user drags an ε slider, showing how
 * connected components merge and loops appear/disappear.
 */
export default function BettiNumberTracker() {
  const { ref, width } = useResizeObserver();
  const height = Math.min(width * 0.6, 400);

  return (
    <div ref={ref} className="my-8">
      <div
        className="flex flex-col items-center justify-center rounded-lg border border-dashed"
        style={{
          height: `${height}px`,
          borderColor: 'var(--color-muted-border)',
          backgroundColor: 'var(--color-muted-bg)',
        }}
      >
        <p className="text-lg font-semibold" style={{ color: 'var(--color-fg)' }}>
          Betti Number Tracker
        </p>
        <p className="mt-2 text-sm" style={{ color: 'var(--color-muted-fg)' }}>
          Interactive visualization coming soon — drag ε and watch β₀, β₁ update as the complex evolves.
        </p>
      </div>
    </div>
  );
}
