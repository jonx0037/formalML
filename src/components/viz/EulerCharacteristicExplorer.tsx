import { useResizeObserver } from './shared/useResizeObserver';

/**
 * EulerCharacteristicExplorer — placeholder component.
 *
 * Full implementation will let users build a simplicial complex by adding/removing
 * simplices, with real-time display of V, E, F counts and χ = V − E + F.
 */
export default function EulerCharacteristicExplorer() {
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
          Euler Characteristic Explorer
        </p>
        <p className="mt-2 text-sm" style={{ color: 'var(--color-muted-fg)' }}>
          Interactive visualization coming soon — build a simplicial complex and watch χ = V − E + F update in real time.
        </p>
      </div>
    </div>
  );
}
