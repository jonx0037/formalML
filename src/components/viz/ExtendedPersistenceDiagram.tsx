import { useResizeObserver } from './shared/useResizeObserver';

/**
 * ExtendedPersistenceDiagram — placeholder component.
 *
 * Full implementation will show ordinary + extended persistence of a
 * height function on a surface, with ordinary pairs above the diagonal,
 * extended pairs below, and relative pairs in a separate quadrant.
 */
export default function ExtendedPersistenceDiagram() {
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
          Extended Persistence Diagram
        </p>
        <p className="mt-2 text-sm" style={{ color: 'var(--color-muted-fg)' }}>
          Interactive visualization coming soon — explore ordinary, relative, and extended persistence pairs on a torus.
        </p>
      </div>
    </div>
  );
}
