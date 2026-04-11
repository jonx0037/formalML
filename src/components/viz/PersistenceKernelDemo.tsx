import { useResizeObserver } from './shared/useResizeObserver';

/**
 * PersistenceKernelDemo — placeholder component.
 *
 * Full implementation will demonstrate the persistence scale-space kernel
 * for SVM classification: show two classes of persistence diagrams,
 * the kernel matrix, and the SVM decision boundary in a 2D embedding.
 */
export default function PersistenceKernelDemo() {
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
          Persistence Kernel Demo
        </p>
        <p className="mt-2 text-sm" style={{ color: 'var(--color-muted-fg)' }}>
          Interactive visualization coming soon — compare persistence diagrams using the scale-space kernel for SVM classification.
        </p>
      </div>
    </div>
  );
}
