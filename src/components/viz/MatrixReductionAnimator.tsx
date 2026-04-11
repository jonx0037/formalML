import { useResizeObserver } from './shared/useResizeObserver';

/**
 * MatrixReductionAnimator — placeholder component.
 *
 * Full implementation will animate the column reduction of a filtered
 * boundary matrix step by step, highlighting pivot elements, column
 * additions (XOR over Z₂), and the resulting birth-death pairings.
 */
export default function MatrixReductionAnimator() {
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
          Matrix Reduction Animator
        </p>
        <p className="mt-2 text-sm" style={{ color: 'var(--color-muted-fg)' }}>
          Interactive visualization coming soon — step through the column reduction algorithm and watch birth-death pairs emerge.
        </p>
      </div>
    </div>
  );
}
