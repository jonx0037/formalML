import { useRef, useState, useEffect } from 'react';

export function useResizeObserver<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });

    observer.observe(el);

    // ResizeObserver may not fire its initial callback reliably for
    // Astro client:visible islands that hydrate at their final size.
    // Fall back to reading dimensions after the next animation frame.
    requestAnimationFrame(() => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      if (w > 0 || h > 0) {
        setDimensions((prev) => (prev.width === 0 && prev.height === 0 ? { width: w, height: h } : prev));
      }
    });

    return () => observer.disconnect();
  }, []);

  return { ref, ...dimensions };
}
