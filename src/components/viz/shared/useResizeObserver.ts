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
    return () => observer.disconnect();
  }, []);

  return { ref, ...dimensions };
}
