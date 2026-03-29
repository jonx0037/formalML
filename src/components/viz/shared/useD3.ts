import { useRef, useEffect } from 'react';
import * as d3 from 'd3';

export function useD3<T extends SVGSVGElement>(
  renderFn: (svg: d3.Selection<T, unknown, null, undefined>) => void,
  deps: React.DependencyList,
) {
  const ref = useRef<T>(null);

  useEffect(() => {
    if (ref.current) {
      try {
        renderFn(d3.select(ref.current));
      } catch (e) {
        const err = e instanceof Error ? e : new Error(String(e));
        console.error('[useD3] Render failed:', err);
        // Clear the SVG so a half-rendered state isn't visible
        d3.select(ref.current).selectAll('*').remove();
      }
    }
  }, deps);

  return ref;
}
