import { useRef, useEffect } from 'react';
import * as d3 from 'd3';

export function useD3<T extends SVGSVGElement>(
  renderFn: (svg: d3.Selection<T, unknown, null, undefined>) => void,
  deps: React.DependencyList,
) {
  const ref = useRef<T>(null);

  useEffect(() => {
    if (ref.current) {
      renderFn(d3.select(ref.current));
    }
  }, deps);

  return ref;
}
