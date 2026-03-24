/**
 * Shared geometry utilities for convex analysis visualizations.
 */

export interface Point {
  x: number;
  y: number;
}

export interface EllipseDef {
  type: 'ellipse';
  cx: number;
  cy: number;
  a: number;
  b: number;
}

export interface PolygonDef {
  type: 'polygon';
  vertices: Point[];
}

export interface RectangleDef {
  type: 'rectangle';
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

export type ConvexSetDef = EllipseDef | PolygonDef | RectangleDef;

/** Test whether a point lies inside a convex set. */
export function isInsideSet(point: Point, set: ConvexSetDef): boolean {
  switch (set.type) {
    case 'ellipse': {
      const dx = (point.x - set.cx) / set.a;
      const dy = (point.y - set.cy) / set.b;
      return dx * dx + dy * dy <= 1;
    }
    case 'rectangle':
      return (
        point.x >= set.xMin &&
        point.x <= set.xMax &&
        point.y >= set.yMin &&
        point.y <= set.yMax
      );
    case 'polygon':
      return isInsidePolygon(point, set.vertices);
  }
}

/** Ray-casting (even-odd rule) for polygon containment. */
function isInsidePolygon(point: Point, vertices: Point[]): boolean {
  const n = vertices.length;
  let inside = false;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const vi = vertices[i];
    const vj = vertices[j];
    if (
      vi.y > point.y !== vj.y > point.y &&
      point.x < ((vj.x - vi.x) * (point.y - vi.y)) / (vj.y - vi.y) + vi.x
    ) {
      inside = !inside;
    }
  }
  return inside;
}

/** Check whether the line segment from p1 to p2 stays entirely inside the set. */
export function segmentInsideSet(
  p1: Point,
  p2: Point,
  set: ConvexSetDef,
  samples = 100,
): boolean {
  for (let i = 0; i <= samples; i++) {
    const t = i / samples;
    const pt = { x: p1.x + t * (p2.x - p1.x), y: p1.y + t * (p2.y - p1.y) };
    if (!isInsideSet(pt, set)) return false;
  }
  return true;
}

/**
 * Return arrays of [t, inside] for the segment from p1 to p2,
 * useful for coloring segments green (inside) vs red (outside).
 */
export function segmentMembership(
  p1: Point,
  p2: Point,
  set: ConvexSetDef,
  samples = 200,
): { t: number; inside: boolean }[] {
  const result: { t: number; inside: boolean }[] = [];
  for (let i = 0; i <= samples; i++) {
    const t = i / samples;
    const pt = { x: p1.x + t * (p2.x - p1.x), y: p1.y + t * (p2.y - p1.y) };
    result.push({ t, inside: isInsideSet(pt, set) });
  }
  return result;
}

/** Compute the closest point on the boundary of a convex set to a given point. */
export function closestBoundaryPoint(point: Point, set: ConvexSetDef): Point {
  switch (set.type) {
    case 'ellipse': {
      const angle = Math.atan2(
        (point.y - set.cy) / set.b,
        (point.x - set.cx) / set.a,
      );
      return {
        x: set.cx + set.a * Math.cos(angle),
        y: set.cy + set.b * Math.sin(angle),
      };
    }
    case 'rectangle': {
      // Sample boundary edges and find closest
      const edges = getRectangleEdges(set);
      return closestOnEdges(point, edges);
    }
    case 'polygon':
      return closestOnEdges(point, getPolygonEdges(set.vertices));
  }
}

function getRectangleEdges(r: RectangleDef): [Point, Point][] {
  const tl = { x: r.xMin, y: r.yMax };
  const tr = { x: r.xMax, y: r.yMax };
  const br = { x: r.xMax, y: r.yMin };
  const bl = { x: r.xMin, y: r.yMin };
  return [
    [tl, tr],
    [tr, br],
    [br, bl],
    [bl, tl],
  ];
}

function getPolygonEdges(vertices: Point[]): [Point, Point][] {
  const edges: [Point, Point][] = [];
  for (let i = 0; i < vertices.length; i++) {
    edges.push([vertices[i], vertices[(i + 1) % vertices.length]]);
  }
  return edges;
}

function closestOnEdges(point: Point, edges: [Point, Point][]): Point {
  let best: Point = edges[0][0];
  let bestDist = Infinity;
  for (const [a, b] of edges) {
    const cp = closestPointOnSegment(point, a, b);
    const d = dist(point, cp);
    if (d < bestDist) {
      bestDist = d;
      best = cp;
    }
  }
  return best;
}

function closestPointOnSegment(p: Point, a: Point, b: Point): Point {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len2 = dx * dx + dy * dy;
  if (len2 === 0) return a;
  let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2;
  t = Math.max(0, Math.min(1, t));
  return { x: a.x + t * dx, y: a.y + t * dy };
}

/** Euclidean distance between two points. */
export function dist(a: Point, b: Point): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/** Outward normal of an ellipse at parameter t (normalized). */
export function ellipseNormal(
  a: number,
  b: number,
  t: number,
): Point {
  const nx = Math.cos(t) / a;
  const ny = Math.sin(t) / b;
  const len = Math.sqrt(nx * nx + ny * ny);
  return { x: nx / len, y: ny / len };
}

/** Compute the convex hull of a set of 2D points (Graham scan). */
export function computeConvexHull(points: Point[]): Point[] {
  if (points.length < 3) return [...points];

  const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);

  const cross = (o: Point, a: Point, b: Point) =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  // Lower hull
  const lower: Point[] = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }

  // Upper hull
  const upper: Point[] = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }

  // Remove last point of each half because it's repeated
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

/** Generate boundary points for an ellipse. */
export function ellipseBoundary(
  cx: number,
  cy: number,
  a: number,
  b: number,
  n = 100,
): Point[] {
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const t = (2 * Math.PI * i) / n;
    pts.push({ x: cx + a * Math.cos(t), y: cy + b * Math.sin(t) });
  }
  return pts;
}

/** Clamp a point to stay within bounds. */
export function clampPoint(
  p: Point,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
): Point {
  return {
    x: Math.max(xMin, Math.min(xMax, p.x)),
    y: Math.max(yMin, Math.min(yMax, p.y)),
  };
}
