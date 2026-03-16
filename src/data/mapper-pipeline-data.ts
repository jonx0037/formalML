/**
 * Pre-computed point cloud for the Mapper pipeline visualization.
 *
 * 80 points sampled from a noisy unit circle. The filter function is the
 * x-coordinate projection, which partitions the circle into overlapping
 * vertical strips — the canonical first example for Mapper.
 *
 * Noise is deterministic via a simple hash:
 *   noise(i) = ((i * 7919 + 104729) % 100000) / 100000 * 0.1 - 0.05
 */
import type { MapperPoint } from '../components/viz/shared/types';

/** Deterministic pseudo-random noise in [-0.05, 0.05]. */
function noise(i: number): number {
  return ((i * 7919 + 104729) % 100000) / 100000 * 0.1 - 0.05;
}

function generateCirclePoints(n: number): MapperPoint[] {
  const points: MapperPoint[] = [];
  for (let i = 0; i < n; i++) {
    const theta = (2 * Math.PI * i) / n;
    const nx = noise(i);
    const ny = noise(i + n);
    const x = Math.cos(theta) + nx;
    const y = Math.sin(theta) + ny;
    points.push({
      id: i,
      x: Math.round(x * 10000) / 10000,
      y: Math.round(y * 10000) / 10000,
      filterValue: Math.round(x * 10000) / 10000,
    });
  }
  return points;
}

/** 80 points on a noisy unit circle with filterValue = x-coordinate. */
export const circlePoints: MapperPoint[] = generateCirclePoints(80);
