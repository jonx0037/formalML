// ─── Smooth Manifold Geometry Utilities ───
// Shared by ChartAtlasExplorer, TangentSpaceExplorer, SmoothMapExplorer, ManifoldGalleryExplorer

// Use local vector types (the existing Point2D has an `id` field for topology viz)
export interface Vec2 {
  x: number;
  y: number;
}

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

// ─── Vector Operations ───

export function vec3Add(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

export function vec3Sub(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

export function vec3Scale(a: Vec3, s: number): Vec3 {
  return { x: a.x * s, y: a.y * s, z: a.z * s };
}

export function vec3Dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

export function vec3Norm(a: Vec3): number {
  return Math.sqrt(vec3Dot(a, a));
}

export function vec3Normalize(a: Vec3): Vec3 {
  const n = vec3Norm(a);
  return n < 1e-12 ? { x: 0, y: 0, z: 0 } : vec3Scale(a, 1 / n);
}

export function vec3Cross(a: Vec3, b: Vec3): Vec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

export function vec2Add(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x + b.x, y: a.y + b.y };
}

export function vec2Scale(a: Vec2, s: number): Vec2 {
  return { x: a.x * s, y: a.y * s };
}

export function vec2Norm(a: Vec2): number {
  return Math.sqrt(a.x * a.x + a.y * a.y);
}

// ─── Stereographic Projection ───

/** Stereographic projection from the north pole: S^2 \ {N} -> R^2 */
export function stereoNorth(p: Vec3): Vec2 {
  const denom = 1 - p.z;
  if (Math.abs(denom) < 1e-10) return { x: 1e6, y: 1e6 };
  return { x: p.x / denom, y: p.y / denom };
}

/** Stereographic projection from the south pole: S^2 \ {S} -> R^2 */
export function stereoSouth(p: Vec3): Vec2 {
  const denom = 1 + p.z;
  if (Math.abs(denom) < 1e-10) return { x: 1e6, y: 1e6 };
  return { x: p.x / denom, y: p.y / denom };
}

/** Inverse stereographic projection (north pole chart) */
export function invStereoNorth(uv: Vec2): Vec3 {
  const d = uv.x * uv.x + uv.y * uv.y;
  return {
    x: (2 * uv.x) / (1 + d),
    y: (2 * uv.y) / (1 + d),
    z: (d - 1) / (1 + d),
  };
}

/** Inverse stereographic projection (south pole chart) */
export function invStereoSouth(uv: Vec2): Vec3 {
  const d = uv.x * uv.x + uv.y * uv.y;
  return {
    x: (2 * uv.x) / (1 + d),
    y: (2 * uv.y) / (1 + d),
    z: (1 - d) / (1 + d),
  };
}

/** Transition map: phi_S ∘ phi_N^{-1}(u, v) = (u, v) / (u^2 + v^2) */
export function transitionNS(uv: Vec2): Vec2 {
  const r2 = uv.x * uv.x + uv.y * uv.y;
  if (r2 < 1e-10) return { x: 1e6, y: 1e6 };
  return { x: uv.x / r2, y: uv.y / r2 };
}

// ─── Parametric Surfaces ───

/** Point on the unit sphere from spherical coordinates (theta = polar, phi = azimuthal) */
export function spherePoint(theta: number, phi: number): Vec3 {
  return {
    x: Math.sin(theta) * Math.cos(phi),
    y: Math.sin(theta) * Math.sin(phi),
    z: Math.cos(theta),
  };
}

/** Geographic to spherical: longitude [-pi,pi], latitude [-pi/2, pi/2] -> (theta, phi) */
export function geoToSpherical(lon: number, lat: number): { theta: number; phi: number } {
  return { theta: Math.PI / 2 - lat, phi: lon };
}

/** Point on a torus with major radius R and minor radius r */
export function torusPoint(u: number, v: number, R = 2, r = 0.8): Vec3 {
  return {
    x: (R + r * Math.cos(v)) * Math.cos(u),
    y: (R + r * Math.cos(v)) * Math.sin(u),
    z: r * Math.sin(v),
  };
}

/** Point on a paraboloid z = x^2 + y^2 parametrized by (u, v) in polar */
export function paraboloidPoint(u: number, v: number): Vec3 {
  const r = u; // radial
  return {
    x: r * Math.cos(v),
    y: r * Math.sin(v),
    z: r * r,
  };
}

/** Point on a Mobius band: u in [0, 2pi], v in [-1, 1] */
export function mobiusPoint(u: number, v: number, R = 1.5): Vec3 {
  const halfU = u / 2;
  return {
    x: (R + (v / 2) * Math.cos(halfU)) * Math.cos(u),
    y: (R + (v / 2) * Math.cos(halfU)) * Math.sin(u),
    z: (v / 2) * Math.sin(halfU),
  };
}

// ─── Orthographic Projection ───

/** Project a 3D point to 2D via orthographic projection with rotation */
export function orthoProject(p: Vec3, rotY: number, rotX: number): Vec2 {
  // Rotate around Y axis
  const cosY = Math.cos(rotY);
  const sinY = Math.sin(rotY);
  const x1 = p.x * cosY + p.z * sinY;
  const y1 = p.y;
  const z1 = -p.x * sinY + p.z * cosY;

  // Rotate around X axis
  const cosX = Math.cos(rotX);
  const sinX = Math.sin(rotX);
  const x2 = x1;
  const y2 = y1 * cosX - z1 * sinX;
  // z2 used for depth but not needed for projection

  return { x: x2, y: y2 };
}

/** Check if a 3D point is on the front side of the orthographic projection */
export function isVisible(p: Vec3, rotY: number, rotX: number): boolean {
  const cosY = Math.cos(rotY);
  const sinY = Math.sin(rotY);
  const z1 = -p.x * sinY + p.z * cosY;

  const cosX = Math.cos(rotX);
  const sinX = Math.sin(rotX);
  const z2 = p.y * sinX + z1 * cosX;

  return z2 >= 0;
}

// ─── Tangent Computation ───

/** Compute tangent vectors on a parametric surface at (u, v) via central differences */
export function surfaceTangents(
  surface: (u: number, v: number) => Vec3,
  u: number,
  v: number,
  h = 1e-5
): { du: Vec3; dv: Vec3 } {
  const pU1 = surface(u + h, v);
  const pU0 = surface(u - h, v);
  const pV1 = surface(u, v + h);
  const pV0 = surface(u, v - h);

  return {
    du: {
      x: (pU1.x - pU0.x) / (2 * h),
      y: (pU1.y - pU0.y) / (2 * h),
      z: (pU1.z - pU0.z) / (2 * h),
    },
    dv: {
      x: (pV1.x - pV0.x) / (2 * h),
      y: (pV1.y - pV0.y) / (2 * h),
      z: (pV1.z - pV0.z) / (2 * h),
    },
  };
}

/** Compute numerical Jacobian of f: R^m -> R^k at x */
export function numericalJacobian(
  f: (x: number[]) => number[],
  x: number[],
  h = 1e-6
): number[][] {
  const fx = f(x);
  const m = x.length;
  const k = fx.length;
  const J: number[][] = Array.from({ length: k }, () => new Array(m).fill(0));

  for (let j = 0; j < m; j++) {
    const xPlus = [...x];
    const xMinus = [...x];
    xPlus[j] += h;
    xMinus[j] -= h;
    const fPlus = f(xPlus);
    const fMinus = f(xMinus);
    for (let i = 0; i < k; i++) {
      J[i][j] = (fPlus[i] - fMinus[i]) / (2 * h);
    }
  }

  return J;
}

// ─── Wireframe Generation ───

/** Generate wireframe lines for a parametric surface */
export function generateWireframe(
  surface: (u: number, v: number) => Vec3,
  uRange: [number, number],
  vRange: [number, number],
  uSteps: number,
  vSteps: number,
  rotY: number,
  rotX: number
): { lines: Vec2[][]; backLines: Vec2[][] } {
  const lines: Vec2[][] = [];
  const backLines: Vec2[][] = [];

  // Constant-u lines
  for (let i = 0; i <= uSteps; i++) {
    const u = uRange[0] + (i / uSteps) * (uRange[1] - uRange[0]);
    const frontPts: Vec2[] = [];
    const backPts: Vec2[] = [];
    for (let j = 0; j <= vSteps * 2; j++) {
      const v = vRange[0] + (j / (vSteps * 2)) * (vRange[1] - vRange[0]);
      const p = surface(u, v);
      const proj = orthoProject(p, rotY, rotX);
      if (isVisible(p, rotY, rotX)) {
        if (backPts.length > 1) backLines.push([...backPts]);
        backPts.length = 0;
        frontPts.push(proj);
      } else {
        if (frontPts.length > 1) lines.push([...frontPts]);
        frontPts.length = 0;
        backPts.push(proj);
      }
    }
    if (frontPts.length > 1) lines.push(frontPts);
    if (backPts.length > 1) backLines.push(backPts);
  }

  // Constant-v lines
  for (let j = 0; j <= vSteps; j++) {
    const v = vRange[0] + (j / vSteps) * (vRange[1] - vRange[0]);
    const frontPts: Vec2[] = [];
    const backPts: Vec2[] = [];
    for (let i = 0; i <= uSteps * 2; i++) {
      const u = uRange[0] + (i / (uSteps * 2)) * (uRange[1] - uRange[0]);
      const p = surface(u, v);
      const proj = orthoProject(p, rotY, rotX);
      if (isVisible(p, rotY, rotX)) {
        if (backPts.length > 1) backLines.push([...backPts]);
        backPts.length = 0;
        frontPts.push(proj);
      } else {
        if (frontPts.length > 1) lines.push([...frontPts]);
        frontPts.length = 0;
        backPts.push(proj);
      }
    }
    if (frontPts.length > 1) lines.push(frontPts);
    if (backPts.length > 1) backLines.push(backPts);
  }

  return { lines, backLines };
}

/** Generate wireframe for the unit sphere using lat/lon grid */
export function sphereWireframe(
  nLat: number,
  nLon: number,
  rotY: number,
  rotX: number
): { lines: Vec2[][]; backLines: Vec2[][] } {
  return generateWireframe(
    (u, v) => spherePoint(u, v),
    [0.05, Math.PI - 0.05], // theta: avoid exact poles for cleaner lines
    [0, 2 * Math.PI],
    nLat,
    nLon,
    rotY,
    rotX
  );
}
