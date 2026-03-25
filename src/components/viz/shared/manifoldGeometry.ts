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

// ─── Riemannian Geometry Utilities ───
// Shared by MetricTensorExplorer, ParallelTransportExplorer,
// ConnectionExplorer, NaturalGradientExplorer

export interface MetricTensor {
  g: [[number, number], [number, number]];
  det: number;
  inv: [[number, number], [number, number]];
}

export interface ChristoffelSymbols {
  /** gamma[k][i][j] = Γ^k_{ij} */
  gamma: number[][][];
}

/** Round metric on S² at colatitude θ: g = dθ² + sin²θ dφ² */
export function sphereMetric(theta: number): MetricTensor {
  const sinTh = Math.sin(theta);
  const sin2 = sinTh * sinTh;
  const det = sin2; // det(diag(1, sin²θ)) = sin²θ
  const invSin2 = sin2 > 1e-12 ? 1 / sin2 : Infinity;
  return {
    g: [[1, 0], [0, sin2]],
    det,
    inv: [[1, 0], [0, invSin2]],
  };
}

/** Christoffel symbols for S² at colatitude θ.
 *  Nonzero: Γ^θ_{φφ} = −sinθ cosθ,  Γ^φ_{θφ} = Γ^φ_{φθ} = cotθ */
export function sphereChristoffel(theta: number): ChristoffelSymbols {
  const sinTh = Math.sin(theta);
  const cosTh = Math.cos(theta);
  const cotTh = Math.abs(sinTh) > 1e-12 ? cosTh / sinTh : 0;

  // gamma[k][i][j] for k,i,j ∈ {0=θ, 1=φ}
  const gamma: number[][][] = [
    // k = 0 (θ)
    [[0, 0], [0, -sinTh * cosTh]],
    // k = 1 (φ)
    [[0, cotTh], [cotTh, 0]],
  ];
  return { gamma };
}

/** Poincaré disk conformal factor λ = 2/(1 − |x|²) at point (x, y) */
export function poincareConformalFactor(x: number, y: number): number {
  const r2 = x * x + y * y;
  const denom = 1 - r2;
  if (denom < 1e-8) return Infinity; // λ → ∞ at the boundary
  return 2 / denom;
}

/** Poincaré disk metric g = λ² I  where λ = 2/(1−|x|²) */
export function poincareMetric(x: number, y: number): MetricTensor {
  const lambda = poincareConformalFactor(x, y);
  const lambda2 = lambda * lambda;
  return {
    g: [[lambda2, 0], [0, lambda2]],
    det: lambda2 * lambda2,
    inv: [[1 / lambda2, 0], [0, 1 / lambda2]],
  };
}

/** Eigendecomposition of a 2×2 symmetric matrix.
 *  Returns eigenvalues (ascending) and orthonormal eigenvectors. */
export function metricEigendecomp(
  g: [[number, number], [number, number]]
): { eigenvalues: [number, number]; eigenvectors: [Vec2, Vec2] } {
  const a = g[0][0];
  const b = g[0][1]; // = g[1][0] by symmetry
  const d = g[1][1];

  const trace = a + d;
  const det = a * d - b * b;
  const disc = Math.sqrt(Math.max(0, trace * trace - 4 * det));

  const lam1 = (trace - disc) / 2;
  const lam2 = (trace + disc) / 2;

  let v1: Vec2;
  let v2: Vec2;
  if (Math.abs(b) > 1e-12) {
    v1 = vec2Normalize({ x: lam1 - d, y: b });
    v2 = vec2Normalize({ x: lam2 - d, y: b });
  } else {
    v1 = { x: 1, y: 0 };
    v2 = { x: 0, y: 1 };
    // swap if a > d so eigenvalues match
    if (a > d) {
      return { eigenvalues: [d, a], eigenvectors: [v2, v1] };
    }
  }

  return { eigenvalues: [lam1, lam2], eigenvectors: [v1, v2] };
}

/** Normalize a Vec2 to unit length */
export function vec2Normalize(v: Vec2): Vec2 {
  const n = vec2Norm(v);
  return n < 1e-12 ? { x: 1, y: 0 } : { x: v.x / n, y: v.y / n };
}

/** Solve parallel transport ODE on S² along a parametric curve.
 *  ODE: dV^k/dt + Γ^k_{ij} γ̇^i V^j = 0 (Forward Euler) */
export function parallelTransportS2(
  curve: (t: number) => [number, number],
  curveDot: (t: number) => [number, number],
  V0: [number, number],
  nSteps = 300
): { t: number; pos: [number, number]; V: [number, number]; normG: number }[] {
  const result: { t: number; pos: [number, number]; V: [number, number]; normG: number }[] = [];
  const dt = 1 / nSteps;
  let V: [number, number] = [V0[0], V0[1]];

  for (let step = 0; step <= nSteps; step++) {
    const t = step * dt;
    const pos = curve(t);
    const [theta] = pos;
    const chris = sphereChristoffel(theta);
    const gam = curveDot(t);

    // Compute metric norm |V|_g = sqrt(g_{ij} V^i V^j)
    const metric = sphereMetric(theta);
    const normG = Math.sqrt(
      metric.g[0][0] * V[0] * V[0] +
      2 * metric.g[0][1] * V[0] * V[1] +
      metric.g[1][1] * V[1] * V[1]
    );

    result.push({ t, pos: [pos[0], pos[1]], V: [V[0], V[1]], normG });

    if (step < nSteps) {
      // dV^k/dt = −Γ^k_{ij} γ̇^i V^j
      const dV0 = -(
        chris.gamma[0][0][0] * gam[0] * V[0] +
        chris.gamma[0][0][1] * gam[0] * V[1] +
        chris.gamma[0][1][0] * gam[1] * V[0] +
        chris.gamma[0][1][1] * gam[1] * V[1]
      );
      const dV1 = -(
        chris.gamma[1][0][0] * gam[0] * V[0] +
        chris.gamma[1][0][1] * gam[0] * V[1] +
        chris.gamma[1][1][0] * gam[1] * V[0] +
        chris.gamma[1][1][1] * gam[1] * V[1]
      );
      V = [V[0] + dV0 * dt, V[1] + dV1 * dt];
    }
  }
  return result;
}

/** Fisher information metric for N(μ, σ²): g = diag(1/σ², 2/σ²) */
export function fisherMetricGaussian(sigma: number): MetricTensor {
  const s2 = sigma * sigma;
  const g11 = 1 / s2;
  const g22 = 2 / s2;
  return {
    g: [[g11, 0], [0, g22]],
    det: g11 * g22,
    inv: [[s2, 0], [0, s2 / 2]],
  };
}

/** KL divergence KL(N(μ,σ²) ‖ N(μ₀,σ₀²)) */
export function klDivGaussian(
  mu: number, sigma: number,
  mu0: number, sigma0: number
): number {
  const s2 = sigma * sigma;
  const s02 = sigma0 * sigma0;
  return Math.log(sigma0 / sigma) + (s2 + (mu - mu0) ** 2) / (2 * s02) - 0.5;
}

/** Partial derivatives of KL(N(μ,σ²) ‖ N(μ₀,σ₀²)) w.r.t. (μ, σ) */
export function klDivGradient(
  mu: number, sigma: number,
  mu0: number, sigma0: number
): [number, number] {
  const s02 = sigma0 * sigma0;
  const dMu = (mu - mu0) / s02;
  const dSigma = -1 / sigma + sigma / s02;
  return [dMu, dSigma];
}

/** Ellipsoid metric: g = diag(a² sin²θ + b² cos²θ, a² sin²θ) for an oblate ellipsoid
 *  with semi-axes a (equatorial) and b (polar). g_θθ is the meridional component,
 *  g_φφ is the azimuthal component. */
export function ellipsoidMetric(theta: number, a = 1, b = 0.6): MetricTensor {
  const sinTh = Math.sin(theta);
  const cosTh = Math.cos(theta);
  const sin2 = sinTh * sinTh;
  const cos2 = cosTh * cosTh;
  const gThTh = a * a * sin2 + b * b * cos2;
  const gPhPh = a * a * sin2;
  const det = gThTh * gPhPh;
  const invThTh = gThTh > 1e-12 ? 1 / gThTh : Infinity;
  const invPhPh = gPhPh > 1e-12 ? 1 / gPhPh : Infinity;
  return {
    g: [[gThTh, 0], [0, gPhPh]],
    det,
    inv: [[invThTh, 0], [0, invPhPh]],
  };
}

/** Point on an oblate ellipsoid from spherical coordinates */
export function ellipsoidPoint(theta: number, phi: number, a = 1, b = 0.6): Vec3 {
  return {
    x: a * Math.sin(theta) * Math.cos(phi),
    y: a * Math.sin(theta) * Math.sin(phi),
    z: b * Math.cos(theta),
  };
}

// ─── Geodesic & Curvature Utilities ───
// Shared by GeodesicExplorer, CurvatureExplorer, GaussBonnetExplorer, JacobiFieldExplorer

export interface GeodesicPoint {
  t: number;
  theta: number;
  phi: number;
  dtheta: number;
  dphi: number;
}

export interface GeodesicPointXY {
  t: number;
  x: number;
  y: number;
  dx: number;
  dy: number;
}

export interface JacobiFieldResult {
  t: number;
  magnitude: number;
}

/** RK4 geodesic solver on the unit sphere S².
 *  Solves: d²θ/dt² − sinθ cosθ (dφ/dt)² = 0
 *          d²φ/dt² + 2 cotθ (dθ/dt)(dφ/dt) = 0
 */
export function solveGeodesicS2(
  theta0: number,
  phi0: number,
  dtheta0: number,
  dphi0: number,
  tMax: number,
  nSteps = 300
): GeodesicPoint[] {
  const dt = tMax / nSteps;
  const result: GeodesicPoint[] = [];

  let state = [theta0, phi0, dtheta0, dphi0];

  function deriv(s: number[]): number[] {
    const [th, _ph, dth, dph] = s;
    const sinTh = Math.sin(th);
    const cosTh = Math.cos(th);
    const cotTh = Math.abs(sinTh) > 1e-12 ? cosTh / sinTh : 0;
    return [
      dth,
      dph,
      sinTh * cosTh * dph * dph,       // −Γ^θ_{φφ} * dφ² = sinθ cosθ dφ²
      -2 * cotTh * dth * dph,           // −2 Γ^φ_{θφ} * dθ dφ = −2 cotθ dθ dφ
    ];
  }

  for (let i = 0; i <= nSteps; i++) {
    const [th, ph, dth, dph] = state;
    result.push({ t: i * dt, theta: th, phi: ph, dtheta: dth, dphi: dph });

    if (i < nSteps) {
      // RK4 step
      const k1 = deriv(state);
      const s2 = state.map((v, j) => v + 0.5 * dt * k1[j]);
      const k2 = deriv(s2);
      const s3 = state.map((v, j) => v + 0.5 * dt * k2[j]);
      const k3 = deriv(s3);
      const s4 = state.map((v, j) => v + dt * k3[j]);
      const k4 = deriv(s4);
      state = state.map((v, j) =>
        v + (dt / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])
      );
    }
  }

  return result;
}

/** Christoffel symbols for the Poincaré disk at (x, y).
 *  Conformal metric g = λ² I where λ = 2/(1 − r²).
 *  Γ^k_{ij} from the conformal formula. Indices: 0=x, 1=y. */
export function poincareChristoffel(x: number, y: number): ChristoffelSymbols {
  const r2 = x * x + y * y;
  const denom = 1 - r2;
  const factor = Math.abs(denom) > 1e-8 ? 2 / denom : 0;
  // ∂_x ln λ = 2x/(1−r²),  ∂_y ln λ = 2y/(1−r²)
  const dlnLx = factor * x;
  const dlnLy = factor * y;

  // For conformal metric g = e^{2f} δ (here f = ln λ):
  //   Γ^k_{ij} = δ^k_i ∂_j f + δ^k_j ∂_i f − δ_{ij} ∂^k f
  // where ∂^k f = δ^{kl} ∂_l f (since conformal metric's Christoffels
  // are computed from the flat inverse for raising)
  const gamma: number[][][] = [
    // k = 0 (x)
    [
      [dlnLx, dlnLy],      // Γ^x_{xx} = ∂_x f, Γ^x_{xy} = ∂_y f
      [dlnLy, -dlnLx],     // Γ^x_{yx} = ∂_y f, Γ^x_{yy} = −∂_x f
    ],
    // k = 1 (y)
    [
      [-dlnLy, dlnLx],     // Γ^y_{xx} = −∂_y f, Γ^y_{xy} = ∂_x f
      [dlnLx, dlnLy],      // Γ^y_{yx} = ∂_x f, Γ^y_{yy} = ∂_y f
    ],
  ];
  return { gamma };
}

/** RK4 geodesic solver on the Poincaré disk.
 *  Coordinates (x, y) with x²+y² < 1. */
export function solveGeodesicPoincare(
  x0: number,
  y0: number,
  dx0: number,
  dy0: number,
  tMax: number,
  nSteps = 300
): GeodesicPointXY[] {
  const dt = tMax / nSteps;
  const result: GeodesicPointXY[] = [];

  let state = [x0, y0, dx0, dy0];

  function deriv(s: number[]): number[] {
    const [px, py, vx, vy] = s;
    const chris = poincareChristoffel(px, py);
    const g = chris.gamma;
    // ddx^k = −Γ^k_{ij} dx^i dx^j
    const ddx = -(
      g[0][0][0] * vx * vx + g[0][0][1] * vx * vy +
      g[0][1][0] * vy * vx + g[0][1][1] * vy * vy
    );
    const ddy = -(
      g[1][0][0] * vx * vx + g[1][0][1] * vx * vy +
      g[1][1][0] * vy * vx + g[1][1][1] * vy * vy
    );
    return [vx, vy, ddx, ddy];
  }

  for (let i = 0; i <= nSteps; i++) {
    const [px, py, vx, vy] = state;
    result.push({ t: i * dt, x: px, y: py, dx: vx, dy: vy });

    // Stop if we leave the disk
    if (px * px + py * py > 0.98) break;

    if (i < nSteps) {
      const k1 = deriv(state);
      const s2 = state.map((v, j) => v + 0.5 * dt * k1[j]);
      const k2 = deriv(s2);
      const s3 = state.map((v, j) => v + 0.5 * dt * k2[j]);
      const k3 = deriv(s3);
      const s4 = state.map((v, j) => v + dt * k3[j]);
      const k4 = deriv(s4);
      state = state.map((v, j) =>
        v + (dt / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])
      );
    }
  }

  return result;
}

/** Gaussian curvature of a torus at parameter v.
 *  K = cos(v) / (r * (R + r*cos(v))) */
export function torusCurvature(v: number, R = 2, r = 0.8): number {
  return Math.cos(v) / (r * (R + r * Math.cos(v)));
}

/** Jacobi field magnitude for constant sectional curvature K.
 *  K > 0: sin(√K t)/√K   (oscillates, zero at t = π/√K)
 *  K = 0: t               (linear growth)
 *  K < 0: sinh(√|K| t)/√|K|  (exponential growth) */
export function jacobiFieldMagnitude(K: number, t: number): number {
  if (Math.abs(K) < 1e-12) return t;
  if (K > 0) {
    const sqK = Math.sqrt(K);
    return Math.sin(sqK * t) / sqK;
  }
  const sqK = Math.sqrt(-K);
  return Math.sinh(sqK * t) / sqK;
}

/** Numerically integrate total curvature ∫∫ K(u,v) √det(g) du dv
 *  over a parametric surface using the trapezoidal rule. */
export function totalCurvature(
  curvatureFn: (u: number, v: number) => number,
  areaElementFn: (u: number, v: number) => number,
  uRange: [number, number],
  vRange: [number, number],
  nU = 50,
  nV = 50
): number {
  const du = (uRange[1] - uRange[0]) / nU;
  const dv = (vRange[1] - vRange[0]) / nV;
  let total = 0;
  for (let i = 0; i < nU; i++) {
    const u = uRange[0] + (i + 0.5) * du;
    for (let j = 0; j < nV; j++) {
      const v = vRange[0] + (j + 0.5) * dv;
      total += curvatureFn(u, v) * areaElementFn(u, v) * du * dv;
    }
  }
  return total;
}

/** Area element √det(g) for the torus parametrized by (u, v). */
export function torusAreaElement(u: number, v: number, R = 2, r = 0.8): number {
  return r * (R + r * Math.cos(v));
}

/** Area element for a sphere of radius rad. */
export function sphereAreaElement(theta: number, _phi: number, rad = 1): number {
  return rad * rad * Math.sin(theta);
}

/** Area element for an ellipsoid with semi-axes (a, a, b). */
export function ellipsoidAreaElement(
  theta: number,
  _phi: number,
  a = 1,
  b = 1
): number {
  const sinTh = Math.sin(theta);
  const cosTh = Math.cos(theta);
  // For (a sinθ cosφ, a sinθ sinφ, b cosθ):
  // |∂_θ × ∂_φ| = sinθ √(a⁴ sin²θ cos²θ/... ) ≈ a sinθ √(a² sin²θ + b² cos²θ)
  return a * sinTh * Math.sqrt(a * a * sinTh * sinTh + b * b * cosTh * cosTh);
}

/** Gaussian curvature of an ellipsoid (a, a, b) at colatitude θ. */
export function ellipsoidCurvature(theta: number, a = 1, b = 1): number {
  const sinTh = Math.sin(theta);
  const cosTh = Math.cos(theta);
  const denom = a * a * sinTh * sinTh + b * b * cosTh * cosTh;
  // K = (a²b²) / denom²
  return (a * a * b * b) / (denom * denom);
}
