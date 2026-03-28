import { useState, useRef, useEffect, useId, useCallback, useMemo } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';

const SM_BREAKPOINT = 640;
const COLORS = {
  input: '#3b82f6', output: '#8b5cf6', groupAction: '#f59e0b', network: '#10b981',
  match: '#22c55e', mismatch: '#ef4444', gridOn: '#3b82f6', gridOff: '#e2e8f0',
  node: '#06b6d4', edge: '#94a3b8', bg: '#f8fafc', text: '#1e293b',
};

type Arch = 'cnn' | 'gnn';
const CNN_INPUT = [[0,0,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,1,1,0,0],[0,0,0,0,0]];
const GNN_ADJ = [[0,1,0,0],[1,0,1,1],[0,1,0,0],[0,1,0,0]];
const GNN_FEAT = [[1,0],[0,1],[1,1],[0,0]];
const PERMS = [
  { label: 'Identity', p: [0,1,2,3] }, { label: 'Swap 0↔1', p: [1,0,2,3] },
  { label: 'Swap 0↔2', p: [2,1,0,3] }, { label: 'Swap 1↔3', p: [0,3,2,1] },
];
const NODE_POS = [{ x: 0.5, y: 0.1 }, { x: 0.9, y: 0.5 }, { x: 0.5, y: 0.9 }, { x: 0.1, y: 0.5 }];

// ─── Math ───

function shiftGrid(g: number[][], amt: number) {
  const n = g[0].length;
  return g.map(r => r.map((_, j) => r[((j - amt % n) + n) % n]));
}
function convolve(g: number[][]) {
  const n = g.length, out: number[][] = [];
  for (let i = 0; i <= n - 3; i++) {
    const row: number[] = [];
    for (let j = 0; j <= n - 3; j++) {
      let s = 0;
      for (let di = 0; di < 3; di++) for (let dj = 0; dj < 3; dj++) s += g[i + di][j + dj];
      row.push(s);
    }
    out.push(row);
  }
  return out;
}
function applyPerm(p: number[], M: number[][]) {
  return p.map((_, i) => M[p[i]]);
}
function permAdj(p: number[], A: number[][]) {
  const n = A.length, R = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) R[i][j] = A[p[i]][p[j]];
  return R;
}
function msgPass(A: number[][], X: number[][]) {
  const n = A.length, d = X[0].length;
  const O = Array.from({ length: n }, () => Array(d).fill(0));
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++)
    if (A[i][j]) for (let k = 0; k < d; k++) O[i][k] += X[j][k];
  return O;
}
function eq(a: number[][], b: number[][]) {
  return a.length === b.length && a.every((r, i) => r.length === b[i].length && r.every((v, j) => v === b[i][j]));
}

// ─── Drawing ───

type G = d3.Selection<SVGGElement, unknown, null, undefined>;

function drawGrid(g: G, grid: number[][], cx: number, cy: number, cs: number, vals: boolean) {
  const rows = grid.length, cols = grid[0].length;
  const ox = cx - (cols * cs) / 2, oy = cy - (rows * cs) / 2;
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const v = grid[i][j], x = ox + j * cs, y = oy + i * cs;
    g.append('rect').attr('x', x).attr('y', y).attr('width', cs).attr('height', cs)
      .attr('fill', vals ? (v > 0 ? COLORS.output : COLORS.gridOff) : v ? COLORS.gridOn : COLORS.gridOff)
      .attr('stroke', '#cbd5e1').attr('stroke-width', 0.5).attr('rx', 1);
    if (vals && v > 0)
      g.append('text').attr('x', x + cs / 2).attr('y', y + cs / 2)
        .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
        .attr('fill', '#fff').attr('font-size', Math.max(8, cs * 0.45)).attr('font-weight', 600)
        .text(v.toFixed(0));
  }
}

function drawGraph(g: G, adj: number[][], feat: number[][], cx: number, cy: number, sz: number, labels: number[]) {
  const r = sz * 0.4, nr = Math.max(8, sz * 0.1);
  const pos = NODE_POS.map(p => ({ x: cx + (p.x - 0.5) * 2 * r, y: cy + (p.y - 0.5) * 2 * r }));
  for (let i = 0; i < adj.length; i++) for (let j = i + 1; j < adj.length; j++)
    if (adj[i][j]) g.append('line').attr('x1', pos[i].x).attr('y1', pos[i].y)
      .attr('x2', pos[j].x).attr('y2', pos[j].y).attr('stroke', COLORS.edge).attr('stroke-width', 1.5);
  for (let i = 0; i < adj.length; i++) {
    g.append('circle').attr('cx', pos[i].x).attr('cy', pos[i].y).attr('r', nr)
      .attr('fill', COLORS.node).attr('stroke', '#fff').attr('stroke-width', 1.5);
    g.append('text').attr('x', pos[i].x).attr('y', pos[i].y - 1)
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
      .attr('fill', '#fff').attr('font-size', Math.max(7, nr * 0.8)).attr('font-weight', 700)
      .text(labels[i]);
    g.append('text').attr('x', pos[i].x).attr('y', pos[i].y + nr + Math.max(8, sz * 0.06))
      .attr('text-anchor', 'middle').attr('fill', COLORS.text).attr('font-size', Math.max(7, sz * 0.055))
      .text(`[${feat[i][0]},${feat[i][1]}]`);
  }
}

function drawArrow(g: G, x1: number, y1: number, x2: number, y2: number, color: string, label: string, mid: string, hl: boolean) {
  let defs = g.select<SVGDefsElement>('defs');
  if (defs.empty()) defs = g.append('defs');
  defs.append('marker').attr('id', mid).attr('viewBox', '0 0 10 10').attr('refX', 8).attr('refY', 5)
    .attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto-start-reverse')
    .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z').attr('fill', color);
  g.append('line').attr('x1', x1).attr('y1', y1).attr('x2', x2).attr('y2', y2)
    .attr('stroke', color).attr('stroke-width', hl ? 3 : 1.5).attr('stroke-dasharray', hl ? 'none' : '4 3')
    .attr('marker-end', `url(#${mid})`).attr('opacity', hl ? 1 : 0.6);
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, hz = Math.abs(y2 - y1) < Math.abs(x2 - x1);
  g.append('text').attr('x', mx + (hz ? 0 : -10)).attr('y', my + (hz ? -6 : 3))
    .attr('text-anchor', 'middle').attr('fill', color).attr('font-size', 10).attr('font-weight', 600).text(label);
}

// ─── Component ───

export default function EquivarianceExplorer() {
  const uid = useId().replace(/:/g, '');
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const [arch, setArch] = useState<Arch>('cnn');
  const [shift, setShift] = useState(1);
  const [permIdx, setPermIdx] = useState(2);
  const [step, setStep] = useState(-1);
  const [match, setMatch] = useState<boolean | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isMobile = width > 0 && width < SM_BREAKPOINT;

  useEffect(() => {
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, []);

  const cnn = useMemo(() => {
    const inp = CNN_INPUT, sh = shiftGrid(inp, shift);
    const cI = convolve(inp), cS = convolve(sh);
    return { inp, sh, cI, cS, ok: eq(cS, shiftGrid(cI, shift)) };
  }, [shift]);

  const gnn = useMemo(() => {
    const p = PERMS[permIdx].p, pF = applyPerm(p, GNN_FEAT), pA = permAdj(p, GNN_ADJ);
    const mpO = msgPass(GNN_ADJ, GNN_FEAT), mpP = msgPass(pA, pF);
    return { pF, pA, mpO, mpP, ok: eq(mpP, applyPerm(p, mpO)), pLabels: p };
  }, [permIdx]);

  const handleVerify = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setMatch(null); setStep(0);
    let s = 0;
    intervalRef.current = setInterval(() => {
      if (++s > 3) { clearInterval(intervalRef.current!); intervalRef.current = null; setMatch(arch === 'cnn' ? cnn.ok : gnn.ok); setStep(-1); return; }
      setStep(s);
    }, 600);
  }, [arch, cnn.ok, gnn.ok]);

  useEffect(() => {
    if (!svgRef.current || width === 0) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const totalH = isMobile ? 480 : 340;
    svg.attr('width', width).attr('height', totalH).attr('viewBox', `0 0 ${width} ${totalH}`);
    const g = svg.append('g');
    const pw = isMobile ? width * 0.45 : width * 0.22, ph = isMobile ? 130 : 130;
    const gap = isMobile ? width * 0.06 : width * 0.04;
    const sx = (width - (2 * pw + gap)) / 2, r0 = 10, r1 = isMobile ? ph + 60 : ph + 50;
    const cs = arch === 'cnn' ? Math.min(pw / 6, ph / 6) : 0;
    const ccs = arch === 'cnn' ? Math.min(pw / 4, ph / 4) : 0;
    const gs = Math.min(pw, ph) * 0.85;

    const cx = (c: number) => sx + c * (pw + gap) + pw / 2;
    const ry = (r: number) => (r === 0 ? r0 : r1) + ph / 2;
    // Panels
    for (let c = 0; c < 2; c++) for (let r = 0; r < 2; r++)
      g.append('rect').attr('x', sx + c * (pw + gap)).attr('y', r === 0 ? r0 : r1)
        .attr('width', pw).attr('height', ph).attr('fill', COLORS.bg)
        .attr('stroke', '#e2e8f0').attr('stroke-width', 1).attr('rx', 6);
    // Labels
    const lbl = arch === 'cnn'
      ? ['x (input)', 'g·x (shifted)', 'f(x) (conv)', 'f(g·x) = g·f(x)']
      : ['x (graph)', 'g·x (permuted)', 'f(x) (msg pass)', 'f(g·x) = g·f(x)'];
    [[0,0],[1,0],[0,1],[1,1]].forEach(([c, r], i) =>
      g.append('text').attr('x', cx(c)).attr('y', (r === 0 ? r0 : r1) + 11)
        .attr('text-anchor', 'middle').attr('fill', COLORS.text)
        .attr('font-size', Math.max(9, Math.min(11, pw * 0.08))).attr('font-weight', 600).text(lbl[i]));
    // Content
    const dy = 6;
    if (arch === 'cnn') {
      drawGrid(g, cnn.inp, cx(0), ry(0) + dy, cs, false);
      drawGrid(g, cnn.sh, cx(1), ry(0) + dy, cs, false);
      drawGrid(g, cnn.cI, cx(0), ry(1) + dy, ccs, true);
      drawGrid(g, cnn.cS, cx(1), ry(1) + dy, ccs, true);
    } else {
      const ol = [0, 1, 2, 3];
      drawGraph(g, GNN_ADJ, GNN_FEAT, cx(0), ry(0) + dy, gs, ol);
      drawGraph(g, gnn.pA, gnn.pF, cx(1), ry(0) + dy, gs, gnn.pLabels);
      drawGraph(g, GNN_ADJ, gnn.mpO, cx(0), ry(1) + dy, gs, ol);
      drawGraph(g, gnn.pA, gnn.mpP, cx(1), ry(1) + dy, gs, gnn.pLabels);
    }
    // Arrows
    const ap = 6, sl = arch === 'cnn' ? 'shift' : 'permute', nl = arch === 'cnn' ? 'conv' : 'msg pass';
    const hx1 = sx + pw + ap, hx2 = sx + pw + gap - ap;
    drawArrow(g, hx1, r0 + ph / 2, hx2, r0 + ph / 2, COLORS.groupAction, sl, `${uid}-ht`, step === 2);
    drawArrow(g, hx1, r1 + ph / 2, hx2, r1 + ph / 2, COLORS.groupAction, sl, `${uid}-hb`, step === 3);
    drawArrow(g, cx(0), r0 + ph + ap, cx(0), r1 - ap, COLORS.network, nl, `${uid}-vl`, step === 1);
    drawArrow(g, cx(1), r0 + ph + ap, cx(1), r1 - ap, COLORS.network, nl, `${uid}-vr`, step === 3);
  }, [width, arch, cnn, gnn, shift, permIdx, step, isMobile, uid]);

  const sel: React.CSSProperties = { padding: '4px 8px', borderRadius: 6, border: '1px solid #cbd5e1', fontSize: 13, background: '#fff', color: COLORS.text };

  return (
    <div ref={containerRef} style={{ width: '100%', maxWidth: 720 }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', marginBottom: 10, fontSize: 13, color: COLORS.text }}>
        <label style={{ fontWeight: 600 }}>Architecture:</label>
        <select value={arch} onChange={e => { setArch(e.target.value as Arch); setMatch(null); setStep(-1); }} style={sel}>
          <option value="cnn">CNN (translation)</option><option value="gnn">GNN (permutation)</option>
        </select>
        {arch === 'cnn' ? (<>
          <label style={{ fontWeight: 600, marginLeft: 6 }}>Shift:</label>
          <select value={shift} onChange={e => { setShift(+e.target.value); setMatch(null); }} style={sel}>
            {[0,1,2,3,4].map(s => <option key={s} value={s}>{s} px</option>)}
          </select>
        </>) : (<>
          <label style={{ fontWeight: 600, marginLeft: 6 }}>Permutation:</label>
          <select value={permIdx} onChange={e => { setPermIdx(+e.target.value); setMatch(null); }} style={sel}>
            {PERMS.map((p, i) => <option key={i} value={i}>{p.label}</option>)}
          </select>
        </>)}
        <button onClick={handleVerify} style={{ padding: '5px 14px', borderRadius: 6, border: 'none', background: COLORS.network, color: '#fff', fontWeight: 600, fontSize: 13, cursor: 'pointer' }}>
          Verify
        </button>
        {match !== null && <span style={{ fontWeight: 700, color: match ? COLORS.match : COLORS.mismatch, fontSize: 13 }}>
          {match ? '✓ Paths match!' : '✗ Mismatch'}
        </span>}
      </div>
      <svg ref={svgRef} style={{ display: 'block', width: '100%' }} />
      <div style={{ marginTop: 8, padding: '10px 14px', background: COLORS.bg, borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12, color: COLORS.text, lineHeight: 1.7, fontFamily: 'ui-monospace, monospace' }}>
        <div style={{ textAlign: 'center', fontWeight: 700, marginBottom: 4, fontSize: 13 }}>Naturality Square</div>
        <div style={{ textAlign: 'center', whiteSpace: 'pre' }}>
          {'   X ──ρ'}<sub>X</sub>{'(g)──▶ X\n'}
          {'   │              │\n'}
          {'   f              f\n'}
          {'   │              │\n'}
          {'   ▼              ▼\n'}
          {'   Y ──ρ'}<sub>Y</sub>{'(g)──▶ Y'}
        </div>
        <div style={{ marginTop: 6, fontSize: 11, lineHeight: 1.6, textAlign: 'center' }}>
          <span>ρ<sub>X</sub>(g) = {arch === 'cnn' ? 'shift input' : 'permute input'}</span>{'  ·  '}
          <span>ρ<sub>Y</sub>(g) = {arch === 'cnn' ? 'shift output' : 'permute output'}</span>{'  ·  '}
          <span>f = {arch === 'cnn' ? 'convolution' : 'message passing'}</span>
        </div>
        <div style={{ textAlign: 'center', marginTop: 4, fontWeight: 700, fontSize: 12, color: COLORS.output }}>
          f(ρ<sub>X</sub>(g)(x)) = ρ<sub>Y</sub>(g)(f(x))
        </div>
      </div>
    </div>
  );
}
