import { useState, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useD3 } from './shared/useD3';
import { useResizeObserver } from './shared/useResizeObserver';
import { dimensionColors } from './shared/colorScales';
import {
  entropy,
  normalize,
  buildHuffmanTree,
  huffmanCodes,
  expectedCodeLength,
} from './shared/informationTheory';
import type { HuffmanNode } from './shared/types';

// ── Constants ────────────────────────────────────────────────────────

const DIST_HEIGHT = 280;
const TREE_HEIGHT = 340;
const SM_BREAKPOINT = 640;
const MARGIN = { top: 30, right: 20, bottom: 40, left: 50 };

const TEAL = dimensionColors[0];
const PURPLE = dimensionColors[1];
const AMBER = '#D97706';

const K_OPTIONS = [2, 3, 4, 5, 8];

const fmt = (x: number) => x.toFixed(3);

function defaultProbs(k: number): number[] {
  // Zipf-like distribution for interesting Huffman trees
  const raw = Array.from({ length: k }, (_, i) => 1 / (i + 1));
  return normalize(raw);
}

function symbolLabels(k: number): string[] {
  return Array.from({ length: k }, (_, i) => String.fromCharCode(65 + i));
}

// ── Convert Huffman tree to d3.hierarchy format ──────────────────

interface TreeNode {
  name: string;
  prob: number;
  code?: string;
  children?: TreeNode[];
}

function toTreeData(node: HuffmanNode): TreeNode {
  if (node.symbol !== undefined) {
    return { name: node.symbol, prob: node.probability, code: node.code };
  }
  const children: TreeNode[] = [];
  if (node.left) children.push(toTreeData(node.left));
  if (node.right) children.push(toTreeData(node.right));
  return { name: '', prob: node.probability, children };
}

// ── Component ────────────────────────────────────────────────────────

export default function SourceCodingExplorer() {
  const { ref: containerRef, width: containerWidth } = useResizeObserver<HTMLDivElement>();

  const [k, setK] = useState(5);
  const [probs, setProbs] = useState(() => defaultProbs(5));
  const [showTree, setShowTree] = useState(true);

  const isStacked = containerWidth > 0 && containerWidth < SM_BREAKPOINT;
  const symbols = useMemo(() => symbolLabels(k), [k]);

  const handleKChange = useCallback((newK: number) => {
    setK(newK);
    setProbs(defaultProbs(newK));
    setShowTree(true);
  }, []);

  // ── Computed values ──────────────────────────────────────────────
  const H = useMemo(() => entropy(probs), [probs]);
  const tree = useMemo(() => buildHuffmanTree(symbols, probs), [symbols, probs]);
  const codes = useMemo(() => huffmanCodes(tree), [tree]);
  const avgLen = useMemo(() => expectedCodeLength(symbols, probs, codes), [symbols, probs, codes]);

  // ── Distribution bar chart (left) ──────────────────────────────
  const distWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.35);

  const distRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (distWidth <= 0) return;

      const w = distWidth - MARGIN.left - MARGIN.right;
      const h = DIST_HEIGHT - MARGIN.top - MARGIN.bottom;
      const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

      const xScale = d3.scaleBand<string>()
        .domain(symbols)
        .range([0, w])
        .padding(0.25);

      const yScale = d3.scaleLinear().domain([0, Math.max(...probs) * 1.15]).range([h, 0]);

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px');

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('fill', 'var(--color-text-secondary)')
        .style('font-size', '12px');

      // Draggable bars — bind {prob, index} to avoid indexOf ambiguity
      const barData = probs.map((p, i) => ({ p, i }));
      const bars = g.selectAll<SVGRectElement, { p: number; i: number }>('.bar')
        .data(barData)
        .enter()
        .append('rect')
        .attr('x', (d) => xScale(symbols[d.i])!)
        .attr('y', (d) => yScale(d.p))
        .attr('width', xScale.bandwidth())
        .attr('height', (d) => h - yScale(d.p))
        .attr('fill', TEAL)
        .attr('rx', 2)
        .style('cursor', 'ns-resize');

      const drag = d3.drag<SVGRectElement, { p: number; i: number }>()
        .on('drag', function (event, d) {
          const newVal = Math.max(0.001, yScale.invert(Math.max(0, Math.min(h, event.y))));
          const updated = [...probs];
          updated[d.i] = newVal;
          setProbs(normalize(updated));
        });

      bars.call(drag);

      // Entropy line
      g.append('line')
        .attr('x1', 0).attr('x2', w)
        .attr('y1', yScale(0)).attr('y2', yScale(0))
        .attr('stroke', 'var(--color-border)');

      // Title
      g.append('text')
        .attr('x', w / 2).attr('y', -10)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '13px')
        .style('font-weight', '500')
        .text('Source Distribution');

      g.selectAll('.domain').style('stroke', 'var(--color-border)');
      g.selectAll('.tick line').style('stroke', 'var(--color-border)');
    },
    [distWidth, probs, symbols]
  );

  // ── Huffman tree (center) ──────────────────────────────────────
  const treeWidth = isStacked ? containerWidth : Math.floor(containerWidth * 0.4);

  const treeRef = useD3<SVGSVGElement>(
    (svg) => {
      svg.selectAll('*').remove();
      if (treeWidth <= 0) return;

      const w = treeWidth - 20;
      const h = TREE_HEIGHT - 40;

      const treeData = toTreeData(tree);
      const root = d3.hierarchy(treeData);
      const treeLayout = d3.tree<TreeNode>().size([w - 40, h - 60]);
      treeLayout(root);

      const g = svg.append('g').attr('transform', 'translate(20, 30)');

      // Links with 0/1 labels
      g.selectAll('.link')
        .data(root.links())
        .enter()
        .append('path')
        .attr('d', (d) => {
          return `M${d.source.x},${d.source.y} C${d.source.x},${(d.source.y + d.target.y) / 2} ${d.target.x},${(d.source.y + d.target.y) / 2} ${d.target.x},${d.target.y}`;
        })
        .attr('fill', 'none')
        .attr('stroke', 'var(--color-border)')
        .attr('stroke-width', 1.5);

      // Edge labels (0 = left, 1 = right)
      root.links().forEach((link) => {
        const mx = (link.source.x + link.target.x) / 2;
        const my = (link.source.y + link.target.y) / 2;
        const isLeft = link.target.x < link.source.x;
        g.append('text')
          .attr('x', mx + (isLeft ? -8 : 8))
          .attr('y', my - 2)
          .attr('text-anchor', 'middle')
          .style('fill', AMBER)
          .style('font-size', '11px')
          .style('font-weight', '600')
          .text(isLeft ? '0' : '1');
      });

      // Nodes
      root.descendants().forEach((d) => {
        const isLeaf = !d.children;
        g.append('circle')
          .attr('cx', d.x).attr('cy', d.y).attr('r', isLeaf ? 16 : 10)
          .attr('fill', isLeaf ? TEAL : 'var(--color-bg)')
          .attr('stroke', isLeaf ? TEAL : 'var(--color-border)')
          .attr('stroke-width', 1.5)
          .attr('fill-opacity', isLeaf ? 0.2 : 1);

        if (isLeaf && d.data.name) {
          g.append('text')
            .attr('x', d.x).attr('y', d.y + 1)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .style('fill', TEAL)
            .style('font-size', '12px')
            .style('font-weight', '700')
            .text(d.data.name);

          // Code below leaf
          const code = codes.get(d.data.name) ?? '';
          g.append('text')
            .attr('x', d.x).attr('y', d.y + 28)
            .attr('text-anchor', 'middle')
            .style('fill', 'var(--color-text-secondary)')
            .style('font-size', '10px')
            .style('font-family', 'var(--font-mono)')
            .text(code);
        } else {
          // Internal node: show probability
          g.append('text')
            .attr('x', d.x).attr('y', d.y - 14)
            .attr('text-anchor', 'middle')
            .style('fill', 'var(--color-text-secondary)')
            .style('font-size', '9px')
            .text(d.data.prob.toFixed(2));
        }
      });

      // Title
      g.append('text')
        .attr('x', (w - 40) / 2).attr('y', -16)
        .attr('text-anchor', 'middle')
        .style('fill', 'var(--color-text)')
        .style('font-size', '13px')
        .style('font-weight', '500')
        .text('Huffman Tree');
    },
    [treeWidth, tree, codes]
  );

  // ── Comparison table (right) ───────────────────────────────────
  const tableData = useMemo(() => {
    return symbols.map((s, i) => ({
      symbol: s,
      prob: probs[i],
      code: codes.get(s) ?? '',
      length: codes.get(s)?.length ?? 0,
    }));
  }, [symbols, probs, codes]);

  return (
    <div ref={containerRef} className="w-full">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3 mb-4 text-sm">
        <label className="flex items-center gap-1.5" style={{ color: 'var(--color-text-secondary)' }}>
          Symbols (k):
          <select
            value={k}
            onChange={(e) => handleKChange(Number(e.target.value))}
            className="rounded border px-2 py-1 text-sm"
            style={{ borderColor: 'var(--color-border)', backgroundColor: 'var(--color-bg)', color: 'var(--color-text)' }}
          >
            {K_OPTIONS.map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>

        <button
          onClick={() => setShowTree((v) => !v)}
          className="rounded px-3 py-1 text-sm font-medium"
          style={{
            backgroundColor: 'var(--color-definition-bg)',
            color: TEAL,
            border: `1px solid ${TEAL}`,
          }}
        >
          {showTree ? 'Hide Tree' : 'Show Tree'}
        </button>
      </div>

      {/* Main panels */}
      <div className={`flex ${isStacked ? 'flex-col' : 'flex-row'} gap-2`}>
        <svg ref={distRef} width={distWidth} height={DIST_HEIGHT} />
        {showTree && <svg ref={treeRef} width={treeWidth} height={TREE_HEIGHT} />}

        {/* Comparison table */}
        <div
          className="flex-1 overflow-auto text-sm"
          style={{ minWidth: isStacked ? '100%' : '25%' }}
        >
          <table className="w-full border-collapse">
            <thead>
              <tr style={{ borderBottom: '2px solid var(--color-border)' }}>
                <th className="text-left px-2 py-1.5" style={{ color: 'var(--color-text-secondary)' }}>Sym</th>
                <th className="text-right px-2 py-1.5" style={{ color: 'var(--color-text-secondary)' }}>p(x)</th>
                <th className="text-center px-2 py-1.5" style={{ color: 'var(--color-text-secondary)' }}>Code</th>
                <th className="text-right px-2 py-1.5" style={{ color: 'var(--color-text-secondary)' }}>ℓ(x)</th>
              </tr>
            </thead>
            <tbody>
              {tableData.map((row) => (
                <tr
                  key={row.symbol}
                  style={{ borderBottom: '1px solid var(--color-border)' }}
                >
                  <td className="px-2 py-1.5 font-semibold" style={{ color: TEAL }}>{row.symbol}</td>
                  <td className="text-right px-2 py-1.5 font-mono text-xs" style={{ color: 'var(--color-text)' }}>{row.prob.toFixed(3)}</td>
                  <td className="text-center px-2 py-1.5 font-mono text-xs" style={{ color: AMBER }}>{row.code}</td>
                  <td className="text-right px-2 py-1.5" style={{ color: 'var(--color-text)' }}>{row.length}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Summary */}
          <div className="mt-3 px-2 space-y-1" style={{ color: 'var(--color-text)' }}>
            <div className="flex justify-between text-xs">
              <span>H(X):</span>
              <span className="font-mono font-semibold" style={{ color: PURPLE }}>{fmt(H)} bits</span>
            </div>
            <div className="flex justify-between text-xs">
              <span>E[ℓ]:</span>
              <span className="font-mono font-semibold" style={{ color: TEAL }}>{fmt(avgLen)} bits</span>
            </div>
            <div className="flex justify-between text-xs">
              <span>Gap (E[ℓ] − H):</span>
              <span className="font-mono font-semibold" style={{ color: AMBER }}>{fmt(avgLen - H)}</span>
            </div>
            <div
              className="mt-2 text-xs px-2 py-1.5 rounded"
              style={{ backgroundColor: 'var(--color-definition-bg)', color: 'var(--color-text-secondary)' }}
            >
              H(X) ≤ E[ℓ] &lt; H(X) + 1
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
