#!/usr/bin/env node
/**
 * audit-cross-site-links.mjs — Cross-repo reciprocity validator.
 *
 * Walks src/content/topics/ in each of the three formal* sibling repos,
 * extracts cross-site frontmatter entries, builds a unified edge map, then
 * checks every edge has a reciprocal with opposite direction (Prereqs ↔
 * Connections) on the target side.
 *
 * Inputs (read-only):
 *   $HOME/Developer/Sites/formalCalculus/src/content/topics/*.mdx
 *   $HOME/Developer/Sites/formalStatistics/src/content/topics/*.mdx
 *   $HOME/Developer/Sites/formalML/src/content/topics/*.mdx
 *
 * Outputs (written under formalML's docs/plans/):
 *   docs/plans/audit-output/<site>-references.json     — per-site edge dump
 *   docs/plans/cross-site-audit-report.md              — consolidated report
 *   docs/plans/deferred-reciprocals.md                 — target-doesn't-exist log
 *
 * Run: node scripts/audit-cross-site-links.mjs
 */

import { readdir, readFile, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { homedir } from 'node:os';
import matter from 'gray-matter';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = dirname(SCRIPT_DIR);
const HOME = process.env.HOME ?? homedir();

const REPOS = {
  formalcalculus: join(HOME, 'Developer/Sites/formalCalculus'),
  formalstatistics: join(HOME, 'Developer/Sites/formalStatistics'),
  formalml: join(HOME, 'Developer/Sites/formalML'),
};

const SITE_LABELS = {
  formalcalculus: 'formalCalculus',
  formalstatistics: 'formalStatistics',
  formalml: 'formalML',
};

const CROSS_SITE_FIELDS = [
  'formalcalculusPrereqs',
  'formalcalculusConnections',
  'formalstatisticsPrereqs',
  'formalstatisticsConnections',
  'formalmlPrereqs',
  'formalmlConnections',
];

const OUTPUT_DIR = join(REPO_ROOT, 'docs/plans/audit-output');
const REPORT_PATH = join(REPO_ROOT, 'docs/plans/cross-site-audit-report.md');
const DEFERRED_PATH = join(REPO_ROOT, 'docs/plans/deferred-reciprocals.md');

const FIELD_RE = /^(formalcalculus|formalstatistics|formalml)(Prereqs|Connections)$/;

function parseField(field) {
  const m = field.match(FIELD_RE);
  if (!m) return null;
  return { targetSite: m[1], direction: m[2] };
}

function reciprocalField(sourceSite, direction) {
  const flipped = direction === 'Prereqs' ? 'Connections' : 'Prereqs';
  return `${sourceSite}${flipped}`;
}

async function loadRepo(site) {
  const root = REPOS[site];
  const topicsDir = join(root, 'src/content/topics');
  if (!existsSync(topicsDir)) {
    return { slugs: new Set(), frontmatter: new Map() };
  }
  const files = (await readdir(topicsDir)).filter((f) => f.endsWith('.mdx')).sort();
  const slugs = new Set();
  const frontmatter = new Map();
  for (const file of files) {
    const slug = file.replace(/\.mdx$/, '');
    slugs.add(slug);
    const raw = await readFile(join(topicsDir, file), 'utf8');
    try {
      const { data } = matter(raw);
      frontmatter.set(slug, data ?? {});
    } catch (err) {
      console.error(`[parse error] ${site}/${file}: ${err.message}`);
      frontmatter.set(slug, {});
    }
  }
  return { slugs, frontmatter };
}

function extractEdges(sourceSite, frontmatter) {
  const edges = [];
  for (const [sourceSlug, fm] of [...frontmatter.entries()].sort()) {
    for (const field of CROSS_SITE_FIELDS) {
      const entries = fm[field];
      if (!Array.isArray(entries)) continue;
      const meta = parseField(field);
      if (!meta) continue;
      for (const entry of entries) {
        if (!entry || typeof entry !== 'object' || typeof entry.topic !== 'string') continue;
        edges.push({
          sourceSite,
          sourceSlug,
          field,
          direction: meta.direction,
          targetSite: meta.targetSite,
          targetSlug: entry.topic,
          declaredSite: typeof entry.site === 'string' ? entry.site : null,
          title: typeof entry.title === 'string' ? entry.title : null,
          relationship: typeof entry.relationship === 'string' ? entry.relationship : '',
        });
      }
    }
  }
  return edges;
}

function checkEdge(edge, repos) {
  const issues = [];
  const targetRepo = repos[edge.targetSite];

  if (edge.declaredSite && edge.declaredSite !== edge.targetSite) {
    issues.push({
      kind: 'site-mismatch',
      detail: `entry.site="${edge.declaredSite}" but field "${edge.field}" implies "${edge.targetSite}"`,
    });
  }
  if (edge.relationship && edge.relationship.length < 40) {
    issues.push({
      kind: 'thin-prose',
      detail: `relationship is ${edge.relationship.length} chars (<40)`,
    });
  }
  if (edge.targetSite === edge.sourceSite) {
    issues.push({
      kind: 'self-site',
      detail: `cross-site field on ${edge.sourceSite} points to ${edge.targetSite}`,
    });
  }

  const targetExists = targetRepo.slugs.has(edge.targetSlug);
  if (!targetExists) {
    return {
      issues,
      reciprocal: { found: false, reason: 'target-missing' },
      targetExists: false,
    };
  }

  const expectedField = reciprocalField(edge.sourceSite, edge.direction);
  const targetFm = targetRepo.frontmatter.get(edge.targetSlug) ?? {};
  const entries = Array.isArray(targetFm[expectedField]) ? targetFm[expectedField] : [];
  const match = entries.find((e) => e && typeof e === 'object' && e.topic === edge.sourceSlug);
  if (match) {
    return { issues, reciprocal: { found: true, field: expectedField }, targetExists: true };
  }

  const wrongField = `${edge.sourceSite}${edge.direction}`;
  const wrongEntries = Array.isArray(targetFm[wrongField]) ? targetFm[wrongField] : [];
  const wrongMatch = wrongEntries.find((e) => e && typeof e === 'object' && e.topic === edge.sourceSlug);
  if (wrongMatch) {
    return {
      issues,
      reciprocal: { found: false, reason: 'direction-mismatch', actualField: wrongField, expectedField },
      targetExists: true,
    };
  }

  return {
    issues,
    reciprocal: { found: false, reason: 'missing', expectedField },
    targetExists: true,
  };
}

function tokenize(slug) {
  return slug.toLowerCase().split(/[-_]/).filter(Boolean).map((t) => t.replace(/s$/, ''));
}

function suggestSimilarSlug(slug, candidates) {
  if (candidates.has(slug)) return slug;
  const targetTokens = tokenize(slug);
  if (targetTokens.length === 0) return null;
  let best = null;
  let bestScore = 0;
  for (const candidate of candidates) {
    const cTokens = tokenize(candidate);
    if (cTokens.length === 0) continue;
    const intersection = targetTokens.filter((t) => cTokens.includes(t)).length;
    if (intersection === 0) continue;
    const score = intersection / Math.min(targetTokens.length, cTokens.length);
    if (score > bestScore && score >= 0.67) {
      best = candidate;
      bestScore = score;
    }
  }
  return best;
}

function renderReport({ repos, edges, results, deferredEdges }) {
  const lines = [];
  const now = new Date().toISOString().slice(0, 10);

  lines.push(`# Cross-Site Audit Report — ${now}`, '');
  lines.push(
    'Generated by `scripts/audit-cross-site-links.mjs`. Walks every `.mdx` in each formal* repo, extracts cross-site frontmatter entries, and validates reciprocity (every `A.x → B.y` must have `B.y → A.x` with the opposite direction).',
    '',
  );

  lines.push('## Summary', '');
  lines.push('| Site | Topics | Outgoing edges | Reciprocated | Missing | Wrong direction | Deferred (target missing) | Slug drift |');
  lines.push('|---|---:|---:|---:|---:|---:|---:|---:|');
  for (const site of Object.keys(REPOS)) {
    const siteEdges = edges.filter((e) => e.sourceSite === site);
    const siteResults = results.filter((r) => r.edge.sourceSite === site);
    const reciprocated = siteResults.filter((r) => r.check.reciprocal.found).length;
    const missing = siteResults.filter((r) => r.check.reciprocal.reason === 'missing').length;
    const wrongDir = siteResults.filter((r) => r.check.reciprocal.reason === 'direction-mismatch').length;
    const deferred = siteResults.filter((r) => r.check.reciprocal.reason === 'target-missing' && !r.suggested).length;
    const drift = siteResults.filter((r) => r.suggested).length;
    lines.push(`| ${SITE_LABELS[site]} | ${repos[site].slugs.size} | ${siteEdges.length} | ${reciprocated} | ${missing} | ${wrongDir} | ${deferred} | ${drift} |`);
  }
  lines.push('');

  lines.push('## Slug drift (target slug doesn\'t exist but a similar one does)', '');
  const driftRows = results.filter((r) => r.suggested).sort((a, b) => a.edge.sourceSlug.localeCompare(b.edge.sourceSlug));
  if (driftRows.length === 0) {
    lines.push('_None._', '');
  } else {
    lines.push('| Source | Field | Declared target | Suggested target |');
    lines.push('|---|---|---|---|');
    for (const r of driftRows) {
      lines.push(`| ${SITE_LABELS[r.edge.sourceSite]}/\`${r.edge.sourceSlug}\` | \`${r.edge.field}\` | \`${r.edge.targetSlug}\` | \`${r.suggested}\` |`);
    }
    lines.push('');
  }

  lines.push('## Direction mismatches (declared on wrong side)', '');
  const mismatchRows = results.filter((r) => r.check.reciprocal.reason === 'direction-mismatch');
  if (mismatchRows.length === 0) {
    lines.push('_None._', '');
  } else {
    lines.push('| Source | Field declared | Target | Reciprocal exists in | Should be |');
    lines.push('|---|---|---|---|---|');
    for (const r of mismatchRows) {
      lines.push(`| ${SITE_LABELS[r.edge.sourceSite]}/\`${r.edge.sourceSlug}\` | \`${r.edge.field}\` | ${SITE_LABELS[r.edge.targetSite]}/\`${r.edge.targetSlug}\` | \`${r.check.reciprocal.actualField}\` | \`${r.check.reciprocal.expectedField}\` |`);
    }
    lines.push('');
  }

  lines.push('## Missing reciprocals (target exists, reciprocal field empty or no matching entry)', '');
  const missingByPair = new Map();
  for (const r of results.filter((r) => r.check.reciprocal.reason === 'missing')) {
    const key = `${r.edge.sourceSite}→${r.edge.targetSite}`;
    if (!missingByPair.has(key)) missingByPair.set(key, []);
    missingByPair.get(key).push(r);
  }
  if (missingByPair.size === 0) {
    lines.push('_None — every declared edge has a reciprocal._', '');
  } else {
    for (const [key, rows] of [...missingByPair.entries()].sort()) {
      lines.push(`### ${key.replace('→', ' → ')} (${rows.length})`, '');
      lines.push('| Source slug | Direction | Target slug | Expected reciprocal field |');
      lines.push('|---|---|---|---|');
      for (const r of rows.sort((a, b) => a.edge.sourceSlug.localeCompare(b.edge.sourceSlug) || a.edge.targetSlug.localeCompare(b.edge.targetSlug))) {
        lines.push(`| \`${r.edge.sourceSlug}\` | ${r.edge.direction} | \`${r.edge.targetSlug}\` | \`${r.check.reciprocal.expectedField}\` |`);
      }
      lines.push('');
    }
  }

  lines.push('## Deferred reciprocals (target slug doesn\'t exist on target repo yet)', '');
  if (deferredEdges.length === 0) {
    lines.push('_None._', '');
  } else {
    const byTarget = new Map();
    for (const r of deferredEdges) {
      const key = `${r.edge.targetSite}/${r.edge.targetSlug}`;
      if (!byTarget.has(key)) byTarget.set(key, []);
      byTarget.get(key).push(r);
    }
    lines.push('Edges that point at slugs the target repo hasn\'t shipped yet. Not failures — log entries for the day those topics ship.', '');
    lines.push('| Missing target | Pointed at by |');
    lines.push('|---|---|');
    for (const [target, rows] of [...byTarget.entries()].sort()) {
      const sources = rows.map((r) => `${SITE_LABELS[r.edge.sourceSite]}/\`${r.edge.sourceSlug}\` (${r.edge.field})`).join('<br/>');
      lines.push(`| \`${target}\` | ${sources} |`);
    }
    lines.push('');
  }

  lines.push('## Frontmatter quality flags', '');
  const flagCounts = { 'thin-prose': 0, 'site-mismatch': 0, 'self-site': 0 };
  const flagRows = [];
  for (const r of results) {
    for (const issue of r.check.issues) {
      flagCounts[issue.kind] = (flagCounts[issue.kind] ?? 0) + 1;
      flagRows.push({ ...r, issue });
    }
  }
  const totalFlags = Object.values(flagCounts).reduce((a, b) => a + b, 0);
  if (totalFlags === 0) {
    lines.push('_None._', '');
  } else {
    lines.push(`Counts: ${Object.entries(flagCounts).filter(([, n]) => n > 0).map(([k, n]) => `${k}=${n}`).join(', ')}`, '');
    lines.push('| Kind | Source | Field | Target | Detail |');
    lines.push('|---|---|---|---|---|');
    for (const r of flagRows.sort((a, b) => a.issue.kind.localeCompare(b.issue.kind))) {
      lines.push(`| ${r.issue.kind} | ${SITE_LABELS[r.edge.sourceSite]}/\`${r.edge.sourceSlug}\` | \`${r.edge.field}\` | ${SITE_LABELS[r.edge.targetSite] ?? '?'}/\`${r.edge.targetSlug}\` | ${r.issue.detail} |`);
    }
    lines.push('');
  }

  return lines.join('\n');
}

function renderDeferred(deferredEdges) {
  if (deferredEdges.length === 0) {
    return '# Deferred reciprocals\n\n_None at the time of the last audit._\n';
  }
  const lines = ['# Deferred reciprocals', ''];
  lines.push('Cross-site edges that point at target slugs not yet shipped on the target repo. These do **not** fail the audit — they\'re a running log of reciprocals to add when those topics publish.', '');
  lines.push('Regenerated by `pnpm audit:cross-site` (or directly via `node scripts/audit-cross-site-links.mjs`).', '');
  const byTarget = new Map();
  for (const r of deferredEdges) {
    const key = `${r.edge.targetSite}/${r.edge.targetSlug}`;
    if (!byTarget.has(key)) byTarget.set(key, []);
    byTarget.get(key).push(r);
  }
  for (const [target, rows] of [...byTarget.entries()].sort()) {
    lines.push(`## When \`${target}\` ships`, '');
    lines.push('Add the reciprocal pointer back to:', '');
    for (const r of rows.sort((a, b) => a.edge.sourceSlug.localeCompare(b.edge.sourceSlug))) {
      lines.push(`- **${SITE_LABELS[r.edge.sourceSite]}/${r.edge.sourceSlug}** declares \`${r.edge.field}\` → \`${r.edge.targetSlug}\` (${r.edge.direction.toLowerCase()})`);
      lines.push(`  - Reciprocal field on the new topic: \`${reciprocalField(r.edge.sourceSite, r.edge.direction)}\``);
      if (r.edge.relationship) {
        const truncated = r.edge.relationship.length > 200 ? `${r.edge.relationship.slice(0, 197)}…` : r.edge.relationship;
        lines.push(`  - Source-side prose (use as a starting point, rewrite from new topic's vantage): _${truncated}_`);
      }
    }
    lines.push('');
  }
  return lines.join('\n');
}

async function main() {
  console.log('Loading repos...');
  const repos = {};
  for (const site of Object.keys(REPOS)) {
    repos[site] = await loadRepo(site);
    console.log(`  ${SITE_LABELS[site]}: ${repos[site].slugs.size} topics`);
  }

  console.log('\nExtracting cross-site edges...');
  const allEdges = [];
  for (const site of Object.keys(REPOS)) {
    const edges = extractEdges(site, repos[site].frontmatter);
    allEdges.push(...edges);
    console.log(`  ${SITE_LABELS[site]}: ${edges.length} outgoing edges`);
  }

  console.log('\nValidating reciprocity...');
  const results = [];
  const deferredEdges = [];
  for (const edge of allEdges) {
    const check = checkEdge(edge, repos);
    let suggested = null;
    if (check.reciprocal.reason === 'target-missing') {
      const candidates = repos[edge.targetSite]?.slugs ?? new Set();
      suggested = suggestSimilarSlug(edge.targetSlug, candidates);
    }
    const result = { edge, check, suggested };
    results.push(result);
    if (check.reciprocal.reason === 'target-missing' && !suggested) {
      deferredEdges.push(result);
    }
  }

  await mkdir(OUTPUT_DIR, { recursive: true });

  for (const site of Object.keys(REPOS)) {
    const siteEdges = allEdges.filter((e) => e.sourceSite === site);
    const path = join(OUTPUT_DIR, `${site}-references.json`);
    await writeFile(path, JSON.stringify(siteEdges, null, 2) + '\n', 'utf8');
    console.log(`  wrote ${path} (${siteEdges.length} edges)`);
  }

  const report = renderReport({ repos, edges: allEdges, results, deferredEdges });
  await writeFile(REPORT_PATH, report, 'utf8');
  console.log(`  wrote ${REPORT_PATH}`);

  const deferred = renderDeferred(deferredEdges);
  await writeFile(DEFERRED_PATH, deferred, 'utf8');
  console.log(`  wrote ${DEFERRED_PATH}`);

  const failures = results.filter((r) =>
    r.check.reciprocal.reason === 'missing' ||
    r.check.reciprocal.reason === 'direction-mismatch' ||
    r.suggested ||
    r.check.issues.some((i) => i.kind === 'site-mismatch' || i.kind === 'self-site'),
  ).length;

  console.log('');
  console.log(`Total edges: ${allEdges.length}`);
  console.log(`Reciprocated: ${results.filter((r) => r.check.reciprocal.found).length}`);
  console.log(`Missing reciprocals: ${results.filter((r) => r.check.reciprocal.reason === 'missing').length}`);
  console.log(`Direction mismatches: ${results.filter((r) => r.check.reciprocal.reason === 'direction-mismatch').length}`);
  console.log(`Slug drift candidates: ${results.filter((r) => r.suggested).length}`);
  console.log(`Deferred (target missing): ${deferredEdges.length}`);
  console.log('');
  console.log(failures === 0 ? '✓ No reconcilable issues.' : `⚠ ${failures} reconcilable issue(s) — see ${REPORT_PATH}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
