/**
 * Validates that all topic connections and prerequisites reference existing topic IDs.
 * Run with: pnpm validate
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join, basename } from 'node:path';

const TOPICS_DIR = join(import.meta.dirname, '..', 'content', 'topics');
const CURRICULUM_GRAPH = join(import.meta.dirname, '..', 'data', 'curriculum-graph.json');

// Collect all valid topic IDs from MDX filenames
const mdxFiles = readdirSync(TOPICS_DIR).filter((f) => f.endsWith('.mdx'));
const validIds = new Set(mdxFiles.map((f) => basename(f, '.mdx')));

// Also collect IDs from curriculum-graph.json
const graph = JSON.parse(readFileSync(CURRICULUM_GRAPH, 'utf-8'));
const graphIds = new Set<string>(graph.nodes.map((n: { id: string }) => n.id));

let errors = 0;

// Check graph ↔ MDX alignment
for (const id of validIds) {
  if (!graphIds.has(id)) {
    console.error(`[MISMATCH] MDX file "${id}.mdx" has no entry in curriculum-graph.json`);
    errors++;
  }
}
for (const id of graphIds) {
  if (!validIds.has(id)) {
    console.error(`[MISMATCH] curriculum-graph.json node "${id}" has no MDX file`);
    errors++;
  }
}

// Check graph edges reference valid nodes
for (const edge of graph.edges) {
  if (!graphIds.has(edge.source)) {
    console.error(`[EDGE] Edge source "${edge.source}" is not a valid node`);
    errors++;
  }
  if (!graphIds.has(edge.target)) {
    console.error(`[EDGE] Edge target "${edge.target}" is not a valid node`);
    errors++;
  }
}

// Parse frontmatter from each MDX file and validate references
for (const file of mdxFiles) {
  const topicId = basename(file, '.mdx');
  const content = readFileSync(join(TOPICS_DIR, file), 'utf-8');

  // Extract YAML frontmatter between --- delimiters
  const match = content.match(/^---\n([\s\S]*?)\n---/);
  if (!match) {
    console.error(`[PARSE] Could not extract frontmatter from ${file}`);
    errors++;
    continue;
  }

  const frontmatter = match[1];

  // Extract prerequisites (YAML array)
  const prereqMatches = frontmatter.matchAll(/prerequisites:\s*\n((?:\s+-\s+"[^"]+"\n?)*)/g);
  for (const m of prereqMatches) {
    const items = m[1].matchAll(/-\s+"([^"]+)"/g);
    for (const item of items) {
      if (!validIds.has(item[1])) {
        console.error(`[PREREQ] ${topicId}: prerequisite "${item[1]}" is not a valid topic`);
        errors++;
      }
    }
  }

  // Also handle inline array format: prerequisites: ["foo", "bar"]
  const inlinePrereq = frontmatter.match(/prerequisites:\s*\[([^\]]*)\]/);
  if (inlinePrereq) {
    const items = inlinePrereq[1].matchAll(/"([^"]+)"/g);
    for (const item of items) {
      if (!validIds.has(item[1])) {
        console.error(`[PREREQ] ${topicId}: prerequisite "${item[1]}" is not a valid topic`);
        errors++;
      }
    }
  }

  // Extract connections (YAML objects with topic field)
  const connectionMatches = frontmatter.matchAll(/- topic:\s*"([^"]+)"/g);
  for (const m of connectionMatches) {
    if (!validIds.has(m[1])) {
      console.error(`[CONNECTION] ${topicId}: connection topic "${m[1]}" is not a valid topic`);
      errors++;
    }
  }
}

if (errors > 0) {
  console.error(`\n${errors} validation error(s) found.`);
  process.exit(1);
} else {
  console.log('All connections, prerequisites, and graph references are valid.');
}
