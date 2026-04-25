import type { DAGEdge, DAGNode } from '../components/viz/shared/types';

type CrossSiteSite = 'formalcalculus' | 'formalstatistics';

interface CrossSiteRef {
  topic: string;
  site: CrossSiteSite;
  relationship: string;
}

interface PublishedTopic {
  id: string;
  data: {
    status: string;
    formalcalculusPrereqs?: CrossSiteRef[];
    formalstatisticsPrereqs?: CrossSiteRef[];
  };
}

export interface ExternalNode extends DAGNode {
  status: 'external';
  domain: CrossSiteSite;
  url: string;
  external: true;
}

const SITE_DOMAINS: Record<CrossSiteSite, string> = {
  formalcalculus: 'https://formalcalculus.com',
  formalstatistics: 'https://formalstatistics.com',
};

export function humanReadableFromSlug(slug: string): string {
  return slug
    .replace(/-and-/g, ' & ')
    .replace(/-/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Walks each published topic's frontmatter and emits ephemeral cross-site
 * nodes (one per unique sister-site prereq) plus edges pointing into the
 * formalML topic. Pure build-time transform — caller merges the result with
 * the canonical curriculum-graph.json data before passing to the renderer.
 */
export function synthesizeCrossSiteNodes(
  topics: PublishedTopic[],
): { nodes: ExternalNode[]; edges: DAGEdge[] } {
  const nodes = new Map<string, ExternalNode>();
  const edges: DAGEdge[] = [];

  const publishedOnly = topics.filter((t) => t.data.status === 'published');

  for (const topic of publishedOnly) {
    const refs: Array<{ refs: CrossSiteRef[] | undefined; site: CrossSiteSite }> = [
      { refs: topic.data.formalcalculusPrereqs, site: 'formalcalculus' },
      { refs: topic.data.formalstatisticsPrereqs, site: 'formalstatistics' },
    ];

    for (const { refs: list, site } of refs) {
      if (!list) continue;
      for (const ref of list) {
        const id = `${site}/${ref.topic}`;
        if (!nodes.has(id)) {
          nodes.set(id, {
            id,
            label: humanReadableFromSlug(ref.topic),
            domain: site,
            status: 'external',
            url: `${SITE_DOMAINS[site]}/topics/${ref.topic}`,
            external: true,
          });
        }
        edges.push({ source: id, target: topic.id });
      }
    }
  }

  return { nodes: Array.from(nodes.values()), edges };
}
