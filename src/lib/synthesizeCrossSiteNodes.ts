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

// Lowercase tokens we want preserved as uppercase when humanizing a slug.
// Common stats / calculus / ML acronyms that appear in sister-site slugs.
const ACRONYMS = new Set([
  'ab',
  'anova',
  'cdf',
  'cqr',
  'em',
  'fft',
  'gp',
  'glm',
  'glmm',
  'hmc',
  'iid',
  'kl',
  'ks',
  'map',
  'mcmc',
  'ml',
  'mle',
  'ode',
  'pac',
  'pca',
  'pde',
  'pdf',
  'pi',
  'pmf',
  'sgd',
  'sgld',
  'svd',
  'tda',
  'tmle',
  'vc',
]);

export function humanReadableFromSlug(slug: string): string {
  return slug
    .split('-')
    .map((word) => {
      if (word === 'and') return '&';
      if (ACRONYMS.has(word)) return word.toUpperCase();
      return word.charAt(0).toUpperCase() + word.slice(1);
    })
    .join(' ');
}

/**
 * Walks each topic's frontmatter and emits ephemeral cross-site nodes
 * (one per unique sister-site prereq) plus edges pointing into the
 * formalML topic. Pure build-time transform — caller merges the result
 * with the canonical curriculum-graph.json data before passing to the
 * renderer.
 *
 * Caller is responsible for filtering to `data.status === 'published'`
 * before calling — this function trusts its input and does not re-filter.
 */
export function synthesizeCrossSiteNodes(
  topics: PublishedTopic[],
): { nodes: ExternalNode[]; edges: DAGEdge[] } {
  const nodes = new Map<string, ExternalNode>();
  const edges: DAGEdge[] = [];

  for (const topic of topics) {
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
