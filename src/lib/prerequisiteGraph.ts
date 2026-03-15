import type { DAGNode, DAGEdge } from '../components/viz/shared/types';

interface TopicEntry {
  id: string;
  data: {
    title: string;
    status: string;
    domain: string;
    prerequisites: string[];
  };
}

export function buildPrerequisiteGraph(topics: TopicEntry[]): { nodes: DAGNode[]; edges: DAGEdge[] } {
  const nodes: DAGNode[] = topics.map((t) => ({
    id: t.id,
    label: t.data.title,
    status: t.data.status,
    domain: t.data.domain,
  }));

  const edges: DAGEdge[] = [];
  for (const topic of topics) {
    for (const prereq of topic.data.prerequisites) {
      edges.push({ source: prereq, target: topic.id });
    }
  }

  return { nodes, edges };
}
