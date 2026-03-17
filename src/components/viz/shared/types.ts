export interface Point2D {
  x: number;
  y: number;
  id: string;
}

export interface PersistenceInterval {
  birth: number;
  death: number;
  dimension: number;
}

export interface Simplex {
  vertices: string[];
  dimension: number;
  birthTime: number;
}

export interface DAGNode {
  id: string;
  label: string;
  status: string;
  domain: string;
  url?: string | null;
}

export interface DAGEdge {
  source: string;
  target: string;
}

// ─── Mapper Algorithm Types ───

export interface MapperPoint {
  x: number;
  y: number;
  id: number;
  filterValue: number;
}

export interface MapperParams {
  nIntervals: number;
  overlap: number;
  clusterEps?: number; // auto-estimated if omitted
  minClusterSize?: number; // defaults to 2
}

export interface MapperCluster {
  intervalIdx: number;
  clusterIdx: number;
  members: number[]; // indices into the original point array
  centroidX: number;
  centroidY: number;
}

export interface MapperResult {
  clusters: MapperCluster[];
  nodes: MapperGraphNode[];
  edges: [number, number][];
  intervals: [number, number][];
  pullbackAssignments: number[][]; // for each interval, which point indices
}

export interface MapperGraphNode {
  id: number;
  size: number;
  members: number[];
  centroidX: number;
  centroidY: number;
}

// ─── Sheaf Theory Types ───

export interface SheafNode {
  id: string;
  x: number;
  y: number;
  stalkDim: number;
  value: number[];
}

export interface SheafEdge {
  source: string;
  target: string;
  stalkDim: number;
  restrictionMap: number[][];
}
