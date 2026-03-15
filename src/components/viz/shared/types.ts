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
}

export interface DAGEdge {
  source: string;
  target: string;
}
