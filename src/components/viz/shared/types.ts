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

// ─── Measure-Theoretic Probability Types ───

export interface ConvergencePath {
  values: number[];
  label: string;
}

export interface MartingalePath {
  values: number[];
  regime?: number[]; // for regime-switching process
}

// ─── Concentration Inequalities Types ───

export interface TailBound {
  name: string;
  values: number[];
  color: string;
  dashed: boolean;
}

export interface ConcentrationConfig {
  n: number;
  epsilon: number;
  delta: number;
  hypothesisClassSize: number;
  lossBound: number;
}

// ─── PAC Learning Types ───

export interface PACBoundConfig {
  hypothesisClassSize: number;
  vcDimension: number;
  delta: number;
  epsilon: number;
  sampleSize: number;
}

export interface ShatteringResult {
  labeling: boolean[];
  realizable: boolean;
}

// ─── Bayesian Nonparametrics Types ───

export interface DPConfig {
  alpha: number;
  truncationLevel: number;
}

export interface GPConfig {
  kernel: 'rbf' | 'matern32' | 'linear';
  lengthScale: number;
  noiseVariance: number;
}

// ─── Information Theory Types ───

export interface DiscreteDistribution {
  labels: string[];
  probabilities: number[];
}

export interface JointDistribution {
  xLabels: string[];
  yLabels: string[];
  joint: number[][]; // joint[i][j] = p(x_i, y_j)
}

export interface HuffmanNode {
  symbol?: string;
  probability: number;
  code?: string;
  left?: HuffmanNode;
  right?: HuffmanNode;
}

// ─── Rate-Distortion Types ───

export interface BlahutArimotoState {
  iteration: number;
  qXhat: number[];
  pXhatGivenX: number[][];
  rate: number;
  distortion: number;
}

export interface RateDistortionPoint {
  rate: number;
  distortion: number;
}

export interface GraphEdge {
  source: number;
  target: number;
  weight: number;
}

export interface GraphPartition {
  sides: [number[], number[]];
  cutSize: number;
  cheegerRatio: number;
}

// ─── Random Walks Types ───

export interface WalkState {
  currentVertex: number;
  visitCounts: number[];
  totalSteps: number;
  trajectory: number[];
}

// ─── Expander Graph Types ───

export interface ExpanderSubsetPair {
  S: number[];
  T: number[];
  actualEdges: number;
  expectedEdges: number;
  deviation: number;
  bound: number;
}

// ─── Category Theory Types ───

export interface CategoryObject {
  label: string;
  x?: number;
  y?: number;
}

export interface CategoryMorphism {
  label: string;
  source: string;
  target: string;
  isIdentity: boolean;
}

export interface FunctorMapping {
  objectMap: Record<string, string>;
  morphismMap: Record<string, string>;
}

export interface NaturalTransformationComponent {
  object: string;        // Source category object label
  morphism: string;      // The component morphism label in the target category
  source: string;        // F(object) in target category
  target: string;        // G(object) in target category
}

export interface NaturalitySquareData {
  topLeft: string;       // F(A)
  topRight: string;      // G(A)
  bottomLeft: string;    // F(B)
  bottomRight: string;   // G(B)
  top: string;           // alpha_A
  bottom: string;        // alpha_B
  left: string;          // F(f)
  right: string;         // G(f)
  commutes: boolean;     // Whether the square commutes
}

// ─── Adjunction Types (Topic 3) ───

export interface AdjunctionData {
  leftAdjointName: string;       // Display name for F
  rightAdjointName: string;      // Display name for G
  sourceCategoryName: string;    // Display name for C
  targetCategoryName: string;    // Display name for D
  unitComponents: Map<string, string>;    // A -> eta_A morphism label
  counitComponents: Map<string, string>;  // B -> epsilon_B morphism label
}

export interface GaloisConnectionData {
  leftPosetElements: string[];
  rightPosetElements: string[];
  leftOrder: [string, string][];   // Covering relations in P
  rightOrder: [string, string][];  // Covering relations in Q
  leftMap: Map<string, string>;    // f: P -> Q (left adjoint)
  rightMap: Map<string, string>;   // g: Q -> P (right adjoint)
}

