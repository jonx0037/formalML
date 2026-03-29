import type * as d3 from 'd3';

/**
 * Shared D3 type utilities for force simulation visualizations.
 *
 * The D3 force simulation mutates nodes in place, adding x, y, vx, vy.
 * These types model that augmented state so we can replace `any` casts
 * in simulation callbacks.
 */

/** A simulation node: your data type T plus the mutable position fields D3 adds. */
export type SimNode<T = Record<string, unknown>> = d3.SimulationNodeDatum & T;

/** A simulation link with resolved (not string) source/target after simulation ticks. */
export interface SimLink<
  N extends d3.SimulationNodeDatum = d3.SimulationNodeDatum,
> {
  source: N;
  target: N;
  index?: number;
}

/** Convenience: a simulation link that also carries extra edge data. */
export type SimLinkDatum<
  E = Record<string, unknown>,
  N extends d3.SimulationNodeDatum = d3.SimulationNodeDatum,
> = SimLink<N> & E;
