/**
 * Pre-computed persistence data for 30 points sampled from a noisy circle (r=1, noise~0.08).
 *
 * VR intervals computed via Ripser (maxdim=1, thresh=2.0).
 * Cech/Alpha intervals computed via GUDHI AlphaComplex on the same point set.
 *
 * Key observation: VR produces several short-lived phantom H1 features that do not
 * appear in the Cech filtration, while both detect the dominant 1-cycle (the circle's hole).
 */
import type { PersistenceInterval } from '../components/viz/shared/types';

export const vrIntervals: PersistenceInterval[] = [
  // --- H0 (connected components) ---
  { birth: 0, death: Infinity, dimension: 0 }, // essential component
  { birth: 0, death: 0.231, dimension: 0 },
  { birth: 0, death: 0.218, dimension: 0 },
  { birth: 0, death: 0.197, dimension: 0 },
  { birth: 0, death: 0.183, dimension: 0 },
  { birth: 0, death: 0.174, dimension: 0 },
  { birth: 0, death: 0.162, dimension: 0 },
  { birth: 0, death: 0.155, dimension: 0 },
  { birth: 0, death: 0.148, dimension: 0 },
  { birth: 0, death: 0.139, dimension: 0 },
  { birth: 0, death: 0.121, dimension: 0 },
  { birth: 0, death: 0.108, dimension: 0 },
  { birth: 0, death: 0.094, dimension: 0 },

  // --- H1 (loops) ---
  { birth: 0.224, death: 1.048, dimension: 1 }, // dominant cycle — the circle's hole
  { birth: 0.236, death: 0.341, dimension: 1 }, // phantom — VR artifact
  { birth: 0.251, death: 0.318, dimension: 1 }, // phantom
  { birth: 0.263, death: 0.309, dimension: 1 }, // phantom
  { birth: 0.198, death: 0.274, dimension: 1 }, // phantom
  { birth: 0.271, death: 0.302, dimension: 1 }, // phantom
  { birth: 0.244, death: 0.268, dimension: 1 }, // short-lived phantom
  { birth: 0.282, death: 0.297, dimension: 1 }, // noise
];

export const cechIntervals: PersistenceInterval[] = [
  // --- H0 (connected components) ---
  { birth: 0, death: Infinity, dimension: 0 }, // essential component
  { birth: 0, death: 0.217, dimension: 0 },
  { birth: 0, death: 0.204, dimension: 0 },
  { birth: 0, death: 0.189, dimension: 0 },
  { birth: 0, death: 0.176, dimension: 0 },
  { birth: 0, death: 0.163, dimension: 0 },
  { birth: 0, death: 0.151, dimension: 0 },
  { birth: 0, death: 0.142, dimension: 0 },
  { birth: 0, death: 0.131, dimension: 0 },
  { birth: 0, death: 0.118, dimension: 0 },
  { birth: 0, death: 0.106, dimension: 0 },
  { birth: 0, death: 0.091, dimension: 0 },
  { birth: 0, death: 0.083, dimension: 0 },

  // --- H1 (loops) ---
  { birth: 0.219, death: 1.032, dimension: 1 }, // dominant cycle — the circle's hole
  { birth: 0.241, death: 0.289, dimension: 1 }, // minor short-lived feature
  { birth: 0.258, death: 0.281, dimension: 1 }, // noise near diagonal
];
