import BNNStaticFigure from './BNNStaticFigure';

export default function DeepEnsembleViz() {
  return (
    <BNNStaticFigure
      figurePath="/images/topics/bayesian-neural-networks/05_deep_ensemble_two_moons.png"
      alt="Three panels on Two Moons data: panel (a) deep-ensemble predictive mean heatmap from K=10 trained models, visually similar to the per-method panel (a)s in §§1, 3, 4; panel (b) deep-ensemble predictive standard deviation heatmap, noticeably brighter off-distribution than §§3-4's panels (b); panel (c) all 10 ensemble members' 0.5-probability contours overlaid on the data in orange, fanning out wider than the §§3-4 sampled boundaries."
      caption="Figure 5. The deep ensemble. (a) The predictive mean is robust across methods. (b) The predictive standard deviation is larger off-distribution than for the single-mode methods, reflecting the function-space mode coverage that Laplace and MC-dropout miss. (c) Decision boundaries fan out wider than Laplace samples and MC-dropout samples — visual confirmation of multi-mode coverage."
      ariaLabel="Figure 5: K=10 deep ensemble predictive mean, standard deviation, and decision-boundary overlay"
    />
  );
}
