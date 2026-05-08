import BNNStaticFigure from './BNNStaticFigure';

export default function SGLDBNNViz() {
  return (
    <BNNStaticFigure
      figurePath="/images/topics/bayesian-neural-networks/06_sgld_two_moons.png"
      alt="Three panels: panel (a) SGLD predictive mean heatmap on Two Moons computed from 100 thinned post-burn-in samples; panel (b) SGLD predictive standard deviation heatmap, comparable to or larger than the deep-ensemble panel from §5; panel (c) trace plot of one weight component across the 1200 SGLD iterations, with the burn-in shaded and visible mixing oscillations after burn-in."
      caption="Figure 6. SGLD samples the posterior. (a) The predictive mean is the topic-invariant. (b) The predictive standard deviation captures multi-mode coverage like deep ensembles. (c) The trace plot is the practical mixing diagnostic — large fluctuations early signal exploration, smaller post-burn-in oscillations signal that the chain has settled into the typical set of the posterior."
      ariaLabel="Figure 6: SGLD predictive mean, standard deviation, and weight-component trace plot"
    />
  );
}
