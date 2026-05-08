import BNNStaticFigure from './BNNStaticFigure';

export default function BNNPredictiveMotivationViz() {
  return (
    <BNNStaticFigure
      figurePath="/images/topics/bayesian-neural-networks/01_point_vs_bayesian_predictive.png"
      alt="Three panels on Two Moons data: panel (a) point-estimate predicted-probability heatmap with a sharp confident decision surface that is correct near the data but arbitrary far from it; panel (b) five independently-trained MLPs' 0.5-probability contours overlaid as red lines, agreeing tightly near the data and fanning out far from any training point; panel (c) a viridis heatmap of predictive variance computed across the five MLP predictions, dark near the data and bright in the off-distribution corners."
      caption="Figure 1. The ensemble preview. (a) A single trained MLP confidently predicts everywhere — including regions far from the training data. (b) Five MLPs trained from different seeds agree on the data and disagree off it — disagreement among independently-trained models is itself a kind of uncertainty quantification. (c) The variance over the input space recovers the desideratum: the model is uncertain where it lacks data."
      ariaLabel="Figure 1: Point-estimate vs. Bayesian-ensemble predictive on Two Moons"
    />
  );
}
