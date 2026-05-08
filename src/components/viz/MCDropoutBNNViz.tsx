import BNNStaticFigure from './BNNStaticFigure';

export default function MCDropoutBNNViz() {
  return (
    <BNNStaticFigure
      figurePath="/images/topics/bayesian-neural-networks/04_mcdropout_two_moons.png"
      alt="Three panels on Two Moons data mirroring §3's panel layout: panel (a) MC-dropout predictive mean heatmap visually similar to the point-estimate; panel (b) MC-dropout predictive standard deviation heatmap, qualitatively similar to §3 Laplace but with flatter off-distribution variance; panel (c) twenty MC-dropout sampled decision boundaries overlaid in green, with narrower fanning-out off-distribution than the Laplace samples in §3."
      caption="Figure 4. MC-dropout. (a) The predictive mean is the point estimate. (b) The predictive standard deviation captures epistemic uncertainty qualitatively but saturates off-distribution — the Foong et al. limitation. (c) Sampled boundaries fan out narrower than Laplace's, concrete visual confirmation of the lower-epistemic-variance regime the Bernoulli family produces."
      ariaLabel="Figure 4: MC-dropout BNN predictive mean, standard deviation, and sampled decision boundaries"
    />
  );
}
