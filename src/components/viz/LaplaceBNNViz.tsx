import BNNStaticFigure from './BNNStaticFigure';

export default function LaplaceBNNViz() {
  return (
    <BNNStaticFigure
      figurePath="/images/topics/bayesian-neural-networks/03_laplace_two_moons.png"
      alt="Three panels on Two Moons data: panel (a) Laplace-BNN predictive mean heatmap visually indistinguishable from the §1 point-estimate predictive; panel (b) Laplace predictive standard deviation heatmap with dark regions hugging the data and bright regions far from any training point; panel (c) twenty Laplace-sampled 0.5-probability decision boundaries overlaid on the data, narrow near the data and fanning out far from it."
      caption="Figure 3. The Laplace BNN. (a) The predictive mean is the point estimate — Laplace doesn't change accuracy. (b) The predictive standard deviation grows away from the data, recovering the §1 desideratum from a single trained model's local Gaussian. (c) Sampled decision boundaries fan out within one mode of the loss landscape, missing the multi-mode structure that §5 will recover."
      ariaLabel="Figure 3: Laplace BNN predictive mean, standard deviation, and sampled decision boundaries"
    />
  );
}
