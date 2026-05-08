import BNNStaticFigure from './BNNStaticFigure';

export default function LossLandscapeModesViz() {
  return (
    <BNNStaticFigure
      figurePath="/images/topics/bayesian-neural-networks/02_loss_landscape_modes.png"
      alt="Two panels: panel (a) PCA scatter of 10 trained MLP weight vectors projected to their first two principal components, color-coded by final training loss with two distinct points labeled 'mode a' and 'mode b'; panel (b) the loss along a linear interpolation between mode a and mode b, rising from the trained-model floor to a peak barrier and back down — a non-convex ridge separating the two modes."
      caption="Figure 2. The loss landscape is genuinely multi-modal. (a) Ten MLPs trained from independent seeds, projected by PCA, cluster into discrete modes corresponding to permutation/scaling classes of the §2.4 hidden-unit symmetry. (b) Linearly interpolating between two modes passes through a region of strictly higher loss — modes are not connected by low-loss corridors, and a single Gaussian (Laplace, mean-field VI) cannot represent more than one mode."
      ariaLabel="Figure 2: PCA-projected loss-landscape modes and the barrier between them"
    />
  );
}
