import BNNStaticFigure from './BNNStaticFigure';

export default function SGHMCBNNViz() {
  return (
    <BNNStaticFigure
      figurePath="/images/topics/bayesian-neural-networks/07_sghmc_two_moons.png"
      alt="Three panels: panel (a) SGHMC predictive mean heatmap on Two Moons; panel (b) SGHMC predictive standard deviation heatmap, comparable to or slightly larger than the SGLD panel from §6; panel (c) autocorrelation function of one weight component for SGLD and SGHMC side-by-side, showing SGHMC's autocorrelation decaying faster than SGLD's."
      caption="Figure 7. SGHMC and the momentum-induced mixing speedup. (a, b) The predictive distribution is similar to SGLD's at the same iteration budget. (c) The autocorrelation function decays faster for SGHMC than SGLD — the visual signature of why momentum helps: each effective sample takes fewer iterations of wall-clock to produce."
      ariaLabel="Figure 7: SGHMC predictive distribution and SGLD-vs-SGHMC autocorrelation comparison"
    />
  );
}
