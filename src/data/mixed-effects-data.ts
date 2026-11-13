// =============================================================================
// mixed-effects-data.ts
//
// Canonical six-classroom dataset for the mixed-effects topic, pinned as it is
// generated in the topic notebook (notebooks/mixed-effects/01_mixed_effects.ipynb,
// cell 4). The rows were extracted by replaying that cell's recipe verbatim:
//
//   rng = np.random.default_rng(20260429)
//   x         = rng.uniform(0.0, 5.0, size=60)          # drawn first
//   classroom = np.repeat(np.arange(6), [4, 6, 8, 10, 12, 20])
//   a_true    = rng.normal(0.0, 5.0, size=6)            # then the intercepts
//   eps       = rng.normal(0.0, 8.0, size=60)           # then the noise
//   y         = 50 + 5 * x + a_true[classroom] + eps
//
// The replay reproduces the notebook's printed fits (complete-pooling intercept
// 51.385, slope 4.300; no-pooling slope 4.019) and its §4 optima: ML
// (tau^2, sigma^2) = (2.139, 45.204), REML (3.298, 46.152). Mulberry32 in TS
// cannot reproduce NumPy's PCG64 stream, so components that must match the
// article's numbers read these rows instead of regenerating data.
// =============================================================================

export interface MixedEffectsRow {
  classroom: number; // 0..5, sorted; group sizes (4, 6, 8, 10, 12, 20)
  x: number;         // prep-hours, Uniform(0, 5)
  y: number;         // exam score
}

export const NOTEBOOK_SIX_CLASSROOM_ROWS: readonly MixedEffectsRow[] = [
  { classroom: 0, x: 1.6737967155288618, y: 49.951149021448536 },
  { classroom: 0, x: 2.0173788306418676, y: 60.13399921419729 },
  { classroom: 0, x: 2.181183561152557, y: 57.25158708193213 },
  { classroom: 0, x: 0.2390792994537061, y: 55.40423747080741 },
  { classroom: 1, x: 4.339080776621183, y: 60.82766535440727 },
  { classroom: 1, x: 3.193746511521473, y: 75.76708592915674 },
  { classroom: 1, x: 2.835862972397443, y: 58.14327998506975 },
  { classroom: 1, x: 0.04738814920264678, y: 42.83606713163698 },
  { classroom: 1, x: 1.172610639664572, y: 49.25289757994951 },
  { classroom: 1, x: 3.5611147229032616, y: 62.054983886930806 },
  { classroom: 2, x: 4.690948660737685, y: 82.04305444651763 },
  { classroom: 2, x: 2.055324731810021, y: 69.49228788282682 },
  { classroom: 2, x: 3.273255762295684, y: 63.55454520429542 },
  { classroom: 2, x: 2.0835186712165914, y: 41.43440716817227 },
  { classroom: 2, x: 1.2359602809441106, y: 52.84147058347416 },
  { classroom: 2, x: 0.6130443303000616, y: 54.32235072091082 },
  { classroom: 2, x: 3.717216435153769, y: 68.92458537788791 },
  { classroom: 2, x: 2.560523275119318, y: 70.322147256715 },
  { classroom: 3, x: 2.3187842525471676, y: 59.76176968084161 },
  { classroom: 3, x: 1.4045810885379035, y: 54.06666188285021 },
  { classroom: 3, x: 2.1151860861710228, y: 59.408720930291814 },
  { classroom: 3, x: 1.9899754044328373, y: 60.27107103671281 },
  { classroom: 3, x: 0.13554248087666865, y: 47.97015954797983 },
  { classroom: 3, x: 1.296179287758306, y: 50.79935007230502 },
  { classroom: 3, x: 3.674788691013933, y: 65.2196025624859 },
  { classroom: 3, x: 2.921676072304513, y: 56.83903017181959 },
  { classroom: 3, x: 1.4101600207457348, y: 62.25739575123241 },
  { classroom: 3, x: 1.2290885728182765, y: 48.18032569140706 },
  { classroom: 4, x: 3.4217902114253014, y: 65.70686360472823 },
  { classroom: 4, x: 3.18589387415799, y: 53.21584488975052 },
  { classroom: 4, x: 2.4953701752909616, y: 64.88474279280894 },
  { classroom: 4, x: 0.510081231992377, y: 52.6145116924525 },
  { classroom: 4, x: 2.0640260079416755, y: 64.94144169126582 },
  { classroom: 4, x: 3.4887646842970015, y: 71.80712454060617 },
  { classroom: 4, x: 0.04510016382870241, y: 54.4175631657809 },
  { classroom: 4, x: 1.8457954638674012, y: 67.67362033127574 },
  { classroom: 4, x: 1.5014756040901556, y: 57.499768706833336 },
  { classroom: 4, x: 3.023407267005526, y: 49.198686138050476 },
  { classroom: 4, x: 1.004059727842403, y: 57.448748255649555 },
  { classroom: 4, x: 4.024585255653478, y: 73.39646800743665 },
  { classroom: 5, x: 1.3457669011506879, y: 69.45445038084738 },
  { classroom: 5, x: 3.939912451203447, y: 76.27751261177357 },
  { classroom: 5, x: 3.278776546364799, y: 56.9683181972065 },
  { classroom: 5, x: 3.635580418936538, y: 63.245045266594786 },
  { classroom: 5, x: 1.622680904888068, y: 65.16595316638437 },
  { classroom: 5, x: 0.8719105209288636, y: 64.96097294818367 },
  { classroom: 5, x: 2.745970938030395, y: 65.56241319976017 },
  { classroom: 5, x: 1.3333188576046284, y: 71.89756497074133 },
  { classroom: 5, x: 4.220110914391536, y: 63.75861635909554 },
  { classroom: 5, x: 0.3245710001705654, y: 50.01665528832249 },
  { classroom: 5, x: 2.0920041343432985, y: 66.53995307594472 },
  { classroom: 5, x: 4.88957153700854, y: 75.20537127205031 },
  { classroom: 5, x: 2.637317235142813, y: 70.36745444038897 },
  { classroom: 5, x: 0.5045775049867884, y: 50.761223466078036 },
  { classroom: 5, x: 2.3211526605819626, y: 62.395306430267944 },
  { classroom: 5, x: 4.106068226799833, y: 63.10776692711125 },
  { classroom: 5, x: 2.426026635594216, y: 68.33253996878051 },
  { classroom: 5, x: 4.821931958713078, y: 71.6266446228277 },
  { classroom: 5, x: 0.1927304872088509, y: 50.79753442258288 },
  { classroom: 5, x: 2.818678318987102, y: 72.46012676452251 },
];
