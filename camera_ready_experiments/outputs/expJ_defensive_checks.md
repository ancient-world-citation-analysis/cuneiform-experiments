### Experiment J: defensive checks (discussion-phase insurance)

**J1 — corpusweight sensitivity (superseded interpretation; see Exps K/L).**
Sweeping Morfessor's only hyperparameter for the ELX source and picking the
*target-oracle* best: →AKK peaks at 0.863 (cw=0.25) vs
0.304 at the published default (cw=1.0); the sweep is razor-peaked (0.060 at
cw≥2). Oracle target-side tuning is inadmissible in zero-shot transfer;
admissible source-side selection and the resulting full matrix are in
Exps K and L. The original headline written for this section ("corpusweight
cannot rescue the collapse") was wrong and is retracted.

**J2 — clamp specificity.** Clamping SUX to its own top-119-by-frequency
signs yields coverage 69.3% of AKK tokens and →AKK F1 = 0.238 ± 0.000;
the ELX-inventory clamp has coverage 74.2% and F1 = 0.283. Both matched-size
low-coverage clamps collapse, and their ordering follows coverage —
consistent with the coverage mediation (the planned "different coverage,
opposite outcome" contrast did not materialize because top-119 SUX signs
cover AKK *worse*, not better).

**J3 — TP under the ELX-inventory clamp.** TP trained on the clamped corpus
still transfers zero-shot at 0.961 (→AKK) / 0.995 (→ELX) under the
cunei-tools convention; see expP for the published-convention values.
