### Experiment P: zero-shot TP recomputed under the published convention

The paper's Table 3 zero-shot TP panel treats transitions from unseen signs as NON-boundaries (notebook 03); cunei-tools treats them as boundaries. All zero-shot TP cells below use the published convention (the two agree within ~0.2 pt on full corpora; they diverge on small subsamples). Target-tuned TP cells elsewhere are unaffected.

| Model | → AKK | → ELX |
|---|---|---|
| TP on SUX@13K subsamples (5 seeds) | 0.947 ± 0.003 | 0.984 ± 0.005 |
| TP on SUX@8,831 subsamples (5 seeds) | 0.941 ± 0.007 | 0.974 ± 0.011 |
| TP on SUX full | 0.960 | 0.993 |
| TP on ELX-inventory-clamped SUX | 0.849 | 0.995 |

TP on ELX-expanded, zero-shot: →AKK 0.924 / →SUX 0.918 (θ=0.45).

All-boundaries trivial baseline (context for the G/I tables): akk: 0.961, sux: 0.968, elx: 0.994, elx25: 0.901.
