### Experiment H: mechanism controls (mKBh W1)

**H1 — inventory clamp (causal test).** Morfessor trained on the full Sumerian corpus with its lexicon clamped to the Elamite sign inventory (142 types; 242,331 Sumerian tokens survive the clamp):

| Source | → AKK | → ELX |
|---|---|---|
| SUX full, unclamped (Table 3) | 0.857 | 0.953 |
| SUX full, clamped to ELX inventory | 0.283 ± 0.000 | 0.599 ± 0.000 |
| ELX full (Table 3) | 0.304 | — |

Reading: the clamp shows missing coverage is causally **sufficient to induce the collapse** (→AKK 0.283; mirror clamp on AKK →SUX 0.285), not that coverage alone guarantees success — the clamped models' →ELX cells (0.599 from SUX, 0.881 from AKK) show distributional fit also matters, consistent with the mediation being strong (r = 0.91) but not total.

**H2 — failure localization.** For the published ELX→AKK model, gold-boundary recall split by whether both flanking tokens are in the source lexicon: covered 0.256 (1,376,411 boundaries) vs uncovered 0.082 (1,051,710) — a 3.1× ratio; the causal weight rests on H1 (absolute recall at covered boundaries is suppressed by Viterbi context effects from adjacent unknown material).
