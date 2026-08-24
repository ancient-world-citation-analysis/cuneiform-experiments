### Experiment A: Sumerian-subsample control (Reviewer mKBh, Weakness 1)

Morfessor and TP trained on Sumerian subsampled to the Elamite training
budget (document-level shuffled-prefix subsampling per the paper's
learning-curve protocol; Morfessor corpusweight=1.0; 5 seeds, mean ± std;
population std as in Table 2). Evaluation: zero-shot on the full target
corpora — identical protocol to Table 3 off-diagonals. Two budgets:
13,000 tokens (the paper's quoted ELX size) and 8,831 tokens (the measured
ELX running-text training size, exact match to what the ELX source saw).

| Source (train size) | Morfessor → AKK | Morfessor → ELX | Morfessor → SUX | TP → AKK (zero-shot) | TP → ELX (zero-shot) |
|---|---|---|---|---|---|
| SUX @ 13K (5 seeds) | **0.595 ± 0.031** | **0.710 ± 0.091** | — | 0.961 ± 0.000 | 0.995 ± 0.000 |
| SUX @ 8,831 = ELX-matched (5 seeds) | **0.542 ± 0.051** | **0.637 ± 0.077** | — | 0.961 ± 0.000 | 0.994 ± 0.000 |
| SUX full, 338K (Table 3; reproduced 0.857) | 0.857 | 0.953 | — | 0.960 | 0.993 |
| ELX full, 8,831 tok (Table 3) | 0.304 | — | 0.267 | 0.959 | — |

Diagnostics (mean over seeds): the SUX-13K lexicon has ~447 sign types and
covers 91.7% of AKK and 96.9% of ELX target tokens; the full ELX lexicon has
142 types and covers 74.2% of AKK / 71.7% of SUX tokens (Table 3 companion
JSON). Morfessor precision stays ≥ 0.97 in every cell; the drop is entirely
recall (0.43 on AKK at 13K), i.e. under-segmentation from lexicon gaps.

**Reading for the rebuttal (honest-engagement branch of the plan):**

1. The reviewer's requested cell — Morfessor trained on ~13K Sumerian,
   tested on Elamite — yields **0.710 ± 0.091**, far above the ELX→SUX
   collapse (0.267). Corpus size alone therefore does NOT reproduce the
   Elamite-source collapse.
2. Size does matter for Morfessor (0.857 → 0.542 on →AKK when SUX shrinks
   from 338K to the ELX-matched budget) — but at *identical* training size
   and *identical* target, the Elamite source is still **24 F1 points below
   the Sumerian source** (0.304 vs 0.542 on →AKK). Lexicon content, not
   sample size, accounts for the residual collapse; the asymmetric
   vocabulary coverage above (96.9% vs 71.7%) gives the mechanism.
3. The paper's central dissociation is unchanged and sharpened: on the same
   subsamples TP is flat at 0.961/0.994 with zero seed variance, while
   Morfessor ranges 0.30–0.86 depending on source size and language. The
   camera-ready should soften "rather than any pure sample-efficiency
   confound" to "over and above a sample-size effect" and cite this control.
