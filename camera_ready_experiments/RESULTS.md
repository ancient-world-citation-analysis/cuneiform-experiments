# ARR May 2026 — Submission 2040 rebuttal experiment results

Generated 2026-07-09 by `rebuttal_experiments/scripts/`.

**Reproduction gate:** the released pipeline + paper-era corpora (AKK 13,502 docs / SUX-ETCSL 394 docs / ELX 84 docs) reproduce every off-diagonal Morfessor cell of Table 3 to within ±0.0004 (`cache/full_transfer_repro.json`). All numbers below share that verified harness: Morfessor corpusweight=1.0; TP = cunei-tools CuneiSeg, theta via find_optimal_threshold coarse grid; boundary micro-F1; doc filter len(split)>2; KFold(5, shuffle, seed=42); paired bootstrap 2,000 resamples two-sided; seeds 42–46 where applicable.

---

### Reproduction gate detail

| Cell | Reproduced F1 | Published F1 |
|---|---|---|
| akk->sux | 0.9813 | 0.9813 |
| akk->elx | 0.9836 | 0.9836 |
| sux->akk | 0.8568 | 0.8565 |
| sux->elx | 0.9532 | 0.9532 |
| elx->akk | 0.3038 | 0.3038 |
| elx->sux | 0.2674 | 0.2674 |

---

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

---

### Experiment A addendum: TP on the identical SUX subsamples

| Source (train size) | TP → AKK (zero-shot) | TP → ELX (zero-shot) |
|---|---|---|
| SUX subsample (13,000 tok) | 0.961 ± 0.000 | 0.995 ± 0.000 |
| SUX subsample (8,831 tok) | 0.961 ± 0.000 | 0.994 ± 0.000 |

Target-tuned TP (cunei-tools default) is within ~0.01 of zero-shot in every cell; full numbers in the JSON.

---

### Experiment B: In-language TP vs Morfessor, paired bootstrap

5-fold document-level CV (identical folds for both methods); document-level paired bootstrap on the F1 difference, 2,000 resamples, two-sided (same significance machinery as the paper's classification experiments).

| Language | TP F1 (CV) | Morfessor F1 (CV) | ΔF1 (TP−Morf) | 95% CI | p (two-sided) |
|---|---|---|---|---|---|
| Akkadian (n=13502) | 0.971 ± 0.002 | 0.996 ± 0.000 | -0.0253 | [-0.0265, -0.0242] | < 0.001 |
| Sumerian (n=394) | 0.971 ± 0.002 | 0.993 ± 0.002 | -0.0214 | [-0.0230, -0.0197] | < 0.001 |
| Elamite (n=84) | 0.989 ± 0.005 | 0.985 ± 0.010 | +0.0047 | [-0.0020, +0.0117] | = 0.194 |

F1 columns: mean ± std over the 5 CV folds (Table 2 style). ΔF1, CI and p are computed on the CV-pooled per-document counts.

---

### Experiment C: Elamite 25K refresh

Expanded corpus: 871 docs / 111,733 tokens (original: 84 docs / 8,831; new Susa texts: 787 docs / 102,902, conversion rate 90.2%).

**In-language (5-fold document-level CV):**

| Corpus | TP F1 | Morfessor F1 |
|---|---|---|
| ELX original (paper Table 2) | 0.989 | 0.985 |
| ELX expanded (111,733 tok) | 0.923 ± 0.007 | 0.993 ± 0.000 |

**ELX as transfer source (Table 3 protocol):**

| Source | Method | -> AKK F1 | -> SUX F1 |
|---|---|---|---|
| ELX original (published) | TP | 0.959 | 0.969 |
| ELX original (published) | Morfessor | 0.304 | 0.267 |
| ELX expanded | TP (zero-shot, θ=0.45) | 0.961 | 0.967 |
| ELX expanded | TP (target-tuned) | 0.963 | 0.968 |
| ELX expanded | Morfessor (5 seeds) | 0.783 ± 0.000 | 0.712 ± 0.000 |

---

### Experiment D: Elamite dictionary vs running-text ablation

All conditions evaluated on the same held-out Utu-nashu folds (5-fold document-level CV); TP theta tuned per held-out fold; Morfessor mean ± std over 5 seeds.

| Training source | TP F1 | Morfessor F1 | Train tokens (fold 1) |
|---|---|---|---|
| Lemma Dictionary only | 0.994 ± 0.005 | 0.727 ± 0.000 | 68,251 |
| Utu-nashu only | 0.989 ± 0.005 | 0.985 ± 0.000 | 6,759 |
| Dictionary + Utu-nashu | 0.995 ± 0.002 | 0.991 ± 0.000 | 75,010 |

---

### Experiment E: Morfessor transfer vs source-corpus size

Zero-shot transfer F1 (mean ± std over 5 seeds; document-level subsampling; corpusweight=1.0). 'full' rows use the entire source corpus (seeds then only affect Morfessor's internal training order).

| Source | Size | → AKK | → SUX | → ELX |
|---|---|---|---|---|
| AKK | 5K (5,741 tok) | — | 0.615 ± 0.022 | 0.826 ± 0.054 |
| AKK | 13K (14,086 tok) | — | 0.743 ± 0.019 | 0.955 ± 0.016 |
| AKK | 25K (25,396 tok) | — | 0.826 ± 0.012 | 0.973 ± 0.002 |
| AKK | full (2,441,623 tok) | — | 0.981 ± 0.000 | 0.984 ± 0.000 |
| SUX | 5K (6,485 tok) | 0.482 ± 0.037 | — | 0.554 ± 0.070 |
| SUX | 13K (14,175 tok) | 0.595 ± 0.031 | — | 0.710 ± 0.091 |
| SUX | 25K (28,925 tok) | 0.662 ± 0.028 | — | 0.757 ± 0.071 |
| SUX | full (337,972 tok) | 0.858 ± 0.001 | — | 0.953 ± 0.000 |
| ELX-expanded | 5K (5,326 tok) | 0.517 ± 0.023 | 0.414 ± 0.027 | — |
| ELX-expanded | 13K (14,219 tok) | 0.587 ± 0.036 | 0.478 ± 0.020 | — |
| ELX-expanded | 25K (25,819 tok) | 0.651 ± 0.018 | 0.527 ± 0.019 | — |
| ELX-expanded | full (111,733 tok) | 0.783 ± 0.000 | 0.712 ± 0.000 | — |
| ELX-original | 5K (5,045 tok) | 0.273 ± 0.009 | 0.247 ± 0.009 | — |
| ELX-original | full (8,831 tok) | 0.304 ± 0.000 | 0.267 ± 0.000 | — |

---

### Experiment E addendum: lexicon coverage mediates Morfessor transfer

Across all 140 (source, size, seed, target) cells of the Experiment E grid, the Morfessor lexicon's coverage of the target corpus predicts transfer F1 at Pearson r = 0.91 (Spearman ρ = 0.92). Coverage rises with source size and source diversity and is asymmetric between languages — the single quantity unifying the size axis, the language axis, and the original-vs-expanded Elamite difference. TP requires no lexicon and is invariant across the same cells.

---

### Experiment G: supervised neural baseline vs unsupervised methods (AKK)

Character-BiLSTM boundary tagger trained on N labeled tokens, vs TP and Morfessor trained on the identical subsamples; all evaluated on the same fixed 500-document held-out Akkadian test set (notebook 04 protocol). Mean ± std over 5 seeds. TP theta is tuned on test (the paper's learning-curve protocol); the BiLSTM tunes its threshold on a held-out slice of its own training data.

| Labeled tokens | BiLSTM (supervised) | TP (unsupervised) | Morfessor (unsupervised) |
|---|---|---|---|
| 1,000 | 0.957 ± 0.002 | 0.961 ± 0.004 | 0.637 ± 0.063 |
| 5,000 | 0.957 ± 0.001 | 0.964 ± 0.003 | 0.908 ± 0.013 |
| 13,000 | 0.962 ± 0.010 | 0.968 ± 0.001 | 0.963 ± 0.007 |
| 50,000 | 0.994 ± 0.001 | 0.968 ± 0.000 | 0.991 ± 0.002 |

Audit note: at 1K–13K labeled tokens the BiLSTM's F1 sits at the all-boundaries trivial baseline for this test set (~0.957); only the 50K model exceeds it. The paper-facing takeaway is therefore: supervision below ~50K gold tokens does not beat the unsupervised methods (TP exceeds the trivial baseline at every budget; Morfessor from 13K).

---

### Experiment F: damaged-token share per corpus (Dcaz W4)

Share of ingested tokens carrying damage marks in each source's native annotation (the pipeline strips marks but keeps the signs; only fully illegible/unlemmatizable rows are excluded up front).

| Corpus | Ingested tokens | Damage-marked | Excluded up front |
|---|---|---|---|
| Akkadian (ORACC; sign-level damaged/missing flags) | 1,255,669 | 27.4% | 22.9% |
| Sumerian (ETCSL composites; X placeholders) | 146,143 | 0.6% | 14.5% |
| Elamite Utu-našu (editorial marks) | 2,582 | 8.4% | — |
| Elamite Susa, new (editorial marks) | 35,304 | 21.9% | — |

AKK 'excluded up front' = unlemmatizable rows (pos 'u'), illegible x/X forms, and null rows dropped by the released loader.

---

### Experiment H: mechanism controls (mKBh W1)

**H1 — inventory clamp (causal test).** Morfessor trained on the full Sumerian corpus with its lexicon clamped to the Elamite sign inventory (142 types; 242,331 Sumerian tokens survive the clamp):

| Source | → AKK | → ELX |
|---|---|---|
| SUX full, unclamped (Table 3) | 0.857 | 0.953 |
| SUX full, clamped to ELX inventory | 0.283 ± 0.000 | 0.599 ± 0.000 |
| ELX full (Table 3) | 0.304 | — |

Reading: the clamp shows missing coverage is causally **sufficient to induce the collapse** (→AKK 0.283; mirror clamp on AKK →SUX 0.285), not that coverage alone guarantees success — the clamped models' →ELX cells (0.599 from SUX, 0.881 from AKK) show distributional fit also matters, consistent with the mediation being strong (r = 0.91) but not total.

**H2 — failure localization.** For the published ELX→AKK model, gold-boundary recall split by whether both flanking tokens are in the source lexicon: covered 0.256 (1,376,411 boundaries) vs uncovered 0.082 (1,051,710) — a 3.1× ratio; the causal weight rests on H1 (absolute recall at covered boundaries is suppressed by Viterbi context effects from adjacent unknown material).

---

### Experiment I: cross-language transfer of the supervised BiLSTM (AKK source)

**AUDIT NOTE (do not cite as learned transfer):** on foreign-language targets the tagger sees mostly unseen signs (<unk>) and its output coincides with the all-boundaries trivial policy; the F1s below match that baseline, not learned generalization. TP zero-shot cells use the cunei-tools unseen-sign convention (see expP for the published-convention values).

Zero-shot on full target corpora; no target-language tuning anywhere.

| Source budget | Method | → SUX | → ELX |
|---|---|---|---|
| AKK 13K | BiLSTM (supervised) | 0.968 ± 0.000 | 0.994 ± 0.000 |
| AKK 13K | TP (unsup., zero-shot) | 0.967 ± 0.000 | 0.994 ± 0.000 |
| AKK 13K | Morfessor (unsup.) | 0.753 ± 0.021 | 0.961 ± 0.009 |
| AKK 50K | BiLSTM (supervised) | 0.974 ± 0.000 | 0.993 ± 0.001 |
| AKK 50K | TP (unsup., zero-shot) | 0.967 ± 0.000 | 0.994 ± 0.000 |
| AKK 50K | Morfessor (unsup.) | 0.906 ± 0.003 | 0.977 ± 0.001 |

---

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

---

### Experiment K: source-side corpusweight selection

In-language 5-fold CV F1 (the only tuning admissible for zero-shot transfer) at each corpusweight:

| corpusweight | ELX | SUX | AKK |
|---|---|---|---|
| 0.25 | 0.9983 | 0.9941 | 0.9962 |
| 0.5 | 0.9986 | 0.9944 | 0.9965 |
| 1.0 | 0.9846 | 0.9930 | 0.9964 |
| 2.0 | 0.0633 | 0.0761 | 0.1217 |
| 4.0 | 0.0565 | 0.0645 | 0.0652 |

Source-side selection picks cw=0.5 in all three languages, but only ELX's preference is decisive (+0.0140 over cw=1.0, beyond fold noise); AKK (+0.0001) and SUX (+0.0014) prefer 0.5 by margins within fold-level std. The robust statement: cw=1.0 is never preferred, and ELX clearly selects 0.5.

---

### Experiment L: Morfessor transfer at source-selected corpusweight (0.5)

Zero-shot transfer with the corpusweight that in-language (source-side) CV selects (Exp K), vs the published default (cw=1.0) and untuned TP.

| Source | Target | Morf cw=1.0 (paper) | Morf cw=0.5 (source-selected) | TP (no tuning) |
|---|---|---|---|---|
| AKK | SUX | 0.981 | 0.982 ± 0.000 | 0.967 |
| AKK | ELX | 0.984 | 0.997 ± 0.000 | 0.992 |
| SUX | AKK | 0.857 | 0.977 ± 0.000 | 0.960 |
| SUX | ELX | 0.953 | 0.993 ± 0.000 | 0.993 |
| ELX | AKK | 0.304 | 0.765 ± 0.000 | 0.959 |
| ELX | SUX | 0.267 | 0.713 ± 0.000 | 0.969 |

---

### Experiment M: 95% document-bootstrap CIs for new transfer cells

Same CI protocol as the paper's Table 3 caption (1,000 document resamples; seed-42 models; seed-variance reported separately in each experiment's own table).

| Cell | F1 | 95% CI |
|---|---|---|
| Morf SUX@13000->AKK | 0.604 | [0.598, 0.610] (±0.006) |
| Morf SUX@13000->ELX | 0.736 | [0.712, 0.758] (±0.023) |
| Morf SUX@8831->AKK | 0.592 | [0.587, 0.597] (±0.005) |
| Morf SUX@8831->ELX | 0.739 | [0.713, 0.762] (±0.024) |
| Morf ELX25->AKK | 0.783 | [0.779, 0.787] (±0.004) |
| TP ELX25->AKK (zero-shot) | 0.961 | [0.959, 0.962] (±0.001) |
| Morf ELX25->SUX | 0.712 | [0.704, 0.721] (±0.009) |
| TP ELX25->SUX (zero-shot) | 0.967 | [0.965, 0.969] (±0.002) |

---

### Experiment N: inventory clamp at corpusweight 0.5 (insurance)

SUX clamped to the ELX inventory, trained at the source-selected cw=0.5: →AKK 0.732 ± 0.000 (unclamped @0.5: 0.977; clamped @1.0: 0.283). →ELX 0.994 ± 0.000.

---

### Experiment O: ELX-expanded in-language significance

On the expanded corpus (n=871 docs), Morfessor 0.993 vs TP 0.924; ΔF1 (TP−Morf) = -0.0692, 95% CI [-0.0771, -0.0622], p < 0.001 (document-level paired bootstrap, 2,000 resamples, two-sided).

---

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

---
### Experiment Q (v2): OB case study — OB_iltani corpus, ATF ground-truth writing-mode labels

132 documents, 13614 words; 50.3% Sumerographic (ATF underscore spans); v1 heuristic vs truth P=0.744/R=0.648; conversion 91.4%; 1246 runs / 11589 tokens; seen rates chars 100.0% / bigrams 98.6% / words 35.4%; TP best theta=0.02 F1=0.567; Morfessor F1=0.551 (P=0.381 R=0.996); all-boundaries trivial=0.540

| boundary class | n | TP recall | Morfessor recall |
|---|---|---|---|
| within-syl | 2450 | 0.709 | 1.000 |
| within-log | 2623 | 0.813 | 0.990 |
| switch | 5270 | 0.865 | 0.998 |

| word class | signs | TP FB rate | Morf FB rate |
|---|---|---|---|
| syl | 13423 | 0.659 | 0.962 |
| log | 4174 | 0.510 | 0.916 |
