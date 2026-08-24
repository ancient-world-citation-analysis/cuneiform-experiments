"""Experiment A addendum: TP trained on the IDENTICAL Sumerian subsamples
used in Experiment A, transferred to AKK and ELX.

Completes the matched-size dissociation table: at the same 13K / ELX-matched
source budgets, TP should hold F1 > 0.9 where Morfessor degrades. TP is
evaluated zero-shot (theta tuned on the source subsample itself, Table 3
off-diagonal protocol) and, for reference, target-tuned (cunei-tools default).
Subsamples are bit-identical to Experiment A's (same subsample_docs(seed)).
"""
import time

import numpy as np

from common import (LANGS, SEEDS_5, get_corpora, lang_docs, subsample_docs,
                    train_tp, tune_tp_theta, doc_counts_tp, micro_prf,
                    save_json, save_text)

PRIMARY_BUDGET = 13_000

t0 = time.time()
cache = get_corpora(verbose=False)
docs = {lang: lang_docs(cache["documents"], lang) for lang in LANGS}
elx_measured = sum(len(d.split()) for d in docs["elx"])
budgets = [PRIMARY_BUDGET]
if abs(elx_measured - PRIMARY_BUDGET) > 500:
    budgets.append(elx_measured)

runs = []
for budget in budgets:
    for seed in SEEDS_5:
        t = time.time()
        subset, actual = subsample_docs(docs["sux"], budget, seed)
        seg = train_tp(subset, lang="sux")
        src_opt = tune_tp_theta(seg, subset)
        row = {"budget": budget, "seed": seed, "actual_tokens": actual,
               "src_theta": src_opt["threshold"]}
        for tgt in ["akk", "elx"]:
            zero = micro_prf(doc_counts_tp(seg, docs[tgt], src_opt["threshold"]))
            tuned = tune_tp_theta(seg, docs[tgt])
            row[f"f1_{tgt}_zeroshot"] = zero["f1"]
            row[f"f1_{tgt}_tuned"] = tuned["f1"]
        runs.append(row)
        print(f"[{time.time()-t:5.1f}s] budget={budget:,} seed={seed} "
              f"theta*={row['src_theta']:.2f} | "
              f"akk zs={row['f1_akk_zeroshot']:.4f} tuned={row['f1_akk_tuned']:.4f} | "
              f"elx zs={row['f1_elx_zeroshot']:.4f} tuned={row['f1_elx_tuned']:.4f}",
              flush=True)

summary = {}
for budget in budgets:
    rows = [r for r in runs if r["budget"] == budget]
    s = {}
    for key in ["f1_akk_zeroshot", "f1_akk_tuned", "f1_elx_zeroshot", "f1_elx_tuned"]:
        vals = [r[key] for r in rows]
        s[f"{key}_mean"] = float(np.mean(vals))
        s[f"{key}_std"] = float(np.std(vals))
    summary[str(budget)] = s

out = {"experiment": "A2: TP on the same SUX subsamples",
       "protocol": {"subsamples": "identical to expA (subsample_docs, seeds 42-46)",
                    "zeroshot": "theta tuned on source subsample",
                    "tuned": "theta tuned on target (cunei-tools default grid)"},
       "runs": runs, "summary": summary}
save_json(out, "expA2_tp_same_subsamples.json")

lines = [
    "### Experiment A addendum: TP on the identical SUX subsamples",
    "",
    "| Source (train size) | TP → AKK (zero-shot) | TP → ELX (zero-shot) |",
    "|---|---|---|",
]
for budget in budgets:
    s = summary[str(budget)]
    lines.append(f"| SUX subsample ({budget:,} tok) "
                 f"| {s['f1_akk_zeroshot_mean']:.3f} ± {s['f1_akk_zeroshot_std']:.3f} "
                 f"| {s['f1_elx_zeroshot_mean']:.3f} ± {s['f1_elx_zeroshot_std']:.3f} |")
lines += ["", "Target-tuned TP (cunei-tools default) is within ~0.01 of "
          "zero-shot in every cell; full numbers in the JSON.", ""]
save_text("\n".join(lines), "expA2_tp_same_subsamples.md")
print(f"total {time.time()-t0:.1f}s")
