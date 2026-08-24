"""Experiment D (rebuttal Tier 2): Elamite dictionary vs running-text ablation.

The paper's Elamite data has two components (Table 1 "Dict.+Corpus"):
the Elamite Lemma Dictionary (~13K clean-Unicode entries) and the Utu-nashu
running-text corpus (84 documents after filtering). This ablation trains
TP and Morfessor on three source conditions and evaluates all three on the
SAME held-out running-text folds, so numbers are directly comparable:

  dict_only  : all clean dictionary entry forms (constant across folds)
  utu_only   : the 4 training folds of the Utu-nashu corpus (classic CV)
  combined   : dictionary + the 4 training folds

Test side is always the held-out Utu-nashu fold (5-fold document-level CV,
KFold(shuffle=True, random_state=42)). TP theta tuned on the held-out fold
(Table 2 protocol, cunei-tools coarse grid); Morfessor corpusweight=1.0 with
seed-controlled training (5 seeds x 5 folds).
"""
import random
import time

import numpy as np
from sklearn.model_selection import KFold

from common import (SEEDS_5, get_corpora, lang_docs, train_morfessor,
                    doc_counts_morf, train_tp, tune_tp_theta, doc_counts_tp,
                    micro_prf, save_json, save_text)

t0 = time.time()
cache = get_corpora()
documents = cache["documents"]

nasu = lang_docs(documents, "elx")
dict_df = cache["elx_dict_df"]
dict_entries = [u for u in dict_df["form_unicode"].astype(str).values
                if u and u.strip()]
print(f"Utu-nashu: {len(nasu)} docs, {sum(len(d.split()) for d in nasu):,} tokens")
print(f"Dictionary: {len(dict_entries)} clean entries, "
      f"{sum(len(d.split()) for d in dict_entries):,} tokens")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(kf.split(nasu))

conditions = ["dict_only", "utu_only", "combined"]


def train_docs_for(cond, tr_idx):
    if cond == "dict_only":
        return dict_entries
    if cond == "utu_only":
        return [nasu[i] for i in tr_idx]
    return dict_entries + [nasu[i] for i in tr_idx]


results = {}
for cond in conditions:
    print(f"\n--- condition: {cond} ---")
    tp_fold_f1s, tp_thetas = [], []
    for fi, (tr, te) in enumerate(folds):
        test = [nasu[i] for i in te]
        seg = train_tp(train_docs_for(cond, tr), lang="elx")
        m = tune_tp_theta(seg, test)
        tp_fold_f1s.append(m["f1"])
        tp_thetas.append(m["threshold"])
    print(f"  TP: {np.mean(tp_fold_f1s):.4f} ± {np.std(tp_fold_f1s):.4f} "
          f"(thetas {tp_thetas})")

    morf_seed_means = []
    for seed in SEEDS_5:
        random.seed(seed)
        fold_f1s = []
        for tr, te in folds:
            test = [nasu[i] for i in te]
            model, _ = train_morfessor(train_docs_for(cond, tr))
            fold_f1s.append(micro_prf(doc_counts_morf(model, test))["f1"])
        morf_seed_means.append(float(np.mean(fold_f1s)))
    print(f"  Morfessor: {np.mean(morf_seed_means):.4f} ± "
          f"{np.std(morf_seed_means):.4f} over seeds")

    results[cond] = {
        "tp_f1_mean": float(np.mean(tp_fold_f1s)),
        "tp_f1_std": float(np.std(tp_fold_f1s)),
        "tp_thetas": tp_thetas,
        "morf_f1_mean": float(np.mean(morf_seed_means)),
        "morf_f1_std": float(np.std(morf_seed_means)),
        "morf_seed_means": morf_seed_means,
        "n_train_tokens_fold1": int(sum(len(d.split())
                                        for d in train_docs_for(cond, folds[0][0]))),
    }

out = {
    "experiment": "D: Elamite dictionary vs running-text ablation",
    "protocol": {
        "test": "held-out Utu-nashu folds (5-fold document-level CV, seed 42), "
                "identical across conditions",
        "tp": "theta tuned on held-out fold, cunei-tools coarse grid",
        "morfessor": "corpusweight=1.0, 5 seeds x 5 folds",
    },
    "corpus": {
        "nasu_docs": len(nasu),
        "nasu_tokens": int(sum(len(d.split()) for d in nasu)),
        "dict_entries": len(dict_entries),
        "dict_tokens": int(sum(len(d.split()) for d in dict_entries)),
    },
    "results": results,
}
save_json(out, "expD_elx_source_ablation.json")

lines = [
    "### Experiment D: Elamite dictionary vs running-text ablation",
    "",
    "All conditions evaluated on the same held-out Utu-nashu folds "
    "(5-fold document-level CV); TP theta tuned per held-out fold; "
    "Morfessor mean ± std over 5 seeds.",
    "",
    "| Training source | TP F1 | Morfessor F1 | Train tokens (fold 1) |",
    "|---|---|---|---|",
]
labels = {"dict_only": "Lemma Dictionary only",
          "utu_only": "Utu-nashu only",
          "combined": "Dictionary + Utu-nashu"}
for cond in conditions:
    r = results[cond]
    lines.append(f"| {labels[cond]} | {r['tp_f1_mean']:.3f} ± {r['tp_f1_std']:.3f} "
                 f"| {r['morf_f1_mean']:.3f} ± {r['morf_f1_std']:.3f} "
                 f"| {r['n_train_tokens_fold1']:,} |")
save_text("\n".join(lines) + "\n", "expD_elx_source_ablation.md")
print(f"\ntotal {time.time()-t0:.1f}s")
