"""Experiment B (rebuttal Tier 1): in-language TP vs Morfessor significance.

For each language (AKK, SUX, ELX) run the paper's Table 2 protocol - 5-fold
document-level CV, KFold(shuffle=True, random_state=42) - for both methods on
identical folds, collect per-document boundary counts from the fold where each
document is held out, and test the F1 difference with a document-level paired
bootstrap (2,000 resamples, two-sided), i.e. the same significance machinery
the paper already uses for the classification experiments (Section 4).

TP protocol per fold (Table 2): train char-bigram TP on the 4 training folds,
tune theta on the held-out fold via cunei-tools' find_optimal_threshold
default coarse grid (0.05..0.95 step 0.05).
Morfessor per fold: corpusweight=1.0, batch training.
"""
import random
import time

import numpy as np
from sklearn.model_selection import KFold

from common import (LANGS, LANG_NAMES, N_BOOTSTRAP, get_corpora, lang_docs,
                    train_morfessor, doc_counts_morf, train_tp, tune_tp_theta,
                    doc_counts_tp, micro_prf, paired_bootstrap, save_json,
                    save_text)

t0 = time.time()
cache = get_corpora()
documents = cache["documents"]

random.seed(42)
np.random.seed(42)

results = {}
for lang in LANGS:
    docs = lang_docs(documents, lang)
    n = len(docs)
    print(f"\n--- {LANG_NAMES[lang]} ({n} docs) ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    tp_counts = [None] * n
    morf_counts = [None] * n
    fold_stats = []

    for fi, (tr_idx, te_idx) in enumerate(kf.split(docs)):
        t = time.time()
        train = [docs[i] for i in tr_idx]
        test = [docs[i] for i in te_idx]

        # TP (Table 2 protocol: theta tuned on the held-out fold, coarse grid)
        seg = train_tp(train, lang=lang)
        tp_metrics = tune_tp_theta(seg, test)
        theta = tp_metrics["threshold"]
        tpc = doc_counts_tp(seg, test, theta)

        # Morfessor (corpusweight=1.0)
        morf, _ = train_morfessor(train)
        mfc = doc_counts_morf(morf, test)

        assert len(tpc) == len(te_idx) and len(mfc) == len(te_idx)
        for j, di in enumerate(te_idx):
            tp_counts[di] = tpc[j]
            morf_counts[di] = mfc[j]

        fold_stats.append({
            "fold": fi + 1,
            "tp_f1": micro_prf(tpc)["f1"],
            "tp_theta": theta,
            "morf_f1": micro_prf(mfc)["f1"],
        })
        print(f"  fold {fi+1} [{time.time()-t:5.1f}s]: "
              f"TP F1={fold_stats[-1]['tp_f1']:.4f} (theta={theta:.2f}) | "
              f"Morf F1={fold_stats[-1]['morf_f1']:.4f}")

    assert all(c is not None for c in tp_counts)
    assert all(c is not None for c in morf_counts)

    pooled_tp = micro_prf(tp_counts)
    pooled_morf = micro_prf(morf_counts)
    boot = paired_bootstrap(tp_counts, morf_counts,
                            n_resamples=N_BOOTSTRAP, seed=42)

    tp_fold_f1s = [f["tp_f1"] for f in fold_stats]
    mf_fold_f1s = [f["morf_f1"] for f in fold_stats]
    results[lang] = {
        "n_docs": n,
        "folds": fold_stats,
        "tp_f1_foldmean": float(np.mean(tp_fold_f1s)),
        "tp_f1_foldstd": float(np.std(tp_fold_f1s)),
        "morf_f1_foldmean": float(np.mean(mf_fold_f1s)),
        "morf_f1_foldstd": float(np.std(mf_fold_f1s)),
        "tp_f1_pooled": pooled_tp["f1"],
        "morf_f1_pooled": pooled_morf["f1"],
        "diff_tp_minus_morf": boot["observed_diff"],
        "bootstrap": boot,
    }
    print(f"  {lang.upper()}: TP {results[lang]['tp_f1_foldmean']:.4f}"
          f"(±{results[lang]['tp_f1_foldstd']:.4f}) vs "
          f"Morf {results[lang]['morf_f1_foldmean']:.4f}"
          f"(±{results[lang]['morf_f1_foldstd']:.4f}) | "
          f"pooled diff {boot['observed_diff']:+.4f}, p={boot['p_two_sided']:.4f}")

out = {
    "experiment": "B: in-language TP vs Morfessor paired bootstrap",
    "protocol": {
        "cv": "5-fold document-level, KFold(shuffle=True, random_state=42)",
        "tp_theta": "tuned per held-out fold, cunei-tools coarse grid",
        "morfessor_corpusweight": 1.0,
        "bootstrap": f"document-level paired, {N_BOOTSTRAP} resamples, two-sided",
    },
    "results": results,
}
save_json(out, "expB_tp_vs_morfessor_significance.json")

def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"

lines = [
    "### Experiment B: In-language TP vs Morfessor, paired bootstrap",
    "",
    "5-fold document-level CV (identical folds for both methods); "
    "document-level paired bootstrap on the F1 difference, 2,000 resamples, "
    "two-sided (same significance machinery as the paper's classification "
    "experiments).",
    "",
    "| Language | TP F1 (CV) | Morfessor F1 (CV) | ΔF1 (TP−Morf) | 95% CI | p (two-sided) |",
    "|---|---|---|---|---|---|",
]
for lang in LANGS:
    r = results[lang]
    b = r["bootstrap"]
    lines.append(
        f"| {LANG_NAMES[lang]} (n={r['n_docs']}) "
        f"| {r['tp_f1_foldmean']:.3f} ± {r['tp_f1_foldstd']:.3f} "
        f"| {r['morf_f1_foldmean']:.3f} ± {r['morf_f1_foldstd']:.3f} "
        f"| {b['observed_diff']:+.4f} "
        f"| [{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}] "
        f"| {fmt_p(b['p_two_sided'])} |")
lines += [
    "",
    "F1 columns: mean ± std over the 5 CV folds (Table 2 style). ΔF1, CI and "
    "p are computed on the CV-pooled per-document counts.",
]
save_text("\n".join(lines) + "\n", "expB_tp_vs_morfessor_significance.md")
print(f"\ntotal {time.time()-t0:.1f}s")
