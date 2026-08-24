"""Experiment I (rebuttal, mKBh W2 completion): does the supervised neural
tagger transfer across languages the way TP does?

Train the Exp G char-BiLSTM on Akkadian subsamples (13K: 5 seeds; 50K:
3 seeds - the budget where it matches Morfessor in-language) and evaluate
ZERO-SHOT on the full Sumerian and Elamite corpora (no target tuning; the
decision threshold is the one tuned on the source-side validation slice;
unseen signs map to <unk>). TP (theta tuned on the source subsample) and
Morfessor are trained on the identical subsamples for comparison.

Same subsample pool as Exp G (notebook 04 protocol: seed-42 shuffle,
500-doc test set removed).
"""
import random
import time

import numpy as np

from common import (SEEDS_5, get_corpora, lang_docs, subsample_docs,
                    train_morfessor, evaluate_morfessor, train_tp,
                    tune_tp_theta, doc_counts_tp, micro_prf,
                    save_json, save_text)
from expG_neural_supervised import (doc_to_example, train_bilstm,
                                    make_batches, eval_tagger)

CONFIGS = [(13_000, SEEDS_5), (50_000, [42, 43, 44])]

t0 = time.time()
cache = get_corpora(verbose=False)
documents = cache["documents"]

random.seed(42)
docs_akk = [d for d in documents["akk"]["unicode"].values()
            if d and len(d.split()) > 2]
random.shuffle(docs_akk)
train_pool = docs_akk[500:]

targets = {lang: lang_docs(documents, lang) for lang in ["sux", "elx"]}
target_examples = {lang: [doc_to_example(d) for d in t]
                   for lang, t in targets.items()}

runs = []
for budget, seeds in CONFIGS:
    for seed in seeds:
        t = time.time()
        subset, actual = subsample_docs(train_pool, budget, seed)

        model, vocab, thresh, epochs = train_bilstm(subset, seed)
        row = {"budget": budget, "seed": seed, "actual_tokens": actual,
               "bilstm_threshold": thresh, "bilstm_epochs": epochs}
        for lang in targets:
            batches = make_batches(target_examples[lang], vocab)
            m = eval_tagger(model, batches, thresh)
            row[f"bilstm_f1_{lang}"] = m["f1"]

        seg = train_tp(subset, lang="akk")
        src_theta = tune_tp_theta(seg, subset)["threshold"]
        for lang in targets:
            row[f"tp_f1_{lang}"] = micro_prf(
                doc_counts_tp(seg, targets[lang], src_theta))["f1"]
        row["tp_src_theta"] = src_theta

        random.seed(seed)
        morf, _ = train_morfessor(subset)
        for lang in targets:
            row[f"morf_f1_{lang}"] = evaluate_morfessor(morf, targets[lang])["f1"]

        runs.append(row)
        print(f"[{time.time()-t:7.1f}s] budget={budget:,} seed={seed}: "
              f"BiLSTM sux={row['bilstm_f1_sux']:.4f} elx={row['bilstm_f1_elx']:.4f} | "
              f"TP sux={row['tp_f1_sux']:.4f} elx={row['tp_f1_elx']:.4f} | "
              f"Morf sux={row['morf_f1_sux']:.4f} elx={row['morf_f1_elx']:.4f}",
              flush=True)

summary = {}
for budget, seeds in CONFIGS:
    rows = [r for r in runs if r["budget"] == budget]
    s = {"n_seeds": len(rows)}
    for k in ["bilstm_f1_sux", "bilstm_f1_elx", "tp_f1_sux", "tp_f1_elx",
              "morf_f1_sux", "morf_f1_elx"]:
        vals = [r[k] for r in rows]
        s[f"{k}_mean"] = float(np.mean(vals))
        s[f"{k}_std"] = float(np.std(vals))
    summary[str(budget)] = s

out = {"experiment": "I: zero-shot cross-language transfer of the supervised "
                     "BiLSTM vs TP vs Morfessor (AKK source)",
       "protocol": {"pool": "notebook 04 train pool (test split removed)",
                    "zero_shot": "no target tuning; BiLSTM threshold from "
                                 "source val slice; TP theta from source",
                    "configs": [{"budget": b, "seeds": s} for b, s in CONFIGS]},
       "runs": runs, "summary": summary}
save_json(out, "expI_bilstm_transfer.json")

lines = [
    "### Experiment I: cross-language transfer of the supervised BiLSTM (AKK source)",
    "",
    "Zero-shot on full target corpora; no target-language tuning anywhere.",
    "",
    "| Source budget | Method | → SUX | → ELX |",
    "|---|---|---|---|",
]
for budget, _ in CONFIGS:
    s = summary[str(budget)]
    lines += [
        f"| AKK {budget//1000}K | BiLSTM (supervised) "
        f"| {s['bilstm_f1_sux_mean']:.3f} ± {s['bilstm_f1_sux_std']:.3f} "
        f"| {s['bilstm_f1_elx_mean']:.3f} ± {s['bilstm_f1_elx_std']:.3f} |",
        f"| AKK {budget//1000}K | TP (unsup., zero-shot) "
        f"| {s['tp_f1_sux_mean']:.3f} ± {s['tp_f1_sux_std']:.3f} "
        f"| {s['tp_f1_elx_mean']:.3f} ± {s['tp_f1_elx_std']:.3f} |",
        f"| AKK {budget//1000}K | Morfessor (unsup.) "
        f"| {s['morf_f1_sux_mean']:.3f} ± {s['morf_f1_sux_std']:.3f} "
        f"| {s['morf_f1_elx_mean']:.3f} ± {s['morf_f1_elx_std']:.3f} |",
    ]
save_text("\n".join(lines) + "\n", "expI_bilstm_transfer.md")
print(f"total {time.time()-t0:.1f}s")
