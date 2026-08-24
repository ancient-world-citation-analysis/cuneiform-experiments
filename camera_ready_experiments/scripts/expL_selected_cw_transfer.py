"""Experiment L: the full transfer matrix at SOURCE-SELECTED corpusweight.

Exp K showed in-language (source-side) selection picks corpusweight 0.5
(not the published default 1.0) for every language. This experiment re-runs
the Table 3 Morfessor off-diagonals with cw = 0.5 so the honest,
protocol-robust version of the transfer story has complete numbers.

3 seeds per cell (training-order randomness only).
"""
import random
import time

import numpy as np

from common import (LANGS, get_corpora, lang_docs, train_morfessor,
                    evaluate_morfessor, save_json, save_text)

CW = 0.5
SEEDS = [42, 43, 44]

t0 = time.time()
cache = get_corpora(verbose=False)
docs = {lang: lang_docs(cache["documents"], lang) for lang in LANGS}

results = {}
for src in LANGS:
    per_tgt = {t: [] for t in LANGS if t != src}
    for seed in SEEDS:
        t = time.time()
        random.seed(seed)
        model, _ = train_morfessor(docs[src], corpusweight=CW)
        for tgt in per_tgt:
            per_tgt[tgt].append(evaluate_morfessor(model, docs[tgt])["f1"])
        print(f"[{time.time()-t:6.1f}s] {src} cw={CW} seed={seed}: " +
              " ".join(f"{t}:{per_tgt[t][-1]:.4f}" for t in per_tgt),
              flush=True)
    results[src] = {f"f1_{t}_mean": float(np.mean(v))
                    for t, v in per_tgt.items()}
    results[src].update({f"f1_{t}_std": float(np.std(v))
                         for t, v in per_tgt.items()})

published_cw1 = {"akk": {"sux": 0.9813, "elx": 0.9836},
                 "sux": {"akk": 0.8565, "elx": 0.9532},
                 "elx": {"akk": 0.3038, "sux": 0.2674}}
tp_reference = {"akk": {"sux": 0.967, "elx": 0.992},
                "sux": {"akk": 0.960, "elx": 0.993},
                "elx": {"akk": 0.959, "sux": 0.969}}

out = {"experiment": "L: Morfessor transfer at source-selected corpusweight 0.5",
       "corpusweight": CW, "seeds": SEEDS, "results": results,
       "published_cw1_reference": published_cw1}
save_json(out, "expL_selected_cw_transfer.json")

lines = [
    "### Experiment L: Morfessor transfer at source-selected corpusweight (0.5)",
    "",
    "Zero-shot transfer with the corpusweight that in-language (source-side) "
    "CV selects (Exp K), vs the published default (cw=1.0) and untuned TP.",
    "",
    "| Source | Target | Morf cw=1.0 (paper) | Morf cw=0.5 (source-selected) | TP (no tuning) |",
    "|---|---|---|---|---|",
]
for src in LANGS:
    for tgt in LANGS:
        if src == tgt:
            continue
        m = results[src]
        lines.append(f"| {src.upper()} | {tgt.upper()} "
                     f"| {published_cw1[src][tgt]:.3f} "
                     f"| {m[f'f1_{tgt}_mean']:.3f} ± {m[f'f1_{tgt}_std']:.3f} "
                     f"| {tp_reference[src][tgt]:.3f} |")
save_text("\n".join(lines) + "\n", "expL_selected_cw_transfer.md")
print(f"total {time.time()-t0:.1f}s")
