"""Experiment K: which corpusweight does LEGITIMATE (source-side) model
selection choose?

Exp J1 shows target-oracle corpusweight tuning would lift ELX->AKK from
0.304 to ~0.86 - but oracle target tuning is inadmissible in zero-shot
transfer. The admissible selection is in-language (source-side) CV. This
experiment runs the paper's Table 2 protocol (5-fold document-level CV,
KFold seed 42) for each corpusweight in {0.25, 0.5, 1.0, 2.0, 4.0} on each
language, reporting which cw source-side selection picks.

If ELX in-language selects cw = 1.0 (Morfessor's published default), then
under every admissible protocol the ELX-source transfer collapse stands,
and the J1 oracle number becomes a *defense* ("even an oracle reaches only
0.86, still below TP's untuned 0.96 - and no admissible selection finds it").
"""
import random
import time

import numpy as np
from sklearn.model_selection import KFold

from common import (LANGS, get_corpora, lang_docs, train_morfessor,
                    doc_counts_morf, micro_prf, save_json, save_text)

CWS = [0.25, 0.5, 1.0, 2.0, 4.0]

t0 = time.time()
cache = get_corpora(verbose=False)
docs_all = {lang: lang_docs(cache["documents"], lang) for lang in LANGS}

results = {}
for lang in ["elx", "sux", "akk"]:
    docs = docs_all[lang]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(docs))
    results[lang] = {}
    for cw in CWS:
        t = time.time()
        random.seed(42)
        fold_f1s = []
        for tr, te in folds:
            model, _ = train_morfessor([docs[i] for i in tr], corpusweight=cw)
            fold_f1s.append(micro_prf(
                doc_counts_morf(model, [docs[i] for i in te]))["f1"])
        results[lang][str(cw)] = {"f1_mean": float(np.mean(fold_f1s)),
                                  "f1_std": float(np.std(fold_f1s))}
        print(f"[{time.time()-t:6.1f}s] {lang} cw={cw}: "
              f"{np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}", flush=True)
    best = max(CWS, key=lambda c: results[lang][str(c)]["f1_mean"])
    results[lang]["selected_cw"] = best
    print(f"  => {lang} source-side selection picks cw={best}")

out = {"experiment": "K: source-side corpusweight selection (Table 2 CV)",
       "corpusweights": CWS, "results": results}
save_json(out, "expK_cw_source_selection.json")

lines = ["### Experiment K: source-side corpusweight selection",
         "",
         "In-language 5-fold CV F1 (the only tuning admissible for zero-shot "
         "transfer) at each corpusweight:",
         "",
         "| corpusweight | ELX | SUX | AKK |",
         "|---|---|---|---|"]
for cw in CWS:
    row = f"| {cw} |"
    for lang in ["elx", "sux", "akk"]:
        r = results[lang][str(cw)]
        row += f" {r['f1_mean']:.4f} |"
    lines.append(row)
lines += ["",
          "Source-side selection picks: " +
          ", ".join(f"{l.upper()}: cw={results[l]['selected_cw']}"
                    for l in ["elx", "sux", "akk"]) + "."]
save_text("\n".join(lines) + "\n", "expK_cw_source_selection.md")
print(f"total {time.time()-t0:.1f}s")
