"""Experiment N (private insurance): does the inventory-clamp causal result
survive at the source-selected corpusweight (0.5)?

The H1 clamp ran at the published default (cw=1.0). If a discussion-phase
exchange ever combines the clamp with the corpusweight question, we must
know whether the causal story is cw-robust: train Morfessor at cw=0.5 on
the ELX-inventory-clamped Sumerian corpus, transfer to AKK.

Interpretation guide:
  - If clamped@0.5 -> AKK stays far below unclamped@0.5 (0.977, Exp L),
    the coverage mechanism is cw-robust and the clamp defense holds at any
    admissible setting.
  - If it recovers toward 0.977, the clamp result is cw=1.0-specific and
    the discussion answer must stay scoped to the published protocol.
"""
import random
import time

import numpy as np

from common import (LANGS, get_corpora, lang_docs, train_morfessor,
                    evaluate_morfessor, save_json, save_text)

t0 = time.time()
cache = get_corpora(verbose=False)
docs = {lang: lang_docs(cache["documents"], lang) for lang in LANGS}
elx_inventory = set(w for d in docs["elx"] for w in d.split() if w)

clamped = []
for d in docs["sux"]:
    kept = [w for w in d.split() if w in elx_inventory]
    if len(kept) > 2:
        clamped.append(" ".join(kept))

per_tgt = {"akk": [], "elx": []}
for seed in [42, 43, 44]:
    t = time.time()
    random.seed(seed)
    model, _ = train_morfessor(clamped, corpusweight=0.5)
    for tgt in per_tgt:
        per_tgt[tgt].append(evaluate_morfessor(model, docs[tgt])["f1"])
    print(f"[{time.time()-t:6.1f}s] seed={seed}: "
          f"akk={per_tgt['akk'][-1]:.4f} elx={per_tgt['elx'][-1]:.4f}",
          flush=True)

out = {"experiment": "N: inventory clamp at source-selected corpusweight 0.5",
       "results": {f"f1_{t}_mean": float(np.mean(v)) for t, v in per_tgt.items()}
       | {f"f1_{t}_std": float(np.std(v)) for t, v in per_tgt.items()},
       "reference": {"clamped_cw1.0->akk": 0.2825,
                     "unclamped_cw0.5->akk (Exp L)": 0.9767,
                     "unclamped_cw1.0->akk": 0.8568}}
save_json(out, "expN_clamp_cw05.json")

m = out["results"]
lines = ["### Experiment N: inventory clamp at corpusweight 0.5 (insurance)",
         "",
         f"SUX clamped to the ELX inventory, trained at the source-selected "
         f"cw=0.5: →AKK {m['f1_akk_mean']:.3f} ± {m['f1_akk_std']:.3f} "
         f"(unclamped @0.5: 0.977; clamped @1.0: 0.283). "
         f"→ELX {m['f1_elx_mean']:.3f} ± {m['f1_elx_std']:.3f}."]
save_text("\n".join(lines) + "\n", "expN_clamp_cw05.md")
print(f"total {time.time()-t0:.1f}s")
