"""Experiment O: paired-bootstrap significance for the ELX-expanded
in-language flip (Morfessor 0.993 vs TP 0.923, Exp C).

Dcaz (or the AE) may ask whether the expanded-corpus in-language difference
is significant. Same machinery as Exp B: identical 5-fold document-level CV
folds for both methods, per-document counts pooled across held-out folds,
document-level paired bootstrap, 2,000 resamples, two-sided.
"""
import json
import os
import random
import time

import numpy as np
from sklearn.model_selection import KFold

import common
from common import (N_BOOTSTRAP, train_morfessor, doc_counts_morf, train_tp,
                    tune_tp_theta, doc_counts_tp, micro_prf, paired_bootstrap,
                    save_json, save_text, get_corpora)

t0 = time.time()
get_corpora(verbose=False)  # ensure cache exists (not otherwise needed)
with open(os.path.join(common.CACHE_DIR, "elx25_docs.json")) as f:
    elx25 = json.load(f)

n = len(elx25)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
tp_counts = [None] * n
morf_counts = [None] * n
random.seed(42)

for fi, (tr, te) in enumerate(kf.split(elx25)):
    t = time.time()
    train = [elx25[i] for i in tr]
    test = [elx25[i] for i in te]
    seg = train_tp(train, lang="elx")
    theta = tune_tp_theta(seg, test)["threshold"]
    tpc = doc_counts_tp(seg, test, theta)
    model, _ = train_morfessor(train)
    mfc = doc_counts_morf(model, test)
    for j, di in enumerate(te):
        tp_counts[di] = tpc[j]
        morf_counts[di] = mfc[j]
    print(f"fold {fi+1} [{time.time()-t:5.1f}s]: TP={micro_prf(tpc)['f1']:.4f} "
          f"(th={theta:.2f}) Morf={micro_prf(mfc)['f1']:.4f}", flush=True)

boot = paired_bootstrap(tp_counts, morf_counts, n_resamples=N_BOOTSTRAP, seed=42)
out = {"experiment": "O: ELX-expanded in-language TP vs Morfessor significance",
       "n_docs": n,
       "tp_f1_pooled": micro_prf(tp_counts)["f1"],
       "morf_f1_pooled": micro_prf(morf_counts)["f1"],
       "bootstrap": boot}
save_json(out, "expO_elx25_significance.json")

p = boot["p_two_sided"]
lines = ["### Experiment O: ELX-expanded in-language significance",
         "",
         f"On the expanded corpus (n={n} docs), Morfessor "
         f"{out['morf_f1_pooled']:.3f} vs TP {out['tp_f1_pooled']:.3f}; "
         f"ΔF1 (TP−Morf) = {boot['observed_diff']:+.4f}, 95% CI "
         f"[{boot['ci95_low']:+.4f}, {boot['ci95_high']:+.4f}], "
         f"p {'< 0.001' if p < 0.001 else f'= {p:.3f}'} (document-level "
         "paired bootstrap, 2,000 resamples, two-sided)."]
save_text("\n".join(lines) + "\n", "expO_elx25_significance.md")
print(f"total {time.time()-t0:.1f}s")
