"""Experiment M: document-bootstrap 95% CIs for the NEW headline transfer
cells quoted in the rebuttal (Exp A and Exp C), matching the paper's Table 3
caption protocol (1,000 document resamples of the target corpus).

The draft currently reports seed-stds for these cells; a careful reviewer
could ask for the same CI treatment Table 3 has. Models are retrained at
seed 42 (the representative seed; seed-variance is reported separately).
"""
import json
import os
import random
import time

import numpy as np

import common
from common import (LANGS, get_corpora, lang_docs, train_morfessor,
                    doc_counts_morf, train_tp, tune_tp_theta, doc_counts_tp,
                    subsample_docs, micro_prf, save_json, save_text)

N_BOOT = 1000

t0 = time.time()
cache = get_corpora(verbose=False)
docs = {lang: lang_docs(cache["documents"], lang) for lang in LANGS}
with open(os.path.join(common.CACHE_DIR, "elx25_docs.json")) as f:
    elx25 = json.load(f)


def boot_ci(counts, seed=42):
    a = np.asarray(counts, dtype=np.int64)
    rng = np.random.RandomState(seed)
    f1s = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.randint(0, len(a), size=len(a))
        s = a[idx].sum(axis=0)
        p = s[0] / (s[0] + s[1]) if (s[0] + s[1]) else 0.0
        r = s[0] / (s[0] + s[2]) if (s[0] + s[2]) else 0.0
        f1s[i] = 2 * p * r / (p + r) if (p + r) else 0.0
    lo, hi = np.percentile(f1s, [2.5, 97.5])
    return float(lo), float(hi), float((hi - lo) / 2)


cells = []

# --- Exp A cells: SUX subsamples (seed 42) -> AKK, ELX ---
for budget in [13_000, 8_831]:
    subset, actual = subsample_docs(docs["sux"], budget, 42)
    random.seed(42)
    model, _ = train_morfessor(subset)
    for tgt in ["akk", "elx"]:
        t = time.time()
        counts = doc_counts_morf(model, docs[tgt])
        point = micro_prf(counts)["f1"]
        lo, hi, half = boot_ci(counts)
        cells.append({"cell": f"Morf SUX@{budget}->{ tgt.upper()}", "f1": point,
                      "ci_low": lo, "ci_high": hi, "ci_half": half})
        print(f"[{time.time()-t:6.1f}s] {cells[-1]['cell']}: {point:.4f} "
              f"[{lo:.4f}, {hi:.4f}] (±{half:.4f})", flush=True)

# --- Exp C cells: ELX-expanded (seed 42) -> AKK, SUX; Morf + TP zero-shot ---
random.seed(42)
morf25, _ = train_morfessor(elx25)
seg25 = train_tp(elx25, lang="elx")
src_theta = tune_tp_theta(seg25, elx25)["threshold"]
for tgt in ["akk", "sux"]:
    t = time.time()
    counts = doc_counts_morf(morf25, docs[tgt])
    point = micro_prf(counts)["f1"]
    lo, hi, half = boot_ci(counts)
    cells.append({"cell": f"Morf ELX25->{tgt.upper()}", "f1": point,
                  "ci_low": lo, "ci_high": hi, "ci_half": half})
    print(f"[{time.time()-t:6.1f}s] {cells[-1]['cell']}: {point:.4f} "
          f"[{lo:.4f}, {hi:.4f}] (±{half:.4f})", flush=True)
    t = time.time()
    tcounts = doc_counts_tp(seg25, docs[tgt], src_theta)
    tpoint = micro_prf(tcounts)["f1"]
    lo, hi, half = boot_ci(tcounts)
    cells.append({"cell": f"TP ELX25->{tgt.upper()} (zero-shot)", "f1": tpoint,
                  "ci_low": lo, "ci_high": hi, "ci_half": half})
    print(f"[{time.time()-t:6.1f}s] {cells[-1]['cell']}: {tpoint:.4f} "
          f"[{lo:.4f}, {hi:.4f}] (±{half:.4f})", flush=True)

out = {"experiment": "M: document-bootstrap CIs for new transfer cells",
       "protocol": f"{N_BOOT} document resamples of the target corpus, "
                   "seed-42 models (Table 3 caption protocol)",
       "cells": cells}
save_json(out, "expM_bootstrap_cis.json")

lines = ["### Experiment M: 95% document-bootstrap CIs for new transfer cells",
         "",
         "Same CI protocol as the paper's Table 3 caption (1,000 document "
         "resamples; seed-42 models; seed-variance reported separately in "
         "each experiment's own table).",
         "",
         "| Cell | F1 | 95% CI |", "|---|---|---|"]
for c in cells:
    lines.append(f"| {c['cell']} | {c['f1']:.3f} "
                 f"| [{c['ci_low']:.3f}, {c['ci_high']:.3f}] (±{c['ci_half']:.3f}) |")
save_text("\n".join(lines) + "\n", "expM_bootstrap_cis.md")
print(f"total {time.time()-t0:.1f}s")
