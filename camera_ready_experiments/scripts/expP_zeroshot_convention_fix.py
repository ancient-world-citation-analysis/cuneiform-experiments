"""Experiment P (audit fix, finding C0): recompute every zero-shot TP cell
under the PUBLISHED zero-shot convention.

The audit proved the paper's Table 3 zero-shot TP panel was produced by
notebook 03's evaluator, which guards `if uni > 0:` (unseen preceding sign
=> NO boundary), while common.doc_counts_tp follows cunei_tools (unseen =>
TP=0.0 => boundary). The two agree on full corpora (<1% unseen) but diverge
by 1-4 F1 points on small subsamples. All zero-shot TP numbers quoted next
to published Table 3 values must use the published (notebook-03) convention.

Recomputed here:
  - expA2 cells: TP on the Exp A SUX subsamples (2 budgets x 5 seeds),
    zero-shot to AKK and ELX (theta tuned on the source subsample).
  - expC cell:   TP on the expanded Elamite corpus, zero-shot to AKK / SUX.
  - expJ J3:     TP on ELX-inventory-clamped SUX, zero-shot to AKK / ELX.
  - Trivial all-boundaries baseline per target (audit findings C10/C11
    context; for prepared answers, not the draft).

Notebook-03 evaluation semantics, replicated exactly: skip docs with
len(words) < 2 or len(continuous) < 3; predict boundary at i iff the
preceding sign was seen in training AND bigram/unigram < theta.
"""
import json
import os
import random
import time
from collections import Counter

import numpy as np

import common
from common import (LANGS, SEEDS_5, get_corpora, lang_docs, subsample_docs,
                    save_json, save_text)


def train_counts(docs):
    uni, bi = Counter(), Counter()
    for doc in docs:
        chars = [c for c in doc if c != " "]
        for c in chars:
            uni[c] += 1
        for i in range(len(chars) - 1):
            bi[(chars[i], chars[i + 1])] += 1
    return uni, bi


def nb03_counts(uni, bi, test_docs, theta):
    """Per-doc (tp,fp,fn) under the published zero-shot convention."""
    out = []
    for doc in test_docs:
        words = doc.split()
        continuous = doc.replace(" ", "")
        if len(continuous) < 3 or len(words) < 2:
            continue
        gold = set()
        pos = 0
        for w in words:
            pos += len(w)
            gold.add(pos)
        gold.discard(len(continuous))
        pred = set()
        for i in range(1, len(continuous)):
            u = uni.get(continuous[i - 1], 0)
            if u > 0 and bi.get((continuous[i - 1], continuous[i]), 0) / u < theta:
                pred.add(i)
        out.append((len(gold & pred), len(pred - gold), len(gold - pred)))
    return out


def micro_f1(counts):
    tp = sum(c[0] for c in counts)
    fp = sum(c[1] for c in counts)
    fn = sum(c[2] for c in counts)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def best_theta_on(uni, bi, docs):
    best_f1, best_t = -1, 0.5
    for t in [i / 100 for i in range(5, 96, 5)]:
        f1 = micro_f1(nb03_counts(uni, bi, docs, t))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def singleton_counts(test_docs):
    out = []
    for doc in test_docs:
        words = doc.split()
        continuous = doc.replace(" ", "")
        if len(continuous) < 3 or len(words) < 2:
            continue
        gold = set()
        pos = 0
        for w in words:
            pos += len(w)
            gold.add(pos)
        gold.discard(len(continuous))
        pred = set(range(1, len(continuous)))
        out.append((len(gold & pred), len(pred - gold), len(gold - pred)))
    return out


t0 = time.time()
cache = get_corpora(verbose=False)
docs = {lang: lang_docs(cache["documents"], lang) for lang in LANGS}
with open(os.path.join(common.CACHE_DIR, "elx25_docs.json")) as f:
    elx25 = json.load(f)

results = {"convention": "published zero-shot (notebook 03: unseen sign -> no boundary)"}

# --- 1. Exp A2 subsample cells ---
a2 = {}
for budget in [13_000, 8_831]:
    rows = []
    for seed in SEEDS_5:
        subset, actual = subsample_docs(docs["sux"], budget, seed)
        uni, bi = train_counts(subset)
        theta, _ = best_theta_on(uni, bi, subset)
        row = {"seed": seed, "theta": theta, "actual_tokens": actual}
        for tgt in ["akk", "elx"]:
            row[f"f1_{tgt}"] = micro_f1(nb03_counts(uni, bi, docs[tgt], theta))
        rows.append(row)
        print(f"[{time.time()-t0:7.1f}s] A2 budget={budget} seed={seed} "
              f"th={theta:.2f}: akk={row['f1_akk']:.4f} elx={row['f1_elx']:.4f}",
              flush=True)
    a2[str(budget)] = {
        "rows": rows,
        **{f"f1_{t}_mean": float(np.mean([r[f"f1_{t}"] for r in rows]))
           for t in ["akk", "elx"]},
        **{f"f1_{t}_std": float(np.std([r[f"f1_{t}"] for r in rows]))
           for t in ["akk", "elx"]},
    }
results["expA2_nb03"] = a2

# --- 2. Full-SUX reference under same convention ---
uni, bi = train_counts(docs["sux"])
theta, _ = best_theta_on(uni, bi, docs["sux"])
ref = {"theta": theta,
       "f1_akk": micro_f1(nb03_counts(uni, bi, docs["akk"], theta)),
       "f1_elx": micro_f1(nb03_counts(uni, bi, docs["elx"], theta))}
results["sux_full_nb03"] = ref
print(f"SUX full (th={theta:.2f}): akk={ref['f1_akk']:.4f} elx={ref['f1_elx']:.4f}")

# --- 3. Exp C: ELX-expanded zero-shot ---
uni, bi = train_counts(elx25)
theta, _ = best_theta_on(uni, bi, elx25)
c = {"theta": theta,
     "f1_akk": micro_f1(nb03_counts(uni, bi, docs["akk"], theta)),
     "f1_sux": micro_f1(nb03_counts(uni, bi, docs["sux"], theta))}
results["elx25_nb03"] = c
print(f"ELX25 (th={theta:.2f}): akk={c['f1_akk']:.4f} sux={c['f1_sux']:.4f}")

# --- 4. Exp J3: clamped-SUX TP ---
elx_inventory = set(w for d in docs["elx"] for w in d.split() if w)
clamped = []
for d in docs["sux"]:
    kept = [w for w in d.split() if w in elx_inventory]
    if len(kept) > 2:
        clamped.append(" ".join(kept))
uni, bi = train_counts(clamped)
theta, _ = best_theta_on(uni, bi, clamped)
j3 = {"theta": theta,
      "f1_akk": micro_f1(nb03_counts(uni, bi, docs["akk"], theta)),
      "f1_elx": micro_f1(nb03_counts(uni, bi, docs["elx"], theta))}
results["j3_clamped_nb03"] = j3
print(f"J3 clamped (th={theta:.2f}): akk={j3['f1_akk']:.4f} elx={j3['f1_elx']:.4f}")

# --- 5. Trivial all-boundaries baseline per corpus ---
results["all_boundaries_baseline"] = {
    lang: micro_f1(singleton_counts(docs[lang])) for lang in LANGS}
results["all_boundaries_baseline"]["elx25"] = micro_f1(singleton_counts(elx25))
print("all-boundaries baselines:", results["all_boundaries_baseline"])

save_json(results, "expP_zeroshot_convention_fix.json")

a13, a88 = a2["13000"], a2["8831"]
lines = [
    "### Experiment P: zero-shot TP recomputed under the published convention",
    "",
    "The paper's Table 3 zero-shot TP panel treats transitions from unseen "
    "signs as NON-boundaries (notebook 03); cunei-tools treats them as "
    "boundaries. All zero-shot TP cells below use the published convention "
    "(the two agree within ~0.2 pt on full corpora; they diverge on small "
    "subsamples). Target-tuned TP cells elsewhere are unaffected.",
    "",
    "| Model | → AKK | → ELX |",
    "|---|---|---|",
    f"| TP on SUX@13K subsamples (5 seeds) | {a13['f1_akk_mean']:.3f} ± "
    f"{a13['f1_akk_std']:.3f} | {a13['f1_elx_mean']:.3f} ± {a13['f1_elx_std']:.3f} |",
    f"| TP on SUX@8,831 subsamples (5 seeds) | {a88['f1_akk_mean']:.3f} ± "
    f"{a88['f1_akk_std']:.3f} | {a88['f1_elx_mean']:.3f} ± {a88['f1_elx_std']:.3f} |",
    f"| TP on SUX full | {ref['f1_akk']:.3f} | {ref['f1_elx']:.3f} |",
    f"| TP on ELX-inventory-clamped SUX | {j3['f1_akk']:.3f} | {j3['f1_elx']:.3f} |",
    "",
    f"TP on ELX-expanded, zero-shot: →AKK {c['f1_akk']:.3f} / →SUX "
    f"{c['f1_sux']:.3f} (θ={c['theta']:.2f}).",
    "",
    "All-boundaries trivial baseline (context for the G/I tables): "
    + ", ".join(f"{k}: {v:.3f}"
                for k, v in results["all_boundaries_baseline"].items()) + ".",
]
save_text("\n".join(lines) + "\n", "expP_zeroshot_convention_fix.md")
print(f"total {time.time()-t0:.1f}s")
