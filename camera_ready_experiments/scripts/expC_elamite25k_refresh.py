"""Experiment C (rebuttal Tier 1): Elamite 25K refresh.

Convert the newly collated Elamite running-text documents
(newelamitedata.csv, Susa texts; one document per row) with the exact
released Nasu conversion pipeline (_setup.load_elamite's convert_nasu path:
normalize -> sign replacement -> final unmatched pass), merge with the
existing Utu-nashu corpus, and re-run:

  1. In-language 5-fold document-level CV (TP + Morfessor) on the expanded
     corpus, compared with the original-corpus numbers (Table 2 protocol).
  2. ELX-as-source cross-language transfer to AKK and SUX (Table 3 protocol),
     TP reported both zero-shot (source-optimal theta) and target-tuned,
     Morfessor zero-shot, compared with the original ELX row.

Seeds: CV split fixed at random_state=42 (paper protocol). Morfessor's
internal epoch shuffle is seed-controlled; in-language CV and transfer
training are repeated over 5 seeds (42-46) and reported mean +/- std where
training randomness applies (Morfessor). TP training is deterministic.
"""
import os
import re
import random
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import common
from common import (LANGS, SEEDS_5, get_corpora, lang_docs, train_morfessor,
                    doc_counts_morf, evaluate_morfessor, train_tp,
                    tune_tp_theta, doc_counts_tp, micro_prf, vocab_overlap,
                    save_json, save_text)

NEW_DATA = os.path.join(common.REPO, "newelamitedata.csv")

t0 = time.time()
cache = get_corpora()
documents = cache["documents"]
sign_dict = cache["sign_dict"]

from _setup import _normalize_for_unmatched_pass  # released helper


def replace_with_unicode(text):
    return " ".join([sign_dict.get(s, s) for s in str(text).split()])


def final_pass(unicode_text):
    txt = _normalize_for_unmatched_pass(unicode_text)
    out, still = [], []
    for tok in txt.split():
        if tok == "md" and "m" in sign_dict and "d" in sign_dict and "md" not in sign_dict:
            out.append(sign_dict["m"])
            out.append(sign_dict["d"])
            continue
        if tok in sign_dict:
            out.append(sign_dict[tok])
        else:
            out.append(tok)
            if re.search(r"[A-Za-z]", tok):
                still.append(tok)
    return " ".join(out), still


def convert_doc(translit):
    """Identical to _setup.load_elamite's convert_nasu, applied to a whole
    document line (words are space-separated in the raw transliteration)."""
    if pd.isna(translit) or str(translit).strip() == "":
        return "", []
    clean = _normalize_for_unmatched_pass(translit)
    return final_pass(replace_with_unicode(clean))


# ---------------------------------------------------------------------------
# 1. Convert the new documents
# ---------------------------------------------------------------------------
new_df = pd.read_csv(NEW_DATA)
new_df = new_df.drop_duplicates(subset="id")
new_df = new_df[new_df["transliteration"].notna()]

new_docs = {}
unmatched_tokens = 0
total_tokens = 0
for _, row in new_df.iterrows():
    uni, still = convert_doc(row["transliteration"])
    toks = uni.split()
    if len(toks) > 2:  # same filter as lang_docs
        new_docs[f"susa_{row['id']}"] = uni
        total_tokens += len(toks)
        unmatched_tokens += sum(1 for t in toks if re.search(r"[A-Za-z]", t))

old_docs = lang_docs(documents, "elx")
old_tokens = sum(len(d.split()) for d in old_docs)
elx25 = old_docs + list(new_docs.values())
elx25_tokens = sum(len(d.split()) for d in elx25)
conv_rate = 1 - unmatched_tokens / total_tokens if total_tokens else 0.0

print(f"new docs: {len(new_docs)} ({total_tokens:,} tokens, "
      f"conversion rate {conv_rate:.1%})")
print(f"ELX original: {len(old_docs)} docs / {old_tokens:,} tokens")
print(f"ELX expanded: {len(elx25)} docs / {elx25_tokens:,} tokens")

corpus_stats = {
    "new_docs": len(new_docs), "new_tokens": int(total_tokens),
    "new_unconverted_token_rate": float(1 - conv_rate),
    "old_docs": len(old_docs), "old_tokens": int(old_tokens),
    "expanded_docs": len(elx25), "expanded_tokens": int(elx25_tokens),
}

# persist the expanded corpus for downstream experiments (expE size grid)
import json
with open(os.path.join(common.CACHE_DIR, "elx25_docs.json"), "w") as f:
    json.dump(elx25, f, ensure_ascii=False)

# ---------------------------------------------------------------------------
# 2. In-language 5-fold CV on the expanded corpus (Table 2 protocol)
# ---------------------------------------------------------------------------
print("\n--- In-language 5-fold CV on ELX-25K ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(kf.split(elx25))

tp_fold_f1s, tp_thetas = [], []
for fi, (tr, te) in enumerate(folds):
    seg = train_tp([elx25[i] for i in tr], lang="elx")
    m = tune_tp_theta(seg, [elx25[i] for i in te])
    tp_fold_f1s.append(m["f1"])
    tp_thetas.append(m["threshold"])
    print(f"  TP fold {fi+1}: F1={m['f1']:.4f} (theta={m['threshold']:.2f})")

morf_cv_by_seed = []
for seed in SEEDS_5:
    random.seed(seed)
    fold_f1s = []
    for tr, te in folds:
        model, _ = train_morfessor([elx25[i] for i in tr])
        fold_f1s.append(micro_prf(doc_counts_morf(model, [elx25[i] for i in te]))["f1"])
    morf_cv_by_seed.append(float(np.mean(fold_f1s)))
    print(f"  Morfessor seed {seed}: CV mean F1={morf_cv_by_seed[-1]:.4f}")

in_language = {
    "tp_f1_mean": float(np.mean(tp_fold_f1s)),
    "tp_f1_std": float(np.std(tp_fold_f1s)),
    "tp_thetas": tp_thetas,
    "morf_f1_mean_over_seeds": float(np.mean(morf_cv_by_seed)),
    "morf_f1_std_over_seeds": float(np.std(morf_cv_by_seed)),
    "morf_cv_by_seed": morf_cv_by_seed,
}

# ---------------------------------------------------------------------------
# 3. ELX-25K as transfer source (Table 3 protocol)
# ---------------------------------------------------------------------------
print("\n--- ELX-25K as source: transfer to AKK / SUX ---")
akk_docs = lang_docs(documents, "akk")
sux_docs = lang_docs(documents, "sux")
targets = {"akk": akk_docs, "sux": sux_docs}

# TP: train on full expanded corpus (deterministic)
seg25 = train_tp(elx25, lang="elx")
src_opt = tune_tp_theta(seg25, elx25)  # source-optimal theta (zero-shot protocol)
tp_transfer = {}
for tgt, tdocs in targets.items():
    zero = micro_prf(doc_counts_tp(seg25, tdocs, src_opt["threshold"]))
    tuned = tune_tp_theta(seg25, tdocs)
    tp_transfer[tgt] = {
        "zeroshot_f1": zero["f1"], "zeroshot_theta": src_opt["threshold"],
        "tuned_f1": tuned["f1"], "tuned_theta": tuned["threshold"],
    }
    print(f"  TP ELX25->{tgt.upper()}: zero-shot F1={zero['f1']:.4f} "
          f"(theta={src_opt['threshold']:.2f}) | tuned F1={tuned['f1']:.4f} "
          f"(theta={tuned['threshold']:.2f})")

# Morfessor: 5 seeds (training-order randomness)
morf_transfer = {t: [] for t in targets}
morf_meta = []
for seed in SEEDS_5:
    random.seed(seed)
    model, wc = train_morfessor(elx25)
    morf_meta.append({"seed": seed, "lexicon_types": len(wc)})
    for tgt, tdocs in targets.items():
        m = evaluate_morfessor(model, tdocs)
        m["vocab_overlap"] = vocab_overlap(wc, tdocs)
        morf_transfer[tgt].append(m)
        print(f"  Morf seed {seed} ELX25->{tgt.upper()}: F1={m['f1']:.4f}")

morf_transfer_summary = {
    tgt: {
        "f1_mean": float(np.mean([m["f1"] for m in runs])),
        "f1_std": float(np.std([m["f1"] for m in runs])),
        "precision_mean": float(np.mean([m["precision"] for m in runs])),
        "recall_mean": float(np.mean([m["recall"] for m in runs])),
        "vocab_overlap_mean": float(np.mean([m["vocab_overlap"] for m in runs])),
    }
    for tgt, runs in morf_transfer.items()
}

# ---------------------------------------------------------------------------
# 4. Reference numbers (original 13K-era corpus)
# ---------------------------------------------------------------------------
import json
with open(os.path.join(common.REPO, "outputs", "table3_morfessor_transfer.json")) as f:
    published = json.load(f)
reference = {
    "published_morf_elx->akk": published["morfessor_transfer"]["elx->akk"]["f1"],
    "published_morf_elx->sux": published["morfessor_transfer"]["elx->sux"]["f1"],
    "published_tp_elx->akk": published["tp_tuned_reference"]["elx->akk"],
    "published_tp_elx->sux": published["tp_tuned_reference"]["elx->sux"],
    "published_morf_elx_inlang_cv": published["morfessor_diagonal_cv"]["elx"]["f1"],
    "published_tp_elx_inlang": published["tp_tuned_reference"]["elx->elx"],
}

out = {
    "experiment": "C: Elamite 25K refresh",
    "corpus": corpus_stats,
    "in_language_cv_25k": in_language,
    "transfer_from_elx25k": {"tp": tp_transfer,
                             "morfessor": morf_transfer_summary,
                             "morfessor_runs": {t: morf_transfer[t] for t in targets},
                             "morf_meta": morf_meta},
    "reference_original_corpus": reference,
}
save_json(out, "expC_elamite25k_refresh.json")

lines = [
    "### Experiment C: Elamite 25K refresh",
    "",
    f"Expanded corpus: {corpus_stats['expanded_docs']} docs / "
    f"{corpus_stats['expanded_tokens']:,} tokens "
    f"(original: {corpus_stats['old_docs']} docs / "
    f"{corpus_stats['old_tokens']:,}; new Susa texts: "
    f"{corpus_stats['new_docs']} docs / {corpus_stats['new_tokens']:,}, "
    f"conversion rate {1-corpus_stats['new_unconverted_token_rate']:.1%}).",
    "",
    "**In-language (5-fold document-level CV):**",
    "",
    "| Corpus | TP F1 | Morfessor F1 |",
    "|---|---|---|",
    f"| ELX original (paper Table 2) | {reference['published_tp_elx_inlang']:.3f} "
    f"| {reference['published_morf_elx_inlang_cv']:.3f} |",
    f"| ELX expanded ({corpus_stats['expanded_tokens']:,} tok) "
    f"| {in_language['tp_f1_mean']:.3f} ± {in_language['tp_f1_std']:.3f} "
    f"| {in_language['morf_f1_mean_over_seeds']:.3f} ± "
    f"{in_language['morf_f1_std_over_seeds']:.3f} |",
    "",
    "**ELX as transfer source (Table 3 protocol):**",
    "",
    "| Source | Method | -> AKK F1 | -> SUX F1 |",
    "|---|---|---|---|",
    f"| ELX original (published) | TP | {reference['published_tp_elx->akk']:.3f} "
    f"| {reference['published_tp_elx->sux']:.3f} |",
    f"| ELX original (published) | Morfessor | "
    f"{reference['published_morf_elx->akk']:.3f} "
    f"| {reference['published_morf_elx->sux']:.3f} |",
    f"| ELX expanded | TP (zero-shot, θ={tp_transfer['akk']['zeroshot_theta']:.2f}) "
    f"| {tp_transfer['akk']['zeroshot_f1']:.3f} "
    f"| {tp_transfer['sux']['zeroshot_f1']:.3f} |",
    f"| ELX expanded | TP (target-tuned) | {tp_transfer['akk']['tuned_f1']:.3f} "
    f"| {tp_transfer['sux']['tuned_f1']:.3f} |",
    f"| ELX expanded | Morfessor (5 seeds) | "
    f"{morf_transfer_summary['akk']['f1_mean']:.3f} ± "
    f"{morf_transfer_summary['akk']['f1_std']:.3f} | "
    f"{morf_transfer_summary['sux']['f1_mean']:.3f} ± "
    f"{morf_transfer_summary['sux']['f1_std']:.3f} |",
]
save_text("\n".join(lines) + "\n", "expC_elamite25k_refresh.md")
print(f"\ntotal {time.time()-t0:.1f}s")
