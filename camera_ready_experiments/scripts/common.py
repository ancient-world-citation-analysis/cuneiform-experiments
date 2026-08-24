"""Shared harness for ARR May-cycle rebuttal experiments.

Data loading is delegated verbatim to the released pipeline
(notebooks/_setup.py::load_corpora); segmentation training / evaluation
helpers are copied unchanged from notebooks 02-04 so every rebuttal number
is produced by the same code paths as the paper's Tables 2-3.

Protocol constants (paper):
  - Morfessor: BaselineModel, corpusweight = 1.0, batch training
  - TP: cunei_tools.CuneiSeg (char bigram TP); theta tuning =
    CuneiSeg.find_optimal_threshold default coarse grid (0.05..0.95 step .05)
  - Boundary F1: micro-averaged over documents (positions of spaces)
  - Document filter: keep docs with len(doc.split()) > 2 (notebooks 02/03)
  - 5-fold document-level CV: sklearn KFold(n_splits=5, shuffle=True,
    random_state=42) (notebook 02)
  - Significance: document-level paired bootstrap, 2000 resamples, two-sided
"""
import os
import sys
import json
import pickle
import random
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REBUTTAL_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(REBUTTAL_DIR)
NOTEBOOKS_DIR = os.path.join(REPO, "notebooks")
CACHE_DIR = os.path.join(REBUTTAL_DIR, "cache")
OUTPUT_DIR = os.path.join(REBUTTAL_DIR, "outputs")
DATA_DIR = os.path.join(CACHE_DIR, "data")

CORPORA_CACHE = os.path.join(CACHE_DIR, "corpora_cache.pkl")

# _setup.py reads $CUNEI_DATA at import time -> must be set before import.
os.environ.setdefault("CUNEI_DATA", DATA_DIR + "/")
sys.path.insert(0, NOTEBOOKS_DIR)

LANGS = ["akk", "sux", "elx"]
LANG_NAMES = {"akk": "Akkadian", "sux": "Sumerian", "elx": "Elamite"}

MORFESSOR_CORPUSWEIGHT = 1.0
N_BOOTSTRAP = 2000
CV_SEED = 42
SEEDS_5 = [42, 43, 44, 45, 46]


# ---------------------------------------------------------------------------
# Corpus loading (via released _setup.py; cached after first build)
# ---------------------------------------------------------------------------

def build_corpora_cache(verbose=True):
    """Run the released loader once and cache everything the rebuttal
    experiments need. Returns the cache dict."""
    from _setup import load_corpora  # noqa: deferred so CUNEI_DATA is set

    corpora = load_corpora(os.environ["CUNEI_DATA"])
    documents = corpora["_documents"]

    cache = {
        "documents": documents,
        "sign_dict": corpora["_sign_dict"],
        "elx_dict_df": corpora["elx"][
            ["form_latin", "form_unicode", "form_unicode_nospace",
             "pos_raw", "lemma", "unicode_clean"]
        ].copy() if "elx" in corpora else None,
    }
    with open(CORPORA_CACHE, "wb") as f:
        pickle.dump(cache, f)
    if verbose:
        for lang in LANGS:
            if lang in documents:
                docs = lang_docs(documents, lang)
                toks = sum(len(d.split()) for d in docs)
                print(f"  {lang.upper()}: {len(docs)} docs (>2-token filter), "
                      f"{toks:,} split-tokens")
    return cache


def get_corpora(verbose=True):
    """Load the cached corpora (building the cache on first call)."""
    if os.path.exists(CORPORA_CACHE):
        with open(CORPORA_CACHE, "rb") as f:
            return pickle.load(f)
    if verbose:
        print("[common] corpora cache not found - building via _setup.load_corpora "
              "(one-time, reads the full ORACC/ETCSL CSVs)...")
    return build_corpora_cache(verbose=verbose)


def lang_docs(documents, lang):
    """Document list exactly as notebooks 02/03 build it."""
    return [d for d in documents[lang]["unicode"].values()
            if d and len(d.split()) > 2]


# ---------------------------------------------------------------------------
# Morfessor (copied from notebook 02 / 04, unchanged)
# ---------------------------------------------------------------------------

def train_morfessor(train_docs, corpusweight=MORFESSOR_CORPUSWEIGHT):
    import morfessor
    word_counts = Counter()
    for doc in train_docs:
        for w in doc.split():
            if w:
                word_counts[w] += 1
    model = morfessor.BaselineModel(corpusweight=corpusweight)
    model.load_data([(c, w) for w, c in word_counts.items()])
    model.train_batch()
    return model, word_counts


def gold_boundaries(segmented):
    g, pos = set(), 0
    for ch in segmented:
        if ch == " ":
            g.add(pos)
        else:
            pos += 1
    return g


def predict_boundaries_morf(model, continuous):
    if not continuous:
        return set()
    try:
        segs, _ = model.viterbi_segment(continuous)
    except Exception:
        return set()
    b, cur = set(), 0
    for s in segs[:-1]:
        cur += len(s)
        b.add(cur)
    return b


def doc_counts_morf(model, test_docs):
    """Per-document (tp, fp, fn) boundary counts; doc filter as in notebooks."""
    counts = []
    for d in test_docs:
        if not d or len(d.split()) < 2:
            continue
        gold = gold_boundaries(d)
        pred = predict_boundaries_morf(model, d.replace(" ", ""))
        counts.append((len(pred & gold), len(pred - gold), len(gold - pred)))
    return counts


# ---------------------------------------------------------------------------
# TP via cunei_tools (canonical implementation released with the paper)
# ---------------------------------------------------------------------------

def train_tp(train_docs, lang=None):
    from cunei_tools import CuneiSeg
    seg = CuneiSeg(lang=lang)
    seg.train(train_docs)
    return seg


def tune_tp_theta(seg, docs):
    """cunei-tools default tuning: coarse grid 0.05..0.95 step 0.05."""
    return seg.find_optimal_threshold(docs)


def doc_counts_tp(seg, test_docs, theta):
    """Per-document (tp, fp, fn) at a fixed theta; mirrors
    CuneiSeg.find_optimal_threshold's inner loop exactly."""
    counts = []
    for doc in test_docs:
        if not doc or len(doc.split()) < 2:
            continue
        gold = gold_boundaries(doc)
        continuous = doc.replace(" ", "")
        pred = set()
        for i in range(1, len(continuous)):
            if seg._transitional_prob(continuous[i - 1], continuous[i]) < theta:
                pred.add(i)
        counts.append((len(pred & gold), len(pred - gold), len(gold - pred)))
    return counts


# ---------------------------------------------------------------------------
# Metrics / aggregation
# ---------------------------------------------------------------------------

def micro_prf(counts):
    tp = sum(c[0] for c in counts)
    fp = sum(c[1] for c in counts)
    fn = sum(c[2] for c in counts)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"f1": f1, "precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn}


def evaluate_morfessor(model, test_docs):
    return micro_prf(doc_counts_morf(model, test_docs))


def vocab_overlap(train_counts, test_docs):
    """Fraction of test split-tokens whose form appears in the trained lexicon
    (notebook 02 diagnostic)."""
    vocab = set(train_counts.keys())
    seen = total = 0
    for d in test_docs:
        for w in d.split():
            if w:
                total += 1
                if w in vocab:
                    seen += 1
    return seen / total if total else 0.0


def subsample_docs(docs, target_tokens, seed):
    """Shuffle documents with `seed`, take the shortest prefix whose cumulative
    split-token count reaches target_tokens (notebook 04 protocol)."""
    pool = list(docs)
    random.Random(seed).shuffle(pool)
    out, cum = [], 0
    for d in pool:
        out.append(d)
        cum += len(d.split())
        if cum >= target_tokens:
            break
    return out, cum


def paired_bootstrap(counts_a, counts_b, n_resamples=N_BOOTSTRAP, seed=42):
    """Document-level paired bootstrap on the micro-F1 difference (A - B).

    counts_a / counts_b: per-document (tp, fp, fn) for the SAME documents in
    the same order. Two-sided p-value + 95% percentile CI of the difference.
    """
    assert len(counts_a) == len(counts_b), "paired bootstrap needs aligned docs"
    a = np.asarray(counts_a, dtype=np.int64)
    b = np.asarray(counts_b, dtype=np.int64)
    n = len(a)

    def f1_of(arr):
        tp, fp, fn = arr[:, 0].sum(), arr[:, 1].sum(), arr[:, 2].sum()
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    observed = f1_of(a) - f1_of(b)
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        diffs[i] = f1_of(a[idx]) - f1_of(b[idx])
    p_two_sided = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p_two_sided = min(1.0, p_two_sided)
    return {
        "observed_diff": float(observed),
        "p_two_sided": float(p_two_sided),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "n_docs": int(n),
        "n_resamples": int(n_resamples),
    }


def save_json(obj, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[saved] {path}")
    return path


def save_text(text, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        f.write(text)
    print(f"[saved] {path}")
    return path
