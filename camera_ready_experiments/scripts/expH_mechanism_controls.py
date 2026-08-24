"""Experiment H (rebuttal, mKBh W1 clincher): two mechanism controls that
turn the coverage-mediation *correlation* (r = 0.91, Exp E2) into a
*causal / microscopic* demonstration.

H1 - Inventory-clamp control (intervention):
    Train Morfessor on the full Sumerian corpus, but with every token not in
    the Elamite sign inventory (142 types) removed from the training stream.
    Language, register, and (residual) corpus size stay Sumerian; only the
    lexicon's possible coverage is clamped to Elamite's. If transfer to AKK
    collapses to the ELX row's level while transfer to ELX stays high, then
    target coverage - not language identity, not corpus size - is causally
    sufficient to reproduce the Table 3 collapse.
    Contrast row: the same clamp applied to AKK-as-source (train on AKK
    tokens restricted to the ELX inventory, test on SUX).

H2 - Boundary-level localization (microscopy):
    For the published ELX -> AKK model, split gold boundaries by whether the
    two flanking tokens are in the source lexicon. If recall is high on
    covered boundaries and near zero on uncovered ones, the failure is
    localized precisely where the mechanism says it must be.

Protocol as everywhere: Morfessor corpusweight=1.0, zero-shot eval on full
target corpora, 5 seeds for training order.
"""
import random
import time
from collections import Counter

import numpy as np

from common import (LANGS, SEEDS_5, get_corpora, lang_docs, train_morfessor,
                    evaluate_morfessor, gold_boundaries,
                    predict_boundaries_morf, save_json, save_text)

t0 = time.time()
cache = get_corpora(verbose=False)
docs = {lang: lang_docs(cache["documents"], lang) for lang in LANGS}

elx_inventory = set(w for d in docs["elx"] for w in d.split() if w)
print(f"ELX inventory: {len(elx_inventory)} types")


def clamp_docs(doc_list, inventory):
    """Remove tokens outside `inventory` from each doc (>=3 tokens kept)."""
    out = []
    for d in doc_list:
        kept = [w for w in d.split() if w in inventory]
        if len(kept) > 2:
            out.append(" ".join(kept))
    return out


# ---------------------------------------------------------------------------
# H1: inventory clamp
# ---------------------------------------------------------------------------
results_h1 = {}
for src, tgts in [("sux", ["akk", "elx"]), ("akk", ["sux", "elx"])]:
    clamped = clamp_docs(docs[src], elx_inventory)
    n_tok = sum(len(d.split()) for d in clamped)
    lex_types = len(set(w for d in clamped for w in d.split()))
    print(f"\n{src.upper()} clamped to ELX inventory: {len(clamped)} docs / "
          f"{n_tok:,} tokens / {lex_types} types")
    per_tgt = {t: [] for t in tgts}
    for seed in SEEDS_5:
        random.seed(seed)
        model, wc = train_morfessor(clamped)
        for tgt in tgts:
            m = evaluate_morfessor(model, docs[tgt])
            per_tgt[tgt].append(m["f1"])
    results_h1[src] = {
        "clamped_docs": len(clamped), "clamped_tokens": int(n_tok),
        "lexicon_types": lex_types,
        **{f"f1_{t}_mean": float(np.mean(v)) for t, v in per_tgt.items()},
        **{f"f1_{t}_std": float(np.std(v)) for t, v in per_tgt.items()},
    }
    for t in tgts:
        print(f"  {src}-clamped -> {t.upper()}: "
              f"{results_h1[src][f'f1_{t}_mean']:.4f} ± "
              f"{results_h1[src][f'f1_{t}_std']:.4f}")

# ---------------------------------------------------------------------------
# H2: boundary-level localization for ELX -> AKK (published setting)
# ---------------------------------------------------------------------------
random.seed(42)
elx_model, elx_wc = train_morfessor(docs["elx"])
lex = set(elx_wc.keys())

cov = {"covered": [0, 0], "uncovered": [0, 0]}  # [recalled, total]
for d in docs["akk"]:
    toks = d.split()
    if len(toks) < 2:
        continue
    pred = predict_boundaries_morf(elx_model, d.replace(" ", ""))
    pos = 0
    for k in range(len(toks) - 1):
        pos += len(toks[k])
        kind = "covered" if (toks[k] in lex and toks[k + 1] in lex) else "uncovered"
        cov[kind][1] += 1
        if pos in pred:
            cov[kind][0] += 1

h2 = {}
for kind, (rec, tot) in cov.items():
    h2[kind] = {"boundaries": int(tot), "recalled": int(rec),
                "recall": rec / tot if tot else 0.0}
    print(f"ELX->AKK boundary recall ({kind}): {rec:,}/{tot:,} = "
          f"{h2[kind]['recall']:.4f}")

out = {"experiment": "H: mechanism controls (inventory clamp + localization)",
       "elx_inventory_types": len(elx_inventory),
       "h1_inventory_clamp": results_h1,
       "h2_boundary_localization_elx_to_akk": h2,
       "reference": {"elx->akk_published": 0.3038,
                     "sux_full->akk_reproduced": 0.8568}}
save_json(out, "expH_mechanism_controls.json")

s = results_h1["sux"]
lines = [
    "### Experiment H: mechanism controls (mKBh W1)",
    "",
    "**H1 — inventory clamp (causal test).** Morfessor trained on the full "
    "Sumerian corpus with its lexicon clamped to the Elamite sign inventory "
    f"({len(elx_inventory)} types; {s['clamped_tokens']:,} Sumerian tokens "
    "survive the clamp):",
    "",
    "| Source | → AKK | → ELX |",
    "|---|---|---|",
    f"| SUX full, unclamped (Table 3) | 0.857 | 0.953 |",
    f"| SUX full, clamped to ELX inventory | {s['f1_akk_mean']:.3f} ± "
    f"{s['f1_akk_std']:.3f} | {s['f1_elx_mean']:.3f} ± {s['f1_elx_std']:.3f} |",
    f"| ELX full (Table 3) | 0.304 | — |",
    "",
    "**H2 — failure localization.** For the published ELX→AKK model, gold-"
    "boundary recall split by whether both flanking tokens are in the source "
    "lexicon: covered "
    f"{h2['covered']['recall']:.3f} ({h2['covered']['boundaries']:,} "
    f"boundaries) vs uncovered {h2['uncovered']['recall']:.3f} "
    f"({h2['uncovered']['boundaries']:,}).",
]
save_text("\n".join(lines) + "\n", "expH_mechanism_controls.md")
print(f"\ntotal {time.time()-t0:.1f}s")
