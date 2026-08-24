"""Experiment A (rebuttal Tier 1): Sumerian-subsample control for the
Morfessor cross-language collapse.

Reviewer concern: Table 3's Morfessor collapse from the Elamite source
(ELX->AKK F1=0.304, ELX->SUX F1=0.267) could be a *corpus size* effect
(ELX is the smallest source) rather than the *lexicon content* effect the
paper claims.

Control: subsample the Sumerian corpus down to the Elamite training budget,
train Morfessor exactly as in the paper (corpusweight=1.0), and evaluate
zero-shot on the full Akkadian and Elamite corpora (Table 3 off-diagonal
protocol). If SUX-at-ELX-size still transfers far better than ELX, corpus
size cannot explain the collapse.

Protocol:
  - Document-level subsampling: shuffle docs, take shortest prefix reaching
    the token budget (notebook 04 learning-curve protocol).
  - Two budgets: 13,000 tokens (the paper's quoted ELX size) and the measured
    ELX split-token count (exact match to what the ELX source model saw).
  - 5 seeds (42-46); each seed controls both the subsample and Morfessor's
    internal compound shuffle. Mean +/- std (population, as in notebook 02).
"""
import json
import os
import random
import time
from multiprocessing import Pool, cpu_count

import numpy as np

import common
from common import (LANGS, SEEDS_5, get_corpora, lang_docs, train_morfessor,
                    evaluate_morfessor, vocab_overlap, subsample_docs,
                    save_json, save_text)

PRIMARY_BUDGET = 13_000
N_WORKERS = 4

_DOCS = None


def _init_worker():
    global _DOCS
    cache = get_corpora(verbose=False)
    _DOCS = {lang: lang_docs(cache["documents"], lang) for lang in LANGS}


def _run_one(args):
    """One (budget, seed) cell: subsample SUX, train Morfessor, evaluate on
    the full AKK and ELX corpora. Deterministic given (budget, seed)."""
    budget, seed = args
    t = time.time()
    train_subset, actual = subsample_docs(_DOCS["sux"], budget, seed)
    random.seed(seed)  # controls morfessor's internal epoch shuffle
    model, wc = train_morfessor(train_subset)
    row = {
        "budget": budget,
        "seed": seed,
        "actual_tokens": actual,
        "n_train_docs": len(train_subset),
        "lexicon_types": len(wc),
    }
    for tgt in ["akk", "elx"]:
        m = evaluate_morfessor(model, _DOCS[tgt])
        row[f"f1_{tgt}"] = m["f1"]
        row[f"precision_{tgt}"] = m["precision"]
        row[f"recall_{tgt}"] = m["recall"]
        row[f"vocab_overlap_{tgt}"] = vocab_overlap(wc, _DOCS[tgt])
    row["wall_s"] = time.time() - t
    return row


def main():
    t0 = time.time()
    cache = get_corpora()
    docs = {lang: lang_docs(cache["documents"], lang) for lang in LANGS}
    elx_measured = sum(len(d.split()) for d in docs["elx"])
    sux_total = sum(len(d.split()) for d in docs["sux"])
    print(f"SUX pool: {len(docs['sux'])} docs, {sux_total:,} tokens | "
          f"ELX measured budget: {elx_measured:,} tokens")

    budgets = [PRIMARY_BUDGET]
    if abs(elx_measured - PRIMARY_BUDGET) > 500:
        budgets.append(elx_measured)

    jobs = [(b, s) for b in budgets for s in SEEDS_5]
    runs = []
    with Pool(processes=min(N_WORKERS, cpu_count()),
              initializer=_init_worker) as pool:
        for row in pool.imap_unordered(_run_one, jobs):
            runs.append(row)
            print(f"[{row['wall_s']:6.1f}s] budget={row['budget']:>6,d} "
                  f"seed={row['seed']}: {row['actual_tokens']:,} toks / "
                  f"{row['n_train_docs']} docs / {row['lexicon_types']} types | "
                  f"F1 akk={row['f1_akk']:.4f} elx={row['f1_elx']:.4f}",
                  flush=True)
    runs.sort(key=lambda r: (r["budget"], r["seed"]))
    finish(runs, budgets, docs, elx_measured, sux_total, t0)

def finish(runs, budgets, docs, elx_measured, sux_total, t0):
    def agg(rows, key):
        vals = [r[key] for r in rows]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {}
    for budget in budgets:
        rows = [r for r in runs if r["budget"] == budget]
        s = {"n_seeds": len(rows),
             "actual_tokens_mean": float(np.mean([r["actual_tokens"] for r in rows]))}
        for tgt in ["akk", "elx"]:
            mean, std = agg(rows, f"f1_{tgt}")
            s[f"f1_{tgt}_mean"] = mean
            s[f"f1_{tgt}_std"] = std
            s[f"precision_{tgt}_mean"] = float(np.mean([r[f"precision_{tgt}"] for r in rows]))
            s[f"recall_{tgt}_mean"] = float(np.mean([r[f"recall_{tgt}"] for r in rows]))
            s[f"vocab_overlap_{tgt}_mean"] = float(np.mean([r[f"vocab_overlap_{tgt}"] for r in rows]))
        summary[str(budget)] = s

    # Reference rows: reproduced full-corpus transfer + published Table 3
    with open(os.path.join(common.CACHE_DIR, "full_transfer_repro.json")) as f:
        repro = json.load(f)
    with open(os.path.join(common.REPO, "outputs", "table3_morfessor_transfer.json")) as f:
        published = json.load(f)["morfessor_transfer"]

    out = {
        "experiment": "A: Morfessor Sumerian-subsample control",
        "protocol": {
            "subsampling": "document-level shuffled prefix to token budget (notebook 04)",
            "morfessor_corpusweight": 1.0,
            "eval": "zero-shot on full target corpora (Table 3 off-diagonal protocol)",
            "seeds": SEEDS_5,
            "budgets": budgets,
            "elx_measured_tokens": elx_measured,
            "sux_full_tokens": sux_total,
        },
        "runs": runs,
        "summary": summary,
        "reference": {
            "sux_full->akk": {"reproduced": repro["reproduced_transfer"]["sux->akk"]["f1"],
                              "published": published["sux->akk"]["f1"]},
            "sux_full->elx": {"reproduced": repro["reproduced_transfer"]["sux->elx"]["f1"],
                              "published": published["sux->elx"]["f1"]},
            "elx->akk": {"reproduced": repro["reproduced_transfer"]["elx->akk"]["f1"],
                         "published": published["elx->akk"]["f1"]},
            "elx->sux": {"reproduced": repro["reproduced_transfer"]["elx->sux"]["f1"],
                         "published": published["elx->sux"]["f1"]},
        },
    }
    save_json(out, "expA_sux13k_subsample_control.json")

    # ---- paste-ready markdown ----
    lines = [
        "### Experiment A: Morfessor Sumerian-subsample control",
        "",
        "Morfessor trained on Sumerian subsampled to the Elamite training budget "
        "(document-level subsampling, 5 seeds, corpusweight=1.0), evaluated "
        "zero-shot on the full target corpora (same protocol as Table 3 "
        "off-diagonals). Mean +/- std over 5 seeds.",
        "",
        "| Source (train size) | -> AKK F1 | -> ELX F1 | -> SUX F1 |",
        "|---|---|---|---|",
    ]
    for budget in budgets:
        s = summary[str(budget)]
        label = f"SUX subsample ({budget:,} tok)"
        lines.append(f"| {label} | {s['f1_akk_mean']:.3f} ± {s['f1_akk_std']:.3f} "
                     f"| {s['f1_elx_mean']:.3f} ± {s['f1_elx_std']:.3f} | — |")
    lines.append(f"| SUX full ({sux_total:,} tok) | "
                 f"{repro['reproduced_transfer']['sux->akk']['f1']:.3f} | "
                 f"{repro['reproduced_transfer']['sux->elx']['f1']:.3f} | — |")
    lines.append(f"| ELX full ({elx_measured:,} tok; paper: 13K) | "
                 f"{published['elx->akk']['f1']:.3f} | — | "
                 f"{published['elx->sux']['f1']:.3f} |")
    lines += [
        "",
        f"ELX row = published Table 3. SUX-full row = reproduction with released "
        f"code (published: {published['sux->akk']['f1']:.3f} / "
        f"{published['sux->elx']['f1']:.3f}).",
    ]
    save_text("\n".join(lines) + "\n", "expA_sux13k_subsample_control.md")

    print("\n=== EXPERIMENT A SUMMARY ===")
    for budget in budgets:
        s = summary[str(budget)]
        print(f"SUX@{budget:,}: ->AKK {s['f1_akk_mean']:.4f}±{s['f1_akk_std']:.4f} | "
              f"->ELX {s['f1_elx_mean']:.4f}±{s['f1_elx_std']:.4f}")
    print(f"ELX(full)->AKK published: {published['elx->akk']['f1']:.4f} | "
          f"ELX(full)->SUX published: {published['elx->sux']['f1']:.4f}")
    print(f"total {time.time()-t0:.1f}s")



if __name__ == "__main__":
    main()
