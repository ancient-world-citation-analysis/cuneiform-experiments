"""Experiment E (rebuttal Tier 2/3): Morfessor cross-language transfer vs
training-corpus size, for every source language.

For each source language, subsample the training corpus at 5K / 13K / 25K /
full split-tokens (document-level shuffled-prefix subsampling, notebook 04
protocol), train Morfessor (corpusweight=1.0), and evaluate zero-shot on the
full corpora of the other two languages (Table 3 off-diagonal protocol).
5 seeds per cell; seeds control both the subsample and Morfessor's internal
epoch shuffle. Budgets >= a corpus's full size are skipped ('full' covers
them). Sources include ELX-expanded (Experiment C's corpus) when available.

Run AFTER expC for the ELX-expanded rows.
"""
import json
import os
import random
import time
from multiprocessing import Pool, cpu_count

import numpy as np

import common
from common import (SEEDS_5, get_corpora, lang_docs, train_morfessor,
                    evaluate_morfessor, subsample_docs, save_json, save_text)

BUDGETS = [5_000, 13_000, 25_000, None]  # None = full corpus
N_WORKERS = 4

TARGETS_OF = {
    "akk": ["sux", "elx"],
    "sux": ["akk", "elx"],
    "elx25": ["akk", "sux"],
    "elx_orig": ["akk", "sux"],
}

_SOURCES = None
_TARGET_DOCS = None


def _load_all():
    cache = get_corpora(verbose=False)
    documents = cache["documents"]
    akk = lang_docs(documents, "akk")
    sux = lang_docs(documents, "sux")
    elx_orig = lang_docs(documents, "elx")
    sources = {"akk": akk, "sux": sux, "elx_orig": elx_orig}
    elx25_path = os.path.join(common.CACHE_DIR, "elx25_docs.json")
    if os.path.exists(elx25_path):
        with open(elx25_path) as f:
            sources["elx25"] = json.load(f)
    return sources, {"akk": akk, "sux": sux, "elx": elx_orig}


def _init_worker():
    global _SOURCES, _TARGET_DOCS
    _SOURCES, _TARGET_DOCS = _load_all()


def _run_one(args):
    src, budget, seed = args
    t = time.time()
    pool = _SOURCES[src]
    full_tokens = sum(len(d.split()) for d in pool)
    if budget is None:
        train_set, actual = pool, full_tokens
        label = "full"
    else:
        train_set, actual = subsample_docs(pool, budget, seed)
        label = f"{budget//1000}K"
    random.seed(seed)
    model, wc = train_morfessor(train_set)
    row = {"source": src, "budget": label, "seed": seed,
           "actual_tokens": actual, "n_docs": len(train_set),
           "lexicon_types": len(wc)}
    for tgt in TARGETS_OF[src]:
        row[f"f1_{tgt}"] = evaluate_morfessor(model, _TARGET_DOCS[tgt])["f1"]
    row["wall_s"] = time.time() - t
    return row


def main():
    t0 = time.time()
    sources, _ = _load_all()

    jobs = []
    for src, pool in sources.items():
        full_tokens = sum(len(d.split()) for d in pool)
        for budget in BUDGETS:
            if budget is not None and budget >= full_tokens:
                continue
            for seed in SEEDS_5:
                jobs.append((src, budget, seed))
    print(f"{len(jobs)} jobs over sources {list(sources)}")

    grid = []
    with Pool(processes=min(N_WORKERS, cpu_count()),
              initializer=_init_worker) as pool:
        for row in pool.imap_unordered(_run_one, jobs):
            grid.append(row)
            f1s = " ".join(f"{t}:{row[f'f1_{t}']:.3f}" for t in TARGETS_OF[row["source"]])
            print(f"[{row['wall_s']:6.1f}s] {row['source']:>8s} "
                  f"@{row['budget']:>4s} seed={row['seed']}: "
                  f"{row['actual_tokens']:>9,d} tok | {f1s}", flush=True)
    grid.sort(key=lambda r: (r["source"], r["budget"], r["seed"]))

    summary = {}
    for src in sources:
        for budget in ["5K", "13K", "25K", "full"]:
            rows = [r for r in grid if r["source"] == src and r["budget"] == budget]
            if not rows:
                continue
            cell = {"n_seeds": len(rows),
                    "actual_tokens_mean": float(np.mean([r["actual_tokens"] for r in rows]))}
            for tgt in TARGETS_OF[src]:
                vals = [r[f"f1_{tgt}"] for r in rows]
                cell[f"f1_{tgt}_mean"] = float(np.mean(vals))
                cell[f"f1_{tgt}_std"] = float(np.std(vals))
            summary[f"{src}@{budget}"] = cell

    out = {
        "experiment": "E: Morfessor cross-language transfer vs source size",
        "protocol": {
            "subsampling": "document-level shuffled prefix (notebook 04)",
            "morfessor_corpusweight": 1.0,
            "eval": "zero-shot on full target corpora",
            "seeds": SEEDS_5,
        },
        "grid": grid,
        "summary": summary,
    }
    save_json(out, "expE_morfessor_size_grid.json")

    lines = [
        "### Experiment E: Morfessor transfer vs source-corpus size",
        "",
        "Zero-shot transfer F1 (mean ± std over 5 seeds; document-level "
        "subsampling; corpusweight=1.0). 'full' rows use the entire source "
        "corpus (seeds then only affect Morfessor's internal training order).",
        "",
        "| Source | Size | → AKK | → SUX | → ELX |",
        "|---|---|---|---|---|",
    ]
    pretty = {"akk": "AKK", "sux": "SUX", "elx25": "ELX-expanded",
              "elx_orig": "ELX-original"}
    for src in ["akk", "sux", "elx25", "elx_orig"]:
        for budget in ["5K", "13K", "25K", "full"]:
            key = f"{src}@{budget}"
            if key not in summary:
                continue
            cell = summary[key]
            cols = []
            for tgt in ["akk", "sux", "elx"]:
                if f"f1_{tgt}_mean" in cell:
                    cols.append(f"{cell[f'f1_{tgt}_mean']:.3f} ± "
                                f"{cell[f'f1_{tgt}_std']:.3f}")
                else:
                    cols.append("—")
            lines.append(f"| {pretty[src]} | {budget} "
                         f"({cell['actual_tokens_mean']:,.0f} tok) | "
                         + " | ".join(cols) + " |")
    save_text("\n".join(lines) + "\n", "expE_morfessor_size_grid.md")
    print(f"\ntotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
