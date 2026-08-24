"""Experiment F (rebuttal Tier 3, Dcaz Weakness 4): what share of the
ingested corpora is damage-marked, and what share was excluded up front?

Damage is encoded differently per source, so each corpus is counted from
its native annotation:

  Akkadian (ORACC)  : token's `gdl` sign-level JSON carries
                      "break": "damaged" / "missing" flags -> a token is
                      damage-marked if ANY of its signs is damaged/missing.
  Sumerian (ETCSL)  : composite editions; damage rarely reaches the
                      lemmatized export - count forms containing the
                      illegible-sign placeholder X.
  Elamite Utu-nashu : editorial marks in the raw transliteration
                      ([ ], half-brackets, #, ?, !, ellipses, isolated x).
  Elamite Susa (new): same, on the raw document transliterations.

Also reported: rows the released loaders exclude before any experiment
(null form/pos; Akkadian pos == 'u' or form x/X = unlemmatizable/illegible).
"""
import os
import re

import pandas as pd

import common
from common import save_json, save_text

DAMAGE_CHARS = re.compile(r"[#\[\]⸢⸣!?]|\.\.\.|…")
ILLEGIBLE = re.compile(r"(?:^|[-.\s])[xX](?:$|[-.\s])")

results = {}

# --- Akkadian (ORACC): chunked full-file scan of gdl flags ---
tot = kept_n = damaged = missing = either = excluded = 0
for chunk in pd.read_csv(os.path.join(common.DATA_DIR, "alltexts_AKK.csv"),
                         usecols=["form", "pos", "gdl"], dtype=str,
                         chunksize=500_000, low_memory=False):
    tot += len(chunk)
    exc = chunk["pos"].isna() | (chunk["pos"] == "u") | chunk["form"].isna() \
        | chunk["form"].isin(["x", "X"])
    excluded += int(exc.sum())
    kept = chunk[~exc]
    kept_n += len(kept)
    g = kept["gdl"].fillna("").astype(str)
    dmg = g.str.contains("'damaged'", regex=False) | g.str.contains('"damaged"', regex=False)
    mis = g.str.contains("'missing'", regex=False) | g.str.contains('"missing"', regex=False)
    damaged += int(dmg.sum())
    missing += int(mis.sum())
    either += int((dmg | mis).sum())
results["akk"] = {
    "raw_rows": tot, "excluded_rows": excluded,
    "excluded_pct": excluded / tot * 100,
    "ingested_tokens": kept_n,
    "damaged_sign_tokens": damaged, "missing_sign_tokens": missing,
    "damage_marked": either, "damage_marked_pct": either / kept_n * 100,
    "note": "token counted if any constituent sign flagged damaged/missing in gdl",
}

# --- Sumerian (ETCSL composite editions) ---
sux = pd.read_csv(os.path.join(common.DATA_DIR, "alltexts_SUX.csv"), dtype=str)
n_raw = len(sux)
exc = sux["pos"].isna() | (sux["pos"].astype(str) == "nan") | sux["form"].isna()
kept = sux[~exc]
xmark = int(kept["form"].str.contains("X", na=False).sum())
results["sux"] = {
    "raw_rows": int(n_raw), "excluded_rows": int(exc.sum()),
    "excluded_pct": float(exc.mean() * 100),
    "ingested_tokens": int(len(kept)),
    "damage_marked": xmark, "damage_marked_pct": float(xmark / len(kept) * 100),
    "note": "ETCSL composite editions; damage largely resolved editorially - "
            "count = forms containing illegible-sign placeholder X",
}

# --- Elamite: Utu-nashu word-level + new Susa documents (raw marks) ---
def mark_stats(forms):
    total = len(forms)
    marked = int((forms.str.contains(DAMAGE_CHARS, na=False)
                  | forms.str.contains(ILLEGIBLE, na=False)).sum())
    return total, marked

nasu = pd.read_csv(os.path.join(common.DATA_DIR, "UnTN-Nasu texts Word-level.csv"),
                   dtype=str)
total, marked = mark_stats(nasu["transliteration"].dropna().astype(str))
results["elx_utu_nashu"] = {
    "ingested_tokens": total, "damage_marked": marked,
    "damage_marked_pct": marked / total * 100,
    "note": "editorial marks in raw word transliterations",
}

susa = pd.read_csv(os.path.join(common.REPO, "newelamitedata.csv"),
                   dtype=str).drop_duplicates(subset="id")
words = susa["transliteration"].dropna().astype(str).str.split().explode()
total, marked = mark_stats(words)
results["elx_susa_new"] = {
    "documents": int(len(susa)), "ingested_tokens": total,
    "damage_marked": marked, "damage_marked_pct": marked / total * 100,
    "note": "editorial marks in raw document transliterations",
}

out = {"experiment": "F: damaged/excluded token share per corpus",
       "results": results}
save_json(out, "expF_damage_fraction.json")

lines = [
    "### Experiment F: damaged-token share per corpus (Dcaz W4)",
    "",
    "Share of ingested tokens carrying damage marks in each source's native "
    "annotation (the pipeline strips marks but keeps the signs; only fully "
    "illegible/unlemmatizable rows are excluded up front).",
    "",
    "| Corpus | Ingested tokens | Damage-marked | Excluded up front |",
    "|---|---|---|---|",
]
lbl = {"akk": "Akkadian (ORACC; sign-level damaged/missing flags)",
       "sux": "Sumerian (ETCSL composites; X placeholders)",
       "elx_utu_nashu": "Elamite Utu-našu (editorial marks)",
       "elx_susa_new": "Elamite Susa, new (editorial marks)"}
for k, r in results.items():
    exc = f"{r['excluded_pct']:.1f}%" if "excluded_pct" in r else "—"
    lines.append(f"| {lbl[k]} | {r['ingested_tokens']:,} "
                 f"| {r['damage_marked_pct']:.1f}% | {exc} |")
lines += ["", "AKK 'excluded up front' = unlemmatizable rows (pos 'u'), "
          "illegible x/X forms, and null rows dropped by the released loader."]
save_text("\n".join(lines) + "\n", "expF_damage_fraction.md")

for k, r in results.items():
    print(k, f"marked={r['damage_marked_pct']:.2f}%",
          f"excluded={r.get('excluded_pct', float('nan')):.2f}%")
