"""Shared setup utilities for the numbered experiment notebooks.

Each notebook in this directory imports `BASE_PATH`, `POS_HARMONIZATION`, and
`load_corpora` from this module. The intent is to keep notebooks focused on a
single experiment while ensuring that data loading, POS harmonization, and
sign-list construction are defined exactly once and stay in sync across
experiments.

Set `BASE_PATH` to the directory containing the input CSV / XLSX files:

    alltexts_AKK.csv                      (from scripts/filter_oracc_data.py)
    alltexts_SUX.csv                      (from scripts/filter_oracc_data.py)
    Elamite_Lemma-base-draft.xlsx
    UnTN-Nasu texts Word-level.csv
    unmatchednew_AAedit - unmatchednew.csv
    unmatchednew - solonew.csv
    7000_hitt_txts_wGloss.csv             (only needed for the Hittite notebook)
"""
from __future__ import annotations

import os
import random
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_PATH = os.environ.get("CUNEI_DATA", "/path/to/cuneiform/data/")
if not BASE_PATH.endswith("/"):
    BASE_PATH += "/"

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Set all RNGs used in the pipeline to a fixed seed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# POS harmonization
# ---------------------------------------------------------------------------

POS_HARMONIZATION: Dict[str, Dict[str, str]] = {
    "akk": {
        "N": "NOUN", "V": "VERB", "AJ": "ADJ", "AV": "ADV",
        "PRP": "ADP", "DET": "DET", "CNJ": "CONJ", "MOD": "MOD",
        "REL": "REL", "SBJ": "SBJN", "IP": "INTJ",
        "PN": "PN", "DN": "DN", "GN": "GN", "CN": "CN",
        "RN": "RN", "QN": "QN", "WN": "WN", "MN": "MN",
        "AN": "AN", "FN": "FN", "TN": "TN", "LN": "LN", "ON": "ON",
        "n": "NUM", "u": "X", "X": "X",
    },
    "sux": {
        "N": "NOUN", "V/t": "VERB", "V/i": "VERB", "V": "VERB",
        "AJ": "ADJ", "AV": "ADV",
        "NU": "NUM", "IP": "INTJ", "QP": "QP",
        "DN": "DN", "GN": "GN", "PN": "PN", "RN": "RN",
        "SN": "SN", "TN": "TN", "WN": "WN", "MN": "MN",
        "NA": "X",
    },
    "elx": {
        "Noun": "NOUN", "Verb": "VERB", "ADJ": "ADJ", "other": "OTHER",
        "PN": "PN", "PN-hyp": "PN", "GN": "GN", "DN": "DN", "Magic": "OTHER",
    },
}

ENTITY_TAGS = {
    "PN", "DN", "GN", "CN", "RN", "QN", "WN", "MN",
    "AN", "FN", "TN", "LN", "ON", "SN", "QP",
}


def map_pos_unified(pos_raw: str, lang: str) -> str:
    return POS_HARMONIZATION.get(lang, {}).get(pos_raw, "X")


def map_pos_grammatical(pos_unified: str) -> str:
    return "PROPN" if pos_unified in ENTITY_TAGS else pos_unified


def map_ner_tag(pos_unified: str) -> str:
    return pos_unified if pos_unified in ENTITY_TAGS else "O"


# ---------------------------------------------------------------------------
# Sign lists
# ---------------------------------------------------------------------------

NUOLENNA_URL = (
    "https://raw.githubusercontent.com/situx/Nuolenna/master/sign_list.json"
)
AKKADEMIA_URL = (
    "https://raw.githubusercontent.com/gaigutherz/Akkademia/master/"
    "cuneiform_to_unicode_fixed.csv"
)


def load_sign_dict(base_path: str = BASE_PATH) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], pd.DataFrame]:
    """Load and merge Nuolenna + Akkademia sign lists plus local manual corrections.

    Returns (sign_dict, manual_dict_1, manual_dict_2, merged_df).
    """
    sign_list = pd.read_json(NUOLENNA_URL, orient="index")
    sign_list.columns = ["unicode"]
    sign_list["sign"] = sign_list.index.tolist()
    sign_list = sign_list[["sign", "unicode"]].reset_index(drop=True)

    akkademia = pd.read_csv(AKKADEMIA_URL)
    merged = pd.merge(sign_list, akkademia, on=["sign", "unicode"], how="outer")
    sign_dict = dict(zip(merged["sign"].astype(str), merged["unicode"].astype(str)))

    manual_dict, manual_dict2 = {}, {}
    p1 = base_path + "unmatchednew_AAedit - unmatchednew.csv"
    p2 = base_path + "unmatchednew - solonew.csv"
    try:
        un1 = pd.read_csv(p1)[["unmatched_sign", "use"]].dropna()
        manual_dict = dict(zip(un1["unmatched_sign"], un1["use"]))
    except FileNotFoundError:
        pass
    try:
        un2 = pd.read_csv(p2)[["value", "SIGN"]].dropna()
        manual_dict2 = dict(zip(un2["value"].str.strip("[]' "), un2["SIGN"]))
    except FileNotFoundError:
        pass

    sign_dict.update(manual_dict)
    sign_dict.update(manual_dict2)
    return sign_dict, manual_dict, manual_dict2, merged


# ---------------------------------------------------------------------------
# Transliteration → Unicode
# ---------------------------------------------------------------------------

def normalize_transliteration(text, lang: str = "akk") -> str:
    if pd.isna(text) or text == "":
        return ""
    s = str(text)
    s = re.sub(r"\{[^}]*\}", "", s)
    s = s.replace("-", " ").replace(".", " ")
    for ch in ["[", "]", "#", "!", "?", "*", "(", ")"]:
        s = s.replace(ch, "")
    return re.sub(r"\s+", " ", s).strip()


def transliteration_to_unicode(text, sign_dict, lang: str = "akk"):
    normalized = normalize_transliteration(text, lang)
    if not normalized:
        return "", []
    out, unmatched = [], []
    for tok in normalized.split():
        if tok in sign_dict:
            out.append(sign_dict[tok])
        elif tok.lower() in sign_dict:
            out.append(sign_dict[tok.lower()])
        elif tok.upper() in sign_dict:
            out.append(sign_dict[tok.upper()])
        else:
            out.append(tok)
            if re.search(r"[A-Za-z]", tok):
                unmatched.append(tok)
    return " ".join(out), unmatched


def convert_column_to_unicode(forms, sign_dict, lang: str = "akk"):
    results = forms.apply(lambda x: transliteration_to_unicode(x, sign_dict, lang))
    unicode_col = results.apply(lambda x: x[0])
    unmatched_col = results.apply(lambda x: x[1])
    n_clean = (unmatched_col.apply(len) == 0).sum()
    rate = n_clean / len(forms) if len(forms) > 0 else 0
    return unicode_col, rate


def _normalize_for_unmatched_pass(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"\(\s*md\s*\)", " m d ", s)
    s = re.sub(r"[.,:;!?()\[\]{}<>\\\"\"\"''/\\|*^`~]", " ", s)
    s = re.sub(r"[-–—]", " ", s)
    s = re.sub(r"([A-Za-z])(\d)([A-Za-z])", r"\1\2 \3", s)
    s = re.sub(r"(\d)([A-Za-z])", r"\1 \2", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# Per-language loaders
# ---------------------------------------------------------------------------

def load_akkadian(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["pos"].notna() & (df["pos"] != "u")]
    df = df[df["form"].notna() & (df["form"] != "x") & (df["form"] != "X")].copy()
    out = pd.DataFrame({
        "token_id": range(len(df)),
        "text_id": df["id_text"].values,
        "form_latin": df["form"].values,
        "pos_raw": df["pos"].values,
        "lemma": df["cf"].values,
        "gloss": df["gw"].values,
        "language": "akk",
    })
    out["pos_unified"] = out["pos_raw"].apply(lambda x: map_pos_unified(x, "akk"))
    out["pos_grammatical"] = out["pos_unified"].apply(map_pos_grammatical)
    out["ner_tag"] = out["pos_unified"].apply(map_ner_tag)
    return out


def load_sumerian(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["pos"].notna() & (df["pos"] != "") & (df["pos"].astype(str) != "nan")]
    df = df[df["form"].notna()].copy()
    out = pd.DataFrame({
        "token_id": range(len(df)),
        "text_id": df["id_text"].values,
        "form_latin": df["form"].values,
        "pos_raw": df["pos"].values,
        "lemma": df["cf"].values,
        "gloss": df["gw"].values,
        "language": "sux",
    })
    out["pos_unified"] = out["pos_raw"].apply(lambda x: map_pos_unified(x, "sux"))
    out["pos_grammatical"] = out["pos_unified"].apply(map_pos_grammatical)
    out["ner_tag"] = out["pos_unified"].apply(map_ner_tag)
    return out


def load_elamite(dict_path: str, nasu_path: str, sign_dict: Dict[str, str], merged: pd.DataFrame, manual_dict: Dict, manual_dict2: Dict):
    """Load Elamite lemma base + Nasu corpus.

    Returns (elx_df, nasu_documents) where nasu_documents = {"latin": {...}, "unicode": {...}}.
    """
    all_sheets = pd.read_excel(dict_path, sheet_name=None)
    target_tabs = ["ADJ", "Noun", "Verb", "other", "PN", "PN-hyp", "GN", "DN", "Magic"]
    columns_to_keep = [
        "transliteration", "sorting", "period", "base",
        "logogram", "morpheme_1", "morpheme_2", "morpheme_3",
        "sense_hk", "certainty-weight_hk", "sense_hk_qid",
        "certainty-weight_MEGA", "sense_MEGA_qid", "POS", "number", "person",
    ]
    combined = []
    for name in target_tabs:
        if name in all_sheets:
            df = all_sheets[name]
            df["category"] = name
            existing = [c for c in columns_to_keep if c in df.columns]
            combined.append(df[existing + ["category"]])
    final_dictionary = pd.concat(combined, ignore_index=True)

    to_unicode = final_dictionary.copy()
    to_unicode["transliteration original"] = to_unicode["transliteration"]
    to_unicode["transliteration"] = to_unicode["transliteration"].astype(str).fillna("")
    to_unicode["transliteration"] = to_unicode["transliteration"].str.replace("-", " ")
    for ch in ["_", "[", "]", "*", "!", "?", "/", ",", ":", ";", "^", "`"]:
        to_unicode["transliteration"] = to_unicode["transliteration"].str.replace(ch, "", regex=False)
    to_unicode["transliteration"] = to_unicode["transliteration"].str.replace(".", " ", regex=False).str.lower()
    to_unicode["transliteration"] = to_unicode["transliteration"].str.replace(r"~.*?……", "……", regex=True)
    to_unicode["transliteration"] = to_unicode["transliteration"].str.replace("X", "", regex=False)
    to_unicode["transliteration"] = to_unicode["transliteration"].str.replace("……", " ", regex=False)

    def replace_with_unicode(text):
        return " ".join([sign_dict.get(s, s) for s in str(text).split()])

    unicode_dict_v2 = dict(zip(merged["sign"].astype(str), merged["unicode"].astype(str)))
    unicode_dict_v2.update(manual_dict)
    unicode_dict_v2.update(manual_dict2)

    def final_pass(unicode_text):
        txt = _normalize_for_unmatched_pass(unicode_text)
        out, still = [], []
        for tok in txt.split():
            if tok == "md" and "m" in unicode_dict_v2 and "d" in unicode_dict_v2 and "md" not in unicode_dict_v2:
                out.append(unicode_dict_v2["m"]); out.append(unicode_dict_v2["d"]); continue
            if tok in unicode_dict_v2:
                out.append(unicode_dict_v2[tok])
            else:
                out.append(tok)
                if re.search(r"[A-Za-z]", tok):
                    still.append(tok)
        return " ".join(out), still

    to_unicode["unicode"] = to_unicode["transliteration"].apply(replace_with_unicode)
    tmp = to_unicode["unicode"].apply(final_pass)
    to_unicode["transliteration"] = to_unicode["transliteration original"].apply(_normalize_for_unmatched_pass)
    to_unicode["roman"] = tmp.apply(lambda x: len(x[1]) > 0)
    to_unicode["unicode"] = tmp.apply(lambda x: x[0])

    dict_clean = to_unicode[to_unicode["roman"] == False].copy()

    elx = pd.DataFrame({
        "token_id": range(len(dict_clean)),
        "text_id": "elx_dict",
        "form_latin": dict_clean["transliteration"].values,
        "form_unicode": dict_clean["unicode"].values,
        "form_unicode_nospace": dict_clean["unicode"].str.replace(" ", "").values,
        "pos_raw": dict_clean["category"].values,
        "lemma": dict_clean["base"].values,
        "gloss": dict_clean["sense_hk"].values,
        "language": "elx",
        "unicode_clean": True,
    })
    if "morpheme_1" in dict_clean.columns:
        elx["morpheme_1"] = dict_clean["morpheme_1"].values
    elx["pos_unified"] = elx["pos_raw"].apply(lambda x: map_pos_unified(x, "elx"))
    elx["pos_grammatical"] = elx["pos_unified"].apply(map_pos_grammatical)
    elx["ner_tag"] = elx["pos_unified"].apply(map_ner_tag)

    nasu_df = pd.read_csv(nasu_path)
    nasu_df["translit_clean"] = nasu_df["transliteration"].apply(_normalize_for_unmatched_pass)

    def convert_nasu(text):
        if pd.isna(text) or text == "":
            return ""
        return final_pass(replace_with_unicode(text))[0]

    nasu_df["unicode"] = nasu_df["translit_clean"].apply(convert_nasu)
    nasu_docs = {
        "latin": {tid: " ".join(g["translit_clean"].dropna().astype(str))
                  for tid, g in nasu_df.groupby("id_text")},
        "unicode": {tid: " ".join(g["unicode"].dropna().astype(str))
                    for tid, g in nasu_df.groupby("id_text")},
    }
    return elx, nasu_docs


def add_unicode_representations(df: pd.DataFrame, sign_dict: Dict[str, str]) -> pd.DataFrame:
    df["form_unicode"], rate = convert_column_to_unicode(
        df["form_latin"], sign_dict, df["language"].iloc[0]
    )
    df["form_unicode_nospace"] = df["form_unicode"].str.replace(" ", "")
    df["unicode_clean"] = ~df["form_unicode"].apply(
        lambda x: bool(re.search(r"[A-Za-z]", str(x))) if pd.notna(x) else True
    )
    return df


def dataset_summary(df: pd.DataFrame, name: str = "") -> None:
    print(f"\n{'='*60}\nDataset: {name}\n{'='*60}")
    print(f"Tokens:     {len(df):,}")
    print(f"Texts:      {df['text_id'].nunique():,}")
    print(f"Vocab:      {df['form_latin'].nunique():,}")
    print(f"Lemmas:     {df['lemma'].nunique():,}")


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def load_corpora(base_path: str = BASE_PATH, languages=("akk", "sux", "elx"), sign_dict: Optional[Dict[str, str]] = None):
    """Load all language datasets in the unified schema.

    Returns a dict like {"akk": df, "sux": df, "elx": df, "_sign_dict": ..., "_documents": {...}}.

    The "_documents" key maps language → {"latin": {tid: str}, "unicode": {tid: str}}.
    Missing files are skipped with a warning rather than raising, so a reviewer
    with only partial data can still run the in-language experiments for the
    languages they have.
    """
    if sign_dict is None:
        sign_dict, manual_dict, manual_dict2, merged = load_sign_dict(base_path)
    else:
        manual_dict, manual_dict2, merged = {}, {}, None

    datasets = {}
    documents = {}

    if "akk" in languages:
        p = base_path + "alltexts_AKK.csv"
        if os.path.exists(p):
            akk = load_akkadian(p)
            akk = add_unicode_representations(akk, sign_dict)
            datasets["akk"] = akk
        else:
            print(f"[warn] {p} not found — skipping Akkadian")

    if "sux" in languages:
        p = base_path + "alltexts_SUX.csv"
        if os.path.exists(p):
            sux = load_sumerian(p)
            sux = add_unicode_representations(sux, sign_dict)
            datasets["sux"] = sux
        else:
            print(f"[warn] {p} not found — skipping Sumerian")

    if "elx" in languages:
        dp = base_path + "Elamite_Lemma-base-draft.xlsx"
        np_ = base_path + "UnTN-Nasu texts Word-level.csv"
        if os.path.exists(dp) and os.path.exists(np_) and merged is not None:
            elx, nasu_docs = load_elamite(dp, np_, sign_dict, merged, manual_dict, manual_dict2)
            datasets["elx"] = elx
            documents["elx"] = nasu_docs
        else:
            print(f"[warn] Elamite files not found at {dp} / {np_} — skipping Elamite")

    # Build per-language document corpora (Akkadian / Sumerian)
    for lang, df in datasets.items():
        if lang == "elx":
            continue
        docs_latin = {tid: " ".join(g["form_latin"].dropna().astype(str))
                      for tid, g in df.groupby("text_id")}
        docs_unicode = {tid: " ".join(g["form_unicode"].dropna().astype(str))
                        for tid, g in df.groupby("text_id")}
        documents[lang] = {"latin": docs_latin, "unicode": docs_unicode}

    return {**datasets, "_sign_dict": sign_dict, "_documents": documents, "_merged_sign_df": merged}


__all__ = [
    "BASE_PATH", "SEED", "set_seeds",
    "POS_HARMONIZATION", "ENTITY_TAGS",
    "map_pos_unified", "map_pos_grammatical", "map_ner_tag",
    "load_sign_dict", "normalize_transliteration", "transliteration_to_unicode",
    "convert_column_to_unicode",
    "load_akkadian", "load_sumerian", "load_elamite",
    "add_unicode_representations", "dataset_summary",
    "load_corpora",
]
