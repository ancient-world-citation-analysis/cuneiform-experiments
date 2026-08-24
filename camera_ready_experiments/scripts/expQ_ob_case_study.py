"""Experiment Q: Old Babylonian case study (camera-ready commitment to
R-ShXS): boundary behavior at language-switch points vs. within-language,
on OB letters that mix syllabic Akkadian with Sumerograms.

Data: the Iltani archive word sheet (56 documents; per-word forms in
oracc_format). Writing-mode label per word: LOGOGRAPHIC (Sumerographic)
iff the transliteration is hyphen-free or dot-joined; SYLLABIC (Akkadian)
iff hyphenated. Validated against the sheet's 99 hand-marked Sumerogram
tokens. Numerals and illegible tokens are excluded from switch classes.

Protocol: TP and Morfessor trained on the full Akkadian corpus (paper
protocol; source-tuned theta via cunei-tools grid on held-out AKK), then
applied zero-shot to the OB Unicode streams (cache sign_dict conversion;
unconverted words break the stream into runs, and boundaries adjacent to
breaks are excluded). For every inter-word gold boundary we record its
class (within-syllabic, within-logographic, switch) and whether each
method predicts it; within-word false boundaries are attributed to the
containing word's class.
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "scripts")
from common import (get_corpora, lang_docs, train_morfessor,
                    predict_boundaries_morf, train_tp, tune_tp_theta,
                    save_json, save_text)

CSV = "iltani_by_word_bart_predictions - iltani_by_word_bart_predictions.csv"
SPLIT = re.compile(r"[-.{}]+")


def sign_split(w):
    return [s for s in SPLIT.split(w) if s]


def label_word(w):
    if not w or w.lower() in ("x", "#n/a"):
        return None
    if re.fullmatch(r"[0-9/]+", w):
        return "num"
    if "-" in w:
        return "syl"          # hyphenated = syllabic Akkadian
    return "log"              # hyphen-free / dot-joined = Sumerographic


def main():
    corpora = get_corpora()
    documents = corpora["documents"]
    sign_dict = corpora["sign_dict"]

    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    # heuristic validation against the 99 hand-marked Sumerograms
    ys = [r["oracc_format"] for r in rows if r["SUX"] == "Y"]
    rec = sum(1 for w in ys if label_word(w) == "log") / len(ys)
    print(f"detector recall on {len(ys)} hand-marked Sumerograms: {rec:.3f}")

    docs = defaultdict(list)
    for r in rows:
        w = r["oracc_format"].strip()
        lab = label_word(w)
        docs[r["source_B"]].append((w, lab))
    print(f"OB documents: {len(docs)}; words: {sum(len(v) for v in docs.values())}")

    # convert to unicode; build runs of consecutively-converted words
    runs = []
    conv_ok = conv_tot = 0
    for doc, words in docs.items():
        cur = []
        for w, lab in words:
            if lab is None:
                if len(cur) >= 2:
                    runs.append(cur)
                cur = []
                continue
            conv_tot += 1
            us = [sign_dict.get(s.lower()) for s in sign_split(w)]
            if all(us) and us:
                conv_ok += 1
                cur.append(("".join(us), lab))
            else:
                if len(cur) >= 2:
                    runs.append(cur)
                cur = []
        if len(cur) >= 2:
            runs.append(cur)
    print(f"conversion rate: {conv_ok/conv_tot:.1%}; runs>=2 words: {len(runs)}; "
          f"tokens in runs: {sum(len(r) for r in runs)}")

    akk_docs = lang_docs(documents, "akk")
    print(f"training TP + Morfessor on full AKK ({len(akk_docs)} docs)...")
    seg = train_tp(akk_docs, lang="akk")
    morf, akk_vocab = train_morfessor(akk_docs)

    # ---- diagnostics: how alien is OB to the NA-Akkadian model? ----
    akk_chars = set()
    akk_bigrams = set()
    for d in akk_docs:
        s = d.replace(" ", "")
        akk_chars.update(s)
        akk_bigrams.update(s[i:i+2] for i in range(len(s) - 1))
    ob_chars = Counter()
    ob_bigrams = Counter()
    ob_words = Counter()
    for run in runs:
        text = "".join(w for w, _ in run)
        ob_chars.update(text)
        ob_bigrams.update(text[i:i+2] for i in range(len(text) - 1))
        ob_words.update(w for w, _ in run)
    cseen = sum(c for ch, c in ob_chars.items() if ch in akk_chars) \
        / sum(ob_chars.values())
    bseen = sum(c for bg, c in ob_bigrams.items() if bg in akk_bigrams) \
        / sum(ob_bigrams.values())
    wseen = sum(c for w, c in ob_words.items() if w in akk_vocab) \
        / sum(ob_words.values())
    print(f"  OB tokens seen in NA-AKK training: chars {cseen:.1%}, "
          f"bigrams {bseen:.1%}, whole words {wseen:.1%}")

    # ---- theta sweep on OB (target-tuned, as Table 3 tuned cells) ----
    def tp_counts(th):
        tp = fp = fn = 0
        for run in runs:
            text = "".join(w for w, _ in run)
            gold = set()
            pos = 0
            for w, _ in run[:-1]:
                pos += len(w)
                gold.add(pos)
            pred = set(i for i in range(1, len(text))
                       if seg._transitional_prob(text[i-1], text[i]) < th)
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        return 2*p*r/max(1e-9, p+r), p, r

    print("  theta sweep on OB:")
    best = (0, None)
    for th in [0.02, 0.06, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        f1, p, r = tp_counts(th)
        print(f"    theta={th}: F1={f1:.3f} P={p:.3f} R={r:.3f}")
        if f1 > best[0]:
            best = (f1, th)
    theta = best[1]
    ngold = sum(len(r_) - 1 for r_ in runs)
    nsigns = sum(len("".join(w for w, _ in r_)) for r_ in runs)
    triv_p = ngold / (nsigns - len(runs))
    triv_f1 = 2 * triv_p / (1 + triv_p)
    print(f"  best theta on OB = {theta} (F1={best[0]:.3f}); "
          f"all-boundaries trivial F1 = {triv_f1:.3f}")

    classes = ["within-syl", "within-log", "switch"]
    hit = {m: Counter() for m in ("tp", "morf")}
    tot = Counter()
    fp = {m: Counter() for m in ("tp", "morf")}
    signs_in = Counter()
    for run in runs:
        text = "".join(w for w, _ in run)
        # gold inter-word boundary positions + class
        pos = 0
        bclass = {}
        for i, (w, lab) in enumerate(run[:-1]):
            pos += len(w)
            nlab = run[i + 1][1]
            if lab == "num" or nlab == "num":
                c = None
            elif lab == nlab:
                c = f"within-{lab}"
            else:
                c = "switch"
            bclass[pos] = c
        gold = set(bclass)
        pred_tp = set()
        for i in range(1, len(text)):
            if seg._transitional_prob(text[i - 1], text[i]) < theta:
                pred_tp.add(i)
        pred_mf = predict_boundaries_morf(morf, text)
        for p, c in bclass.items():
            if c is None:
                continue
            tot[c] += 1
            hit["tp"][c] += p in pred_tp
            hit["morf"][c] += p in pred_mf
        # within-word false boundaries by containing word class
        pos = 0
        for w, lab in run:
            for q in range(pos + 1, pos + len(w)):
                signs_in[lab] += 1
                fp["tp"][lab] += q in pred_tp
                fp["morf"][lab] += q in pred_mf
            pos += len(w)

    lines = ["### Experiment Q: OB case study (Iltani archive)", "",
             f"Detector recall on hand-marked Sumerograms: {rec:.3f} "
             f"(n={len(ys)}); conversion {conv_ok/conv_tot:.1%}; "
             f"{len(runs)} runs, {sum(len(r) for r in runs)} tokens; "
             f"OB-vs-NA seen rates: chars {cseen:.1%} / bigrams {bseen:.1%} "
             f"/ words {wseen:.1%}; best target-tuned theta={theta} "
             f"(F1={best[0]:.3f}); all-boundaries trivial F1={triv_f1:.3f}", "",
             "| boundary class | n | TP recall | Morfessor recall |",
             "|---|---|---|---|"]
    for c in classes:
        if tot[c]:
            lines.append(f"| {c} | {tot[c]} | {hit['tp'][c]/tot[c]:.3f} | "
                         f"{hit['morf'][c]/tot[c]:.3f} |")
    lines += ["", "| word class | signs | TP false-boundary rate | Morf FB rate |",
              "|---|---|---|---|"]
    for lab in ("syl", "log"):
        if signs_in[lab]:
            lines.append(f"| {lab} | {signs_in[lab]} | "
                         f"{fp['tp'][lab]/signs_in[lab]:.3f} | "
                         f"{fp['morf'][lab]/signs_in[lab]:.3f} |")
    out = "\n".join(lines)
    print(out)
    save_text(out, "expQ_ob_case_study.md")
    save_json({"recall_detector": rec, "theta": theta,
               "boundary_totals": dict(tot),
               "tp_hits": dict(hit["tp"]), "morf_hits": dict(hit["morf"]),
               "signs_in": dict(signs_in),
               "tp_fp": dict(fp["tp"]), "morf_fp": dict(fp["morf"])},
              "expQ_ob_case_study.json")


if __name__ == "__main__":
    main()
