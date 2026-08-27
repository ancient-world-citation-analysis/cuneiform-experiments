"""Experiment Q (v2): Old Babylonian case study on the OB_iltani corpus.

v2 replaces v1's data and labels: 132 CDLI documents / 13,614 words
(letters AND administrative texts) from the atf_by_word export, with
GROUND-TRUTH writing-mode labels from the ATF underscore markup
(IN_UNDERSCORE_SUX: tokens inside _..._ spans are Sumerographic).
v1 used a 56-document subset with an orthographic heuristic label;
the heuristic is now VALIDATED against the ATF truth and reported.

Protocol unchanged: TP and Morfessor trained on the full Neo-Assyrian
Akkadian corpus, applied zero-shot to OB Unicode streams (cache
sign-dict conversion; damaged tokens and conversion failures break the
stream into runs; runs >= 2 words analyzed). TP threshold target-tuned
as in Table 3's tuned cells. Per gold inter-word boundary: class =
within-syl / within-log / switch; within-word false boundaries
attributed to the containing word's class.
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "scripts")
from common import (get_corpora, lang_docs, train_morfessor,
                    predict_boundaries_morf, train_tp, save_json,
                    save_text)

CSV = "iltani_by_word_bart_predictions - OB_iltani.csv"
SPLIT = re.compile(r"[-.{}()\s]+")
SUBS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
DAMAGED = re.compile(r"^[x#?\[\]<>…\.]+$")


def components(w):
    return [s for s in SPLIT.split(w.lower().translate(SUBS)) if s]


def heuristic_log(w):
    """v1's orthographic rule, for validation against ATF truth."""
    if "-" in w:
        return False
    return not re.fullmatch(r"[0-9/]+", w)


def main():
    corpora = get_corpora()
    documents = corpora["documents"]
    sign_dict = corpora["sign_dict"]

    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    rows.sort(key=lambda r: (r["CDLI_ID"], int(r["LINE_ORDER"] or 0),
                             int(r["WORD_ID"] or 0)))
    print(f"OB_iltani: {len(rows)} words, "
          f"{len(set(r['CDLI_ID'] for r in rows))} documents")

    # heuristic validation against ATF ground truth
    tp_ = fp_ = fn_ = 0
    n_lab = Counter()
    for r in rows:
        w = r["ORACC_FORMAT"].strip()
        if not w or DAMAGED.match(w):
            continue
        truth = r["IN_UNDERSCORE_SUX"] == "TRUE"
        n_lab["log" if truth else "syl"] += 1
        pred = heuristic_log(w)
        tp_ += pred and truth
        fp_ += pred and not truth
        fn_ += (not pred) and truth
    hp = tp_ / max(1, tp_ + fp_)
    hr = tp_ / max(1, tp_ + fn_)
    print(f"label distribution (ATF truth): {dict(n_lab)} "
          f"({n_lab['log']/(n_lab['log']+n_lab['syl']):.1%} Sumerographic)")
    print(f"v1 heuristic vs ATF truth: P={hp:.3f} R={hr:.3f}")

    # unicode conversion + runs (break at doc/surface change, damage,
    # or conversion failure)
    runs, cur = [], []
    conv_ok = conv_tot = 0
    prev_key = None
    for r in rows:
        key = (r["CDLI_ID"], r["SURFACE"])
        if key != prev_key:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
            prev_key = key
        w = r["ORACC_FORMAT"].strip()
        if not w or DAMAGED.match(w):
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
            continue
        lab = "log" if r["IN_UNDERSCORE_SUX"] == "TRUE" else "syl"
        conv_tot += 1
        us = [sign_dict.get(s) for s in components(w)]
        if us and all(us):
            conv_ok += 1
            cur.append(("".join(us), lab))
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
    if len(cur) >= 2:
        runs.append(cur)
    n_tok = sum(len(r_) for r_ in runs)
    print(f"conversion {conv_ok/conv_tot:.1%}; runs>=2: {len(runs)}; "
          f"tokens in runs: {n_tok}")

    akk_docs = lang_docs(documents, "akk")
    print(f"training TP + Morfessor on full AKK ({len(akk_docs)} docs)...")
    seg = train_tp(akk_docs, lang="akk")
    morf, akk_vocab = train_morfessor(akk_docs)

    # diagnostics: how alien is OB to the NA model?
    akk_chars, akk_bigrams = set(), set()
    for d in akk_docs:
        s = d.replace(" ", "")
        akk_chars.update(s)
        akk_bigrams.update(s[i:i+2] for i in range(len(s) - 1))
    ob_chars, ob_bigrams, ob_words = Counter(), Counter(), Counter()
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
    print(f"OB seen in NA-AKK training: chars {cseen:.1%}, "
          f"bigrams {bseen:.1%}, whole words {wseen:.1%}")

    def tp_counts(th):
        tp = fp = fn = 0
        for run in runs:
            text = "".join(w for w, _ in run)
            gold, pos = set(), 0
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
        return 2 * p * r / max(1e-9, p + r), p, r

    print("theta sweep on OB:")
    best = (0, None)
    for th in [0.02, 0.06, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        f1, p, r = tp_counts(th)
        print(f"  theta={th}: F1={f1:.3f} P={p:.3f} R={r:.3f}")
        if f1 > best[0]:
            best = (f1, th)
    theta = best[1]
    ngold = sum(len(r_) - 1 for r_ in runs)
    nsigns = sum(len("".join(w for w, _ in r_)) for r_ in runs)
    triv_p = ngold / (nsigns - len(runs))
    triv_f1 = 2 * triv_p / (1 + triv_p)
    print(f"best theta={theta} (F1={best[0]:.3f}); "
          f"all-boundaries trivial F1={triv_f1:.3f}")

    # Morfessor overall F1 on the same runs
    mtp = mfp = mfn = 0
    hit = {m: Counter() for m in ("tp", "morf")}
    tot = Counter()
    fpc = {m: Counter() for m in ("tp", "morf")}
    signs_in = Counter()
    for run in runs:
        text = "".join(w for w, _ in run)
        pos, bclass = 0, {}
        for i, (w, lab) in enumerate(run[:-1]):
            pos += len(w)
            nlab = run[i + 1][1]
            bclass[pos] = ("switch" if lab != nlab else f"within-{lab}")
        gold = set(bclass)
        pred_tp = set(i for i in range(1, len(text))
                      if seg._transitional_prob(text[i-1], text[i]) < theta)
        pred_mf = predict_boundaries_morf(morf, text)
        mtp += len(pred_mf & gold)
        mfp += len(pred_mf - gold)
        mfn += len(gold - pred_mf)
        for p_, c in bclass.items():
            tot[c] += 1
            hit["tp"][c] += p_ in pred_tp
            hit["morf"][c] += p_ in pred_mf
        pos = 0
        for w, lab in run:
            for q in range(pos + 1, pos + len(w)):
                signs_in[lab] += 1
                fpc["tp"][lab] += q in pred_tp
                fpc["morf"][lab] += q in pred_mf
            pos += len(w)
    mp = mtp / max(1, mtp + mfp)
    mr = mtp / max(1, mtp + mfn)
    mf1 = 2 * mp * mr / max(1e-9, mp + mr)

    lines = ["### Experiment Q (v2): OB case study — OB_iltani corpus, "
             "ATF ground-truth writing-mode labels", "",
             f"{len(set(r['CDLI_ID'] for r in rows))} documents, "
             f"{len(rows)} words; {n_lab['log']/(n_lab['log']+n_lab['syl']):.1%} "
             f"Sumerographic (ATF underscore spans); v1 heuristic vs truth "
             f"P={hp:.3f}/R={hr:.3f}; conversion {conv_ok/conv_tot:.1%}; "
             f"{len(runs)} runs / {n_tok} tokens; seen rates chars "
             f"{cseen:.1%} / bigrams {bseen:.1%} / words {wseen:.1%}; "
             f"TP best theta={theta} F1={best[0]:.3f}; Morfessor F1={mf1:.3f} "
             f"(P={mp:.3f} R={mr:.3f}); all-boundaries trivial={triv_f1:.3f}",
             "",
             "| boundary class | n | TP recall | Morfessor recall |",
             "|---|---|---|---|"]
    for c in ("within-syl", "within-log", "switch"):
        if tot[c]:
            lines.append(f"| {c} | {tot[c]} | {hit['tp'][c]/tot[c]:.3f} | "
                         f"{hit['morf'][c]/tot[c]:.3f} |")
    lines += ["", "| word class | signs | TP FB rate | Morf FB rate |",
              "|---|---|---|---|"]
    for lab in ("syl", "log"):
        if signs_in[lab]:
            lines.append(f"| {lab} | {signs_in[lab]} | "
                         f"{fpc['tp'][lab]/signs_in[lab]:.3f} | "
                         f"{fpc['morf'][lab]/signs_in[lab]:.3f} |")
    out = "\n".join(lines)
    print(out)
    save_text(out, "expQ_ob_case_study.md")
    save_json({"docs": len(set(r['CDLI_ID'] for r in rows)),
               "heuristic_p": hp, "heuristic_r": hr,
               "log_share": n_lab["log"]/(n_lab["log"]+n_lab["syl"]),
               "conversion": conv_ok/conv_tot, "theta": theta,
               "tp_f1": best[0], "morf_f1": mf1, "trivial_f1": triv_f1,
               "seen": {"chars": cseen, "bigrams": bseen, "words": wseen},
               "boundary_totals": dict(tot),
               "tp_hits": dict(hit["tp"]), "morf_hits": dict(hit["morf"]),
               "signs_in": dict(signs_in),
               "tp_fp": dict(fpc["tp"]), "morf_fp": dict(fpc["morf"])},
              "expQ_ob_case_study.json")


if __name__ == "__main__":
    main()
