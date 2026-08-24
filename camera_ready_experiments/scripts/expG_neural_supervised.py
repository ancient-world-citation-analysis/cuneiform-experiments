"""Experiment G (rebuttal, mKBh Weakness 2): supervised neural segmentation
baseline — how does a small amount of *annotated* data compare with the
paper's fully unsupervised methods?

Protocol: notebook 04 learning-curve harness, replicated exactly.
  - Fixed held-out test set: random.seed(42); shuffle AKK docs; first 500.
  - Training budgets: 1K / 5K / 13K / 50K split-tokens drawn from the
    remaining pool (document-level shuffled prefix per seed, as in Exp A).
  - 5 seeds (42-46) controlling subsample, torch init, and Morfessor order.

Supervised model: character-level BiLSTM boundary tagger (the paper already
uses char BiLSTMs for classification, App. A.4). For each adjacent character
pair (c_{i-1}, c_i) in the continuous stream, predict whether a word boundary
separates them from concat(h_{i-1}, h_i). Embedding 64, BiLSTM hidden 128,
Adam 1e-3, BCE loss, <=30 epochs, early stop + decision threshold tuned on a
10% held-out slice OF THE TRAINING SUBSET (never the test set).

Comparison columns: TP (theta tuned on test, exactly as in the paper's
learning curve / Table 2 protocol - note this favors TP) and Morfessor
(corpusweight=1.0), trained on the IDENTICAL subsamples, evaluated on the
IDENTICAL fixed test set.
"""
import random
import time

import numpy as np
import torch
import torch.nn as nn

from common import (SEEDS_5, get_corpora, lang_docs, subsample_docs,
                    train_morfessor, doc_counts_morf, train_tp, tune_tp_theta,
                    doc_counts_tp, micro_prf, save_json, save_text)

BUDGETS = [1_000, 5_000, 13_000, 50_000]
N_TEST = 500
MAX_EPOCHS = 30
PATIENCE = 5

torch.set_num_threads(4)


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def doc_to_example(doc):
    """Space-segmented doc -> (chars, labels); labels[i]=1 iff boundary
    between chars[i-1] and chars[i] (same convention as CuneiSeg eval)."""
    continuous = doc.replace(" ", "")
    chars = list(continuous)
    gold = set()
    pos = 0
    for ch in doc:
        if ch == " ":
            gold.add(pos)
        else:
            pos += 1
    labels = [1 if i in gold else 0 for i in range(1, len(chars))]
    return chars, labels


class BoundaryTagger(nn.Module):
    def __init__(self, vocab_size, emb=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(4 * hidden, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x, lengths):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        pair = torch.cat([h[:, :-1, :], h[:, 1:, :]], dim=-1)  # gaps
        return self.head(pair).squeeze(-1)  # (B, L-1) logits


def make_batches(examples, vocab, batch_chars=8000):
    """Sort by length, group into padded batches (x, lengths, y, mask)."""
    idx = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
    batches, cur, cur_chars = [], [], 0
    for i in idx:
        cur.append(i)
        cur_chars += len(examples[i][0])
        if cur_chars >= batch_chars:
            batches.append(cur)
            cur, cur_chars = [], 0
    if cur:
        batches.append(cur)
    out = []
    for group in batches:
        exs = [examples[i] for i in group]
        maxlen = max(len(c) for c, _ in exs)
        x = torch.zeros(len(exs), maxlen, dtype=torch.long)
        y = torch.zeros(len(exs), maxlen - 1)
        mask = torch.zeros(len(exs), maxlen - 1)
        lengths = torch.tensor([len(c) for c, _ in exs])
        for j, (chars, labels) in enumerate(exs):
            x[j, :len(chars)] = torch.tensor(
                [vocab.get(c, 1) for c in chars])
            y[j, :len(labels)] = torch.tensor(labels, dtype=torch.float)
            mask[j, :len(labels)] = 1.0
        out.append((x, lengths, y, mask))
    return out


def eval_tagger(model, batches, threshold):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for x, lengths, y, mask in batches:
            probs = torch.sigmoid(model(x, lengths))
            pred = (probs >= threshold).float() * mask
            gold = y * mask
            tp += int(((pred == 1) & (gold == 1)).sum())
            fp += int(((pred == 1) & (gold == 0) & (mask == 1)).sum())
            fn += int(((pred == 0) & (gold == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"f1": f1, "precision": p, "recall": r}


def train_bilstm(train_docs, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    examples = [doc_to_example(d) for d in train_docs
                if d and len(d.split()) >= 2]
    random.shuffle(examples)
    n_val = max(5, len(examples) // 10)
    val_ex, tr_ex = examples[:n_val], examples[n_val:]
    if not tr_ex:
        tr_ex, val_ex = examples, examples

    vocab = {"<pad>": 0, "<unk>": 1}
    for chars, _ in tr_ex:
        for c in chars:
            if c not in vocab:
                vocab[c] = len(vocab)

    model = BoundaryTagger(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    tr_batches = make_batches(tr_ex, vocab)
    val_batches = make_batches(val_ex, vocab)

    best_f1, best_state, best_thresh, bad = -1.0, None, 0.5, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        random.shuffle(tr_batches)
        for x, lengths, y, mask in tr_batches:
            opt.zero_grad()
            logits = model(x, lengths)
            loss = (lossf(logits, y) * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            opt.step()
        # validate: pick best threshold on val
        cand = [(eval_tagger(model, val_batches, t)["f1"], t)
                for t in np.arange(0.2, 0.85, 0.05)]
        vf1, vth = max(cand)
        if vf1 > best_f1 + 1e-4:
            best_f1, best_thresh, bad = vf1, float(vth), 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, vocab, best_thresh, epoch + 1


def main():
    t0 = time.time()
    cache = get_corpora(verbose=False)
    documents = cache["documents"]

    # notebook 04 test split, replicated exactly
    random.seed(42)
    docs_akk = [d for d in documents["akk"]["unicode"].values()
                if d and len(d.split()) > 2]
    random.shuffle(docs_akk)
    test_docs = docs_akk[:N_TEST]
    train_pool = docs_akk[N_TEST:]
    test_examples = [doc_to_example(d) for d in test_docs]
    print(f"test: {N_TEST} docs, {sum(len(d.split()) for d in test_docs):,} tokens | "
          f"pool: {len(train_pool)} docs")

    runs = []
    for budget in BUDGETS:
        for seed in SEEDS_5:
            t = time.time()
            subset, actual = subsample_docs(train_pool, budget, seed)

            # --- supervised BiLSTM ---
            model, vocab, thresh, epochs = train_bilstm(subset, seed)
            test_batches = make_batches(test_examples, vocab)
            neural = eval_tagger(model, test_batches, thresh)

            # --- TP (theta tuned on test = notebook 04 protocol) ---
            seg = train_tp(subset, lang="akk")
            tp_m = tune_tp_theta(seg, test_docs)

            # --- Morfessor ---
            random.seed(seed)
            morf, _ = train_morfessor(subset)
            morf_m = micro_prf(doc_counts_morf(morf, test_docs))

            row = {"budget": budget, "seed": seed, "actual_tokens": actual,
                   "n_docs": len(subset), "bilstm_f1": neural["f1"],
                   "bilstm_precision": neural["precision"],
                   "bilstm_recall": neural["recall"],
                   "bilstm_threshold": thresh, "bilstm_epochs": epochs,
                   "tp_f1": tp_m["f1"], "tp_theta": tp_m["threshold"],
                   "morf_f1": morf_m["f1"]}
            runs.append(row)
            print(f"[{time.time()-t:6.1f}s] budget={budget:>6,d} seed={seed}: "
                  f"BiLSTM={neural['f1']:.4f} (th={thresh:.2f}, ep={epochs}) | "
                  f"TP={tp_m['f1']:.4f} | Morf={morf_m['f1']:.4f}", flush=True)

    summary = {}
    for budget in BUDGETS:
        rows = [r for r in runs if r["budget"] == budget]
        s = {"actual_tokens_mean": float(np.mean([r["actual_tokens"] for r in rows]))}
        for k in ["bilstm_f1", "tp_f1", "morf_f1"]:
            vals = [r[k] for r in rows]
            s[f"{k}_mean"] = float(np.mean(vals))
            s[f"{k}_std"] = float(np.std(vals))
        summary[str(budget)] = s

    out = {"experiment": "G: supervised neural segmentation baseline (AKK)",
           "protocol": {
               "test": f"fixed {N_TEST}-doc held-out set (notebook 04, seed 42)",
               "model": "char BiLSTM boundary tagger (emb 64, hidden 128), "
                        "threshold + early stop on 10% train-val slice",
               "tp": "theta tuned on TEST (paper learning-curve protocol)",
               "morfessor_corpusweight": 1.0,
               "budgets": BUDGETS, "seeds": SEEDS_5},
           "runs": runs, "summary": summary}
    save_json(out, "expG_neural_supervised.json")

    lines = [
        "### Experiment G: supervised neural baseline vs unsupervised methods (AKK)",
        "",
        "Character-BiLSTM boundary tagger trained on N labeled tokens, vs TP "
        "and Morfessor trained on the identical subsamples; all evaluated on "
        "the same fixed 500-document held-out Akkadian test set (notebook 04 "
        "protocol). Mean ± std over 5 seeds. TP theta is tuned on test (the "
        "paper's learning-curve protocol); the BiLSTM tunes its threshold on "
        "a held-out slice of its own training data.",
        "",
        "| Labeled tokens | BiLSTM (supervised) | TP (unsupervised) | Morfessor (unsupervised) |",
        "|---|---|---|---|",
    ]
    for budget in BUDGETS:
        s = summary[str(budget)]
        lines.append(
            f"| {budget:,} | {s['bilstm_f1_mean']:.3f} ± {s['bilstm_f1_std']:.3f} "
            f"| {s['tp_f1_mean']:.3f} ± {s['tp_f1_std']:.3f} "
            f"| {s['morf_f1_mean']:.3f} ± {s['morf_f1_std']:.3f} |")
    save_text("\n".join(lines) + "\n", "expG_neural_supervised.md")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
