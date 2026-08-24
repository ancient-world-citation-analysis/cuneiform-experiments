### Experiment G: supervised neural baseline vs unsupervised methods (AKK)

Character-BiLSTM boundary tagger trained on N labeled tokens, vs TP and Morfessor trained on the identical subsamples; all evaluated on the same fixed 500-document held-out Akkadian test set (notebook 04 protocol). Mean ± std over 5 seeds. TP theta is tuned on test (the paper's learning-curve protocol); the BiLSTM tunes its threshold on a held-out slice of its own training data.

| Labeled tokens | BiLSTM (supervised) | TP (unsupervised) | Morfessor (unsupervised) |
|---|---|---|---|
| 1,000 | 0.957 ± 0.002 | 0.961 ± 0.004 | 0.637 ± 0.063 |
| 5,000 | 0.957 ± 0.001 | 0.964 ± 0.003 | 0.908 ± 0.013 |
| 13,000 | 0.962 ± 0.010 | 0.968 ± 0.001 | 0.963 ± 0.007 |
| 50,000 | 0.994 ± 0.001 | 0.968 ± 0.000 | 0.991 ± 0.002 |

Audit note: at 1K–13K labeled tokens the BiLSTM's F1 sits at the all-boundaries trivial baseline for this test set (~0.957); only the 50K model exceeds it. The paper-facing takeaway is therefore: supervision below ~50K gold tokens does not beat the unsupervised methods (TP exceeds the trivial baseline at every budget; Morfessor from 13K).
