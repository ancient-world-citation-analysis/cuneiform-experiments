### Experiment I: cross-language transfer of the supervised BiLSTM (AKK source)

**AUDIT NOTE (do not cite as learned transfer):** on foreign-language targets the tagger sees mostly unseen signs (<unk>) and its output coincides with the all-boundaries trivial policy; the F1s below match that baseline, not learned generalization. TP zero-shot cells use the cunei-tools unseen-sign convention (see expP for the published-convention values).

Zero-shot on full target corpora; no target-language tuning anywhere.

| Source budget | Method | → SUX | → ELX |
|---|---|---|---|
| AKK 13K | BiLSTM (supervised) | 0.968 ± 0.000 | 0.994 ± 0.000 |
| AKK 13K | TP (unsup., zero-shot) | 0.967 ± 0.000 | 0.994 ± 0.000 |
| AKK 13K | Morfessor (unsup.) | 0.753 ± 0.021 | 0.961 ± 0.009 |
| AKK 50K | BiLSTM (supervised) | 0.974 ± 0.000 | 0.993 ± 0.001 |
| AKK 50K | TP (unsup., zero-shot) | 0.967 ± 0.000 | 0.994 ± 0.000 |
| AKK 50K | Morfessor (unsup.) | 0.906 ± 0.003 | 0.977 ± 0.001 |
