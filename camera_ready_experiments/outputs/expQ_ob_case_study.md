### Experiment Q: OB case study (Iltani archive)

Detector recall on hand-marked Sumerograms: 0.970 (n=99); conversion 79.0%; 768 runs, 5462 tokens; OB-vs-NA seen rates: chars 100.0% / bigrams 97.7% / words 39.8%; best target-tuned theta=0.02 (F1=0.559); all-boundaries trivial F1=0.534

| boundary class | n | TP recall | Morfessor recall |
|---|---|---|---|
| within-syl | 463 | 0.691 | 1.000 |
| within-log | 1283 | 0.822 | 0.999 |
| switch | 2577 | 0.847 | 1.000 |

| word class | signs | TP false-boundary rate | Morf FB rate |
|---|---|---|---|
| syl | 6009 | 0.685 | 0.962 |
| log | 2148 | 0.543 | 0.917 |