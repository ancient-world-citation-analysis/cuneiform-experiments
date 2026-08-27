### Experiment Q (v2): OB case study — OB_iltani corpus, ATF ground-truth writing-mode labels

132 documents, 13614 words; 50.3% Sumerographic (ATF underscore spans); v1 heuristic vs truth P=0.744/R=0.648; conversion 91.4%; 1246 runs / 11589 tokens; seen rates chars 100.0% / bigrams 98.6% / words 35.4%; TP best theta=0.02 F1=0.567; Morfessor F1=0.551 (P=0.381 R=0.996); all-boundaries trivial=0.540

| boundary class | n | TP recall | Morfessor recall |
|---|---|---|---|
| within-syl | 2450 | 0.709 | 1.000 |
| within-log | 2623 | 0.813 | 0.990 |
| switch | 5270 | 0.865 | 0.998 |

| word class | signs | TP FB rate | Morf FB rate |
|---|---|---|---|
| syl | 13423 | 0.659 | 0.962 |
| log | 4174 | 0.510 | 0.916 |