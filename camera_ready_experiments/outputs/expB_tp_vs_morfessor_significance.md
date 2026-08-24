### Experiment B: In-language TP vs Morfessor, paired bootstrap

5-fold document-level CV (identical folds for both methods); document-level paired bootstrap on the F1 difference, 2,000 resamples, two-sided (same significance machinery as the paper's classification experiments).

| Language | TP F1 (CV) | Morfessor F1 (CV) | ΔF1 (TP−Morf) | 95% CI | p (two-sided) |
|---|---|---|---|---|---|
| Akkadian (n=13502) | 0.971 ± 0.002 | 0.996 ± 0.000 | -0.0253 | [-0.0265, -0.0242] | < 0.001 |
| Sumerian (n=394) | 0.971 ± 0.002 | 0.993 ± 0.002 | -0.0214 | [-0.0230, -0.0197] | < 0.001 |
| Elamite (n=84) | 0.989 ± 0.005 | 0.985 ± 0.010 | +0.0047 | [-0.0020, +0.0117] | = 0.194 |

F1 columns: mean ± std over the 5 CV folds (Table 2 style). ΔF1, CI and p are computed on the CV-pooled per-document counts.
