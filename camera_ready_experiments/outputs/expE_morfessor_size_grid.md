### Experiment E: Morfessor transfer vs source-corpus size

Zero-shot transfer F1 (mean ± std over 5 seeds; document-level subsampling; corpusweight=1.0). 'full' rows use the entire source corpus (seeds then only affect Morfessor's internal training order).

| Source | Size | → AKK | → SUX | → ELX |
|---|---|---|---|---|
| AKK | 5K (5,741 tok) | — | 0.615 ± 0.022 | 0.826 ± 0.054 |
| AKK | 13K (14,086 tok) | — | 0.743 ± 0.019 | 0.955 ± 0.016 |
| AKK | 25K (25,396 tok) | — | 0.826 ± 0.012 | 0.973 ± 0.002 |
| AKK | full (2,441,623 tok) | — | 0.981 ± 0.000 | 0.984 ± 0.000 |
| SUX | 5K (6,485 tok) | 0.482 ± 0.037 | — | 0.554 ± 0.070 |
| SUX | 13K (14,175 tok) | 0.595 ± 0.031 | — | 0.710 ± 0.091 |
| SUX | 25K (28,925 tok) | 0.662 ± 0.028 | — | 0.757 ± 0.071 |
| SUX | full (337,972 tok) | 0.858 ± 0.001 | — | 0.953 ± 0.000 |
| ELX-expanded | 5K (5,326 tok) | 0.517 ± 0.023 | 0.414 ± 0.027 | — |
| ELX-expanded | 13K (14,219 tok) | 0.587 ± 0.036 | 0.478 ± 0.020 | — |
| ELX-expanded | 25K (25,819 tok) | 0.651 ± 0.018 | 0.527 ± 0.019 | — |
| ELX-expanded | full (111,733 tok) | 0.783 ± 0.000 | 0.712 ± 0.000 | — |
| ELX-original | 5K (5,045 tok) | 0.273 ± 0.009 | 0.247 ± 0.009 | — |
| ELX-original | full (8,831 tok) | 0.304 ± 0.000 | 0.267 ± 0.000 | — |
