### Experiment C: Elamite 25K refresh

Expanded corpus: 871 docs / 111,733 tokens (original: 84 docs / 8,831; new Susa texts: 787 docs / 102,902, conversion rate 90.2%).

**In-language (5-fold document-level CV):**

| Corpus | TP F1 | Morfessor F1 |
|---|---|---|
| ELX original (paper Table 2) | 0.989 | 0.985 |
| ELX expanded (111,733 tok) | 0.923 ± 0.007 | 0.993 ± 0.000 |

**ELX as transfer source (Table 3 protocol):**

| Source | Method | -> AKK F1 | -> SUX F1 |
|---|---|---|---|
| ELX original (published) | TP | 0.959 | 0.969 |
| ELX original (published) | Morfessor | 0.304 | 0.267 |
| ELX expanded | TP (zero-shot, θ=0.45) | 0.961 | 0.967 |
| ELX expanded | TP (target-tuned) | 0.963 | 0.968 |
| ELX expanded | Morfessor (5 seeds) | 0.783 ± 0.000 | 0.712 ± 0.000 |
