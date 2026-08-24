### Experiment L: Morfessor transfer at source-selected corpusweight (0.5)

Zero-shot transfer with the corpusweight that in-language (source-side) CV selects (Exp K), vs the published default (cw=1.0) and untuned TP.

| Source | Target | Morf cw=1.0 (paper) | Morf cw=0.5 (source-selected) | TP (no tuning) |
|---|---|---|---|---|
| AKK | SUX | 0.981 | 0.982 ± 0.000 | 0.967 |
| AKK | ELX | 0.984 | 0.997 ± 0.000 | 0.992 |
| SUX | AKK | 0.857 | 0.977 ± 0.000 | 0.960 |
| SUX | ELX | 0.953 | 0.993 ± 0.000 | 0.993 |
| ELX | AKK | 0.304 | 0.765 ± 0.000 | 0.959 |
| ELX | SUX | 0.267 | 0.713 ± 0.000 | 0.969 |
