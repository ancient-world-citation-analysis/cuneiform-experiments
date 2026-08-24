### Experiment K: source-side corpusweight selection

In-language 5-fold CV F1 (the only tuning admissible for zero-shot transfer) at each corpusweight:

| corpusweight | ELX | SUX | AKK |
|---|---|---|---|
| 0.25 | 0.9983 | 0.9941 | 0.9962 |
| 0.5 | 0.9986 | 0.9944 | 0.9965 |
| 1.0 | 0.9846 | 0.9930 | 0.9964 |
| 2.0 | 0.0633 | 0.0761 | 0.1217 |
| 4.0 | 0.0565 | 0.0645 | 0.0652 |

Source-side selection picks cw=0.5 in all three languages, but only ELX's preference is decisive (+0.0140 over cw=1.0, beyond fold noise); AKK (+0.0001) and SUX (+0.0014) prefer 0.5 by margins within fold-level std. The robust statement: cw=1.0 is never preferred, and ELX clearly selects 0.5.
