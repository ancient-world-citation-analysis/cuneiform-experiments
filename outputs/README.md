# Outputs index

Pre-computed artifacts for the experiments. Every file here is regenerable
from a corresponding notebook; they are checked in so that a reviewer with
no GPU and no Zenodo download can still inspect the headline numbers.

| File | Paper element | Produced by |
|---|---|---|
| `dataset_summary.json` | §3 Table 1 (token / type counts per language) | `01_data_preparation.ipynb` |
| `table2_in_language_f1.csv` | §4 Table 2 (in-language F1) | `02_in_language_segmentation.ipynb` |
| `threshold_sweep_akk.csv`, `threshold_sweep_sux.csv`, `threshold_sweep_elx.csv` | §4 Figure 1 (F1 vs threshold per language) | `02_in_language_segmentation.ipynb` |
| `morfessor_corpusweight_sweep.json` | §4 robustness column of Table 2 | `02_in_language_segmentation.ipynb` |
| `table3_transfer_matrix.csv` | §5 Table 3 (cross-language transfer F1) | `03_cross_language_transfer_matrix.ipynb` |
| `table3_transfer_matrix_ci.csv` | §5 Table 3 bootstrap CIs | `03_cross_language_transfer_matrix.ipynb` |
| `figure2_transfer_bars.png` | §5 Figure 2 | `03_cross_language_transfer_matrix.ipynb` |
| `table3_morfessor_transfer.json` | §5 Morfessor baseline row of Table 3 | `03_cross_language_transfer_matrix.ipynb` |
| `figure3_learning_curve.png`, `figure3_learning_curve.pdf` | §4.3 Figure 3 | `04_learning_curve.ipynb` |
| `learning_curve_results.json` | §4.3 Figure 3 raw numbers | `04_learning_curve.ipynb` |
| `table4_hittite_in_language.json` | §6 Table 4 (Hittite in-language) | `05_hittite_substrate_violation.ipynb` |
| `table5_hittite_substrate_logogram.json` | §6 Table 5 (logogram-rate substrate analysis) | `05_hittite_substrate_violation.ipynb` |
| `table6_logogram_robustness.csv` | §7 Table 6 (logogram-masking robustness, AKK / SUX) | `06_logogram_robustness.ipynb` |
| `table7_pos_classification.csv` | §8.1 Table 7 (POS classification) | `07_pos_classification.ipynb` |
| `table8_concatenation.csv` | §8.2 Table 8 (concatenation rows) | `08_representation_concatenation.ipynb` |
| `table8_gated_dual_encoder.json` | §8.3 Table 8 (gated dual encoder rows) | `08_representation_concatenation.ipynb` |
| `figure4_gating_weights.png` | §8.3 Figure 4 | `08_representation_concatenation.ipynb` |

Files that already exist in this directory are the most recent re-runs;
re-executing the originating notebook will overwrite them. Files listed in
the table but not yet present have not been re-run after the last anonymization
pass.
