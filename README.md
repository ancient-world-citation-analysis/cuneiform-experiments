# Reproduction code: *What Transfers Across Cuneiform? Script-Level Substrate and Language-Specific Regularities via Segmentation and Representation*

**Chuanjun Zhou and Adam Anderson** &middot; **EMNLP 2026 Main Conference**

**Camera-ready experiments** (review-period controls, robustness analyses, and the Old Babylonian case study) live in [`camera_ready_experiments/`](camera_ready_experiments/) with a full experiment-to-paper index.

This repository contains the code, curated data, and pre-computed result
artifacts needed to reproduce the experiments reported in the paper. The
pipeline covers three southern-Mesopotamian languages — **Akkadian (akk)**,
**Sumerian (sux)**, **Elamite (elx)** — plus a **Hittite (hit)** extension
used as a substrate-violation test.

The paper frames cuneiform as having two superimposable kinds of statistical
regularity:

* **Script-level regularities**, which are shared across languages, recoverable
  from very little segmented text, and exposed by a transitional-probability
  word-boundary segmenter (TP). Notebooks 02, 03, 04 quantify this; notebook 06
  shows it is not driven by logograms.
* **Language-specific regularities**, which do not transfer across languages
  and require representation-aware models. Notebook 05 stresses this with a
  Hittite substrate test; notebooks 07 and 08 quantify it by contrasting Latin
  transliteration against Unicode cuneiform on POS classification and
  representation-concatenation / gated-dual-encoder probes.

The companion library used by these notebooks — Unicode conversion, sign-list
loaders, the `CuneiSeg` transitional-probability segmenter — is released as a
separate package:
<https://github.com/ancient-world-citation-analysis/cunei-tools>.

---

## Notebook execution order

| # | Notebook | Paper section | Approx. runtime |
|---|---|---|---|
| 01 | `01_data_preparation.ipynb` | §3 Data & preprocessing | ~3 min (curated only); ~15 min with ORACC |
| 02 | `02_in_language_segmentation.ipynb` | §4 In-language segmentation (Table 2) | ~5 min CPU |
| 03 | `03_cross_language_transfer_matrix.ipynb` | §5 Cross-language transfer (Table 3) | ~10 min CPU |
| 04 | `04_learning_curve.ipynb` | §4.3 Learning curve (Figure 3) | ~15 min CPU |
| 05 | `05_hittite_substrate_violation.ipynb` | §6 Hittite substrate (Tables 4-5) | ~5 min CPU |
| 06 | `06_logogram_robustness.ipynb` | §7 Logogram robustness (Table 6) | ~5 min CPU |
| 07 | `07_pos_classification.ipynb` | §8.1 POS classification (Table 7) | ~10 min CPU / 3 min GPU |
| 08 | `08_representation_concatenation.ipynb` | §8.2-3 Concatenation & gated dual encoder (Table 8, Figure 4) | ~20 min CPU / 5 min GPU |

The shared setup logic (sign-list merging, POS harmonization, per-language
loaders) lives in `notebooks/_setup.py` and is imported by every numbered
notebook:

```python
from _setup import load_corpora, POS_HARMONIZATION, BASE_PATH
```

`BASE_PATH` defaults to `/path/to/cuneiform/data/` and can be overridden by
exporting `CUNEI_DATA=/your/data/dir` before launching Jupyter.

---

## Setup

The pipeline was developed and tested on Python 3.10.

```bash
git clone https://github.com/ancient-world-citation-analysis/cuneiform-experiments
cd cuneiform-experiments

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Companion library
pip install git+https://github.com/ancient-world-citation-analysis/cunei-tools

# Point the notebooks at your data directory
export CUNEI_DATA=/path/to/cuneiform/data/
jupyter lab notebooks/
```

The deep-learning notebooks (07, 08, and the optional `ByT5_Cuneiform.ipynb`)
will use a CUDA GPU automatically if one is available; everything else is
CPU-only.

---

## Data sources

This repository ships only data that is **not** publicly available elsewhere.
Two large public sources must be downloaded separately to run the full
Akkadian / Sumerian pipeline:

1. **ORACC corpus dump** — `finaldf.csv` (~1.5 GB) on Zenodo:
   <https://zenodo.org/records/10794626/files/finaldf.csv>
   After download, run `python scripts/filter_oracc_data.py` from the
   directory containing `finaldf.csv` to produce `alltexts_AKK.csv` and
   `alltexts_SUX.csv`. Move both files into `$CUNEI_DATA`.
2. **ETCSL (Sumerian Literature)** — used as a complementary source for
   Sumerian text: <https://etcsl.orinst.ox.ac.uk/>.

Sign-list resources are pulled from public GitHub mirrors at runtime (no
manual download needed):

- *Nuolenna* sign list — `situx/Nuolenna`
- *Akkademia* sign list — `gaigutherz/Akkademia`

The remaining curated resources (Hittite glossed corpus, Elamite lemma base,
Utu-Našu word-level corpus, manual sign corrections) are bundled under `data/`.

---

## Reproducing Table 3 (cross-language transfer matrix)

Table 3 is the headline result. To reproduce it:

```bash
export CUNEI_DATA=/path/to/cuneiform/data/
jupyter nbconvert --to notebook --execute \
  notebooks/01_data_preparation.ipynb \
  notebooks/03_cross_language_transfer_matrix.ipynb
```

Outputs written:

```
outputs/table3_transfer_matrix.csv      # F1 matrix
outputs/table3_transfer_matrix_ci.csv   # bootstrap CIs
outputs/figure2_transfer_bars.png       # per-row bars
outputs/table3_morfessor_transfer.json  # Morfessor baseline row
```

Expected runtime end-to-end: ~12 minutes on CPU.

---

## Repository layout

```
cuneiform-experiments/
├── README.md                 # this file
├── requirements.txt
├── notebooks/
│   ├── _setup.py             # shared setup utilities (imported by 01-08)
│   ├── 01_data_preparation.ipynb
│   ├── 02_in_language_segmentation.ipynb
│   ├── 03_cross_language_transfer_matrix.ipynb
│   ├── 04_learning_curve.ipynb
│   ├── 05_hittite_substrate_violation.ipynb
│   ├── 06_logogram_robustness.ipynb
│   ├── 07_pos_classification.ipynb
│   ├── 08_representation_concatenation.ipynb
│   ├── ByT5_Cuneiform.ipynb              # ByT5 fine-tuning baseline (optional)
│   ├── ElamiteExp4_Lemmatization.ipynb   # lemmatization deep-dive (Appendix)
│   ├── ElamiteExp5_LLM_Evaluation.ipynb  # LLM eval (Appendix)
│   ├── ElamiteExperiments.ipynb          # Elamite-only embedding analysis (Appendix)
│   ├── ElamiteNewData.ipynb              # preprocessing reference (Appendix)
│   ├── GraphAnalysis_ThreeLanguages.ipynb # graph analysis figure (Appendix)
│   └── archive/                          # superseded monoliths kept for traceability
├── scripts/
│   └── filter_oracc_data.py
├── data/
│   ├── 7000_hitt_txts_wGloss.csv
│   ├── Elamite_Lemma-base-draft.xlsx
│   ├── Elamite-Lemma-Base-Unicode.csv
│   ├── UnTN-Nasu texts Word-level.csv
│   ├── to_unicode_nasu_clean.csv
│   ├── unmatchednew*.csv
│   └── three_lang_results.json
└── outputs/
    ├── README.md             # paper-element ↔ file mapping
    ├── figure3_learning_curve.{png,pdf}
    ├── learning_curve_results.json
    ├── table3_morfessor_transfer.json
    └── table4_hittite_in_language.json
```

---

## Notes for reviewers

- All randomness uses `SEED = 42`. Every numbered notebook seeds `random`,
  `numpy`, and `torch` at the top.
- Notebook outputs (figures, CSVs, JSON) are written to `outputs/` with
  filenames mirroring the paper artifact they correspond to. See
  `outputs/README.md` for the table-by-table mapping.
- All previous Colab Drive paths (`/content/drive/MyDrive/…`) have been
  rewritten to relative `data/` paths. Setting `$CUNEI_DATA` once is enough.
- Author / affiliation strings have been stripped throughout: the notebooks
  contain no `displayName` / `userId` metadata, no institutional names, and
  no personal email addresses.
- The Appendix notebooks (`ElamiteExp4_Lemmatization`, `ElamiteExp5_LLM_Evaluation`,
  `ElamiteExperiments`, `ElamiteNewData`, `GraphAnalysis_ThreeLanguages`,
  `ByT5_Cuneiform`) cover extended analyses that are not in the main paper
  but are referenced for completeness.
- The two superseded monoliths (`ThreeLanguagePipeline.ipynb`, the original
  `HittitePipeline.ipynb`) are preserved under `notebooks/archive/` so a
  reviewer can verify the split was content-preserving.
