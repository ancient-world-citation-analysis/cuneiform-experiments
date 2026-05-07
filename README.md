# Reproduction Code for: Word Boundaries Without Dictionaries

**Anonymous Authors** &middot; *Anonymous Affiliation*

**Anonymous mirror for review:** <https://anonymous.4open.science/r/cuneiform-experiments>

This repository contains the code and curated data needed to reproduce the
experiments reported in *Word Boundaries Without Dictionaries*. The pipeline
covers three cuneiform languages — **Elamite (elx)**, **Akkadian (akk)**, and
**Sumerian (sux)** — and compares Latin transliteration against Unicode
cuneiform across five tasks: embedding evaluation, transitional-probability
word-boundary inference, POS classification, lemmatization, and LLM
evaluation.

The companion library used by the notebooks (Unicode conversion, sign-list
loaders, POS harmonization helpers) is available as a separate anonymous
release: <https://anonymous.4open.science/r/cunei-tools>.

---

## Repository layout

```
cuneiform-experiments/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── ThreeLanguagePipeline.ipynb        # master pipeline: all 5 experiments × 3 languages
│   ├── ElamiteExperiments.ipynb           # Elamite-focused deep dive (Exp 1–3)
│   ├── ElamiteExp4_Lemmatization.ipynb    # standalone Elamite lemmatization
│   ├── ElamiteExp5_LLM_Evaluation.ipynb   # standalone Elamite LLM eval
│   ├── GraphAnalysis_ThreeLanguages.ipynb # cosine-similarity edge graphs
│   ├── HittitePipeline.ipynb              # Hittite-only run (additional language)
│   ├── ByT5_Cuneiform.ipynb               # ByT5 fine-tuning comparison
│   └── ElamiteNewData.ipynb               # data preprocessing / Unicode conversion
├── scripts/
│   └── filter_oracc_data.py               # turns Zenodo finaldf.csv → AKK / SUX subsets
└── data/
    ├── 7000_hitt_txts_wGloss.csv          # Hittite glossed texts (curated)
    ├── Elamite_Lemma-base-draft.xlsx      # Elamite lemma base (curated)
    ├── Elamite-Lemma-Base-Unicode.csv     # Unicode-converted lemma base
    ├── UnTN-Nasu texts Word-level.csv     # Utu-Našu word-level corpus
    ├── to_unicode_nasu_clean.csv          # cleaned Nasu transliterations
    ├── unmatchednew - solonew.csv         # manual sign mapping (curated)
    ├── unmatchednew_AAedit - unmatchednew.csv
    ├── unmatchednewfinal.csv
    └── three_lang_results.json            # cached run output (regenerable)
```

## Paper artifact → notebook map

The main paper experiments are all driven from
`notebooks/ThreeLanguagePipeline.ipynb`. The standalone notebooks contain the
extended analyses and per-language deep-dives referenced in the paper.

| Paper artifact | Topic | Produced by |
|---|---|---|
| Dataset statistics table | Per-language token / type counts | `ThreeLanguagePipeline.ipynb` § 5 (Dataset Summary) |
| Rare-token analysis table | Type/token rates and rare-token coverage | `ThreeLanguagePipeline.ipynb` § *Rare Token Analysis* |
| Cross-language transfer table | Transfer between Latin / Unicode / languages | `ThreeLanguagePipeline.ipynb` § *Cross Language Transfer* |
| Embedding comparison table (Exp 1) | Silhouette + morpheme coherence, Latin vs Unicode | `ThreeLanguagePipeline.ipynb` § Experiment 1; `ElamiteExperiments.ipynb` § 2 |
| t-SNE / silhouette figures (Exp 1) | Per-class silhouette bars, embedding plots | `ThreeLanguagePipeline.ipynb` § Experiment 1 |
| Graph-analysis figures | Modularity, community count, eigenvector centrality, reinforced-edge networks | `GraphAnalysis_ThreeLanguages.ipynb` |
| Word-boundary inference table (Exp 2) | TP precision / recall / F1, Latin vs Unicode | `ThreeLanguagePipeline.ipynb` § Experiment 2; `ElamiteExperiments.ipynb` § 3 |
| TP threshold-sweep figures | F1 vs threshold, per language | `ThreeLanguagePipeline.ipynb` § Experiment 2 (writes `threshold_sweep_{lang}.csv`) |
| POS classification table (Exp 3) | n-gram LR / k-NN / BiLSTM, two tagsets | `ThreeLanguagePipeline.ipynb` § Experiment 3 / 3b / 3c; `ElamiteExperiments.ipynb` § 4 |
| Lemmatization table (Exp 4) | Char-seq2seq exact-match, Latin vs Unicode | `ThreeLanguagePipeline.ipynb` § Experiment 4; `ElamiteExp4_Lemmatization.ipynb` |
| LLM evaluation table (Exp 5) | Zero/few-shot Claude + GPT-4o | `ThreeLanguagePipeline.ipynb` § Experiment 5; `ElamiteExp5_LLM_Evaluation.ipynb` |
| ByT5 fine-tuning table | ByT5 classifier, Latin vs Unicode | `ByT5_Cuneiform.ipynb` § Cell 13 (Summary Comparison) |
| Gated dual-encoder + baseline transformer table | Learned per-input gating analysis | `ThreeLanguagePipeline.ipynb` § Experiment 6 |
| Hittite-only results | Single-language replication | `HittitePipeline.ipynb` § 11 (Results Summary) |
| Radar / summary figure | Multi-task summary across languages | `ThreeLanguagePipeline.ipynb` § Results Summary |
| LaTeX result tables | Direct paper-table source | `ThreeLanguagePipeline.ipynb` § Results Summary & LaTeX Tables |

## Setup

The pipeline was developed in Python 3.10. The notebooks are designed to run
in Google Colab (which is the simplest path because of GPU and Drive
integration), but they also run locally if the data files are placed under
`./data/` next to the notebook.

```bash
git clone <this anonymous repo URL>
cd cuneiform-experiments

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Plus the anonymous companion library:
pip install git+https://anonymous.4open.science/r/cunei-tools
```

Additional packages used by individual notebooks (install on demand):
`fasttext`, `gensim`, `scikit-learn`, `torch`, `transformers`,
`sentencepiece`, `nltk`, `pandas`, `numpy`, `matplotlib`, `seaborn`,
`networkx`, `python-louvain`, `umap-learn`. The LLM-evaluation notebook
additionally requires `anthropic` and `openai` clients with valid API keys.

### Running in Colab

`ElamiteNewData.ipynb` and the `*Pipeline*.ipynb` notebooks contain a Drive
mount step. After the anonymization pass in this release the previous
`/content/drive/MyDrive/...` paths have been rewritten to `data/`, so once
you upload the contents of `./data/` to your Colab session (or mount Drive
and put them at `MyDrive/data/`) the notebooks read them transparently.

## Data sources

This repository ships only data that is **not** publicly available
elsewhere. Two large public sources must be downloaded separately before
running the full Akkadian / Sumerian experiments:

1. **ORACC corpus dump** — `finaldf.csv` (≈1.5 GB), distributed via Zenodo:
   <https://zenodo.org/records/10794626/files/finaldf.csv>
   After download, run `python scripts/filter_oracc_data.py` from the
   directory containing `finaldf.csv` to produce `alltexts_AKK.csv` and
   `alltexts_SUX.csv`. The `*Pipeline*.ipynb` notebooks expect those two
   files in `data/`.
2. **ETCSL (Sumerian Literature)** — used as a complementary source for
   Sumerian text, available at <https://etcsl.orinst.ox.ac.uk/>.

The notebooks also pull the following sign-list resources directly from
GitHub at runtime (no manual download needed):

- *Akkademia* sign list — `https://raw.githubusercontent.com/gaigutherz/Akkademia/master/cuneiform_to_unicode_fixed.csv`
- *Nuolenna* sign list — `https://raw.githubusercontent.com/situx/Nuolenna/master/sign_list.json`

The remaining curated resources (Hittite glossed corpus, Elamite lemma
base, Utu-Našu word-level corpus, manual sign corrections) are bundled
under `data/` because they were assembled by the authors and are not
otherwise distributed.

## Reproducing a specific table or figure

1. Set up the environment and data as above.
2. Open `notebooks/ThreeLanguagePipeline.ipynb`.
3. Run the *Setup*, *POS Harmonization*, *Sign Lists*, and *Load All
   Three Languages* cells (sections 0–3).
4. Jump to the experiment section corresponding to the target artifact in
   the table above.
5. The final two cells of the pipeline write `data/three_lang_results.json`
   (numerical results) and emit LaTeX source for the paper tables.

For the standalone deep dives (`ElamiteExperiments.ipynb`,
`ElamiteExp4_Lemmatization.ipynb`, `ElamiteExp5_LLM_Evaluation.ipynb`,
`GraphAnalysis_ThreeLanguages.ipynb`, `HittitePipeline.ipynb`,
`ByT5_Cuneiform.ipynb`) just run the notebook end-to-end — each is
self-contained given `data/` and `requirements.txt`.

## Notes for reviewers

- All paths under the original Drive layout have been rewritten to
  relative `data/` paths.
- Author / affiliation strings in markdown cells have been replaced with
  `Anonymous Authors` / `Anonymous Affiliation`.
- No execution metadata that could identify the authors is retained (Colab
  user IDs, execution info, and notebook-level author fields have been
  stripped).
- Random seeds are fixed where stated in the notebooks
  (`SEED = 42` is the default).
