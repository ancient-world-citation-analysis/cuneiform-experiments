# SUX corpus provenance (rebuttal experiments)

The repo's `EMNLP/alltexts_SUX.csv` (881MB) is an ORACC dump export
(98,570 texts, admin-dominated) and does NOT match the paper's Sumerian
corpus (ETCSL literary, 394 texts / 146,143 POS-bearing tokens; see Table 1
"146K", outputs/table3_morfessor_transfer.json doc_counts sux=394).

`alltexts_SUX_etcsl_paper.csv` (copied 2026-07-08 from
~/Desktop/UCB/Stat238FinalProject/alltexts_SUX.csv, byte-identical to
~/Downloads/"alltexts_SUX - alltexts.csv") reproduces the paper's corpus
exactly after `_setup.load_sumerian` filters: 394 texts, 146,143 rows,
ETCSL `c.N.N.N` composition ids. The `alltexts_SUX.csv` symlink used by the
rebuttal harness points at this file.
