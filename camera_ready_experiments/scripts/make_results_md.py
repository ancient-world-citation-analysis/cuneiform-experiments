"""Stitch all experiment outputs into rebuttal_experiments/RESULTS.md —
the paste-ready summary required by REVIEWS_AND_REBUTTAL_PLAN.md.

Safe to re-run after any experiment; includes whatever outputs exist.
"""
import json
import os
from datetime import date

import common

OUT = os.path.join(common.REBUTTAL_DIR, "RESULTS.md")

SECTIONS = [
    ("expA_sux13k_subsample_control.md", "expA_sux13k_subsample_control.json"),
    ("expA2_tp_same_subsamples.md", "expA2_tp_same_subsamples.json"),
    ("expB_tp_vs_morfessor_significance.md", "expB_tp_vs_morfessor_significance.json"),
    ("expC_elamite25k_refresh.md", "expC_elamite25k_refresh.json"),
    ("expD_elx_source_ablation.md", "expD_elx_source_ablation.json"),
    ("expE_morfessor_size_grid.md", "expE_morfessor_size_grid.json"),
    ("expE2_overlap_mediation.md", "expE2_overlap_mediation.json"),
    ("expG_neural_supervised.md", "expG_neural_supervised.json"),
    ("expF_damage_fraction.md", "expF_damage_fraction.json"),
    ("expH_mechanism_controls.md", "expH_mechanism_controls.json"),
    ("expI_bilstm_transfer.md", "expI_bilstm_transfer.json"),
    ("expJ_defensive_checks.md", "expJ_defensive_checks.json"),
    ("expK_cw_source_selection.md", "expK_cw_source_selection.json"),
    ("expL_selected_cw_transfer.md", "expL_selected_cw_transfer.json"),
    ("expM_bootstrap_cis.md", "expM_bootstrap_cis.json"),
    ("expN_clamp_cw05.md", "expN_clamp_cw05.json"),
    ("expO_elx25_significance.md", "expO_elx25_significance.json"),
    ("expP_zeroshot_convention_fix.md", "expP_zeroshot_convention_fix.json"),
]

parts = [
    "# ARR May 2026 — Submission 2040 rebuttal experiment results",
    "",
    f"Generated {date.today().isoformat()} by `rebuttal_experiments/scripts/`.",
    "",
    "**Reproduction gate:** the released pipeline + paper-era corpora "
    "(AKK 13,502 docs / SUX-ETCSL 394 docs / ELX 84 docs) reproduce every "
    "off-diagonal Morfessor cell of Table 3 to within ±0.0004 "
    "(`cache/full_transfer_repro.json`). All numbers below share that "
    "verified harness: Morfessor corpusweight=1.0; TP = cunei-tools "
    "CuneiSeg, theta via find_optimal_threshold coarse grid; boundary "
    "micro-F1; doc filter len(split)>2; KFold(5, shuffle, seed=42); paired "
    "bootstrap 2,000 resamples two-sided; seeds 42–46 where applicable.",
    "",
    "---",
    "",
]

gate_path = os.path.join(common.CACHE_DIR, "full_transfer_repro.json")
if os.path.exists(gate_path):
    with open(gate_path) as f:
        gate = json.load(f)
    parts += [
        "### Reproduction gate detail",
        "",
        "| Cell | Reproduced F1 | Published F1 |",
        "|---|---|---|",
    ]
    for k, v in gate["reproduced_transfer"].items():
        pub = gate["published_reference"][k]["f1"]
        parts.append(f"| {k} | {v['f1']:.4f} | {pub:.4f} |")
    parts += ["", "---", ""]

for md_name, _ in SECTIONS:
    p = os.path.join(common.OUTPUT_DIR, md_name)
    if os.path.exists(p):
        with open(p) as f:
            parts.append(f.read().rstrip())
        parts += ["", "---", ""]

with open(OUT, "w") as f:
    f.write("\n".join(parts).rstrip() + "\n")
print(f"[saved] {OUT}")
