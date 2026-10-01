# Where every reported number comes from

All paths are relative to `results/`. "01", "02", "03" refer to the scripts in `scripts/`; `[n]` is the cell number inside script 01.

## BRCA (primary analysis)

| Paper item | Script | Output |
|---|---|---|
| Cohort size, deaths, RPPA/mutation missingness, zero follow-up, reconciliation (Data section; appendix reconciliation table) | 01 [1] | `results.json` → `reconciliation`; `cohort_manifest.csv`; `manifest.json` |
| Clinical comparator cohort (n = 962, 125 deaths) | 01 [1] | printed; `results.json` |
| Out-of-fold discrimination table (7 methods) | 01 [4], 02 | `table_discrimination.csv`; `results.json` → `discrimination`; `summary_brca.md` |
| Paired bootstrap table (token vs indicator, augmentation, classical baselines) | 01 [4], 02 | `table_paired_bootstrap.csv`; `results.json` → `paired_bootstrap` |
| Parameterization equivalence (max abs. difference, ranks of W_p, W_g, parameter counts) | 01 [5] | `results.json` → `equivalence` |
| Synthetic missingness stress test (text + figure) | 01 [8] | `results.json` → `stress_test`; `stress_test_heldout.png` |
| Learning-curve table and figure (incl. naive intervals, p ≈ 0.016) | 01 [9] | `table_learning_curve.csv`; `learning_curve_folds.csv`; `learning_curve_diff.png` |
| Natural-missingness subgroups: C-index per subgroup | 01 [4] | `results.json` → `subgroups` |
| RPPA-missing subgroup paired difference (+0.1070, p = 0.068) | 03 | `extension/summary_extension.md`; `extension/results_extension.json` |
| Five-year overall accuracy table (Brier, constant risk, IPA) and calibration table (O/E, slope) | 01 [6] | `table_calibration.csv`; `results.json` → `calibration`; `summary_brca.md` |
| Combined vs clinical paired ΔC (+0.0056 [−0.0026, +0.0142], p = 0.162) | 01 [6], 03 | `results.json` → `combo_vs_clinical_C`; `extension/summary_extension.md` |
| Decision curves | 01 [7] | `results.json` → `dca`; `dca_nested.png` |
| Calibration plot | 01 [6] | `calibration_plot.png` |
| Proportional-hazards test (appendix) | 01 [6] | `results.json` → `ph_test` |
| Site (tissue source site) sensitivity (Limitations) | 02 | `results.json` → `site_sensitivity`; `summary_brca.md` |

## Replication and additional mechanisms (all three cohorts)

| Paper item | Script | Output |
|---|---|---|
| LUAD and KIRC cohort descriptions (n, deaths, missingness, panel, clinical cohorts, censoring survival at 5 years) | 03 | printed `[LUAD] {...}`, `[KIRC] {...}`; `extension/results_extension.json` → `summary` |
| Cross-cohort discrimination table | 03 | `extension/summary_extension.md`; `extension/results_extension.json` → `discrimination` |
| Cross-cohort paired differences vs indicator, RPPA-missing subgroup replication | 03 | `extension/summary_extension.md`; `extension/results_extension.json` → `paired`, `subgroup_rppa_missing` |
| Cross-cohort absolute risk and incremental value (age + stage comparators) | 03 | `extension/summary_extension.md`; `extension/results_extension.json` → `absolute_risk`, `combo_vs_clinical` |
| KIRC with histologic grade in the clinical comparator | 03 (`RUN_KIRC_GRADE`) | `extension/summary_kirc_grade.md`; `extension/results_KIRC_grade.json` |
| Parameter counts of regression-imputation (984) and late-fusion (964) arms | 03 | `extension/results_extension.json` → `params` |
| Cross-cohort forest plot (not in the paper; diagnostic) | 03 | `extension/cross_cohort_forest.png` |

## Not reproduced by this repository

The "Earlier estimate" column of the evaluation-errors table comes from the superseded analysis that the paper audits. Its corrected counterparts ("Corrected estimate" column) are all produced by the scripts above.
