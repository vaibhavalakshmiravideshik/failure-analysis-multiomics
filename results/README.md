# Results

Scripts write all outputs here (`$PGR_BASE_DIR/results`). Extension outputs go to `results/extension/`.

- 01: `results.json`, `manifest.json`, `cohort_manifest.csv`, `fold_features.json`, `oof_arrays.npz`, `oof_predictions.csv`, `table_*.csv`, `learning_curve_folds.csv`, figures (`stress_test_heldout.png`, `calibration_plot.png`, `dca_nested.png`, `learning_curve.png`, `learning_curve_diff.png`)
- 02: updates `results.json`, `oof_arrays.npz`, `table_discrimination.csv`, `table_paired_bootstrap.csv`; writes `summary_brca.md`
- 03: `extension/arrays_{BRCA,LUAD,KIRC}.npz`, `extension/arrays_KIRC_grade.npz`, `extension/results_extension.json`, `extension/results_KIRC_grade.json`, `extension/summary_extension.md`, `extension/summary_kirc_grade.md`, `extension/cross_cohort_forest.png`

See `../PAPER_MAP.md` for which output feeds which table or figure in the paper.
