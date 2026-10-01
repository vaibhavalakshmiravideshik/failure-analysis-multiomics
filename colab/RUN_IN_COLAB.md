# Running on Google Colab

1. **Install (own cell, once per session):**
   ```
   !pip install -q lifelines scikit-survival==0.28.0
   ```
   Then **Runtime → Restart session**. Never put `!pip` in the same cell as the scripts; scikit-survival only imports cleanly after the restart.

2. **Data location.** The scripts mount Google Drive and use `/content/drive/MyDrive/proteogenomic_revision/` with subfolders `snapshot/` and `results/`. To use the frozen snapshot of the paper, copy the contents of this repository's `snapshot/` folder into `.../proteogenomic_revision/snapshot/` first.

3. **Script 01 (GPU recommended, 1–2 h).** Paste `scripts/01_brca_main_pipeline.py` into a cell (or paste each `# %%` block into its own cell) and run. For a quick dry run set `N_REPEATS=1` and `RUN_LEARNING_CURVE=False` in `CFG`.

4. **Script 02 (CPU, 15–25 min).** Paste `scripts/02_brca_tree_baselines_site_sensitivity.py` into one cell and run. It reloads everything it needs from Drive, so a fresh session is fine.

5. **Script 03 (CPU, ~2 h, resumable).** Paste `scripts/03_extension_luad_kirc_mechanisms.py` into one cell and run. If the session disconnects, run it again: finished cohorts are loaded from `results/extension/` and skipped.

All outputs land in `.../proteogenomic_revision/results/`.
