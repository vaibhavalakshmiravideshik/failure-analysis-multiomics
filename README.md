TCGA Multi-omic Survival Prediction under Missing Modalities
Code for the paper "Learnable Missing-Modality Tokens Show No Benefit over Imputation with Indicators: A Three-Cohort Failure Analysis of Proteogenomic Survival Prediction" (under review).
Contents: 1. What this repository is about · 2. Main findings · 3. Repository map · 4. Quick start · 5. The pipeline step by step · 6. Every file explained · 7. Configuration · 8. Data · 9. Reproducibility · 10. Troubleshooting · 11. Glossary · 12. License
1. What this repository is about
The problem. Survival models that combine several molecular data types ("modalities") for the same patient must decide what to do when one modality was never measured. In TCGA, about 11–30% of patients have no reverse-phase protein array (RPPA) proteomics data, depending on the cancer type. Many deep-learning papers handle this with learned mechanisms, such as a learnable missing-modality token, and judge their models mainly by the C-index.
What we test. Two practices:
Does a learned missing-modality mechanism beat a simple one? We compare a learnable token with median imputation plus a missingness indicator, regression imputation and late fusion. All arms use the same network, folds, features and hyperparameters.
Does reporting only the C-index hide problems? We also measure absolute-risk accuracy, calibration, and whether the omics score adds anything to a model built from routine clinical variables.
Where. Three TCGA cohorts with naturally missing RPPA data: breast cancer (BRCA), lung adenocarcinoma (LUAD) and clear-cell renal carcinoma (KIRC).
How. Every step that learns from data is fit only on training data, under repeated 5×5-fold cross-validation. This covers feature selection, imputation, scaling, early stopping, recalibration and the clinical models. Second-stage models use nested cross-fitting, so every evaluated prediction is out of sample.
2. Main findings
BRCA LUAD KIRC Patients / deaths 1,084 / 151 505 / 182 511 / 170 Patients without RPPA data 19.2% 30.3% 11.2% Token − indicator, ΔC-index [95% CI] +0.008 [−0.012, +0.027] −0.018 [−0.041, +0.004] −0.005 [−0.015, +0.004] Late fusion − indicator, ΔC-index +0.005 (not significant) −0.033 (p = 0.045) −0.026 (p = 0.027) Omics added to clinical model, ΔC-index +0.006 (not significant) −0.005 (p = 0.047) +0.018 (p = 0.036; clinical model includes grade)
The learnable token never beat imputation plus an indicator. With a linear first layer, the token model is an exact reparameterization of a subset of the indicator model, so the comparison tests parameterization, not expressive power.
Regression imputation matched the indicator model. Late fusion was worse in two of three cohorts.
A post hoc BRCA subgroup signal (patients without RPPA: ΔC = +0.107, p = 0.068) did not replicate in LUAD or KIRC.
Whether omics add value depends on the cohort.
BRCA: absolute-risk accuracy was indistinguishable from predicting the same risk for everyone (IPA 1.7%).
LUAD: the omics score was at chance.
KIRC: the omics score added to age, stage and histologic grade.
3. Repository map
Path Type What it is README.md doc This file. PAPER_MAP.md doc Maps every table, figure and number in the paper to the script and output file that produce it. requirements.txt config Python packages needed to run the code. LICENSE legal MIT license. .gitignore config Files Git should ignore (Python caches, notebook checkpoints). scripts/ folder The three analysis scripts, run in order 01 → 02 → 03. scripts/01_brca_main_pipeline.py code BRCA primary analysis: data snapshot, cohort, four neural arms, elastic-net Cox, nested second stage, calibration, Brier/IPA, decision curves, stress test, equivalence check, learning curves. scripts/02_brca_tree_baselines_site_sensitivity.py code BRCA random survival forest and gradient-boosted survival on the identical folds; tissue-source-site sensitivity; BRCA summary. scripts/03_extension_luad_kirc_mechanisms.py code LUAD and KIRC replication; regression imputation and late fusion in all three cohorts; subgroup replication; KIRC with histologic grade. colab/ folder Instructions for running on Google Colab. colab/RUN_IN_COLAB.md doc Step-by-step Colab guide (installation, restart, cell order). snapshot/ folder Frozen copy of the cBioPortal input data (JSON). Described in §6.3. snapshot/README.md doc Layout of the snapshot folder. results/ folder Everything the scripts produce (tables, arrays, figures, summaries). Described in §6.4. results/README.md doc Short list of output files.
4. Quick start
4.1 Install
git clone <this repository>
cd <repository folder>
pip install -r requirements.txt

Requires Python 3.10 or newer. The paper's runs used scikit-learn 1.9.1 and scikit-survival 0.28.0 on Google Colab. A GPU speeds up script 01 but is not required.
4.2 Run
export PGR_BASE_DIR=$(pwd)        # scripts read ./snapshot and write ./results
python scripts/01_brca_main_pipeline.py
python scripts/02_brca_tree_baselines_site_sensitivity.py
python scripts/03_extension_luad_kirc_mechanisms.py

Script Needs Hardware Approx. time 01 nothing (downloads data if snapshot/ is empty) GPU recommended 1–2 h 02 outputs of 01 CPU 15–25 min 03 outputs of 01 (and 02) CPU ~2 h, resumable
For a quick test of script 01 (~15 min), set N_REPEATS=1 and RUN_LEARNING_CURVE=False in its CFG block (see §7).
On Google Colab, follow colab/RUN_IN_COLAB.md instead.
4.3 Check the results
results/summary_brca.md (from 02) and results/extension/summary_extension.md (from 03) list every key number in plain text.
PAPER_MAP.md shows which paper table each number belongs to.
5. The pipeline step by step
cBioPortal API ──► snapshot/  (frozen JSON; downloaded once)
                      │
                      ▼
   01_brca_main_pipeline.py ──► results/  (BRCA: arrays, tables, figures, results.json)
                      │
                      ▼
   02_brca_tree_baselines_site_sensitivity.py ──► results/  (adds RSF/GBM, site analysis, summary_brca.md)
                      │
                      ▼
   03_extension_luad_kirc_mechanisms.py ──► snapshot/<LUAD, KIRC>/ + results/extension/

Script 01: BRCA primary analysis
Script 01 is organized in numbered cells (# %% [n]). They can be run as one script, or pasted cell by cell into a notebook.
Cell Purpose [0] Setup Imports, CFG settings, output folders, the 44-gene breast-cancer driver panel. [1] Data snapshot and cohort Downloads or loads the cBioPortal data and builds survival times and events. Defines missing modalities from cBioPortal sample lists: a sequenced sample with no mutation in the panel is wild-type, not missing. Builds the clinical variables (age, stage, PAM50 subtype). Writes cohort_manifest.csv and manifest.json. [2] Model and preprocessing Survival network (1 hidden layer of 20 units, 4 discrete time intervals ending at 1, 3, 5 and 7 years) and its loss. Per-fold feature screening (top 25 proteins and 15 genes by univariate Cox p-value), imputation and standardization, all on training rows only. [3] Cross-validation 5 repeats × 5 folds. Trains the four neural arms (token, indicator, each with and without augmentation) and the classical baselines. Inside each training fold, an inner 5-fold CV produces out-of-fold predictions for recalibration and for the clinical and clinical + omics Cox models. [4] Discrimination C-index per method, repeat-ensemble scores, and paired bootstrap comparisons. Also C-index in subgroups (complete data, RPPA missing, mutation missing). [5] Equivalence check Numerically verifies that a trained token model equals an indicator model with u_k = W_k (t_k − c_k), and reports parameter counts. [6] Calibration and accuracy 5-year calibration slope, O/E, IPCW Brier score, constant-risk baseline, IPA, combined-vs-clinical comparison, proportional-hazards test, calibration plot. [7] Decision curves Net benefit of clinical + omics vs clinical across risk thresholds of 3–40%. [8] Stress test Forces RPPA to be missing at increasing rates on held-out complete-data patients. [9] Learning curves Token vs indicator at 25/50/75/100% of each training fold. [10] Save Writes results.json, oof_predictions.csv and a printed summary.
Script 02: BRCA tree baselines and site sensitivity
Repeats the setup and data cells of 01, then loads 01's saved arrays.
Fits a random survival forest and a gradient-boosted survival model on exactly the same folds and features.
Fits Cox models of survival on the omics score with and without stratification by tissue source site (the 2-character center code in the TCGA sample ID), to check that the score is not explained by which hospital collected the sample.
Writes summary_brca.md.
Needed when scikit-survival was unavailable during 01, and always needed for the site analysis.
Script 03: replication cohorts and additional mechanisms
Downloads and freezes LUAD and KIRC with the same rules as BRCA.
Runs four neural arms in every cohort: token, indicator, regression imputation and late fusion. In BRCA it trains only the two new arms and reuses 01's saved arrays.
Runs the three classical baselines in LUAD and KIRC, plus the nested absolute-risk and incremental-value analysis.
Tests whether the BRCA RPPA-missing subgroup result replicates.
Re-runs KIRC with histologic grade added to the clinical comparator.
Resumable: each cohort's arrays are saved when finished, and rerunning skips finished cohorts.
6. Every file explained
6.1 Documentation and configuration
File Contents README.md Overview, usage and file reference (this document). PAPER_MAP.md Table: paper item → script (and cell) → output file and JSON key. requirements.txt numpy, pandas, scipy, matplotlib, requests, torch, lifelines, scikit-learn==1.9.1, scikit-survival==0.28.0. colab/RUN_IN_COLAB.md Install with !pip, restart the runtime, then run 01, 02, 03; where Drive files go. LICENSE MIT. .gitignore Ignores __pycache__/, *.pyc, .ipynb_checkpoints/, .DS_Store.
6.2 Code
File Input Output scripts/01_brca_main_pipeline.py snapshot/ (BRCA JSON) or the cBioPortal API BRCA files in results/ (see §6.4) scripts/02_brca_tree_baselines_site_sensitivity.py snapshot/, results/oof_arrays.npz, results/results.json updated results/ files + summary_brca.md scripts/03_extension_luad_kirc_mechanisms.py snapshot/, results/oof_arrays.npz, the cBioPortal API for LUAD and KIRC if not yet frozen snapshot/luad_.../, snapshot/kirc_.../, results/extension/
6.3 Input data: snapshot/
BRCA files sit directly in snapshot/. LUAD and KIRC files sit in subfolders named after their cBioPortal study (snapshot/luad_tcga_pan_can_atlas_2018/, snapshot/kirc_tcga_pan_can_atlas_2018/). Files per cohort:
File Contents retrieval_meta.json UTC time at which the data were downloaded. clinical_patient.json Patient-level clinical data: overall survival months and status, age, AJCC stage, PAM50 subtype (BRCA). clinical_sample.json Sample-level clinical data, including mutation count (BRCA) and histologic grade (KIRC). Present for BRCA and KIRC. samples.json All samples in the study with their patient IDs. sample_lists.json The study's sample lists (which samples were profiled by which assay). samples_in_mut_list.json IDs of samples profiled for mutations. A sample not in this list has its mutation data treated as missing. samples_in_rppa_list.json IDs of samples profiled by RPPA. rppa_zscores.json RPPA protein-level z-scores per sample and protein. mutations.json Mutation calls per sample and gene. molecular_profiles.json List of the study's molecular profiles (written by script 03, all cohorts).
6.4 Outputs: results/
From script 01 (BRCA)
File Contents results.json Every BRCA number. Keys: reconciliation, discrimination, paired_bootstrap, subgroups, equivalence, calibration, combo_vs_clinical_C, ph_test, dca, stress_test, learning_curve, brier_crosscheck, abstract_C_range (plus site_sensitivity after 02). manifest.json Study ID, profile and sample-list IDs, retrieval time, gene panel, full CFG, package versions, sample-count reconciliation. cohort_manifest.csv One row per patient: sample and patient ID, survival time and event, RPPA and mutation availability (new and old definitions), age, stage, PAM50. fold_features.json Proteins and genes selected in each repeat and fold. oof_arrays.npz Raw out-of-fold predictions. haz_<arm>: hazards, shape (repeats, patients, 4 intervals). base_<model>: classical-baseline risk scores. nest_<model>: nested 5-year risks for recal, combo, clinical, stage_only, null. oof_predictions.csv Per-patient 5-year risks averaged over repeats, for every arm and nested model. table_discrimination.csv Mean and SD of C-index over repeats, ensemble C-index and bootstrap 95% CI, per method. table_paired_bootstrap.csv Paired C-index differences with 95% CI and p-value. table_calibration.csv 5-year Brier, constant-risk Brier, IPA, O/E and calibration slope with CIs, per model. table_learning_curve.csv Learning-curve summary per training fraction, including naive 95% intervals and p-values. learning_curve_folds.csv Learning-curve C-index per repeat, fold and training fraction. stress_test_heldout.png C-index under forced RPPA missingness (paper figure). calibration_plot.png Predicted vs observed 5-year risk by decile, before and after recalibration (paper figure). dca_nested.png Decision curves, clinical + omics vs clinical (paper figure). learning_curve_diff.png Token − indicator C-index vs training events (paper figure). learning_curve.png Token and indicator C-index vs training events (diagnostic).
From script 02 (BRCA)
File Contents summary_brca.md Readable summary: tree baselines, absolute-risk table, combined vs clinical, decision-curve range, stress test, site sensitivity. results.json, oof_arrays.npz, table_discrimination.csv, table_paired_bootstrap.csv Updated with random survival forest and gradient-boosted survival (and site_sensitivity in JSON).
From script 03 (all cohorts), in results/extension/
File Contents arrays_BRCA.npz, arrays_LUAD.npz, arrays_KIRC.npz Out-of-fold predictions per cohort, same layout as oof_arrays.npz, including haz_regimpute and haz_latefusion. arrays_KIRC_grade.npz KIRC token/indicator predictions and nested models with grade in the clinical comparator. results_extension.json Per cohort: summary (cohort description), discrimination, paired, subgroup_rppa_missing, absolute_risk, combo_vs_clinical, params, data_ids. results_KIRC_grade.json Same structure for the KIRC-with-grade analysis. summary_extension.md Readable summary of all three cohorts (the source of the cross-cohort tables in the paper). summary_kirc_grade.md Readable summary of the KIRC-with-grade analysis. cross_cohort_forest.png Forest plot of paired differences across cohorts (diagnostic, not in the paper).
7. Configuration
Each script has a CFG dictionary near the top. The defaults reproduce the paper.
Setting Default Meaning N_REPEATS, N_OUTER, N_INNER 5, 5, 5 Repeats of outer CV, outer folds, inner folds for nested models. TOP_K_PROT, TOP_K_GEN 25, 15 Proteins and genes kept by per-fold screening. TIME_BINS, HORIZON 1/3/5/7 years, 5 years Discrete-time intervals and the evaluation horizon (in days). HIDDEN, DROPOUT 20, 0.35 Hidden units and dropout of the survival network. LR, WD, BATCH, MAX_EPOCHS, PATIENCE 8e-4, 3e-2, 32, 100, 20 AdamW optimizer settings and early stopping. AUG_PROT, AUG_GEN 0.20, 0.10 Training-time modality dropout probabilities (augmented arms, script 01). N_BOOT, N_BOOT_DCA 2000, 200 Bootstrap resamples for CIs/tests and for decision curves. RUN_BASELINES, RUN_NESTED, RUN_STRESS, RUN_LEARNING_CURVE True Switch analysis blocks on or off (script 01). RIDGE_ALPHA 1.0 Ridge penalty for regression imputation (script 03). RUN_KIRC_GRADE True Run the KIRC-with-grade analysis (script 03). PGR_BASE_DIR (environment variable) not set Base folder containing snapshot/ and results/. If unset: Google Drive on Colab, otherwise ./proteogenomic_revision.
8. Data
Public TCGA PanCancer Atlas studies from cBioPortal, with no login required:
Cohort cBioPortal study ID RPPA proteins Mutation panel BRCA (breast) brca_tcga_pan_can_atlas_2018 208 44 breast-cancer driver genes LUAD (lung adenocarcinoma) luad_tcga_pan_can_atlas_2018 209 44 genes + KEAP1, SMARCA4, RBM10, U2AF1, SETD2, MET (50) KIRC (clear-cell renal) kirc_tcga_pan_can_atlas_2018 209 44 genes + VHL, PBRM1, SETD2, BAP1, KDM5C (49)
Outcome: overall survival, evaluated at 5 years.
Clinical comparators:
BRCA: age, AJCC stage and PAM50 subtype.
LUAD: age and stage.
KIRC: age and stage, with and without histologic grade.
Snapshot: the frozen data used in the paper are in snapshot/. BRCA was retrieved on 30 September 2026 (UTC); LUAD and KIRC on 1 October 2026 (UTC). If the folder is empty, the scripts download fresh data, which may differ slightly if cBioPortal has been updated since.
9. Reproducibility
Seeds. Outer folds use StratifiedKFold(5, shuffle=True, random_state=r) for repeats r = 0…4. Network seeds are derived from repeat and fold indices, and bootstrap seeds are fixed in the code. Neural-network results can differ in the last decimal across hardware and PyTorch versions.
Traceability. Every number in the paper is written to results/. PAPER_MAP.md says where each one comes from.
Scope. The paper also audits evaluation errors in an earlier version of the analysis. That superseded pipeline is not included here; all corrected estimates are reproduced.
Two implementation details discussed in the paper:
A modality counts as missing only when the sample is absent from cBioPortal's sample list for that assay.
In LUAD no patient lacks mutation profiling, so the always-zero mutation-missing indicator is dropped from the elastic-net Cox model within each training fold.
10. Troubleshooting
Problem Fix ImportError from scikit-survival on Colab Run !pip install -q lifelines scikit-survival==0.28.0 in its own cell, then Runtime → Restart session, then run the script. Never put !pip in the same cell as the script. scikit-survival unavailable -> tree baselines skipped in script 01 Run script 02 afterwards; it adds the tree baselines on the same folds. cBioPortal returns 502/503 The scripts retry automatically with back-off. If it keeps failing, wait and rerun; data already in snapshot/ are not downloaded again. Script 03 stopped part-way Rerun it. Finished cohorts are loaded from results/extension/arrays_*.npz and skipped. FileNotFoundError: oof_arrays.npz in script 02 or 03 Run script 01 first, with the same PGR_BASE_DIR. WARNING: cohort differs from the paper The snapshot is not the frozen one used in the paper (probably a fresh download). Results will be close but not identical.
11. Glossary
Term Meaning Modality One type of molecular measurement for a patient. Here: RPPA proteomics and somatic mutations. RPPA Reverse-phase protein array: measures the abundance of about 200 proteins. Learnable token Missing features are replaced by a vector of parameters learned during training. Impute + indicator Missing features are replaced by the training-fold median, and a 0/1 "modality missing" input is added. Regression imputation Missing RPPA values are predicted from the patient's mutation features (ridge regression), plus an indicator. Late fusion Each modality has its own encoder. Their outputs are averaged over the modalities the patient actually has. Augmentation During training only, a present modality is randomly hidden so the model learns to cope with missing data. C-index Concordance index: the probability that, of two patients, the one who dies first was given the higher predicted risk. 0.5 is chance. ΔC Difference in C-index between two models on the same patients, with a paired bootstrap CI. Brier score (IPCW) Mean squared error of predicted 5-year risk, weighted to account for censored patients. Lower is better. Constant-risk model Predicts the same risk for everyone: the Kaplan–Meier 5-year risk of the training fold. IPA Index of prediction accuracy = 1 − Brier / Brier of the constant-risk model. 0 means no better than giving everyone the same risk. Calibration slope 1 means predicted risks are spread correctly; below 1 means predictions are too extreme. O/E Observed over expected 5-year risk; 1 means correct on average. Nested cross-fitting Second-stage models are trained on inner-fold predictions within each training fold, so they never see the test patients. Decision curve / net benefit Clinical usefulness of a model across risk thresholds at which one would act. Repeat ensemble For each method, the average over the 5 repeats of rank-normalized risk scores; used for CIs and paired tests. Tissue source site The hospital or center that contributed a TCGA sample.
12. License
MIT. See LICENSE.
