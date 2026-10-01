# TCGA Multi-omic Survival Prediction under Missing Modalities

Code for the paper **"Learnable Missing-Modality Tokens Show No Benefit over Imputation with Indicators: A Three-Cohort Failure Analysis of Proteogenomic Survival Prediction"** (under review).

Multi-omic survival models often add learned machinery to handle missing modalities, and are usually judged on the C-index alone. This repository tests both habits on three TCGA cohorts with naturally missing reverse-phase protein array (RPPA) proteomics. Every step that learns from data runs inside the training folds, and models are evaluated on calibration, absolute-risk accuracy and value over clinical variables, not only on discrimination.

---

## Main findings

| | BRCA | LUAD | KIRC |
|---|---|---|---|
| Patients / deaths | 1,084 / 151 | 505 / 182 | 511 / 170 |
| RPPA missing | 19.2% | 30.3% | 11.2% |
| Token − indicator, ΔC [95% CI] | +0.008 [−0.012, +0.027] | −0.018 [−0.041, +0.004] | −0.005 [−0.015, +0.004] |
| Late fusion − indicator, ΔC | +0.005 (n.s.) | **−0.033** (p = 0.045) | **−0.026** (p = 0.027) |
| Omics added to clinical model, ΔC | +0.006 (n.s.) | −0.005 (p = 0.047) | **+0.018** (p = 0.036, with grade) |

- **A learnable missing-modality token never beat median imputation plus a missingness indicator.** With a linear first layer, the token is an exact reparameterization of a subset of the indicator model, so the comparison tests parameterization, not expressive power.
- **Regression imputation matched the indicator model; late fusion was worse** in two of three cohorts.
- **A post hoc BRCA subgroup signal** (patients lacking RPPA, ΔC = +0.107, p = 0.068) **did not replicate** in LUAD or KIRC.
- **Whether omics add value depends on the cohort.** In BRCA, absolute-risk accuracy was indistinguishable from a constant-risk model (IPA 1.7%). In LUAD the omics score was at chance. In KIRC it added to age, stage and histologic grade.

---

## What is compared

**Missing-modality mechanisms** (identical encoder, folds, feature screening and hyperparameters):

| Arm | How a missing modality is handled | Parameters |
|---|---|---|
| Learnable token | missing features replaced by a learned vector | 984 |
| Impute + indicator | training-fold median + binary missingness indicator | 984 |
| Token / indicator + augmentation | as above, with matched training-time modality dropout (BRCA) | 984 |
| Regression imputation | ridge prediction of RPPA from mutations + indicator | 984 |
| Late fusion | per-modality encoders averaged over available modalities | 964 |
| Classical baselines | elastic-net Cox, random survival forest, gradient-boosted survival | — |

**Evaluation** (5-year horizon unless stated):
- discrimination: Harrell's C-index, mean over 5 repeats of 5-fold CV, plus paired bootstrap tests (2,000 resamples);
- overall accuracy: IPCW Brier score, compared with a constant-risk model (index of prediction accuracy, IPA);
- calibration: Cox calibration slope and O/E;
- incremental value: nested clinical vs clinical + omics Cox models;
- decision curves (BRCA);
- robustness: forced missingness on held-out patients, learning curves, tissue-source-site sensitivity.

**Leakage control:** feature screening, imputation, standardization, early stopping, recalibration and all second-stage clinical models are fit within training folds; second-stage models use nested cross-fitting, so every reported prediction is out of sample.

---

## Repository structure

```text
.
├── README.md
├── PAPER_MAP.md                 every table, figure and number in the paper → script → output file
├── requirements.txt
├── LICENSE
├── scripts/
│   ├── 01_brca_main_pipeline.py                     BRCA primary analysis
│   ├── 02_brca_tree_baselines_site_sensitivity.py   BRCA tree baselines + site sensitivity
│   └── 03_extension_luad_kirc_mechanisms.py         LUAD/KIRC replication + extra mechanisms (all cohorts)
├── colab/
│   └── RUN_IN_COLAB.md          step-by-step Colab instructions
├── snapshot/                    frozen cBioPortal data (JSON)
└── results/                     all outputs (JSON, CSV, figures, summaries)
```

| Script | What it does | Main outputs |
|---|---|---|
| `01` | Builds the BRCA cohort with corrected missingness masks; trains the four neural arms and elastic-net Cox under repeated nested CV; calibration, Brier/IPA, decision curves, stress test, equivalence check, learning curves | `results/results.json`, `table_*.csv`, figures |
| `02` | Random survival forest and gradient-boosted survival on the identical folds; tissue-source-site sensitivity; BRCA summary | `results/summary_brca.md` |
| `03` | LUAD and KIRC cohorts; regression imputation and late fusion in all three cohorts; subgroup replication; KIRC with histologic grade | `results/extension/summary_extension.md`, `summary_kirc_grade.md` |

---

## Installation

```bash
git clone <this repository>
cd <repository>
pip install -r requirements.txt
```

Python 3.10+. Tested with scikit-learn 1.9.1 and scikit-survival 0.28.0. A GPU speeds up script 01 but is not required.

---

## Usage

```bash
export PGR_BASE_DIR=$(pwd)        # reads ./snapshot, writes ./results
python scripts/01_brca_main_pipeline.py
python scripts/02_brca_tree_baselines_site_sensitivity.py
python scripts/03_extension_luad_kirc_mechanisms.py
```

Run the scripts in order: 02 and 03 reuse outputs saved by 01. On Google Colab, follow [`colab/RUN_IN_COLAB.md`](colab/RUN_IN_COLAB.md). If `PGR_BASE_DIR` is not set, the scripts use Google Drive on Colab and `./proteogenomic_revision` elsewhere.

| Script | Hardware | Approx. time |
|---|---|---|
| 01 | GPU | 1–2 h (dry run with `N_REPEATS=1`, `RUN_LEARNING_CURVE=False`: ~15 min) |
| 02 | CPU | 15–25 min |
| 03 | CPU | ~2 h; resumable (finished cohorts are skipped on rerun) |

---

## Data

Public TCGA PanCancer Atlas studies from [cBioPortal](https://www.cbioportal.org) (no authentication required):

| Cohort | cBioPortal study | Mutation panel |
|---|---|---|
| BRCA | `brca_tcga_pan_can_atlas_2018` | 44 breast-cancer driver genes |
| LUAD | `luad_tcga_pan_can_atlas_2018` | 44 genes + KEAP1, SMARCA4, RBM10, U2AF1, SETD2, MET |
| KIRC | `kirc_tcga_pan_can_atlas_2018` | 44 genes + VHL, PBRM1, SETD2, BAP1, KDM5C |

A modality counts as missing only when the sample is absent from cBioPortal's sample list for that profile. A sequenced sample with no panel mutation is wild-type, not missing.

The frozen snapshot used in the paper is in `snapshot/` (BRCA retrieved 30 September 2026 UTC; LUAD and KIRC 1 October 2026 UTC). If the folder is empty, the scripts download the data from the cBioPortal API on first run; later cBioPortal updates may change the cohorts slightly.

---

## Reproducibility

- Fixed seeds throughout: outer folds use `StratifiedKFold(5, shuffle=True, random_state=r)` for repeats `r = 0..4`; network and bootstrap seeds are fixed in the code. Neural-network results may differ in the last decimal across hardware and PyTorch versions.
- Every reported number is written to `results/`; [`PAPER_MAP.md`](PAPER_MAP.md) maps each table and figure to its script and output file.
- The paper also audits evaluation errors in an earlier version of the analysis. That superseded pipeline is not included; all corrected estimates are reproduced here.

---

## License

MIT. See [`LICENSE`](LICENSE).
