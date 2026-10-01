# Missing-modality tokens vs. imputation with indicators in proteogenomic survival prediction

Code for the paper *"Learnable Missing-Modality Tokens Show No Benefit over Imputation with Indicators: A Three-Cohort Failure Analysis of Proteogenomic Survival Prediction"* (under review).

The analysis compares a learnable missing-modality token with median imputation plus a missingness indicator, regression imputation and late fusion, on TCGA BRCA, LUAD and KIRC (PanCancer Atlas, via cBioPortal) with naturally missing RPPA proteomics. Everything is fit inside the training data under repeated 5x5-fold cross-validation: screening, imputation, standardization, early stopping, recalibration and the clinical second-stage models (nested). Evaluation reports discrimination, absolute-risk accuracy (IPCW Brier, IPA), calibration, incremental value over clinical variables, and decision curves.

## Repository layout

```
.
├── README.md                 this file
├── PAPER_MAP.md              which script and output file produces every table, figure and number in the paper
├── requirements.txt
├── LICENSE
├── scripts/
│   ├── 01_brca_main_pipeline.py                  BRCA: snapshot, 4 neural arms, classical baselines, nested
│   │                                             second stage, calibration/Brier/IPA, DCA, stress test,
│   │                                             parameterization-equivalence check, learning curves
│   ├── 02_brca_tree_baselines_site_sensitivity.py  BRCA: RSF + GBM on identical folds (if skipped in 01),
│   │                                             tissue-source-site sensitivity, BRCA summary
│   └── 03_extension_luad_kirc_mechanisms.py      LUAD + KIRC replication; regression imputation and late
│                                                 fusion in all three cohorts; RPPA-missing subgroup replication;
│                                                 KIRC with histologic grade in the clinical comparator
├── colab/RUN_IN_COLAB.md     step-by-step instructions for Google Colab
├── snapshot/                 frozen cBioPortal data (JSON); see snapshot/README.md
└── results/                  outputs written by the scripts; see results/README.md
```

## Quick start (local)

```bash
pip install -r requirements.txt
export PGR_BASE_DIR=$(pwd)          # scripts read  $PGR_BASE_DIR/snapshot  and write  $PGR_BASE_DIR/results
python scripts/01_brca_main_pipeline.py
python scripts/02_brca_tree_baselines_site_sensitivity.py
python scripts/03_extension_luad_kirc_mechanisms.py
```

Run the scripts in this order: 02 and 03 load outputs saved by 01. If `snapshot/` already contains the frozen JSON files, nothing is downloaded and the exact cohorts of the paper are used. If it is empty, the scripts download the data from the public cBioPortal API on first run and freeze it there; cBioPortal data can change over time, so a fresh download may differ slightly from the paper.

Without `PGR_BASE_DIR`, the scripts use Google Drive (`/content/drive/MyDrive/proteogenomic_revision`) on Colab and `./proteogenomic_revision` elsewhere.

## Runtime

| Script | Hardware | Approx. time |
|---|---|---|
| 01 | Colab GPU | 1-2 h (dry run: set `N_REPEATS=1`, `RUN_LEARNING_CURVE=False`, ~15 min) |
| 02 | CPU | 15-25 min |
| 03 | CPU | ~2 h (resumable: each cohort's arrays are saved when finished and skipped on rerun) |

## Reproducibility notes

- Seeds: outer folds use `StratifiedKFold(5, shuffle=True, random_state=r)` for repeats `r = 0..4`; network seeds are derived from repeat and fold indices; bootstrap seeds are fixed (see the code). Neural-network results can differ in the last decimal across hardware and PyTorch versions.
- Every number reported in the paper is written by the scripts to `results/` (JSON, CSV and the `summary_*.md` files); `PAPER_MAP.md` lists where each one comes from.
- The "Earlier estimate" column of the evaluation-errors table in the paper comes from the superseded analysis that the paper audits; that pipeline is intentionally not included. All corrected estimates are reproduced here.
- Two implementation details discussed in the paper: (i) a modality is missing only if the sample is absent from the cBioPortal sample list for that profile (a sequenced sample with no panel mutation is wild-type, not missing); (ii) in LUAD, no patient lacks mutation profiling, so the constant mutation-missing indicator is dropped from the elastic-net Cox design within each training fold.

## Data

TCGA PanCancer Atlas studies `brca_tcga_pan_can_atlas_2018`, `luad_tcga_pan_can_atlas_2018` and `kirc_tcga_pan_can_atlas_2018`, accessed through the public cBioPortal API (https://www.cbioportal.org), which requires no authentication. These are open-access TCGA data.

## License

MIT (see `LICENSE`).
