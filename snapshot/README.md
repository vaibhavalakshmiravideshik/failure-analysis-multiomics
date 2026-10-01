# Frozen data snapshot

The scripts read cBioPortal data from this folder and download it only if a file is missing.

Layout (created automatically on first run):

```
snapshot/
├── retrieval_meta.json, clinical_patient.json, clinical_sample.json, samples.json,
│   sample_lists.json, samples_in_mut_list.json, samples_in_rppa_list.json, ...   <- BRCA (written by 01)
├── luad_tcga_pan_can_atlas_2018/    <- LUAD (written by 03)
└── kirc_tcga_pan_can_atlas_2018/    <- KIRC (written by 03)
```

To reproduce the paper exactly, place the frozen JSON files used for the paper here (BRCA retrieved 30 September 2026 UTC; LUAD and KIRC retrieved 1 October 2026 UTC; the retrieval time is recorded in each `retrieval_meta.json`). With an empty folder the data are downloaded fresh from the public cBioPortal API, which may differ slightly from the paper if cBioPortal has been updated.
