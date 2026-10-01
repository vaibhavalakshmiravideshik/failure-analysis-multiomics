# =============================================================================
# 02 — BRCA tree baselines (RSF, GBM) on the identical folds/features + site (TSS) sensitivity
# Self-contained: repeats the setup/snapshot/definition cells of 01, then loads 01's saved
# outputs (results/oof_arrays.npz, results/results.json). Does NOT retrain neural arms.
# Needed when scikit-survival was unavailable during 01 (tree baselines skipped there);
# always needed for the site-sensitivity analysis. ~15-25 min on CPU.
# =============================================================================

# =============================================================================
# MAIN PIPELINE — Proteogenomic survival failure analysis (TCGA-BRCA)
# Leakage-controlled re-analysis; fixes [F1]-[F14] are listed below.
#
# Colab: paste each "# %%" block into its own cell, run top to bottom.
# Runtime: ~1–2 h on a Colab GPU with defaults. Dry run: set N_REPEATS=1,
# RUN_LEARNING_CURVE=False (~15 min).
#
# FIXES vs. previous notebook
#  [F1] Mutation mask from cBioPortal sequenced-sample list (not "has >=1 call")
#  [F2] 5-year risk (bin idx 2) used for C-index, calibration, Brier, DCA
#       (old oof_risk_v2 was 7-year risk evaluated at 5 years)
#  [F3] Early stopping on a split carved from the TRAINING fold only
#  [F4] Nested cross-fitting for recalibration + clinical second stage
#  [F5] Per-fold RPPA within-panel imputation + standardization
#  [F6] Augmented impute+indicator arm (matched ablation)
#  [F7] Forward-pass spec, parameter counts, exact-equivalence check
#  [F8] Explicit IPCW Brier (Graf), constant-risk null, IPA, bootstrap CIs
#  [F9] Calibration slope CI + Wald test vs 1; calibration-in-the-large (O/E)
#  [F10] Clinical comparator = age + stage (dummies) + PAM50 (dummies)
#  [F11] Stress test evaluated on held-out complete-data patients, both arms
#  [F12] Learning curves (token vs indicator) over training-set size/events
#  [F13] Frozen data snapshot + manifest (IDs, retrieval date, versions, masks)
#  [F14] Repeated 5x5 CV; every manuscript number written to results.json
# =============================================================================

# %% [0] Setup ----------------------------------------------------------------
# Run once in its own cell first, then Runtime > Restart session, then continue:
#   !pip install -q lifelines scikit-survival
import os, json, copy, hashlib, datetime, warnings, platform
import numpy as np, pandas as pd, requests, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import rankdata, norm
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import proportional_hazard_test
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

try:
    from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
    from sksurv.util import Surv
    from sksurv.metrics import brier_score as sksurv_brier
    HAS_SKSURV = True
except Exception as e:  # if import fails on Colab: Runtime > Restart, rerun
    print("scikit-survival unavailable -> tree baselines skipped:", e)
    HAS_SKSURV = False

# Working directory: set PGR_BASE_DIR to override. Default: Google Drive on Colab, else ./proteogenomic_revision
BASE = os.environ.get("PGR_BASE_DIR")
if BASE is None:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        BASE = "/content/drive/MyDrive/proteogenomic_revision"
    except Exception:
        BASE = "./proteogenomic_revision"

CFG = dict(
    STUDY_ID="brca_tcga_pan_can_atlas_2018",
    SNAPSHOT_DIR=f"{BASE}/snapshot", RESULTS_DIR=f"{BASE}/results",
    FORCE_REFETCH=False,
    N_REPEATS=5, N_OUTER=5, N_INNER=5, ES_FRAC=0.15,
    TOP_K_PROT=25, TOP_K_GEN=15,
    TIME_BINS=[365.0, 1095.0, 1825.0, 2555.0], HORIZON=1825.0, HORIZON_IDX=2,
    HIDDEN=20, DROPOUT=0.35, LR=8e-4, WD=3e-2, BATCH=32, MAX_EPOCHS=100, PATIENCE=20,
    AUG_PROT=0.20, AUG_GEN=0.10,
    N_BOOT=2000, N_BOOT_DCA=200,
    RUN_BASELINES=True, RUN_NESTED=True, RUN_STRESS=True, RUN_LEARNING_CURVE=True,
    STRESS_SEEDS=5, STRESS_PROBS=[0.0, 0.15, 0.30, 0.50, 0.75, 1.0],
    LC_FRACTIONS=[0.25, 0.50, 0.75, 1.0], LC_REPEATS=3,
)
os.makedirs(CFG["SNAPSHOT_DIR"], exist_ok=True)
os.makedirs(CFG["RESULTS_DIR"], exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S, H, HIDX = CFG["STUDY_ID"], CFG["HORIZON"], CFG["HORIZON_IDX"]
RESULTS = {}

def out(p):
    return os.path.join(CFG["RESULTS_DIR"], p)

BRCA_DRIVER_PANEL = sorted(set([
    'TP53','PIK3CA','GATA3','MAP3K1','CDH1','PTEN','AKT1','ERBB2','ESR1',
    'MYC','CCND1','RB1','BRCA1','BRCA2','NF1','ARID1A','KMT2C','FOXA1',
    'RUNX1','TBX3','CBFB','PIK3R1','NCOR1','SF3B1','CTCF','MAP2K4',
    'CDKN1B','SMAD4','ATR','STK11','ERBB3','KRAS','NRAS','BRAF','EGFR',
    'MTOR','TSC1','TSC2','CDKN2A','MDM2','MDM4','ATM','CHEK2','PALB2',
]))

# %% [1] Frozen data snapshot + corrected modality masks [F1][F13] --------------
API = "https://www.cbioportal.org/api"

import time

def _request(method, ep, tries=8, **kw):
    """Retry with exponential backoff on 429/5xx and connection errors."""
    for a in range(tries):
        try:
            r = requests.request(method, API + ep, **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} for {ep}", response=r)
            r.raise_for_status()
            return r.json()
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as ex:
            code = getattr(getattr(ex, "response", None), "status_code", None)
            if code is not None and code < 500 and code != 429:
                raise                              # real client error: don't retry
            if a == tries - 1:
                raise
            wait = min(2 ** a * 5, 300)
            print(f"  cBioPortal {ep}: {ex} -> retry {a+1}/{tries-1} in {wait}s")
            time.sleep(wait)

def _get(ep, params=None):
    return _request("GET", ep, params=params, timeout=120)

def _post(ep, body, params=None):
    return _request("POST", ep, json=body, params=params, timeout=300)

def _portal_info():
    try:
        return _request("GET", "/info", tries=3, timeout=60)
    except Exception as ex:                        # version info is metadata, not data
        print("  /info unavailable, recording without portal version:", ex)
        return {"unavailable": str(ex)}

def snap(name, fn):
    p = os.path.join(CFG["SNAPSHOT_DIR"], name + ".json")
    if os.path.exists(p) and not CFG["FORCE_REFETCH"]:
        with open(p) as f:
            return json.load(f)
    d = fn()
    with open(p, "w") as f:
        json.dump(d, f)
    return d

def _paged_clinical(kind):
    res, page = [], 0
    while True:
        b = _get(f"/studies/{S}/clinical-data", {"clinicalDataType": kind, "pageSize": 10000,
                                                  "pageNumber": page, "projection": "SUMMARY"})
        res += b
        if len(b) < 10000:
            break
        page += 1
    return res

def _chunked(ep, ids, params):
    res = []
    for i in range(0, len(ids), 500):
        res += _post(ep, {"sampleIds": ids[i:i + 500]}, params)
    return res

meta = snap("retrieval_meta", lambda: {
    "retrieved_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "portal_info": _portal_info()})
pat_clin = snap("clinical_patient", lambda: _paged_clinical("PATIENT"))
smp_clin = snap("clinical_sample", lambda: _paged_clinical("SAMPLE"))
samples_all = snap("samples", lambda: _get(f"/studies/{S}/samples", {"pageSize": 100000}))
sample_lists = snap("sample_lists", lambda: _get(f"/studies/{S}/sample-lists"))

def find_list(category, suffix):
    for sl in sample_lists:
        if sl.get("category") == category:
            return sl["sampleListId"]
    cand = f"{S}_{suffix}"
    if any(sl["sampleListId"] == cand for sl in sample_lists):
        return cand
    raise RuntimeError(f"No sample list for {category}")

print("Sample lists in study:")
for sl in sample_lists:
    print(f"  {sl['sampleListId']:55s} {sl.get('category')}")
MUT_LIST = find_list("all_cases_with_mutation_data", "sequenced")
RPPA_LIST = find_list("all_cases_with_rppa_data", "rppa")
mut_listed = set(snap("samples_in_mut_list", lambda: _get(f"/sample-lists/{MUT_LIST}/sample-ids")))
rppa_listed = set(snap("samples_in_rppa_list", lambda: _get(f"/sample-lists/{RPPA_LIST}/sample-ids")))

all_sample_ids = [s["sampleId"] for s in samples_all]
RPPA_PROFILE, MUT_PROFILE = f"{S}_rppa_Zscores", f"{S}_mutations"
rppa_raw = snap("rppa_zscores", lambda: _chunked(
    f"/molecular-profiles/{RPPA_PROFILE}/molecular-data/fetch", all_sample_ids, {"projection": "DETAILED"}))
mut_raw = snap("mutations", lambda: _chunked(
    f"/molecular-profiles/{MUT_PROFILE}/mutations/fetch", all_sample_ids, {"projection": "DETAILED"}))

# --- clinical ---
clin = pd.DataFrame(pat_clin).pivot_table(index="patientId", columns="clinicalAttributeId",
                                          values="value", aggfunc="first")
sclin = pd.DataFrame(smp_clin).pivot_table(index="sampleId", columns="clinicalAttributeId",
                                           values="value", aggfunc="first")
os_m = pd.to_numeric(clin["OS_MONTHS"], errors="coerce")
valid_os = clin["OS_STATUS"].notna() & os_m.notna()
clin = clin[valid_os].copy()
clin["T"] = os_m[valid_os] * 30.4375
clin["E"] = clin["OS_STATUS"].astype(str).str.startswith("1").astype(int)

smp = pd.DataFrame(samples_all)[["sampleId", "patientId"]]
smp = smp[smp["patientId"].isin(clin.index)]
if smp["patientId"].duplicated().any():  # keep primary tumour (-01) if multiple
    smp = smp[smp["sampleId"].str.endswith("-01")].drop_duplicates("patientId")
smp = smp.reset_index(drop=True)
N = len(smp)
T = clin.loc[smp["patientId"], "T"].to_numpy(float)
E = clin.loc[smp["patientId"], "E"].to_numpy(int)

# --- RPPA ---
rp = pd.DataFrame(rppa_raw)
rp["gene"] = rp["gene"].apply(lambda g: g.get("hugoGeneSymbol"))
n_dup = rp.duplicated(["sampleId", "gene"]).sum()
Xp_raw = rp.pivot_table(index="sampleId", columns="gene", values="value",
                        aggfunc="first").reindex(smp["sampleId"]).reset_index(drop=True)
has_rppa = Xp_raw.notna().any(axis=1).to_numpy()
in_rppa_list = smp["sampleId"].isin(rppa_listed).to_numpy()

# --- Mutations: CORRECTED mask [F1] ---
mu = pd.DataFrame(mut_raw)
mu["gene"] = mu["gene"].apply(lambda g: g.get("hugoGeneSymbol"))
called_any = set(mu["sampleId"])
has_mut = smp["sampleId"].isin(mut_listed).to_numpy()        # NEW: profiled per sample list
old_has_mut = smp["sampleId"].isin(called_any).to_numpy()    # OLD: >=1 call anywhere
panel = pd.crosstab(mu.loc[mu["gene"].isin(BRCA_DRIVER_PANEL), "sampleId"],
                    mu.loc[mu["gene"].isin(BRCA_DRIVER_PANEL), "gene"]).gt(0).astype(float)
Xg_raw = (panel.reindex(columns=BRCA_DRIVER_PANEL, fill_value=0.0)
          .reindex(smp["sampleId"]).fillna(0.0).reset_index(drop=True))
Xg_raw.loc[~has_mut, :] = np.nan   # only unprofiled samples are missing

mc = pd.to_numeric(sclin.get("MUTATION_COUNT", pd.Series(dtype=float)), errors="coerce")
study_ids = set(all_sample_ids)
recon = {
    "FULL_STUDY_samples": len(study_ids),
    "FULL_STUDY_in_mut_list": len(study_ids & mut_listed),
    "FULL_STUDY_not_in_mut_list": len(study_ids - mut_listed),
    "FULL_STUDY_in_list_with_MUTATION_COUNT_0": int(((mc.reindex(list(study_ids & mut_listed)) == 0)).sum()),
    "COHORT_n": N, "COHORT_events": int(E.sum()),
    "COHORT_mut_profiled_NEW": int(has_mut.sum()),
    "COHORT_profiled_with_0_calls": int((has_mut & ~old_has_mut).sum()),
    "COHORT_calls_but_not_in_list": int((~has_mut & old_has_mut).sum()),
    "COHORT_mut_missing_OLD": int((~old_has_mut).sum()),
    "COHORT_mut_missing_NEW": int((~has_mut).sum()),
    "COHORT_rppa_present_data": int(has_rppa.sum()),
    "COHORT_rppa_in_list": int(in_rppa_list.sum()),
    "COHORT_rppa_list_vs_data_disagree": int((has_rppa != in_rppa_list).sum()),
    "RPPA_duplicate_sample_gene_rows_collapsed": int(n_dup),
}
RESULTS["reconciliation"] = recon
print("\n=== Missingness reconciliation ===")
for k, v in recon.items():
    print(f"  {k:48s} {v}")
print(f"\nRPPA missing: {(~has_rppa).mean():.1%} | Mutation missing OLD: "
      f"{(~old_has_mut).mean():.1%}  NEW: {(~has_mut).mean():.1%}")
if N != 1084 or E.sum() != 151:
    print("WARNING: cohort differs from the paper (1084 patients / 151 deaths); check the snapshot.")

# --- clinical design (age + stage dummies + PAM50 dummies) [F10] ---
def stage_group(s):
    if not isinstance(s, str):
        return np.nan
    s = s.upper().replace("STAGE", "").strip()
    for g in ["IV", "III", "II", "I"]:
        if s.startswith(g):
            return g
    return np.nan

age = pd.to_numeric(clin.loc[smp["patientId"], "AGE"], errors="coerce").to_numpy()
stage = clin.loc[smp["patientId"], "AJCC_PATHOLOGIC_TUMOR_STAGE"].map(stage_group).to_numpy()
pam = clin.loc[smp["patientId"], "SUBTYPE"].astype(str).str.replace("BRCA_", "").replace("nan", np.nan).to_numpy()
CLIN = pd.DataFrame({"age": age})
for g in ["II", "III", "IV"]:
    CLIN[f"st_{g}"] = (stage == g).astype(float)
for g in ["LumB", "Her2", "Basal", "Normal"]:
    CLIN[f"pam_{g}"] = (pam == g).astype(float)
valid_clin = ~np.isnan(age) & pd.notna(stage) & pd.Series(pam).isin(["LumA", "LumB", "Her2", "Basal", "Normal"]).to_numpy()
CLIN_COLS = list(CLIN.columns)
STAGE_COLS = [c for c in CLIN_COLS if c.startswith("st_")]
print(f"Clinical comparator cohort (age+stage+PAM50 complete): n={valid_clin.sum()}, "
      f"events={E[valid_clin].sum()}")

# --- manifest + masks export ---
pd.DataFrame({"sampleId": smp["sampleId"], "patientId": smp["patientId"], "OS_days": T, "OS_event": E,
              "has_rppa_data": has_rppa, "in_rppa_list": in_rppa_list,
              "in_mut_list(NEW mask)": has_mut, "has_any_mut_call(OLD mask)": old_has_mut,
              "age": age, "stage": stage, "pam50": pam}).to_csv(out("cohort_manifest.csv"), index=False)
import lifelines, sklearn
manifest = {
    "study_id": S, "rppa_profile": RPPA_PROFILE, "mutation_profile": MUT_PROFILE,
    "mutation_sample_list": MUT_LIST, "rppa_sample_list": RPPA_LIST,
    "retrieval": meta, "driver_panel": BRCA_DRIVER_PANEL, "config": CFG,
    "versions": {"python": platform.python_version(), "torch": torch.__version__,
                 "lifelines": lifelines.__version__, "sklearn": sklearn.__version__,
                 "numpy": np.__version__, "pandas": pd.__version__},
    "reconciliation": recon,
}
with open(out("manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, default=str)

# %% [2] Model, loss, per-fold preprocessing [F3][F5] ---------------------------
_SCREEN = {}

def screen(df, idx, present, k, tag):
    key = (tag, k, hashlib.md5(np.sort(idx).tobytes()).hexdigest())
    if key in _SCREEN:
        return _SCREEN[key]
    el = idx[present[idx]]
    res = []
    for col in df.columns:
        x = df[col].to_numpy(float)[el]
        ok = ~np.isnan(x)
        if ok.sum() < 30 or np.unique(x[ok]).size < 2:
            continue
        try:
            c = CoxPHFitter().fit(pd.DataFrame({"x": x[ok], "T": T[el][ok], "E": E[el][ok]}), "T", "E")
            res.append((col, c.summary["p"].iloc[0]))
        except Exception:
            pass
    sel = [c for c, _ in sorted(res, key=lambda z: z[1])[:k]]
    _SCREEN[key] = sel
    return sel

def prep_fold(tr):
    """Screening, within-panel imputation, standardization — train rows only."""
    sp = screen(Xp_raw, tr, has_rppa, CFG["TOP_K_PROT"], "p")
    sg = screen(Xg_raw, tr, has_mut, CFG["TOP_K_GEN"], "g")
    Xp = Xp_raw[sp].to_numpy(float).copy()
    trp = tr[has_rppa[tr]]
    med = np.nanmedian(Xp[trp], 0)
    Xp[has_rppa] = np.where(np.isnan(Xp[has_rppa]), med, Xp[has_rppa])
    mu_, sd_ = Xp[trp].mean(0), Xp[trp].std(0) + 1e-6
    Xp = (Xp - mu_) / sd_
    c_p = np.median(Xp[trp], 0)                  # imputation value for missing modality
    Xp[~has_rppa] = c_p
    Xg = Xg_raw[sg].to_numpy(float).copy()
    trg = tr[has_mut[tr]]
    c_g = np.median(Xg[trg], 0)
    Xg[~has_mut] = c_g
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    return {"sel_p": sp, "sel_g": sg, "Xp": Xp, "Xg": Xg, "c_p_np": c_p, "c_g_np": c_g,
            "xp": t(Xp), "xg": t(Xg), "mp": t((~has_rppa).astype(float)), "mg": t((~has_mut).astype(float)),
            "c_p": t(c_p), "c_g": t(c_g), "T": t(T), "E": t(E)}

class SurvNet(nn.Module):
    """token:     x_mod <- t_mod if modality missing; h = [x_p, x_g]
       indicator: x_mod <- c_mod (train median) if missing; h = [x_p, x_g, m_p, m_g]
       both:      Linear(h,20) -> LayerNorm -> ReLU -> Dropout(.35) -> Linear(20,4) -> Sigmoid"""
    def __init__(self, n_p, n_g, mode):
        super().__init__()
        self.mode = mode
        if mode == "token":
            self.tok_p = nn.Parameter(torch.randn(n_p) * 0.02)
            self.tok_g = nn.Parameter(torch.randn(n_g) * 0.02)
            d = n_p + n_g
        else:
            d = n_p + n_g + 2
        self.fc1 = nn.Linear(d, CFG["HIDDEN"])
        self.ln = nn.LayerNorm(CFG["HIDDEN"])
        self.drop = nn.Dropout(CFG["DROPOUT"])
        self.fc2 = nn.Linear(CFG["HIDDEN"], len(CFG["TIME_BINS"]))

    def first(self, xp, xg, mp, mg):
        if self.mode == "token":
            xp = torch.where(mp[:, None] > 0.5, self.tok_p.expand_as(xp), xp)
            xg = torch.where(mg[:, None] > 0.5, self.tok_g.expand_as(xg), xg)
            return self.fc1(torch.cat([xp, xg], 1))
        return self.fc1(torch.cat([xp, xg, mp[:, None], mg[:, None]], 1))

    def forward(self, xp, xg, mp, mg):
        return torch.sigmoid(self.fc2(self.drop(torch.relu(self.ln(self.first(xp, xg, mp, mg))))))

def surv_loss(h, t, e):
    h = torch.clamp(h, 1e-4, 1 - 1e-4)
    bins = CFG["TIME_BINS"]
    st = torch.tensor([0.0] + bins[:-1], device=h.device)
    en = torch.tensor(bins, device=h.device)
    tot = 0.0
    for b in range(h.shape[1]):
        ar = (t >= st[b]).float()
        ev = ((t >= st[b]) & (t < en[b]) & (e == 1)).float()
        sv = (ar - ev) * ((t >= en[b]) | (e == 0)).float()
        tot = tot + (-ev * torch.log(h[:, b]) - sv * torch.log(1 - h[:, b])).sum() / (ar.sum() + 1e-8)
    return tot / h.shape[1]

def _inputs(F, idx):
    i = torch.as_tensor(idx, device=DEV)
    return F["xp"][i], F["xg"][i], F["mp"][i], F["mg"][i]

def train_net(mode, F, tr, seed, aug=(0.0, 0.0)):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed)
    fit, es = train_test_split(tr, test_size=CFG["ES_FRAC"], random_state=seed, stratify=E[tr])  # [F3]
    m = SurvNet(len(F["sel_p"]), len(F["sel_g"]), mode).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=CFG["LR"], weight_decay=CFG["WD"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["MAX_EPOCHS"])
    best, best_state, bad = np.inf, None, 0
    es_in = _inputs(F, es)
    es_i = torch.as_tensor(es, device=DEV)
    for _ in range(CFG["MAX_EPOCHS"]):
        m.train()
        perm = rng.permutation(fit)
        for s in range(0, len(perm), CFG["BATCH"]):
            bi = torch.as_tensor(perm[s:s + CFG["BATCH"]], device=DEV)
            xp, xg = F["xp"][bi].clone(), F["xg"][bi].clone()
            mp, mg = F["mp"][bi].clone(), F["mg"][bi].clone()
            if aug[0] > 0:
                add = (mp < 0.5) & (torch.rand(len(bi), device=DEV) < aug[0])
                mp = torch.where(add, torch.ones_like(mp), mp)
                if mode == "indicator":
                    xp[add] = F["c_p"]          # indicator arm must also lose the values
            if aug[1] > 0:
                add = (mg < 0.5) & (torch.rand(len(bi), device=DEV) < aug[1])
                mg = torch.where(add, torch.ones_like(mg), mg)
                if mode == "indicator":
                    xg[add] = F["c_g"]
            opt.zero_grad()
            loss = surv_loss(m(xp, xg, mp, mg), F["T"][bi], F["E"][bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            l = surv_loss(m(*es_in), F["T"][es_i], F["E"][es_i]).item()
        if l < best:
            best, best_state, bad = l, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= CFG["PATIENCE"]:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m

def predict(m, F, idx):
    with torch.no_grad():
        return m(*_inputs(F, idx)).cpu().numpy()

def risk_at(h, j=HIDX):
    return 1 - np.prod(1 - h[:, :j + 1], axis=1)

def cll(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(-np.log(1 - p))

def cox_risk(cph, df, t=H):
    return 1 - cph.predict_survival_function(df, times=[t]).iloc[0].to_numpy()

def km_risk(t_, e_, horizon=H):
    if len(t_) < 5 or e_.sum() == 0:
        return np.nan
    return 1 - KaplanMeierFitter().fit(t_, e_).survival_function_at_times(horizon).iloc[0]

def cidx(risk, idx=None):
    idx = np.arange(N) if idx is None else idx
    return concordance_index(T[idx], -risk[idx], E[idx])


# %% ===================== TREE BASELINES + SITE SENSITIVITY =====================
import json, numpy as np, pandas as pd
from scipy.stats import rankdata
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
import sksurv

print("scikit-survival", sksurv.__version__)

# ---- load saved state -------------------------------------------------------
arr = np.load(out("oof_arrays.npz"))
VARIANTS = ["token", "token_aug", "indicator", "indicator_aug"]
HAZ = {v: arr[f"haz_{v}"] for v in VARIANTS}
BASE = {b: arr[f"base_{b}"] for b in ["elastic_net_cox", "rsf", "gbm_survival"]}
NEST = {k: arr[f"nest_{k}"] for k in ["recal", "combo", "clinical", "stage_only", "null"]}
R = HAZ["token"].shape[0]
with open(out("results.json")) as f:
    RES = json.load(f)
assert HAZ["token"].shape[1] == N, "Saved arrays do not match current cohort — stop and check snapshot."
print(f"Loaded {R} repeats, N={N}")

# ---- A. RSF + GBM on identical folds ----------------------------------------
for r in range(R):
    skf = StratifiedKFold(CFG["N_OUTER"], shuffle=True, random_state=r)
    for k, (tr, te) in enumerate(skf.split(np.zeros(N), E)):
        F = prep_fold(tr)                       # same screening as main run (deterministic)
        Xb = np.hstack([F["Xp"], F["Xg"], (~has_rppa)[:, None], (~has_mut)[:, None]]).astype(float)
        ytr = Surv.from_arrays(E[tr].astype(bool), T[tr])
        BASE["rsf"][r, te] = RandomSurvivalForest(
            n_estimators=200, min_samples_leaf=10, max_depth=4, random_state=r, n_jobs=-1
        ).fit(Xb[tr], ytr).predict(Xb[te])
        BASE["gbm_survival"][r, te] = GradientBoostingSurvivalAnalysis(
            n_estimators=100, learning_rate=0.05, max_depth=2, random_state=r
        ).fit(Xb[tr], ytr).predict(Xb[te])
    print(f"baselines repeat {r+1}/{R} done")

np.savez(out("oof_arrays.npz"), **{f"haz_{v}": a for v, a in HAZ.items()},
         **{f"base_{b}": a for b, a in BASE.items()}, **{f"nest_{k}": a for k, a in NEST.items()})

# ---- B. discrimination + paired tests ---------------------------------------
def risk5(h):
    return 1 - np.prod(1 - h[:, :HIDX + 1], axis=1)

def c_(score, idx=None):
    idx = np.arange(N) if idx is None else idx
    return concordance_index(T[idx], -score[idx], E[idx])

rng = np.random.RandomState(42)                  # same bootstrap indices as main run
BOOT = [rng.randint(0, N, N) for _ in range(CFG["N_BOOT"])]
SCORES = {v: np.stack([risk5(HAZ[v][r]) for r in range(R)]) for v in VARIANTS}
SCORES.update({b: a for b, a in BASE.items() if not np.isnan(a).all()})
ENS = {m: np.mean([rankdata(s[r]) / N for r in range(R)], 0) for m, s in SCORES.items()}

rows = []
for m, s in SCORES.items():
    per = [c_(s[r]) for r in range(R)]
    boots = [concordance_index(T[i], -ENS[m][i], E[i]) for i in BOOT if E[i].sum() > 0]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rows.append({"method": m, "C_mean_over_repeats": np.mean(per), "C_sd_over_repeats": np.std(per),
                 "C_ensemble": c_(ENS[m]), "CI_lo": lo, "CI_hi": hi})
disc = pd.DataFrame(rows)
disc.to_csv(out("table_discrimination.csv"), index=False)
print("\n=== Discrimination (all methods) ===\n" + disc.round(4).to_string(index=False))

def paired(a, b):
    d = np.array([concordance_index(T[i], -a[i], E[i]) - concordance_index(T[i], -b[i], E[i])
                  for i in BOOT if E[i].sum() > 0])
    return {"diff": c_(a) - c_(b), "CI_lo": np.percentile(d, 2.5), "CI_hi": np.percentile(d, 97.5),
            "p_two_sided": 2 * min((d <= 0).mean(), (d >= 0).mean())}

COMPARISONS = [("token", "indicator"), ("token_aug", "indicator_aug"), ("token_aug", "token"),
               ("indicator_aug", "indicator"), ("token_aug", "indicator"),
               ("token", "elastic_net_cox"), ("token", "rsf"), ("token", "gbm_survival")]
pdiff = pd.DataFrame([{"A": a, "B": b, **paired(ENS[a], ENS[b])} for a, b in COMPARISONS])
pdiff.to_csv(out("table_paired_bootstrap.csv"), index=False)
print("\n=== Paired bootstrap ===\n" + pdiff.round(4).to_string(index=False))

# consistency check: neural/EN rows must match the main run exactly
old = {d["method"]: d for d in RES["discrimination"]}
for m in ["token", "token_aug", "indicator", "indicator_aug", "elastic_net_cox"]:
    new = disc.set_index("method").loc[m]
    for col in ["C_mean_over_repeats", "C_ensemble", "CI_lo", "CI_hi"]:
        if abs(new[col] - old[m][col]) > 1e-9:
            print(f"WARNING: {m} {col} changed {old[m][col]:.6f} -> {new[col]:.6f}; check saved results.")

cs = disc.set_index("method")["C_mean_over_repeats"]
RES["discrimination"] = disc.to_dict("records")
RES["paired_bootstrap"] = pdiff.to_dict("records")
RES["abstract_C_range"] = f"{cs.min():.3f}–{cs.max():.3f} (mean over {R} repeats of 5-fold CV; all methods)"

# ---- D. site (TSS) sensitivity on revised token ensemble score ---------------
tss = smp["sampleId"].str.split("-").str[1].to_numpy()
site_rows = []
for s in pd.Series(tss).value_counts().index:
    idx = np.where(tss == s)[0]
    ev = int(E[idx].sum())
    if len(idx) >= 30 and ev >= 5:
        site_rows.append({"site": s, "n": len(idx), "deaths": ev, "C_token": c_(ENS["token"], idx),
                          "rppa_missing_rate": float((~has_rppa[idx]).mean())})
sites = pd.DataFrame(site_rows)
keep = np.isin(tss, pd.Series(tss).value_counts()[lambda v: v >= 30].index)
dS = pd.DataFrame({"score_z": (ENS["token"][keep] - ENS["token"].mean()) / ENS["token"].std(),
                   "site": tss[keep], "T": T[keep], "E": E[keep]})
c_plain = CoxPHFitter().fit(dS[["score_z", "T", "E"]], "T", "E").summary.loc["score_z"]
c_strat = CoxPHFitter().fit(dS, "T", "E", strata=["site"]).summary.loc["score_z"]
RES["site_sensitivity"] = {
    "sites": site_rows, "n_sites": len(site_rows),
    "C_range": [float(sites["C_token"].min()), float(sites["C_token"].max())],
    "C_sd": float(sites["C_token"].std()),
    "HR_per_SD_unstratified": [float(np.exp(c_plain["coef"])), float(np.exp(c_plain["coef lower 95%"])),
                               float(np.exp(c_plain["coef upper 95%"])), float(c_plain["p"])],
    "HR_per_SD_site_stratified": [float(np.exp(c_strat["coef"])), float(np.exp(c_strat["coef lower 95%"])),
                                  float(np.exp(c_strat["coef upper 95%"])), float(c_strat["p"])],
    "n_in_cox": int(keep.sum()), "deaths_in_cox": int(E[keep].sum()),
}
print("\n=== Site sensitivity (token ensemble score) ===\n" + sites.round(4).to_string(index=False))

with open(out("results.json"), "w") as f:
    json.dump(RES, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

# ---- C + E. print a summary of all BRCA values reported in the paper -----------
cal = {d["model"]: d for d in RES["calibration"]}
st, dca, cvc, ss = RES.get("stress_test"), RES.get("dca"), RES.get("combo_vs_clinical_C"), RES["site_sensitivity"]

def f4(x):
    return f"{float(x):.4f}"

L = ["## BRCA SUMMARY (tree baselines, absolute risk, stress test, site sensitivity)"]
L.append(f"- Snapshot retrieved_utc: {meta['retrieved_utc']}; RPPA proteins in snapshot: {Xp_raw.shape[1]}")
d = disc.set_index("method")
for m, lab in [("rsf", "Random survival forest"), ("gbm_survival", "Gradient-boosted survival")]:
    if m in d.index:
        L.append(f"- {lab}: {f4(d.loc[m,'C_mean_over_repeats'])} ± {f4(d.loc[m,'C_sd_over_repeats'])}; "
                 f"{f4(d.loc[m,'C_ensemble'])} ({f4(d.loc[m,'CI_lo'])}, {f4(d.loc[m,'CI_hi'])})")
for _, rw in pdiff[pdiff["B"].isin(["rsf", "gbm_survival"])].iterrows():
    L.append(f"- Token − {rw['B']}: {rw['diff']:+.4f} [{rw['CI_lo']:+.4f}, {rw['CI_hi']:+.4f}], p = {rw['p_two_sided']:.3f}")
L.append(f"- Abstract C-index range (per-repeat means, all 7 methods): {RES['abstract_C_range']}")
for name, rw in cal.items():
    L.append(f"- {name}: Brier 95% CI [{rw['brier_CI'][0]:.4f}, {rw['brier_CI'][1]:.4f}]; "
             f"Brier−constant {rw['brier_minus_null']:+.4f} [{rw['brier_minus_null_CI'][0]:+.4f}, {rw['brier_minus_null_CI'][1]:+.4f}]; "
             f"IPA {rw['IPA']:.4f} [{rw['IPA_CI'][0]:.4f}, {rw['IPA_CI'][1]:.4f}]")
if cvc:
    L.append(f"- Combined − clinical C-index: {cvc['diff']:+.4f} [{cvc['CI_lo']:+.4f}, {cvc['CI_hi']:+.4f}], p = {cvc['p_two_sided']:.3f}")
if dca:
    thr = np.array(dca["thresholds"]); df_ = np.array(dca["diff_combo_minus_clinical"])
    lo_, hi_ = np.array(dca["diff_CI_lo"]), np.array(dca["diff_CI_hi"])
    excl = thr[(lo_ > 0) | (hi_ < 0)]
    L.append(f"- DCA combined − clinical net benefit over thresholds {thr.min():.2f}–{thr.max():.2f}: "
             f"diff range [{np.nanmin(df_):+.4f}, {np.nanmax(df_):+.4f}]; thresholds where 95% CI excludes 0: "
             + (", ".join(f"{t:.3f}" for t in excl) if len(excl) else "none"))
if st:
    L.append(f"- Stress test (held-out complete-data patients n = {st['n_eval']}, deaths = {st['events_eval']}; "
             f"mean C ± SD over seeds by forced RPPA-missing probability):")
    for j, p in enumerate(st["probs"]):
        L.append(f"  - p = {p:.2f}: token {st['token_mean'][j]:.4f} ± {st['token_sd'][j]:.4f}; "
                 f"indicator {st['indicator_mean'][j]:.4f} ± {st['indicator_sd'][j]:.4f}")
L.append(f"- Site sensitivity: {ss['n_sites']} sites with n ≥ 30 and ≥ 5 deaths; token C range "
         f"{ss['C_range'][0]:.4f}–{ss['C_range'][1]:.4f} (SD {ss['C_sd']:.4f}); HR per SD of score "
         f"unstratified {ss['HR_per_SD_unstratified'][0]:.3f} [{ss['HR_per_SD_unstratified'][1]:.3f}, "
         f"{ss['HR_per_SD_unstratified'][2]:.3f}], p = {ss['HR_per_SD_unstratified'][3]:.4f}; site-stratified "
         f"{ss['HR_per_SD_site_stratified'][0]:.3f} [{ss['HR_per_SD_site_stratified'][1]:.3f}, "
         f"{ss['HR_per_SD_site_stratified'][2]:.3f}], p = {ss['HR_per_SD_site_stratified'][3]:.4f} "
         f"(n = {ss['n_in_cox']}, deaths = {ss['deaths_in_cox']})")
block = "\n".join(L)
with open(out("summary_brca.md"), "w") as f:
    f.write(block)
print("\n" + "=" * 80 + "\n" + block + "\n" + "=" * 80)
print("Saved:", out("summary_brca.md"))
