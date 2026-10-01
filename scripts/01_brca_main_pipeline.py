# =============================================================================
# MAIN PIPELINE — Proteogenomic survival failure analysis (TCGA-BRCA)
# Leakage-controlled re-analysis; fixes [F1]-[F14] are listed below.
#
# Run as a script (python scripts/01_brca_main_pipeline.py) or paste each "# %%" block into its own
# Colab cell and run top to bottom.
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

# %% [3] Repeated 5x5 CV: 4 neural arms + baselines + nested second stage --------
VARIANTS = {"token": ("token", (0, 0)),
            "token_aug": ("token", (CFG["AUG_PROT"], CFG["AUG_GEN"])),
            "indicator": ("indicator", (0, 0)),
            "indicator_aug": ("indicator", (CFG["AUG_PROT"], CFG["AUG_GEN"]))}
R = CFG["N_REPEATS"]
HAZ = {v: np.full((R, N, len(CFG["TIME_BINS"])), np.nan) for v in VARIANTS}
BASE = {b: np.full((R, N), np.nan) for b in ["elastic_net_cox", "rsf", "gbm_survival"]}
NEST = {k: np.full((R, N), np.nan) for k in ["recal", "combo", "clinical", "stage_only", "null"]}
FOLD_FEATS, EQUIV_CTX = [], None

for r in range(R):
    skf = StratifiedKFold(CFG["N_OUTER"], shuffle=True, random_state=r)
    for k, (tr, te) in enumerate(skf.split(np.zeros(N), E)):
        F = prep_fold(tr)
        FOLD_FEATS.append({"repeat": r, "fold": k, "prot": F["sel_p"], "gen": F["sel_g"]})
        models = {}
        for v, (mode, aug) in VARIANTS.items():
            models[v] = train_net(mode, F, tr, seed=1000 * r + 10 * k, aug=aug)
            HAZ[v][r, te] = predict(models[v], F, te)
        if EQUIV_CTX is None:
            EQUIV_CTX = (models["token"], F)
        NEST["null"][r, te] = km_risk(T[tr], E[tr])

        if CFG["RUN_BASELINES"]:
            Xb = np.hstack([F["Xp"], F["Xg"], (~has_rppa)[:, None], (~has_mut)[:, None]]).astype(float)
            cols = [f"f{i}" for i in range(Xb.shape[1])]
            dtr = pd.DataFrame(Xb[tr], columns=cols).assign(T=T[tr], E=E[tr])
            try:
                en = CoxPHFitter(penalizer=0.1, l1_ratio=0.5).fit(dtr, "T", "E")
                BASE["elastic_net_cox"][r, te] = en.predict_log_partial_hazard(
                    pd.DataFrame(Xb[te], columns=cols)).to_numpy()
            except Exception as ex:
                print("EN-Cox failed:", ex)
            if HAS_SKSURV:
                ytr = Surv.from_arrays(E[tr].astype(bool), T[tr])
                BASE["rsf"][r, te] = RandomSurvivalForest(n_estimators=200, min_samples_leaf=10, max_depth=4,
                                                          random_state=r, n_jobs=-1).fit(Xb[tr], ytr).predict(Xb[te])
                BASE["gbm_survival"][r, te] = GradientBoostingSurvivalAnalysis(
                    n_estimators=100, learning_rate=0.05, max_depth=2, random_state=r).fit(Xb[tr], ytr).predict(Xb[te])

        if CFG["RUN_NESTED"]:   # [F4] inner OOF on outer-train -> fit 2nd stage -> apply to outer-test
            inner = np.full(N, np.nan)
            iskf = StratifiedKFold(CFG["N_INNER"], shuffle=True, random_state=100 + r)
            for j, (a, b) in enumerate(iskf.split(np.zeros(len(tr)), E[tr])):
                itr, ite = tr[a], tr[b]
                Fi = prep_fold(itr)
                inner[ite] = risk_at(predict(train_net("token", Fi, itr, seed=5000 + 100 * r + 10 * k + j), Fi, ite))
            raw_te = risk_at(HAZ["token"][r, te])
            rc = CoxPHFitter().fit(pd.DataFrame({"cll": cll(inner[tr]), "T": T[tr], "E": E[tr]}), "T", "E")
            NEST["recal"][r, te] = cox_risk(rc, pd.DataFrame({"cll": cll(raw_te)}))
            trc, tec = tr[valid_clin[tr]], te[valid_clin[te]]
            raw_full = np.full(N, np.nan); raw_full[te] = raw_te
            Dtr = CLIN.iloc[trc].assign(cll_model=cll(inner[trc]), T=T[trc], E=E[trc])
            Dte = CLIN.iloc[tec].assign(cll_model=cll(raw_full[tec]))
            for key, cols in [("combo", CLIN_COLS + ["cll_model"]), ("clinical", CLIN_COLS), ("stage_only", STAGE_COLS)]:
                cph = CoxPHFitter(penalizer=0.01).fit(Dtr[cols + ["T", "E"]], "T", "E")
                NEST[key][r, tec] = cox_risk(cph, Dte[cols])
        print(f"repeat {r+1}/{R} fold {k+1}/{CFG['N_OUTER']} done")

np.savez(out("oof_arrays.npz"), **{f"haz_{v}": a for v, a in HAZ.items()},
         **{f"base_{b}": a for b, a in BASE.items()}, **{f"nest_{k}": a for k, a in NEST.items()})
with open(out("fold_features.json"), "w") as f:
    json.dump(FOLD_FEATS, f)

# %% [4] Discrimination table + paired bootstrap [F6][F14] -----------------------
def boot_idx(n_boot, seed=42, n=N):
    rng = np.random.RandomState(seed)
    return [rng.randint(0, n, n) for _ in range(n_boot)]

BOOT = boot_idx(CFG["N_BOOT"])
SCORES_RAW = {v: np.stack([risk_at(HAZ[v][r]) for r in range(R)]) for v in VARIANTS}
SCORES_RAW.update({b: a for b, a in BASE.items() if not np.isnan(a).all()})
ENS = {m: np.mean([rankdata(s[r]) / N for r in range(R)], 0) for m, s in SCORES_RAW.items()}  # repeat-ensemble

def boot_ci_c(score):
    vals = [concordance_index(T[i], -score[i], E[i]) for i in BOOT if E[i].sum() > 0]
    return np.percentile(vals, [2.5, 97.5])

rows = []
for m, s in SCORES_RAW.items():
    per = [cidx(s[r]) for r in range(R)]
    lo, hi = boot_ci_c(ENS[m])
    rows.append({"method": m, "C_mean_over_repeats": np.mean(per), "C_sd_over_repeats": np.std(per),
                 "C_ensemble": cidx(ENS[m]), "CI_lo": lo, "CI_hi": hi})
disc = pd.DataFrame(rows)
disc.to_csv(out("table_discrimination.csv"), index=False)
print(disc.round(4).to_string(index=False))
RESULTS["discrimination"] = disc.to_dict("records")

def paired(a, b):
    d = []
    for i in BOOT:
        if E[i].sum() == 0:
            continue
        d.append(concordance_index(T[i], -a[i], E[i]) - concordance_index(T[i], -b[i], E[i]))
    d = np.array(d)
    return {"diff": cidx(a) - cidx(b), "CI_lo": np.percentile(d, 2.5), "CI_hi": np.percentile(d, 97.5),
            "p_two_sided": 2 * min((d <= 0).mean(), (d >= 0).mean())}

COMPARISONS = [("token", "indicator"), ("token_aug", "indicator_aug"),     # matched arms
               ("token_aug", "token"), ("indicator_aug", "indicator"),     # augmentation effect
               ("token_aug", "indicator"),                                 # old comparison
               ("token", "elastic_net_cox"), ("token", "rsf"), ("token", "gbm_survival")]
prow = []
for a, b in COMPARISONS:
    if a in ENS and b in ENS:
        prow.append({"A": a, "B": b, **paired(ENS[a], ENS[b])})
pdiff = pd.DataFrame(prow)
pdiff.to_csv(out("table_paired_bootstrap.csv"), index=False)
print("\n=== Paired bootstrap (repeat-ensemble scores) ===\n" + pdiff.round(4).to_string(index=False))
RESULTS["paired_bootstrap"] = pdiff.to_dict("records")

# Real-missingness subgroups
sub = []
for name, mask in [("complete", has_rppa & has_mut), ("rppa_missing", ~has_rppa), ("mut_missing", ~has_mut)]:
    idx = np.where(mask)[0]
    for m in ["token", "indicator"]:
        sub.append({"subgroup": name, "method": m, "n": len(idx), "events": int(E[idx].sum()),
                    "C": cidx(ENS[m], idx) if E[idx].sum() >= 5 else np.nan})
RESULTS["subgroups"] = sub
print(pd.DataFrame(sub).round(4).to_string(index=False))

# %% [5] Forward-pass equivalence + parameter counts [F7] -------------------------
tm, F0 = EQUIV_CTX
n_p, n_g = len(F0["sel_p"]), len(F0["sel_g"])
im = SurvNet(n_p, n_g, "indicator").to(DEV)
with torch.no_grad():
    W = tm.fc1.weight
    Wp, Wg = W[:, :n_p], W[:, n_p:]
    im.fc1.weight[:, :n_p + n_g] = W
    im.fc1.weight[:, -2] = Wp @ (tm.tok_p - F0["c_p"])
    im.fc1.weight[:, -1] = Wg @ (tm.tok_g - F0["c_g"])
    im.fc1.bias.copy_(tm.fc1.bias)
    im.ln.load_state_dict(tm.ln.state_dict())
    im.fc2.load_state_dict(tm.fc2.state_dict())
im.eval()
allidx = np.arange(N)
maxdiff = float(np.abs(predict(tm, F0, allidx) - predict(im, F0, allidx)).max())
nparams = lambda m: sum(p.numel() for p in m.parameters())
equiv = {"n_prot": n_p, "n_gen": n_g, "hidden": CFG["HIDDEN"],
         "params_token": nparams(tm), "params_indicator": nparams(im),
         "extra_params_token(tokens)": n_p + n_g, "extra_params_indicator(2 columns)": 2 * CFG["HIDDEN"],
         "rank_Wp": int(torch.linalg.matrix_rank(Wp).item()), "rank_Wg": int(torch.linalg.matrix_rank(Wg).item()),
         "max_abs_output_diff_token_vs_constructed_indicator": maxdiff}
RESULTS["equivalence"] = equiv
print("\n=== Token vs indicator: parameterization ===")
for k, v in equiv.items():
    print(f"  {k:52s} {v}")
print("Every token model is exactly an indicator model with u_mod = W_mod (t_mod - c_mod).\n"
      "Indicator -> token only if u_mod lies in col(W_mod); with rank(Wg) <= n_gen < hidden\n"
      "this fails, so the indicator class STRICTLY CONTAINS the token class here. The\n"
      "comparison tests parameterization/regularization, not added expressivity.")

# %% [6] Calibration, IPCW Brier, IPA (nested, out-of-sample) [F2][F8][F9] ------
kmc = KaplanMeierFitter().fit(T, 1 - E)       # censoring distribution G (full cohort; stated in Methods)

def G(t):
    return kmc.survival_function_at_times(np.atleast_1d(t)).to_numpy()

case_event = (T <= H) & (E == 1)
case_surv = T > H
W_IPCW = np.zeros(N)
W_IPCW[case_event] = 1 / np.clip(G(np.maximum(T[case_event] - 1e-6, 0)), 1e-8, None)
W_IPCW[case_surv] = 1 / G(H)[0]
TARGET = np.where(case_event, 1.0, 0.0)  # event by H (risk scale); censored-before-H get weight 0

def brier(risk, idx):
    return np.sum(W_IPCW[idx] * (TARGET[idx] - risk[idx]) ** 2) / len(idx)

def cal_slope(risk, idx):
    c = CoxPHFitter().fit(pd.DataFrame({"x": cll(risk[idx]), "T": T[idx], "E": E[idx]}), "T", "E")
    s = c.summary.iloc[0]
    z = (s["coef"] - 1) / s["se(coef)"]
    return {"slope": s["coef"], "slope_CI_lo": s["coef lower 95%"], "slope_CI_hi": s["coef upper 95%"],
            "p_vs_1": 2 * (1 - norm.cdf(abs(z))), "p_vs_0": s["p"]}

def calib_block(name, risk, null, idx, B=CFG["N_BOOT"]):
    rng = np.random.RandomState(7)
    bs, bn = brier(risk, idx), brier(null, idx)
    oe = km_risk(T[idx], E[idx]) / risk[idx].mean()
    bb = []
    for _ in range(B):
        i = idx[rng.randint(0, len(idx), len(idx))]
        m_, n_ = brier(risk, i), brier(null, i)
        bb.append((m_, m_ - n_, 1 - m_ / n_))
    bb = np.array(bb)
    ci = lambda c: (np.percentile(bb[:, c], 2.5), np.percentile(bb[:, c], 97.5))
    return {"model": name, "n": len(idx), "events": int(E[idx].sum()), "C": cidx(risk, idx),
            "brier": bs, "brier_CI": ci(0), "brier_null_constant": bn,
            "brier_minus_null": bs - bn, "brier_minus_null_CI": ci(1),
            "IPA": 1 - bs / bn, "IPA_CI": ci(2), "O_over_E": oe, **cal_slope(risk, idx)}

raw5 = np.mean([risk_at(HAZ["token"][r]) for r in range(R)], 0)
null5 = np.nanmean(NEST["null"], 0)
ALL, CI_ = np.arange(N), np.where(valid_clin)[0]
cal = [calib_block("token_raw_5y (uncalibrated)", raw5, null5, ALL)]
if CFG["RUN_NESTED"]:
    recal5 = np.nanmean(NEST["recal"], 0)
    cal.append(calib_block("token_recalibrated_5y (nested)", recal5, null5, ALL))
    for key in ["combo", "clinical", "stage_only"]:
        cal.append(calib_block(f"{key}_5y (nested)", np.nanmean(NEST[key], 0), null5, CI_))
calt = pd.DataFrame(cal)
calt.to_csv(out("table_calibration.csv"), index=False)
print(calt.drop(columns=[c for c in calt.columns if c.endswith("_CI")]).round(4).to_string(index=False))
RESULTS["calibration"] = calt.to_dict("records")

if HAS_SKSURV:   # cross-check manual Graf estimator against sksurv
    y = Surv.from_arrays(E.astype(bool), T)
    _, bs_sk = sksurv_brier(y, y, (1 - raw5).reshape(-1, 1), [H])
    RESULTS["brier_crosscheck"] = {"manual": brier(raw5, ALL), "sksurv": float(bs_sk[0])}
    print("Brier cross-check (manual vs sksurv):", RESULTS["brier_crosscheck"])

if CFG["RUN_NESTED"]:
    # Paired bootstrap restricted to the clinical cohort (patients with age, stage and PAM50),
    # identical to the procedure in 03_extension (seed 1). Paper: +0.0056 [-0.0026, +0.0142], p = 0.162.
    _cmb, _cln = np.nanmean(NEST["combo"], 0), np.nanmean(NEST["clinical"], 0)
    _rr, _d = np.random.RandomState(1), []
    for _ in range(CFG["N_BOOT"]):
        _i = CI_[_rr.randint(0, len(CI_), len(CI_))]
        if E[_i].sum() == 0:
            continue
        _d.append(concordance_index(T[_i], -_cmb[_i], E[_i]) - concordance_index(T[_i], -_cln[_i], E[_i]))
    _d = np.array(_d)
    RESULTS["combo_vs_clinical_C"] = {"diff": cidx(_cmb, CI_) - cidx(_cln, CI_),
                                      "CI_lo": np.percentile(_d, 2.5), "CI_hi": np.percentile(_d, 97.5),
                                      "p_two_sided": 2 * min((_d <= 0).mean(), (_d >= 0).mean())}
    print("Combined - clinical C:", RESULTS["combo_vs_clinical_C"])
    # PH diagnostics (descriptive, full-cohort fit)
    D = CLIN.iloc[CI_].assign(cll_model=cll(raw5[CI_]), T=T[CI_], E=E[CI_])
    ph = proportional_hazard_test(CoxPHFitter(penalizer=0.01).fit(D, "T", "E"), D, time_transform="rank")
    RESULTS["ph_test"] = ph.summary.reset_index().to_dict("records")
    print(ph.summary.round(4).to_string())

# calibration plot
fig, axes = plt.subplots(1, 2 if CFG["RUN_NESTED"] else 1, figsize=(11, 5.2), squeeze=False)
for ax, (lab, rk) in zip(axes[0], [("Uncalibrated", raw5)] + ([("Recalibrated (nested)", recal5)] if CFG["RUN_NESTED"] else [])):
    q = pd.qcut(rk, 10, labels=False, duplicates="drop")
    pts = [(rk[q == b].mean(), km_risk(T[q == b], E[q == b])) for b in np.unique(q)]
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.scatter(*zip(*pts), color="#2b5c8f")
    lim = max(max(p) for p in pts) * 1.15
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Mean predicted 5-year risk (decile)"); ax.set_ylabel("KM observed 5-year risk")
    ax.set_title(lab); ax.grid(ls="--", alpha=.4)
fig.tight_layout(); fig.savefig(out("calibration_plot.png"), dpi=300, bbox_inches="tight"); plt.show()

# %% [7] Decision curves (nested, out-of-sample) ----------------------------------
def net_benefit(risk, t_, e_, thr):
    nb = []
    for pt in thr:
        hi = risk >= pt
        if hi.sum() == 0:
            nb.append(0.0); continue
        p = km_risk(t_[hi], e_[hi])
        nb.append(np.nan if np.isnan(p) else hi.mean() * p - hi.mean() * (1 - p) * pt / (1 - pt))
    return np.array(nb)

if CFG["RUN_NESTED"]:
    thr = np.linspace(0.03, 0.40, 30)
    tt, ee = T[CI_], E[CI_]
    curves = {k: np.nanmean(NEST[k], 0)[CI_] for k in ["combo", "clinical"]}
    rng = np.random.RandomState(42)
    bidx = [rng.randint(0, len(CI_), len(CI_)) for _ in range(CFG["N_BOOT_DCA"])]
    dca = {}
    for k, rk in curves.items():
        pt = net_benefit(rk, tt, ee, thr)
        bs = np.array([net_benefit(rk[i], tt[i], ee[i], thr) for i in bidx])
        dca[k] = (pt, np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0))
    diff_b = np.array([net_benefit(curves["combo"][i], tt[i], ee[i], thr) -
                       net_benefit(curves["clinical"][i], tt[i], ee[i], thr) for i in bidx])
    p_all = km_risk(tt, ee)
    treat_all = p_all - (1 - p_all) * thr / (1 - thr)
    fig, ax = plt.subplots(figsize=(6, 6))
    sty = {"combo": ("#2b5c8f", "-", "Clinical + omics (nested)"), "clinical": ("#e08214", "--", "Clinical: age + stage + PAM50")}
    hs = []
    for k, (pt, lo, hi) in dca.items():
        c, ls, lab = sty[k]
        hs.append(ax.plot(thr, pt, color=c, ls=ls, lw=2.2, label=lab)[0])
        ax.fill_between(thr, lo, hi, color=c, alpha=.15)
    hs.append(ax.plot(thr, treat_all, color="gray", ls=":", lw=1.5, label="Treat all")[0])
    hs.append(ax.axhline(0, color="black", lw=1, label="Treat none"))
    ax.set_xlabel("Risk threshold (5-year mortality probability)"); ax.set_ylabel("Net benefit")
    ax.set_ylim(-0.02, max(dca["combo"][2].max(), treat_all.max()) + 0.02); ax.grid(ls="--", alpha=.4)
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0., 1.02, 1., .102), ncol=2, mode="expand",
              borderaxespad=0., fontsize=9)
    fig.tight_layout(); fig.savefig(out("dca_nested.png"), dpi=300, bbox_inches="tight"); plt.show()
    RESULTS["dca"] = {"thresholds": thr.tolist(),
                      **{f"{k}_nb": v[0].tolist() for k, v in dca.items()},
                      "diff_combo_minus_clinical": (dca["combo"][0] - dca["clinical"][0]).tolist(),
                      "diff_CI_lo": np.nanpercentile(diff_b, 2.5, 0).tolist(),
                      "diff_CI_hi": np.nanpercentile(diff_b, 97.5, 0).tolist()}

# %% [8] Synthetic stress test — held-out patients, both arms [F11] ---------------
if CFG["RUN_STRESS"]:
    tr_s, te_s = train_test_split(np.arange(N), test_size=0.25, random_state=42, stratify=E)
    Fs = prep_fold(tr_s)
    te_c = te_s[has_rppa[te_s] & has_mut[te_s]]
    res = {m: np.zeros((CFG["STRESS_SEEDS"], len(CFG["STRESS_PROBS"]))) for m in ["token", "indicator"]}
    for sd in range(CFG["STRESS_SEEDS"]):
        for mode in res:
            m = train_net(mode, Fs, tr_s, seed=9000 + sd)
            for j, p in enumerate(CFG["STRESS_PROBS"]):
                force = np.random.RandomState(sd * 100 + j).rand(len(te_c)) < p
                xp, xg, mp, mg = [z.clone() for z in _inputs(Fs, te_c)]
                ft = torch.as_tensor(force, device=DEV)
                mp = torch.where(ft, torch.ones_like(mp), mp)
                if mode == "indicator":
                    xp[ft] = Fs["c_p"]
                with torch.no_grad():
                    rk = risk_at(m(xp, xg, mp, mg).cpu().numpy())
                res[mode][sd, j] = concordance_index(T[te_c], -rk, E[te_c])
    fig, ax = plt.subplots(figsize=(6, 6))
    for mode, c in [("token", "#2b5c8f"), ("indicator", "#e08214")]:
        mu_, lo_, hi_ = res[mode].mean(0), res[mode].min(0), res[mode].max(0)
        ax.plot(CFG["STRESS_PROBS"], mu_, "o-", color=c, lw=2, label=f"{mode} (mean of {CFG['STRESS_SEEDS']} seeds)")
        ax.fill_between(CFG["STRESS_PROBS"], lo_, hi_, color=c, alpha=.15)
    ax.axhline(.5, color="gray", ls=":", label="Chance")
    ax.set_xlabel("Forced RPPA missingness (held-out complete-data patients)"); ax.set_ylabel("C-index")
    ax.grid(ls="--", alpha=.4)
    ax.legend(loc="lower left", bbox_to_anchor=(0., 1.02, 1., .102), ncol=2, mode="expand", fontsize=9)
    fig.tight_layout(); fig.savefig(out("stress_test_heldout.png"), dpi=300, bbox_inches="tight"); plt.show()
    RESULTS["stress_test"] = {"n_eval": len(te_c), "events_eval": int(E[te_c].sum()), "probs": CFG["STRESS_PROBS"],
                              **{f"{m}_mean": v.mean(0).tolist() for m, v in res.items()},
                              **{f"{m}_sd": v.std(0).tolist() for m, v in res.items()}}

# %% [9] Learning curves: token vs indicator over training size [F12] -------------
if CFG["RUN_LEARNING_CURVE"]:
    lc = []
    for r in range(CFG["LC_REPEATS"]):
        skf = StratifiedKFold(CFG["N_OUTER"], shuffle=True, random_state=r)
        for k, (tr, te) in enumerate(skf.split(np.zeros(N), E)):
            for frac in CFG["LC_FRACTIONS"]:
                sub_tr = tr if frac == 1.0 else train_test_split(tr, train_size=frac, stratify=E[tr], random_state=r)[0]
                Fl = prep_fold(sub_tr)
                row = {"repeat": r, "fold": k, "frac": frac, "n_train": len(sub_tr), "events_train": int(E[sub_tr].sum())}
                for mode in ["token", "indicator"]:
                    m_lc = train_net(mode, Fl, sub_tr, seed=7000 + 100 * r + 10 * k)
                    row[mode] = concordance_index(T[te], -risk_at(predict(m_lc, Fl, te)), E[te])
                lc.append(row)
        print(f"learning-curve repeat {r+1} done")
    lc = pd.DataFrame(lc)
    lc["diff"] = lc["token"] - lc["indicator"]
    lcs = lc.groupby("frac").agg(n_train=("n_train", "mean"), events_train=("events_train", "mean"),
                                 token=("token", "mean"), indicator=("indicator", "mean"),
                                 diff_mean=("diff", "mean"), diff_sd=("diff", "std")).reset_index()
    lc.to_csv(out("learning_curve_folds.csv"), index=False)
    _nf = lc.groupby("frac").size().to_numpy()
    lcs["n_folds"] = _nf
    lcs["diff_CI_lo_naive"] = lcs["diff_mean"] - 1.96 * lcs["diff_sd"] / np.sqrt(_nf)   # treats folds as independent
    lcs["diff_CI_hi_naive"] = lcs["diff_mean"] + 1.96 * lcs["diff_sd"] / np.sqrt(_nf)
    lcs["p_naive_normal"] = 2 * (1 - norm.cdf(np.abs(lcs["diff_mean"] / (lcs["diff_sd"] / np.sqrt(_nf)))))
    lcs.to_csv(out("table_learning_curve.csv"), index=False)
    print(lcs.round(4).to_string(index=False))
    RESULTS["learning_curve"] = lcs.to_dict("records")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(lcs["events_train"], lcs["token"], "o-", color="#2b5c8f", label="Learnable token")
    ax.plot(lcs["events_train"], lcs["indicator"], "s--", color="#e08214", label="Impute + indicator")
    ax.set_xlabel("Training events (mean over folds)"); ax.set_ylabel("Held-out C-index"); ax.grid(ls="--", alpha=.4)
    ax.legend(); fig.tight_layout(); fig.savefig(out("learning_curve.png"), dpi=300, bbox_inches="tight"); plt.show()
    # Paper figure: token - indicator difference with naive fold-level 95% intervals
    g = lc.groupby("frac"); m_ = g["diff"].mean(); se_ = g["diff"].std() / np.sqrt(g.size())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(g["events_train"].mean(), m_, yerr=1.96 * se_, fmt="o-", color="#2b5c8f", capsize=4)
    ax.axhline(0, color="gray", ls=":"); ax.set_xlabel("Training events (mean over folds)")
    ax.set_ylabel("C-index difference (token - indicator)"); ax.grid(ls="--", alpha=.4)
    fig.tight_layout(); fig.savefig(out("learning_curve_diff.png"), dpi=300, bbox_inches="tight"); plt.show()

# %% [10] Save everything + numbers for the manuscript ----------------------------
pd.DataFrame({"sampleId": smp["sampleId"], "event": E, "time_days": T,
              **{f"risk5_{v}_mean": np.mean([risk_at(HAZ[v][r]) for r in range(R)], 0) for v in VARIANTS},
              **({f"nested_{k}": np.nanmean(NEST[k], 0) for k in NEST} if CFG["RUN_NESTED"] else {})}
             ).to_csv(out("oof_predictions.csv"), index=False)

cs = disc.set_index("method")["C_mean_over_repeats"]
RESULTS["abstract_C_range"] = f"{cs.min():.3f}–{cs.max():.3f} (mean over {R} repeats of 5-fold CV; all methods)"
with open(out("results.json"), "w") as f:
    json.dump(RESULTS, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

print("\n================ MANUSCRIPT NUMBERS (verify against results.json) ================")
print("Cohort:", recon["COHORT_n"], "patients,", recon["COHORT_events"], "events")
print(f"Missingness: RPPA {(~has_rppa).mean():.1%}; mutation {(~has_mut).mean():.1%} "
      f"(previously {(~old_has_mut).mean():.1%} under the 'no calls = missing' definition)")
print("Abstract C-index range:", RESULTS["abstract_C_range"])
for rw in RESULTS["paired_bootstrap"]:
    print(f"  {rw['A']} - {rw['B']}: {rw['diff']:+.4f} [{rw['CI_lo']:+.4f}, {rw['CI_hi']:+.4f}] p={rw['p_two_sided']:.3f}")
for rw in RESULTS["calibration"]:
    print(f"  {rw['model']}: slope {rw['slope']:.3f} [{rw['slope_CI_lo']:.3f},{rw['slope_CI_hi']:.3f}] "
          f"p(vs 1)={rw['p_vs_1']:.3g}; O/E {rw['O_over_E']:.2f}; Brier {rw['brier']:.4f} vs null "
          f"{rw['brier_null_constant']:.4f}; IPA {rw['IPA']:.3f}")
print("All outputs in", CFG["RESULTS_DIR"])
