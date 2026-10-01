# =============================================================================
# 03 — EXTENSION: replication cohorts and additional missing-modality mechanisms
#   (1) More cohorts: TCGA LUAD and KIRC (PanCancer Atlas), both with natural RPPA missingness
#   (2) More missing-modality mechanisms, run on ALL cohorts (incl. BRCA):
#       - "regimpute": regression imputation of RPPA from mutation features + indicator
#                      (classical model-based imputation)
#       - "latefusion": per-modality encoders, masked mean pooling over available modalities
#                       (common deep multimodal pattern; NOT reducible to an input-level token)
#
# HOW TO RUN: python scripts/03_extension_luad_kirc_mechanisms.py, or paste into ONE Colab cell
# (scikit-survival must import; do not put !pip in the same cell). Requires 01 (and 02 if
# 01 skipped tree baselines) to have been run: reuses the frozen BRCA snapshot and saved BRCA arrays.
# Also runs the KIRC analysis with histologic grade added to the clinical comparator (RUN_KIRC_GRADE).
# New cohorts are snapshotted to Drive on first fetch.
# RESUMABLE: each cohort's arrays are saved when finished; rerunning skips finished cohorts.
# Runtime, Colab CPU: BRCA new arms ~15 min; each new cohort ~45-60 min. ~2 h total.
# =============================================================================
import os, json, copy, hashlib, datetime, time, warnings
import numpy as np, pandas as pd, requests, torch, torch.nn as nn
from scipy.stats import rankdata, norm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import Ridge
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
try:
    from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
    from sksurv.util import Surv
    HAS_SKSURV = True
except Exception as ex:
    print("scikit-survival unavailable -> RSF/GBM skipped:", ex); HAS_SKSURV = False
BASE_DIR = os.environ.get("PGR_BASE_DIR")
if BASE_DIR is None:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        BASE_DIR = "/content/drive/MyDrive/proteogenomic_revision"
    except Exception:
        BASE_DIR = "./proteogenomic_revision"

CFG = dict(
    SNAPSHOT_DIR=f"{BASE_DIR}/snapshot", RESULTS_DIR=f"{BASE_DIR}/results",
    EXT_DIR=f"{BASE_DIR}/results/extension",
    N_REPEATS=5, N_OUTER=5, N_INNER=5, ES_FRAC=0.15, TOP_K_PROT=25, TOP_K_GEN=15,
    TIME_BINS=[365.0, 1095.0, 1825.0, 2555.0], HORIZON=1825.0, HORIZON_IDX=2,
    HIDDEN=20, DROPOUT=0.35, LR=8e-4, WD=3e-2, BATCH=32, MAX_EPOCHS=100, PATIENCE=20,
    N_BOOT=2000, RIDGE_ALPHA=1.0,
    RUN_KIRC_GRADE=True,
)
os.makedirs(CFG["EXT_DIR"], exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H, HIDX = CFG["HORIZON"], CFG["HORIZON_IDX"]
BRCA = "brca_tcga_pan_can_atlas_2018"

BRCA_DRIVER_PANEL = sorted(set([
    'TP53','PIK3CA','GATA3','MAP3K1','CDH1','PTEN','AKT1','ERBB2','ESR1',
    'MYC','CCND1','RB1','BRCA1','BRCA2','NF1','ARID1A','KMT2C','FOXA1',
    'RUNX1','TBX3','CBFB','PIK3R1','NCOR1','SF3B1','CTCF','MAP2K4',
    'CDKN1B','SMAD4','ATR','STK11','ERBB3','KRAS','NRAS','BRAF','EGFR',
    'MTOR','TSC1','TSC2','CDKN2A','MDM2','MDM4','ATM','CHEK2','PALB2',
]))
# New cohorts: same fixed panel + canonical cohort drivers from the TCGA marker papers.
COHORTS = [
    dict(study=BRCA, label="BRCA", extra=[], pam50=True),
    dict(study="luad_tcga_pan_can_atlas_2018", label="LUAD",
         extra=["KEAP1", "SMARCA4", "RBM10", "U2AF1", "SETD2", "MET"], pam50=False),
    dict(study="kirc_tcga_pan_can_atlas_2018", label="KIRC",
         extra=["VHL", "PBRM1", "SETD2", "BAP1", "KDM5C"], pam50=False),
]
NEW_ARMS_ONLY_BRCA = ["regimpute", "latefusion"]           # BRCA old arms are loaded from disk
ARMS_NEW_COHORT = ["token", "indicator", "regimpute", "latefusion"]
ARM_MODE = {"token": "token", "indicator": "indicator", "regimpute": "regimpute",
            "latefusion": "latefusion", "token_aug": "token", "indicator_aug": "indicator"}

# ---------------------------------------------------------------- API + snapshot
API = "https://www.cbioportal.org/api"

def _request(method, ep, tries=8, **kw):
    for a in range(tries):
        try:
            r = requests.request(method, API + ep, **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} for {ep}", response=r)
            r.raise_for_status(); return r.json()
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as ex:
            code = getattr(getattr(ex, "response", None), "status_code", None)
            if (code is not None and code < 500 and code != 429) or a == tries - 1:
                raise
            w = min(2 ** a * 5, 300); print(f"  cBioPortal {ep}: {ex} -> retry in {w}s"); time.sleep(w)

def _get(ep, params=None): return _request("GET", ep, params=params, timeout=120)
def _post(ep, body, params=None): return _request("POST", ep, json=body, params=params, timeout=300)

def stage_group(s):
    if not isinstance(s, str): return np.nan
    s = s.upper().replace("STAGE", "").strip()
    for g in ["IV", "III", "II", "I"]:
        if s.startswith(g): return g
    return np.nan

def load_cohort(c):
    study = c["study"]
    sdir = CFG["SNAPSHOT_DIR"] if study == BRCA else os.path.join(CFG["SNAPSHOT_DIR"], study)
    os.makedirs(sdir, exist_ok=True)

    def snap(name, fn):
        p = os.path.join(sdir, name + ".json")
        if os.path.exists(p):
            with open(p) as f: return json.load(f)
        d = fn()
        with open(p, "w") as f: json.dump(d, f)
        return d

    def paged(kind):
        out, page = [], 0
        while True:
            b = _get(f"/studies/{study}/clinical-data", {"clinicalDataType": kind, "pageSize": 10000,
                                                         "pageNumber": page, "projection": "SUMMARY"})
            out += b
            if len(b) < 10000: return out
            page += 1

    def chunked(ep, ids, params):
        out = []
        for i in range(0, len(ids), 500): out += _post(ep, {"sampleIds": ids[i:i + 500]}, params)
        return out

    meta = snap("retrieval_meta", lambda: {"retrieved_utc": datetime.datetime.utcnow().isoformat() + "Z"})
    profiles = snap("molecular_profiles", lambda: _get(f"/studies/{study}/molecular-profiles"))
    pids = {p["molecularProfileId"] for p in profiles}
    rppa_id = next((p for p in [f"{study}_rppa_Zscores", f"{study}_rppa"] if p in pids), None)
    mut_id = f"{study}_mutations"
    if rppa_id is None or mut_id not in pids:
        print(f"[{c['label']}] missing RPPA or mutation profile -> cohort skipped"); return None
    pat = snap("clinical_patient", lambda: paged("PATIENT"))
    samples = snap("samples", lambda: _get(f"/studies/{study}/samples", {"pageSize": 100000}))
    lists = snap("sample_lists", lambda: _get(f"/studies/{study}/sample-lists"))
    def find_list(cat, suf):
        for sl in lists:
            if sl.get("category") == cat: return sl["sampleListId"]
        cand = f"{study}_{suf}"
        return cand if any(sl["sampleListId"] == cand for sl in lists) else None
    mut_list, rppa_list = find_list("all_cases_with_mutation_data", "sequenced"), find_list("all_cases_with_rppa_data", "rppa")
    if mut_list is None:
        print(f"[{c['label']}] no mutation sample list -> skipped"); return None
    mut_listed = set(snap("samples_in_mut_list", lambda: _get(f"/sample-lists/{mut_list}/sample-ids")))
    rppa_listed = set(snap("samples_in_rppa_list", lambda: _get(f"/sample-lists/{rppa_list}/sample-ids"))) if rppa_list else set()
    all_ids = [s["sampleId"] for s in samples]
    rppa_raw = snap("rppa_zscores", lambda: chunked(f"/molecular-profiles/{rppa_id}/molecular-data/fetch", all_ids, {"projection": "DETAILED"}))
    mut_raw = snap("mutations", lambda: chunked(f"/molecular-profiles/{mut_id}/mutations/fetch", all_ids, {"projection": "DETAILED"}))

    clin = pd.DataFrame(pat).pivot_table(index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first")
    os_m = pd.to_numeric(clin.get("OS_MONTHS"), errors="coerce")
    ok = clin["OS_STATUS"].notna() & os_m.notna()
    clin = clin[ok].copy(); clin["T"] = os_m[ok] * 30.4375
    clin["E"] = clin["OS_STATUS"].astype(str).str.startswith("1").astype(int)
    smp = pd.DataFrame(samples)[["sampleId", "patientId"]]
    smp = smp[smp["patientId"].isin(clin.index)]
    if smp["patientId"].duplicated().any():
        smp = smp[smp["sampleId"].str.endswith("-01")].drop_duplicates("patientId")
    smp = smp.reset_index(drop=True)
    T = clin.loc[smp["patientId"], "T"].to_numpy(float); E = clin.loc[smp["patientId"], "E"].to_numpy(int)

    rp = pd.DataFrame(rppa_raw); rp["gene"] = rp["gene"].apply(lambda g: g.get("hugoGeneSymbol"))
    Xp_raw = rp.pivot_table(index="sampleId", columns="gene", values="value", aggfunc="first").reindex(smp["sampleId"]).reset_index(drop=True)
    has_rppa = Xp_raw.notna().any(axis=1).to_numpy()

    panel = sorted(set(BRCA_DRIVER_PANEL) | set(c["extra"]))
    mu = pd.DataFrame(mut_raw); mu["gene"] = mu["gene"].apply(lambda g: g.get("hugoGeneSymbol"))
    has_mut = smp["sampleId"].isin(mut_listed).to_numpy()
    sub = mu[mu["gene"].isin(panel)]
    Xg_raw = (pd.crosstab(sub["sampleId"], sub["gene"]).gt(0).astype(float)
              .reindex(columns=panel, fill_value=0.0).reindex(smp["sampleId"]).fillna(0.0).reset_index(drop=True))
    Xg_raw.loc[~has_mut, :] = np.nan

    age = pd.to_numeric(clin.loc[smp["patientId"]].get("AGE"), errors="coerce").to_numpy()
    stage = clin.loc[smp["patientId"]].get("AJCC_PATHOLOGIC_TUMOR_STAGE", pd.Series(index=smp["patientId"], dtype=object)).map(stage_group).to_numpy()
    CLIN = pd.DataFrame({"age": age})
    for g in ["II", "III", "IV"]: CLIN[f"st_{g}"] = (stage == g).astype(float)
    valid = ~np.isnan(age) & pd.notna(stage)
    if c["pam50"]:
        pam = clin.loc[smp["patientId"], "SUBTYPE"].astype(str).str.replace("BRCA_", "").replace("nan", np.nan).to_numpy()
        for g in ["LumB", "Her2", "Basal", "Normal"]: CLIN[f"pam_{g}"] = (pam == g).astype(float)
        valid &= pd.Series(pam).isin(["LumA", "LumB", "Her2", "Basal", "Normal"]).to_numpy()
    cols = [k for k in CLIN.columns if k == "age" or CLIN.loc[valid, k].sum() >= 5]   # drop near-empty dummies
    D = dict(label=c["label"], study=study, smp=smp, T=T, E=E, N=len(smp), Xp_raw=Xp_raw, Xg_raw=Xg_raw,
             has_rppa=has_rppa, has_mut=has_mut, CLIN=CLIN, CLIN_COLS=cols,
             STAGE_COLS=[k for k in cols if k.startswith("st_")], valid_clin=valid, meta=meta,
             rppa_profile=rppa_id, mut_list=mut_list, rppa_list=rppa_list, panel_size=len(panel))
    kmc = KaplanMeierFitter().fit(T, 1 - E)
    D["summary"] = {"n": int(len(smp)), "deaths": int(E.sum()), "rppa_missing": int((~has_rppa).sum()),
                    "rppa_missing_pct": float((~has_rppa).mean() * 100), "rppa_missing_deaths": int(E[~has_rppa].sum()),
                    "mut_missing": int((~has_mut).sum()), "mut_missing_pct": float((~has_mut).mean() * 100),
                    "rppa_list_vs_data_disagree": int((has_rppa != smp["sampleId"].isin(rppa_listed).to_numpy()).sum()) if rppa_list else None,
                    "n_proteins": int(Xp_raw.shape[1]), "panel_genes": len(panel),
                    "clinical_n": int(valid.sum()), "clinical_deaths": int(E[valid].sum()), "clinical_cols": cols,
                    "censoring_survival_at_horizon": float(kmc.survival_function_at_times(H).iloc[0]),
                    "zero_followup": int((T <= 0).sum()), "retrieved_utc": meta.get("retrieved_utc")}
    print(f"[{c['label']}] " + json.dumps(D["summary"]))
    return D

# ---------------------------------------------------------------- per-fold prep + models
_SCREEN = {}
def screen(D, df, idx, present, k, tag):
    key = (D["label"], tag, k, hashlib.md5(np.sort(idx).tobytes()).hexdigest())
    if key in _SCREEN: return _SCREEN[key]
    T, E = D["T"], D["E"]; el = idx[present[idx]]; res = []
    for col in df.columns:
        x = df[col].to_numpy(float)[el]; ok = ~np.isnan(x)
        if ok.sum() < 30 or np.unique(x[ok]).size < 2: continue
        try:
            cc = CoxPHFitter().fit(pd.DataFrame({"x": x[ok], "T": T[el][ok], "E": E[el][ok]}), "T", "E")
            res.append((col, cc.summary["p"].iloc[0]))
        except Exception:
            pass
    sel = [c for c, _ in sorted(res, key=lambda z: z[1])[:k]]
    _SCREEN[key] = sel; return sel

def prep_fold(D, tr):
    hr, hm = D["has_rppa"], D["has_mut"]
    sp = screen(D, D["Xp_raw"], tr, hr, CFG["TOP_K_PROT"], "p")
    sg = screen(D, D["Xg_raw"], tr, hm, CFG["TOP_K_GEN"], "g")
    Xp = D["Xp_raw"][sp].to_numpy(float).copy(); trp = tr[hr[tr]]
    med = np.nanmedian(Xp[trp], 0); Xp[hr] = np.where(np.isnan(Xp[hr]), med, Xp[hr])
    mu_, sd_ = Xp[trp].mean(0), Xp[trp].std(0) + 1e-6; Xp = (Xp - mu_) / sd_
    c_p = np.median(Xp[trp], 0); Xp[~hr] = c_p
    Xg = D["Xg_raw"][sg].to_numpy(float).copy(); trg = tr[hm[tr]]
    c_g = np.median(Xg[trg], 0); Xg[~hm] = c_g
    rid = Ridge(alpha=CFG["RIDGE_ALPHA"]).fit(Xg[trp], Xp[trp])       # regression imputation, train rows only
    Pp = rid.predict(Xg); Xp_reg = Xp.copy(); Xp_reg[~hr] = Pp[~hr]
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    return {"sel_p": sp, "sel_g": sg, "Xp": Xp, "Xg": Xg, "xp": t(Xp), "xp_reg": t(Xp_reg), "pp": t(Pp),
            "xg": t(Xg), "mp": t((~hr).astype(float)), "mg": t((~hm).astype(float)),
            "c_p": t(c_p), "c_g": t(c_g), "T": t(D["T"]), "E": t(D["E"])}

class SurvNet(nn.Module):
    """token / indicator / regimpute (indicator net with regression-imputed fill)."""
    def __init__(s, n_p, n_g, mode):
        super().__init__(); s.mode = mode
        if mode == "token":
            s.tok_p = nn.Parameter(torch.randn(n_p) * 0.02); s.tok_g = nn.Parameter(torch.randn(n_g) * 0.02); d = n_p + n_g
        else:
            d = n_p + n_g + 2
        s.fc1 = nn.Linear(d, CFG["HIDDEN"]); s.ln = nn.LayerNorm(CFG["HIDDEN"])
        s.drop = nn.Dropout(CFG["DROPOUT"]); s.fc2 = nn.Linear(CFG["HIDDEN"], len(CFG["TIME_BINS"]))
    def forward(s, xp, xg, mp, mg):
        if s.mode == "token":
            xp = torch.where(mp[:, None] > 0.5, s.tok_p.expand_as(xp), xp)
            xg = torch.where(mg[:, None] > 0.5, s.tok_g.expand_as(xg), xg)
            h = s.fc1(torch.cat([xp, xg], 1))
        else:
            h = s.fc1(torch.cat([xp, xg, mp[:, None], mg[:, None]], 1))
        return torch.sigmoid(s.fc2(s.drop(torch.relu(s.ln(h)))))

class LateFusion(nn.Module):
    """Per-modality encoders; masked mean over AVAILABLE modalities; shared head."""
    def __init__(s, n_p, n_g):
        super().__init__(); Hd = CFG["HIDDEN"]
        s.ep = nn.Linear(n_p, Hd); s.eg = nn.Linear(n_g, Hd); s.ln = nn.LayerNorm(Hd)
        s.drop = nn.Dropout(CFG["DROPOUT"]); s.fc2 = nn.Linear(Hd, len(CFG["TIME_BINS"]))
    def forward(s, xp, xg, mp, mg):
        ap, ag = (1 - mp)[:, None], (1 - mg)[:, None]
        z = (torch.relu(s.ep(xp)) * ap + torch.relu(s.eg(xg)) * ag) / torch.clamp(ap + ag, min=1.0)
        return torch.sigmoid(s.fc2(s.drop(torch.relu(s.ln(z)))))

def make_model(mode, F):
    n_p, n_g = len(F["sel_p"]), len(F["sel_g"])
    return (LateFusion(n_p, n_g) if mode == "latefusion" else SurvNet(n_p, n_g, mode)).to(DEV)

def surv_loss(h, t, e):
    h = torch.clamp(h, 1e-4, 1 - 1e-4); bins = CFG["TIME_BINS"]
    st = torch.tensor([0.0] + bins[:-1], device=h.device); en = torch.tensor(bins, device=h.device); tot = 0.0
    for b in range(h.shape[1]):
        ar = (t >= st[b]).float(); ev = ((t >= st[b]) & (t < en[b]) & (e == 1)).float()
        sv = (ar - ev) * ((t >= en[b]) | (e == 0)).float()
        tot = tot + (-ev * torch.log(h[:, b]) - sv * torch.log(1 - h[:, b])).sum() / (ar.sum() + 1e-8)
    return tot / h.shape[1]

def _inputs(F, idx, mode):
    i = torch.as_tensor(idx, device=DEV)
    xp = F["xp_reg"] if mode == "regimpute" else F["xp"]
    return xp[i], F["xg"][i], F["mp"][i], F["mg"][i]

def train_net(D, mode, F, tr, seed, aug=(0.0, 0.0)):
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.RandomState(seed)
    fit, es = train_test_split(tr, test_size=CFG["ES_FRAC"], random_state=seed, stratify=D["E"][tr])
    m = make_model(mode, F)
    opt = torch.optim.AdamW(m.parameters(), lr=CFG["LR"], weight_decay=CFG["WD"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["MAX_EPOCHS"])
    best, best_state, bad = np.inf, None, 0
    es_in = _inputs(F, es, mode); es_i = torch.as_tensor(es, device=DEV)
    xsrc = F["xp_reg"] if mode == "regimpute" else F["xp"]
    for _ in range(CFG["MAX_EPOCHS"]):
        m.train(); perm = rng.permutation(fit)
        for s in range(0, len(perm), CFG["BATCH"]):
            bi = torch.as_tensor(perm[s:s + CFG["BATCH"]], device=DEV)
            xp, xg = xsrc[bi].clone(), F["xg"][bi].clone(); mp, mg = F["mp"][bi].clone(), F["mg"][bi].clone()
            if aug[0] > 0:
                add = (mp < 0.5) & (torch.rand(len(bi), device=DEV) < aug[0]); mp = torch.where(add, torch.ones_like(mp), mp)
                if mode == "indicator": xp[add] = F["c_p"]
                if mode == "regimpute": xp[add] = F["pp"][bi][add]
            if aug[1] > 0:
                add = (mg < 0.5) & (torch.rand(len(bi), device=DEV) < aug[1]); mg = torch.where(add, torch.ones_like(mg), mg)
                if mode in ("indicator", "regimpute"): xg[add] = F["c_g"]
            opt.zero_grad(); loss = surv_loss(m(xp, xg, mp, mg), F["T"][bi], F["E"][bi]); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step(); m.eval()
        with torch.no_grad(): l = surv_loss(m(*es_in), F["T"][es_i], F["E"][es_i]).item()
        if l < best: best, best_state, bad = l, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= CFG["PATIENCE"]: break
    m.load_state_dict(best_state); m.eval(); return m

def predict(m, F, idx, mode):
    with torch.no_grad(): return m(*_inputs(F, idx, mode)).cpu().numpy()

def risk_at(h, j=HIDX): return 1 - np.prod(1 - h[:, :j + 1], axis=1)
def cll(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(-np.log(1 - p))
def cox_risk(cph, df, t=H): return 1 - cph.predict_survival_function(df, times=[t]).iloc[0].to_numpy()
def km_risk(t_, e_, horizon=H):
    if len(t_) < 5 or e_.sum() == 0: return np.nan
    return 1 - KaplanMeierFitter().fit(t_, e_).survival_function_at_times(horizon).iloc[0]

# ---------------------------------------------------------------- CV runner
def run_cv(D, arms, baselines=True, nested=True):
    R, N, T, E = CFG["N_REPEATS"], D["N"], D["T"], D["E"]
    HAZ = {a: np.full((R, N, len(CFG["TIME_BINS"])), np.nan) for a in arms}
    BASE = {b: np.full((R, N), np.nan) for b in ["elastic_net_cox", "rsf", "gbm_survival"]}
    NEST = {k: np.full((R, N), np.nan) for k in ["recal", "combo", "clinical", "stage_only", "null"]}
    PARAMS = {}
    for r in range(R):
        skf = StratifiedKFold(CFG["N_OUTER"], shuffle=True, random_state=r)
        for k, (tr, te) in enumerate(skf.split(np.zeros(N), E)):
            F = prep_fold(D, tr)
            for a in arms:
                m = train_net(D, ARM_MODE[a], F, tr, seed=1000 * r + 10 * k)
                HAZ[a][r, te] = predict(m, F, te, ARM_MODE[a])
                PARAMS.setdefault(a, sum(p.numel() for p in m.parameters()))
            NEST["null"][r, te] = km_risk(T[tr], E[tr])
            if baselines:
                Xb = np.hstack([F["Xp"], F["Xg"], (~D["has_rppa"])[:, None], (~D["has_mut"])[:, None]]).astype(float)
                # Elastic-net Cox: drop columns constant on the training fold (e.g. the mutation-missing
                # indicator in LUAD, where no patient lacks mutation profiling). Trees use all columns.
                Xe = Xb[:, Xb[tr].std(0) > 1e-8]
                cols = [f"f{i}" for i in range(Xe.shape[1])]
                try:
                    en = CoxPHFitter(penalizer=0.1, l1_ratio=0.5).fit(pd.DataFrame(Xe[tr], columns=cols).assign(T=T[tr], E=E[tr]), "T", "E")
                    BASE["elastic_net_cox"][r, te] = en.predict_log_partial_hazard(pd.DataFrame(Xe[te], columns=cols)).to_numpy()
                except Exception as ex:
                    print("EN-Cox failed:", ex)
                if HAS_SKSURV:
                    ytr = Surv.from_arrays(E[tr].astype(bool), T[tr])
                    BASE["rsf"][r, te] = RandomSurvivalForest(n_estimators=200, min_samples_leaf=10, max_depth=4,
                                                              random_state=r, n_jobs=-1).fit(Xb[tr], ytr).predict(Xb[te])
                    BASE["gbm_survival"][r, te] = GradientBoostingSurvivalAnalysis(
                        n_estimators=100, learning_rate=0.05, max_depth=2, random_state=r).fit(Xb[tr], ytr).predict(Xb[te])
            if nested and "token" in arms:
                inner = np.full(N, np.nan)
                iskf = StratifiedKFold(CFG["N_INNER"], shuffle=True, random_state=100 + r)
                for j, (a_, b_) in enumerate(iskf.split(np.zeros(len(tr)), E[tr])):
                    itr, ite = tr[a_], tr[b_]; Fi = prep_fold(D, itr)
                    inner[ite] = risk_at(predict(train_net(D, "token", Fi, itr, seed=5000 + 100 * r + 10 * k + j), Fi, ite, "token"))
                raw_te = risk_at(HAZ["token"][r, te])
                rc = CoxPHFitter().fit(pd.DataFrame({"cll": cll(inner[tr]), "T": T[tr], "E": E[tr]}), "T", "E")
                NEST["recal"][r, te] = cox_risk(rc, pd.DataFrame({"cll": cll(raw_te)}))
                vc = D["valid_clin"]; trc, tec = tr[vc[tr]], te[vc[te]]
                raw_full = np.full(N, np.nan); raw_full[te] = raw_te
                Dtr = D["CLIN"].iloc[trc].assign(cll_model=cll(inner[trc]), T=T[trc], E=E[trc])
                Dte = D["CLIN"].iloc[tec].assign(cll_model=cll(raw_full[tec]))
                for key, cols in [("combo", D["CLIN_COLS"] + ["cll_model"]), ("clinical", D["CLIN_COLS"]), ("stage_only", D["STAGE_COLS"])]:
                    if not cols: continue
                    cph = CoxPHFitter(penalizer=0.01).fit(Dtr[cols + ["T", "E"]], "T", "E")
                    NEST[key][r, tec] = cox_risk(cph, Dte[cols])
            print(f"  [{D['label']}] repeat {r+1}/{R} fold {k+1}/{CFG['N_OUTER']} done")
    return HAZ, BASE, NEST, PARAMS

# ---------------------------------------------------------------- analysis
def analyze(D, HAZ, BASE, NEST):
    T, E, N = D["T"], D["E"], D["N"]; R = next(iter(HAZ.values())).shape[0]
    rng = np.random.RandomState(42); BOOT = [rng.randint(0, N, N) for _ in range(CFG["N_BOOT"])]
    SC = {a: np.stack([risk_at(HAZ[a][r]) for r in range(R)]) for a in HAZ}
    SC.update({b: v for b, v in BASE.items() if not np.isnan(v).all()})
    ENS = {m: np.mean([rankdata(s[r]) / N for r in range(R)], 0) for m, s in SC.items()}
    cidx = lambda s, idx=None: concordance_index(T[np.arange(N) if idx is None else idx], -s[np.arange(N) if idx is None else idx], E[np.arange(N) if idx is None else idx])
    out = {"summary": D["summary"], "discrimination": [], "paired": [], "subgroup_rppa_missing": [], "absolute_risk": []}
    for m, s in SC.items():
        per = [cidx(s[r]) for r in range(R)]
        bt = [concordance_index(T[i], -ENS[m][i], E[i]) for i in BOOT if E[i].sum() > 0]
        out["discrimination"].append({"method": m, "C_mean": float(np.mean(per)), "C_sd": float(np.std(per)),
                                      "C_ensemble": float(cidx(ENS[m])), "CI": [float(x) for x in np.percentile(bt, [2.5, 97.5])]})

    def paired(a, b, idx=None, seed=42):
        base = np.arange(N) if idx is None else idx
        rr = np.random.RandomState(seed); d = []
        for _ in range(CFG["N_BOOT"]):
            i = base[rr.randint(0, len(base), len(base))]
            if E[i].sum() == 0: continue
            d.append(concordance_index(T[i], -a[i], E[i]) - concordance_index(T[i], -b[i], E[i]))
        d = np.array(d)
        return {"diff": float(cidx(a, base) - cidx(b, base)), "CI": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "p": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}

    pairs = [("token", "indicator"), ("regimpute", "indicator"), ("latefusion", "indicator"),
             ("token", "latefusion"), ("token", "regimpute"),
             ("token", "elastic_net_cox"), ("token", "rsf"), ("token", "gbm_survival")]
    for a, b in pairs:
        if a in ENS and b in ENS: out["paired"].append({"A": a, "B": b, **paired(ENS[a], ENS[b])})
    miss = np.where(~D["has_rppa"])[0]
    if E[miss].sum() >= 5:
        for a in ["token", "regimpute", "latefusion"]:
            if a in ENS:
                out["subgroup_rppa_missing"].append({"A": a, "B": "indicator", "n": int(len(miss)), "deaths": int(E[miss].sum()),
                                                     "C_A": float(cidx(ENS[a], miss)), "C_B": float(cidx(ENS["indicator"], miss)),
                                                     **paired(ENS[a], ENS["indicator"], miss, seed=0)})
    if not np.isnan(NEST["recal"]).all():
        kmc = KaplanMeierFitter().fit(T, 1 - E)
        G = lambda t: kmc.survival_function_at_times(np.atleast_1d(t)).to_numpy()
        ce, cs = (T <= H) & (E == 1), T > H
        W = np.zeros(N); W[ce] = 1 / np.clip(G(np.maximum(T[ce] - 1e-6, 0)), 1e-8, None); W[cs] = 1 / G(H)[0]
        TG = ce.astype(float)
        brier = lambda rk, idx: np.sum(W[idx] * (TG[idx] - rk[idx]) ** 2) / len(idx)
        def block(name, rk, null, idx):
            rr = np.random.RandomState(7); bb = []
            for _ in range(CFG["N_BOOT"]):
                i = idx[rr.randint(0, len(idx), len(idx))]; m_, n_ = brier(rk, i), brier(null, i); bb.append((m_, m_ - n_, 1 - m_ / n_))
            bb = np.array(bb); ci = lambda c: [float(np.percentile(bb[:, c], 2.5)), float(np.percentile(bb[:, c], 97.5))]
            cc = CoxPHFitter().fit(pd.DataFrame({"x": cll(rk[idx]), "T": T[idx], "E": E[idx]}), "T", "E").summary.iloc[0]
            bs, bn = brier(rk, idx), brier(null, idx)
            return {"model": name, "n": int(len(idx)), "deaths": int(E[idx].sum()), "C": float(cidx(rk, idx)),
                    "brier": float(bs), "brier_CI": ci(0), "brier_constant": float(bn), "brier_minus_constant": float(bs - bn),
                    "brier_minus_constant_CI": ci(1), "IPA": float(1 - bs / bn), "IPA_CI": ci(2),
                    "O_over_E": float(km_risk(T[idx], E[idx]) / rk[idx].mean()), "slope": float(cc["coef"]),
                    "slope_CI": [float(cc["coef lower 95%"]), float(cc["coef upper 95%"])],
                    "p_slope_vs_1": float(2 * (1 - norm.cdf(abs((cc["coef"] - 1) / cc["se(coef)"]))))}
        raw5 = np.mean([risk_at(HAZ["token"][r]) for r in range(R)], 0); null5 = np.nanmean(NEST["null"], 0)
        ALL, CI_ = np.arange(N), np.where(D["valid_clin"])[0]
        out["absolute_risk"].append(block("token_raw", raw5, null5, ALL))
        out["absolute_risk"].append(block("token_recal_nested", np.nanmean(NEST["recal"], 0), null5, ALL))
        for key in ["combo", "clinical", "stage_only"]:
            if not np.isnan(NEST[key]).all():
                out["absolute_risk"].append(block(f"{key}_nested", np.nanmean(NEST[key], 0), null5, CI_))
        if not np.isnan(NEST["combo"]).all():
            cmb, cln = np.nanmean(NEST["combo"], 0), np.nanmean(NEST["clinical"], 0)
            out["combo_vs_clinical"] = paired(np.nan_to_num(cmb), np.nan_to_num(cln), CI_, seed=1)
    return out

# ---------------------------------------------------------------- main loop (resumable)
ALL = {}
for c in COHORTS:
    tag = c["label"]; npz = os.path.join(CFG["EXT_DIR"], f"arrays_{tag}.npz")
    D = load_cohort(c)
    if D is None: continue
    if os.path.exists(npz):
        print(f"[{tag}] found saved arrays -> skipping training"); z = np.load(npz, allow_pickle=True)
        HAZ = {k[4:]: z[k] for k in z.files if k.startswith("haz_")}
        BASE = {k[5:]: z[k] for k in z.files if k.startswith("base_")}
        NEST = {k[5:]: z[k] for k in z.files if k.startswith("nest_")}
        PARAMS = json.loads(str(z["params"])) if "params" in z.files else {}
    else:
        t0 = time.time()
        if c["study"] == BRCA:
            old = np.load(os.path.join(CFG["RESULTS_DIR"], "oof_arrays.npz"))
            assert old["haz_token"].shape[1] == D["N"], "BRCA cohort size changed vs saved arrays — stop."
            HAZn, _, _, PARAMS = run_cv(D, NEW_ARMS_ONLY_BRCA, baselines=False, nested=False)
            HAZ = {k[4:]: old[k] for k in old.files if k.startswith("haz_")}; HAZ.update(HAZn)
            BASE = {k[5:]: old[k] for k in old.files if k.startswith("base_")}
            NEST = {k[5:]: old[k] for k in old.files if k.startswith("nest_")}
        else:
            HAZ, BASE, NEST, PARAMS = run_cv(D, ARMS_NEW_COHORT, baselines=True, nested=True)
        np.savez(npz, **{f"haz_{k}": v for k, v in HAZ.items()}, **{f"base_{k}": v for k, v in BASE.items()},
                 **{f"nest_{k}": v for k, v in NEST.items()}, params=json.dumps(PARAMS))
        print(f"[{tag}] finished in {(time.time()-t0)/60:.1f} min")
    res = analyze(D, HAZ, BASE, NEST); res["params"] = PARAMS
    res["data_ids"] = {"study": D["study"], "rppa_profile": D["rppa_profile"], "mut_list": D["mut_list"], "rppa_list": D["rppa_list"]}
    ALL[tag] = res
    with open(os.path.join(CFG["EXT_DIR"], "results_extension.json"), "w") as f:
        json.dump(ALL, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

# ---------------------------------------------------------------- forest plot
rows = []
for tag, res in ALL.items():
    for p in res["paired"]:
        if p["B"] == "indicator": rows.append((f"{tag}: {p['A']} - indicator", p["diff"], *p["CI"]))
    if "combo_vs_clinical" in res:
        p = res["combo_vs_clinical"]; rows.append((f"{tag}: clinical+omics - clinical", p["diff"], *p["CI"]))
if rows:
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    ax.errorbar([r[1] for r in rows], y, xerr=[[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]],
                fmt="o", color="#2b5c8f", capsize=3)
    ax.axvline(0, color="gray", ls=":"); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("Paired C-index difference (95% bootstrap CI)"); ax.grid(axis="x", ls="--", alpha=.4)
    fig.tight_layout(); fig.savefig(os.path.join(CFG["EXT_DIR"], "cross_cohort_forest.png"), dpi=300, bbox_inches="tight"); plt.show()

# ---------------------------------------------------------------- paste-ready verified block
f4 = lambda x: f"{x:.4f}"
L = ["## EXTENSION SUMMARY (all three cohorts)"]
for tag, res in ALL.items():
    s = res["summary"]
    L.append(f"- **{tag} cohort**: n = {s['n']}, deaths = {s['deaths']}; RPPA missing {s['rppa_missing']} ({s['rppa_missing_pct']:.1f}%, "
             f"{s['rppa_missing_deaths']} deaths); mutation missing {s['mut_missing']} ({s['mut_missing_pct']:.1f}%); "
             f"{s['n_proteins']} proteins; panel {s['panel_genes']} genes; clinical cohort n = {s['clinical_n']}, "
             f"deaths = {s['clinical_deaths']} ({', '.join(s['clinical_cols'])}); censoring survival at 5y {s['censoring_survival_at_horizon']:.3f}; "
             f"zero follow-up {s['zero_followup']}; retrieved {s['retrieved_utc']}; params {res['params']}")
    for d in res["discrimination"]:
        L.append(f"  - {tag} {d['method']}: mean C {f4(d['C_mean'])} (SD {f4(d['C_sd'])}); ensemble {f4(d['C_ensemble'])} ({f4(d['CI'][0])}, {f4(d['CI'][1])})")
    for p in res["paired"]:
        L.append(f"  - {tag} {p['A']} - {p['B']}: {p['diff']:+.4f} [{p['CI'][0]:+.4f}, {p['CI'][1]:+.4f}], p = {p['p']:.3f}")
    for p in res["subgroup_rppa_missing"]:
        L.append(f"  - {tag} RPPA-missing subgroup (n = {p['n']}, deaths = {p['deaths']}) {p['A']} vs indicator: "
                 f"C {f4(p['C_A'])} vs {f4(p['C_B'])}; diff {p['diff']:+.4f} [{p['CI'][0]:+.4f}, {p['CI'][1]:+.4f}], p = {p['p']:.3f}")
    for a in res["absolute_risk"]:
        L.append(f"  - {tag} {a['model']}: n {a['n']}, deaths {a['deaths']}, C {f4(a['C'])}, Brier {f4(a['brier'])} "
                 f"[{f4(a['brier_CI'][0])}, {f4(a['brier_CI'][1])}] vs constant {f4(a['brier_constant'])} "
                 f"(diff {a['brier_minus_constant']:+.4f} [{a['brier_minus_constant_CI'][0]:+.4f}, {a['brier_minus_constant_CI'][1]:+.4f}]); "
                 f"IPA {f4(a['IPA'])} [{f4(a['IPA_CI'][0])}, {f4(a['IPA_CI'][1])}]; O/E {a['O_over_E']:.2f}; "
                 f"slope {a['slope']:.2f} [{a['slope_CI'][0]:.2f}, {a['slope_CI'][1]:.2f}], p vs 1 = {a['p_slope_vs_1']:.3g}")
    if "combo_vs_clinical" in res:
        p = res["combo_vs_clinical"]
        L.append(f"  - {tag} clinical+omics - clinical C: {p['diff']:+.4f} [{p['CI'][0]:+.4f}, {p['CI'][1]:+.4f}], p = {p['p']:.3f}")
block = "\n".join(L)
with open(os.path.join(CFG["EXT_DIR"], "summary_extension.md"), "w") as f: f.write(block)
print("\n" + "=" * 80 + "\n" + block + "\n" + "=" * 80)
print("Saved in", CFG["EXT_DIR"])


# ---------------------------------------------------------------- KIRC: clinical comparator with histologic grade
# Grade (G3, G4 vs G1-G2) is added to age + stage; patients with GX/missing grade are excluded.
# Only token and indicator arms are retrained (nested second stage needs the token arm). ~10-30 min.
if CFG["RUN_KIRC_GRADE"]:
    cK = next(c for c in COHORTS if c["label"] == "KIRC")
    DK = load_cohort(cK)
    sdirK = os.path.join(CFG["SNAPSHOT_DIR"], cK["study"])
    pat = pd.DataFrame(json.load(open(os.path.join(sdirK, "clinical_patient.json"))))
    clinK = pat.pivot_table(index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first")
    gcols = [c for c in clinK.columns if "GRADE" in c.upper()]
    print("[KIRC] patient-level grade attributes:", gcols)
    grade_by_patient = None
    if gcols:
        grade_by_patient = clinK[gcols[0]]
    else:                                                    # fall back to sample-level attributes (snapshotted)
        p_s = os.path.join(sdirK, "clinical_sample.json")
        if not os.path.exists(p_s):
            out_, page = [], 0
            while True:
                b = _get(f"/studies/{cK['study']}/clinical-data", {"clinicalDataType": "SAMPLE", "pageSize": 10000,
                                                                   "pageNumber": page, "projection": "SUMMARY"})
                out_ += b
                if len(b) < 10000: break
                page += 1
            json.dump(out_, open(p_s, "w"))
        smpc = pd.DataFrame(json.load(open(p_s)))
        smpc = smpc[smpc["clinicalAttributeId"].str.upper().str.contains("GRADE")]
        print("[KIRC] sample-level grade attributes:", sorted(smpc["clinicalAttributeId"].unique().tolist()))
        if len(smpc):
            aid = sorted(smpc["clinicalAttributeId"].unique())[0]
            grade_by_patient = smpc[smpc["clinicalAttributeId"] == aid].groupby("patientId")["value"].first()

    if grade_by_patient is None:
        print("[KIRC] no grade attribute found -> grade analysis skipped.")
    else:
        g = grade_by_patient.reindex(DK["smp"]["patientId"]).astype(str).str.upper().str.strip().to_numpy()
        print("[KIRC] grade value counts:", pd.Series(g).value_counts(dropna=False).to_dict())
        gvalid = np.isin(g, ["G1", "G2", "G3", "G4"])
        if gvalid.mean() < 0.80:
            print(f"[KIRC] grade available for only {gvalid.mean():.1%} -> grade analysis skipped.")
        else:
            D2 = copy.copy(DK)                                  # shallow copy; only clinical fields replaced
            CL2 = DK["CLIN"].copy()
            CL2["gr_G3"] = (g == "G3").astype(float); CL2["gr_G4"] = (g == "G4").astype(float)   # ref: G1 + G2
            D2["CLIN"] = CL2
            D2["valid_clin"] = DK["valid_clin"] & gvalid
            D2["CLIN_COLS"] = DK["CLIN_COLS"] + ["gr_G3", "gr_G4"]
            D2["STAGE_COLS"] = DK["STAGE_COLS"]
            s2 = dict(DK["summary"]); s2["clinical_n"] = int(D2["valid_clin"].sum())
            s2["clinical_deaths"] = int(DK["E"][D2["valid_clin"]].sum()); s2["clinical_cols"] = D2["CLIN_COLS"]
            D2["summary"] = s2
            npzG = os.path.join(CFG["EXT_DIR"], "arrays_KIRC_grade.npz")
            if os.path.exists(npzG):
                z = np.load(npzG, allow_pickle=True)
                HAZ2 = {k[4:]: z[k] for k in z.files if k.startswith("haz_")}
                NEST2 = {k[5:]: z[k] for k in z.files if k.startswith("nest_")}
            else:
                t0 = time.time()
                HAZ2, _, NEST2, _ = run_cv(D2, ["token", "indicator"], baselines=False, nested=True)
                np.savez(npzG, **{f"haz_{k}": v for k, v in HAZ2.items()}, **{f"nest_{k}": v for k, v in NEST2.items()})
                print(f"[KIRC grade] finished in {(time.time()-t0)/60:.1f} min")
            resG = analyze(D2, HAZ2, {}, NEST2)
            json.dump(resG, open(os.path.join(CFG["EXT_DIR"], "results_KIRC_grade.json"), "w"), indent=2,
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
            f4 = lambda x: f"{x:.4f}"
            L = ["## KIRC WITH GRADE IN THE CLINICAL COMPARATOR",
                 f"- KIRC clinical cohort with grade: n = {s2['clinical_n']}, deaths = {s2['clinical_deaths']} ({', '.join(s2['clinical_cols'])}); grade ref = G1+G2"]
            for a in resG["absolute_risk"]:
                if a["model"] in ("combo_nested", "clinical_nested", "stage_only_nested"):
                    L.append(f"  - KIRC+grade {a['model']}: n {a['n']}, deaths {a['deaths']}, C {f4(a['C'])}, Brier {f4(a['brier'])} "
                             f"[{f4(a['brier_CI'][0])}, {f4(a['brier_CI'][1])}] vs constant {f4(a['brier_constant'])}; "
                             f"IPA {f4(a['IPA'])} [{f4(a['IPA_CI'][0])}, {f4(a['IPA_CI'][1])}]; O/E {a['O_over_E']:.2f}; "
                             f"slope {a['slope']:.2f} [{a['slope_CI'][0]:.2f}, {a['slope_CI'][1]:.2f}], p vs 1 = {a['p_slope_vs_1']:.3g}")
            p = resG.get("combo_vs_clinical")
            if p:
                L.append(f"  - KIRC+grade clinical+omics - clinical C: {p['diff']:+.4f} [{p['CI'][0]:+.4f}, {p['CI'][1]:+.4f}], p = {p['p']:.3f}")
            blk = "\n".join(L)
            open(os.path.join(CFG["EXT_DIR"], "summary_kirc_grade.md"), "w").write(blk)
            print("\n" + "=" * 80 + "\n" + blk + "\n" + "=" * 80)
