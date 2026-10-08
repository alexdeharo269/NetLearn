#!/usr/bin/env python3
"""
Analyses added in v3 of the manuscript.

  1. Parcellation of each cohort. The released networks of HCPd, HCPya, camCAN and HCPa follow the
     AAL3 region order (labels 1-94 without the empty labels 35, 36, 81, 82); the other cohorts follow
     AAL90. Region-level quantities below use the labels of each cohort.
  2. Where the stable rank is set: exact factorisation 0.9801 r_s = N G D L (global, degree, local),
     dominance of the strongest edge, and the share of squared weight on homotopic left-right edges.
  3. New null models (closed form, same 100 subjects as paper/data/surrogates.csv):
     equal strength (symmetric Sinkhorn balancing) and halved interhemispheric weights.
  4. Lifespan: within-cohort aging explained by interhemispheric weight; virtual rescaling of the
     interhemispheric edges of young camCAN/HCPa connectomes; sensitivity to the 82 shared regions
     and to removing the strongest edge.
  5. CALM: children referred for problems of attention, learning or memory vs comparison children
     (same site and pipeline), from all_data.mat of the original release.

Usage (repo root):
  python paper/analysis_v3.py --data <export folder> --allmat <all_data.mat> --demo <demographics.csv>
Writes paper/data/v3/.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "paper")); sys.path.insert(0, str(REPO / "chain"))
import paper_analysis as pa      # noqa: E402  closed form (input Gramian, memory curves)
import chain_analysis as ca      # noqa: E402  loaders

AAL90 = """Precentral_L Precentral_R Frontal_Sup_L Frontal_Sup_R Frontal_Sup_Orb_L Frontal_Sup_Orb_R Frontal_Mid_L
Frontal_Mid_R Frontal_Mid_Orb_L Frontal_Mid_Orb_R Frontal_Inf_Oper_L Frontal_Inf_Oper_R Frontal_Inf_Tri_L Frontal_Inf_Tri_R
Frontal_Inf_Orb_L Frontal_Inf_Orb_R Rolandic_Oper_L Rolandic_Oper_R Supp_Motor_Area_L Supp_Motor_Area_R Olfactory_L
Olfactory_R Frontal_Sup_Medial_L Frontal_Sup_Medial_R Frontal_Med_Orb_L Frontal_Med_Orb_R Rectus_L Rectus_R Insula_L
Insula_R Cingulum_Ant_L Cingulum_Ant_R Cingulum_Mid_L Cingulum_Mid_R Cingulum_Post_L Cingulum_Post_R Hippocampus_L
Hippocampus_R ParaHippocampal_L ParaHippocampal_R Amygdala_L Amygdala_R Calcarine_L Calcarine_R Cuneus_L Cuneus_R
Lingual_L Lingual_R Occipital_Sup_L Occipital_Sup_R Occipital_Mid_L Occipital_Mid_R Occipital_Inf_L Occipital_Inf_R
Fusiform_L Fusiform_R Postcentral_L Postcentral_R Parietal_Sup_L Parietal_Sup_R Parietal_Inf_L Parietal_Inf_R
SupraMarginal_L SupraMarginal_R Angular_L Angular_R Precuneus_L Precuneus_R Paracentral_Lobule_L Paracentral_Lobule_R
Caudate_L Caudate_R Putamen_L Putamen_R Pallidum_L Pallidum_R Thalamus_L Thalamus_R Heschl_L Heschl_R Temporal_Sup_L
Temporal_Sup_R Temporal_Pole_Sup_L Temporal_Pole_Sup_R Temporal_Mid_L Temporal_Mid_R Temporal_Pole_Mid_L
Temporal_Pole_Mid_R Temporal_Inf_L Temporal_Inf_R""".split()
AAL3_94 = """Precentral_L Precentral_R Frontal_Sup_2_L Frontal_Sup_2_R Frontal_Mid_2_L Frontal_Mid_2_R Frontal_Inf_Oper_L
Frontal_Inf_Oper_R Frontal_Inf_Tri_L Frontal_Inf_Tri_R Frontal_Inf_Orb_2_L Frontal_Inf_Orb_2_R Rolandic_Oper_L Rolandic_Oper_R
Supp_Motor_Area_L Supp_Motor_Area_R Olfactory_L Olfactory_R Frontal_Sup_Medial_L Frontal_Sup_Medial_R Frontal_Med_Orb_L
Frontal_Med_Orb_R Rectus_L Rectus_R OFCmed_L OFCmed_R OFCant_L OFCant_R OFCpost_L OFCpost_R OFClat_L OFClat_R Insula_L
Insula_R EMPTY35 EMPTY36 Cingulate_Mid_L Cingulate_Mid_R Cingulate_Post_L Cingulate_Post_R Hippocampus_L
Hippocampus_R ParaHippocampal_L ParaHippocampal_R Amygdala_L Amygdala_R Calcarine_L Calcarine_R Cuneus_L Cuneus_R Lingual_L
Lingual_R Occipital_Sup_L Occipital_Sup_R Occipital_Mid_L Occipital_Mid_R Occipital_Inf_L Occipital_Inf_R Fusiform_L
Fusiform_R Postcentral_L Postcentral_R Parietal_Sup_L Parietal_Sup_R Parietal_Inf_L Parietal_Inf_R SupraMarginal_L
SupraMarginal_R Angular_L Angular_R Precuneus_L Precuneus_R Paracentral_Lobule_L Paracentral_Lobule_R Caudate_L Caudate_R
Putamen_L Putamen_R Pallidum_L Pallidum_R EMPTY81 EMPTY82 Heschl_L Heschl_R Temporal_Sup_L Temporal_Sup_R
Temporal_Pole_Sup_L Temporal_Pole_Sup_R Temporal_Mid_L Temporal_Mid_R Temporal_Pole_Mid_L Temporal_Pole_Mid_R
Temporal_Inf_L Temporal_Inf_R""".split()
AAL3 = [n for n in AAL3_94 if not n.startswith("EMPTY")]
assert len(AAL90) == 90 and len(AAL3) == 90
AAL3_COHORTS = {6, 7, 8, 9}                      # HCPd, HCPya, camCAN, HCPa
CANON = lambda n: n.replace("_2_", "_").replace("Cingulate", "Cingulum")
SHARED = [n for n in AAL90 if n in [CANON(m) for m in AAL3]]           # 82 regions
MEDIAL = ("Supp_Motor_Area", "Paracentral_Lobule", "Frontal_Sup_Medial", "Precuneus", "Cuneus", "Calcarine",
          "Lingual", "Cingulum", "Cingulate", "Frontal_Med_Orb", "Rectus", "Olfactory", "OFCmed")
DS = pa.DS_NAMES
OUT = REPO / "paper" / "data" / "v3"


def labels(ds):
    return AAL3 if int(ds) in AAL3_COHORTS else AAL90


def edge_sets(ds):
    """Index arrays of homotopic (medial / lateral) and all interhemispheric pairs for a cohort's atlas."""
    L = labels(ds); idx = {n: i for i, n in enumerate(L)}
    hom_med, hom_lat = [], []
    for n, i in idx.items():
        if n.endswith("_L") and n[:-2] + "_R" in idx:
            (hom_med if n.startswith(MEDIAL) else hom_lat).append((i, idx[n[:-2] + "_R"]))
    left = np.array([n.endswith("_L") for n in L])
    return np.array(hom_med), np.array(hom_lat), np.logical_xor.outer(left, left)


EDGES = {a: edge_sets(6 if a == 3 else 1) for a in (1, 3)}   # 1: AAL90, 3: AAL3


def sets_of(ds):
    return EDGES[3 if int(ds) in AAL3_COHORTS else 1]


def mc_closed(A, seed, eta=pa.ETA0):
    """Closed-form memory capacity (SI Prop. S1) with the numpy input vector of seed `seed`."""
    rho = np.abs(np.linalg.eigvalsh(A)).max(); Wn = A * (pa.RHO / rho)
    w = np.random.default_rng(int(seed)).uniform(-1.0, 1.0, A.shape[0])
    C = pa.input_gramian(Wn, w); c, U = np.linalg.eigh(C); c, U = np.clip(c[::-1], 0, None), U[:, ::-1]
    return float(pa.theory_curves(Wn, w, c, U, [eta])[0].sum())


def structure(A, ds):
    """Per-connectome quantities: stable rank, its exact factors, dominance and interhemispheric shares."""
    lam, V = np.linalg.eigh(A); i1 = int(np.argmax(np.abs(lam))); rho = abs(lam[i1]); v1 = V[:, i1]
    s = A.sum(1); k = (A > 0).sum(1).astype(float); ok = s > 0
    Y = np.zeros_like(s); Y[ok] = (A[ok] ** 2).sum(1) / s[ok] ** 2
    st2 = (pa.RHO * s / rho) ** 2
    d = np.where(k > 0, 2.0 / (k + 1.0), 0.0)                    # null expectation of Y_i
    R = np.where(d > 0, Y / np.where(d > 0, d, 1), 0.0)            # disparity relative to the null
    G = st2.mean(); D = (st2 * d).sum() / st2.sum(); Lf = (st2 * d * R).sum() / (st2 * d).sum()
    F2 = (A ** 2).sum(); iu = np.triu_indices(A.shape[0], 1); wv = A[iu]; e = int(np.argmax(wv))
    med, lat, inter = sets_of(ds); tot = A.sum() / 2
    hm, hl = A[med[:, 0], med[:, 1]], A[lat[:, 0], lat[:, 1]]
    l2 = (pa.RHO * lam / rho) ** 2
    L = labels(ds)
    return dict(rs=F2 / rho ** 2, G=G, D=D, L=Lf, top_F2=2 * wv[e] ** 2 / F2,
                top_edge=f"{CANON(L[iu[0][e]])}--{CANON(L[iu[1][e]])}",
                hom_F2=2 * ((hm ** 2).sum() + (hl ** 2).sum()) / F2,
                hom_med=hm.sum() / tot, hom_lat=hl.sum() / tot, inter_frac=A[inter].sum() / 2 / tot,
                v1_top2=float(np.sort(v1 ** 2)[-2:].sum()), S_inf=float((l2 / (1 - l2)).mean()),
                Ratio_w=float(np.average(R[k > 0], weights=st2[k > 0])))


def balance(A, tol=1e-10, iters=200000):
    """Symmetric Sinkhorn-Knopp: D A D with all strengths equal to the mean strength of A."""
    on = A.sum(1) > 0; B = A[np.ix_(on, on)]; target = B.sum() / B.shape[0]; x = np.ones(B.shape[0])
    for _ in range(iters):
        xn = np.sqrt(x * target / (B @ x))
        if np.max(np.abs(xn - x)) < tol * xn.max():
            x = xn; break
        x = xn
    out = np.zeros_like(A); out[np.ix_(on, on)] = B * np.outer(x, x)
    return out if np.all(np.isfinite(out)) and np.allclose(out.sum(1)[on], target, rtol=1e-4) else None


def r_(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def pcor(d, x, y, z):
    Z = np.c_[np.ones(len(d)), d[z].values]
    res = lambda v: d[v].values - Z @ np.linalg.lstsq(Z, d[v].values, rcond=None)[0]
    return r_(res(x), res(y))


def shares(d):
    v = d["log_rs"].var()
    return {f: float(np.cov(d["log_rs"], d["log_" + f])[0, 1] / v) for f in ("G", "D", "L")}


def ols_group(d, y, group_col):
    """y ~ 1 + group + age + sex; returns group coefficient, its t and p, and Cohen's d of residuals."""
    X = np.c_[np.ones(len(d)), d[group_col].values, d["age"].values, d["sex"].values]
    b, *_ = np.linalg.lstsq(X, d[y].values, rcond=None)
    res = d[y].values - X @ b; dof = len(d) - X.shape[1]
    cov = (res @ res / dof) * np.linalg.inv(X.T @ X); t = b[1] / np.sqrt(cov[1, 1])
    p = 2 * stats.t.sf(abs(t), dof)
    rz = d[y].values - np.c_[np.ones(len(d)), d["age"].values, d["sex"].values] @ np.linalg.lstsq(
        np.c_[np.ones(len(d)), d["age"].values, d["sex"].values], d[y].values, rcond=None)[0]
    g = d[group_col].values.astype(bool)
    dd = (rz[g].mean() - rz[~g].mean()) / np.sqrt(((g.sum() - 1) * rz[g].var(ddof=1) + ((~g).sum() - 1) * rz[~g].var(ddof=1)) / (len(g) - 2))
    return dict(coef=float(b[1]), t=float(t), p=float(p), d=float(dd),
                mwu_p=float(stats.mannwhitneyu(rz[g], rz[~g]).pvalue))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="export folder of the 3,901 connectomes (chain/export_data.py)")
    ap.add_argument("--allmat", required=True, help="all_data.mat of the original release (4,216 networks)")
    ap.add_argument("--demo", required=True, help="demographics.csv of the original release")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sid, mats, meta = ca.load_export(args.data)
    sub = pd.read_csv(REPO / "paper/data/subject_v2.csv")
    sub = sub.set_index("sid").loc[sid].reset_index()
    ds = sub["dataset"].values

    # ---- 1-2. per-connectome structure, 82-region and top-edge sensitivity
    I90 = np.array([AAL90.index(n) for n in SHARED]); I3 = np.array([[CANON(m) for m in AAL3].index(n) for n in SHARED])
    rows = []
    cached = OUT / "subjects_v3.csv"
    for n, (s_, A) in enumerate(zip(sid, mats) if not cached.exists() else []):
        q = structure(A, ds[n]); q["sid"] = int(s_)
        ii = I3 if ds[n] in AAL3_COHORTS else I90; B = A[np.ix_(ii, ii)]
        q["MC82"] = mc_closed(B, s_); q["rs82"] = (B ** 2).sum() / np.abs(np.linalg.eigvalsh(B)).max() ** 2
        iu = np.triu_indices(90, 1); e = int(np.argmax(A[iu])); T = A.copy(); T[iu[0][e], iu[1][e]] = T[iu[1][e], iu[0][e]] = 0
        q["MC_notop"] = mc_closed(T, s_); q["rs_notop"] = (T ** 2).sum() / np.abs(np.linalg.eigvalsh(T)).max() ** 2
        rows.append(q)
        if (n + 1) % 1000 == 0: print(f"  structure {n + 1}/{len(sid)}", flush=True)
    st = pd.DataFrame(rows) if rows else pd.read_csv(cached)[["sid"] + [c for c in pd.read_csv(cached, nrows=1).columns
                                                                    if c not in sub.columns and c not in ("MC", "MCth", "cohort", "atlas")
                                                                    and not c.startswith("log_")]]
    d = sub[["sid", "age", "sex", "dataset", pa.MC, "MC_theory_py", "stable_rank", "Ratio", "s2t", "gram_Tinf"]].merge(st, on="sid")
    d = d.rename(columns={pa.MC: "MC", "MC_theory_py": "MCth"})
    d["cohort"] = d["dataset"].map(DS); d["atlas"] = np.where(d["dataset"].isin(AAL3_COHORTS), "AAL3", "AAL90")
    for f in ("rs", "G", "D", "L"):
        d["log_" + f] = np.log(d[f])
    d.to_csv(OUT / "subjects_v3.csv", index=False)

    N = {"n": len(d)}
    dev, agi = d[d.age <= pa.AGE_SPLIT], d[d.age > pa.AGE_SPLIT]
    N["factor_shares"] = {"all": shares(d), "development": shares(dev), "aging": shares(agi),
                          **{c: shares(g) for c, g in d.groupby("cohort")}}
    N["disparity"] = dict(r_MC_ratio=r_(d.MC, d.Ratio), pr_MC_ratio_given_G=pcor(d, "MC", "Ratio", ["log_G"]),
                          r_MC_logL=r_(d.MC, d.log_L), pr_MC_logL_given_GD=pcor(d, "MC", "log_L", ["log_G", "log_D"]),
                          r_logG_logL=r_(d.log_G, d.log_L), r_MC_logG=r_(d.MC, d.log_G))
    X = np.c_[np.ones(len(d)), d.hom_F2]; b = np.linalg.lstsq(X, d.MC, rcond=None)[0]
    N["interhemispheric"] = dict(
        r_MC_homF2=r_(d.MC, d.hom_F2), R2_MC_homF2=float(1 - ((d.MC - X @ b) ** 2).sum() / ((d.MC - d.MC.mean()) ** 2).sum()),
        r_MC_topF2=r_(d.MC, d.top_F2), r_logrs_logtop=r_(d.log_rs, np.log(d.top_F2)),
        by_cohort={c: dict(top_edge=g.top_edge.value_counts().index[0], top_edge_frac=float(g.top_edge.value_counts().iloc[0] / len(g)),
                           top_F2=float(g.top_F2.mean()), v1_top2=float(g.v1_top2.mean()), hom_F2=float(g.hom_F2.mean()),
                           inter_frac=float(g.inter_frac.mean()), r_MC_homF2=r_(g.MC, g.hom_F2))
                   for c, g in d.groupby("cohort")})
    N["robustness"] = dict(r_MC_logrs=r_(d.MC, d.log_rs), r_MC82_logrs82=r_(d.MC82, np.log(d.rs82)), r_MC82_MCth=r_(d.MC82, d.MCth),
                           r_MCnotop_logrsnotop=r_(d.MC_notop, np.log(d.rs_notop)),
                           age=dict(dev=dict(MC=r_(dev.age, dev.MC), MC82=r_(dev.age, dev.MC82), MC_notop=r_(dev.age, dev.MC_notop)),
                                    aging=dict(MC=r_(agi.age, agi.MC), MC82=r_(agi.age, agi.MC82), MC_notop=r_(agi.age, agi.MC_notop))))
    N["energy"] = dict(r_S_MC=r_(d.gram_Tinf, d.MC), pr_S_MC_given_logrs=pcor(d, "gram_Tinf", "MC", ["log_rs"]),
                       top_mode_share=float((0.9801 / (1 - 0.9801) / 90 / d.gram_Tinf).mean()))
    bins = pd.cut(d.age, [-1, 0.5, 2, 5, 9, 13, 17, 21, 26, 31, 36, 45, 55, 65, 75, 91])
    N["lifespan_bins"] = d.groupby(bins, observed=True)[["MC", "hom_F2", "inter_frac", "rs"]].mean().round(3).reset_index().astype(str).to_dict("records")
    life = dict(dev=dict(r_age_MC=r_(dev.age, dev.MC), r_age_hom=r_(dev.age, dev.hom_F2), pr_age_MC_given_hom=pcor(dev, "age", "MC", ["hom_F2"])),
                aging=dict(r_age_MC=r_(agi.age, agi.MC), r_age_hom=r_(agi.age, agi.hom_F2), pr_age_MC_given_hom=pcor(agi, "age", "MC", ["hom_F2"])))
    for c in ("camCAN", "HCPa"):
        g = d[d.cohort == c]
        life[c] = dict(n=len(g), r_age_MC=r_(g.age, g.MC), r_age_hom=r_(g.age, g.hom_F2), r_age_inter=r_(g.age, g.inter_frac),
                       r_age_med=r_(g.age, g.hom_med), r_age_lat=r_(g.age, g.hom_lat), r_MC_hom=r_(g.MC, g.hom_F2),
                       pr_age_MC_given_hom=pcor(g, "age", "MC", ["hom_F2"]), pr_age_MC_given_top=pcor(g, "age", "MC", ["top_F2"]))
    cc = d[d.cohort == "camCAN"]
    life["camCAN_bins"] = cc.groupby(pd.cut(cc.age, [17, 25, 30, 35, 40, 50, 60, 70, 80, 90]), observed=True)[["hom_F2", "MC"]].mean().round(3).reset_index().astype(str).to_dict("records")
    young = cc[cc.age <= 37]; hy = d[d.cohort == "HCPya"]
    life["HCPya_vs_camCAN_young"] = dict(hom_HCPya=float(hy.hom_F2.mean()), hom_camCAN=float(young.hom_F2.mean()),
                                         MC_HCPya=float(hy.MC.mean()), MC_camCAN=float(young.MC.mean()))
    N["lifespan"] = life

    # regional aging pattern (camCAN vs HCPa; AAL3 labels): per-region share of the order-2 sum vs age
    def share_mat(sel):
        M = []
        for A in mats[sel]:
            s = A.sum(1); Y = np.where(s > 0, (A ** 2).sum(1) / np.where(s > 0, s, 1) ** 2, 0); c = s ** 2 * Y; M.append(c / c.sum())
        return np.array(M)
    reg = {}
    for c, k in (("camCAN", 8), ("HCPa", 9)):
        sel = ds == k; M = share_mat(sel); a = d.age.values[sel]
        reg[c] = np.array([r_(a, M[:, j]) if M[:, j].std() > 0 else 0.0 for j in range(90)])
    rg = pd.DataFrame(dict(region=[CANON(n) for n in AAL3], r_age_camCAN=reg["camCAN"], r_age_HCPa=reg["HCPa"]))
    rg.to_csv(OUT / "regional_aging.csv", index=False)
    N["regional_aging"] = dict(r_camCAN_HCPa=r_(rg.r_age_camCAN, rg.r_age_HCPa),
                               losses=rg.sort_values("r_age_camCAN").head(6).round(2).values.tolist(),
                               gains=rg.sort_values("r_age_camCAN").tail(6).round(2).values.tolist())

    # ---- 3. null models (closed form) on the 100 subjects of the stored surrogates
    surr = pd.read_csv(REPO / "paper/data/surrogates.csv")
    old = surr.groupby(["model", "subject_id"])[["MC_lin", "m2"]].mean().reset_index()
    real_old = old[old.model == "Real"].set_index("subject_id")
    nrows = []
    for m in ("Uniform", "BrokenStick", "Reshuffle"):
        g = old[old.model == m].set_index("subject_id")
        for s_ in g.index:
            nrows.append(dict(sid=int(s_), model=m, dMC=float(g.loc[s_, "MC_lin"] - real_old.loc[s_, "MC_lin"]),
                              m2=float(g.loc[s_, "m2"]), m2_real=float(real_old.loc[s_, "m2"])))
    pos = {int(s_): i for i, s_ in enumerate(sid)}
    for s_ in sorted(surr.subject_id.unique()):
        A = mats[pos[int(s_)]]; k = int(ds[pos[int(s_)]]); base = mc_closed(A, s_)
        m2 = lambda B: float(pa.RHO ** 2 * (B ** 2).sum() / np.abs(np.linalg.eigvalsh(B)).max() ** 2 / 90)
        for m, B in (("EqualStrength", balance(A)), ("InterHalf", A * np.where(sets_of(k)[2], 0.5, 1.0))):
            if B is None:
                print(f"  balancing did not converge for subject {s_}; skipped", flush=True); continue
            nrows.append(dict(sid=int(s_), model=m, dMC=mc_closed(B, s_) - base, m2=m2(B), m2_real=m2(A)))
    nl = pd.DataFrame(nrows); slope = np.polyfit(d.log_rs, d.MC, 1)[0]     # MC = a + b log r_s across real connectomes
    nl["dMC_pred"] = slope * np.log(nl.m2 / nl.m2_real)
    nl.to_csv(OUT / "nulls_v3.csv", index=False)
    pv = {m: float(stats.wilcoxon(g.dMC).pvalue) for m, g in nl.groupby("model")}
    q = dict(zip(pv, stats.false_discovery_control(list(pv.values()))))
    N["nulls"] = {m: dict(dMC_median=float(g.dMC.median()), dMC_pred_median=float(g.dMC_pred.median()),
                          r_obs_pred=r_(g.dMC, g.dMC_pred) if g.dMC_pred.std() > 0 else None, p=pv[m], p_fdr=float(q[m]))
                  for m, g in nl.groupby("model")}

    # ---- 4. virtual rescaling of interhemispheric weights (camCAN, HCPa)
    KAP = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]; krows = []; vr = {}
    for c, k in (("camCAN", 8), ("HCPa", 9)):
        idx = np.where(ds == k)[0]; inter = sets_of(k)[2]
        for i in idx:
            for kap in KAP:
                krows.append(dict(cohort=c, sid=int(sid[i]), age=float(d.age.values[i]), kappa=kap,
                                  MC=mc_closed(mats[i] * np.where(inter, kap, 1.0), sid[i])))
        yi, oi = idx[d.age.values[idx] < 40], idx[d.age.values[idx] >= 70]
        f_old = float(np.mean([mats[i][inter].sum() / mats[i].sum() for i in oi])); pred = []
        for i in yi:
            A = mats[i]; I = A[inter].sum(); T = A.sum(); kap = f_old * (T - I) / (I * (1 - f_old))
            pred.append(mc_closed(A * np.where(inter, kap, 1.0), sid[i]))
        vr[c] = dict(n_young=len(yi), n_old=len(oi), MC_young=float(d.MCth.values[yi].mean()), MC_young_rescaled=float(np.mean(pred)),
                     MC_old=float(d.MCth.values[oi].mean()), inter_young=float(np.mean([mats[i][inter].sum() / mats[i].sum() for i in yi])),
                     inter_old=f_old)
    pd.DataFrame(krows).to_csv(OUT / "kappa.csv", index=False)
    N["virtual_rescaling"] = vr

    # ---- 5. CALM referred vs comparison children (all_data.mat)
    import scipy.io as sio
    allm = sio.loadmat(args.allmat, squeeze_me=True, struct_as_record=False)["all_data"].connectomes.astype(float)
    demo = pd.read_csv(args.demo)
    keep = np.where(demo.group.values == 1)[0]                       # the 3,901 analysed networks, in order
    assert len(keep) == len(mats) and np.allclose(np.triu(allm[keep[:50]], 1), np.triu(mats[:50], 1)), "all_data.mat does not match the export"
    crows = []
    for i in np.where(demo.dataset.values == 3)[0]:
        A = allm[i].copy(); np.fill_diagonal(A, 0)
        q = structure(A, 3); q.update(idx=int(i), age=float(demo.age[i]), sex=int(demo.sex[i]), referred=int(demo.group[i] == 2),
                                      MCth=mc_closed(A, 10_000_000 + i))
        crows.append(q)
    cm = pd.DataFrame(crows); cm.to_csv(OUT / "calm.csv", index=False)
    N["calm"] = dict(n_referred=int(cm.referred.sum()), n_comparison=int((1 - cm.referred).sum()),
                     age_referred=float(cm[cm.referred == 1].age.mean()), age_comparison=float(cm[cm.referred == 0].age.mean()),
                     **{y: ols_group(cm, y, "referred") for y in ("MCth", "rs", "hom_F2", "top_F2")},
                     means={y: [float(cm[cm.referred == 0][y].mean()), float(cm[cm.referred == 1][y].mean())] for y in ("MCth", "rs", "hom_F2")})

    # ---- per-cohort table
    N["cohorts"] = {c: dict(n=len(g), atlas=g.atlas.iloc[0], age=[float(g.age.min()), float(g.age.max())], MC=float(g.MC.mean()),
                            R2_MC_logrs=r_(g.MC, g.log_rs) ** 2, r_age_MC=r_(g.age, g.MC) if g.age.std() > 0 else None,
                            hom_F2=float(g.hom_F2.mean()))
                    for c, g in d.groupby("cohort")}
    json.dump(N, open(OUT / "numbers_v3.json", "w"), indent=1, default=float)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
