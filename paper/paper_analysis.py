#!/usr/bin/env python3
"""
Analyses behind the v2 manuscript and its Supplementary Information.

Reads the connectomes and the outputs of chain/chain_analysis.py (chain_subject.csv and
chain_surrogates_scale1e-05.csv, definitive run at input scale 1e-5) and writes everything
the figures and the text need to paper/data/:

  subject_v2.csv       one row per connectome: chain columns + stable rank, local/global
                       heterogeneity factors, truncated Gramian series, input-Gramian
                       spectrum summaries (effective rank, decay rate q), closed-form MC
  eta_sweep.csv        closed-form MC and effective rank for noise floors eta/eta0 = 1e-4..1e4
  examples.csv         input-Gramian eigenvalues c_k / eta0 for three subjects (p5, p50, p95 of MC)
  examples_curves.csv  closed-form forgetting curves m(tau) for the same subjects
  nodes_v2.csv         node-level sample (150 subjects): degree, disparity, walk-expansion terms
  surrogates.csv       copy of chain_surrogates_scale1e-05.csv
  ridge_sweep.csv      (--ridge-sweep) simulated tanh ESN at several ridge penalties, 300 subjects
  dynamics_noise.csv   closed-form MC with noise injected into the dynamics instead of at the
                       readout (White et al. 2004; Ganguli et al. 2008), 600 subjects
  numbers.json         every number quoted in the manuscript and SI

paper/make_figures.py draws the figures from paper/data only, so figures can be edited
without the connectomes.

Closed-form theory (SI Note 2). Linear reservoir x(t) = W~ x(t-1) + w u(t), u iid with
variance s2u = 1/3, W~ = 0.99 W / rho(W):
  C   = s2u sum_j W~^j w w' W~'^j                       (input controllability Gramian)
  g_t = s2u W~^t w                                      (covariance with u(t - t))
  m(t) = (g' M g)^2 / (s2u g' M C M g),  M = (C + eta I)^-1,  eta = ridge / n_train
  sum_{t>=0} g' M g / s2u = sum_k c_k / (c_k + eta)     (effective rank)
The C++ engine (ESNcpp/esn.hpp, mc_linear_theory) computes the same m(t) with its own input
vector; here w is drawn with numpy (seed = subject id), so MC_theory_py is an independent
realisation of the input mask.

Usage (repo root):
  python paper/paper_analysis.py --data procdata/data_with_metrics.pkl --chain chain/results
  python paper/paper_analysis.py --data procdata/data_with_metrics.pkl --chain chain/results --ridge-sweep
"""
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "chain"))
import chain_analysis as ca  # noqa: E402  (loaders, ESN runner, statistics helpers)

RHO, S2U, TAU = 0.99, 1.0 / 3.0, 20
STEPS, WASHOUT, TRAIN = 10000, 1000, 0.7
NTR = min(max(int(round(TRAIN * (STEPS - WASHOUT))), 10), STEPS - WASHOUT - 10)   # as in esn.hpp
RIDGE = 1e-6
ETA0 = RIDGE / NTR                     # noise floor of the manuscript (input scale cancels)
ETA_REL = [1e4, 1e3, 1e2, 1e1, 1.0, 1e-1, 1e-2, 1e-3, 1e-4]
T_SERIES = [1, 2, 3, 5, 10, 20]
EPS_DYN = [1e-9, 1e-7, 1e-5, 1e-3]    # variance of the noise injected per node and step (sigma_in = 1 units)
AGE_SPLIT = 32.0
MC = "MC_s1e-05"
DS_NAMES = {1: "dHCP", 2: "BCP", 3: "CALM", 4: "RED", 5: "ACE", 6: "HCPd", 7: "HCPya", 8: "camCAN", 9: "HCPa"}


# ------------------------------------------------------------------------ per subject
def input_gramian(Wn, w):
    """C = s2u sum_j Wn^j w w' Wn'^j by doubling (2^16 terms; |lambda| <= 0.99)."""
    C, A = S2U * np.outer(w, w), Wn.copy()
    for _ in range(16):
        C = C + A @ C @ A.T
        A = A @ A
    return 0.5 * (C + C.T)


def theory_curves(Wn, w, c, U, etas):
    """Closed-form m(tau), tau = 1..TAU, for each eta, from the eigendecomposition of C."""
    g = S2U * w
    P = np.empty((TAU, len(c)))
    for t in range(TAU):
        g = Wn @ g
        P[t] = (U.T @ g) ** 2                         # squared projections of g_tau on C's eigenvectors
    out = []
    for eta in etas:
        num = (P / (c + eta)).sum(1)                  # g' M g
        den = S2U * (P * c / (c + eta) ** 2).sum(1)   # s2u g' M C M g
        out.append(num ** 2 / den)
    return np.array(out)                              # (len(etas), TAU)


def decay_rate(c, k_max):
    """Geometric decay ratio q of the leading input-Gramian eigenvalues (log-linear fit)."""
    k = np.arange(min(max(k_max, 3), len(c)))
    slope = np.polyfit(k, np.log(c[k]), 1)[0]
    return float(np.exp(slope))


def subject_quantities(A, seed):
    N = A.shape[0]
    lam_raw, V = np.linalg.eigh(A)
    i = int(np.argmax(np.abs(lam_raw)))
    rho = abs(lam_raw[i]); v1 = V[:, i]
    Wn, lam = A * (RHO / rho), lam_raw * (RHO / rho)
    s = A.sum(1)
    lam2 = lam ** 2
    q = dict(stable_rank=float((A ** 2).sum() / rho ** 2),
             s2t=float(RHO ** 2 * (s ** 2).mean() / rho ** 2),       # global factor <s~^2>
             rho_over_srms=float(rho / np.sqrt((s ** 2).mean())),
             ipr_v1=float((v1 ** 4).sum()), cv_s=float(s.std() / s.mean()),
             m3_py=float((lam ** 3).mean()))
    for T in T_SERIES:
        q[f"gram_T{T}"] = float((lam2 * (1 - lam2 ** T) / (1 - lam2)).mean())
    q["gram_Tinf"] = float((lam2 / (1 - lam2)).mean())                 # = <(G_I)_ii> - 1
    q["n_slow"] = int((np.abs(lam) > 0.9).sum())

    w = np.random.default_rng(int(seed)).uniform(-1.0, 1.0, N)
    C = input_gramian(Wn, w)
    c, U = np.linalg.eigh(C)
    c, U = np.clip(c[::-1], 0.0, None), U[:, ::-1]
    curves = theory_curves(Wn, w, c, U, [ETA0 * e for e in ETA_REL])
    j0 = ETA_REL.index(1.0)
    q.update(MC_theory_py=float(curves[j0].sum()), eff_rank=float((c / (c + ETA0)).sum()),
             n_above=int((c > ETA0).sum()), c1_over_eta=float(c[0] / ETA0))
    q["q_decay"] = decay_rate(c, q["n_above"])
    q["mc_formula"] = float(1 + np.log(q["c1_over_eta"]) / np.log(1 / q["q_decay"]))
    sweep = [dict(eta_rel=e, MC_theory=float(curves[j].sum()),
                  eff_rank=float((c / (c + ETA0 * e)).sum())) for j, e in enumerate(ETA_REL)]
    return q, sweep, c, curves[j0]


def node_sample(A):
    N = A.shape[0]
    lam_raw, V = np.linalg.eigh(A)
    rho = np.abs(lam_raw).max()
    Wn, lam = A * (RHO / rho), lam_raw * (RHO / rho)
    k = (A > 0).sum(1).astype(float); s = A.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Y = np.where(s > 0, (A ** 2).sum(1) / s ** 2, np.nan)
    W2 = Wn @ Wn
    return pd.DataFrame(dict(k=k, Y=Y, Ups=k * Y, Ups_null=2 * k / (k + 1),
                             Cii=(V ** 2) @ np.exp(lam), T2=0.5 * np.diag(W2),
                             T3=np.einsum("ij,ji->i", W2, Wn) / 6.0, s_tilde=Wn.sum(1)))


def dynamics_noise(A, seed, eps_list=EPS_DYN):
    """MC (tau = 1..TAU) when white noise of variance eps is injected into every node at every step,
    x(t) = W~ x(t-1) + w u(t) + z(t), with the optimal linear readout. The state covariance is then
    C + eps C_n with C_n = sum_j W~^j W~^j' = (I - W~^2)^-1 for symmetric W~, and the squared correlation
    at delay tau is g' (C + eps C_n)^-1 g / s2u (White et al. 2004, Eq. 3)."""
    N = A.shape[0]
    lam_raw, V = np.linalg.eigh(A)
    rho = np.abs(lam_raw).max()
    Wn, lam = A * (RHO / rho), lam_raw * (RHO / rho)
    w = np.random.default_rng(int(seed)).uniform(-1.0, 1.0, N)    # same input vector as subject_quantities
    C = input_gramian(Wn, w)
    Cn = (V / (1 - lam ** 2)) @ V.T
    G = np.empty((TAU, N)); g = S2U * w
    for t in range(TAU):
        g = Wn @ g; G[t] = g
    out = {}
    for e in eps_list:
        X = np.linalg.solve(C + e * Cn, G.T)
        out[e] = float((G * X.T).sum() / S2U)
    return out


# ----------------------------------------------------------------------- statistics
def r_(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float(stats.pearsonr(x[m], y[m])[0])


def pcor(d, a, b, z):
    """Partial correlation r(a, b | z), z a column name or list of names."""
    z = [z] if isinstance(z, str) else list(z)
    return float(ca.partial_r(d[a].values, d[b].values, d[z].values)["r"])


def r2(d, y, cols):
    return float(ca.r2_ols(d, y, cols))


def smoothed_min_age(d, col):
    ab = np.floor(d["age"]).astype(int)
    g = d.groupby(ab)[col].mean(); n = d.groupby(ab)[col].size(); g = g[n >= 5]
    sm = g.rolling(5, center=True, min_periods=3).mean()
    a = sm[(sm.index >= 10) & (sm.index <= 60)]
    return int(a.idxmin())


def quad_vertex(age, y, n_boot=2000, seed=0):
    z = (age - age.mean()) / age.std()
    A = np.column_stack([np.ones(len(z)), z, z ** 2])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ b; s2 = (e ** 2).sum() / (len(y) - 3); cov = s2 * np.linalg.inv(A.T @ A)
    t = b[2] / np.sqrt(cov[2, 2]); p = 2 * stats.t.sf(abs(t), len(y) - 3)
    p2 = np.polyfit(age, y, 2); vtx = -p2[1] / (2 * p2[0])
    rng = np.random.default_rng(seed); vs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(age), len(age)); pb = np.polyfit(age[i], y[i], 2)
        vs.append(-pb[1] / (2 * pb[0]) if pb[0] > 0 else np.nan)
    vs = np.array(vs)
    return dict(vertex=float(vtx), ci=[float(np.nanpercentile(vs, 2.5)), float(np.nanpercentile(vs, 97.5))],
                p_quadratic=float(p), coef=[float(x) for x in p2])


def numbers(df, sweep, nodes, surr, ridge, dyn=None):
    N = {}
    d = df.dropna(subset=[MC]).copy()
    d["log_rs"] = np.log(d["stable_rank"])
    dev, agi = d[d["age"] <= AGE_SPLIT], d[d["age"] > AGE_SPLIT]
    N["n"] = int(len(d))
    N["MC"] = dict(min=float(d[MC].min()), p5=float(d[MC].quantile(.05)), median=float(d[MC].median()),
                   p95=float(d[MC].quantile(.95)), max=float(d[MC].max()), mean=float(d[MC].mean()))
    N["MC_scale1"] = dict(min=float(d["MC_s1"].min()), median=float(d["MC_s1"].median()), max=float(d["MC_s1"].max()))
    N["theory"] = dict(r_sim_cpp=r_(d[MC].values, d["MClin_s1e-05"].values),
                       r_sim_py=r_(d[MC].values, d["MC_theory_py"].values),
                       r_sim_cpp_scale1=r_(d["MC_s1"].values, d["MClin_s1"].values),
                       mean_sim=float(d[MC].mean()), mean_cpp=float(d["MClin_s1e-05"].mean()),
                       mean_py=float(d["MC_theory_py"].mean()))
    N["stable_rank"] = dict(median=float(d["stable_rank"].median()), min=float(d["stable_rank"].min()),
                            max=float(d["stable_rank"].max()),
                            r_MC={k: r_(x["stable_rank"].values, x[MC].values) for k, x in [("all", d), ("dev", dev), ("aging", agi)]},
                            R2_lin=r2(d, MC, ["stable_rank"]), R2_log=r2(d, MC, ["log_rs"]),
                            R2_log_plus_Ginf=r2(d, MC, ["log_rs", "gram_Tinf"]),
                            r_Cii=r_(d["Cii_mean"].values, d["stable_rank"].values),
                            r_Ginf=r_(d["gram_Tinf"].values, d["stable_rank"].values),
                            identity_max_relerr=float(d["identity_relerr"].max()) if "identity_relerr" in d else None)
    N["effective_rank"] = dict(mean=float(d["eff_rank"].mean()), min=float(d["eff_rank"].min()), max=float(d["eff_rank"].max()),
                               r_MC=r_(d["eff_rank"].values, d[MC].values), r_rs=r_(d["eff_rank"].values, d["stable_rank"].values),
                               n_above_mean=float(d["n_above"].mean()), r_MCpy=r_(d["eff_rank"].values, d["MC_theory_py"].values),
                               q_median=float(d["q_decay"].median()),
                               r_q_rs=r_(d["q_decay"].values, d["stable_rank"].values),
                               r_formula_effrank=r_(d["mc_formula"].values, d["eff_rank"].values))
    N["gram_series_r_MC"] = {f"T{T}": r_(d[f"gram_T{T}"].values, d[MC].values) for T in T_SERIES}
    N["gram_series_r_MC"]["Tinf"] = r_(d["gram_Tinf"].values, d[MC].values)
    N["n_slow_r_MC"] = r_(d["n_slow"].values.astype(float), d[MC].values)
    preds = {"stable_rank": "stable_rank", "Cii": "Cii_mean", "abs_lambda": "abs_lam_mean",
             "communicability": "comm_mean", "m3": "m3", "clustering_Cw": "C_w", "disparity_ratio": "Ratio",
             "strength_raw": "strength", "degree": "degree", "avg_controllability": "gram_Tinf",
             "global_factor": "s2t", "density": "density"}
    race = {}
    for name, x in [("all", d), ("dev", dev), ("aging", agi)]:
        race[name] = {}
        for lab, c in preds.items():
            if c not in x: continue
            race[name][lab] = dict(r=r_(x[c].values, x[MC].values),
                                   partial_given_rs=None if c == "stable_rank" else pcor(x, c, MC, "stable_rank"))
    N["predictors"] = race
    N["global_factor"] = dict(r_rho_over_srms=r_(d["s2t"].values, d["rho_over_srms"].values),
                              r_cv_s=r_(d["s2t"].values, d["cv_s"].values),
                              r_ipr=r_(d["s2t"].values, d["ipr_v1"].values),
                              max=float(d["s2t"].max()))
    ages = {}
    for name, x in [("dev", dev), ("aging", agi)]:
        ages[name] = {c: r_(x[c].values, x["age"].values) for c in
                      [MC, "stable_rank", "Ratio", "s2t", "strength", "density", "Cii_mean"] if c in x}
        ages[name]["MC_vs_strength"] = r_(x["strength"].values, x[MC].values)
        ages[name]["partial_MC_age_given_density"] = pcor(x, MC, "age", "density")
    N["age_correlations"] = ages
    N["smoothed_min_age"] = {c: smoothed_min_age(d, c) for c in [MC, "stable_rank", "Cii_mean", "s2t"]}
    coh = {}
    for code, x in d.groupby("dataset"):
        nm = DS_NAMES.get(int(code), str(code))
        e = dict(n=int(len(x)), age_min=float(x["age"].min()), age_max=float(x["age"].max()),
                 MC_mean=float(x[MC].mean()), rs_mean=float(x["stable_rank"].mean()),
                 R2_log_rs=r2(x, MC, ["log_rs"]),
                 partial_ratio_given_global=pcor(x, "Ratio", MC, "s2t"),
                 partial_global_given_ratio=pcor(x, "s2t", MC, "Ratio"))
        if x["age"].std() > 0:
            e.update(r_MC_age=r_(x[MC].values, x["age"].values), r_rs_age=r_(x["stable_rank"].values, x["age"].values),
                     r_ratio_age=r_(x["Ratio"].values, x["age"].values), r_global_age=r_(x["s2t"].values, x["age"].values))
        if "sex" in x and x["sex"].nunique() == 2:
            a, b = x[x["sex"] == 0][MC], x[x["sex"] == 1][MC]
            sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
            e.update(sex_n0=int(len(a)), sex_n1=int(len(b)), sex_d=float((b.mean() - a.mean()) / sp),
                     sex_p=float(stats.mannwhitneyu(a, b).pvalue))
        coh[nm] = e
    N["cohorts"] = coh
    cc = d[d["dataset"] == 8]
    if len(cc) > 50:
        bins = [18, 25, 32, 40, 50, 60, 70, 80, 90]
        g = cc.groupby(pd.cut(cc["age"], bins), observed=True)
        N["camcan"] = dict(bins=[f"{int(i.left)}-{int(i.right)}" for i in g.groups.keys()],
                           MC=[float(v) for v in g[MC].mean()], rs=[float(v) for v in g["stable_rank"].mean()],
                           ratio=[float(v) for v in g["Ratio"].mean()], s2t=[float(v) for v in g["s2t"].mean()],
                           n=[int(v) for v in g.size()],
                           MC_quad=quad_vertex(cc["age"].values, cc[MC].values),
                           rs_quad=quad_vertex(cc["age"].values, cc["stable_rank"].values))
        lo, hi = cc[cc["age"] <= 40], cc[cc["age"] >= 35]
        N["camcan"].update(r_le40=r_(lo[MC].values, lo["age"].values), p_le40=float(stats.pearsonr(lo[MC], lo["age"])[1]),
                           r_ge35=r_(hi[MC].values, hi["age"].values), p_ge35=float(stats.pearsonr(hi[MC], hi["age"])[1]))
    def overlap(a, b, lo, hi):
        A = d[(d["dataset"] == a) & d["age"].between(lo, hi)]; B = d[(d["dataset"] == b) & d["age"].between(lo, hi)]
        out = {}
        for c in [MC, "stable_rank", "Ratio", "s2t"]:
            out[c] = dict(a=float(A[c].mean()), b=float(B[c].mean()), na=int(len(A)), nb=int(len(B)),
                          d_sd=float((A[c].mean() - B[c].mean()) / d[c].std()), p=float(stats.mannwhitneyu(A[c], B[c]).pvalue))
        return out
    N["overlap"] = {"HCPya_vs_camCAN_22_37": overlap(7, 8, 22, 37), "camCAN_vs_HCPa_36_89": overlap(8, 9, 36, 89)}
    # sweeps
    sw = sweep.merge(d[["sid", "stable_rank"]], on="sid")
    N["eta_sweep"] = {f"{e:g}": dict(MC_mean=float(g["MC_theory"].mean()), eff_rank_mean=float(g["eff_rank"].mean()),
                                     r_rs=r_(g["MC_theory"].values, g["stable_rank"].values))
                      for e, g in sw.groupby("eta_rel")}
    if ridge is not None:
        rr = ridge.merge(d[["sid", "stable_rank"]], on="sid")
        N["ridge_sweep"] = {f"{e:g}": dict(n=int(len(g)), MC_mean=float(g["MC"].mean()), r_rs=r_(g["MC"].values, g["stable_rank"].values),
                                           r_sim_theory=r_(g["MC"].values, g["MC_lin"].values))
                            for e, g in rr.groupby("ridge")}
    if dyn is not None:
        dd = dyn.merge(d[["sid", "stable_rank", "MC_theory_py"]], on="sid")
        N["dynamics_noise"] = {f"{e:g}": dict(n=int(len(g)), MC_mean=float(g["MC"].mean()),
                                              r_log_rs=r_(g["MC"].values, np.log(g["stable_rank"].values)),
                                              r_readout=r_(g["MC"].values, g["MC_theory_py"].values))
                               for e, g in dd.groupby("eps")}
        N["dynamics_noise"]["readout_same_subjects"] = dict(
            r_log_rs=r_(dd.drop_duplicates("sid")["MC_theory_py"].values, np.log(dd.drop_duplicates("sid")["stable_rank"].values)))
    # node level
    nd = nodes.dropna()
    N["nodes"] = dict(n=int(len(nd)), R2_order2=float(np.corrcoef(nd["Cii"], 1 + nd["T2"])[0, 1] ** 2),
                      R2_order3=float(np.corrcoef(nd["Cii"], 1 + nd["T2"] + nd["T3"])[0, 1] ** 2))
    for lab, sel in [("k_lt20", nd["k"] < 20), ("k_20_40", nd["k"].between(20, 40)), ("k_gt40", nd["k"] > 40)]:
        x = nd[sel]
        N["nodes"][f"order3_share_{lab}"] = float((x["T3"] / (x["Cii"] - 1)).median())
        N["nodes"][f"order2_relerr_{lab}"] = float(((x["Cii"] - 1 - x["T2"]) / (x["Cii"] - 1)).median())
    N["subject_level_R2_order2"] = float(np.corrcoef(d["Cii_mean"], 1 + d["m2"] / 2)[0, 1] ** 2)
    # surrogates
    b, a = np.polyfit(d["m2"], d[MC], 1)
    sm = surr.groupby(["subject_id", "model"])[["MC", "m2", "MC_lin"]].mean().reset_index()
    wide = sm.pivot(index="subject_id", columns="model", values="MC")
    mods, pv = {}, {}
    for m in ["Real", "Uniform", "BrokenStick", "Reshuffle", "SignFlip", "RealAsym"]:
        if m not in wide: continue
        g = sm[sm["model"] == m]
        e = dict(MC=float(g["MC"].mean()), m2=float(g["m2"].mean()), MC_lin=float(g["MC_lin"].mean()),
                 MC_from_line=float(a + b * g["m2"].mean()))
        if m != "Real":
            dd = (wide[m] - wide["Real"]).dropna()
            e.update(dMC_median=float(dd.median()), n=int(len(dd)))
            pv[m] = float(stats.wilcoxon(dd).pvalue)
        mods[m] = e
    ks = list(pv); padj = stats.false_discovery_control([pv[k] for k in ks])
    for k, p, q in zip(ks, [pv[k] for k in ks], padj):
        mods[k].update(wilcoxon_p=p, p_fdr=float(q))
    N["surrogates"] = dict(line=dict(intercept=float(a), slope=float(b)), models=mods,
                           n_real=int(surr["realization"].nunique()), n_subjects=int(surr["subject_id"].nunique()))
    return N


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="export folder (chain/export_data.py) or data_with_metrics.pkl")
    ap.add_argument("--chain", default=str(REPO / "chain" / "results"), help="folder with chain_analysis.py outputs")
    ap.add_argument("--out", default=str(REPO / "paper" / "data"))
    ap.add_argument("--ridge-sweep", action="store_true", help="also simulate the tanh ESN at several ridge values")
    ap.add_argument("--ridge-n", type=int, default=300)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--eigen", default=None)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    chain = Path(args.chain)

    print("loading connectomes ...", flush=True)
    sid, mats, meta = ca.load_pickle(args.data) if str(args.data).endswith(".pkl") else ca.load_export(args.data)
    cs = pd.read_csv(chain / "chain_subject.csv")
    if MC not in cs:
        raise SystemExit(f"{chain}/chain_subject.csv has no {MC}: run chain_analysis.py with --scales 1 1e-5")

    print("per-subject quantities ...", flush=True)
    rows, sweep, ex_c, ex_m = [], [], {}, {}
    for n, (s, A) in enumerate(zip(sid, mats)):
        q, sw, c, m = subject_quantities(A, s)
        q["sid"] = int(s); rows.append(q)
        for x in sw: x["sid"] = int(s); sweep.append(x)
        ex_c[int(s)], ex_m[int(s)] = c, m
        if (n + 1) % 500 == 0: print(f"  {n + 1}/{len(sid)}", flush=True)
    new = pd.DataFrame(rows)
    keep = [c for c in cs.columns if c not in new.columns or c == "sid"]
    df = cs[keep].merge(new, on="sid", how="inner")
    sweep = pd.DataFrame(sweep)

    # three example subjects: closest to the 5th, 50th and 95th percentile of simulated MC
    ex_rows, cur_rows = [], []
    for lab, pct in [("p5", 5), ("p50", 50), ("p95", 95)]:
        target = np.percentile(df[MC].dropna(), pct)
        s = int(df.loc[(df[MC] - target).abs().idxmin(), "sid"])
        c = ex_c[s]
        ex_rows += [dict(label=lab, sid=s, k=k + 1, c_over_eta=float(v / ETA0)) for k, v in enumerate(c)]
        cur_rows += [dict(label=lab, sid=s, tau=t + 1, m=float(v)) for t, v in enumerate(ex_m[s])]

    print("node sample ...", flush=True)
    pick = np.random.default_rng(0).choice(len(sid), size=min(150, len(sid)), replace=False)
    nodes = pd.concat([node_sample(mats[i]).assign(sid=int(sid[i])) for i in pick], ignore_index=True)

    surr_src = chain / "chain_surrogates_scale1e-05.csv"
    surr = pd.read_csv(surr_src)

    ridge = None
    if args.ridge_sweep:
        print("ridge sweep (tanh ESN) ...", flush=True)
        work = out / "work"; work.mkdir(exist_ok=True)
        sub = np.sort(np.random.default_rng(2).choice(len(sid), min(args.ridge_n, len(sid)), replace=False))
        csv = work / "connectomes_ridge.csv"; ca.write_connectomes_csv(csv, sid[sub], mats[sub])
        exe = ca.build("esn_mc", ca.find_eigen(args.eigen))
        parts = []
        for rg in [1e-3, 1e-6, 1e-9, 1e-12]:
            o = work / f"ridge_{rg:g}.csv"
            ca.run_cfg(exe, work / f"cfg_ridge_{rg:g}.txt", {**ca.ESN_CFG, "ridge": rg, "input_scale": 1e-5,
                       "log_theory": 1, "threads": args.threads, "in_csv": str(csv), "out_csv": str(o)})
            t = pd.read_csv(o).rename(columns={"subject_id": "sid", "MC_Glob": "MC"})
            parts.append(t[["sid", "MC", "MC_lin"]].assign(ridge=rg))
        ridge = pd.concat(parts, ignore_index=True)
        ridge.to_csv(out / "ridge_sweep.csv", index=False)
    elif (out / "ridge_sweep.csv").exists():
        ridge = pd.read_csv(out / "ridge_sweep.csv")

    print("dynamics noise (600 subjects) ...", flush=True)
    pick_d = np.random.default_rng(5).choice(len(sid), min(600, len(sid)), replace=False)
    dyn = pd.DataFrame([dict(sid=int(sid[i]), eps=e, MC=v) for i in pick_d
                        for e, v in dynamics_noise(mats[i], sid[i]).items()])

    print("numbers ...", flush=True)
    N = numbers(df, sweep, nodes, surr, ridge, dyn)
    N["constants"] = dict(rho=RHO, s2u=S2U, n_train=NTR, ridge=RIDGE, eta0=ETA0, tau_max=TAU, age_split=AGE_SPLIT)

    df.to_csv(out / "subject_v2.csv", index=False)
    sweep.to_csv(out / "eta_sweep.csv", index=False)
    pd.DataFrame(ex_rows).to_csv(out / "examples.csv", index=False)
    pd.DataFrame(cur_rows).to_csv(out / "examples_curves.csv", index=False)
    nodes.to_csv(out / "nodes_v2.csv", index=False)
    dyn.to_csv(out / "dynamics_noise.csv", index=False)
    shutil.copy(surr_src, out / "surrogates.csv")
    (out / "numbers.json").write_text(json.dumps(N, indent=2))
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
