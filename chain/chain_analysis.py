#!/usr/bin/env python3
"""
Analytical chain  disparity -> subgraph centrality -> Gramian -> spectrum -> memory capacity,
on the real (symmetric) dMRI connectomes and on symmetric weight surrogates.

For every connectome W (diagonal zeroed, rescaled to spectral radius rho, W~ = rho W / rho(W)):

  step 1  C_ii = (e^W~)_ii = 1 + (W~^2)_ii/2 + ...,   (W~^2)_ii = s~_i^2 Y_i       (node level)
          sum_i s~_i^2 Y_i = Tr W~^2 = sum_k lambda_k^2                          (exact)
  step 2  (G_I)_ii = [(I - W~^2)^-1]_ii = 1 + s~_i^2 Y_i + ...   (Gramian, B = I)
  step 3  linear theory: MC of the linear reservoir with the same Win and the same ridge
          penalty per sample, in closed form (ESNcpp/esn.hpp, mc_linear_theory), against
          the simulated tanh ESN at each input scale
  step 4  structure -> MC: <s~^2 Y> = sum lambda^2 / N, <|lambda|> (Aceituno et al.),
          higher spectral moments, <C_ii>, clustering; R^2 increments over sum lambda^2
  regime  development (age <= 32) vs aging: shape (Upsilon ratio) vs scale (<s~^2>)
  nulls   esn_surrogates (symmetric Uniform / Broken Stick / Reshuffle / Sign Flip, plus the
          real weights made asymmetric as a control) with sum lambda^2 and theory per matrix

The ESN runs use ESNcpp/esn_mc and ESNcpp/esn_surrogates with the manuscript
hyper-parameters (RES + MAIN/SURR of ESNcpp/TFM_closing_figures.ipynb); at input_scale=1
esn_mc reproduces the manuscript MC, and the surrogate subset is the notebook's one when
<data>/cache/mc_surrogates.* exists.

Usage (from the repo root):
    python chain/chain_analysis.py --data ../NetLearn-data            # exported data
    python chain/chain_analysis.py --data procdata/data_with_metrics.pkl
    python chain/chain_analysis.py --synthetic 60 --out chain/results_synthetic   # pipeline test
"""
import argparse, json, os, subprocess, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
ESN_DIR = REPO / "ESNcpp"
# Manuscript ESN hyper-parameters = RES + MAIN in ESNcpp/TFM_closing_figures.ipynb
ESN_CFG = dict(reservoir_size=90, spectral_radius=0.99, ridge=1e-6, train_ratio=0.7,
               washout=1000, steps=10000, tau=20, win_reps=1, seed=42)
SURR_CFG = dict(steps=10000, tau=20)          # SURR in the notebook (n_real from --surr-real)
AGE_SPLIT = 32.0                      # development: age <= 32 ; aging: age > 32 (as in the notebook)
MODELS = ["Real", "Uniform", "BrokenStick", "Reshuffle", "SignFlip", "RealAsym"]
NULLS = ["Uniform", "BrokenStick", "Reshuffle", "SignFlip", "RealAsym"]


# ----------------------------------------------------------------------------- data
def as_matrix(x):
    A = np.asarray(x, dtype=float)
    if A.ndim == 1:
        n = int(round(np.sqrt(A.size))); A = A.reshape(n, n)
    A = A.copy(); np.fill_diagonal(A, 0.0)
    return A


def load_export(folder):
    """Folder written by chain/export_data.py (connectomes_partXX.npz + metadata.csv)."""
    folder = Path(folder)
    parts = sorted(folder.glob("connectomes_part*.npz"))
    if not parts:
        raise FileNotFoundError(f"no connectomes_part*.npz in {folder}")
    sids, mats = [], []
    for f in parts:
        z = np.load(f, allow_pickle=False)
        N, layout, w = int(z["N"]), str(z["layout"]), z["weights"].astype(float)
        for row in w:
            if layout == "upper":
                A = np.zeros((N, N)); iu = np.triu_indices(N, 1); A[iu] = row; A = A + A.T
            else:
                A = row.reshape(N, N).copy(); np.fill_diagonal(A, 0.0)
            mats.append(A)
        sids.append(z["sid"])
    sid = np.concatenate(sids)
    meta = pd.read_csv(folder / "metadata.csv") if (folder / "metadata.csv").exists() else pd.DataFrame({"sid": sid})
    return sid, np.stack(mats), meta


def load_pickle(path, matrix_col="connectome"):
    df = pd.read_pickle(path).reset_index(drop=True)
    sid = np.arange(len(df))
    mats = np.stack([as_matrix(x) for x in df[matrix_col]])
    scalar = [c for c in df.columns if c != matrix_col and
              not isinstance(df[c].dropna().iloc[0] if df[c].notna().any() else 0, (np.ndarray, list, tuple, dict))]
    meta = df[scalar].copy(); meta.insert(0, "sid", sid)
    return sid, mats, meta


def synthetic(S, N=90, seed=0):
    """Connectome-like test data: heterogeneous degrees, heavy-tailed weights, s ~ k^1.5,
    with weight heterogeneity drifting with a fake age (only to exercise the pipeline)."""
    rng = np.random.default_rng(seed)
    age = np.sort(rng.uniform(0, 90, S)); mats = []
    for a in age:
        dens = rng.uniform(0.25, 0.5)
        sig = 1.6 - 0.9 * np.exp(-((a - 32) / 25) ** 2) + rng.normal(0, 0.15)   # U-shaped heterogeneity
        kexp = rng.lognormal(0, 0.6, N); kexp *= dens * (N - 1) / kexp.mean()
        P = np.clip(np.outer(kexp, kexp) / kexp.sum(), 0, 1)
        A = (rng.random((N, N)) < P).astype(float); A = np.triu(A, 1); A = A + A.T
        k = A.sum(1); Wt = np.sqrt(np.outer(k, k)) * rng.lognormal(0, max(sig, 0.2), (N, N))
        Wt = np.round(np.triu(Wt, 1) * 10); mats.append(A * (Wt + Wt.T))
    meta = pd.DataFrame({"sid": np.arange(S), "age": age, "dataset": "synthetic"})
    return np.arange(S), np.stack(mats), meta


# ------------------------------------------------------------------------ structure
def structural_chain(A, rho, tau):
    """Node- and subject-level quantities of the chain for one connectome."""
    N = A.shape[0]
    sym = np.allclose(A, A.T, rtol=0, atol=1e-10 * max(1.0, np.abs(A).max()))
    As = A if sym else 0.5 * (A + A.T)                  # base model is symmetric; guard only
    ev = np.linalg.eigvalsh(As); sr = np.abs(ev).max()
    Wn = As * (rho / sr)
    lam, V = np.linalg.eigh(Wn)
    V2 = V ** 2
    k = (As > 0).sum(1).astype(float)
    s = Wn.sum(1)
    sY = (Wn ** 2).sum(1)                               # (W~^2)_ii = s~_i^2 Y_i
    with np.errstate(divide="ignore", invalid="ignore"):
        Y = np.where(s > 0, sY / s ** 2, np.nan)
        ratio = np.where(k > 1, k * Y / (2 * k / (k + 1)), np.nan)   # Upsilon / mu_null(k)
    Cii = V2 @ np.exp(lam)
    GI = V2 @ (1.0 / (1.0 - lam ** 2))
    approx2 = 1.0 + 0.5 * sY
    lam2 = lam ** 2
    subj = dict(
        symmetric=bool(sym),
        m2=float(lam2.mean()),                                  # = <s~^2 Y>  (exact)
        m3=float((lam ** 3).mean()),                            # = Tr W~^3 / N : closed triangles
        m4=float((lam ** 4).mean()),
        identity_relerr=float(abs(sY.sum() - lam2.sum()) / lam2.sum()),
        stable_rank=float((As ** 2).sum() / sr ** 2),
        imc=float((lam2 * (1 - lam2 ** tau)).sum()),           # sum lam^2 (1 - lam^(2 tau))
        Cii_mean=float(Cii.mean()), GI_mean=float(GI.mean()),
        abs_lam_mean=float(np.abs(lam).mean()),                 # Aceituno et al. proxy
        Y_mean=float(np.nanmean(Y)), Ups_mean=float(np.nanmean(k * Y)), ratio_mean=float(np.nanmean(ratio)),
        s_tilde_mean=float(s.mean()), s2_mean=float((s ** 2).mean()),
        strength_raw_mean=float(As.sum(1).mean()),
        density=float((As > 0).sum() / (N * (N - 1))),
        r_C_order2=float(np.corrcoef(Cii, approx2)[0, 1]),
        r_C_GI=float(np.corrcoef(Cii, GI)[0, 1]),
        rho_C_GI=float(stats.spearmanr(Cii, GI)[0]),
    )
    nodes = dict(Cii=Cii, approx2=approx2, GI=GI, sY=sY)
    return subj, nodes


def run_structure(sid, mats, rho, tau, n_plot_subjects=150, seed=0):
    rows, pooled = [], {k: [] for k in ("Cii", "approx2", "GI", "sY")}
    for i, A in enumerate(mats):
        subj, nodes = structural_chain(A, rho, tau)
        subj["sid"] = int(sid[i]); rows.append(subj)
        for k in pooled: pooled[k].append(nodes[k])
        if (i + 1) % 500 == 0: print(f"  structure {i + 1}/{len(mats)}", flush=True)
    struct = pd.DataFrame(rows)
    P = {k: np.concatenate(v) for k, v in pooled.items()}
    pooled_stats = dict(
        n_nodes=int(P["Cii"].size),
        step1_R2_order2=float(np.corrcoef(P["Cii"], P["approx2"])[0, 1] ** 2),
        step1_R2_identity=float(1 - ((P["Cii"] - P["approx2"]) ** 2).sum() / ((P["Cii"] - P["Cii"].mean()) ** 2).sum()),
        step2_pearson_C_GI=float(np.corrcoef(P["Cii"], P["GI"])[0, 1]),
        step2_spearman_C_GI=float(stats.spearmanr(P["Cii"], P["GI"])[0]),
    )
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(mats), size=min(n_plot_subjects, len(mats)), replace=False)
    node_sample = pd.DataFrame({k: np.concatenate([pooled[k][j] for j in pick]) for k in pooled})
    return struct, node_sample, pooled_stats


# ------------------------------------------------------------------------------ ESN
def find_eigen(user=None):
    for p in [user, os.environ.get("EIGEN_INCLUDE"), "/usr/include/eigen3",
              str(ESN_DIR / "eigen-master"), "/usr/local/include/eigen3", "/opt/homebrew/include/eigen3",
              "/opt/homebrew/opt/eigen/include/eigen3", "/usr/local/opt/eigen/include/eigen3"]:
        if p and (Path(p) / "Eigen" / "Dense").exists():
            return p
    raise RuntimeError("Eigen not found: install it (apt install libeigen3-dev / brew install eigen) "
                       "or pass --eigen /path/to/eigen")


def build(prog, eigen):
    exe = ESN_DIR / prog
    src = [ESN_DIR / f"{prog}.cpp", ESN_DIR / "esn.hpp"]
    if exe.exists() and exe.stat().st_mtime > max(f.stat().st_mtime for f in src):
        return exe
    cmd = [os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-fopenmp", f"-I{eigen}", "-w",
           str(ESN_DIR / f"{prog}.cpp"), "-o", str(exe)]
    print("  building:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return exe


def write_connectomes_csv(path, sid, mats):
    """Same format as export_connectomes() in TFM_closing_figures.ipynb (values as %.6g)."""
    N = mats.shape[1]
    with open(path, "w") as f:
        f.write("subject_id," + ",".join(f"w{k}" for k in range(N * N)) + "\n")
        for s, A in zip(sid, mats):
            f.write(f"{int(s)}," + ",".join(f"{v:.6g}" for v in A.flatten()) + "\n")


def run_cfg(exe, cfg_path, kv):
    cfg_path.write_text("".join(f"{k}={v}\n" for k, v in kv.items()))
    t0 = time.time()
    subprocess.run([str(exe), str(cfg_path)], check=True)
    return time.time() - t0


def mc_columns(sc):
    return {"subject_id": "sid", "MC_Glob": f"MC_s{sc:g}", "r2_mean": f"r2_s{sc:g}", "MC_lin": f"MClin_s{sc:g}"}


def run_esn(exe, csv, workdir, scale, threads):
    out_csv = workdir / f"mc_scale{scale:g}.csv"
    dt = run_cfg(exe, workdir / f"cfg_scale{scale:g}.txt", {
        **ESN_CFG, "input_scale": scale, "log_state": 1, "log_theory": 1, "threads": threads,
        "in_csv": str(csv), "out_csv": str(out_csv)})
    print(f"  esn_mc input_scale={scale:g} done in {dt:.0f}s", flush=True)
    return pd.read_csv(out_csv).rename(columns=mc_columns(scale))


def run_surrogates(exe, csv, workdir, scale, threads, n_real):
    out_csv = workdir / f"surr_scale{scale:g}.csv"
    cfg = {k: v for k, v in ESN_CFG.items() if k not in ("steps", "tau", "win_reps")}
    dt = run_cfg(exe, workdir / f"cfg_surr_scale{scale:g}.txt", {
        **cfg, **SURR_CFG, "n_real": n_real, "input_scale": scale, "log_theory": 1, "asym_control": 1,
        "threads": threads, "in_csv": str(csv), "out_csv": str(out_csv)})
    print(f"  esn_surrogates input_scale={scale:g} done in {dt:.0f}s", flush=True)
    return pd.read_csv(out_csv)


def read_cache_table(cache, stem):
    if cache is None: return None
    for p in [cache / f"{stem}.csv", cache / f"{stem}.pkl"]:
        if p.exists():
            return pd.read_csv(p) if p.suffix == ".csv" else pd.read_pickle(p)
    return None


# ---------------------------------------------------------------------------- stats
def pr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4: return dict(r=np.nan, p=np.nan, rho=np.nan, n=int(m.sum()))
    r, p = stats.pearsonr(x[m], y[m]); rho = stats.spearmanr(x[m], y[m])[0]
    return dict(r=float(r), p=float(p), rho=float(rho), n=int(m.sum()))


def partial_r(x, y, Z):
    Z = np.column_stack([np.ones(len(x)), Z])
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), 1)
    rx = x[m] - Z[m] @ np.linalg.lstsq(Z[m], x[m], rcond=None)[0]
    ry = y[m] - Z[m] @ np.linalg.lstsq(Z[m], y[m], rcond=None)[0]
    return pr(rx, ry)


def r2_ols(df, y, cols):
    d = df[[y, *cols]].dropna()
    if len(d) < len(cols) + 3: return np.nan
    Z = np.column_stack([np.ones(len(d)), d[cols].values])
    res = d[y].values - Z @ np.linalg.lstsq(Z, d[y].values, rcond=None)[0]
    return float(1 - res.var() / d[y].var(ddof=0))


def lifespan(df, col):
    if "age" not in df: return {}
    dev, agi = df[df["age"] <= AGE_SPLIT], df[df["age"] > AGE_SPLIT]
    q = pd.qcut(df["age"], 24, duplicates="drop")
    traj = df.groupby(q, observed=True).agg(age=("age", "mean"), mc=(col, "mean"))
    return dict(r_age_development=pr(dev["age"], dev[col]),
                r_age_aging=pr(agi["age"], agi[col]),
                age_of_min_binned=float(traj.loc[traj["mc"].idxmin(), "age"]))


def shape_col(df):
    return "Ratio" if "Ratio" in df else "ratio_mean"      # Ratio = the manuscript's R = Y / Y_null


PREDICTORS = ["m2", "abs_lam_mean", "imc", "Cii_mean", "GI_mean", "m3", "m4", "Ratio", "ratio_mean",
              "Ups_mean", "Y_mean", "s2_mean", "C_w", "strength_raw_mean", "density"]


def regime_block(df, mc):
    if "age" not in df: return {}
    R = shape_col(df); out = {}
    for name, sub in [("development", df[df["age"] <= AGE_SPLIT]), ("aging", df[df["age"] > AGE_SPLIT]), ("all", df)]:
        blk = dict(n=int(len(sub)),
                   r={c: pr(sub[c], sub[mc]) for c in [R, "s2_mean", "m2", "abs_lam_mean", "C_w"] if c in sub},
                   R2={"m2": r2_ols(sub, mc, ["m2"]), R: r2_ols(sub, mc, [R])})
        if "C_w" in sub:
            blk["R2"].update({"m2+C_w": r2_ols(sub, mc, ["m2", "C_w"]), f"{R}+C_w": r2_ols(sub, mc, [R, "C_w"])})
        out[name] = blk
    return out


def surrogate_block(surr, df, sc):
    """Subject means per model; Delta MC vs Real; sum lambda^2 against the real-subject fit."""
    mc = f"MC_s{sc:g}"
    b, a = np.polyfit(df["m2"], df[mc], 1)                 # MC = a + b <s~^2 Y>, real connectomes
    sm = surr.groupby(["subject_id", "model"])[["MC", "m2", "MC_lin"]].mean().reset_index()
    wide = sm.pivot(index="subject_id", columns="model", values="MC")
    models = {}
    for m in [x for x in MODELS if x in wide.columns]:
        g = sm[sm["model"] == m]
        blk = dict(MC_mean=float(g["MC"].mean()), MC_lin_mean=float(g["MC_lin"].mean()),
                   m2_mean=float(g["m2"].mean()), MC_pred_from_m2=float(a + b * g["m2"].mean()))
        if m != "Real":
            d = (wide[m] - wide["Real"]).dropna()
            blk.update(dMC_median=float(d.median()), dMC_mean=float(d.mean()),
                       wilcoxon_p=float(stats.wilcoxon(d).pvalue) if (d != 0).any() else 1.0, n=int(len(d)))
        models[m] = blk
    sym = sm[sm["model"] != "RealAsym"]
    return dict(fit_real=dict(intercept=float(a), slope=float(b)), models=models,
                r_MC_MClin_all=pr(sm["MC"], sm["MC_lin"]), r_MC_m2_symmetric=pr(sym["MC"], sym["m2"]),
                n_subjects=int(sm["subject_id"].nunique()), n_real=int(surr["realization"].nunique()))


def manuscript_nulls(cache):
    old = read_cache_table(cache, "mc_surrogates")
    if old is None or "model" not in old: return None
    w = old.groupby(["subject_id", "model"])["MC"].mean().unstack()
    return {m: dict(dMC_median=float((w[m] - w["Real"]).median()),
                    wilcoxon_p=float(stats.wilcoxon((w[m] - w["Real"]).dropna()).pvalue))
            for m in w.columns if m != "Real"}


def summarize(df, scales, pooled, cache_check, surr, cache):
    out = dict(n_subjects=int(len(df)), esn_config=ESN_CFG, scales=scales, age_split=AGE_SPLIT, structure=pooled,
               identity_max_relerr=float(df["identity_relerr"].max()),
               step1_median_subject_r=float(df["r_C_order2"].median()),
               step2_median_subject_pearson=float(df["r_C_GI"].median()),
               step2_median_subject_spearman=float(df["rho_C_GI"].median()))
    preds = [p for p in PREDICTORS if p in df]
    sets = [["m2"], ["m2", "m3"], ["m2", "m3", "m4"], ["abs_lam_mean"], ["Cii_mean"], [shape_col(df)]]
    if "C_w" in df: sets += [["m2", "C_w"], [shape_col(df), "C_w"]]
    for sc in scales:
        mc, r2, lin = f"MC_s{sc:g}", f"r2_s{sc:g}", f"MClin_s{sc:g}"
        blk = dict(MC_range=[float(df[mc].min()), float(df[mc].max())], MC_median=float(df[mc].median()),
                   r2_median=float(df[r2].median()), r2_p05_p95=[float(df[r2].quantile(.05)), float(df[r2].quantile(.95))],
                   theory=dict(r=pr(df[lin], df[mc]), MC_mean=float(df[mc].mean()), MC_lin_mean=float(df[lin].mean()),
                               median_abs_diff=float((df[mc] - df[lin]).abs().median()),
                               r_lin_m2=pr(df["m2"], df[lin])),
                   corr={p: pr(df[p], df[mc]) for p in preds},
                   R2={"+".join(s): r2_ols(df, mc, s) for s in sets},
                   partial_Cii_given_strength_density=partial_r(df["Cii_mean"].values, df[mc].values,
                                                                df[["strength_raw_mean", "density"]].values),
                   partial_m2_given_strength_density=partial_r(df["m2"].values, df[mc].values,
                                                               df[["strength_raw_mean", "density"]].values),
                   regime=regime_block(df, mc), lifespan=lifespan(df, mc))
        if "C_w" in df:
            blk["partial_Cw_given_m2"] = partial_r(df["C_w"].values, df[mc].values, df[["m2"]].values)
        blk["partial_m3_given_m2"] = partial_r(df["m3"].values, df[mc].values, df[["m2"]].values)
        if sc in surr: blk["surrogates"] = surrogate_block(surr[sc], df, sc)
        out[f"scale_{sc:g}"] = blk
    if len(scales) >= 2:
        out["MC_between_scales"] = pr(df[f"MC_s{scales[0]:g}"], df[f"MC_s{scales[1]:g}"])
    mn = manuscript_nulls(cache)
    if mn: out["manuscript_directed_nulls_scale1"] = mn
    if cache_check: out["reproduction_vs_cached_MC"] = cache_check
    return out


def f3(x): return "—" if x is None or not np.isfinite(x) else f"{x:+.3f}"


def summary_markdown(S, scales):
    st = S["structure"]
    L = [f"# Analytical chain — {S['n_subjects']} connectomes\n",
         "ESN: " + ", ".join(f"{k}={v}" for k, v in S["esn_config"].items()) + "\n",
         "## Step 1 · disparity → subgraph centrality",
         f"- identity Σᵢ s̃ᵢ²Yᵢ = Σₖ λₖ²: max relative error {S['identity_max_relerr']:.1e}",
         f"- C_ii vs 1 + ½ s̃ᵢ²Yᵢ (pooled nodes): R² = {st['step1_R2_order2']:.4f} (identity-line R² = {st['step1_R2_identity']:.4f}); "
         f"median within-subject r = {S['step1_median_subject_r']:.3f}",
         "## Step 2 · subgraph centrality ↔ Gramian (G_I)_ii",
         f"- pooled nodes: Pearson {st['step2_pearson_C_GI']:.3f}, Spearman {st['step2_spearman_C_GI']:.3f}; "
         f"median within subject: Pearson {S['step2_median_subject_pearson']:.3f}, Spearman {S['step2_median_subject_spearman']:.3f}",
         "## Step 3 · linear theory (closed form, same Win, ridge/n_train) vs simulated ESN",
         "| input scale | r(MC_lin, MC) | mean MC | mean MC_lin | median abs diff | r(⟨s̃²Y⟩, MC_lin) |", "|---|---|---|---|---|---|"]
    for s in scales:
        t = S[f"scale_{s:g}"]["theory"]
        L.append(f"| {s:g} | {t['r']['r']:.3f} | {t['MC_mean']:.2f} | {t['MC_lin_mean']:.2f} | {t['median_abs_diff']:.2f} | {t['r_lin_m2']['r']:.3f} |")
    L += ["\n## Step 4 · structure → MC (subject level)",
          "| predictor | " + " | ".join(f"r (scale {s:g})" for s in scales) + " |", "|---|" + "---|" * len(scales)]
    names = dict(m2="⟨s̃²Y⟩ = Σλ²/N", abs_lam_mean="⟨|λ|⟩ (Aceituno)", imc="Σλ²(1−λ^2τ)", Cii_mean="⟨C_ii⟩",
                 GI_mean="⟨(G_I)_ii⟩", m3="Σλ³/N (triangles)", m4="Σλ⁴/N", Ratio="R = Y/Y_null (manuscript)",
                 ratio_mean="⟨Υ/μ_null⟩ (node mean)", Ups_mean="⟨Υ⟩", Y_mean="⟨Y⟩", s2_mean="⟨s̃²⟩",
                 C_w="C_w (weighted clustering)", strength_raw_mean="raw strength", density="density")
    first = S[f"scale_{scales[0]:g}"]
    for p, nm in names.items():
        if p in first["corr"]:
            L.append(f"| {nm} | " + " | ".join(f"{S[f'scale_{s:g}']['corr'][p]['r']:+.3f}" for s in scales) + " |")
    L += ["\nR² of MC (OLS):", "| model | " + " | ".join(f"scale {s:g}" for s in scales) + " |", "|---|" + "---|" * len(scales)]
    for k in first["R2"]:
        L.append(f"| {k} | " + " | ".join(f"{S[f'scale_{s:g}']['R2'][k]:.3f}" for s in scales) + " |")
    L.append("\npartial r with MC given ⟨s̃²Y⟩: " + "; ".join(
        f"scale {s:g}: C_w {f3(S[f'scale_{s:g}'].get('partial_Cw_given_m2', {}).get('r'))}, "
        f"Σλ³ {f3(S[f'scale_{s:g}']['partial_m3_given_m2']['r'])}" for s in scales))
    if first["regime"]:
        L += [f"\n## Regime · development (age ≤ {S['age_split']:g}) vs aging",
              "| scale | regime | n | " + " | ".join(f"r({c}, MC)" for c in first["regime"]["development"]["r"]) +
              " | " + " | ".join(f"R² {k}" for k in first["regime"]["development"]["R2"]) + " |",
              "|---" * (3 + len(first["regime"]["development"]["r"]) + len(first["regime"]["development"]["R2"])) + "|"]
        for s in scales:
            for reg, b in S[f"scale_{s:g}"]["regime"].items():
                L.append(f"| {s:g} | {reg} | {b['n']} | " + " | ".join(f"{v['r']:+.3f}" for v in b["r"].values()) +
                         " | " + " | ".join(f"{v:.3f}" for v in b["R2"].values()) + " |")
    L.append("\n## Weight surrogates (symmetric) and the asymmetric control")
    for s in scales:
        sb = S[f"scale_{s:g}"].get("surrogates")
        if not sb: continue
        L += [f"\n**Input scale {s:g}** — {sb['n_subjects']} subjects × {sb['n_real']} realizations; "
              f"r(MC, MC_lin) over subject×model = {sb['r_MC_MClin_all']['r']:.3f}; "
              f"r(MC, Σλ²/N) over symmetric models = {sb['r_MC_m2_symmetric']['r']:.3f}; "
              f"real-subject fit MC = {sb['fit_real']['intercept']:.2f} + {sb['fit_real']['slope']:.1f}·⟨s̃²Y⟩",
              "| model | median ΔMC | Wilcoxon p | mean MC | mean MC_lin | Σλ²/N | MC predicted from Σλ²/N |",
              "|---|---|---|---|---|---|---|"]
        for m, b in sb["models"].items():
            p_txt = f"{b['wilcoxon_p']:.1e}" if "wilcoxon_p" in b else "—"
            L.append(f"| {m} | {f3(b.get('dMC_median'))} | {p_txt} | {b['MC_mean']:.2f} | "
                     f"{b['MC_lin_mean']:.2f} | {b['m2_mean']:.4f} | {b['MC_pred_from_m2']:.2f} |")
    if "manuscript_directed_nulls_scale1" in S:
        L.append("\nManuscript (directed Reshuffle / Broken Stick, input scale 1, cached): " + "; ".join(
            f"{m} {b['dMC_median']:+.3f} (p={b['wilcoxon_p']:.1e})" for m, b in S["manuscript_directed_nulls_scale1"].items()))
    L.append("\n## Input scale, linearity and lifespan")
    for s in scales:
        b = S[f"scale_{s:g}"]
        L.append(f"- scale {s:g}: MC {b['MC_range'][0]:.2f}–{b['MC_range'][1]:.2f} (median {b['MC_median']:.2f}); "
                 f"⟨r²⟩ median {b['r2_median']:.2e}")
        if b["lifespan"]:
            ls = b["lifespan"]
            L.append(f"  lifespan: r(age, MC) development {ls['r_age_development']['r']:+.3f}, aging "
                     f"{ls['r_age_aging']['r']:+.3f}; binned minimum at age ≈ {ls['age_of_min_binned']:.1f}")
    if "MC_between_scales" in S:
        c = S["MC_between_scales"]
        L.append(f"- MC(scale {scales[0]:g}) vs MC(scale {scales[1]:g}): Pearson {c['r']:.3f}, Spearman {c['rho']:.3f}")
    if "reproduction_vs_cached_MC" in S:
        c = S["reproduction_vs_cached_MC"]
        L.append(f"- reproduction of the cached manuscript MC: Pearson {c['r']:.5f}, max |ΔMC| {c['max_abs_diff']:.2e} (n={c['n']})")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- figure
# categorical slots of the dataviz reference palette (validated, light surface)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"
MODEL_COL = {"Real": INK, "Uniform": AQUA, "BrokenStick": RED, "Reshuffle": BLUE, "SignFlip": YELLOW, "RealAsym": MAGENTA}
MODEL_LAB = {"Real": "Real", "Uniform": "Uniform", "BrokenStick": "Broken Stick", "Reshuffle": "Reshuffle",
             "SignFlip": "Sign Flip", "RealAsym": "Real, asymmetric"}
REGIME_COL = {"development": ORANGE, "aging": VIOLET}       # as in the manuscript figure


def make_figure(df, nodes, scales, surr, path_stem):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    lin_sc = min(scales)                                    # the near-linear run
    scale_col = {s: (BLUE if s == lin_sc else MUTED) for s in scales}
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.8), constrained_layout=True)
    ax = ax.ravel()

    a = ax[0]
    a.scatter(nodes["approx2"], nodes["Cii"], s=4, color=BLUE, alpha=0.35, lw=0)
    lo, hi = nodes[["approx2", "Cii"]].min().min(), nodes[["approx2", "Cii"]].max().max()
    a.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
    a.set(xlabel=r"$1+\frac{1}{2}\tilde s_i^2Y_i$", ylabel=r"$C_{ii}=(e^{\tilde W})_{ii}$",
          title="(a) Step 1: disparity → $C_{ii}$")

    a = ax[1]
    a.scatter(nodes["Cii"], nodes["GI"], s=4, color=BLUE, alpha=0.35, lw=0)
    a.set_yscale("log")
    a.set(xlabel=r"$C_{ii}$", ylabel=r"$(G_I)_{ii}=[(I-\tilde W^2)^{-1}]_{ii}$", title="(b) Step 2: $C_{ii}$ ↔ Gramian")

    a = ax[2]
    for sc in sorted(scales, reverse=True):
        a.scatter(df[f"MClin_s{sc:g}"], df[f"MC_s{sc:g}"], s=5, color=scale_col[sc], alpha=0.5, lw=0,
                  label=f"input scale {sc:g}")
    lo = min(df[[f"MC_s{s:g}" for s in scales]].min().min(), df[f"MClin_s{lin_sc:g}"].min())
    hi = max(df[[f"MC_s{s:g}" for s in scales]].max().max(), df[f"MClin_s{lin_sc:g}"].max())
    a.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
    a.set(xlabel="MC, linear theory (closed form)", ylabel="MC, simulated ESN", title="(c) Step 3: theory vs simulation")
    a.legend(frameon=False, markerscale=2)

    a = ax[3]
    for sc in sorted(scales, reverse=True):
        a.scatter(df["m2"], df[f"MC_s{sc:g}"], s=5, color=scale_col[sc], alpha=0.5, lw=0, label=f"input scale {sc:g}")
    a.set(xlabel=r"$\langle\tilde s^2Y\rangle=\sum_k\lambda_k^2/N$", ylabel="MC", title="(d) Step 4: structure → MC")
    a.legend(frameon=False, markerscale=2)

    a = ax[4]
    a.scatter(df["m2"], df[f"MC_s{lin_sc:g}"], s=4, color=GRID, lw=0, zorder=1)
    b, c0 = np.polyfit(df["m2"], df[f"MC_s{lin_sc:g}"], 1)
    xs = np.linspace(df["m2"].min(), df["m2"].max(), 2)
    a.plot(xs, c0 + b * xs, color=INK2, lw=1, ls="--", zorder=2)
    if lin_sc in surr:
        sm = surr[lin_sc].groupby(["subject_id", "model"])[["MC", "m2"]].mean().reset_index()
        for m in [x for x in MODELS if x in set(sm["model"])]:
            g = sm[sm["model"] == m]
            a.scatter(g["m2"], g["MC"], s=9, color=MODEL_COL[m], alpha=0.35, lw=0, zorder=3)
            a.scatter(g["m2"].mean(), g["MC"].mean(), s=60, color=MODEL_COL[m], edgecolor="white", lw=1.5, zorder=4)
            a.annotate(MODEL_LAB[m], (g["m2"].mean(), g["MC"].mean()), xytext=(6, -3), textcoords="offset points",
                       fontsize=7, color=INK, zorder=5)
    a.set(xlabel=r"$\sum_k\lambda_k^2/N$", ylabel=f"MC (input scale {lin_sc:g})",
          title="(e) Surrogates on the real-network line")

    a = ax[5]
    reg = regime_r(df, f"MC_s{lin_sc:g}")
    if reg is not None:
        labels, keys = reg
        y = np.arange(len(keys)); h = 0.38
        for j, (name, col) in enumerate(REGIME_COL.items()):
            vals = [labels[name][k] for k in keys]
            a.barh(y + (h / 2 if j == 0 else -h / 2), vals, h, color=col, label=name)
        a.axvline(0, color=INK2, lw=0.7)
        a.set_yticks(y); a.set_yticklabels([REG_LAB.get(k, k) for k in keys])
        a.invert_yaxis()
        a.set(xlabel=f"r(·, MC), input scale {lin_sc:g}", title="(f) Regime: shape vs scale")
        a.legend(frameon=False)
    for a in ax:
        a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    fig.savefig(f"{path_stem}.png", dpi=200); fig.savefig(f"{path_stem}.pdf")
    plt.close(fig)


REG_LAB = {"Ratio": r"$R=Y/Y_{null}$ (shape)", "ratio_mean": r"$\langle\Upsilon/\mu_{null}\rangle$ (shape)",
           "s2_mean": r"$\langle\tilde s^2\rangle$ (scale)", "m2": r"$\langle\tilde s^2Y\rangle$ (order 2)",
           "C_w": r"$C_w$ (clustering)"}


def regime_r(df, mc):
    if "age" not in df: return None
    keys = [k for k in [shape_col(df), "s2_mean", "m2", "C_w"] if k in df]
    out = {}
    for name, sub in [("development", df[df["age"] <= AGE_SPLIT]), ("aging", df[df["age"] > AGE_SPLIT])]:
        out[name] = {k: pr(sub[k], sub[mc])["r"] for k in keys}
    return out, keys


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="export folder (chain/export_data.py) or data_with_metrics.pkl")
    src.add_argument("--synthetic", type=int, metavar="S", help="test the pipeline on S synthetic connectomes")
    ap.add_argument("--out", default=str(REPO / "chain" / "results"))
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 1e-5])
    ap.add_argument("--subset", type=int, default=0, help="run on a random subset of this many subjects")
    ap.add_argument("--cache", default=None, help="folder with cached notebook results (default: <data>/cache)")
    ap.add_argument("--surr-n", type=int, default=100, help="surrogate subset size if no cached subset is found")
    ap.add_argument("--surr-real", type=int, default=5, help="surrogate realizations per subject (notebook: 5)")
    ap.add_argument("--skip-surrogates", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--eigen", default=None)
    ap.add_argument("--skip-esn", action="store_true", help="reuse mc_scale*.csv / surr_scale*.csv already in <out>/work")
    args = ap.parse_args()

    out = Path(args.out); work = out / "work"; work.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else (Path(args.data) / "cache" if args.data and Path(args.data).is_dir() else None)
    print("loading data ...", flush=True)
    if args.synthetic:
        sid, mats, meta = synthetic(args.synthetic)
    elif str(args.data).endswith(".pkl"):
        sid, mats, meta = load_pickle(args.data)
    else:
        sid, mats, meta = load_export(args.data)
    if args.subset and args.subset < len(sid):
        pick = np.sort(np.random.default_rng(0).choice(len(sid), args.subset, replace=False))
        sid, mats = sid[pick], mats[pick]
    print(f"  {len(sid)} connectomes, N = {mats.shape[1]}", flush=True)
    if mats.shape[1] != ESN_CFG["reservoir_size"]:
        ESN_CFG["reservoir_size"] = int(mats.shape[1])

    print("structural chain ...", flush=True)
    struct, nodes, pooled = run_structure(sid, mats, ESN_CFG["spectral_radius"], ESN_CFG["tau"])
    if not struct["symmetric"].all():
        print(f"  WARNING: {int((~struct['symmetric']).sum())} non-symmetric connectomes were symmetrised for the structural chain")

    # surrogate subset: the notebook's (cached ids) when available, else a random one
    surr_ids = None
    if not args.skip_surrogates:
        old = read_cache_table(cache, "mc_surrogates")
        if old is not None and set(old["subject_id"]).issubset(set(sid)):
            surr_ids = np.sort(old["subject_id"].unique()); print(f"  surrogate subset: {len(surr_ids)} cached ids")
        else:
            surr_ids = np.sort(np.random.default_rng(1).choice(sid, min(args.surr_n, len(sid)), replace=False))
            print(f"  surrogate subset: {len(surr_ids)} random subjects")

    csv, csv_s = work / "connectomes.csv", work / "connectomes_surr.csv"
    if not args.skip_esn:
        print("writing connectomes for the C++ engine ...", flush=True)
        write_connectomes_csv(csv, sid, mats)
        eig = find_eigen(args.eigen)
        exe = build("esn_mc", eig)
        if surr_ids is not None:
            idx = np.searchsorted(sid, surr_ids) if np.all(np.diff(sid) > 0) else [int(np.where(sid == s)[0][0]) for s in surr_ids]
            write_connectomes_csv(csv_s, sid[idx], mats[idx])
            exe_s = build("esn_surrogates", eig)
    res, surr = [], {}
    for sc in args.scales:
        if args.skip_esn:
            res.append(pd.read_csv(work / f"mc_scale{sc:g}.csv").rename(columns=mc_columns(sc)))
            if surr_ids is not None and (work / f"surr_scale{sc:g}.csv").exists():
                surr[sc] = pd.read_csv(work / f"surr_scale{sc:g}.csv")
        else:
            res.append(run_esn(exe, csv, work, sc, args.threads))
            if surr_ids is not None:
                surr[sc] = run_surrogates(exe_s, csv_s, work, sc, args.threads, args.surr_real)
    df = struct.copy()
    for r in res: df = df.merge(r, on="sid", how="left")
    meta_cols = [c for c in meta.columns if c not in df.columns or c == "sid"]
    df = df.merge(meta[meta_cols], on="sid", how="left")

    cache_check = None
    if 1.0 in args.scales:
        old = read_cache_table(cache, "mc_main")
        if old is not None:
            old = old.rename(columns={"subject_id": "sid", "MC_Glob": "MC"})[["sid", "MC"]]
            m = df[["sid", "MC_s1"]].merge(old, on="sid")
            cache_check = dict(**pr(m["MC_s1"], m["MC"]), max_abs_diff=float((m["MC_s1"] - m["MC"]).abs().max()))

    S = summarize(df, args.scales, pooled, cache_check, surr, cache)
    df.to_csv(out / "chain_subject.csv", index=False)
    nodes.to_csv(out / "chain_nodes_sample.csv", index=False)
    for sc, t in surr.items(): t.to_csv(out / f"chain_surrogates_scale{sc:g}.csv", index=False)
    (out / "chain_summary.json").write_text(json.dumps(S, indent=2))
    (out / "chain_summary.md").write_text(summary_markdown(S, args.scales))
    make_figure(df, nodes, args.scales, surr, str(out / "fig_chain"))
    print("\n" + summary_markdown(S, args.scales))
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
