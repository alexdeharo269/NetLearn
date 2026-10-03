#!/usr/bin/env python3
"""
Analytical chain  disparity -> subgraph centrality -> Gramian -> memory capacity,
evaluated on the BASE model (the real, symmetric connectomes; no surrogates).

For every connectome W (diagonal zeroed, rescaled to spectral radius rho, W~ = rho W / rho(W)):

  step 1  C_ii = (e^W~)_ii = 1 + (W~^2)_ii/2 + ...,   (W~^2)_ii = s~_i^2 Y_i       (node level)
          sum_i s~_i^2 Y_i = Tr W~^2 = sum_k lambda_k^2                          (exact)
  step 2  (G_I)_ii = [(I - W~^2)^-1]_ii = 1 + s~_i^2 Y_i + ...   (Gramian, B = I)
  step 3  linear-ESN memory: sum_k lambda_k^2 (1 - lambda_k^(2 tau)) (independent-mode
          estimate), <C_ii>, <(G_I)_ii>, against the simulated MC
  step 4  tanh ESN at the manuscript input scale (1) and in the near-linear regime
          (0.1), with the mean squared state <r^2> as linearity diagnostic

The ESN runs use ESNcpp/esn_mc with the manuscript hyper-parameters (RES + MAIN of
ESNcpp/TFM_closing_figures.ipynb), so MC at input_scale=1 reproduces the paper's MC.

Usage (from the repo root):
    python chain/chain_analysis.py --data ../NetLearn-data            # exported data
    python chain/chain_analysis.py --data procdata/data_with_metrics.pkl
    python chain/chain_analysis.py --synthetic 60 --out chain/results_synthetic   # pipeline test
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
ESN_DIR = REPO / "ESNcpp"
# Manuscript ESN hyper-parameters = RES + MAIN in ESNcpp/TFM_closing_figures.ipynb
ESN_CFG = dict(reservoir_size=90, spectral_radius=0.99, ridge=1e-6, train_ratio=0.7,
               washout=1000, steps=10000, tau=20, win_reps=1, seed=42)
AGE_SPLIT = 32.0                      # development: age <= 32 ; aging: age > 32 (as in the notebook)


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
        identity_relerr=float(abs(sY.sum() - lam2.sum()) / lam2.sum()),
        stable_rank=float((As ** 2).sum() / sr ** 2),
        imc=float((lam2 * (1 - lam2 ** tau)).sum()),           # sum lam^2 (1 - lam^(2 tau))
        Cii_mean=float(Cii.mean()), GI_mean=float(GI.mean()),
        abs_lam_mean=float(np.abs(lam).mean()),                 # Aceituno et al. 2020 proxy
        Y_mean=float(np.nanmean(Y)), Ups_mean=float(np.nanmean(k * Y)), ratio_mean=float(np.nanmean(ratio)),
        s_tilde_mean=float(s.mean()), strength_raw_mean=float(As.sum(1).mean()),
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
              str(ESN_DIR / "eigen-master"), "/usr/local/include/eigen3", "/opt/homebrew/include/eigen3"]:
        if p and (Path(p) / "Eigen" / "Dense").exists():
            return p
    raise RuntimeError("Eigen not found: install it (apt install libeigen3-dev / brew install eigen) "
                       "or pass --eigen /path/to/eigen")


def build_esn_mc(eigen):
    exe = ESN_DIR / "esn_mc"
    src = [ESN_DIR / "esn_mc.cpp", ESN_DIR / "esn.hpp"]
    if exe.exists() and exe.stat().st_mtime > max(f.stat().st_mtime for f in src):
        return exe
    cmd = [os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-fopenmp", f"-I{eigen}", "-w",
           str(ESN_DIR / "esn_mc.cpp"), "-o", str(exe)]
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


def run_esn(exe, csv, workdir, scale, threads):
    out_csv = workdir / f"mc_scale{scale:g}.csv"
    cfg = workdir / f"cfg_scale{scale:g}.txt"
    cfg.write_text("".join(f"{k}={v}\n" for k, v in {
        **ESN_CFG, "input_scale": scale, "log_state": 1, "threads": threads,
        "in_csv": str(csv), "out_csv": str(out_csv)}.items()))
    t0 = time.time()
    subprocess.run([str(exe), str(cfg)], check=True)
    print(f"  esn_mc input_scale={scale:g} done in {time.time() - t0:.0f}s", flush=True)
    return pd.read_csv(out_csv).rename(columns={"subject_id": "sid", "MC_Glob": f"MC_s{scale:g}",
                                                "r2_mean": f"r2_s{scale:g}"})


# ---------------------------------------------------------------------------- stats
def pr(x, y):
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


def lifespan(df, col):
    if "age" not in df: return {}
    dev, agi = df[df["age"] <= AGE_SPLIT], df[df["age"] > AGE_SPLIT]
    q = pd.qcut(df["age"], 24, duplicates="drop")
    traj = df.groupby(q, observed=True).agg(age=("age", "mean"), mc=(col, "mean"))
    return dict(r_age_development=pr(dev["age"].values, dev[col].values),
                r_age_aging=pr(agi["age"].values, agi[col].values),
                age_of_min_binned=float(traj.loc[traj["mc"].idxmin(), "age"]))


def summarize(df, scales, pooled, cache_check):
    out = dict(n_subjects=int(len(df)), esn_config=ESN_CFG, scales=scales, structure=pooled,
               identity_max_relerr=float(df["identity_relerr"].max()),
               step1_median_subject_r=float(df["r_C_order2"].median()),
               step2_median_subject_pearson=float(df["r_C_GI"].median()),
               step2_median_subject_spearman=float(df["rho_C_GI"].median()))
    preds = ["m2", "imc", "Cii_mean", "GI_mean", "abs_lam_mean", "ratio_mean", "Ups_mean",
             "strength_raw_mean", "density"]
    for sc in scales:
        mc, r2 = f"MC_s{sc:g}", f"r2_s{sc:g}"
        blk = dict(MC_range=[float(df[mc].min()), float(df[mc].max())], MC_median=float(df[mc].median()),
                   r2_median=float(df[r2].median()), r2_p05_p95=[float(df[r2].quantile(.05)), float(df[r2].quantile(.95))],
                   corr={p: pr(df[p].values, df[mc].values) for p in preds},
                   partial_Cii_given_strength_density=partial_r(df["Cii_mean"].values, df[mc].values,
                                                                df[["strength_raw_mean", "density"]].values),
                   partial_m2_given_strength_density=partial_r(df["m2"].values, df[mc].values,
                                                               df[["strength_raw_mean", "density"]].values),
                   lifespan=lifespan(df, mc))
        out[f"scale_{sc:g}"] = blk
    if len(scales) >= 2:
        a, b = f"MC_s{scales[0]:g}", f"MC_s{scales[1]:g}"
        out["step4_MC_between_scales"] = pr(df[a].values, df[b].values)
    if cache_check: out["reproduction_vs_cached_MC"] = cache_check
    return out


def summary_markdown(S, scales):
    L = [f"# Analytical chain on the base model — {S['n_subjects']} connectomes\n",
         "ESN: " + ", ".join(f"{k}={v}" for k, v in S["esn_config"].items()) + "\n",
         "## Step 1 · disparity → subgraph centrality",
         f"- identity Σ s̃²Y = Σ λ²: max relative error {S['identity_max_relerr']:.1e}",
         f"- order-2 reconstruction of C_ii (pooled nodes): R² = {S['structure']['step1_R2_order2']:.4f} "
         f"(identity-line R² = {S['structure']['step1_R2_identity']:.4f})",
         "## Step 2 · subgraph centrality ↔ Gramian (G_I)_ii",
         f"- pooled nodes: Pearson {S['structure']['step2_pearson_C_GI']:.3f}, Spearman {S['structure']['step2_spearman_C_GI']:.3f}; "
         f"median within subject: Pearson {S['step2_median_subject_pearson']:.3f}, Spearman {S['step2_median_subject_spearman']:.3f}",
         "## Steps 3–4 · structure → MC (subject level)",
         "| predictor | " + " | ".join(f"r (scale {s:g})" for s in scales) + " |",
         "|---|" + "---|" * len(scales)]
    names = dict(m2="⟨s̃²Y⟩ = Σλ²/N", imc="Σλ²(1−λ^2τ)", Cii_mean="⟨C_ii⟩", GI_mean="⟨(G_I)_ii⟩",
                 abs_lam_mean="⟨|λ|⟩ (Aceituno)", ratio_mean="⟨Υ/Υ_null⟩", Ups_mean="⟨Υ⟩",
                 strength_raw_mean="raw strength", density="density")
    for p, nm in names.items():
        L.append(f"| {nm} | " + " | ".join(f"{S[f'scale_{s:g}']['corr'][p]['r']:+.3f}" for s in scales) + " |")
    for s in scales:
        b = S[f"scale_{s:g}"]
        L += [f"\n**Input scale {s:g}**: MC {b['MC_range'][0]:.2f}–{b['MC_range'][1]:.2f} (median {b['MC_median']:.2f}); "
              f"⟨r²⟩ median {b['r2_median']:.4f} (5–95%: {b['r2_p05_p95'][0]:.4f}–{b['r2_p05_p95'][1]:.4f}); "
              f"partial r(⟨C_ii⟩, MC | strength, density) = {b['partial_Cii_given_strength_density']['r']:+.3f}"]
        if b["lifespan"]:
            ls = b["lifespan"]
            L.append(f"lifespan: r(age, MC) development {ls['r_age_development']['r']:+.3f}, aging "
                     f"{ls['r_age_aging']['r']:+.3f}; binned minimum at age ≈ {ls['age_of_min_binned']:.1f}")
    if "step4_MC_between_scales" in S:
        c = S["step4_MC_between_scales"]
        L.append(f"\n**Step 4**: MC(scale {scales[0]:g}) vs MC(scale {scales[1]:g}): Pearson {c['r']:.3f}, Spearman {c['rho']:.3f}")
    if "reproduction_vs_cached_MC" in S:
        c = S["reproduction_vs_cached_MC"]
        L.append(f"\n**Reproduction** of the cached manuscript MC: Pearson {c['r']:.5f}, max |ΔMC| {c['max_abs_diff']:.2e} (n={c['n']})")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- figure
BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical slots 1-2 (dataviz palette)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def make_figure(df, nodes, scales, path_stem):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    ages = LinearSegmentedColormap.from_list("age", ["#9ec5f4", "#2a78d6", "#0d366b"])
    cols = [BLUE, ORANGE]
    fig, ax = plt.subplots(1, 5, figsize=(15, 3.2), constrained_layout=True)

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
    for sc, c in zip(scales, cols):
        a.scatter(df["m2"], df[f"MC_s{sc:g}"], s=5, color=c, alpha=0.45, lw=0, label=f"input scale {sc:g}")
    a.set(xlabel=r"$\langle\tilde s^2Y\rangle=\sum_k\lambda_k^2/N$", ylabel="MC", title="(c) Steps 3–4: structure → MC")
    a.legend(frameon=False, markerscale=2)

    a = ax[3]
    if len(scales) >= 2:
        x, y = df[f"MC_s{scales[0]:g}"], df[f"MC_s{scales[1]:g}"]
        sc_ = a.scatter(x, y, s=5, c=df["age"] if "age" in df else BLUE, cmap=ages, lw=0, alpha=0.7)
        if "age" in df: fig.colorbar(sc_, ax=a, label="age (years)", shrink=0.85)
        a.set(xlabel=f"MC (input scale {scales[0]:g})", ylabel=f"MC (input scale {scales[1]:g})",
              title="(d) Step 4: robustness to input scale")

    a = ax[4]
    if "age" in df:
        q = pd.qcut(df["age"], 24, duplicates="drop")
        for sc, c in zip(scales, cols):
            z = (df[f"MC_s{sc:g}"] - df[f"MC_s{sc:g}"].mean()) / df[f"MC_s{sc:g}"].std()
            t = pd.DataFrame({"age": df["age"], "z": z}).groupby(q, observed=True).mean()
            a.plot(t["age"], t["z"], "-o", color=c, lw=2, ms=4, label=f"input scale {sc:g}")
        a.axvline(AGE_SPLIT, color=INK2, lw=0.8, ls=":")
        a.set(xlabel="age (years)", ylabel="MC (z-score, binned)", title="(e) Lifespan trajectory")
        a.legend(frameon=False)
    for a in ax:
        a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    fig.savefig(f"{path_stem}.png", dpi=200); fig.savefig(f"{path_stem}.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="export folder (chain/export_data.py) or data_with_metrics.pkl")
    src.add_argument("--synthetic", type=int, metavar="S", help="test the pipeline on S synthetic connectomes")
    ap.add_argument("--out", default=str(REPO / "chain" / "results"))
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.1])
    ap.add_argument("--subset", type=int, default=0, help="run on a random subset of this many subjects")
    ap.add_argument("--cache", default=None, help="folder with mc_main.csv/pkl to check reproduction (default: <data>/cache)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--eigen", default=None)
    ap.add_argument("--skip-esn", action="store_true", help="reuse mc_scale*.csv already in <out>/work")
    args = ap.parse_args()

    out = Path(args.out); work = out / "work"; work.mkdir(parents=True, exist_ok=True)
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

    csv = work / "connectomes.csv"
    if not args.skip_esn:
        print("writing connectomes for the C++ engine ...", flush=True)
        write_connectomes_csv(csv, sid, mats)
        exe = build_esn_mc(find_eigen(args.eigen))
    res = []
    for sc in args.scales:
        if args.skip_esn:
            r = pd.read_csv(work / f"mc_scale{sc:g}.csv").rename(
                columns={"subject_id": "sid", "MC_Glob": f"MC_s{sc:g}", "r2_mean": f"r2_s{sc:g}"})
        else:
            r = run_esn(exe, csv, work, sc, args.threads)
        res.append(r)
    df = struct.copy()
    for r in res: df = df.merge(r, on="sid", how="left")
    meta_cols = [c for c in meta.columns if c not in df.columns or c == "sid"]
    df = df.merge(meta[meta_cols], on="sid", how="left")

    cache_check = None
    cache = Path(args.cache) if args.cache else (Path(args.data) / "cache" if args.data and Path(args.data).is_dir() else None)
    if cache and 1.0 in args.scales:
        f = next((p for p in [cache / "mc_main.csv", cache / "mc_main.pkl"] if p.exists()), None)
        if f is not None:
            old = pd.read_csv(f) if f.suffix == ".csv" else pd.read_pickle(f)
            old = old.rename(columns={"subject_id": "sid", "MC_Glob": "MC"})[["sid", "MC"]]
            m = df[["sid", "MC_s1"]].merge(old, on="sid")
            cache_check = dict(**pr(m["MC_s1"].values, m["MC"].values),
                               max_abs_diff=float((m["MC_s1"] - m["MC"]).abs().max()))

    S = summarize(df, args.scales, pooled, cache_check)
    df.to_csv(out / "chain_subject.csv", index=False)
    nodes.to_csv(out / "chain_nodes_sample.csv", index=False)
    (out / "chain_summary.json").write_text(json.dumps(S, indent=2))
    (out / "chain_summary.md").write_text(summary_markdown(S, args.scales))
    make_figure(df, nodes, args.scales, str(out / "fig_chain"))
    print("\n" + summary_markdown(S, args.scales))
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
