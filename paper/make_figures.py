#!/usr/bin/env python3
"""
Figures of the v2 manuscript (main: fig2_order2, fig4_memory, fig5_lifespan) and of its
Supplementary Information (figS_*), drawn from paper/data only (run paper/paper_analysis.py
first). Each figure is one function, so single panels can be edited and re-drawn quickly:

    python paper/make_figures.py                 # all figures -> paper/figures/*.pdf, *.png
    python paper/make_figures.py fig4_memory     # one figure

Figures 1 (atlas and ages) and 3 (memory paradigm) of the manuscript are unchanged
(plots/figdata_atlas_age.pdf, plots/trace_fit.pdf).
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "figures"
MC, MCL = "MC_s1e-05", "MClin_s1e-05"
N_NODES, AGE_SPLIT = 90, 32.0

# cohort colours of the manuscript figures (matplotlib tab10 order, purple skipped)
COHORTS = {1: ("dHCP", "#1f77b4"), 2: ("BCP", "#ff7f0e"), 3: ("CALM", "#2ca02c"), 4: ("RED", "#d62728"),
           5: ("ACE", "#8c564b"), 6: ("HCPd", "#e377c2"), 7: ("HCPya", "#7f7f7f"), 8: ("camCAN", "#bcbd22"),
           9: ("HCPa", "#17becf")}
# surrogate colours (Okabe-Ito), real network in black
MODELS = {"Real": ("Real", "#000000"), "Uniform": ("Uniform", "#E69F00"), "BrokenStick": ("Broken stick", "#56B4E9"),
          "Reshuffle": ("Reshuffle", "#009E73"), "SignFlip": ("Sign flip", "#CC79A7"),
          "RealAsym": ("Directed control", "#D55E00")}
EXAMPLES = {"p5": "#9ecae1", "p50": "#4292c6", "p95": "#08306b"}          # one hue, light -> dark
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
W2, W1 = 7.0, 3.42                                                        # PNAS widths (inches)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "savefig.dpi": 300, "pdf.fonttype": 42, "legend.frameon": False,
    "mathtext.fontset": "cm",
})


# ------------------------------------------------------------------------------ data
def load():
    D = dict(s=pd.read_csv(DATA / "subject_v2.csv"), eta=pd.read_csv(DATA / "eta_sweep.csv"),
             ex=pd.read_csv(DATA / "examples.csv"), cur=pd.read_csv(DATA / "examples_curves.csv"),
             nodes=pd.read_csv(DATA / "nodes_v2.csv"), surr=pd.read_csv(DATA / "surrogates.csv"),
             N=json.loads((DATA / "numbers.json").read_text()))
    D["ridge"] = pd.read_csv(DATA / "ridge_sweep.csv") if (DATA / "ridge_sweep.csv").exists() else None
    D["s"] = D["s"].dropna(subset=[MC])
    return D


# ---------------------------------------------------------------------------- helpers
def label(ax, letter, x=-0.2, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left", color=INK)


def note(ax, text, x=0.04, y=0.96, ha="left", va="top"):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=6, ha=ha, va=va, color=INK)


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def cohort_legend(ax, loc="lower right", ncol=1, codes=None):
    codes = codes or list(COHORTS)
    h = [Line2D([], [], marker="o", ls="", ms=3.5, mfc=COHORTS[c][1], mec="white", mew=0.3) for c in codes]
    ax.legend(h, [COHORTS[c][0] for c in codes], loc=loc, ncol=ncol, handletextpad=0.1, columnspacing=0.6,
              borderaxespad=0.2, labelspacing=0.25)


def scatter_cohorts(ax, d, x, y, s=2.5, alpha=0.55):
    for c, (nm, col) in COHORTS.items():
        g = d[d["dataset"] == c]
        ax.scatter(g[x], g[y], s=s, color=col, alpha=alpha, lw=0, rasterized=True)


def age_means(ax, d, col, ms=2.6):
    """Mean of `col` per cohort and age (1-y bins below 40 y, 3-y bins above), as in the v1 figures."""
    for c, (nm, colr) in COHORTS.items():
        g = d[d["dataset"] == c]
        if g.empty: continue
        a = g["age"].values
        b = np.where(a < 40, np.floor(a), 40 + 3 * np.floor((a - 40) / 3) + 1)
        m = g.groupby(b)[col].mean()
        ax.plot(m.index, m.values, "-o", color=colr, ms=ms, lw=0.7, mec="white", mew=0.25)
    ax.axvline(AGE_SPLIT, color=MUTED, lw=0.6, ls=":", zorder=0)


def rank_axis(ax):
    """Log x-axis for the stable rank with plain tick labels."""
    ax.set_xscale("log")
    ax.set_xticks([3, 4, 5, 6, 8, 10]); ax.set_xticklabels(["3", "4", "5", "6", "8", "10"])
    ax.xaxis.set_minor_formatter(NullFormatter())


def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return np.corrcoef(x[m], y[m])[0, 1]


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  ", name)


def binned(x, y, edges):
    """Median and interquartile range of y in bins of x."""
    idx = np.digitize(x, edges) - 1
    rows = []
    for b in range(len(edges) - 1):
        v = y[idx == b]
        if len(v) >= 20:
            rows.append((0.5 * (edges[b] + edges[b + 1]), *np.percentile(v, [25, 50, 75])))
    return np.array(rows)


def null_upsilon(k):
    mu = 2 * k / (k + 1)
    var = k ** 2 * ((20 + 4 * k) / ((k + 1) * (k + 2) * (k + 3)) - 4 / (k + 1) ** 2)
    return mu, np.sqrt(np.clip(var, 0, None))


# ------------------------------------------------------------------- main figure 2
def fig2_order2(D):
    """From local disparity to the spectrum."""
    s, nd = D["s"], D["nodes"].dropna()
    fig, axs = plt.subplots(1, 4, figsize=(W2, 1.85), gridspec_kw=dict(wspace=0.55))

    ax = axs[0]                                         # (a) disparity against degree
    ax.scatter(nd["k"], nd["Ups"], s=1.2, color="#bdbdbd", lw=0, alpha=0.5, rasterized=True)
    k = np.arange(2, int(nd["k"].max()) + 1)
    mu, sd = null_upsilon(k.astype(float))
    ax.fill_between(k, mu - 2 * sd, mu + 2 * sd, color="#9e9e9e", alpha=0.35, lw=0, label="null $\\pm2\\sigma$")
    ax.plot(k, mu, color=MUTED, lw=0.8)
    ax.plot(k, k, "--", color="#d62728", lw=0.8, label="$\\Upsilon=k$")
    e = np.unique(np.round(np.logspace(np.log10(3), np.log10(nd["k"].max()), 14)))
    bm = binned(nd["k"].values, nd["Ups"].values, e)
    ax.plot(bm[:, 0], bm[:, 2], color="#1f4e79", lw=1.4, label="median")
    ax.set(xscale="log", yscale="log", xlabel="degree $k$", ylabel="disparity $\\Upsilon=kY$")
    ax.legend(loc="upper left", handlelength=1.4)
    label(ax, "a", x=-0.32)

    ax = axs[1]                                         # (b) accuracy of the truncations by degree
    rel2 = (nd["Cii"] - 1 - nd["T2"]) / (nd["Cii"] - 1)
    rel3 = (nd["Cii"] - 1 - nd["T2"] - nd["T3"]) / (nd["Cii"] - 1)
    edges = np.arange(0, nd["k"].max() + 8, 8)
    for rel, nm, col in [(rel2, "order 2", "#1f4e79"), (rel3, "order 3", "#6baed6")]:
        bm = binned(nd["k"].values, 100 * rel.values, edges)
        ax.fill_between(bm[:, 0], bm[:, 1], bm[:, 3], color=col, alpha=0.25, lw=0)
        ax.plot(bm[:, 0], bm[:, 2], "-o", color=col, ms=2.5, lw=1.0, label=nm)
    ax.set(xlabel="degree $k$", ylabel="error in $C_{ii}-1$ (%)", ylim=(0, None))
    grid(ax, "y"); ax.legend(loc="upper left")
    label(ax, "b", x=-0.3)

    ax = axs[2]                                         # (c) subgraph centrality and stable rank
    ax.scatter(s["stable_rank"], s["Cii_mean"] - 1, s=2, color="#4d4d4d", alpha=0.4, lw=0, rasterized=True)
    x = np.linspace(s["stable_rank"].min(), s["stable_rank"].max(), 10)
    ax.plot(x, 0.9801 * x / (2 * N_NODES), color="#d62728", lw=0.8, ls="--", label="order-2 term")
    ax.set(xlabel="stable rank $r_s$", ylabel="$\\langle C_{ii}\\rangle-1$")
    note(ax, f"$r={corr(s['stable_rank'].values, s['Cii_mean'].values):.3f}$")
    ax.legend(loc="lower right")
    label(ax, "c", x=-0.32)

    ax = axs[3]                                         # (d) local and global heterogeneity
    sc = ax.scatter(s["Ratio"], s["s2t"], c=s["stable_rank"], cmap="Blues", s=2, lw=0, rasterized=True,
                    vmin=s["stable_rank"].quantile(0.01) - 1.5, vmax=s["stable_rank"].quantile(0.99))
    ax.set(xlabel="local: $\\langle\\Upsilon/\\Upsilon_{null}\\rangle$", ylabel="global: $\\langle\\tilde s^{2}\\rangle$")
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02); cb.set_label("$r_s$", fontsize=6)
    cb.ax.tick_params(labelsize=5); cb.outline.set_linewidth(0.4)
    label(ax, "d", x=-0.32)
    save(fig, "fig2_order2")


# ------------------------------------------------------------------- main figure 4
def surrogate_means(D):
    sm = D["surr"].groupby(["subject_id", "model"])[["MC", "m2", "MC_lin"]].mean().reset_index()
    return sm.groupby("model").agg(MC=("MC", "mean"), MCsd=("MC", "std"), m2=("m2", "mean"), m2sd=("m2", "std"),
                                   MC_lin=("MC_lin", "mean"))


def fig4_memory(D):
    """Memory counts input directions above the noise floor; the stable rank sets the count."""
    s, N = D["s"], D["N"]
    fig = plt.figure(figsize=(W2, 3.9))
    gs = fig.add_gridspec(2, 6, hspace=0.62, wspace=1.6)
    axa, axb, axc = fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])
    axd, axe = fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])

    ax = axa                                            # (a) closed form against simulation
    ax.scatter(s[MCL], s[MC], s=2, color="#4d4d4d", alpha=0.4, lw=0, rasterized=True)
    lo, hi = s[[MC, MCL]].min().min() - 0.2, s[[MC, MCL]].max().max() + 0.2
    ax.plot([lo, hi], [lo, hi], color="#d62728", lw=0.8, ls="--")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="MC, closed form", ylabel="MC, simulated reservoir")
    note(ax, f"$r={N['theory']['r_sim_cpp']:.3f}$\n3,901 connectomes")
    label(ax, "a")

    ax = axb                                            # (b) input-Gramian spectra and the noise floor
    ex = D["ex"]
    for lab, g in ex.groupby("label"):
        g = g[g["k"] <= 22]
        mc = s.loc[s["sid"] == g["sid"].iloc[0], MC].iloc[0]
        nab = int((g["c_over_eta"] > 1).sum())
        ax.plot(g["k"], g["c_over_eta"], "-o", color=EXAMPLES[lab], ms=2.2, lw=0.9,
                label=f"MC = {mc:.1f}: {nab} directions")
    ax.axhline(1, color=INK, lw=0.8, ls="--")
    ax.text(22, 3, "noise floor $\\eta$", fontsize=6, ha="right", va="bottom", color=INK)
    ax.set(yscale="log", xlim=(0, 22.5), ylim=(1e-3, 1e12), xlabel="direction $k$ of the input Gramian",
           ylabel="variance $c_k/\\eta$")
    ax.set_yticks([1e-3, 1, 1e3, 1e6, 1e9, 1e12])
    ax.legend(loc="upper right", handlelength=1.2, fontsize=5.5)
    label(ax, "b")

    ax = axc                                            # (c) stable rank predicts memory
    scatter_cohorts(ax, s, "stable_rank", MC, s=2)
    x = np.linspace(s["stable_rank"].min(), s["stable_rank"].max(), 50)
    p = np.polyfit(np.log(s["stable_rank"]), s[MC], 1)
    ax.plot(x, np.polyval(p, np.log(x)), color=INK, lw=0.9)
    rank_axis(ax); ax.set(xlabel="stable rank $r_s$", ylabel="MC, simulated")
    r = N["stable_rank"]["r_MC"]
    note(ax, f"$r={r['all']:.2f}$ (dev. {r['dev']:.2f}, aging {r['aging']:.2f})", y=0.98)
    cohort_legend(ax, loc="lower right", ncol=2)
    ax.get_legend().set_bbox_to_anchor((1.04, -0.03))
    for t in ax.get_legend().get_texts(): t.set_fontsize(5)
    label(ax, "c")

    ax = axd                                            # (d) energy versus dimension
    T = ["1", "2", "3", "5", "10", "20", "$\\infty$"]
    rv = [N["gram_series_r_MC"][f"T{t}"] for t in [1, 2, 3, 5, 10, 20]] + [N["gram_series_r_MC"]["Tinf"]]
    xs = np.arange(len(T))
    ax.bar(xs, rv, width=0.6, color=["#1f4e79"] + ["#6baed6"] * 5 + ["#bdbdbd"], edgecolor="white", lw=0.5)
    for xi, v in zip(xs, rv):
        ax.text(xi, v + (0.03 if v >= 0 else -0.03), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=5.5, color=INK)
    ax.axhline(0, color=MUTED, lw=0.6)
    T[0], T[-1] = "1\n(stable rank)", "$\\infty$\n(avg. controllability)"
    ax.set_xticks(xs); ax.set_xticklabels(T)
    ax.set(ylim=(-0.3, 1.05), xlabel="terms $T$ of the Gramian series $S_T$", ylabel="$r$ with MC")
    grid(ax, "y")
    label(ax, "d", x=-0.12)

    ax = axe                                            # (e) surrogates on the stable-rank line
    ax.scatter(s["m2"], s[MC], s=1.5, color="#d9d9d9", lw=0, rasterized=True)
    x = np.linspace(0.025, 0.09, 10); L = N["surrogates"]["line"]
    ax.plot(x, L["intercept"] + L["slope"] * x, color=MUTED, lw=0.8)
    sm = surrogate_means(D)
    for m, (nm, col) in MODELS.items():
        if m not in sm.index: continue
        r_ = sm.loc[m]
        ax.plot(r_["m2"], r_["MC"], "o", ms=5, color=col, mec="white", mew=0.6, zorder=3)
        dx, dy, ha = {"Real": (-0.002, 0.45, "right"), "Uniform": (0.002, -0.4, "left"),
                      "BrokenStick": (0.002, -0.45, "left"), "Reshuffle": (0.0015, 0.55, "left"),
                      "SignFlip": (0.003, -0.45, "left"), "RealAsym": (0.003, 0.0, "left")}[m]
        ax.text(r_["m2"] + dx, r_["MC"] + dy, nm, fontsize=5.5, ha=ha, va="center", color=INK)
    ax.set(xlabel="$m_2=\\mathrm{Tr}\\,\\tilde W^2/N$ (stable rank / $N$)", ylabel="MC, simulated",
           xlim=(0.02, 0.095))
    label(ax, "e", x=-0.12)
    save(fig, "fig4_memory")


# ------------------------------------------------------------------- main figure 5
def fig5_lifespan(D):
    """Lifespan of memory, stable rank and the two heterogeneity factors."""
    s, N = D["s"], D["N"]
    fig, axs = plt.subplots(2, 3, figsize=(W2, 3.9), gridspec_kw=dict(hspace=0.55, wspace=0.45, top=0.9))
    panels = [(axs[0, 0], MC, "MC", "a"), (axs[0, 1], "stable_rank", "stable rank $r_s$", "b"),
              (axs[1, 0], "Ratio", "local: $\\langle\\Upsilon/\\Upsilon_{null}\\rangle$", "d"),
              (axs[1, 1], "s2t", "global: $\\langle\\tilde s^{2}\\rangle$", "e"),
              (axs[1, 2], "strength", "raw strength (streamlines)", "f")]
    for ax, col, yl, let in panels:
        ax.scatter(s["age"], s[col], s=1, color="#e0e0e0", lw=0, rasterized=True, zorder=1)
        age_means(ax, s, col)
        ax.set(xlabel="age (y)", ylabel=yl, xlim=(-3, 93))
        label(ax, let)
    ages = N["age_correlations"]
    for ax, col in [(axs[0, 0], MC), (axs[0, 1], "stable_rank"), (axs[1, 0], "Ratio"), (axs[1, 1], "s2t"),
                    (axs[1, 2], "strength")]:
        note(ax, f"$r_{{age}}$: {ages['dev'][col]:+.2f} | {ages['aging'][col]:+.2f}", x=0.98, y=0.98, ha="right")
    h = [Line2D([], [], marker="o", ls="-", lw=0.7, ms=3, mfc=c[1], color=c[1], mec="white", mew=0.3) for c in COHORTS.values()]
    fig.legend(h, [c[0] for c in COHORTS.values()], loc="upper center", ncol=9, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=1.0)

    ax = axs[0, 2]                                      # (c) camCAN alone
    cc = s[s["dataset"] == 8]
    bins = [18, 25, 32, 40, 50, 60, 70, 80, 90]
    g = cc.groupby(pd.cut(cc["age"], bins), observed=True)
    xm, ym, ye = g["age"].mean(), g[MC].mean(), g[MC].sem()
    ax.scatter(cc["age"], cc[MC], s=1.5, color="#e5e5a8", lw=0, rasterized=True)
    q = N["camcan"]["MC_quad"]
    xa = np.linspace(19, 89, 100)
    ax.axvspan(*q["ci"], color="#bcbd22", alpha=0.15, lw=0)
    ax.plot(xa, np.polyval(q["coef"], xa), color=INK, lw=0.9)
    ax.errorbar(xm, ym, yerr=ye, fmt="o", ms=3, color="#8c8d10", mec="white", mew=0.4, elinewidth=0.7)
    ax.axvline(q["vertex"], color=INK, lw=0.6, ls=":")
    ax.set(xlabel="age (y)", ylabel="MC (camCAN only)", xlim=(15, 92), ylim=(cc[MC].quantile(0.01), cc[MC].quantile(0.995)))
    ax.text(0.04, 0.04, f"single site, n = {len(cc)}\nminimum {q['vertex']:.1f} y (95% CI {q['ci'][0]:.0f}-{q['ci'][1]:.0f})",
            transform=ax.transAxes, fontsize=6, va="bottom", color=INK,
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.85))
    label(ax, "c")
    save(fig, "fig5_lifespan")


# ----------------------------------------------------------------- supplementary
def figS_theory(D):
    s, cur, N = D["s"], D["cur"], D["N"]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.0), gridspec_kw=dict(wspace=0.45))
    for ax, (x, y, ttl, let) in zip(axs[:2], [(MCL, MC, "$\\sigma_{in}=10^{-5}$ (linear)", "a"),
                                              ("MClin_s1", "MC_s1", "$\\sigma_{in}=1$ (nonlinear)", "b")]):
        ax.scatter(s[x], s[y], s=2, color="#4d4d4d", alpha=0.4, lw=0, rasterized=True)
        lo, hi = min(s[x].min(), s[y].min()) - 0.2, max(s[x].max(), s[y].max()) + 0.2
        ax.plot([lo, hi], [lo, hi], color="#d62728", lw=0.8, ls="--")
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="MC, closed form", ylabel="MC, simulated", title=ttl)
        note(ax, f"$r={corr(s[x].values, s[y].values):.3f}$")
        label(ax, let)
    ax = axs[2]
    for lab, g in cur.groupby("label"):
        mc = s.loc[s["sid"] == g["sid"].iloc[0], MC].iloc[0]
        ax.plot(g["tau"], g["m"], "-o", color=EXAMPLES[lab], ms=2.2, lw=0.9, label=f"MC = {mc:.1f}")
    ax.set(xlabel="delay $\\tau$", ylabel="$m(\\tau)$, closed form", ylim=(0, 1.05), xticks=[1, 5, 10, 15, 20])
    grid(ax, "y"); ax.legend(loc="upper right")
    label(ax, "c")
    save(fig, "figS_theory")


def figS_gramian(D):
    s, ex, N = D["s"], D["ex"], D["N"]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.0), gridspec_kw=dict(wspace=0.45))
    ax = axs[0]
    for lab, g in ex.groupby("label"):
        g = g[g["k"] <= 24]
        mc = s.loc[s["sid"] == g["sid"].iloc[0], MC].iloc[0]
        ax.plot(g["k"], g["c_over_eta"], "-o", color=EXAMPLES[lab], ms=2, lw=0.9, label=f"MC = {mc:.1f}")
    ax.axhline(1, color=INK, lw=0.8, ls="--")
    ax.axhspan(1e-8, 1e-4, color="#f0f0f0", lw=0, zorder=0)
    ax.text(23.5, 2e-6, "double-precision limit", fontsize=5.5, ha="right", va="center", color=MUTED)
    ax.set(yscale="log", xlabel="direction $k$", ylabel="$c_k/\\eta_0$", xlim=(0, 24), ylim=(1e-8, 1e12))
    ax.legend(loc="upper right"); label(ax, "a")
    ax = axs[1]
    scatter_cohorts(ax, s, "eff_rank", MC, s=2)
    ax.set(xlabel="effective rank $\\sum_k c_k/(c_k+\\eta_0)$", ylabel="MC, simulated")
    note(ax, f"$r={N['effective_rank']['r_MC']:.3f}$"); label(ax, "b")
    ax = axs[2]
    scatter_cohorts(ax, s, "stable_rank", "q_decay", s=2)
    rank_axis(ax); ax.set(xlabel="stable rank $r_s$", ylabel="decay ratio $q$")
    note(ax, f"$r={N['effective_rank']['r_q_rs']:.2f}$", x=0.96, y=0.04, ha="right", va="bottom")
    cohort_legend(ax, loc="upper left", ncol=2)
    label(ax, "c")
    save(fig, "figS_gramian")


def figS_noise(D):
    eta, ridge, N = D["eta"], D["ridge"], D["N"]
    s = D["s"][["sid", "stable_rank"]]
    fig, axs = plt.subplots(1, 2, figsize=(W1 * 1.6, 2.0), gridspec_kw=dict(wspace=0.45))
    g = eta.groupby("eta_rel")["MC_theory"]
    xs = np.array(sorted(eta["eta_rel"].unique()))
    q = g.quantile([0.25, 0.5, 0.75]).unstack().loc[xs]
    ax = axs[0]
    ax.fill_between(xs, q[0.25], q[0.75], color="#6baed6", alpha=0.3, lw=0)
    ax.plot(xs, g.mean().loc[xs], "-", color="#1f4e79", lw=1.2, label="closed form, 3,901")
    rr = [N["eta_sweep"][f"{x:g}"]["r_rs"] for x in xs]
    axs[1].plot(xs, rr, "-", color="#1f4e79", lw=1.2, label="closed form")
    if ridge is not None:
        # ridge penalty (relative to sigma_in^2) -> noise floor relative to the main text (ridge 1e-6)
        rs = ridge.merge(s, on="sid")
        for rg, h in rs.groupby("ridge"):
            x = rg / 1e-6
            bad = rg < 1e-10
            col = "#bdbdbd" if bad else "#d62728"
            ax.errorbar(x, h["MC"].mean(), yerr=h["MC"].std(), fmt="o", ms=3.5, color=col, mec="white", mew=0.4,
                        elinewidth=0.7)
            axs[1].plot(x, corr(h["MC"].values, h["stable_rank"].values), "o", ms=3.5, color=col, mec="white", mew=0.4)
        for a in axs:
            a.axvspan(1e-7, 1e-5, color="#f0f0f0", lw=0, zorder=0)
        h1 = [Line2D([], [], color="#1f4e79", lw=1.2), Line2D([], [], marker="o", ls="", color="#d62728", ms=3.5),
              Line2D([], [], marker="o", ls="", color="#bdbdbd", ms=3.5)]
        axs[1].legend(h1, ["closed form", "simulated (300)", "rounding-limited"], loc="lower left", numpoints=1)
    ax.axvline(1, color=MUTED, lw=0.6, ls=":")
    axs[1].axvline(1, color=MUTED, lw=0.6, ls=":")
    ax.set(xscale="log", xlabel="noise floor $\\eta/\\eta_0$", ylabel="mean MC")
    axs[1].set(xscale="log", xlabel="noise floor $\\eta/\\eta_0$", ylabel="$r$(MC, stable rank)", ylim=(0.5, 1.0))
    for a in axs: grid(a)
    label(axs[0], "a"); label(axs[1], "b")
    save(fig, "figS_noise")


def figS_energy(D):
    s, N = D["s"], D["N"]
    fig, axs = plt.subplots(1, 2, figsize=(W1 * 1.7, 2.1), gridspec_kw=dict(wspace=0.55))
    ax = axs[0]
    T = [1, 2, 3, 5, 10, 20]
    rv = [N["gram_series_r_MC"][f"T{t}"] for t in T]
    ax.plot(T, rv, "-o", color="#1f4e79", ms=3, lw=1)
    ax.axhline(N["gram_series_r_MC"]["Tinf"], color=MUTED, lw=0.8, ls="--")
    ax.text(20, N["gram_series_r_MC"]["Tinf"] + 0.03, "$T=\\infty$", fontsize=6, ha="right", va="bottom", color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.set(xscale="log", xlabel="terms $T$", ylabel="$r(S_T,\\mathrm{MC})$", xticks=T)
    ax.set_xticklabels([str(t) for t in T]); grid(ax, "y")
    label(ax, "a")
    ax = axs[1]
    sc = ax.scatter(s["stable_rank"], s[MC], c=s["gram_Tinf"], cmap="Blues", s=2, lw=0, rasterized=True,
                    vmin=s["gram_Tinf"].quantile(0.01) * 0.6, vmax=s["gram_Tinf"].quantile(0.99))
    rank_axis(ax); ax.set(xlabel="stable rank $r_s$", ylabel="MC, simulated")
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02); cb.set_label("$S_\\infty$ (energy)", fontsize=6)
    cb.ax.tick_params(labelsize=5); cb.outline.set_linewidth(0.4)
    note(ax, f"partial $r(S_\\infty,\\mathrm{{MC}}\\,|\\,r_s)={N['predictors']['all']['avg_controllability']['partial_given_rs']:.2f}$")
    label(ax, "b")
    save(fig, "figS_energy")


def figS_global(D):
    s, N = D["s"], D["N"]
    fig, axs = plt.subplots(1, 4, figsize=(W2, 1.9), gridspec_kw=dict(wspace=0.5))
    for ax, (x, xl, let) in zip(axs[:2], [("cv_s", "CV of node strength", "a"),
                                          ("ipr_v1", "IPR of leading eigenvector", "b")]):
        scatter_cohorts(ax, s, x, "s2t", s=2)
        ax.set(xlabel=xl, ylabel="$\\langle\\tilde s^{2}\\rangle$")
        note(ax, f"$r={corr(s[x].values, s['s2t'].values):.2f}$", x=0.96, ha="right")
        label(ax, let, x=-0.3)
    axs[1].set_xscale("log"); axs[1].xaxis.set_minor_formatter(NullFormatter())
    axs[1].set_xticks([0.02, 0.05, 0.1]); axs[1].set_xticklabels(["0.02", "0.05", "0.1"])
    for ax, (col, yl, let) in zip(axs[2:], [("strength", "raw strength", "c"), ("s2t", "$\\langle\\tilde s^{2}\\rangle$", "d")]):
        ax.scatter(s["age"], s[col], s=1, color="#e0e0e0", lw=0, rasterized=True)
        age_means(ax, s, col, ms=2)
        ax.set(xlabel="age (y)", ylabel=yl)
        label(ax, let, x=-0.3)
    h = [Line2D([], [], marker="o", ls="", ms=3, mfc=c[1], mec="white", mew=0.3) for c in COHORTS.values()]
    fig.legend(h, [c[0] for c in COHORTS.values()], loc="lower center", ncol=9, bbox_to_anchor=(0.5, 0.98),
               handletextpad=0.1, columnspacing=0.9)
    save(fig, "figS_global")


def figS_surrogates(D):
    surr, N = D["surr"], D["N"]
    sm = surr.groupby(["subject_id", "model"])["MC"].mean().unstack()
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.8, 2.2), gridspec_kw=dict(wspace=0.4, width_ratios=[1.1, 1]))
    ax = axs[0]
    order = [m for m in ["Uniform", "BrokenStick", "Reshuffle", "SignFlip", "RealAsym"] if m in sm]
    data = [(sm[m] - sm["Real"]).dropna().values for m in order]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=0.9),
                    whiskerprops=dict(lw=0.6, color=MUTED), capprops=dict(lw=0.6, color=MUTED))
    for b, m in zip(bp["boxes"], order):
        b.set(facecolor=MODELS[m][1], alpha=0.8, lw=0.5, edgecolor="white")
    rng = np.random.default_rng(0)
    for i, v in enumerate(data):
        ax.scatter(i + 1 + rng.uniform(-0.15, 0.15, len(v)), v, s=1.5, color=INK, alpha=0.35, lw=0, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.6, ls="--")
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels([MODELS[m][0] for m in order], rotation=30, ha="right")
    ax.set(ylabel="$\\Delta$MC (surrogate $-$ real)")
    grid(ax, "y"); label(ax, "a")
    ax = axs[1]
    s = D["s"]
    ax.scatter(s["m2"], s[MC], s=1.5, color="#d9d9d9", lw=0, rasterized=True)
    x = np.linspace(0.025, 0.09, 10); L = N["surrogates"]["line"]
    ax.plot(x, L["intercept"] + L["slope"] * x, color=MUTED, lw=0.8)
    means = surrogate_means(D)
    for m, (nm, col) in MODELS.items():
        if m not in means.index: continue
        r_ = means.loc[m]
        ax.plot(r_["m2"], r_["MC"], "o", ms=4.5, color=col, mec="white", mew=0.5, zorder=3)
        ax.plot(r_["m2"], r_["MC_lin"], "o", ms=7, mfc="none", mec=col, mew=0.7, zorder=3)
    h = [Line2D([], [], marker="o", ls="", ms=4, color=MODELS[m][1]) for m in MODELS] + \
        [Line2D([], [], marker="o", ls="", ms=6, mfc="none", mec=MUTED)]
    ax.legend(h, [MODELS[m][0] for m in MODELS] + ["closed form"], loc="lower right", fontsize=5.5)
    ax.set(xlabel="$m_2$", ylabel="MC", xlim=(0.02, 0.095))
    label(ax, "b")
    save(fig, "figS_surrogates")


def figS_taylor(D):
    nd = D["nodes"].dropna()
    fig, axs = plt.subplots(1, 2, figsize=(W1 * 1.6, 2.0), gridspec_kw=dict(wspace=0.45))
    ax = axs[0]
    edges = np.arange(0, nd["k"].max() + 8, 8)
    for rel, nm, col in [((nd["Cii"] - 1 - nd["T2"]) / (nd["Cii"] - 1), "order 2", "#1f4e79"),
                         ((nd["Cii"] - 1 - nd["T2"] - nd["T3"]) / (nd["Cii"] - 1), "order 3", "#6baed6")]:
        bm = binned(nd["k"].values, 100 * rel.values, edges)
        ax.fill_between(bm[:, 0], bm[:, 1], bm[:, 3], color=col, alpha=0.25, lw=0)
        ax.plot(bm[:, 0], bm[:, 2], "-o", color=col, ms=2.5, lw=1.0, label=nm)
    ax.set(xlabel="degree $k$", ylabel="relative error in $C_{ii}-1$ (%)", ylim=(0, None))
    grid(ax, "y"); ax.legend(loc="upper left"); label(ax, "a")
    ax = axs[1]
    ax.scatter(1 + nd["T2"], nd["Cii"], s=1, color="#4d4d4d", alpha=0.3, lw=0, rasterized=True)
    lo, hi = nd["Cii"].min(), nd["Cii"].max()
    ax.plot([lo, hi], [lo, hi], color="#d62728", lw=0.8, ls="--")
    ax.set(xlabel="$1+\\frac{1}{2}\\tilde s_i^2Y_i$", ylabel="$C_{ii}$")
    note(ax, f"$R^2={D['N']['nodes']['R2_order2']:.4f}$\n{len(nd):,} nodes")
    label(ax, "b")
    save(fig, "figS_taylor")


def figS_cohorts(D):
    s, N = D["s"], D["N"]
    fig = plt.figure(figsize=(W2, 4.3))
    gs = fig.add_gridspec(3, 6, hspace=0.75, wspace=1.4)
    xl = (s["stable_rank"].min() * 0.95, s["stable_rank"].max() * 1.05)
    yl = (s[MC].min() - 0.2, s[MC].max() + 0.2)
    for i, (c, (nm, col)) in enumerate(COHORTS.items()):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        g = s[s["dataset"] == c]
        ax.scatter(g["stable_rank"], g[MC], s=2, color=col, alpha=0.6, lw=0, rasterized=True)
        rank_axis(ax); ax.set(xlim=xl, ylim=yl)
        ax.set_xticks([3, 5, 10]); ax.set_xticklabels(["3", "5", "10"] if i // 3 == 2 else [])
        if i % 3: ax.set_yticklabels([])
        ax.set_title(f"{nm}  $R^2$={N['cohorts'][nm]['R2_log_rs']:.2f}", fontsize=6, pad=2)
        if i == 7: ax.set_xlabel("stable rank $r_s$")
        if i == 3: ax.set_ylabel("MC")
        if i == 0: label(ax, "a", x=-0.45, y=1.12)
    ax = fig.add_subplot(gs[0, 3:6])
    names = [COHORTS[c][0] for c in COHORTS if "r_MC_age" in N["cohorts"][COHORTS[c][0]]]
    vals = [N["cohorts"][n]["r_MC_age"] for n in names]
    cols = [COHORTS[c][1] for c in COHORTS if COHORTS[c][0] in names]
    ax.bar(range(len(names)), vals, color=cols, width=0.65, edgecolor="white", lw=0.5)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set(ylabel="$r$(MC, age) within cohort"); grid(ax, "y"); label(ax, "b", x=-0.12)
    cc = s[s["dataset"] == 8]
    bins = [18, 25, 32, 40, 50, 60, 70, 80, 90]
    g = cc.groupby(pd.cut(cc["age"], bins), observed=True)
    for j, (col, yl_, key, let) in enumerate([(MC, "MC (camCAN)", "MC_quad", "c"), ("stable_rank", "$r_s$ (camCAN)", "rs_quad", "d")]):
        ax = fig.add_subplot(gs[1 + j, 3:6])
        q = N["camcan"][key]; xa = np.linspace(19, 89, 100)
        ax.axvspan(*q["ci"], color="#bcbd22", alpha=0.15, lw=0)
        ax.plot(xa, np.polyval(q["coef"], xa), color=INK, lw=0.9)
        ax.errorbar(g["age"].mean(), g[col].mean(), yerr=g[col].sem(), fmt="o", ms=3, color="#8c8d10", mec="white",
                    mew=0.4, elinewidth=0.7)
        ax.axvline(q["vertex"], color=INK, lw=0.6, ls=":")
        ax.set(ylabel=yl_, xlim=(15, 92)); grid(ax, "y")
        if j == 1: ax.set_xlabel("age (y)")
        note(ax, f"minimum {q['vertex']:.1f} y", x=0.04, y=0.96)
        label(ax, let, x=-0.12)
    save(fig, "figS_cohorts")


FIGS = [fig2_order2, fig4_memory, fig5_lifespan, figS_theory, figS_gramian, figS_noise, figS_energy, figS_global,
        figS_surrogates, figS_taylor, figS_cohorts]

if __name__ == "__main__":
    D = load()
    want = set(sys.argv[1:])
    for f in FIGS:
        if not want or f.__name__ in want:
            f(D)
