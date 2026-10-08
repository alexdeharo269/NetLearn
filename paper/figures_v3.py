#!/usr/bin/env python3
"""
Figures of the v3 manuscript (single-column, 8.7 cm wide, 400 dpi), drawn from paper/data only.
  python paper/figures_v3.py            -> paper/figures/v3/*.pdf and *.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
DATA, D3, OUT = REPO / "paper/data", REPO / "paper/data/v3", REPO / "paper/figures/v3"
W1 = 3.42                                                        # PNAS single column (8.7 cm)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e2e1dc"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"                     # validated categorical slots 1-3
AGE_CMAP = mpl.colors.LinearSegmentedColormap.from_list("age", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
ORDER = ["dHCP", "BCP", "CALM", "RED", "ACE", "HCPd", "HCPya", "camCAN", "HCPa"]

sns.set_theme(context="paper", style="ticks", font="DejaVu Sans")
mpl.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "pdf.fonttype": 42, "savefig.dpi": 400, "lines.linewidth": 1.2})


def letter(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig)


def age_scatter(ax, x, y, age, s=2.5):
    return ax.scatter(x, y, c=age, cmap=AGE_CMAP, vmin=0, vmax=90, s=s, lw=0, alpha=0.7, rasterized=True)


def age_bar(fig, sc, ax):
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05, aspect=25)
    cb.set_label("age (years)", fontsize=6.5); cb.ax.tick_params(labelsize=6, width=0.5, length=2); cb.outline.set_visible(False)


def bin_means(d, col, edges=(-1, 0.5, 2, 5, 9, 13, 17, 21, 26, 31, 36, 45, 55, 65, 75, 91)):
    g = d.groupby(pd.cut(d.age, edges), observed=True)
    return pd.DataFrame(dict(age=g.age.mean(), m=g[col].mean(), se=g[col].sem())).dropna()


def fig1_cohorts(d, calm):
    """Age coverage of each cohort and the parcellation of its released networks."""
    rows = d[["cohort", "age", "atlas"]].copy()
    ref = calm[calm.referred == 1][["age"]].assign(cohort="CALM referred", atlas="AAL90")
    rows = pd.concat([rows, ref]); order = ORDER[:3] + ["CALM referred"] + ORDER[3:]
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    pal = {"AAL90": C1, "AAL3": C2}
    sns.stripplot(data=rows, y="cohort", x="age", hue="atlas", order=order, palette=pal, size=1.6, jitter=0.28,
                  alpha=0.45, ax=ax, legend=False, rasterized=True)
    sns.boxplot(data=rows, y="cohort", x="age", order=order, color="white", width=0.55, fliersize=0, linewidth=0.6,
                boxprops=dict(facecolor="none", edgecolor=INK2), whiskerprops=dict(color=INK2), medianprops=dict(color=INK),
                capprops=dict(color=INK2), ax=ax)
    for i, c in enumerate(order):
        ax.text(93, i, f"{(rows.cohort == c).sum()}", va="center", ha="left", fontsize=6, color=INK2)
    ax.text(93, -0.9, "n", fontsize=6, color=INK2)
    ax.set(xlim=(-2, 92), xlabel="age (years)", ylabel="")
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); sns.despine(ax=ax, left=True); ax.tick_params(axis="y", length=0)
    hs = [mpl.lines.Line2D([], [], marker="o", ls="", color=pal[k], ms=3.5, label=f"{k} order") for k in pal]
    ax.legend(handles=hs, loc="upper right", frameon=False, handletextpad=0.1, borderaxespad=0.2)
    save(fig, "fig1_cohorts")


def fig3_memory(v2, ex):
    """Memory is a count of input directions above the noise floor; the stable rank sets the count."""
    fig, axs = plt.subplots(3, 1, figsize=(W1, 6.2), gridspec_kw=dict(hspace=0.55, height_ratios=[1, 1, 1.1]))
    ax = axs[0]
    hb = ax.hexbin(v2["MClin_s1e-05"], v2["MC_s1e-05"], gridsize=45, cmap=AGE_CMAP, mincnt=1, bins="log", linewidths=0, rasterized=True)
    lo, hi = 6.5, 12.6; ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.6, ls="--")
    r = np.corrcoef(v2["MClin_s1e-05"], v2["MC_s1e-05"])[0, 1]
    ax.text(0.04, 0.9, f"r = {r:.3f}", transform=ax.transAxes)
    ax.set(xlabel="memory capacity, closed form", ylabel="memory capacity,\nsimulated reservoir", xlim=(lo, hi), ylim=(lo, hi))
    cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.05, aspect=25); cb.set_label("connectomes", fontsize=6.5)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2); cb.outline.set_visible(False); letter(ax, "a")
    ax = axs[1]
    for lab, col, name in (("p5", C2, "5th"), ("p50", C1, "50th"), ("p95", C3, "95th")):
        e = ex[ex.label == lab]; e = e[e.k <= 30]
        ax.semilogy(e.k, e.c_over_eta, "-o", color=col, ms=2.2, lw=0.9, label=f"{name} percentile of MC")
    ax.axhline(1, color=INK2, lw=0.7, ls="--"); ax.text(29.5, 2.2, "noise floor", ha="right", fontsize=6, color=INK2)
    ax.set(xlabel="rank $k$ of the input-driven direction", ylabel="variance $c_k$ / noise floor $\\eta$", ylim=(1e-12, 1e12), xlim=(0, 30.5))
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(numticks=7)); ax.legend(frameon=False, loc="upper right")
    letter(ax, "b")
    ax = axs[2]
    sc = age_scatter(ax, v2.stable_rank, v2["MC_s1e-05"], v2.age); ax.set_xscale("log")
    b = np.polyfit(np.log(v2.stable_rank), v2["MC_s1e-05"], 1); xs = np.geomspace(v2.stable_rank.min(), v2.stable_rank.max(), 50)
    ax.plot(xs, np.polyval(b, np.log(xs)), color=INK, lw=0.9)
    r = np.corrcoef(np.log(v2.stable_rank), v2["MC_s1e-05"])[0, 1]
    ax.text(0.04, 0.9, f"r = {r:.2f}", transform=ax.transAxes)
    ax.set_xticks([3, 4, 5, 6, 8, 10]); ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set(xlabel="stable rank $r_s$", ylabel="memory capacity"); age_bar(fig, sc, ax); letter(ax, "c")
    for a in axs: sns.despine(ax=a)
    save(fig, "fig3_memory")


def fig4_nulls(nl):
    """Change in memory produced by each null model, against the change predicted by the stable rank."""
    names = {"Uniform": "Uniform", "BrokenStick": "Broken stick", "Reshuffle": "Reshuffle",
             "EqualStrength": "Equal strength", "InterHalf": "Interhemispheric × ½"}
    order = ["Uniform", "BrokenStick", "Reshuffle", "EqualStrength", "InterHalf"]
    nl = nl.assign(null=nl.model.map(names)); lab = [names[m] for m in order]
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    ax.axvline(0, color=INK2, lw=0.6)
    sns.boxplot(data=nl, y="null", x="dMC", order=lab, color="#cde2fb", width=0.55, fliersize=0, linewidth=0.6,
                boxprops=dict(edgecolor=INK2), whiskerprops=dict(color=INK2), medianprops=dict(color=INK), capprops=dict(color=INK2), ax=ax)
    sns.stripplot(data=nl, y="null", x="dMC", order=lab, color=C1, size=1.8, jitter=0.18, alpha=0.6, ax=ax, rasterized=True)
    pred = nl.groupby("null").dMC_pred.median().reindex(lab)
    ax.scatter(pred.values, range(len(lab)), marker="D", s=16, color=C2, edgecolor="white", lw=0.5, zorder=5, label="predicted by stable rank")
    ax.set(xlabel="change in memory capacity (null $-$ real)", ylabel="")
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); sns.despine(ax=ax, left=True); ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper right", handletextpad=0.2)
    save(fig, "fig4_nulls")


def fig5_interhemispheric(d, kap):
    """Interhemispheric weight limits memory: across connectomes and by intervention."""
    fig, axs = plt.subplots(2, 1, figsize=(W1, 4.3), gridspec_kw=dict(hspace=0.5))
    ax = axs[0]
    sc = age_scatter(ax, d.hom_F2, d.MC, d.age)
    r = np.corrcoef(d.hom_F2, d.MC)[0, 1]; ax.text(0.96, 0.9, f"r = {r:.2f}", transform=ax.transAxes, ha="right")
    ax.set(xlabel="homotopic share of squared weight, $h$", ylabel="memory capacity"); age_bar(fig, sc, ax); letter(ax, "a")
    ax = axs[1]
    for c, col in (("camCAN", C1), ("HCPa", C2)):
        g = kap[kap.cohort == c].groupby("kappa").MC.agg(["mean", "sem"]).reset_index()
        ax.errorbar(g.kappa, g["mean"], yerr=1.96 * g["sem"], color=col, marker="o", ms=3, lw=1, capsize=1.5, label=c)
    ax.axvline(1, color=INK2, lw=0.6, ls="--"); ax.text(1.02, ax.get_ylim()[0], " real weights", fontsize=6, color=INK2, va="bottom")
    ax.set(xlabel="interhemispheric weights scaled by $\\kappa$", ylabel="memory capacity")
    ax.legend(frameon=False, loc="upper right"); letter(ax, "b")
    for a in axs: sns.despine(ax=a)
    save(fig, "fig5_interhemispheric")


def fig6_lifespan(d, N):
    """Memory and interhemispheric weight across the lifespan; the aging rise within cohorts."""
    fig = plt.figure(figsize=(W1, 5.9)); gs = fig.add_gridspec(3, 1, hspace=0.55, height_ratios=[1, 1, 0.8])
    for k, (col, lab) in enumerate((("MC", "memory capacity"), ("hom_F2", "homotopic share $h$"))):
        ax = fig.add_subplot(gs[k])
        ax.scatter(d.age, d[col], s=1.5, color="#9a9994", alpha=0.25, lw=0, rasterized=True)
        b = bin_means(d, col); ax.errorbar(b.age, b.m, yerr=1.96 * b.se, color=INK, lw=1, marker="o", ms=2.5, capsize=1.2, label="all cohorts")
        for c, cc in (("camCAN", C1), ("HCPa", C2)):
            bc = bin_means(d[d.cohort == c], col, edges=(17, 30, 40, 50, 60, 70, 80, 91))
            ax.plot(bc.age, bc.m, color=cc, lw=1.2, marker="s", ms=2.2, label=c)
        ax.set(xlabel="age (years)" if k == 1 else "", ylabel=lab, xlim=(-2, 92)); letter(ax, "ab"[k])
        if k == 0: ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.98), columnspacing=1)
        sns.despine(ax=ax)
    ax = fig.add_subplot(gs[2]); L = N["lifespan"]
    vals = pd.DataFrame([dict(cohort=c, kind=k, r=L[c][key]) for c in ("camCAN", "HCPa")
                         for k, key in (("age", "r_age_MC"), ("age | $h$", "pr_age_MC_given_hom"))])
    sns.barplot(data=vals, x="r", y="cohort", hue="kind", palette=[INK2, "#cde2fb"], ax=ax, width=0.6, edgecolor=INK2, linewidth=0.5)
    ax.axvline(0, color=INK2, lw=0.6); ax.set(xlabel="correlation of memory capacity with age", ylabel="", xlim=(-0.1, 0.42))
    hs = [mpl.patches.Patch(facecolor=c, edgecolor=INK2, lw=0.5, label=l) for c, l in ((INK2, "age"), ("#cde2fb", "age, controlling for $h$"))]
    ax.legend(handles=hs, frameon=False, loc="lower right")
    ax.tick_params(axis="y", length=0); sns.despine(ax=ax, left=True); letter(ax, "c")
    save(fig, "fig6_lifespan")


def main():
    d = pd.read_csv(D3 / "subjects_v3.csv"); calm = pd.read_csv(D3 / "calm.csv")
    N = json.load(open(D3 / "numbers_v3.json"))
    v2 = pd.read_csv(DATA / "subject_v2.csv"); ex = pd.read_csv(DATA / "examples.csv")
    fig1_cohorts(d, calm)
    fig3_memory(v2, ex)
    fig4_nulls(pd.read_csv(D3 / "nulls_v3.csv"))
    fig5_interhemispheric(d, pd.read_csv(D3 / "kappa.csv"))
    fig6_lifespan(d, N)
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
