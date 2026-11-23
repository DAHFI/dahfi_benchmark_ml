"""
Effect of each factor of the design on the AUC: signal length vs preprocessing vs feature
set vs classifier.

Reads fold_results.csv (training/train_models.py) and runs three analyses:

1. FACTORIAL ANOVA as a VARIANCE DECOMPOSITION
   Model: AUC ~ duration x preprocessing x feature set x classifier + seed (block)
   Unit of analysis: the mean AUC of each seed (mean over its outer partitions). The
   partitions are not used as replicates: they share 75 % of their training data
   (pseudo-replication). Reported:
       eta^2          = SS_effect / SS_total      (fraction of the total variance)
       partial eta^2  = SS_effect / (SS_effect + SS_residual)
   Seed x factor interactions go to the residual (block design).
   NOTE: the F p-values are optimistic, because every cell uses the same records and, for a
   given seed, the same partitions (observations are not independent [1]). Interpret
   mainly the eta^2 values.

2. FRIEDMAN + POST-HOC per factor [2]
   For each factor, each block is a combination of the other factors and the seed; within
   each block its levels are ranked by AUC (rank 1 = best).
     - Friedman: do the mean ranks differ? (with 2 levels: Wilcoxon signed-rank test)
     - Nemenyi: critical difference CD = q_alpha * sqrt(k(k+1) / (6N)); two levels whose mean
       ranks differ by less than CD are not distinguishable.
     - Pairwise Wilcoxon tests with Holm correction [3] and the median AUC difference.
     - Critical-difference (CD) diagram: levels joined by a bar do not differ.

3. INTERACTION PLOTS
   For each pair of factors (A, B): mean AUC of each level of A, one line per level of B
   (averaging the rest). Parallel lines = no interaction.

Outputs (in --out_dir, by default <results_dir>/factor_analysis), each figure as PDF + PNG:
    anova_eta2          bars with the eta^2 of each effect
    cd_<factor>         critical-difference diagram of each factor
    interaction_<A>_<B> interaction plot of each pair of factors
    anova.csv, friedman.csv, posthoc_<factor>.csv, ranks_<factor>.csv

Usage:
    python factor_analysis.py --results_dir outputs/results
    python factor_analysis.py --results_dir outputs/results --metric ap
    python factor_analysis.py --results_dir outputs/results --models LASSO RIDGE SVM RF

References
[1] Nadeau C, Bengio Y. Inference for the generalization error. Mach Learn. 2003;52:239-281.
[2] Demsar J. Statistical comparisons of classifiers over multiple data sets.
    J Mach Learn Res. 2006;7:1-30.
[3] Holm S. A simple sequentially rejective multiple test procedure.
    Scand J Stat. 1979;6(2):65-70.
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd

from common import (DARK_COLORS, FACTORS, GRID, INK, INK2, LABELS, LOWER_IS_BETTER, METRIC_NAMES,
                    anova, cliques, friedman_factor, load_results, per_seed_table)

# Okabe-Ito: colour-blind-safe categorical palette (interaction plots)
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
LEVEL_LABELS = {"KNN": "kNN"}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle=":")
    ax.set_axisbelow(True)


def save(fig, path, tight=True):
    """Saves the figure as PDF and PNG (200 dpi) and closes it. With tight=False the exact
    figure size is kept (so that several figures share the same width)."""
    import matplotlib.pyplot as plt

    bb = "tight" if tight else None
    fig.savefig(path, bbox_inches=bb)
    fig.savefig(os.path.splitext(path)[0] + ".png", dpi=200, bbox_inches=bb)
    plt.close(fig)
    return os.path.basename(path)


def fig_eta2(tab, metric, path):
    """Horizontal bars with the eta^2 of each effect (main effects, two-way interactions and
    higher-order interactions >= 1 %)."""
    import matplotlib.pyplot as plt

    eff = tab[tab.order.between(1, 4)].sort_values("eta2", ascending=False)
    show = eff[(eff.order <= 2) | (eff.eta2 >= 0.01)]
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(show) + 1.6))
    ypos = np.arange(len(show))[::-1]
    colors = ["#0072B2" if o == 1 else "#56B4E9" if o == 2 else "#a9c7dd" for o in show.order]
    ax.barh(ypos, show.eta2, color=colors, height=0.65)
    ax.set_yticks(ypos, show.short, fontsize=9.5, color=INK)
    for yy, v in zip(ypos, show.eta2):
        ax.text(v, yy, f" {100 * v:.1f} %", va="center", fontsize=8.5, color=INK2)
    res = tab.loc[tab.effect == "Residual", "eta2"].iloc[0]
    ax.set_xlabel(f"eta$^2$ (fraction of the total variance of {METRIC_NAMES.get(metric, metric)}); "
                  f"residual: {100 * res:.1f} %", fontsize=9.5, color=INK2)
    style_axis(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, linestyle=":")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(show.eta2.max() * 1.25, 0.05))
    fig.text(0.01, -0.02, "D = signal length, P = preprocessing, F = feature set, C = classifier. "
             "Dark: main effects; light: interactions.", fontsize=8, color=INK2, va="top")
    return save(fig, path)


def fig_cd(mean_rank, cd, factor, summary, metric, alpha, path, fs=1.8, title=False):
    """Critical-difference diagram of one factor. fs scales the font size; title=True adds
    a title with the test result. Every CD figure has the same width and label margin, so
    when they are included with the same width the font size is the same in all of them.
    Levels joined by a thick bar (coloured by factor) do not differ (Nemenyi)."""
    import matplotlib.pyplot as plt

    k = len(mean_rank)
    half = int(np.ceil(k / 2))
    fig = plt.figure(figsize=(8.5, (0.7 + 0.34 * half) * (0.55 + 0.45 * fs)))
    m = min(0.36, 0.175 * fs)  # margin for the labels
    ax = fig.add_axes([m, 0.0, 1 - 2 * m, 1.0])
    lo, hi = 1, k
    ax.set_xlim(lo - 0.3, hi + 0.3)
    yaxis = 0.8
    step = 0.62 / max(half, 1)
    ax.set_ylim(yaxis - 0.16 - (half - 1) * step - 0.08, 1.1)
    ax.axis("off")
    ax.plot([lo, hi], [yaxis, yaxis], color=INK2, lw=1.2)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [yaxis, yaxis + 0.04], color=INK2, lw=1.2)
        ax.text(r, yaxis + 0.065, str(r), ha="center", va="bottom", fontsize=9 * fs, color=INK2)
    if np.isfinite(cd):
        ax.plot([lo, lo + cd], [1.02, 1.02], color=INK, lw=1.8)
        ax.text(lo + cd + 0.05, 1.02, f"CD = {cd:.3f}", ha="left", va="center", fontsize=8.5 * fs, color=INK)
    names, ranks = list(mean_rank.index), mean_rank.to_numpy()
    for i, (n, r) in enumerate(zip(names, ranks)):
        left = i < half
        level = i if left else k - 1 - i
        yy = yaxis - 0.16 - level * step
        xend = lo - 0.25 if left else hi + 0.25
        ax.plot([r, r, xend], [yaxis, yy, yy], color=INK, lw=1.0)
        ax.text(xend + (-0.03 if left else 0.03), yy, f"{LEVEL_LABELS.get(n, n)} ({r:.2f})",
                ha="right" if left else "left", va="center", fontsize=9.5 * fs, color=INK)
    color = DARK_COLORS.get(factor, "#D55E00")
    for b, (_, _, r1, r2) in enumerate(cliques(mean_rank, cd)):
        yy = yaxis - 0.05 - b * 0.04
        ax.plot([r1 - 0.03, r2 + 0.03], [yy, yy], color=color, lw=3.5, solid_capstyle="round", zorder=4)
    if title:
        p = summary["p"]
        ptxt = "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
        w = summary["kendall_W"]
        wtxt = f" · Kendall's W = {w:.2f}" if np.isfinite(w) else ""
        ax.set_title(f"{LABELS[factor]}: mean rank (1 = best {METRIC_NAMES.get(metric, metric)})  ·  "
                     f"{summary['test']} {ptxt}{wtxt}  ·  {summary['blocks']} blocks  ·  alpha = {alpha}",
                     loc="left", fontsize=10.5, color=INK, pad=14)
    return save(fig, path, tight=False)


def fig_interaction(d, metric, factors, a, b, eta, path):
    """Mean metric of each level of A (x axis) with one line per level of B."""
    import matplotlib.pyplot as plt

    if d[b].nunique() > d[a].nunique():  # the factor with more levels goes on the x axis
        a, b = b, a
    m = d.groupby([a, b], observed=True)[metric].mean().unstack(b)
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    x = np.arange(len(m.index))
    for i, col in enumerate(m.columns):
        ax.plot(x, m[col].to_numpy(), marker="o", ms=5, lw=1.9, color=PALETTE[i % len(PALETTE)], label=str(col))
    ax.set_xticks(x, [str(v) for v in m.index], fontsize=9.5, rotation=20 if len(x) > 4 else 0)
    ax.set_ylabel(f"Mean {METRIC_NAMES.get(metric, metric)}", fontsize=9.5, color=INK2)
    key = " x ".join(LABELS[f] for f in factors if f in (a, b))
    e = eta.get(key, np.nan)
    ax.set_title(f"{LABELS[a]} x {LABELS[b]}  (eta$^2$ = {100 * e:.1f} %)", fontsize=11, loc="left", color=INK)
    ax.legend(title=LABELS[b], fontsize=8.5, title_fontsize=9, frameon=False,
              bbox_to_anchor=(1.01, 1), loc="upper left")
    style_axis(ax)
    return save(fig, path)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True, help="output folder of train_models.py")
    ap.add_argument("--out_dir", default=None, help="default: <results_dir>/factor_analysis")
    ap.add_argument("--metric", default="auc", choices=list(METRIC_NAMES))
    ap.add_argument("--alpha", type=float, default=0.05, choices=[0.05, 0.10])
    ap.add_argument("--models", nargs="+", default=None, help="restrict to these models")
    ap.add_argument("--methods", nargs="+", default=None, help="restrict to these preprocessing pipelines")
    ap.add_argument("--feature_sets", nargs="+", default=None, help="restrict to these feature sets")
    ap.add_argument("--durations", type=int, nargs="+", default=None, help="restrict to these durations")
    ap.add_argument("--cd_fontscale", type=float, default=1.8, help="font scale of the CD diagrams")
    ap.add_argument("--cd_title", action="store_true", help="add a title (test, p, W) to the CD diagrams")
    a = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")

    out_dir = a.out_dir or os.path.join(a.results_dir, "factor_analysis")
    os.makedirs(out_dir, exist_ok=True)
    higher_better = a.metric not in LOWER_IS_BETTER

    df = load_results(a.results_dir)
    if a.metric not in df.columns:
        raise SystemExit(f"Metric '{a.metric}' is not in these results.")
    for col, vals in (("model", a.models), ("preprocessing", a.methods),
                      ("feature_set", a.feature_sets), ("duration", a.durations)):
        if vals:
            df = df[df[col].astype(str).str.upper().isin([str(v).upper() for v in vals])]
    df["duration"] = df["duration"].astype(int).astype(str) + " min"
    factors = [f for f in FACTORS if df[f].nunique() > 1]  # single-level factors are dropped
    if not factors:
        raise SystemExit("No factor has more than one level.")

    d = per_seed_table(df, a.metric, factors)
    n_seeds = d.seed.nunique()
    print("Design: " + " x ".join(f"{LABELS[f]} ({d[f].nunique()})" for f in factors)
          + f" x {n_seeds} seeds = {len(d)} observations\n")

    # 1. ANOVA
    tab = anova(d, a.metric, factors)
    tab.round(6).to_csv(os.path.join(out_dir, "anova.csv"), index=False)
    show = tab[["effect", "df", "F", "p", "eta2", "eta2_partial"]].copy()
    show["eta2"] = (100 * show.eta2).round(1).astype(str) + " %"
    print("1. Factorial ANOVA (eta2 = % of the total variance)")
    print(show.round(4).to_string(index=False), "\n")

    # 2. Friedman
    results, summaries = {}, []
    for f in factors:
        summary, mean_rank, post, cd = friedman_factor(d, a.metric, f, factors, higher_better, a.alpha)
        results[f] = (summary, mean_rank, post, cd)
        summaries.append(summary)
        post.round(5).to_csv(os.path.join(out_dir, f"posthoc_{f}.csv"), index=False)
        mean_rank.rename("mean_rank").to_csv(os.path.join(out_dir, f"ranks_{f}.csv"))
        print(f"2. {LABELS[f]}: {summary['test']} p = {summary['p']:.3g}, "
              f"Kendall's W = {summary['kendall_W']:.2f}, CD = {cd:.3f}; mean ranks:")
        print("   " + "  ".join(f"{n} {r:.2f}" for n, r in mean_rank.items()))
    pd.DataFrame(summaries).round(6).to_csv(os.path.join(out_dir, "friedman.csv"), index=False)

    # Figures, each in its own file (PDF + PNG)
    eta = dict(zip(tab.effect, tab.eta2))
    files = [fig_eta2(tab, a.metric, os.path.join(out_dir, "anova_eta2.pdf"))]
    for f, (summary, mean_rank, _, cd) in results.items():
        files.append(fig_cd(mean_rank, cd, f, summary, a.metric, a.alpha, os.path.join(out_dir, f"cd_{f}.pdf"),
                            fs=a.cd_fontscale, title=a.cd_title))
    for fa, fb in itertools.combinations(factors, 2):
        files.append(fig_interaction(d, a.metric, factors, fa, fb, eta,
                                     os.path.join(out_dir, f"interaction_{fa}_{fb}.pdf")))
    print(f"\nFigures in {out_dir} (each as .pdf and .png):")
    for name in files:
        print(f"   {name}")
    print("Tables: anova.csv, friedman.csv, posthoc_<factor>.csv, ranks_<factor>.csv")


if __name__ == "__main__":
    main()
