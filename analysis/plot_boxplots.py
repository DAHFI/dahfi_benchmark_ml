"""
Boxplots of the results of training/train_models.py grouped by factor:

    - Signal length   (only if there is more than one)
    - Preprocessing   (SOTA, BASIC, FOURIER, SPLINES, LLR)
    - Feature set     (M, M+L, M+L+NL, M+L+NL+IBTF)
    - Classifier      (SVM, LASSO, RIDGE, RF, ANN, KNN, DT)

Each point is one pipeline (duration, preprocessing, feature set, classifier) with its
metric averaged over seeds, i.e. one cell of the heatmap. Each panel distributes all those
points according to one factor, so each box summarises the performance of one level
averaged over the other factors. Within each panel, boxes are sorted by median (best on
top). With --mean_sd the mean +/- standard deviation of each box is written under its label.

Usage:
    python plot_boxplots.py --results_dir outputs/results --mean_sd
    python plot_boxplots.py --results_dir outputs/results --metric ap
    python plot_boxplots.py --results_dir outputs/results --xlim 0.5 0.75
"""

import argparse
import os

import numpy as np
import pandas as pd

from common import BOX_COLORS, INK, INK2, GRID, LABELS, LOWER_IS_BETTER, METRIC_NAMES, UNIT_KEYS, load_results

PANELS = ["duration", "preprocessing", "feature_set", "model"]  # top to bottom
LEVEL_LABELS = {"KNN": "kNN"}


def summarize(df, metric):
    """Mean and standard deviation across seeds (mean over partitions within each seed)."""
    per_seed = df.groupby(UNIT_KEYS)[metric].mean()
    g = per_seed.groupby(["duration", "preprocessing", "feature_set", "model"])
    return pd.DataFrame({"mean": g.mean(), "std": g.std(), "n": g.size()}).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True, help="output folder of train_models.py")
    ap.add_argument("--out_dir", default=None, help="where to save the figure (default: results_dir)")
    ap.add_argument("--metric", default="auc", choices=list(METRIC_NAMES))
    ap.add_argument("--xlim", type=lambda v: None if v.lower() == "none" else float(v), nargs=2,
                    default=[0.55, None], help="x-axis limits, e.g. 0.5 0.75 (default: start at 0.55)")
    ap.add_argument("--fontscale", type=float, default=1.8, help="font scale")
    ap.add_argument("--width", type=float, default=6.5, help="figure width in inches")
    ap.add_argument("--mean_sd", action="store_true", help="write the mean +/- SD of each box")
    ap.add_argument("--seed", type=int, default=0, help="seed of the random jitter of the points")
    a = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.transforms import blended_transform_factory

    out_dir = a.out_dir or a.results_dir
    os.makedirs(out_dir, exist_ok=True)
    higher_better = a.metric not in LOWER_IS_BETTER

    df = load_results(a.results_dir)
    if a.metric not in df.columns:
        raise SystemExit(f"Metric '{a.metric}' is not in these results.")
    summ = summarize(df, a.metric).dropna(subset=["mean"])
    if summ.empty:
        raise SystemExit("No completed results.")
    summ["duration"] = summ["duration"].astype(int).astype(str) + " min"
    panels = [p for p in PANELS if summ[p].nunique() > 1]  # single-level factors are dropped

    n_levels = [summ[p].nunique() for p in panels]
    fs = a.fontscale
    fig, axes = plt.subplots(
        len(panels), 1, sharex=True,
        figsize=(a.width, (1.1 + 0.42 * sum(n_levels) + 0.5 * len(panels)) * (0.6 + 0.4 * fs)),
        gridspec_kw={"height_ratios": [n + 0.8 for n in n_levels], "hspace": 0.12},
    )
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(a.seed)

    for ax, col in zip(axes, panels):
        med = summ.groupby(col)["mean"].median()
        levels = list(med.sort_values(ascending=not higher_better).index)  # best on top
        data = [summ.loc[summ[col] == lv, "mean"].to_numpy() for lv in levels]
        pos = np.arange(len(levels))[::-1]

        style = dict(
            positions=pos, widths=0.55, patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=BOX_COLORS[col], edgecolor="#6e6d68", linewidth=0.9),
            medianprops=dict(color="#4a4945", linewidth=1.1),
            whiskerprops=dict(color="#6e6d68", linewidth=0.9),
            capprops=dict(color="#6e6d68", linewidth=0.9),
        )
        try:  # matplotlib >= 3.10
            ax.boxplot(data, orientation="horizontal", **style)
        except TypeError:  # older versions
            ax.boxplot(data, vert=False, **style)
        for p, vals in zip(pos, data):
            ax.scatter(vals, p + rng.uniform(-0.18, 0.18, vals.size), s=10, color="#8a8984",
                       alpha=0.65, linewidths=0, zorder=3)

        names = [LEVEL_LABELS.get(lv, lv) for lv in levels]
        if a.mean_sd:  # level name and, below it, mean +/- SD across the pipelines of the box
            ytr = blended_transform_factory(ax.transAxes, ax.transData)
            for p, name, vals in zip(pos, names, data):
                ax.text(-0.015, p + 0.17, name, transform=ytr, ha="right", va="center",
                        fontsize=10.5 * fs, color=INK)
                ax.text(-0.015, p - 0.25, f"{vals.mean():.3f} ± {vals.std(ddof=1):.3f}", transform=ytr,
                        ha="right", va="center", fontsize=8 * fs, color=INK2)
            ax.set_yticks(pos, [""] * len(pos))
        else:
            ax.set_yticks(pos, names, fontsize=10.5 * fs, color=INK)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_ylim(-0.7, len(levels) + 0.1)  # room for the panel title
        ax.tick_params(axis="y", length=0)
        ax.text(1.0, 1.0, LABELS[col], transform=ax.transAxes, ha="right", va="top",
                fontsize=11.5 * fs, fontweight="bold", color=INK2)
        ax.grid(axis="x", color=GRID, linewidth=0.8, linestyle=":")
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_visible(ax is axes[-1])
        ax.spines["bottom"].set_color("#8a8984")
        ax.tick_params(axis="x", colors=INK2, labelsize=10 * fs, length=0 if ax is not axes[-1] else 3)

    if a.xlim:
        axes[-1].set_xlim(left=a.xlim[0], right=a.xlim[1])
        if a.xlim[0] is not None:
            n_out = int((summ["mean"] < a.xlim[0]).sum())
            if n_out:
                print(f"Warning: {n_out} points are below {a.xlim[0]} and are not shown "
                      f"(use --xlim 0.5 None to show them)")
    axes[-1].set_xlabel(METRIC_NAMES.get(a.metric, a.metric), color=INK2, fontsize=10 * fs)

    path = os.path.join(out_dir, f"boxplots_{a.metric}.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Table with the summary of each level
    rows = []
    for col in panels:
        for lv, g in summ.groupby(col)["mean"]:
            rows.append({"factor": LABELS[col], "level": lv, "n": g.size, "mean": g.mean(),
                         "sd": g.std(ddof=1), "median": g.median(), "q25": g.quantile(0.25),
                         "q75": g.quantile(0.75), "min": g.min(), "max": g.max()})
    pd.DataFrame(rows).round(4).to_csv(os.path.join(out_dir, f"boxplots_{a.metric}.csv"), index=False)
    print(f"Figure: {path} (and .png) · table: boxplots_{a.metric}.csv")


if __name__ == "__main__":
    main()
