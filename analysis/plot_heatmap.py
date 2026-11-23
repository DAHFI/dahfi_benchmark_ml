"""
Landscape heatmap of the mean performance of every pipeline.

    rows    : classifiers
    columns : preprocessing x feature set (M, L, NL, IBTF; cumulative sets), grouped by
              preprocessing
    cell    : mean metric across seeds (mean over the partitions of each seed) and, below it,
              +/- standard deviation across seeds (--no_sd to remove it); best cell in bold
    legend  : only minimum, median and maximum

Width and height:
    --width       width in inches (4.8 = text width of a 6-page ESANN paper)
    --row_height  height of each row in inches (smaller = flatter)

Usage:
    python plot_heatmap.py --results_dir outputs/results --out_dir outputs/figures
    python plot_heatmap.py --results_dir outputs/results --duration 30 --decimals 3
"""

import argparse
import os

import numpy as np

from common import FEATURE_SET_ORDER, INK, INK2, METHOD_ORDER, MODEL_ORDER, load_results, ordered

MODEL_LABELS = {"KNN": "kNN"}
SET_LABELS = {"M": "M", "M+L": "L", "M+L+NL": "NL", "M+L+NL+IBTF": "IBTF"}  # cumulative sets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_dir", default="outputs/figures")
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--metric", default="auc")
    ap.add_argument("--width", type=float, default=4.8, help="width (inches)")
    ap.add_argument("--row_height", type=float, default=0.27, help="height of each row (inches)")
    ap.add_argument("--no_sd", action="store_true", help="do not write the standard deviation")
    ap.add_argument("--decimals", type=int, default=2, help="decimals (2 = 0.75 +/-0.01)")
    ap.add_argument("--name", default="heatmap")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    plt.rcParams.update({"font.size": 6.5, "pdf.fonttype": 42})

    df = load_results(a.results_dir)
    df = df[df.duration.astype(int) == a.duration]
    if df.empty:
        raise SystemExit(f"No results for {a.duration} min.")
    cell = ["preprocessing", "feature_set", "model"]
    per_seed = df.groupby(cell + ["seed"])[a.metric].mean()
    mean = per_seed.groupby(cell).mean()
    sd = per_seed.groupby(cell).std(ddof=1)  # across seeds
    methods = ordered(mean.index.get_level_values(0).unique(), METHOD_ORDER)
    sets = ordered(mean.index.get_level_values(1).unique(), FEATURE_SET_ORDER)
    models = ordered(mean.index.get_level_values(2).unique(), MODEL_ORDER)
    cols = [(m, g) for m in methods for g in sets]
    M = np.array([[mean.get((m, g, mo), np.nan) for (m, g) in cols] for mo in models])
    S = np.array([[sd.get((m, g, mo), np.nan) for (m, g) in cols] for mo in models])

    vals = M[np.isfinite(M)]
    cmap = LinearSegmentedColormap.from_list("div", ["#97a4e8", "#f4f3f7", "#ec8a84"])
    cmap.set_bad("#f0efec")
    norm = TwoSlopeNorm(vmin=vals.min(), vcenter=float(np.median(vals)), vmax=vals.max())

    n_r, n_c = M.shape
    fig, ax = plt.subplots(figsize=(a.width, 0.45 + a.row_height * n_r))
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, aspect="auto")
    best = np.unravel_index(np.nanargmax(M), M.shape)
    fs = max(3.6, min(6.0, 0.95 * a.width / n_c * 72 / (a.decimals + 1.9)))  # number size from cell width
    for i in range(n_r):
        for j in range(n_c):
            if not np.isfinite(M[i, j]):
                continue
            bold = "bold" if (i, j) == best else "normal"
            if a.no_sd or not np.isfinite(S[i, j]):
                ax.text(j, i, f"{M[i, j]:.{a.decimals}f}", ha="center", va="center", fontsize=fs,
                        color=INK, fontweight=bold)
            else:  # mean on top, +/- SD below
                ax.text(j, i - 0.17, f"{M[i, j]:.{a.decimals}f}", ha="center", va="center",
                        fontsize=fs, color=INK, fontweight=bold)
                ax.text(j, i + 0.22, f"±{S[i, j]:.{a.decimals}f}", ha="center", va="center",
                        fontsize=fs * 0.82, color=INK2)
    ax.set_yticks(range(n_r), [MODEL_LABELS.get(m, m) for m in models], fontsize=6)
    ax.set_xticks(range(n_c), [SET_LABELS.get(g, g) for _, g in cols], fontsize=5, rotation=90)
    ax.tick_params(length=0, pad=1.5)
    ax.set_xticks(np.arange(-0.5, n_c, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_r, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.4)
    ax.tick_params(which="minor", length=0)
    for k in range(1, len(methods)):
        ax.axvline(k * len(sets) - 0.5, color="white", lw=2)
    for s in ax.spines.values():
        s.set_visible(False)
    for k, m in enumerate(methods):
        ax.text(k * len(sets) + (len(sets) - 1) / 2, -0.7, m, ha="center", va="bottom",
                fontsize=6, fontweight="bold", color=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, aspect=14)
    cb.outline.set_visible(False)
    ticks = [vals.min(), float(np.median(vals)), vals.max()]  # only min, median and max
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.2f}" for t in ticks])
    cb.minorticks_off()
    cb.ax.tick_params(labelsize=5, length=0)
    cb.set_label(a.metric.upper(), fontsize=5.5)

    path = os.path.join(a.out_dir, a.name + ".pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.splitext(path)[0] + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"{path} (and .png) · {n_r} classifiers x {n_c} pipelines · best: "
          f"{models[best[0]]} | {cols[best[1]][0]} | {cols[best[1]][1]} = {M[best]:.3f}")


if __name__ == "__main__":
    main()
