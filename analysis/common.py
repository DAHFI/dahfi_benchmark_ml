"""
Shared helpers of the analysis scripts: loading the results of training/train_models.py,
level ordering, the factorial ANOVA and the Friedman/Nemenyi comparison.

References
[1] Nadeau C, Bengio Y. Inference for the generalization error. Mach Learn. 2003;52:239-281.
[2] Demsar J. Statistical comparisons of classifiers over multiple data sets.
    J Mach Learn Res. 2006;7:1-30.
[3] Holm S. A simple sequentially rejective multiple test procedure.
    Scand J Stat. 1979;6(2):65-70.
"""

import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

UNIT_KEYS = ["duration", "preprocessing", "feature_set", "model", "seed"]
FACTORS = ["duration", "preprocessing", "feature_set", "model"]

METHOD_ORDER = ["SOTA", "BASIC", "FOURIER", "SPLINES", "LLR"]
FEATURE_SET_ORDER = ["M", "M+L", "M+L+NL", "M+L+NL+IBTF"]
MODEL_ORDER = ["SVM", "LASSO", "RIDGE", "RF", "ANN", "KNN", "DT"]
ORDERS = {"preprocessing": METHOD_ORDER, "feature_set": FEATURE_SET_ORDER, "model": MODEL_ORDER}

LABELS = {"duration": "Signal length", "preprocessing": "Preprocessing", "feature_set": "Feature set",
          "model": "Classifier", "seed": "Seed (block)"}
SHORT = {"duration": "D", "preprocessing": "P", "feature_set": "F", "model": "C"}
METRIC_NAMES = {"auc": "AUC", "ap": "AP (area under the PR curve)", "brier": "Brier score",
                "bal_acc": "Balanced accuracy", "sens": "Sensitivity", "spec": "Specificity"}
LOWER_IS_BETTER = {"brier"}

# Factor colours (light: boxes; dark: CD bars and titles)
BOX_COLORS = {"duration": "#f6d6b8", "preprocessing": "#cfe3ec", "feature_set": "#e6cdef", "model": "#dbe3a8"}
DARK_COLORS = {"duration": "#c07a3a", "preprocessing": "#4f93b0", "feature_set": "#a46cc0", "model": "#8a9c2e"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

# Column names of older (Spanish) result files, so that they can also be analysed
_LEGACY_COLUMNS = {"duracion": "duration", "preprocesado": "preprocessing", "grupo": "feature_set",
                   "modelo": "model", "semilla": "seed"}
_LEGACY_FILES = {"fold_results.csv": "resultados_folds.csv", "completed.txt": "completados.txt"}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def _path(results_dir, name):
    p = os.path.join(results_dir, name)
    if not os.path.exists(p) and name in _LEGACY_FILES:
        p = os.path.join(results_dir, _LEGACY_FILES[name])
    return p


def load_results(results_dir):
    """Rows of fold_results.csv belonging to completed units (without duplicates)."""
    path = _path(results_dir, "fold_results.csv")
    if not os.path.exists(path):
        raise SystemExit(f"No fold_results.csv in {results_dir}.")
    df = pd.read_csv(path).rename(columns=_LEGACY_COLUMNS)
    done_path = _path(results_dir, "completed.txt")
    if os.path.exists(done_path):
        with open(done_path, encoding="utf-8") as f:
            done = {line.strip() for line in f if line.strip()}
        df = df[df[UNIT_KEYS].astype(str).agg("|".join, axis=1).isin(done)]
    return df.drop_duplicates(subset=UNIT_KEYS + ["fold"], keep="last")


def ordered(values, order):
    """Sorts according to `order` and appends the values that are not in it."""
    values = list(dict.fromkeys(values))
    return [v for v in order if v in values] + sorted(v for v in values if v not in order)


def per_seed_table(df, metric, factors):
    """Mean metric per seed in each cell (mean of its outer partitions); keeps a complete,
    balanced design."""
    d = df.groupby(factors + ["seed"])[metric].mean().reset_index()
    n_cells = int(np.prod([d[f].nunique() for f in factors]))
    counts = d.groupby("seed").size()
    complete = counts[counts == n_cells].index
    dropped = sorted(set(counts.index) - set(complete))
    if dropped:
        print(f"Warning: seeds {dropped} are discarded because they do not have every cell.")
    d = d[d.seed.isin(complete)]
    if d.empty:
        raise SystemExit(f"The design is not complete (fewer than {n_cells} cells with results). "
                         "Finish training or filter with --models/--methods/--feature_sets/--durations.")
    for f in factors:
        d[f] = pd.Categorical(d[f], ordered(d[f].unique(), ORDERS.get(f, sorted(d[f].unique()))))
    return d


# --------------------------------------------------------------------------
# Factorial ANOVA (balanced design, exact decomposition of means)
# --------------------------------------------------------------------------
def short_label(S):
    """Main effects with their name; interactions abbreviated (P x F x C)."""
    return LABELS[S[0]] if len(S) == 1 else " x ".join(SHORT[f] for f in S)


def anova(d, metric, factors):
    """Sums of squares of every effect, with the seed as a block. In a complete, balanced
    design the effects are obtained exactly by inclusion-exclusion of marginal means (types I,
    II and III coincide). The F p-values are optimistic because the observations are not
    independent [1]: interpret mainly eta^2."""
    y = d[metric].to_numpy(float)
    N, grand = len(y), y.mean()
    ss_total = ((y - grand) ** 2).sum()
    levels = {f: d[f].nunique() for f in factors + ["seed"]}

    marg = {(): np.full(N, grand)}  # marginal mean of each subset of factors, per observation
    for r in range(1, len(factors) + 1):
        for S in itertools.combinations(factors, r):
            marg[S] = d.groupby(list(S), observed=True)[metric].transform("mean").to_numpy(float)

    def effect(S):
        """Effect of S by inclusion-exclusion of the marginal means (Moebius inversion)."""
        out = np.zeros(N)
        for r in range(len(S) + 1):
            for T in itertools.combinations(S, r):
                out += (-1) ** (len(S) - r) * marg[T]
        return out

    rows = []
    for r in range(1, len(factors) + 1):
        for S in itertools.combinations(factors, r):
            rows.append({"effect": " x ".join(LABELS[f] for f in S), "short": short_label(S), "order": r,
                         "SS": (effect(S) ** 2).sum(), "df": int(np.prod([levels[f] - 1 for f in S]))})
    seed_mean = d.groupby("seed", observed=True)[metric].transform("mean").to_numpy(float)
    rows.append({"effect": LABELS["seed"], "short": LABELS["seed"], "order": 0,
                 "SS": ((seed_mean - grand) ** 2).sum(), "df": levels["seed"] - 1})

    tab = pd.DataFrame(rows)
    ss_res = ss_total - tab.SS.sum()
    df_res = (N - 1) - tab.df.sum()
    if df_res <= 0:
        raise SystemExit("No residual degrees of freedom: at least 2 seeds are needed.")
    ms_res = ss_res / df_res
    tab["MS"] = tab.SS / tab.df
    tab["F"] = tab.MS / ms_res
    tab["p"] = stats.f.sf(tab.F, tab.df, df_res)
    tab["eta2"] = tab.SS / ss_total
    tab["eta2_partial"] = tab.SS / (tab.SS + ss_res)
    res = pd.DataFrame([
        {"effect": "Residual", "short": "Residual", "order": 9, "SS": ss_res, "df": df_res,
         "MS": ms_res, "eta2": ss_res / ss_total},
        {"effect": "Total", "short": "Total", "order": 10, "SS": ss_total, "df": N - 1, "eta2": 1.0},
    ])
    return pd.concat([tab, res], ignore_index=True)


# --------------------------------------------------------------------------
# Friedman + post-hoc [2]
# --------------------------------------------------------------------------
# Nemenyi critical values (studentised range / sqrt(2)), alpha = 0.05 and 0.10, k = 2..10 [2]
Q_NEMENYI = {
    0.05: [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164],
    0.10: [1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920],
}


def holm(p):
    """Holm-adjusted p-values [3]."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[idx]))
        adj[idx] = running
    return adj


def friedman_factor(d, metric, factor, factors, higher_better=True, alpha=0.05):
    """Each block is a combination of the other factors and the seed; within each block the
    levels of `factor` are ranked (1 = best). Returns (summary, mean ranks, pairwise
    post-hoc table, Nemenyi critical difference)."""
    others = [f for f in factors if f != factor] + ["seed"]
    wide = d.pivot_table(index=others, columns=factor, values=metric, observed=True).dropna()
    levels = list(wide.columns)
    k, N = len(levels), len(wide)
    X = wide.to_numpy(float)
    ranks = stats.rankdata(-X if higher_better else X, axis=1)  # 1 = best
    mean_rank = pd.Series(ranks.mean(axis=0), index=levels)

    if k >= 3:
        stat, p = stats.friedmanchisquare(*X.T)
        test = "Friedman"
    else:
        stat, p = stats.wilcoxon(X[:, 0], X[:, 1])
        test = "Wilcoxon"
    qa = Q_NEMENYI[alpha][k - 2] if k - 2 < len(Q_NEMENYI[alpha]) else np.nan
    cd = qa * np.sqrt(k * (k + 1) / (6.0 * N))
    # Kendall's W: effect size of Friedman (0 = no agreement, 1 = same order in every block)
    w = stat / (N * (k - 1)) if test == "Friedman" else np.nan

    pairs = []
    for i, j in itertools.combinations(range(k), 2):
        diff = X[:, i] - X[:, j]
        try:
            pw = stats.wilcoxon(X[:, i], X[:, j]).pvalue if np.any(diff != 0) else 1.0
        except ValueError:
            pw = 1.0
        pairs.append({
            "level_A": levels[i], "level_B": levels[j],
            "median_diff": np.median(diff), "mean_diff": diff.mean(),
            "pct_A_better": 100 * np.mean(diff > 0 if higher_better else diff < 0),
            "rank_A": mean_rank.iloc[i], "rank_B": mean_rank.iloc[j],
            "nemenyi_different": abs(mean_rank.iloc[i] - mean_rank.iloc[j]) > cd,
            "p_wilcoxon": pw,
        })
    post = pd.DataFrame(pairs)
    post["p_holm"] = holm(post.p_wilcoxon)
    summary = {"factor": LABELS[factor], "test": test, "k": k, "blocks": N, "statistic": stat,
               "p": p, "kendall_W": w, "CD_nemenyi": cd, "alpha": alpha}
    return summary, mean_rank.sort_values(), post, cd


def cliques(mean_rank, cd):
    """Maximal groups of consecutive levels (by rank) whose difference is < CD."""
    names, r = list(mean_rank.index), mean_rank.to_numpy()
    out = []
    for i in range(len(r)):
        j = i
        while j + 1 < len(r) and r[j + 1] - r[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in out):
            out.append((i, j))
    return [(names[a], names[b], r[a], r[b]) for a, b in out]
