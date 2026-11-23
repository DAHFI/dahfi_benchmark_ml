"""
Training and evaluation of the classifiers on the extracted features.

Experimental design
-------------------
For every combination of

    duration       : segment length (one feature folder per duration, e.g. 30 min)
    preprocessing  : SOTA, BASIC, FOURIER, SPLINES, LLR
    feature set (cumulative):
        M            morphological                (baseline, DCC, ACC)
        M+L          + linear                     (mean, std, LTV, delta, STV, II)
        M+L+NL       + nonlinear                  (ApEn, SampEn, LZC)
        M+L+NL+IBTF  + image-based time-frequency (16 GLCM features)
    model          : those of models.get_classical_models() (SVM, LASSO, RIDGE, RF, DT, KNN, ANN)
    seed           : several repetitions (--seeds)

a NESTED CROSS-VALIDATION is run [1-3]:

    - outer loop: StratifiedKFold (--outer, 5 folds) -> performance estimate;
    - inner loop: hyperparameter search (GridSearchCV or RandomizedSearchCV) with
      StratifiedKFold (--inner, 3 folds) inside each outer training partition,
      maximising the AUC.

The seed controls both the shuffling of the (outer and inner) partitions and the
random_state of the models. For a given seed, every pipeline uses exactly the same
partitions, so comparisons between pipelines are paired. Repeating the nested CV with
several seeds reduces the variability due to a particular partition [4].

No data leakage
---------------
    - Features are computed from each record separately (features/build_features.py).
    - Everything fitted on data is INSIDE the Pipeline: the imputer (median) and the
      StandardScaler. In each partition they are fitted only on the training data and then
      applied to the validation/test data [2, 5].
    - The hyperparameter search only sees the training part of each outer partition; the
      outer test part is not involved in any decision [1, 3].
    Tree models (RF, DT) are not scaled, because they are invariant to monotone
    transformations, but they do use the imputer. If a scale-sensitive model arrives from
    models.py without a scaler, a StandardScaler is added here. The Pipeline of each model is
    printed at the start.

Class imbalance (--class_weight)
--------------------------------
With few positives, models that minimise an unweighted loss tend to favour the majority
class [8, 9]. By default (--class_weight grid), in the models that support class_weight the
inner search chooses between None and 'balanced' as one more hyperparameter. KNN does not
support it. 'balanced' shifts the probabilities towards the positive class: it does not
change the AUC for that reason, but it does change the Brier score and the
sensitivity/specificity at the 0.5 threshold.

Label
-----
Acidemia from the umbilical artery pH, read from the .hea files of CTU-UHB [6]:
y = 1 if pH < --ph_threshold (7.15 by default). Alternative: --labels_csv with columns id,y.
Besides the AUC, the area under the precision-recall curve (average precision) is stored,
which is more informative with imbalanced classes [7].

Saving and resuming
-------------------
After each (duration, preprocessing, feature set, model, seed) unit, the results are
appended to:
    fold_results.csv   metrics of each outer partition and best hyperparameters
    predictions.csv    predicted probability for each test record (for ROC curves)
    completed.txt      finished units
If training is interrupted, running the same command again skips the completed units.
At the end, summary.csv is written (mean and standard deviation across seeds).
label_config.json stores the label definition, so results obtained with different labels
are never mixed in the same folder.

References
[1] Varma S, Simon R. Bias in error estimation when using cross-validation for model
    selection. BMC Bioinformatics. 2006;7:91.
[2] Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning. 2nd ed.
    Springer; 2009. Section 7.10.2.
[3] Cawley GC, Talbot NLC. On over-fitting in model selection and subsequent selection
    bias in performance evaluation. J Mach Learn Res. 2010;11:2079-2107.
[4] Krstajic D, Buturovic LJ, Leahy DE, Thomas S. Cross-validation pitfalls when selecting
    and assessing regression and classification models. J Cheminform. 2014;6:10.
[5] Pedregosa F, et al. Scikit-learn: machine learning in Python.
    J Mach Learn Res. 2011;12:2825-2830.
[6] Chudacek V, Spilka J, Bursa M, et al. Open access intrapartum CTG database.
    BMC Pregnancy Childbirth. 2014;14:16.
[7] Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot
    when evaluating binary classifiers on imbalanced datasets. PLoS One. 2015;10(3):e0118432.
[8] He H, Garcia EA. Learning from imbalanced data. IEEE Trans Knowl Data Eng.
    2009;21(9):1263-1284.
[9] King G, Zeng L. Logistic regression in rare events data. Polit Anal. 2001;9(2):137-163.

Usage
-----
    python train_models.py --features 30=outputs/features_30 \\
        --data_dir data/ctu-uhb-ctgdb/1.0.0 --out_dir outputs/results

    # several durations
    python train_models.py --features 15=outputs/features_15 30=outputs/features_30 ...

    # quick test
    python train_models.py ... --seeds 0 --outer 3 --models LASSO KNN --methods BASIC

    # show the pipeline and grid of each model without training
    python train_models.py --show_grid
"""

import argparse
import glob
import importlib
import json
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import BaseEnsemble
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from sklearn.tree import BaseDecisionTree

METHODS = ["SOTA", "BASIC", "FOURIER", "SPLINES", "LLR"]

# Columns of each family (names from feature_names.json, written by build_features.py)
FAMILIES = {
    "M": ["baseline", "DCC", "ACC"],
    "L": ["mean", "std", "LTV", "delta", "STV", "II"],
    "NL": ["ApEn015", "ApEn020", "SampEn015", "SampEn020", "LZC"],
    "IBTF": None,  # every column starting with "IBTF_"
}
FEATURE_SETS = {  # cumulative sets
    "M": ["M"],
    "M+L": ["M", "L"],
    "M+L+NL": ["M", "L", "NL"],
    "M+L+NL+IBTF": ["M", "L", "NL", "IBTF"],
}
UNIT_KEYS = ["duration", "preprocessing", "feature_set", "model", "seed"]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_features(features_dir, method):
    """Returns a DataFrame (index = id) with the features of one preprocessing pipeline."""
    with open(os.path.join(features_dir, "feature_names.json"), encoding="utf-8") as f:
        names = json.load(f)
    X = np.load(os.path.join(features_dir, f"features_{method.lower()}.npy"))
    df = pd.DataFrame(X, columns=names)
    df["id"] = df["id"].astype(int)
    return df.set_index("id")


def set_columns(columns, feature_set):
    """Columns of a feature set (in the order of the file)."""
    cols = []
    for fam in FEATURE_SETS[feature_set]:
        if FAMILIES[fam] is None:
            cols += [c for c in columns if c.startswith("IBTF_")]
        else:
            cols += [c for c in FAMILIES[fam] if c in columns]
    missing = [
        c
        for fam in FEATURE_SETS[feature_set]
        if FAMILIES[fam]
        for c in FAMILIES[fam]
        if c not in columns
    ]
    if missing:
        raise ValueError(f"Missing columns of feature set {feature_set}: {missing}")
    return cols


def read_ph_labels(data_dir, threshold):
    """Reads the umbilical artery pH from the CTU-UHB .hea files -> Series id -> y (0/1)."""
    rows = {}
    for hea in glob.glob(os.path.join(data_dir, "*.hea")):
        rid = int(os.path.splitext(os.path.basename(hea))[0])
        with open(hea, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = re.match(r"^#?\s*pH\s+([0-9]+(?:\.[0-9]+)?)\s*$", line.strip())
                if m:
                    rows[rid] = float(m.group(1))
                    break
    if not rows:
        raise SystemExit(
            f"No pH found in the .hea files of {data_dir}. Use --labels_csv with columns id,y."
        )
    ph = pd.Series(rows, name="pH").sort_index()
    return (ph < threshold).astype(int).rename("y"), ph


def load_labels(args):
    if args.labels_csv:
        lab = pd.read_csv(args.labels_csv)
        return lab.set_index(lab["id"].astype(int))["y"].astype(int)
    y, _ = read_ph_labels(args.data_dir, args.ph_threshold)
    print(
        f"Labels: {len(y)} records with pH, {int(y.sum())} with pH < {args.ph_threshold} "
        f"({100 * y.mean():.1f} %)"
    )
    return y


def parse_features(items):
    """['30=path', '15=path'] -> {30: 'path', 15: 'path'}; a bare path means 30 min."""
    out = {}
    for it in items:
        dur, path = it.split("=", 1) if "=" in it else ("30", it)
        out[int(float(dur))] = path
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
SCALERS = (StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler)


def _grids(grid):
    """The grid as a list of dicts (scikit-learn accepts a dict or a list of dicts)."""
    return [dict(g) for g in grid] if isinstance(grid, (list, tuple)) else [dict(grid)]


def _final_estimator(est):
    while isinstance(est, Pipeline):
        est = est.steps[-1][1]
    if type(est).__name__ == "CalibratedClassifierCV":
        est = _final_estimator(est.estimator)
    return est


def _flat_steps(est):
    """Pipeline steps, unfolding nested Pipelines (for printing)."""
    if not isinstance(est, Pipeline):
        return [type(est).__name__]
    return [n for _, s in est.steps for n in _flat_steps(s)]


def has_scaler(est):
    return isinstance(est, SCALERS) or any(
        isinstance(v, SCALERS) for v in est.get_params(deep=True).values()
    )


def is_tree_based(est):
    return isinstance(_final_estimator(est), (BaseDecisionTree, BaseEnsemble))


def with_imputer(model, grid):
    """Pipeline: imputer (median) -> [StandardScaler if needed] -> model.

    Everything is fitted within each partition (no data leakage). Scaling is set by
    models.get_classical_model(use_scaler=True); if a scale-sensitive model (SVM, logistic
    regression, KNN, ANN) arrives without a scaler, a StandardScaler is added here. Trees do
    not need it. Hyperparameters get the prefix 'clf__' (e.g. 'clf__model__C').
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    added_scaler = not has_scaler(model) and not is_tree_based(model)
    if added_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", model))
    grids = [{f"clf__{k}": v for k, v in g.items()} for g in _grids(grid)]
    return Pipeline(steps), (grids if len(grids) > 1 else grids[0]), added_scaler


def class_weight_param(pipe):
    """Name of the class_weight parameter of the final classifier, or None if unsupported (KNN)."""
    keys = [
        k
        for k in pipe.get_params(deep=True)
        if k == "class_weight" or k.endswith("__class_weight")
    ]
    return min(keys, key=len) if keys else None


def apply_class_weight(pipe, grid, mode):
    """Class-imbalance correction through class_weight [8, 9].

    mode = 'grid'     : class_weight in {None, 'balanced'} is chosen in the inner search,
                        as one more hyperparameter (unless models.py already has it in the grid)
           'balanced' : fixed to 'balanced' in every model that supports it
           'none'     : fixed to None (no correction)
           'model'    : untouched, whatever models.py defines
    'balanced' weights each class by n / (2 * n_class): with 20 % positives, each positive
    weighs 4 times more than a negative and both classes contribute equally to the loss.
    Returns (pipe, grid, description).
    """
    key = class_weight_param(pipe)
    if key is None:
        return pipe, grid, "class_weight not supported"
    if mode == "model":
        return pipe, grid, f"class_weight from models.py ({pipe.get_params()[key]})"
    grids = _grids(grid)
    if mode == "grid":
        if all(key in g for g in grids):
            return pipe, grid, f"class_weight in the grid of models.py: {grids[0][key]}"
        for g in grids:
            g.setdefault(key, [None, "balanced"])
        desc = "class_weight in {None, 'balanced'} (inner search)"
    else:
        value = "balanced" if mode == "balanced" else None
        pipe.set_params(**{key: value})
        for g in grids:
            g.pop(key, None)
        desc = f"class_weight fixed = {value}"
    return pipe, (grids if len(grids) > 1 else grids[0]), desc


def n_combinations(grid):
    return int(sum(np.prod([len(v) for v in g.values()]) for g in _grids(grid)))


def make_search(estimator, grid, args, seed):
    inner = StratifiedKFold(n_splits=args.inner, shuffle=True, random_state=seed)
    if args.search == "random":
        return RandomizedSearchCV(
            estimator,
            grid,
            n_iter=min(args.n_iter, n_combinations(grid)),
            scoring="roc_auc",
            cv=inner,
            n_jobs=args.n_jobs,
            random_state=seed,
            refit=True,
        )
    return GridSearchCV(
        estimator, grid, scoring="roc_auc", cv=inner, n_jobs=args.n_jobs, refit=True
    )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def fold_metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {
        "auc": roc_auc_score(y_true, prob),
        "ap": average_precision_score(y_true, prob),
        "brier": brier_score_loss(y_true, prob),
        "bal_acc": balanced_accuracy_score(y_true, pred),
        "sens": tp / (tp + fn) if tp + fn else np.nan,
        "spec": tn / (tn + fp) if tn + fp else np.nan,
    }


# --------------------------------------------------------------------------
# Saving / resuming
# --------------------------------------------------------------------------
def unit_id(d):
    return "|".join(str(d[k]) for k in UNIT_KEYS)


def load_done(out_dir):
    path = os.path.join(out_dir, "completed.txt")
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_csv(path, rows):
    pd.DataFrame(rows).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False
    )


def save_unit(out_dir, unit, fold_rows, pred_rows):
    """Appends the results of a unit and marks it as completed (in this order: if the run is
    interrupted in between, the unit is not marked as completed and is repeated)."""
    append_csv(os.path.join(out_dir, "fold_results.csv"), fold_rows)
    append_csv(os.path.join(out_dir, "predictions.csv"), pred_rows)
    with open(os.path.join(out_dir, "completed.txt"), "a", encoding="utf-8") as f:
        f.write(unit_id(unit) + "\n")
        f.flush()
        os.fsync(f.fileno())


def completed_rows(out_dir, filename):
    """Rows of the completed units (discards leftovers of an interrupted unit)."""
    path = os.path.join(out_dir, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    keep = df[UNIT_KEYS].astype(str).agg("|".join, axis=1).isin(load_done(out_dir))
    subset = (
        UNIT_KEYS + (["fold"] if "fold" in df else []) + (["id"] if "id" in df else [])
    )
    return df[keep].drop_duplicates(subset=subset, keep="last")


def write_summary(out_dir):
    df = completed_rows(out_dir, "fold_results.csv")
    if df.empty:
        return None
    metrics = ["auc", "ap", "brier", "bal_acc", "sens", "spec"]
    cell = ["duration", "preprocessing", "feature_set", "model"]
    # First average over the partitions of each seed and then across seeds, so that the
    # standard deviation reflects the variability between repetitions.
    per_seed = df.groupby(cell + ["seed"])[metrics].mean()
    summ = per_seed.groupby(cell).agg(["mean", "std"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ["n_seeds"] = per_seed.groupby(cell).size()
    summ = summ.reset_index().sort_values(
        ["duration", "auc_mean"], ascending=[True, False]
    )
    summ.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    return summ


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def check_label_config(args):
    """Prevents mixing, in the same folder, results trained with different labels.

    The pH threshold (or the labels CSV) and the class_weight mode are stored in
    label_config.json the first time. When resuming, the run stops if they do not match:
    otherwise completed.txt would skip units trained with other labels.
    """
    cfg = (
        {"labels_csv": os.path.abspath(args.labels_csv)}
        if args.labels_csv
        else {"ph_threshold": args.ph_threshold}
    )
    cfg["class_weight"] = args.class_weight
    path = os.path.join(args.out_dir, "label_config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        if old != cfg:
            raise SystemExit(
                f"{args.out_dir} contains results with another configuration ({old}); "
                f"now {cfg} is requested. Use another --out_dir."
            )
    elif os.path.exists(os.path.join(args.out_dir, "completed.txt")):
        raise SystemExit(
            f"{args.out_dir} already has results but no label_config.json. "
            "Use another --out_dir so that results are not mixed."
        )
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)


def show_grids(args):
    """Prints the final Pipeline and hyperparameter grid of each model, without training."""
    models_mod = importlib.import_module(args.models_module)
    names = [
        m.upper() for m in (args.models or models_mod.get_classical_models().keys())
    ]
    for name in names:
        model, grid = models_mod.get_classical_model(
            name, random_state=0, use_scaler=True
        )
        est, grid, added = with_imputer(model, grid)
        est, grid, cw_desc = apply_class_weight(est, grid, args.class_weight)
        clf = _final_estimator(est)
        print(
            f"\n=== {name} ===  ({type(clf).__name__}; {n_combinations(grid)} combinations; "
            f"{n_combinations(grid) * args.inner} fits per outer partition)"
        )
        print(
            "  Pipeline : "
            + " -> ".join(_flat_steps(est))
            + ("  [StandardScaler added by train_models]" if added else "")
        )
        print(f"  Imbalance: {cw_desc}")
        for i, g in enumerate(_grids(grid)):
            if len(_grids(grid)) > 1:
                print(f"  Sub-grid {i + 1}:")
            for k, v in g.items():
                print(f"    {k.split('__')[-1]:22s} {list(v)}")


def run(args):
    if args.show_grid:
        show_grids(args)
        return
    os.makedirs(args.out_dir, exist_ok=True)
    check_label_config(args)
    models_mod = importlib.import_module(args.models_module)
    y_all = load_labels(args)
    feature_dirs = parse_features(args.features)

    model_names = [
        m.upper() for m in (args.models or models_mod.get_classical_models().keys())
    ]
    units = [
        (d, p, g, m, s)
        for d in feature_dirs
        for p in args.methods
        for g in args.feature_sets
        for m in model_names
        for s in args.seeds
    ]
    done = load_done(args.out_dir)
    pending = [u for u in units if unit_id(dict(zip(UNIT_KEYS, u))) not in done]
    print(
        f"Units: {len(units)} in total, {len(units) - len(pending)} already completed, "
        f"{len(pending)} pending.\n"
    )

    cache, described = {}, set()
    t_start = time.time()
    for i, (dur, method, fset, model_name, seed) in enumerate(pending, 1):
        unit = dict(zip(UNIT_KEYS, (dur, method, fset, model_name, seed)))

        if (dur, method) not in cache:
            cache[(dur, method)] = load_features(feature_dirs[dur], method)
        feats = cache[(dur, method)]
        ids = feats.index.intersection(y_all.index)
        cols = set_columns(feats.columns, fset)
        X = feats.loc[ids, cols].to_numpy(dtype=float)
        y = y_all.loc[ids].to_numpy(dtype=int)
        if y.sum() < args.outer:
            raise SystemExit(
                f"Only {y.sum()} positive cases: fewer than outer partitions ({args.outer})."
            )

        model, grid = models_mod.get_classical_model(
            model_name, random_state=seed, use_scaler=True
        )
        estimator, grid, added_scaler = with_imputer(model, grid)
        estimator, grid, cw_desc = apply_class_weight(
            estimator, grid, args.class_weight
        )
        if model_name not in described:  # once per model: which pipeline is used
            described.add(model_name)
            scaling = (
                "StandardScaler added by train_models"
                if added_scaler
                else (
                    "scaler from models.py"
                    if has_scaler(model)
                    else "no scaling (trees do not need it)"
                )
            )
            print(
                f"  {model_name}: {' -> '.join(_flat_steps(estimator))}\n"
                f"    scaling: {scaling} · imbalance: {cw_desc} · "
                f"{n_combinations(grid)} hyperparameter combinations",
                flush=True,
            )
        outer = StratifiedKFold(n_splits=args.outer, shuffle=True, random_state=seed)

        t0 = time.time()
        fold_rows, pred_rows = [], []
        for fold, (tr, te) in enumerate(outer.split(X, y)):
            search = make_search(clone(estimator), grid, args, seed)
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )  # convergence warnings during the search
                search.fit(X[tr], y[tr])
            prob = search.predict_proba(X[te])[:, 1]
            fold_rows.append(
                {
                    **unit,
                    "fold": fold,
                    "n_features": len(cols),
                    "n_train": len(tr),
                    "n_test": len(te),
                    "n_pos_test": int(y[te].sum()),
                    **fold_metrics(y[te], prob),
                    "inner_auc": search.best_score_,
                    "best_params": json.dumps(search.best_params_, default=str),
                }
            )
            pred_rows += [
                {**unit, "fold": fold, "id": int(rid), "y": int(yy), "prob": float(pp)}
                for rid, yy, pp in zip(ids[te], y[te], prob)
            ]

        save_unit(args.out_dir, unit, fold_rows, pred_rows)
        auc = np.mean([r["auc"] for r in fold_rows])
        elapsed = time.time() - t_start
        eta = elapsed / i * (len(pending) - i)
        print(
            f"[{i}/{len(pending)}] {dur} min · {method:7s} · {fset:12s} · {model_name:5s} · seed {seed}"
            f" -> AUC {auc:.3f}  ({time.time() - t0:.0f} s; ~{eta / 60:.0f} min left)",
            flush=True,
        )

    summ = write_summary(args.out_dir)
    if summ is not None:
        print("\nBest combinations (mean AUC across seeds):")
        cols = [
            "duration",
            "preprocessing",
            "feature_set",
            "model",
            "auc_mean",
            "auc_std",
            "ap_mean",
            "n_seeds",
        ]
        for _, g in summ.groupby("duration"):
            print(g[cols].head(10).round(3).to_string(index=False))
        print(
            f"\nResults in {args.out_dir} (summary.csv, fold_results.csv, predictions.csv)"
        )


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--features",
        nargs="+",
        default=[],
        help="feature folders of build_features.py as DURATION=FOLDER (e.g. 30=outputs/features_30)",
    )
    ap.add_argument("--data_dir", help="CTU-UHB folder (.hea) to read the pH")
    ap.add_argument(
        "--labels_csv", help="alternative to --data_dir: CSV with columns id,y"
    )
    ap.add_argument(
        "--ph_threshold", type=float, default=7.15, help="y = 1 if pH < threshold"
    )
    ap.add_argument(
        "--show_grid",
        action="store_true",
        help="print the Pipeline and grid of each model and exit (no training)",
    )
    ap.add_argument(
        "--class_weight",
        choices=["grid", "balanced", "none", "model"],
        default="grid",
        help="imbalance correction in models supporting class_weight: "
        "'grid' = {None, 'balanced'} as a hyperparameter (default); "
        "'balanced'/'none' = fixed; 'model' = whatever models.py says",
    )
    ap.add_argument("--out_dir", default="outputs/results", help="results folder")
    ap.add_argument(
        "--models_module",
        default="models",
        help="module with get_classical_models / get_classical_model",
    )
    ap.add_argument(
        "--models", nargs="+", default=None, help="models to train (default: all)"
    )
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument(
        "--feature_sets",
        nargs="+",
        default=list(FEATURE_SETS),
        choices=list(FEATURE_SETS),
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="seeds of the repeated nested CV",
    )
    ap.add_argument("--outer", type=int, default=5, help="outer partitions")
    ap.add_argument("--inner", type=int, default=3, help="inner partitions")
    ap.add_argument("--search", choices=["grid", "random"], default="grid")
    ap.add_argument(
        "--n_iter", type=int, default=30, help="combinations with --search random"
    )
    ap.add_argument(
        "--n_jobs", type=int, default=-1, help="processes for the search (-1 = all)"
    )
    return ap


if __name__ == "__main__":
    a = build_parser().parse_args()
    if not a.show_grid:
        if not a.labels_csv and not a.data_dir:
            raise SystemExit(
                "Give --data_dir (to read the pH from the .hea files) or --labels_csv."
            )
        if not a.features:
            raise SystemExit(
                "Give at least one feature folder with --features 30=FOLDER."
            )
    run(a)
