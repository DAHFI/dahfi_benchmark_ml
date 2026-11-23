"""
Builds, for each preprocessing pipeline, a .npy file with the record id + all features.

Input (folder --in_dir, created by preprocessing/process_ctu_uhb.py):
    ids.npy                        record id of each row
    fhr_sota.npy, fhr_basic.npy, fhr_fourier.npy, fhr_splines.npy, fhr_llr.npy
                                   matrices (n_records x n_samples), rows = records

Output (folder --out_dir):
    features_sota.npy, features_basic.npy, features_fourier.npy,
    features_splines.npy, features_llr.npy
        matrix (n_records x (1 + n_features)):
            column 0     = record id (as a number)
            columns 1..  = features, in the order of feature_names.json
    feature_names.json             column names (including 'id')

Feature groups (in this column order):
    linear (6) -> morphological (3) -> nonlinear (5) -> ibtf (16) -> clinical (optional)

Usage:
    python build_features.py --in_dir outputs/preprocessed_30 --out_dir outputs/features_30
    python build_features.py --in_dir ... --out_dir ... --methods SOTA LLR --max_records 5
    python build_features.py --in_dir ... --out_dir ... --groups linear morphological clinical \\
        --clinical_csv clinical.csv --clinical_numeric Age "Gest. weeks" --clinical_binary Parity
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # never open windows

import argparse
import contextlib
import importlib
import io
import json
import sys
import time
import warnings

import numpy as np

METHODS = ["SOTA", "BASIC", "FOURIER", "SPLINES", "LLR"]
GROUPS = ["linear", "morphological", "nonlinear", "ibtf", "clinical"]

FEATURE_NAMES = {
    "linear": ["mean", "std", "LTV", "delta", "STV", "II"],
    "morphological": ["baseline", "DCC", "ACC"],
    "nonlinear": ["ApEn015", "ApEn020", "SampEn015", "SampEn020", "LZC"],
    "ibtf": [f"IBTF_{band}_{prop}"
             for band in ("VLF", "LF", "MF", "HF")
             for prop in ("contrast", "correlation", "energy", "homogeneity")],
}
MODULES = {  # group -> (module, function)
    "linear": ("linear_features", "get_linear_features"),
    "morphological": ("morphological_features", "get_morphological_features"),
    "nonlinear": ("nonlinear_features", "get_nonlinear_features"),
    "ibtf": ("ibtf_features", "get_IBTF_features"),
}
_FUNCS = {}


# --------------------------------------------------------------------------
def _trim_padding(row):
    """Removes trailing NaN (padding of matrices with records of different lengths)."""
    valid = np.flatnonzero(~np.isnan(row))
    return row[: valid[-1] + 1] if len(valid) else row[:0]


def load_modules(groups):
    """Imports the feature modules at the start, so that an import error stops the script
    with a clear message instead of leaving NaN in every record."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for g in groups:
        if g not in MODULES:
            continue
        mod, fn = MODULES[g]
        try:
            _FUNCS[g] = getattr(importlib.import_module(mod), fn)
        except Exception as e:
            raise SystemExit(f"Could not import {mod}.py ({type(e).__name__}: {e}).")


def compute_group(group, row):
    """Features of one group for ONE signal. Returns a 1D vector."""
    # Silences the prints of each module and numpy RuntimeWarnings on all-NaN blocks
    # (long gaps): "Mean of empty slice", "All-NaN slice encountered". Results are unchanged.
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.asarray(_FUNCS[group](row[None, :]), dtype=float)[:, 0]


def compute_clinical(ids, args):
    """Clinical features of all ids (computed once)."""
    from clinical_features import get_clinical_features

    if not args.clinical_csv:
        raise SystemExit("The 'clinical' group needs --clinical_csv (and the column lists).")
    names, X = get_clinical_features(list(ids), args.clinical_csv, args.clinical_numeric,
                                     args.clinical_binary, get_names=True)  # raw: NaN if missing
    return list(names), np.asarray(X, dtype=float)


# --------------------------------------------------------------------------
def build(in_dir, out_dir, methods, groups, max_records=None, args=None):
    os.makedirs(out_dir, exist_ok=True)
    load_modules(groups)

    ids = np.load(os.path.join(in_dir, "ids.npy"))
    if max_records:
        ids = ids[:max_records]
    try:
        ids_num = ids.astype(float)
    except ValueError:
        raise SystemExit("Record ids are not numeric, so they cannot go in column 0 of a numeric matrix.")

    # --- column names ------------------------------------------------------
    names = ["id"]
    clinical_X = None
    for g in groups:
        if g == "clinical":
            cl_names, clinical_X = compute_clinical(ids, args)
            names += cl_names
        else:
            names += FEATURE_NAMES[g]
    with open(os.path.join(out_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)

    # --- features of each pipeline --------------------------------------------
    for m in methods:
        path = os.path.join(in_dir, f"fhr_{m.lower()}.npy")
        signals = np.load(path)[: len(ids)]
        if signals.shape[0] != len(ids):
            raise SystemExit(f"{path}: {signals.shape[0]} rows but there are {len(ids)} ids.")

        n_valid = int(np.isfinite(signals).sum())
        print(f"[{m}] {path}: shape={signals.shape}, valid values={n_valid}")
        if n_valid == 0:
            raise SystemExit(f"{path} has no valid values (all NaN). Check process_ctu_uhb.py.")

        errors = {g: 0 for g in groups}
        first_error = {}
        out = np.full((len(ids), len(names)), np.nan)
        out[:, 0] = ids_num
        t0 = time.time()

        for i, row in enumerate(signals):
            row = _trim_padding(row)
            col = 1
            for g in groups:
                if g == "clinical":
                    k = clinical_X.shape[1]
                    out[i, col:col + k] = clinical_X[i]
                    col += k
                    continue
                k = len(FEATURE_NAMES[g])
                if len(row):
                    try:
                        out[i, col:col + k] = compute_group(g, row)
                    except Exception as e:  # one failure does not stop the rest
                        errors[g] += 1
                        first_error.setdefault(g, f"{type(e).__name__}: {e}")
                        if errors[g] <= 3:
                            print(f"  [{m}] record {ids[i]}, group {g}: ERROR -> {type(e).__name__}: {e}")
                col += k
            if (i + 1) % 10 == 0 or i + 1 == len(ids):
                print(f"[{m}] {i + 1}/{len(ids)}  ({time.time() - t0:.0f} s)")

        for g, n in errors.items():
            if n:
                print(f"[{m}] WARNING: group '{g}' failed in {n} of {len(ids)} records. "
                      f"First error: {first_error[g]}")
        np.save(os.path.join(out_dir, f"features_{m.lower()}.npy"), out)

        n_nan = np.isnan(out[:, 1:]).sum(axis=0)
        with_nan = {names[j + 1]: int(n) for j, n in enumerate(n_nan) if n}
        print(f"[{m}] saved features_{m.lower()}.npy  shape={out.shape}")
        if with_nan:
            print(f"[{m}] features with NaN (number of records): {with_nan}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir", required=True, help="folder with ids.npy and fhr_<method>.npy")
    ap.add_argument("--out_dir", default="outputs/features", help="output folder")
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--groups", nargs="+", default=["linear", "morphological", "nonlinear", "ibtf"],
                    choices=GROUPS, help="feature groups (add 'clinical' if needed)")
    ap.add_argument("--max_records", type=int, default=None, help="only the first N (for testing)")
    ap.add_argument("--clinical_csv", default=None, help="CSV with clinical data (first column = id)")
    ap.add_argument("--clinical_numeric", nargs="*", default=[], help="numeric clinical columns")
    ap.add_argument("--clinical_binary", nargs="*", default=[], help="binary clinical columns")
    a = ap.parse_args()
    build(a.in_dir, a.out_dir, a.methods, a.groups, a.max_records, a)
