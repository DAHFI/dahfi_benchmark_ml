"""
Loads the CTU-UHB intrapartum CTG database (PhysioNet, WFDB format) and applies the five
pipelines of `preprocess_fhr.py` (SOTA, BASIC, FOURIER, SPLINES, LLR).

Database format (see the PhysioNet page):
    - 552 records, each with a <id>.hea + <id>.dat pair
    - each record has 2 signals at 4 Hz: FHR (fetal heart rate) and UC (uterine contractions)
    - the clinical data (pH, Apgar, ...) are stored as comments in the .hea file

Download:
    wget -r -N -c -np https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/

Usage:
    python process_ctu_uhb.py --data_dir data/ctu-uhb-ctgdb/1.0.0 --out_dir outputs/preprocessed_30 \\
        --f_cut 0.15 --cut_min 30
    (with --f_cut the number of basis functions depends on the segment length;
     without it, --n_basis is used)

Outputs (in --out_dir):
    preprocessed.npz   keys '<method>/<id>' and 'raw/<id>' (raw segment)
    fhr_<method>.npy   one matrix per method (n_records x n_samples), rows = records,
                       column j = j / fs seconds
    ids.npy            record id of each row of those matrices (same order)
    summary.csv        length, number and % of NaN per record and method
"""

import argparse
import csv
import glob
import os

import numpy as np

from preprocess_fhr import METHODS, plot_record, preprocess, segment


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def list_records(data_dir):
    """Record identifiers (name of each .hea file without extension)."""
    heas = sorted(glob.glob(os.path.join(data_dir, "*.hea")))
    return [os.path.splitext(os.path.basename(h))[0] for h in heas]


def load_record(data_dir, rid):
    """
    Reads a WFDB record and returns (fhr, fs, comments).
    fhr is a 1D array in bpm; missing values may be 0 or NaN (the preprocessing treats
    both as invalid).
    """
    import wfdb

    rec = wfdb.rdrecord(os.path.join(data_dir, rid))
    names = [n.upper() for n in rec.sig_name]
    idx = next((i for i, n in enumerate(names) if "FHR" in n), 0)  # first signal if no 'FHR'
    fhr = np.asarray(rec.p_signal[:, idx], dtype=float)
    return fhr, float(rec.fs), list(rec.comments or [])


# --------------------------------------------------------------------------
# Whole-database processing
# --------------------------------------------------------------------------
def process_database(
    data_dir,
    out_dir,
    methods=METHODS,
    max_records=None,
    plot_dir=None,
    align="start",
    **kwargs,
):
    """
    Loads each record and applies the requested pipelines.

    align : 'start' | 'end'. If records have different lengths (without cut_min), the
            matrices are padded with NaN: 'start' aligns the beginning of each record at
            column 0; 'end' aligns the end (delivery) and pads at the beginning.
    kwargs: n_basis, f_cut, cut_min, detrend_deg, bandwidth_s, max_gap_s, ... (passed to
            `preprocess`).
    """
    os.makedirs(out_dir, exist_ok=True)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    rids = list_records(data_dir)
    if not rids:
        raise FileNotFoundError(f"No .hea files found in {data_dir}")
    if max_records:
        rids = rids[:max_records]

    arrays, rows, failed = {}, [], []
    by_method = {m: {} for m in methods}  # {method: {id: signal}}
    for k, rid in enumerate(rids, 1):
        try:
            fhr, fs, _ = load_record(data_dir, rid)
            raw = segment(fhr, kwargs.get("cut_min"), fs)
            arrays[f"raw/{rid}"] = raw

            results = {}
            for m in methods:
                y = preprocess(fhr, m, fs=fs, **kwargs)
                results[m] = y
                arrays[f"{m}/{rid}"] = y
                by_method[m][rid] = y
                n_nan = int(np.isnan(y).sum())
                rows.append([rid, m, len(y), n_nan, round(100 * n_nan / len(y), 2)])

            if plot_dir:
                plot_record(raw, results, os.path.join(plot_dir, f"{rid}.png"), fs=fs, title=rid)
        except Exception as e:  # a bad record does not stop the rest
            failed.append((rid, str(e)))
            print(f"[{k}/{len(rids)}] {rid}: ERROR -> {e}")
            continue
        print(f"[{k}/{len(rids)}] {rid} ok")

    np.savez_compressed(os.path.join(out_dir, "preprocessed.npz"), **arrays)
    ok_ids = [r for r in rids if r not in {f for f, _ in failed}]
    if ok_ids:
        np.save(os.path.join(out_dir, "ids.npy"), np.array(ok_ids))
        for m, signals in by_method.items():  # one matrix (records x time) per method
            T = max(len(signals[r]) for r in ok_ids)
            mat = np.full((len(ok_ids), T), np.nan)
            for i, r in enumerate(ok_ids):
                y = signals[r]
                if align == "end":
                    mat[i, T - len(y):] = y
                else:
                    mat[i, : len(y)] = y
            np.save(os.path.join(out_dir, f"fhr_{m.lower()}.npy"), mat)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record", "method", "n_samples", "n_nan", "pct_nan"])
        w.writerows(rows)

    print(f"\nProcessed: {len(rids) - len(failed)} of {len(rids)} records -> {out_dir}")
    if failed:
        print("Failed:", ", ".join(r for r, _ in failed))
    return arrays


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="folder with the CTU-UHB .hea/.dat files")
    ap.add_argument("--out_dir", default="outputs/preprocessed", help="output folder")
    ap.add_argument("--n_basis", type=int, default=101, help="number of basis functions (FOURIER, SPLINES)")
    ap.add_argument(
        "--f_cut",
        type=float,
        default=None,
        help="cut-off frequency in Hz (FOURIER, SPLINES); if given, it replaces --n_basis and "
        "the number of basis functions depends on the segment length (e.g. 0.15)",
    )
    ap.add_argument("--cut_min", type=float, default=None, help="final minutes to keep (15, 30, ...)")
    ap.add_argument("--detrend_deg", type=int, default=2, help="detrending degree (SOTA)")
    ap.add_argument("--bandwidth_s", type=float, default=10.0, help="bandwidth in s (LLR)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--max_records", type=int, default=None, help="process only the first N (for testing)")
    ap.add_argument(
        "--align",
        choices=["start", "end"],
        default="start",
        help="alignment when records have different lengths (without --cut_min)",
    )
    ap.add_argument("--plot_dir", default=None, help="folder for one plot per record")
    a = ap.parse_args()

    process_database(
        a.data_dir,
        a.out_dir,
        methods=a.methods,
        max_records=a.max_records,
        plot_dir=a.plot_dir,
        align=a.align,
        n_basis=a.n_basis,
        f_cut=a.f_cut,
        cut_min=a.cut_min,
        detrend_deg=a.detrend_deg,
        bandwidth_s=a.bandwidth_s,
    )
