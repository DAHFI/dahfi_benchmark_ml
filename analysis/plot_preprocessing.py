"""
Compact figure of the five preprocessing pipelines on one CTU-UHB record.

Layout (one page wide, short):
    top    : the 30 min of the raw signal, with the long gaps shaded and the zoomed window
             marked
    bottom : 2 x 3 panels with the zoomed window (4 min by default): raw signal and the five
             pipelines (SOTA, BASIC, FOURIER, SPLINES, LLR), each over the raw signal in light grey

It uses exactly preprocessing/preprocess_fhr.preprocess. Give the number of basis functions
as in the preprocessing: --f_cut 0.15 (or --n_basis 101).

If --record is not given, "illustrative" records are chosen automatically: with at least one
short (interpolated) gap and one long gap (> 15 s), and enough valid signal. With
--n_examples N, N different figures are produced to choose from.

Never overwrites: figures are saved in --out_dir with the record id; if the file already
exists, a suffix _2, _3, ... is added.

Usage:
    python plot_preprocessing.py --data_dir data/ctu-uhb-ctgdb/1.0.0 --f_cut 0.15 --n_examples 5
    python plot_preprocessing.py --data_dir data/ctu-uhb-ctgdb/1.0.0 --f_cut 0.15 --record 1001 --zoom_start 12
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preprocessing"))
from preprocess_fhr import METHODS, clean_out_of_range, n_basis_from_cutoff, preprocess, segment, split_gaps  # noqa: E402

FS = 4.0
COLORS = {"SOTA": "#d6453d", "BASIC": "#0b0b0b", "FOURIER": "#2a78d6", "SPLINES": "#2e9e5b", "LLR": "#9b59b6"}
RAW, GAP, INK2 = "#b9b8b3", "#f1e3c8", "#52514e"


def load_fhr(data_dir, rid):
    import wfdb

    rec = wfdb.rdrecord(os.path.join(data_dir, str(rid)))
    names = [n.upper() for n in rec.sig_name]
    idx = next((i for i, n in enumerate(names) if "FHR" in n), 0)
    return np.asarray(rec.p_signal[:, idx], dtype=float)


def list_ids(data_dir):
    return sorted(os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith(".hea"))


def gap_runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def score_record(raw_seg):
    """How illustrative a record is (short and long gaps, valid signal)."""
    x = clean_out_of_range(raw_seg)
    short, long_ = split_gaps(x, FS)
    valid = np.isfinite(x).mean()
    if valid < 0.75 or not short.any() or not long_.any():
        return -1
    return valid + 0.5 * min(len(gap_runs(long_)), 3) / 3 + 0.5 * min(len(gap_runs(short)), 10) / 10


def best_window(raw_seg, zoom_len):
    """Start of the window with a long gap and the largest FHR variability."""
    x = clean_out_of_range(raw_seg)
    _, long_ = split_gaps(x, FS)
    step = int(30 * FS)
    best, best_s = 0, -np.inf
    for s in range(0, max(1, len(x) - zoom_len), step):
        w = x[s:s + zoom_len]
        lw = long_[s:s + zoom_len]
        if np.isfinite(w).mean() < 0.6:
            continue
        sc = np.nanstd(w) + (10 if lw.any() and not lw[:20].any() and not lw[-20:].any() else 0)
        if sc > best_s:
            best, best_s = s, sc
    return best


def unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    k = 2
    while os.path.exists(f"{base}_{k}{ext}"):
        k += 1
    return f"{base}_{k}{ext}"


def make_figure(raw_seg, results, rid, zoom_start, zoom_len, path, label_bases):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(raw_seg)
    t = (np.arange(n) - n) / FS / 60  # minutes before the end
    x = clean_out_of_range(raw_seg)
    _, long_ = split_gaps(x, FS)
    z0, z1 = zoom_start, min(zoom_start + zoom_len, n)

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8})
    fig = plt.figure(figsize=(7.2, 3.9))
    gs = GridSpec(3, 3, height_ratios=[0.8, 1, 1], hspace=0.55, wspace=0.08, figure=fig)

    # --- Overview
    ax0 = fig.add_subplot(gs[0, :])
    for s, e in gap_runs(long_):
        ax0.axvspan(t[s], t[min(e, n - 1)], color=GAP, lw=0)
    ax0.plot(t, x, color=INK2, lw=0.45)
    ax0.axvspan(t[z0], t[z1 - 1], facecolor="none", edgecolor="#0b0b0b", lw=0.9)
    ax0.set_xlim(t[0], 0)
    ax0.set_ylabel("bpm")
    ax0.set_title(f"Raw FHR, last {n / FS / 60:.0f} min (record {rid}); shaded: gaps > 15 s; box: zoomed window",
                  loc="left")

    # --- Zoomed window: raw + 5 pipelines
    tz = t[z0:z1]
    lo = np.nanmin([np.nanmin(x[z0:z1])] + [np.nanmin(r[z0:z1]) for r in results.values()])
    hi = np.nanmax([np.nanmax(x[z0:z1])] + [np.nanmax(r[z0:z1]) for r in results.values()])
    pad = 0.08 * (hi - lo)
    panels = [("Raw", None)] + [(m, results[m]) for m in METHODS]
    for k, (name, y) in enumerate(panels):
        ax = fig.add_subplot(gs[1 + k // 3, k % 3])
        for s, e in gap_runs(long_[z0:z1]):
            ax.axvspan(tz[s], tz[min(e, len(tz) - 1)], color=GAP, lw=0)
        if y is None:
            ax.plot(tz, x[z0:z1], color=INK2, lw=0.6)
        else:
            ax.plot(tz, x[z0:z1], color=RAW, lw=0.6)
            ax.plot(tz, y[z0:z1], color=COLORS[name], lw=1.0)
        title = name if name not in ("FOURIER", "SPLINES") else f"{name} ({label_bases[name]} bases)"
        ax.set_title(title, loc="left", pad=2)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(tz[0], tz[-1])
        if k % 3:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("bpm")
        if k < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("min before delivery")
    for a_ in fig.axes:
        for side in ("top", "right"):
            a_.spines[side].set_visible(False)
        a_.tick_params(length=2, colors=INK2)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="CTU-UHB folder (.hea/.dat)")
    ap.add_argument("--record", default=None, help="record id (otherwise chosen automatically)")
    ap.add_argument("--n_examples", type=int, default=5, help="number of automatic records")
    ap.add_argument("--cut_min", type=float, default=30)
    ap.add_argument("--n_basis", type=int, default=101, help="number of basis functions (as in preprocessing)")
    ap.add_argument("--f_cut", type=float, default=None, help="cut-off frequency (if used in preprocessing)")
    ap.add_argument("--bandwidth_s", type=float, default=10.0)
    ap.add_argument("--zoom_min", type=float, default=4.0, help="duration of the zoomed window (min)")
    ap.add_argument("--zoom_start", type=float, default=None,
                    help="minutes before the end at which the window STARTS (otherwise chosen automatically)")
    ap.add_argument("--out_dir", default="outputs/figures/preprocessing")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if a.record:
        ids = [a.record]
    else:
        print("Looking for illustrative records...", flush=True)
        scored = []
        for rid in list_ids(a.data_dir):
            try:
                seg = segment(load_fhr(a.data_dir, rid), a.cut_min, FS)
            except Exception:
                continue
            sc = score_record(seg)
            if sc > 0:
                scored.append((sc, rid))
        scored.sort(reverse=True)
        ids = [rid for _, rid in scored[: a.n_examples]]
        print(f"  {len(scored)} candidates; plotting: {', '.join(ids)}")

    kw = dict(n_basis=a.n_basis, f_cut=a.f_cut, bandwidth_s=a.bandwidth_s, cut_min=a.cut_min)
    zoom_len = int(a.zoom_min * 60 * FS)
    for rid in ids:
        raw = load_fhr(a.data_dir, rid)
        seg = segment(raw, a.cut_min, FS)
        results = {m: preprocess(raw, m, **kw) for m in METHODS}
        if a.f_cut:
            bases = {m: n_basis_from_cutoff(m, len(seg), FS, a.f_cut) for m in ("FOURIER", "SPLINES")}
        else:
            bases = {"FOURIER": a.n_basis, "SPLINES": a.n_basis}
        z0 = max(0, len(seg) - int(a.zoom_start * 60 * FS)) if a.zoom_start is not None else best_window(seg, zoom_len)
        path = unique_path(os.path.join(a.out_dir, f"preprocessing_{rid}.pdf"))
        make_figure(seg, results, rid, z0, zoom_len, path, bases)
        start_min = (len(seg) - z0) / FS / 60
        print(f"  {path} (and .png) · window: from -{start_min:.1f} to -{start_min - a.zoom_min:.1f} min")
    print("\nTo change the window of a record: --record ID --zoom_start MIN (minutes before the end)")


if __name__ == "__main__":
    main()
