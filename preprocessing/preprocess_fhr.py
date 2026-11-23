"""
Fetal heart rate (FHR) preprocessing pipelines: SOTA, BASIC, FOURIER, SPLINES and LLR.

Common steps:
    values >= 200 or <= 50 bpm  ->  NaN
    trailing missing samples are discarded (they are not part of the signal)
    (optional) keep only the last X minutes (15, 30, ...)

Pipelines:
    SOTA   : interpolate the whole signal -> store the positions of gaps > 15 s
             -> second-order polynomial detrending -> restore (NaN) the gaps > 15 s
    BASIC  : interpolate gaps <= 15 s (gaps > 15 s stay as NaN)
    FOURIER: gaps <= 15 s interpolated, gaps > 15 s temporarily filled with the mean
             -> least-squares smoothing with a Fourier basis -> restore gaps > 15 s
    SPLINES: same, with a cubic B-spline basis
    LLR    : same, with local linear regression (Gaussian kernel)

The number of basis functions of FOURIER and SPLINES can be given directly (n_basis) or
through a cut-off frequency (f_cut), in which case it is derived from the segment length
(see n_basis_from_cutoff).

Quick use:
    from preprocess_fhr import preprocess
    y = preprocess(fhr, method="FOURIER", f_cut=0.15, fs=4, cut_min=30)

Dependencies: numpy, scipy (and matplotlib only for plots).
"""

import argparse
import os

import numpy as np
from scipy.interpolate import BSpline

METHODS = ("SOTA", "BASIC", "FOURIER", "SPLINES", "LLR")

# Defaults
FS = 4.0  # Hz (CTU-UHB)
MIN_BPM = 50.0
MAX_BPM = 200.0
MAX_GAP_S = 15.0  # seconds: gaps up to this length are interpolated


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def cut_final_segment(fhr, minutes, fs=FS):
    """Keeps the last `minutes` minutes of the signal (None = no cut)."""
    fhr = np.asarray(fhr, dtype=float)
    if minutes is None:
        return fhr
    n = int(round(minutes * 60 * fs))
    return fhr[-n:] if n < len(fhr) else fhr


def clean_out_of_range(fhr, low=MIN_BPM, high=MAX_BPM):
    """Values >= high or <= low become NaN (existing NaN and zeros stay missing)."""
    x = np.asarray(fhr, dtype=float).copy()
    with np.errstate(invalid="ignore"):
        x[(x >= high) | (x <= low)] = np.nan
    return x


def segment(fhr, minutes=None, fs=FS):
    """
    Segment of the raw signal (values unchanged) that will be preprocessed:
    1) trailing missing samples are discarded, counting out-of-range values
       (>= 200 or <= 50 bpm, zeros included) as missing;
    2) if `minutes` is not None, the last `minutes` minutes of what remains are kept.
    """
    x = np.asarray(fhr, dtype=float)
    valid = ~np.isnan(clean_out_of_range(x))
    if not valid.any():
        raise ValueError("The signal has no valid values.")
    last = np.where(valid)[0][-1]
    return cut_final_segment(x[: last + 1], minutes, fs)


def _nan_runs(nan_mask):
    """Returns (starts, exclusive_ends) of each run of consecutive NaN."""
    padded = np.concatenate(([0], nan_mask.astype(int), [0]))
    d = np.diff(padded)
    return np.where(d == 1)[0], np.where(d == -1)[0]


def split_gaps(x, fs=FS, max_gap_s=MAX_GAP_S):
    """Boolean masks of short (<= max_gap_s) and long (> max_gap_s) gaps."""
    nan_mask = np.isnan(x)
    short = np.zeros(len(x), dtype=bool)
    long_ = np.zeros(len(x), dtype=bool)
    for s, e in zip(*_nan_runs(nan_mask)):
        if (e - s) / fs <= max_gap_s:
            short[s:e] = True
        else:
            long_[s:e] = True
    return short, long_


def _interp_linear(x):
    """Linear interpolation of all NaN (the ends take the nearest valid value)."""
    idx = np.arange(len(x))
    valid = ~np.isnan(x)
    if not valid.any():
        raise ValueError("The signal has no valid values.")
    return np.interp(idx, idx[valid], x[valid])


# --------------------------------------------------------------------------
# Base pipelines
# --------------------------------------------------------------------------
def sota(x, fs=FS, max_gap_s=MAX_GAP_S, detrend_deg=2):
    """
    SOTA (Comert et al.): interpolates the signal, stores the position of the gaps > 15 s,
    removes a polynomial trend (fitted only on the observed values) and restores (NaN) the
    gaps > 15 s. The mean is added back after detrending so the signal stays in bpm.
    """
    _, long_mask = split_gaps(x, fs, max_gap_s)
    filled = _interp_linear(x)

    t = np.arange(len(x), dtype=float)
    obs = ~np.isnan(x)
    poly = np.polynomial.Polynomial.fit(t[obs], x[obs], deg=detrend_deg)
    out = filled - poly(t) + np.nanmean(x)
    out[long_mask] = np.nan
    return out


def basic(x, fs=FS, max_gap_s=MAX_GAP_S):
    """BASIC: interpolates gaps <= 15 s. Gaps > 15 s stay as NaN."""
    y = x.copy()
    short, _ = split_gaps(x, fs, max_gap_s)
    valid = ~np.isnan(x)
    idx = np.arange(len(x))
    if not valid.any():
        raise ValueError("The signal has no valid values.")
    y[short] = np.interp(idx[short], idx[valid], x[valid])
    return y


def fill_gaps(x, fs=FS, max_gap_s=MAX_GAP_S):
    """For FOURIER/SPLINES/LLR (which do not accept NaN): gaps <= 15 s are interpolated
    and gaps > 15 s are filled with the mean of the valid values."""
    y = basic(x, fs, max_gap_s)
    y[np.isnan(y)] = np.nanmean(x)
    return y


# --------------------------------------------------------------------------
# Smoothers (applied to the gap-filled signal, see fill_gaps)
# --------------------------------------------------------------------------
def _lstsq_smooth(B, y, w):
    """Least squares with binary weights w on the design matrix B."""
    coef, *_ = np.linalg.lstsq(B[w], y[w], rcond=None)
    return B @ coef


def fourier_basis(n, n_basis):
    """(n x n_basis) Fourier basis matrix: 1, sin(t), cos(t), sin(2t), cos(2t), ...
    The period is the length of the record. An odd n_basis is recommended."""
    if n_basis < 1:
        raise ValueError("n_basis must be >= 1")
    t = 2 * np.pi * np.arange(n) / n
    cols = [np.ones(n)]
    k = 1
    while len(cols) < n_basis:
        cols.append(np.sin(k * t))
        if len(cols) < n_basis:
            cols.append(np.cos(k * t))
        k += 1
    return np.column_stack(cols)


def bspline_basis(n, n_basis, degree=3):
    """(n x n_basis) B-spline basis matrix of the given degree with equally spaced interior
    knots. n_basis must be >= degree + 1."""
    if n_basis < degree + 1:
        raise ValueError(f"For splines of degree {degree}, n_basis must be >= {degree + 1}")
    n_interior = n_basis - degree - 1
    interior = np.linspace(0, 1, n_interior + 2)[1:-1]
    knots = np.r_[np.zeros(degree + 1), interior, np.ones(degree + 1)]
    x = np.arange(n) / (n - 1)
    return BSpline(knots, np.eye(n_basis), degree)(x)


def n_basis_from_cutoff(method, n, fs, f_cut, degree=3):
    """
    Number of basis functions that preserves FHR oscillations up to `f_cut` Hz in a segment
    of n samples at fs Hz (duration T = n / fs seconds).

    FOURIER : harmonic m has frequency m / T, so harmonics m = 1 .. floor(f_cut * T) are kept,
              plus the constant:  K = 2 * floor(f_cut * T) + 1.
    SPLINES : knot spacing of 1 / (2 * f_cut) s (two knots per period of the fastest
              oscillation), i.e. round(2 * f_cut * T) intervals and K = intervals + degree.

    Example (f_cut = 0.15 Hz): 15 min -> K Fourier 271, K splines 273;
                               30 min -> K Fourier 541, K splines 543.
    """
    if not 0 < f_cut <= fs / 2:
        raise ValueError(f"f_cut must be between 0 and fs/2 = {fs / 2} Hz")
    T = n / fs
    if method == "FOURIER":
        k = 2 * int(np.floor(f_cut * T)) + 1
    else:
        k = max(int(round(2 * f_cut * T)), 1) + degree
    return int(min(k, n))


def fourier_smooth(y, w, n_basis):
    """Least-squares smoothing with a Fourier basis (fourier_basis)."""
    return _lstsq_smooth(fourier_basis(len(y), n_basis), y, w)


def bspline_smooth(y, w, n_basis, degree=3):
    """Least-squares smoothing with B-splines (bspline_basis)."""
    return _lstsq_smooth(bspline_basis(len(y), n_basis, degree), y, w)


def llr_smooth(y, w, bandwidth_s=10.0, fs=FS):
    """Local linear regression with a Gaussian kernel (standard deviation = bandwidth_s
    seconds). Computed with convolutions, so it is fast even for long records."""
    h = bandwidth_s * fs  # bandwidth in samples
    half = int(np.ceil(4 * h))
    m = np.arange(-half, half + 1, dtype=float)
    K = np.exp(-0.5 * (m / h) ** 2)
    k0, k1, k2 = K, -m * K, m**2 * K  # kernel moments (x_j - x_i = -m)

    wt = w.astype(float)
    wy = wt * y

    def conv(a, k):
        return np.convolve(a, k, mode="same")

    S0, S1, S2 = conv(wt, k0), conv(wt, k1), conv(wt, k2)
    T0, T1 = conv(wy, k0), conv(wy, k1)

    den = S0 * S2 - S1**2
    with np.errstate(divide="ignore", invalid="ignore"):
        fit = np.where(np.abs(den) > 1e-12, (S2 * T0 - S1 * T1) / den, T0 / S0)
    return fit


# --------------------------------------------------------------------------
# Main function
# --------------------------------------------------------------------------
def preprocess(
    fhr,
    method,
    n_basis=101,
    fs=FS,
    cut_min=None,
    max_gap_s=MAX_GAP_S,
    detrend_deg=2,
    spline_degree=3,
    bandwidth_s=10.0,
    f_cut=None,
):
    """
    Applies one of the five pipelines to a 1D FHR signal.

    Parameters
    ----------
    fhr          : 1D array with the FHR (bpm).
    method       : 'SOTA' | 'BASIC' | 'FOURIER' | 'SPLINES' | 'LLR'
    n_basis      : number of basis functions (FOURIER and SPLINES). Ignored if f_cut is given.
    f_cut        : cut-off frequency in Hz (FOURIER and SPLINES). If given, the number of
                   basis functions is computed from the length of each segment
                   (n_basis_from_cutoff), so the resolution is the same for any duration.
                   E.g. 0.15 Hz keeps the VLF and LF bands.
    fs           : sampling frequency in Hz.
    cut_min      : if not None, keep the last `cut_min` minutes (15, 30, ...) after
                   discarding the trailing missing samples.
    max_gap_s    : threshold (s) between short (interpolated) and long gaps.
    detrend_deg  : degree of the detrending polynomial (SOTA only).
    spline_degree: degree of the B-splines.
    bandwidth_s  : bandwidth (s) of the local linear regression (LLR).
    """
    method = method.upper()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")

    x = clean_out_of_range(segment(fhr, cut_min, fs))

    if method == "SOTA":
        return sota(x, fs, max_gap_s, detrend_deg)  # already restores gaps > 15 s
    if method == "BASIC":
        return basic(x, fs, max_gap_s)  # gaps > 15 s remain NaN

    _, long_mask = split_gaps(x, fs, max_gap_s)  # positions of gaps > 15 s
    y = fill_gaps(x, fs, max_gap_s)
    w = np.ones(len(y), dtype=bool)
    if f_cut is not None and method in ("FOURIER", "SPLINES"):
        n_basis = n_basis_from_cutoff(method, len(y), fs, f_cut, spline_degree)
    if method == "FOURIER":
        out = fourier_smooth(y, w, n_basis)
    elif method == "SPLINES":
        out = bspline_smooth(y, w, n_basis, spline_degree)
    else:  # LLR
        out = llr_smooth(y, w, bandwidth_s, fs)

    out[long_mask] = np.nan  # restore gaps > 15 s
    return out


def plot_record(raw, results, path, fs=FS, title=None):
    """
    Saves a PNG with the raw signal and, below it, one panel per pipeline.

    raw     : raw signal (already cut if cut_min is used).
    results : dict {method: preprocessed_signal}
    path    : output PNG path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = np.asarray(raw, dtype=float)
    t = np.arange(len(raw)) / fs / 60  # minutes
    n = 1 + len(results)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)

    axes[0].plot(t, raw, lw=0.7, color="k")
    axes[0].set_title("Raw signal", loc="left", fontsize=10)
    for ax, (m, y) in zip(axes[1:], results.items()):
        ax.plot(t, y, lw=0.8)
        ax.set_ylim(MIN_BPM, MAX_BPM)
        ax.set_title(m, loc="left", fontsize=10)
    for ax in axes:
        ax.set_ylabel("FHR (bpm)")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (min)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def preprocess_database(signals, methods=METHODS, plot_dir=None, **kwargs):
    """
    signals  : dict {record_id: FHR array}
    methods  : pipelines to apply (default: all five)
    plot_dir : if given, saves one PNG per record ("<record_id>.png") in that folder.
    kwargs (n_basis, fs, cut_min, ...) are passed to `preprocess`.
    Returns {method: {record_id: preprocessed_signal}}.
    """
    out = {m: {} for m in methods}
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)

    for rid, sig in signals.items():
        for m in methods:
            out[m][rid] = preprocess(sig, m, **kwargs)

        if plot_dir is not None:
            fs = kwargs.get("fs", FS)
            raw = segment(sig, kwargs.get("cut_min"), fs)
            plot_record(
                raw,
                {m: out[m][rid] for m in methods},
                os.path.join(plot_dir, f"{rid}.png"),
                fs=fs,
                title=str(rid),
            )
    return out


# --------------------------------------------------------------------------
# Demo with a synthetic signal:  python preprocess_fhr.py --f_cut 0.15
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_basis", type=int, default=101, help="number of basis functions")
    ap.add_argument("--f_cut", type=float, default=None, help="cut-off frequency (Hz)")
    ap.add_argument("--cut_min", type=float, default=30, help="final minutes to keep")
    ap.add_argument("--plot_dir", default=None, help="folder for the plots")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    n = 4 * 60 * 40  # 40 min at 4 Hz
    t = np.arange(n) / 4
    demo = 140 + 10 * np.sin(2 * np.pi * t / 300) + 0.02 * t + rng.normal(0, 2, n)
    demo[1000:1020] = 0  # short gap (5 s)
    demo[3000:3400] = 0  # long gap (100 s)
    demo[5000:5003] = 230  # out-of-range values
    demo[-200:] = 0  # trailing missing samples (discarded)

    res = preprocess_database(
        {"demo": demo},
        plot_dir=args.plot_dir,
        n_basis=args.n_basis,
        f_cut=args.f_cut,
        cut_min=args.cut_min,
    )
    for m, d in res.items():
        s = d["demo"]
        print(f"{m:8s} n={len(s)}  NaN={int(np.isnan(s).sum()):4d}  mean={np.nanmean(s):.1f}")
