"""
Nonlinear FHR features: ApEn, SampEn and Lempel-Ziv complexity.

    ApEn015, ApEn020     : Approximate Entropy, m = 2, r = 0.15 and 0.20 [1]
    SampEn015, SampEn020 : Sample Entropy, m = 2, r = 0.15 and 0.20 [2]
    LZC                  : normalised Lempel-Ziv complexity, binary coding [3, 4]

Analysed series: as in Signorini et al. [5] and Ferrario et al. [6], the indices are
computed on the series of 2.5-s epoch means (24 values per minute) of the beat-to-beat
interval in ms (units="ms"). Averaging over epochs avoids the aliasing produced by
simple decimation.

Tolerance: r * standard deviation of the analysed series, as in [1, 2].

Gaps are not interpolated. Templates (ApEn, SampEn) containing an epoch without data are
discarded, and in LZC the sequence is split at the gaps. Interpolating long gaps with
straight lines would create artificial regularity and lower all three measures.

SampEn is undefined when there are no matches (A = 0 or B = 0) [2]: NaN is returned.
These NaN must be imputed inside the model Pipeline, not here.

No data leakage: everything is computed from a single record.

References
[1] Pincus SM. Approximate entropy as a measure of system complexity.
    Proc Natl Acad Sci USA. 1991;88(6):2297-2301.
[2] Richman JS, Moorman JR. Physiological time-series analysis using approximate entropy
    and sample entropy. Am J Physiol Heart Circ Physiol. 2000;278(6):H2039-H2049.
[3] Lempel A, Ziv J. On the complexity of finite sequences.
    IEEE Trans Inf Theory. 1976;22(1):75-81.
[4] Kaspar F, Schuster HG. Easily calculable measure for the complexity of
    spatiotemporal patterns. Phys Rev A. 1987;36(2):842-848.
[5] Signorini MG, Magenes G, Cerutti S, Arduini D. Linear and nonlinear parameters for
    the analysis of fetal heart rate signal from cardiotocographic recordings.
    IEEE Trans Biomed Eng. 2003;50(3):365-374.
[6] Ferrario M, Signorini MG, Magenes G, Cerutti S. Comparison of entropy-based
    regularity estimators: application to the fetal heart rate signal for the
    identification of fetal distress. IEEE Trans Biomed Eng. 2006;53(1):119-125.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

FS = 4.0  # Hz (CTU-UHB)


# --------------------------------------------------------------------------
# Epoch series
# --------------------------------------------------------------------------
def epoch_series(signal, fs=FS, epoch_s=2.5, min_valid=0.5, units="ms"):
    """Means over epochs of `epoch_s` s (NaN if the epoch has less than min_valid of data)."""
    x = np.asarray(signal, dtype=float)
    if units == "ms":
        with np.errstate(divide="ignore", invalid="ignore"):
            x = 60000.0 / x
    ep = int(round(epoch_s * fs))
    n_ep = x.size // ep
    blocks = x[: n_ep * ep].reshape(n_ep, ep)
    n_ok = np.sum(~np.isnan(blocks), axis=1)
    return np.where(n_ok >= min_valid * ep, np.nansum(blocks, axis=1) / np.maximum(n_ok, 1), np.nan)


def _templates(y, m):
    """Templates of length m without any NaN."""
    if y.size < m:
        return np.empty((0, m))
    X = sliding_window_view(y, m)
    return X[~np.isnan(X).any(axis=1)]


def _close(X, tol):
    """Boolean matrix: Chebyshev distance between templates <= tol."""
    close = np.ones((X.shape[0], X.shape[0]), dtype=bool)
    for k in range(X.shape[1]):
        close &= np.abs(X[:, k][:, None] - X[:, k][None, :]) <= tol
    return close


# --------------------------------------------------------------------------
# ApEn (Pincus [1])
# --------------------------------------------------------------------------
def approximate_entropy(y, m=2, r=0.2):
    """
    ApEn(m, r) = Phi^m(r) - Phi^(m+1)(r),
    Phi^m(r) = mean of log C_i^m(r),  C_i^m(r) = fraction of templates j with d(i, j) <= r
    (self-matches included, as in the original definition).
    """
    y = np.asarray(y, dtype=float)
    tol = r * np.nanstd(y)
    phi = []
    for mm in (m, m + 1):
        X = _templates(y, mm)
        if X.shape[0] < 2:
            return np.nan
        phi.append(np.mean(np.log(_close(X, tol).mean(axis=1))))
    return float(phi[0] - phi[1])


# --------------------------------------------------------------------------
# SampEn (Richman and Moorman [2])
# --------------------------------------------------------------------------
def sample_entropy(y, m=2, r=0.2):
    """
    SampEn(m, r) = -ln(A / B)
        B: pairs of templates (i < j) of length m with d <= r
        A: pairs of templates (i < j) of length m+1 with d <= r
    Without self-matches and with the same set of templates for A and B (the first N - m),
    as in [2]. NaN if A = 0 or B = 0.
    """
    y = np.asarray(y, dtype=float)
    tol = r * np.nanstd(y)
    X1 = _templates(y, m + 1)
    if X1.shape[0] < 2:
        return np.nan
    close_m = _close(X1[:, :m], tol)
    close_m1 = close_m & (np.abs(X1[:, m][:, None] - X1[:, m][None, :]) <= tol)
    iu = np.triu_indices(X1.shape[0], k=1)
    B = int(close_m[iu].sum())
    A = int(close_m1[iu].sum())
    if A == 0 or B == 0:
        return np.nan
    return float(-np.log(A / B))


# --------------------------------------------------------------------------
# LZC (Lempel-Ziv [3], Kaspar and Schuster algorithm [4])
# --------------------------------------------------------------------------
def _lz76(seq):
    """Number of words (c) of the Lempel-Ziv 1976 decomposition."""
    n = len(seq)
    if n == 0:
        return 0
    if n == 1:
        return 1
    i, k, l, c, k_max = 0, 1, 1, 1, 1
    while True:
        if seq[i + k - 1] == seq[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            k_max = max(k_max, k)
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c


def lempel_ziv_complexity(y):
    """
    Binary coding of the increments: 1 if y(i+1) > y(i), 0 otherwise.
    The sequence is split at gaps; c is the sum of the words of each segment.
    Normalisation of [3]:  C = c * log2(n) / n,  with n = total number of symbols.
    """
    y = np.asarray(y, dtype=float)
    d = np.diff(y)
    ok = ~np.isnan(d)
    bits = (d > 0).astype(int)

    c, n = 0, 0
    edges = np.diff(np.concatenate(([0], ok.astype(int), [0])))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        c += _lz76(bits[s:e].tolist())
        n += e - s
    if n < 2:
        return np.nan
    return float(c * np.log2(n) / n)


# --------------------------------------------------------------------------
# Convenience functions (4-Hz FHR signal as input)
# --------------------------------------------------------------------------
def get_ApEn(signal, dim=2, r=0.2, fs=FS, epoch_s=2.5, units="ms"):
    return approximate_entropy(epoch_series(signal, fs, epoch_s, units=units), dim, r)


def get_SampEn(signal, dim=2, r=0.2, fs=FS, epoch_s=2.5, units="ms"):
    return sample_entropy(epoch_series(signal, fs, epoch_s, units=units), dim, r)


def get_LZC(signal, fs=FS, epoch_s=2.5, units="ms"):
    return lempel_ziv_complexity(epoch_series(signal, fs, epoch_s, units=units))


def get_nonlinear_features(signals, fs=FS, epoch_s=2.5, units="ms"):
    """
    Output: array (5, n_signals) in the order ApEn015, ApEn020, SampEn015, SampEn020, LZC.
    """
    rows = []
    for signal in signals:
        y = epoch_series(signal, fs, epoch_s, units=units)
        rows.append([
            approximate_entropy(y, 2, 0.15),
            approximate_entropy(y, 2, 0.20),
            sample_entropy(y, 2, 0.15),
            sample_entropy(y, 2, 0.20),
            lempel_ziv_complexity(y),
        ])
    nonlinear = np.array(rows, dtype=float).T

    names = ["ApEn015", "ApEn020", "SampEn015", "SampEn020", "LZC"]
    for name, values in zip(names, nonlinear):
        bad = ~np.isfinite(values)
        if bad.any():
            print(f"WARNING: {bad.sum()} non-finite values in {name} (signals {np.flatnonzero(bad)})")
    return nonlinear
