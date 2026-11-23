"""
Linear (time-domain) FHR features.

    mean  : mean FHR (bpm).
    std   : standard deviation of the FHR (bpm).
    LTV   : long-term variation, Dawes-Redman [1, 2] (ms).
    delta : baseline variability, FIGO [3] (bpm).
    STV   : short-term variation, Dawes-Redman [1, 2] (ms).
    II    : interval index of de Haan [4], in the form used for FHR analysis by
            Signorini et al. [5] (dimensionless).

Units: Dawes-Redman and de Haan work with the beat-to-beat interval in ms
(T = 60000 / FHR). Therefore STV, LTV and II use ms by default (units="ms");
with units="bpm" they are computed on the FHR in bpm.

Event exclusion: FIGO [3] measures variability excluding accelerations and decelerations,
and Dawes-Redman [1, 2] excludes minutes with decelerations from the LTV. With
exclude_events=True (default), minutes containing an acceleration or deceleration
(detected in morphological_features) do not count for LTV or delta.

References
[1] Dawes GS, Moulden M, Redman CWG. System 8000: computerized antenatal FHR analysis.
    J Perinat Med. 1991;19(1-2):47-51.
[2] Pardey J, Moulden M, Redman CWG. A computer system for the numerical analysis of
    nonstress tests. Am J Obstet Gynecol. 2002;186(5):1095-1103.
[3] Ayres-de-Campos D, Spong CY, Chandraharan E; FIGO Intrapartum Fetal Monitoring
    Expert Consensus Panel. FIGO consensus guidelines on intrapartum fetal monitoring:
    Cardiotocography. Int J Gynaecol Obstet. 2015;131(1):13-24.
[4] de Haan J, van Bemmel JH, Versteeg B, Veth AF, Stolte LA, Janssens J, Eskes TK.
    Quantitative evaluation of fetal heart rate patterns. I. Processing methods.
    Eur J Obstet Gynecol. 1971;1(3):95-102.
[5] Signorini MG, Magenes G, Cerutti S, Arduini D. Linear and nonlinear parameters for
    the analysis of fetal heart rate signal from cardiotocographic recordings.
    IEEE Trans Biomed Eng. 2003;50(3):365-374.
"""

import numpy as np

from morphological_features import get_event_mask

FS = 4.0  # Hz (CTU-UHB)


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def _convert(signal, units):
    x = np.asarray(signal, dtype=float)
    if units == "ms":
        with np.errstate(divide="ignore", invalid="ignore"):
            x = 60000.0 / x  # beat-to-beat interval (ms)
    return x


def _epoch_means(x, ep, min_valid):
    """Mean of each epoch of `ep` samples; NaN if it has fewer than min_valid * ep valid samples."""
    n_ep = x.size // ep
    blocks = x[: n_ep * ep].reshape(n_ep, ep)
    n_ok = np.sum(~np.isnan(blocks), axis=1)
    return np.where(
        n_ok >= min_valid * ep, np.nansum(blocks, axis=1) / np.maximum(n_ok, 1), np.nan
    )


def _minutes_with_events(event_mask, n_win, block):
    """True for each minute that contains any acceleration or deceleration sample."""
    if event_mask is None:
        return np.zeros(n_win, dtype=bool)
    m = np.asarray(event_mask, dtype=bool)[: n_win * block]
    return m.reshape(n_win, block).any(axis=1)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def get_STV(signal, fs=FS, epoch_s=3.75, window_s=60.0, min_valid=0.5, units="ms"):
    """
    Short-Term Variation, Dawes-Redman [1, 2].

    The signal is divided into epochs of 1/16 min (3.75 s) and the mean of each one is
    computed. In each minute the absolute differences between consecutive epochs are
    averaged, and the STV is the mean of those values over the valid minutes.
    """
    x = _convert(signal, units)
    ep = int(round(epoch_s * fs))
    per_win = int(round(window_s / epoch_s))
    e = _epoch_means(x, ep, min_valid)
    n_win = e.size // per_win
    if n_win == 0:
        return np.nan

    w = e[: n_win * per_win].reshape(n_win, per_win)
    d = np.abs(np.diff(w, axis=1))  # NaN if any epoch is invalid
    n_d = np.sum(~np.isnan(d), axis=1)
    keep = n_d >= min_valid * (per_win - 1)
    if not keep.any():
        return np.nan
    return float(np.mean(np.nansum(d[keep], axis=1) / n_d[keep]))


def get_LTV(
    signal,
    fs=FS,
    epoch_s=3.75,
    window_s=60.0,
    min_valid=0.5,
    units="ms",
    event_mask=None,
):
    """
    Long-Term Variation, Dawes-Redman [1, 2].

    Range (maximum - minimum) of the epoch means (1/16 min) within each minute, averaged over
    the valid minutes. Minutes with accelerations or decelerations (event_mask) are excluded.
    """
    x = _convert(signal, units)
    ep = int(round(epoch_s * fs))
    per_win = int(round(window_s / epoch_s))
    e = _epoch_means(x, ep, min_valid)
    n_win = e.size // per_win
    if n_win == 0:
        return np.nan

    w = e[: n_win * per_win].reshape(n_win, per_win)
    keep = np.sum(~np.isnan(w), axis=1) >= min_valid * per_win
    keep &= ~_minutes_with_events(event_mask, n_win, per_win * ep)
    if not keep.any():
        return np.nan
    return float(np.mean(np.nanmax(w[keep], axis=1) - np.nanmin(w[keep], axis=1)))


def get_delta(
    signal,
    fs=FS,
    window_s=60.0,
    q=(2.5, 97.5),
    min_valid=0.5,
    agg="mean",
    event_mask=None,
):
    """
    Baseline variability, FIGO [3]: amplitude of the FHR oscillations in 1-min segments,
    excluding accelerations and decelerations (event_mask).

    q: percentiles that define the amplitude. (0, 100) is maximum - minimum; (2.5, 97.5)
       ignores isolated samples (artifacts), like the visual reading of the band.

    Returns (delta, delta_total); delta_total is the amplitude over the whole signal.
    """
    x = np.asarray(signal, dtype=float)
    v_all = x[~np.isnan(x)]
    if v_all.size == 0:
        return np.nan, np.nan
    delta_total = float(np.percentile(v_all, q[1]) - np.percentile(v_all, q[0]))

    block = int(round(window_s * fs))
    n_win = x.size // block
    if n_win == 0:
        return np.nan, delta_total

    with_events = _minutes_with_events(event_mask, n_win, block)
    amps = []
    for k, part in enumerate(x[: n_win * block].reshape(n_win, block)):
        v = part[~np.isnan(part)]
        if not with_events[k] and v.size >= min_valid * block:
            amps.append(np.percentile(v, q[1]) - np.percentile(v, q[0]))
    if not amps:
        return np.nan, delta_total
    return float(np.median(amps) if agg == "median" else np.mean(amps)), delta_total


def get_II(signal, fs=FS, epoch_s=2.5, min_valid=0.5, units="ms"):
    """
    Interval Index of de Haan [4], as used by Signorini et al. [5]:

        T(i) : mean beat-to-beat interval in 2.5-s epochs (24 per minute)
        dT   : differences between consecutive valid epochs
        STV* : mean of |dT|
        II   = standard deviation of dT / STV*
    """
    x = _convert(signal, units)
    e = _epoch_means(x, int(round(epoch_s * fs)), min_valid)
    d = np.diff(e)
    d = d[~np.isnan(d)]
    if d.size < 2:
        return np.nan
    stv = np.mean(np.abs(d))
    return float(np.std(d) / stv) if stv > 0 else np.nan


def get_linear_features(signals, fs=FS, units="ms", exclude_events=True):
    """
    Output: array (6, n_signals) in the order mean, std, LTV, delta, STV, II.
    """
    rows = []
    for i, signal in enumerate(signals):
        x = np.asarray(signal, dtype=float)
        if x.size == 0 or np.isnan(x).all():
            print(f"Signal {i} is all NaN")
            rows.append([np.nan] * 6)
            continue

        mask = get_event_mask(x, fs) if exclude_events else None
        rows.append(
            [
                float(np.nanmean(x)),
                float(np.nanstd(x)),
                get_LTV(x, fs=fs, units=units, event_mask=mask),
                get_delta(x, fs=fs, event_mask=mask)[0],
                get_STV(x, fs=fs, units=units),
                get_II(x, fs=fs, units=units),
            ]
        )

    linear = np.array(rows, dtype=float)
    if np.isnan(linear).any():
        print("WARNING linear features: NaN values")
    return linear.T
