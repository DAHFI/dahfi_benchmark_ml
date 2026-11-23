"""
Morphological FHR features: baseline, accelerations and decelerations.

    baseline : mean of the baseline curve (bpm)
    DCC      : total deceleration area (bpm*s) per 10 min of valid signal
    ACC      : total acceleration area (bpm*s) per 10 min of valid signal

Both events are summarised in the same way (area per 10 min of valid signal), so that
records with different amounts of valid signal are comparable. Deceleration area is more
strongly associated with acidemia than the number of decelerations [6].

Definitions (FIGO 2015, term pregnancies) [1]
---------------------------------------------
- Baseline: mean FHR level in the most stable segments (without accelerations or
  decelerations), estimated over 10-min periods.
- Acceleration: abrupt increase of more than 15 bpm lasting more than 15 s and less
  than 10 min.
- Deceleration: decrease of more than 15 bpm lasting more than 10 s. (Prolonged: > 3 min.
  A decrease lasting > 10 min is a baseline change, not a deceleration.)
  Other guidelines (NICHD [2]) use 15 s also for decelerations (parameter dec_min_s).

Implementation
--------------
1. Jump cleaning: samples deviating more than 20 bpm from the 5-s running median are
   treated as invalid (signal loss and abrupt jumps, which would otherwise be mistaken for
   decelerations) [5]. This is only used internally; the signal passed to the other
   feature modules is not modified.
2. Baseline by iterative exclusion with decreasing bands, inspired by Mantel et al. [3]
   (one of the method families compared in [4]). The exclusion is ASYMMETRIC: below the
   current trend only points within HALF the band are kept (above it, the full band), so
   that frequent decelerations (common in the second stage of labour) do not drag the
   baseline down.
3. An event starts and ends when the FHR deviates more than 5 bpm from the baseline, so
   shallow tails are not included and consecutive decelerations that almost return to the
   baseline are not merged.

References
[1] Ayres-de-Campos D, Spong CY, Chandraharan E; FIGO Intrapartum Fetal Monitoring
    Expert Consensus Panel. FIGO consensus guidelines on intrapartum fetal monitoring:
    Cardiotocography. Int J Gynaecol Obstet. 2015;131(1):13-24.
[2] Macones GA, Hankins GDV, Spong CY, Hauth J, Moore T. The 2008 National Institute of
    Child Health and Human Development workshop report on electronic fetal monitoring.
    Obstet Gynecol. 2008;112(3):661-666.
[3] Mantel R, van Geijn HP, Caron FJ, Swartjes JM, van Woerden EE, Jongsma HW. Computer
    analysis of antepartum fetal heart rate: 1. Baseline determination.
    Int J Biomed Comput. 1990;25(4):261-272.
[4] Houze de l'Aulnoit A, Boudet S, Demailly R, et al. Automated fetal heart rate
    analysis for baseline determination and acceleration/deceleration detection:
    A comparison of 11 methods versus expert consensus.
    Biomed Signal Process Control. 2019;49:113-123.
[5] Spilka J, Chudacek V, Koucky M, et al. Using nonlinear features for fetal heart rate
    classification. Biomed Signal Process Control. 2012;7(4):350-357.
[6] Cahill AG, Tuuli MG, Stout MJ, Lopez JD, Macones GA. A prospective cohort study of
    fetal heart rate monitoring: deceleration area is predictive of fetal acidemia.
    Am J Obstet Gynecol. 2018;218(5):523.e1-523.e12.
"""

import numpy as np
import pandas as pd

FS = 4.0  # Hz (CTU-UHB)

CLEAN_JUMPS = True  # remove abrupt jumps before detecting events
ASYMMETRIC_BASELINE = True  # asymmetric exclusion in the baseline estimate
ONSET_BPM = 5.0  # an event starts/ends when |FHR - baseline| exceeds this value


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def _rolling(x, win, how):
    """Centred running mean or median that ignores NaN."""
    r = pd.Series(x).rolling(int(win), center=True, min_periods=1)
    return np.array(
        r.median() if how == "median" else r.mean(), dtype=float
    )  # writable copy


def _fill(x):
    """Linear interpolation of NaN (the ends take the nearest valid value)."""
    x = np.asarray(x, dtype=float).copy()
    bad = np.isnan(x)
    if bad.all() or not bad.any():
        return x
    x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
    return x


def _runs(mask):
    """(start, exclusive_end) of each run of True."""
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def clean_jumps(signal, fs=FS, threshold=20.0, window_s=5.0):
    """Sets to NaN the samples deviating more than `threshold` bpm from the running median
    of `window_s` s (abrupt jumps and signal loss) [5]. Returns a copy."""
    x = np.asarray(signal, dtype=float).copy()
    med = _rolling(x, max(1, int(round(window_s * fs))), "median")
    x[np.abs(x - med) > threshold] = np.nan
    return x


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------
def get_baseline_curve(
    signal,
    fs=FS,
    window_s=600.0,
    bands=(25.0, 20.0, 15.0, 10.0),
    asymmetric=ASYMMETRIC_BASELINE,
):
    """
    Baseline curve by iterative exclusion (inspired by Mantel et al. [3]).

    1. Initial trend: running median over `window_s` (10 min, as the FIGO period [1]).
    2. At each iteration, with band tau = 25, 20, 15 and 10 bpm, the points kept are
           symmetric : |x - B| <= tau
           asymmetric: -tau/2 <= x - B <= tau   (points below are excluded earlier, so that
                                                 decelerations do not drag the baseline down)
       and the trend is recomputed as the running mean of the kept points.

    Returns an array of the same length as the signal (NaN if there are no valid data).
    """
    x = np.asarray(signal, dtype=float)
    valid = ~np.isnan(x)
    if not valid.any():
        return np.full(x.shape, np.nan)

    win = max(1, int(round(window_s * fs)))
    b = _fill(_rolling(x, win, "median"))
    for thr in bands:
        d = x - b
        keep = valid & (
            ((d >= -thr / 2) & (d <= thr)) if asymmetric else (np.abs(d) <= thr)
        )
        if keep.sum() < 2:
            break
        b = _fill(_rolling(np.where(keep, x, np.nan), win, "mean"))
    return b


def get_baseline(signal, fs=FS, clean=CLEAN_JUMPS, **kwargs):
    """Baseline value (bpm): mean of the baseline curve over the valid samples."""
    x = np.asarray(signal, dtype=float)
    if clean:
        x = clean_jumps(x, fs)
    valid = ~np.isnan(x)
    if not valid.any():
        return np.nan
    return float(np.mean(get_baseline_curve(x, fs, **kwargs)[valid]))


# --------------------------------------------------------------------------
# Accelerations and decelerations (FIGO 2015 [1])
# --------------------------------------------------------------------------
def get_events(
    signal,
    fs=FS,
    baseline_curve=None,
    smooth_s=2.5,
    onset=ONSET_BPM,
    clean=CLEAN_JUMPS,
    acc_amp=15.0,
    acc_min_s=15.0,
    acc_max_s=600.0,
    dec_amp=15.0,
    dec_min_s=10.0,
    dec_max_s=600.0,
):
    """
    Detects accelerations and decelerations following FIGO 2015 [1].

    r[n] = xs[n] - B[n], with xs the smoothed FHR (running mean of `smooth_s` s, only used to
    find the crossings) and B the baseline. An event is a continuous segment with
        r > onset   (acceleration)      or      r < -onset   (deceleration).
    Its amplitude is the maximum distance to the baseline and its area sum(|r|) / fs.
    It is accepted if:
        acceleration : amplitude > acc_amp  and  acc_min_s < duration < acc_max_s
        deceleration : amplitude > dec_amp  and  dec_min_s < duration <= dec_max_s
    Segments are cut at gaps (NaN). With clean=True, jumps are removed first.
    If baseline_curve is given, it must be computed on the same (cleaned) signal.

    Returns a list of dicts with: type ('acc'/'dec'), start, end (indices, exclusive end),
    duration_s, amplitude (bpm, positive), area (bpm*s).
    """
    x = np.asarray(signal, dtype=float)
    if clean:
        x = clean_jumps(x, fs)
    if not (~np.isnan(x)).any():
        return []
    b = (
        get_baseline_curve(x, fs)
        if baseline_curve is None
        else np.asarray(baseline_curve, float)
    )

    xs = _rolling(x, max(1, int(round(smooth_s * fs))), "mean")
    xs[np.isnan(x)] = np.nan  # smoothing does not fill gaps
    r = xs - b

    events = []
    for kind, mask in (
        ("acc", r > onset),
        ("dec", r < -onset),
    ):  # NaN -> False: cut at gaps
        for s, e in zip(*_runs(mask)):
            dur = (e - s) / fs
            seg = r[s:e]
            amp = float(np.max(seg)) if kind == "acc" else float(-np.min(seg))
            if kind == "acc":
                ok = amp > acc_amp and acc_min_s < dur < acc_max_s
            else:
                ok = amp > dec_amp and dec_min_s < dur <= dec_max_s
            if ok:
                events.append(
                    {
                        "type": kind,
                        "start": int(s),
                        "end": int(e),
                        "duration_s": dur,
                        "amplitude": amp,
                        "area": float(np.abs(seg).sum() / fs),
                    }
                )
    events.sort(key=lambda ev: ev["start"])
    return events


def get_event_mask(signal, fs=FS, **kwargs):
    """Boolean mask (True at samples belonging to an acceleration or deceleration).
    Used by the linear features to exclude events, as indicated by FIGO [1] and Dawes-Redman.
    """
    x = np.asarray(signal, dtype=float)
    mask = np.zeros(x.size, dtype=bool)
    for ev in get_events(x, fs, **kwargs):
        mask[ev["start"] : ev["end"]] = True
    return mask


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def analyze_single_ctg(signal, fhr=True, fs=FS):
    """
    fhr=True : (baseline, DCC, ACC), with DCC and ACC the deceleration and acceleration
               area (bpm*s) per 10 min of valid signal
    fhr=False: baseline
    """
    x = np.asarray(signal, dtype=float)
    if CLEAN_JUMPS:
        x = clean_jumps(x, fs)
    valid = ~np.isnan(x)
    if x.size == 0 or not valid.any():
        return (np.nan, np.nan, np.nan) if fhr else np.nan

    b = get_baseline_curve(x, fs)
    baseline = float(np.mean(b[valid]))
    if not fhr:
        return baseline

    events = get_events(x, fs, baseline_curve=b, clean=False)  # x is already clean
    minutes = valid.sum() / fs / 60
    dcc = 10 * sum(ev["area"] for ev in events if ev["type"] == "dec") / minutes
    acc = 10 * sum(ev["area"] for ev in events if ev["type"] == "acc") / minutes
    return baseline, dcc, acc


def get_morphological_features(signals, fhr=True, fs=FS):
    """
    fhr=True : array (3, n_signals) -> baseline, DCC, ACC
    fhr=False: array (1, n_signals) -> baseline
    """
    rows = [analyze_single_ctg(s, fhr=fhr, fs=fs) for s in signals]
    morphological = (
        np.array(rows, dtype=float).T if fhr else np.array([rows], dtype=float)
    )
    if np.isnan(morphological).any():
        print("WARNING morphological features: NaN values")
    return morphological
