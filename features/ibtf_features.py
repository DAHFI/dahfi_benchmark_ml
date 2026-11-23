"""
Image-based time-frequency (IBTF) FHR features.

Following Comert et al. [1]: the spectrogram (STFT) of the FHR is computed, the image of
each frequency band is taken, and its texture is described with the grey-level
co-occurrence matrix (GLCM) of Haralick [2]: contrast, correlation, energy and
homogeneity, in 4 bands -> 16 features.

FHR frequency bands (Signorini et al. [3]):
    VLF 0-0.03 Hz | LF 0.03-0.15 Hz | MF 0.15-0.5 Hz | HF 0.5-1 Hz

Implementation details:
    - The real sampling frequency is used (4 Hz in CTU-UHB).
    - The GLCM is computed directly on the spectrogram matrix (in dB) restricted to each
      band, not on a rendered figure, so figure margins, colour maps and interpolation do
      not enter the GLCM and the result does not depend on the plotting library.
    - Each band image is quantised globally between its minimum and maximum (as MATLAB's
      graycomatrix does by default).
    - STFT window of 1024 samples (256 s), so that the VLF band has enough frequency rows.
      The mean of each window is removed (detrend='constant') and the 0-Hz component is
      excluded.
    - Gaps are interpolated only for the spectrogram, which does not accept NaN.
    - GLCM: 12 grey levels, distance 1, angle pi/2.

No data leakage: everything is computed from a single record.

References
[1] Comert Z, Kocamaz AF, Subha V. Prognostic model based on image-based time-frequency
    features and genetic algorithm for fetal hypoxia assessment.
    Comput Biol Med. 2018;99:85-97.
[2] Haralick RM, Shanmugam K, Dinstein I. Textural features for image classification.
    IEEE Trans Syst Man Cybern. 1973;SMC-3(6):610-621.
[3] Signorini MG, Magenes G, Cerutti S, Arduini D. Linear and nonlinear parameters for
    the analysis of fetal heart rate signal from cardiotocographic recordings.
    IEEE Trans Biomed Eng. 2003;50(3):365-374.
"""

import numpy as np
from scipy.signal import spectrogram
from skimage.feature import graycomatrix, graycoprops

FS = 4.0  # Hz (CTU-UHB)

BANDS = {  # Signorini et al. [3]
    "VLF": (0.0, 0.03),
    "LF": (0.03, 0.15),
    "MF": (0.15, 0.50),
    "HF": (0.50, 1.00),
}
PROPS = ("contrast", "correlation", "energy", "homogeneity")


def _fill(x):
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if bad.all() or not bad.any():
        return x
    x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
    return x


def get_spectrogram_db(signal, fs=FS, nperseg=1024, overlap=0.75):
    """Spectrogram (STFT, Hann window) in dB: returns (frequencies, times, S_dB)."""
    x = _fill(signal)
    nperseg = min(nperseg, x.size)
    f, t, S = spectrogram(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=int(nperseg * overlap),
        detrend="constant",
        scaling="density",
        mode="psd",
    )
    floor = np.max(S) * 1e-12 if np.max(S) > 0 else 1e-12  # avoids log10(0)
    return f, t, 10 * np.log10(np.maximum(S, floor))


def band_glcm(f, S_db, band, levels=12, distance=1, angle=np.pi / 2):
    """Contrast, correlation, energy and homogeneity of the GLCM of one band."""
    lo, hi = band
    rows = (f > lo) & (f <= hi)  # non-overlapping bands; VLF excludes the 0-Hz component
    img = S_db[rows][::-1]  # rows = frequency (high at the top, as in the image), columns = time
    if img.shape[0] < 2 or img.shape[1] < 2:
        return np.full(len(PROPS), np.nan)

    lo_v, hi_v = img.min(), img.max()
    if hi_v > lo_v:
        q = np.round((img - lo_v) / (hi_v - lo_v) * (levels - 1)).astype(np.uint8)
    else:
        q = np.zeros(img.shape, dtype=np.uint8)

    P = graycomatrix(q, [distance], [angle], levels=levels, symmetric=False, normed=True)
    return np.array([float(graycoprops(P, p).mean()) for p in PROPS])


def get_IBTF_features(signals, fs=FS, nperseg=1024, overlap=0.75, levels=12, distance=1, angle=np.pi / 2):
    """
    Output: array (16, n_signals), in the order
        VLF: contrast, correlation, energy, homogeneity
        LF : ...   MF : ...   HF : ...
    """
    out = []
    for signal in signals:
        x = np.asarray(signal, dtype=float)
        if x.size == 0 or not np.isfinite(x).any():
            out.append(np.full(4 * len(PROPS), np.nan))
            continue
        f, _, S_db = get_spectrogram_db(x, fs, nperseg, overlap)
        out.append(np.concatenate([band_glcm(f, S_db, BANDS[b], levels, distance, angle)
                                   for b in ("VLF", "LF", "MF", "HF")]))
    ibtf = np.array(out, dtype=float).T

    if not np.isfinite(ibtf).all():
        print("WARNING IBTF features: NaN or Inf values")
    return ibtf
