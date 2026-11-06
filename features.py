"""
Windowing + the five hand-crafted features of the report (Section 5.1).

Window: 50 ms (500 samples @ 10 kHz), stride 10 ms (100 samples) -> new decision every 10 ms.
Features per window:
  1. p_rms        RMS of p' (kPa)                         -> loudness
  2. kurtosis     4th standardised moment of p'           -> spikiness (noise ~3, sine ~1.5, bursts > 4)
  3. peak_ratio   share of spectral energy within +/-30 Hz of the dominant acoustic peak
  4. zcr_reg      zero-crossing rate / (2 * f_peak)        -> frequency regularity (1 = clean tone)
  5. phase_lag    |angle| between p' and q' at the peak (deg), waveguide delay calibrated out
"""
from __future__ import annotations
import numpy as np

FS = 10_000
WIN = 500
STRIDE = 100
NFFT = 2048
WAVEGUIDE_DELAY_S = 0.5e-3

FEATURE_NAMES = ["p_rms", "kurtosis", "peak_ratio", "zcr_reg", "phase_lag"]
FEATURE_LABELS = ["RMS p′ (kPa)", "Kurtosis", "FFT peak ratio", "Zero-crossing regularity", "Phase lag Δθ (deg)"]


def make_windows(x, win=WIN, stride=STRIDE):
    n = (len(x) - win) // stride + 1
    idx = np.arange(win)[None, :] + stride * np.arange(n)[:, None]
    return x[idx], idx[:, -1]  # windows, index of last sample (decision time)


_hann_cache = {}


def _hann(n):
    if n not in _hann_cache:
        _hann_cache[n] = np.hanning(n).astype(np.float64)
    return _hann_cache[n]


def window_features(P, Q=None, fs=FS, fmin=80.0, fmax=2000.0, band=30.0, delay_s=WAVEGUIDE_DELAY_S):
    """P, Q: (n_windows, win) arrays. Returns (n_windows, 5) (phase_lag = nan if Q is None)."""
    P = np.asarray(P, dtype=np.float64)
    n, w = P.shape
    Pc = P - P.mean(1, keepdims=True)
    rms = np.sqrt((Pc ** 2).mean(1))
    var = (Pc ** 2).mean(1)
    kurt = (Pc ** 4).mean(1) / (var ** 2 + 1e-20)

    h = _hann(w)
    FP = np.fft.rfft(Pc * h, n=NFFT, axis=1)
    freqs = np.fft.rfftfreq(NFFT, 1.0 / fs)
    psd = np.abs(FP) ** 2
    sel = (freqs >= fmin) & (freqs <= fmax)
    tot_sel = (freqs >= 20) & (freqs <= fmax)
    kpk = np.argmax(np.where(sel[None, :], psd, -1), axis=1)
    fpk = freqs[kpk]
    inband = np.abs(freqs[None, :] - fpk[:, None]) <= band
    peak_ratio = (psd * inband).sum(1) / ((psd * tot_sel[None, :]).sum(1) + 1e-20)

    s = np.signbit(Pc)
    zc = (s[:, 1:] != s[:, :-1]).sum(1) * fs / (w - 1)
    zcr_reg = zc / (2.0 * fpk)

    if Q is not None:
        Q = np.asarray(Q, dtype=np.float64)
        Qc = Q - Q.mean(1, keepdims=True)
        FQ = np.fft.rfft(Qc * h, n=NFFT, axis=1)
        # smooth cross-spectrum over the peak band for a robust phase estimate
        cross = (FP * np.conj(FQ) * inband).sum(1)
        ang = np.angle(cross)                                  # phase of p relative to q
        ang = ang - 2 * np.pi * fpk * delay_s                  # remove waveguide lag of p
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        phase = np.degrees(np.abs(ang))
    else:
        phase = np.full(n, np.nan)
    return np.stack([rms, kurt, peak_ratio, zcr_reg, phase], axis=1).astype(np.float32), fpk
