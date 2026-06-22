import numpy as np
from scipy.signal import find_peaks, correlate
from scipy.fft import fft, fftfreq


def extract_period_peaks(x_coords, fps, prominence=15.0, min_dist_sec=0.3):
    """Peak-based period extraction (original method, improved)."""
    x = np.array(x_coords)
    x_c = x - x.mean()
    min_dist = int(fps * min_dist_sec)

    peaks_max, props_max = find_peaks(x_c, distance=min_dist, prominence=prominence)
    peaks_min, props_min = find_peaks(-x_c, distance=min_dist, prominence=prominence)

    periods = []
    if len(peaks_max) >= 2:
        periods.extend((np.diff(peaks_max) / fps).tolist())
    if len(peaks_min) >= 2:
        periods.extend((np.diff(peaks_min) / fps).tolist())

    if not periods:
        return None, None, peaks_max, peaks_min

    T = float(np.mean(periods))
    T_std = float(np.std(periods, ddof=1)) if len(periods) > 1 else 0.0
    return T, T_std, peaks_max, peaks_min


def extract_period_fft(x_coords, fps):
    """FFT-based period extraction. More robust to noise."""
    x = np.array(x_coords)
    x_c = x - x.mean()
    n = len(x_c)
    window = np.hanning(n)
    x_windowed = x_c * window

    yf = fft(x_windowed)
    xf = fftfreq(n, 1.0 / fps)

    pos_mask = xf > 0
    xf_pos = xf[pos_mask]
    magnitudes = np.abs(yf[pos_mask])

    dc_range = xf_pos < 0.1
    if np.any(dc_range):
        magnitudes[dc_range] = 0

    if len(magnitudes) == 0 or np.max(magnitudes) < 1e-10:
        return None, xf_pos, magnitudes

    peak_idx = np.argmax(magnitudes)
    f_peak = xf_pos[peak_idx]
    T = 1.0 / f_peak if f_peak > 0 else None

    if T is None or T > n / fps:
        return None, xf_pos, magnitudes

    return T, xf_pos, magnitudes


def extract_period_autocorrelation(x_coords, fps):
    """Autocorrelation-based period extraction."""
    x = np.array(x_coords)
    x_c = x - x.mean()

    acf = correlate(x_c, x_c, mode="full")
    acf = acf[len(acf) // 2:]
    acf = acf / acf[0]

    min_lag = max(1, int(fps * 0.3))
    if min_lag >= len(acf):
        return None, acf

    peaks, _ = find_peaks(acf[min_lag:], distance=int(fps * 0.3), prominence=0.05)
    if len(peaks) == 0:
        return None, acf

    peaks_actual = peaks + min_lag
    best_idx = peaks_actual[np.argmax(acf[peaks_actual])]
    T = best_idx / fps

    return T, acf


def extract_period(x_coords, fps, method="fft", prominence=15.0, min_dist_sec=0.3):
    """
    Extract oscillation period using specified method.
    Returns dict with T, method name, and debug info.
    """
    result = {"method": method, "fps": fps}

    if method == "fft" or method == "all":
        T_fft, freqs, mags = extract_period_fft(x_coords, fps)
        result["T_fft"] = T_fft
        result["fft_freqs"] = freqs
        result["fft_magnitudes"] = mags

    if method == "autocorrelation" or method == "all":
        T_acf, acf_data = extract_period_autocorrelation(x_coords, fps)
        result["T_acf"] = T_acf
        result["acf_data"] = acf_data

    if method == "peaks" or method == "all":
        T_pk, T_std, pk_max, pk_min = extract_period_peaks(x_coords, fps, prominence, min_dist_sec)
        result["T_peaks"] = T_pk
        result["T_std"] = T_std
        result["peaks_max"] = pk_max
        result["peaks_min"] = pk_min

    if method == "all":
        values = [v for v in [result.get("T_fft"), result.get("T_acf"), result.get("T_peaks")] if v is not None]
        result["T"] = float(np.mean(values)) if values else None
        result["all_T_values"] = values
    elif method == "fft":
        result["T"] = result.get("T_fft")
    elif method == "autocorrelation":
        result["T"] = result.get("T_acf")
    elif method == "peaks":
        result["T"] = result.get("T_peaks")

    return result
