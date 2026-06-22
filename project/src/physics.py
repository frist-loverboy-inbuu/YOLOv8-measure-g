import numpy as np
from scipy.optimize import curve_fit


def calc_g_simple(L_m, T):
    """Simple pendulum: g = 4pi²L/T²"""
    return 4 * np.pi ** 2 * L_m / (T ** 2)


def damped_oscillation(t, A, gamma, f, phi, C):
    """Damped oscillation model: x(t) = A * exp(-gamma*t) * cos(2*pi*f*t + phi) + C"""
    return A * np.exp(-gamma * t) * np.cos(2 * np.pi * f * t + phi) + C


def correct_for_damping(T_measured, gamma):
    """
    Damped pendulum period correction:
    T0 = T_measured / sqrt(1 + (gamma*T_measured/(2*pi))^2)
    where gamma is the damping coefficient.
    """
    factor = 1 + (gamma * T_measured / (2 * np.pi)) ** 2
    return T_measured / np.sqrt(factor)


def fit_damped_model(t, x, fps, p0=None):
    """
    Fit damped oscillation model to trajectory data.
    Returns fitted parameters: (A, gamma, f, phi, C) and covariance.
    """
    x_clean = np.array(x[~np.isnan(x)])
    t_clean = np.array(t[~np.isnan(x)]) if hasattr(t, '__len__') else np.arange(len(x_clean)) / fps

    if len(x_clean) < 20:
        return None, None

    x_centered = x_clean - x_clean.mean()
    T_guess = _rough_period_estimate(x_centered, fps)
    if T_guess is None:
        T_guess = 1.0
    f_guess = 1.0 / T_guess

    if p0 is None:
        A_guess = (x_clean.max() - x_clean.min()) / 2
        gamma_guess = 0.02
        phi_guess = 0.0
        C_guess = x_clean.mean()
        p0 = [A_guess, gamma_guess, f_guess, phi_guess, C_guess]

    bounds_lower = [0, 0, f_guess * 0.5, -2 * np.pi, -np.inf]
    bounds_upper = [np.inf, 0.5, f_guess * 2.0, 2 * np.pi, np.inf]

    try:
        popt, pcov = curve_fit(damped_oscillation, t_clean, x_clean,
                               p0=p0, bounds=(bounds_lower, bounds_upper),
                               maxfev=10000)
        return popt, pcov
    except (RuntimeError, ValueError):
        try:
            popt, pcov = curve_fit(damped_oscillation, t_clean, x_clean, p0=p0, maxfev=20000)
            return popt, pcov
        except (RuntimeError, ValueError):
            return None, None


def _rough_period_estimate(x, fps):
    """Quick period estimate via zero crossings and peak detection for initial guess."""
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(x, distance=int(fps * 0.3))
    if len(peaks) >= 2:
        return float(np.mean(np.diff(peaks))) / fps
    zero_crossings = np.where(np.diff(np.signbit(x)))[0]
    if len(zero_crossings) >= 4:
        return float(np.mean(np.diff(zero_crossings)[::2])) * 2 / fps
    return None


def uncertainty_g(L_m, dL_m, T, dT):
    """
    Error propagation for g = 4pi²L/T²:
    dg/g = sqrt((dL/L)² + (2*dT/T)²)
    """
    g = calc_g_simple(L_m, T)
    dg = g * np.sqrt((dL_m / L_m) ** 2 + (2 * dT / T) ** 2)
    return g, dg


def analyze_damped(traj_x, fps, L_m=0.4):
    """
    Full damped oscillation analysis:
    1. Fit damped model
    2. Extract T0 (undamped period) and gamma
    3. Calculate g with and without damping correction
    """
    t = np.arange(len(traj_x)) / fps
    popt, pcov = fit_damped_model(t, traj_x, fps)
    if popt is None:
        return {"error": "Damped fit failed"}

    A, gamma, f, phi, C = popt
    T_raw = 1.0 / f
    T_corrected = correct_for_damping(T_raw, gamma)

    g_simple = calc_g_simple(L_m, T_raw)
    g_damped = calc_g_simple(L_m, T_corrected)

    param_names = ["amplitude", "gamma", "frequency", "phase", "offset"]
    param_errors = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros(5)

    return {
        "A": A, "gamma": gamma, "f": f, "phi": phi, "C": C,
        "param_errors": dict(zip(param_names, param_errors)),
        "T_raw": T_raw, "T_corrected": T_corrected,
        "g_simple": g_simple, "g_damped": g_damped,
        "fitted_curve": damped_oscillation(t, *popt),
        "popt": popt, "pcov": pcov,
    }
