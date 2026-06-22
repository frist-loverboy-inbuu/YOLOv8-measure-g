import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from pathlib import Path


def plot_trajectory_comparison(traj_raw, traj_filt, save_path="output/trajectory_comparison.png"):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    colors = {"raw": "#378ADD40", "filt": "#E84855"}
    axes[0].plot(traj_raw[:, 0], alpha=0.4, linewidth=0.8, color="#378ADD", label="Raw X")
    axes[0].plot(traj_filt[:, 0], linewidth=1.2, color="#E84855", label="Kalman X")
    axes[0].set_ylabel("X (px)"); axes[0].legend(); axes[0].set_title("X-coordinate: Raw vs Kalman Filtered")
    axes[1].plot(traj_raw[:, 1], alpha=0.4, linewidth=0.8, color="#378ADD", label="Raw Y")
    axes[1].plot(traj_filt[:, 1], linewidth=1.2, color="#E84855", label="Kalman Y")
    axes[1].set_ylabel("Y (px)"); axes[1].set_xlabel("Frame"); axes[1].legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_period_analysis(period_result, fps, traj_x, save_path="output/period_analysis.png"):
    method = period_result.get("method", "unknown")
    T = period_result.get("T")

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(np.arange(len(traj_x)) / fps, traj_x, linewidth=1, color="#2E86AB")
    if method in ("peaks", "all") and "peaks_max" in period_result:
        pmax = period_result["peaks_max"]
        pmin = period_result["peaks_min"]
        ax1.plot(pmax / fps, [traj_x[i] for i in pmax], "rv", markersize=6, label="Peaks")
        ax1.plot(pmin / fps, [traj_x[i] for i in pmin], "g^", markersize=6, label="Valleys")
        ax1.legend()
    ax1.set_ylabel("X (px)"); ax1.set_title(f"Pendulum Trajectory  |  T = {T:.4f}s  (method: {method})")

    if method in ("fft", "all") and "fft_freqs" in period_result:
        ax2 = fig.add_subplot(3, 1, 2)
        freqs = period_result["fft_freqs"]
        mags = period_result["fft_magnitudes"]
        ax2.plot(freqs, mags, color="#D85A30", linewidth=1.5)
        ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("Magnitude")
        ax2.set_title("FFT Spectrum")
        f_peak = 1.0 / T if T else None
        if f_peak:
            ax2.axvline(x=f_peak, color="green", linestyle="--", label=f"f={f_peak:.3f} Hz")
            ax2.legend()

    if method in ("autocorrelation", "all") and "acf_data" in period_result:
        ax3 = fig.add_subplot(3, 1, 3)
        acf = period_result["acf_data"]
        lags = np.arange(len(acf)) / fps
        ax3.plot(lags, acf, color="#7F77DD", linewidth=1.5)
        ax3.set_xlabel("Lag (s)"); ax3.set_ylabel("Autocorrelation")
        ax3.set_title("Autocorrelation")
        T_acf = period_result.get("T_acf")
        if T_acf:
            ax3.axvline(x=T_acf, color="green", linestyle="--", label=f"T={T_acf:.3f}s")
            ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_damped_fit(t, traj_x, damped_result, save_path="output/damped_fit.png"):
    if damped_result is None or "error" in damped_result:
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(t, traj_x, alpha=0.5, linewidth=1, color="#378ADD", label="Measured")
    ax.plot(t, damped_result["fitted_curve"], linewidth=2, color="#E84855",
            label=f"Fit: T={damped_result['T_raw']:.4f}s  gamma={damped_result['gamma']:.4f}")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("X (px)")
    ax.set_title(f"Damped Oscillation Fit  |  T_corrected={damped_result.get('T_corrected', 0):.4f}s  "
                 f"g_damped={damped_result.get('g_damped', 0):.4f} m/s²")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_results_summary(results, save_path="output/results_summary.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    lines = []
    lines.append("Pendulum g Measurement Report")
    lines.append("=" * 45)
    for k, v in results.items():
        if isinstance(v, float):
            lines.append(f"  {k:<20s}: {v:.6f}")
        elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], float):
            vs = [f"{x:.4f}" for x in v[:5]]
            lines.append(f"  {k:<20s}: [{', '.join(vs)}{'...' if len(v) > 5 else ''}]")
        else:
            lines.append(f"  {k:<20s}: {v}")
    y_pos = np.linspace(0.9, 0.1, len(lines))
    for i, line in enumerate(lines):
        ax.text(0.05, y_pos[i], line, fontfamily="monospace", fontsize=11, va="top")
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
