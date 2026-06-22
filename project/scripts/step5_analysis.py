import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics import calc_g_simple

Path("output").mkdir(exist_ok=True)

EXPERIMENTS = [
    (30.0, 1.10),
    (40.0, 1.27),
    (50.0, 1.42),
    (60.0, 1.55),
    (70.0, 1.68),
]
USE_SAVED = False
G_STANDARD = 9.80


def main():
    data = list(EXPERIMENTS)
    if USE_SAVED:
        try:
            m = np.load("output/measurement.npy", allow_pickle=True).item()
            L_cm_saved = m["L_m"] * 100
            T_saved = m["T_s"]
            data.append((L_cm_saved, T_saved))
            print(f"Loaded saved measurement: L={L_cm_saved:.1f}cm, T={T_saved:.4f}s")
        except Exception:
            print("No saved measurement found, skipping.")

    L_arr = np.array([d[0] for d in data]) / 100.0
    T_arr = np.array([d[1] for d in data])
    T2_arr = T_arr ** 2

    slope, intercept, r_value, p_value, std_err = stats.linregress(L_arr, T2_arr)
    g_fit = 4 * np.pi ** 2 / slope
    g_single = 4 * np.pi ** 2 * L_arr / T2_arr

    slope_uncertainty = std_err
    g_fit_uncertainty = g_fit * (slope_uncertainty / slope)

    print("\n" + "=" * 50)
    print("  Linear Regression: T^2 vs L")
    print("=" * 50)
    print(f"  slope k     = {slope:.6f} +/- {slope_uncertainty:.6f}")
    print(f"  R^2         = {r_value ** 2:.6f}")
    print(f"  g (fit)     = {g_fit:.4f} +/- {g_fit_uncertainty:.4f} m/s^2")
    print(f"  Error       = {abs(g_fit - G_STANDARD) / G_STANDARD * 100:.2f}%")
    print("=" * 50)
    for i, (l, t) in enumerate(data):
        gs = calc_g_simple(l / 100, t)
        err = abs(gs - G_STANDARD) / G_STANDARD * 100
        print(f"  Group {i+1}: L={l:.1f}cm  T={t:.4f}s  g={gs:.4f}  err={err:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Pendulum g Measurement  g_fit = {g_fit:.4f} m/s^2  (R^2={r_value**2:.4f})",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    L_fit = np.linspace(L_arr.min() * 0.9, L_arr.max() * 1.05, 100)
    T2_fit = slope * L_fit + intercept
    ax.scatter(L_arr * 100, T2_arr, color="#378ADD", s=60, zorder=5, label="Data")
    ax.plot(L_fit * 100, T2_fit, color="#D85A30", linewidth=2,
            label=f"Fit: T^2 = {slope:.4f}*L {intercept:+.4f}")
    ax.fill_between(L_fit * 100,
                    (slope + slope_uncertainty) * L_fit + intercept,
                    (slope - slope_uncertainty) * L_fit + intercept,
                    alpha=0.15, color="#D85A30")
    ax.set_xlabel("L (cm)"); ax.set_ylabel("T^2 (s^2)")
    ax.set_title("T^2 vs L"); ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    x_pos = np.arange(len(g_single))
    bars = ax2.bar(x_pos, g_single, color="#7F77DD", alpha=0.8, label="Per-group g")
    ax2.axhline(y=G_STANDARD, color="gray", linestyle="--", alpha=0.7, label=f"Standard {G_STANDARD}")
    ax2.axhline(y=g_fit, color="#D85A30", linestyle="-", linewidth=2, label=f"Fit {g_fit:.4f}")
    ax2.fill_between([x_pos[0] - 0.5, x_pos[-1] + 0.5],
                     g_fit - g_fit_uncertainty, g_fit + g_fit_uncertainty,
                     alpha=0.15, color="#D85A30")
    ax2.set_xlabel("Group"); ax2.set_ylabel("g (m/s^2)")
    ax2.set_title("Per-group g Comparison")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"Group {i+1}" for i in x_pos])
    ax2.legend(); ax2.grid(True, alpha=0.3, axis="y")
    for bar, gv in zip(bars, g_single):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{gv:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("output/results_report.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nReport saved to output/results_report.png")


if __name__ == "__main__":
    main()
