# ============================================================
# step5_calculate.py —— 多组线性回归提升精度 + 最终报告
#
# 用途：
#   把多次不同摆长的实验数据做线性拟合
#   利用 T² = (4π²/g)·L 的线性关系，精确求 g
#
# ▶ 使用前修改：EXPERIMENTS 列表，填入多组(L_cm, T)实验数据
#   每次实验改变摆长，重新跑 step3+step4，记录结果
# ▶ 运行方式：python step5_calculate.py
# ▶ 输出：results_report.png（结果图）
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ──────────────────────────────────────────────
# ⚠️ 修改这里：填入你多次实验的结果
# 格式：(摆长L单位cm, 周期T单位秒)
# 至少做3组，越多越准，每次改变摆长重复实验
#
# 例子（先用这个测试，再换成你自己的数据）：
EXPERIMENTS = [
    # (L_cm, T_秒)
    (30.0, 1.10),    # 第1组：摆长30cm，周期1.10s
    (40.0, 1.27),    # 第2组：摆长40cm，周期1.27s
    (50.0, 1.42),    # 第3组：摆长50cm，周期1.42s
    (60.0, 1.55),    # 第4组：摆长60cm，周期1.55s
    (70.0, 1.68),    # 第5组：摆长70cm，周期1.68s
]
# ──────────────────────────────────────────────

# 也可以读取上一步自动保存的单次测量
# （如果你只做了一组实验，就先看单次结果）
USE_SAVED = False    # 改成 True 则自动读取 measurement.npy 里的数据追加

def main():
    data = list(EXPERIMENTS)

    # 可选：读取 measurement.npy 追加
    if USE_SAVED:
        try:
            m = np.load("measurement.npy", allow_pickle=True).item()
            L_cm_saved = m["L_m"] * 100
            T_saved    = m["T"]
            data.append((L_cm_saved, T_saved))
            print(f"📥 读取到保存的测量：L={L_cm_saved:.1f}cm, T={T_saved:.4f}s")
        except:
            print("⚠️  没有找到 measurement.npy，跳过")

    # ── 数据处理 ──────────────────────────────────────
    L_arr  = np.array([d[0] for d in data]) / 100.0   # cm → m
    T_arr  = np.array([d[1] for d in data])
    T2_arr = T_arr ** 2                                 # T²

    # 线性回归：T² = k·L + b，其中 k = 4π²/g
    slope, intercept, r_value, p_value, std_err = stats.linregress(L_arr, T2_arr)
    g_fit = 4 * np.pi**2 / slope

    # 每组单次 g
    g_single = 4 * np.pi**2 * L_arr / T2_arr

    print("\n" + "="*50)
    print("  📊 各组单次 g 值")
    print("="*50)
    for i, (l, t) in enumerate(data):
        gs = 4 * np.pi**2 * (l/100) / t**2
        err = abs(gs - 9.80) / 9.80 * 100
        print(f"  第{i+1}组 L={l:.1f}cm  T={t:.4f}s  g={gs:.4f}  误差{err:.2f}%")

    print("\n" + "="*50)
    print("  🎯 线性拟合结果")
    print("="*50)
    print(f"  斜率 k = {slope:.4f}")
    print(f"  R²   = {r_value**2:.6f}  （越接近1越好）")
    print(f"  g（拟合）= 4π²/k = {g_fit:.4f} m/s²")
    print(f"  误差     = {abs(g_fit - 9.80) / 9.80 * 100:.2f}%")
    print("="*50)

    # ── 画报告图 ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"单摆测量重力加速度  g = {g_fit:.4f} m/s²  (误差 {abs(g_fit-9.80)/9.80*100:.2f}%)",
                 fontsize=13, fontweight="bold")

    # 左图：T² vs L 拟合直线
    ax = axes[0]
    L_fit = np.linspace(L_arr.min() * 0.9, L_arr.max() * 1.05, 100)
    T2_fit = slope * L_fit + intercept
    ax.scatter(L_arr * 100, T2_arr, color="#378ADD", s=60, zorder=5, label="实验数据")
    ax.plot(L_fit * 100, T2_fit, color="#D85A30", linewidth=2, label=f"拟合直线 R²={r_value**2:.4f}")
    ax.set_xlabel("摆长 L (cm)")
    ax.set_ylabel("周期² T² (s²)")
    ax.set_title("T² - L 线性关系")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图：各组 g 值对比
    ax2 = axes[1]
    x_pos = np.arange(len(g_single))
    bars = ax2.bar(x_pos, g_single, color="#7F77DD", alpha=0.8, label="单次g")
    ax2.axhline(y=9.80,    color="gray",    linestyle="--", alpha=0.7, label="标准值 9.80")
    ax2.axhline(y=g_fit,   color="#D85A30", linestyle="-",  linewidth=2, label=f"拟合值 {g_fit:.4f}")
    ax2.set_xlabel("实验组")
    ax2.set_ylabel("g (m/s²)")
    ax2.set_title("各组 g 值对比")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"第{i+1}组" for i in x_pos])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    # 在柱子上标数值
    for bar, gv in zip(bars, g_single):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{gv:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("results_report.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n📊 报告图已保存到 results_report.png")

if __name__ == "__main__":
    main()
