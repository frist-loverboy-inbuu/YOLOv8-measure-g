# ============================================================
# step4_measure.py —— 周期提取 + 物理标定（像素→真实长度）
#
# 修复内容：
#   1. FPS自动从track_meta.npy读取（step3保存的），不再硬编码
#   2. PIXEL_PER_CM自动从track_meta.npy读取（step3标定的）
#   3. 如果没有track_meta.npy，才使用下面的手动配置值作为后备
#
# ▶ 运行方式：python step4_measure.py
# ▶ 输出：measurement.npy（给 step5 用）
# ============================================================

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ──────────────────────────────────────────────
# 后备值：只有在没有track_meta.npy时才用这里的值
# （正常情况下step3会自动保存，不需要手动改）
FALLBACK_FPS        = 30.0    # 手动填的FPS后备值
FALLBACK_PPcm       = None    # 像素/cm后备值，None表示不用像素换算
KNOWN_L_CM          = 15.5    # 已知摆长(cm)，如果做了像素标定可以设为None
# ──────────────────────────────────────────────


def load_meta():
    """读取step3保存的元数据（FPS + 像素标定）"""
    try:
        meta = np.load("track_meta.npy", allow_pickle=True).item()
        fps = float(meta["fps"])
        ppcm = meta.get("pixel_per_cm", None)
        print(f"📂 读取元数据：FPS={fps:.2f}, PIXEL_PER_CM={ppcm}")
        return fps, ppcm
    except FileNotFoundError:
        print(f"⚠️  未找到track_meta.npy，使用后备值：FPS={FALLBACK_FPS}, PIXEL_PER_CM={FALLBACK_PPcm}")
        return FALLBACK_FPS, FALLBACK_PPcm


def extract_period(x_coords, fps):
    """从x坐标序列提取摆动周期"""
    x = np.array(x_coords)
    x_centered = x - x.mean()

    # 找极大值峰 / 极小值谷
    peaks_max, _ = find_peaks( x_centered, distance=fps * 0.5, prominence=15)
    peaks_min, _ = find_peaks(-x_centered, distance=fps * 0.5, prominence=15)

    # 画图
    plt.figure(figsize=(13, 4))
    plt.plot(x, color="#378ADD", linewidth=1, alpha=0.8, label="质心x")
    plt.plot(peaks_max, x[peaks_max], "rv", markersize=8, label="极大值")
    plt.plot(peaks_min, x[peaks_min], "g^", markersize=8, label="极小值")
    plt.xlabel("帧编号")
    plt.ylabel("x坐标（像素）")
    plt.title("峰值检测结果（红色=极大，绿色=极小）")
    plt.legend()
    plt.tight_layout()
    plt.savefig("peaks_detection.png", dpi=150)
    plt.show()

    # 用相邻同类峰值间距计算周期
    periods = []
    if len(peaks_max) >= 2:
        periods.extend((np.diff(peaks_max) / fps).tolist())
    if len(peaks_min) >= 2:
        periods.extend((np.diff(peaks_min) / fps).tolist())

    if len(periods) == 0:
        print("❌ 没有找到足够的峰值，请检查轨迹数据或调整参数")
        return None, None, None

    T_mean = float(np.mean(periods))
    T_std  = float(np.std(periods))

    print(f"\n📐 周期分析：")
    print(f"   FPS（自动读取）= {fps:.2f}")
    print(f"   找到极大值：{len(peaks_max)} 个")
    print(f"   找到极小值：{len(peaks_min)} 个")
    print(f"   周期均值 T = {T_mean:.4f} 秒")
    print(f"   标准差     = {T_std:.4f} 秒")

    return T_mean, T_std, periods


def calc_pendulum_length(y_coords, ppcm):
    """
    通过像素换算摆长：
    悬挂点≈y最小处，摆球≈y均值，两者像素差 / PIXEL_PER_CM = 摆长(cm)
    """
    pivot_y     = y_coords.min()
    ball_y_mean = y_coords.mean()
    L_pixel     = abs(ball_y_mean - pivot_y)
    L_cm        = L_pixel / ppcm

    print(f"\n📏 摆长（像素换算）：")
    print(f"   悬挂点估算 y = {pivot_y:.0f} px")
    print(f"   摆球均值  y  = {ball_y_mean:.0f} px")
    print(f"   像素摆长     = {L_pixel:.0f} px")
    print(f"   PIXEL_PER_CM = {ppcm:.4f}")
    print(f"   真实摆长   L = {L_cm:.2f} cm")
    return L_cm


def main():
    # ── 读取元数据（FPS + 标定值）──
    fps, ppcm = load_meta()

    # ── 读取轨迹 ──
    try:
        traj = np.load("trajectory.npy")
    except FileNotFoundError:
        print("❌ 找不到 trajectory.npy，请先运行 step3_track.py")
        return

    x_coords = traj[:, 0]
    y_coords = traj[:, 1]

    # ── 提取周期 T ──
    T, T_std, all_periods = extract_period(x_coords, fps)
    if T is None:
        return

    # ── 确定摆长 L ──
    if KNOWN_L_CM is not None:
        L_cm = KNOWN_L_CM
        print(f"\n📏 摆长（手动输入）：L = {L_cm:.2f} cm")
    elif ppcm is not None:
        L_cm = calc_pendulum_length(y_coords, ppcm)
    else:
        print("❌ 没有摆长数据，请：")
        print("   方案A：在step3标定时拖动画线框选参照物")
        print("   方案B：在step4顶部设置 KNOWN_L_CM = 你的摆长(cm)")
        return

    L_m = L_cm / 100.0

    # ── 计算 g ──
    g = 4 * np.pi**2 * L_m / T**2
    print(f"\n⚡ 单次 g 值计算：")
    print(f"   T = {T:.4f} s")
    print(f"   L = {L_m:.4f} m")
    print(f"   g = 4π²L/T² = {g:.4f} m/s²")
    print(f"   误差 = {abs(g - 9.80) / 9.80 * 100:.2f}%（对比标准值9.80）")

    # ── 保存 ──
    result = {
        "T": T,
        "T_std": T_std,
        "L_m": L_m,
        "g_single": g,
        "all_periods": all_periods,
        "FPS": fps,
        "pixel_per_cm": ppcm,
    }
    np.save("measurement.npy", result, allow_pickle=True)
    print(f"\n💾 结果已保存 → measurement.npy")


if __name__ == "__main__":
    main()
