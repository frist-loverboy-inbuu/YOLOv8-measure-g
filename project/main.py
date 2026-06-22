import argparse
import sys
import numpy as np
from pathlib import Path

from src.config import load_config
from src.period import extract_period
from src.physics import calc_g_simple, analyze_damped
from src.visualize import (
    plot_trajectory_comparison,
    plot_period_analysis,
    plot_damped_fit,
    plot_results_summary,
)

Path("output").mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Pendulum g Measurement System")
    p.add_argument("--config", default="config.yaml", help="Config file path")
    p.add_argument("--source", default=None, help="Video path or 0 for webcam")
    p.add_argument("--model", default=None, help="YOLO model weight path")
    p.add_argument("--L", type=float, default=None, help="Pendulum length (cm)")
    p.add_argument("--fps", type=float, default=None, help="FPS fallback")
    p.add_argument("--method", default=None, choices=["fft", "autocorrelation", "peaks", "all"],
                   help="Period extraction method")
    p.add_argument("--save-video", action="store_true", default=None, help="Save annotated video")
    p.add_argument("--calibrate", action="store_true", help="Enable interactive pixel calibration")
    p.add_argument("--skip-track", action="store_true", help="Skip tracking, load existing trajectory.npy")
    p.add_argument("--headless", action="store_true", help="Run without GUI windows")
    p.add_argument("--test-synthetic", action="store_true", help="Run with synthetic pendulum data for testing")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    source = args.source or cfg.paths.video
    model_path = args.model or cfg.paths.model
    length_cm = args.L if args.L is not None else cfg.pendulum.length_cm
    fps_fb = args.fps if args.fps is not None else cfg.video.fps_fallback
    method = args.method or cfg.period.method
    save_video = args.save_video if args.save_video is not None else cfg.video.save_annotated
    do_calibrate = args.calibrate or cfg.pendulum.use_pixel_calibration
    L_m = length_cm / 100.0

    print("=" * 60)
    print("  AI Pendulum Gravitational Acceleration Measurement")
    print("=" * 60)
    print(f"  Video:    {source}")
    print(f"  Model:    {model_path}")
    print(f"  Length:   {length_cm} cm  ({L_m:.4f} m)")
    print(f"  Method:   {method}")
    print("=" * 60 + "\n")

    if args.test_synthetic:
        print("  [TEST MODE] Generating synthetic pendulum data")
        fps = 30.0
        t = np.arange(0, 10, 1 / fps)
        true_freq = 0.85
        true_T = 1.0 / true_freq
        x_clean = 120 * np.cos(2 * np.pi * true_freq * t + 0.3) + 320
        y_clean = 80 * np.sin(2 * np.pi * true_freq * t + 0.3) + 240
        np.random.seed(42)
        traj = np.column_stack([x_clean + np.random.randn(len(t)) * 3,
                                 y_clean + np.random.randn(len(t)) * 3])
        np.save("output/trajectory.npy", traj)
        np.save("output/track_meta.npy", {"fps": fps, "total_frames": len(traj)}, allow_pickle=True)
        print(f"  Generated {len(traj)} synthetic frames, FPS={fps}, true T={true_T:.4f}s")

    elif not args.skip_track:
        from src.tracker import PendulumTracker
        tracker = PendulumTracker(
            model_path,
            conf=cfg.tracking.conf_threshold,
            iou=cfg.tracking.iou_threshold,
            track_history=cfg.tracking.track_history,
            box_thickness=cfg.tracking.box_thickness,
            target_class=cfg.tracking.target_class,
        )

        cap = __import__("cv2").VideoCapture(source)
        ret, first_frame = cap.read()
        cap.release()

        ppcm = None
        if do_calibrate and ret:
            ppcm = tracker.calibrate_pixel_per_cm(first_frame)
            if ppcm:
                print(f"\nPixel calibration: {ppcm:.4f} px/cm")

        traj_raw, traj_filt, fps = tracker.process_video(
            source, fps_fallback=fps_fb, save_video=save_video, headless=args.headless
        )

        plot_trajectory_comparison(traj_raw, traj_filt)
        print("  Saved: output/trajectory_comparison.png")

        traj = traj_filt
    else:
        try:
            traj = np.load("output/trajectory.npy")
            fps = fps_fb
            meta_path = Path("output/track_meta.npy")
            if meta_path.exists():
                meta = np.load(meta_path, allow_pickle=True).item()
                fps = meta.get("fps", fps_fb)
            print(f"Loaded existing trajectory: {traj.shape[0]} points, FPS={fps}")
        except FileNotFoundError:
            print("No saved trajectory found. Run without --skip-track first.")
            sys.exit(1)

    x_coords = traj[:, 0]
    print(f"\n{'='*40}")
    print("  Period Extraction")
    print("=" * 40)

    period_result = extract_period(
        x_coords, fps, method=method,
        prominence=cfg.period.peak_prominence,
        min_dist_sec=cfg.period.peak_min_distance_sec,
    )

    T = period_result.get("T")
    if T is None:
        print("Period extraction failed.")
        sys.exit(1)

    if method == "all":
        print(f"  T_fft  = {period_result.get('T_fft', 'N/A')}")
        print(f"  T_acf  = {period_result.get('T_acf', 'N/A')}")
        print(f"  T_peaks= {period_result.get('T_peaks', 'N/A')}")
    print(f"  T      = {T:.6f} s")

    plot_period_analysis(period_result, fps, x_coords)
    print("  Saved: output/period_analysis.png")

    g_simple = calc_g_simple(L_m, T)
    print(f"\n{'='*40}")
    print("  Simple Pendulum g")
    print("=" * 40)
    print(f"  g = 4*pi^2*L/T^2 = {g_simple:.6f} m/s^2")
    print(f"  Error vs 9.80 = {abs(g_simple - 9.80) / 9.80 * 100:.3f}%")

    print(f"\n{'='*40}")
    print("  Damped Oscillation Analysis")
    print("=" * 40)
    damped = analyze_damped(x_coords, fps, L_m)
    if damped and "error" not in damped:
        print(f"  Amplitude A    = {damped['A']:.2f} px")
        print(f"  Damping gamma  = {damped['gamma']:.6f} s^-1")
        print(f"  T_raw          = {damped['T_raw']:.6f} s")
        print(f"  T_corrected    = {damped['T_corrected']:.6f} s")
        print(f"  g_simple       = {damped['g_simple']:.6f} m/s^2")
        print(f"  g_damped       = {damped['g_damped']:.6f} m/s^2")
        print(f"  Damping corr.  = {(damped['g_damped'] - damped['g_simple']):.6f} m/s^2")
        plot_damped_fit(np.arange(len(x_coords)) / fps, x_coords, damped)
        print("  Saved: output/damped_fit.png")
        g_final = damped["g_damped"]
    else:
        g_final = g_simple

    print(f"\n{'='*40}")
    print(f"  Final g = {g_final:.6f} m/s^2")
    print(f"  Error   = {abs(g_final - 9.80) / 9.80 * 100:.3f}%")
    print("=" * 40)

    results = {
        "T_s": T,
        "L_m": L_m,
        "L_cm": length_cm,
        "g_simple": g_simple,
        "g_final": g_final,
        "method": method,
        "fps": fps,
        "n_frames": len(x_coords),
    }
    np.save("output/measurement.npy", results, allow_pickle=True)
    plot_results_summary(results)
    print("\n  Results saved to output/\n")


if __name__ == "__main__":
    main()
