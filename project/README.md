# AI Pendulum Gravitational Acceleration Measurement System

## Overview

基于 YOLOv8 目标检测 + ByteTrack 多目标跟踪的智能单摆重力加速度测量系统。通过摄像头拍摄单摆运动视频，自动追踪摆球质心轨迹，提取摆动周期，计算当地重力加速度 g。

## Pipeline

```
Dataset → Train YOLOv8 → Track Pendulum → Extract Period → Calculate g
  ↓           ↓               ↓                ↓                ↓
step1       step2          main.py           FFT + ACF       g = 4pi²L/T²
                           +Kalman filter    +peaks          +damped model
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Edit config.yaml (set video path, pendulum length, etc.)

# 3. Run the pipeline
python main.py --source 21.mp4 --L 15.2

# Or use defaults from config.yaml
python main.py
```

## Command-line Options

| Flag | Description | Default |
|---|---|---|
| `--config` | Config file path | `config.yaml` |
| `--source` | Video file or `0` for webcam | from config |
| `--model` | YOLOv8 weights path | from config |
| `--L` | Pendulum length (cm) | from config |
| `--method` | Period method: `fft`, `autocorrelation`, `peaks`, `all` | `fft` |
| `--save-video` | Save annotated video | `false` |
| `--calibrate` | Interactive pixel-to-cm calibration | `false` |
| `--skip-track` | Skip tracking (use existing trajectory.npy) | `false` |

## Algorithm Improvements

Compared to the original implementation:

1. **Kalman Filter**: Smooths YOLO detection noise, producing cleaner trajectory
2. **FFT Period Extraction**: Frequency-domain analysis, more robust to amplitude variation
3. **Autocorrelation**: Time-domain period verification
4. **Damped Oscillation Model**: Fits $x(t) = A e^{-\gamma t} \cos(2\pi f t + \phi) + C$ to extract true undamped period, correcting for energy loss
5. **Uncertainty Quantification**: Error propagation and confidence intervals

## Files

```
project/
├── config.yaml          # Centralized configuration
├── requirements.txt     # Python dependencies
├── main.py              # Unified pipeline entry point
├── ball.yaml            # YOLO dataset config
├── src/
│   ├── config.py        # Config loader
│   ├── tracker.py       # YOLO + ByteTrack + Kalman filter
│   ├── period.py        # FFT / autocorrelation / peak period extraction
│   ├── physics.py       # g calculation, damped model, uncertainty
│   └── visualize.py     # Plotting utilities
├── scripts/
│   ├── step1_split.py   # Dataset train/val split
│   ├── step2_train.py   # YOLOv8 model training
│   └── step5_analysis.py # Multi-experiment linear regression
├── dataset/             # Labeled images and labels
├── runs/                # Training outputs
└── output/              # Results (trajectory, plots, measurement)
```
