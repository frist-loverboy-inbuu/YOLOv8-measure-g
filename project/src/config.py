import yaml
from pathlib import Path
from dataclasses import dataclass, field
@dataclass
class PathsConfig:
    model: str = "runs/detect/ball_detector/weights/best.pt"
    video: str = "21.mp4"
    output_dir: str = "output"


@dataclass
class PendulumConfig:
    length_cm: float = 15.5
    use_pixel_calibration: bool = False


@dataclass
class TrackingConfig:
    conf_threshold: float = 0.1
    iou_threshold: float = 0.3
    track_history: int = 80
    box_thickness: int = 5
    target_class: int = 0


@dataclass
class PeriodConfig:
    method: str = "fft"
    peak_prominence: float = 15.0
    peak_min_distance_sec: float = 0.3


@dataclass
class VideoConfig:
    fps_fallback: float = 30.0
    save_annotated: bool = False


@dataclass
class DatasetConfig:
    train_ratio: float = 0.8
    image_dir: str = "dataset/images"
    label_dir: str = "dataset/labels"
    output_dir: str = "dataset"


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    pendulum: PendulumConfig = field(default_factory=PendulumConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    period: PeriodConfig = field(default_factory=PeriodConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


def load_config(path: str = "config.yaml") -> AppConfig:
    p = Path(path)
    if not p.exists():
        print(f"config.yaml not found at {p}, using defaults")
        return AppConfig()
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = AppConfig()
    if raw:
        for section, cls in [
            ("paths", PathsConfig),
            ("pendulum", PendulumConfig),
            ("tracking", TrackingConfig),
            ("period", PeriodConfig),
            ("video", VideoConfig),
            ("dataset", DatasetConfig),
        ]:
            if section in raw:
                for k, v in raw[section].items():
                    if hasattr(getattr(cfg, section), k):
                        setattr(getattr(cfg, section), k, v)
    return cfg
