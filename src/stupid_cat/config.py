from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from stupid_cat.geometry import is_convex_polygon

T = TypeVar("T")

FUSION_MODES = frozenset({"weighted_median", "weighted_mean", "best_frame"})
INPUT_MODES = frozenset({"bgr", "gray3"})


class ConfigError(ValueError):
    """Invalid configuration."""


@dataclass
class ServiceConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    pause_on_start: bool = False


@dataclass
class InferenceConfig:
    device: str = "cpu"
    yolo_model: str = "models/yolo11s.pt"
    yolo_conf: float = 0.20
    yolo_confirm_frames: int = 3
    imgsz: int = 640
    active_fps: int = 6
    idle_fps: int = 1
    motion_threshold: int = 25
    min_crop_px: int = 80
    reid_backbone: str = "efficientnet_b0"
    similarity_threshold: float = 0.55
    fusion: str = "weighted_median"
    fusion_max_frames: int = 64
    fp16: bool = False  # half precision; only applied on CUDA devices
    # COCO class ids accepted as "the cat". A hunched/back-to-camera cat in the
    # litter box is often misclassified by YOLO (e.g. as sheep/dog/bear), and the
    # box only ever contains a cat, so accept the commonly-confused animal classes
    # to avoid losing the cat (which would split one visit into several).
    # 15 cat, 16 dog, 17 horse, 18 sheep, 19 cow, 21 bear, 77 teddy bear.
    detect_class_ids: list[int] = field(default_factory=lambda: [15, 16, 17, 18, 19, 21, 77])


@dataclass
class CatSeed:
    id: str
    name: str


@dataclass
class CatsConfig:
    min_refs: int = 5
    seed: list[CatSeed] = field(
        default_factory=lambda: [
            CatSeed("mimi", "咪咪"),
            CatSeed("cat2", "猫2"),
            CatSeed("cat3", "猫3"),
            CatSeed("cat4", "猫4"),
            CatSeed("cat5", "猫5"),
        ]
    )


@dataclass
class CameraConfig:
    id: str
    name: str = ""
    rtsp_url: str = ""
    weight: float = 0.5
    enabled: bool = True
    stream_width: int = 1920
    stream_height: int = 1080
    roi_polygon: list[list[float]] = field(default_factory=list)


@dataclass
class SessionConfig:
    roi_overlap_min: float = 0.25
    enter_overlap_sec: float = 2.0
    exit_no_cat_sec: float = 8.0
    cooldown_sec: float = 3.0
    min_visit_sec: float = 3.0


@dataclass
class PreprocessConfig:
    input_mode: str = "bgr"
    clahe_enabled: bool = True
    denoise_enabled: bool = False


@dataclass
class RecorderConfig:
    enabled: bool = True
    primary_camera: str = "cam1"
    # Which cameras to write visit clips from. Defaults to primary only; set
    # [cam1, cam2] (or all enabled ids) for dual-angle review on the timeline.
    record_cameras: list[str] | None = None
    max_seconds: int = 30
    min_free_mb: int = 500  # skip recording a visit if free disk space is below this


@dataclass
class MqttConfig:
    enabled: bool = False
    broker_host: str = "192.168.1.1"
    broker_port: int = 1883
    topic_prefix: str = "stupid-cat/visit"
    username: str = ""
    password: str = ""


@dataclass
class WasteConfig:
    enabled: bool = False
    min_duration_sec: int = 8  # shorter visits are 'unknown' (a sniff, not elimination)
    pee_max_duration_sec: int = 90
    poop_min_duration_sec: int = 120
    dig_motion_threshold: int = 40


def _default_cameras() -> list[CameraConfig]:
    return [
        CameraConfig(
            id="cam1",
            rtsp_url="rtsp://user:pass@192.168.1.101:554/stream1",
            roi_polygon=[[100, 80], [500, 80], [500, 400], [100, 400]],
        ),
        CameraConfig(
            id="cam2",
            rtsp_url="rtsp://user:pass@192.168.1.102:554/stream1",
            roi_polygon=[[100, 80], [500, 80], [500, 400], [100, 400]],
        ),
    ]


@dataclass
class AppConfig:
    service: ServiceConfig = field(default_factory=ServiceConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    cats: CatsConfig = field(default_factory=CatsConfig)
    cameras: list[CameraConfig] = field(default_factory=_default_cameras)
    session: SessionConfig = field(default_factory=SessionConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    mqtt: MqttConfig = field(default_factory=MqttConfig)
    waste: WasteConfig = field(default_factory=WasteConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _only_known_keys(section: str, data: dict[str, Any], cls: type[T]) -> dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"unknown keys in {section}: {sorted(unknown)}")
    return data


def _merge_section(defaults: dict[str, Any], override: dict[str, Any], cls: type[T]) -> dict[str, Any]:
    filtered = _only_known_keys(cls.__name__, override, cls)
    return {**defaults, **filtered}


def _dict_to_config(data: dict[str, Any]) -> AppConfig:
    cfg = AppConfig()
    if "service" in data:
        merged = _merge_section(cfg.service.__dict__, data["service"], ServiceConfig)
        cfg.service = ServiceConfig(**merged)
    if "inference" in data:
        merged = _merge_section(cfg.inference.__dict__, data["inference"], InferenceConfig)
        cfg.inference = InferenceConfig(**merged)
    if "cats" in data:
        cats_raw = _only_known_keys("cats", data["cats"], CatsConfig)
        seed_raw = cats_raw.get("seed", [])
        cfg.cats = CatsConfig(
            min_refs=cats_raw.get("min_refs", cfg.cats.min_refs),
            seed=[CatSeed(**_only_known_keys("cat seed", s, CatSeed)) for s in seed_raw]
            if seed_raw
            else cfg.cats.seed,
        )
    if "cameras" in data:
        cameras = []
        for i, raw in enumerate(data["cameras"]):
            merged = _merge_section(CameraConfig(id="").__dict__, raw, CameraConfig)
            cameras.append(CameraConfig(**merged))
        cfg.cameras = cameras
    if "session" in data:
        merged = _merge_section(cfg.session.__dict__, data["session"], SessionConfig)
        cfg.session = SessionConfig(**merged)
    if "preprocess" in data:
        merged = _merge_section(cfg.preprocess.__dict__, data["preprocess"], PreprocessConfig)
        cfg.preprocess = PreprocessConfig(**merged)
    if "recorder" in data:
        merged = _merge_section(cfg.recorder.__dict__, data["recorder"], RecorderConfig)
        cfg.recorder = RecorderConfig(**merged)
    if "mqtt" in data:
        merged = _merge_section(cfg.mqtt.__dict__, data["mqtt"], MqttConfig)
        cfg.mqtt = MqttConfig(**merged)
    if "waste" in data:
        merged = _merge_section(cfg.waste.__dict__, data["waste"], WasteConfig)
        cfg.waste = WasteConfig(**merged)
    return cfg


def validate_config(cfg: AppConfig) -> None:
    if not cfg.cameras:
        raise ConfigError("cameras must contain at least one entry")

    camera_ids: list[str] = []
    enabled_ids: list[str] = []
    for cam in cfg.cameras:
        if not cam.id:
            raise ConfigError("each camera requires a non-empty id")
        if cam.id in camera_ids:
            raise ConfigError(f"duplicate camera id: {cam.id}")
        camera_ids.append(cam.id)
        if not cam.enabled:
            continue  # disabled cameras are excluded from ingest/FSM/ROI checks
        enabled_ids.append(cam.id)
        if len(cam.roi_polygon) < 3:
            raise ConfigError(f"camera {cam.id}: roi_polygon needs at least 3 points")
        if not is_convex_polygon(cam.roi_polygon):
            raise ConfigError(f"camera {cam.id}: roi_polygon must be convex")

    if not enabled_ids:
        raise ConfigError("at least one camera must be enabled")

    # Only required when recording is on; otherwise primary_camera is unused at
    # runtime (the --video debug path falls back to the first enabled camera), so
    # validating it unconditionally would reject otherwise-valid RTSP configs.
    if cfg.recorder.enabled and cfg.recorder.primary_camera not in enabled_ids:
        raise ConfigError(
            f"recorder.primary_camera '{cfg.recorder.primary_camera}' "
            f"must be an enabled camera, one of {enabled_ids}"
        )

    if cfg.recorder.enabled and cfg.recorder.record_cameras is not None:
        # Reject typos (ids that don't exist) but allow a currently-disabled
        # camera to remain listed — record_camera_ids() filters disabled ones at
        # runtime. Only checked when recording is on (else record_cameras is
        # unused, like primary_camera above).
        for cam_id in cfg.recorder.record_cameras:
            if cam_id not in camera_ids:
                raise ConfigError(
                    f"recorder.record_cameras contains unknown camera '{cam_id}'; "
                    f"known cameras: {camera_ids}"
                )

    if cfg.inference.fusion not in FUSION_MODES:
        raise ConfigError(
            f"inference.fusion must be one of {sorted(FUSION_MODES)}, got {cfg.inference.fusion!r}"
        )

    if cfg.preprocess.input_mode not in INPUT_MODES:
        raise ConfigError(
            f"preprocess.input_mode must be one of {sorted(INPUT_MODES)}, "
            f"got {cfg.preprocess.input_mode!r}"
        )


def load_config(path: Path | str, local_path: Path | str | None = None) -> AppConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    if local_path is not None:
        local_path = Path(local_path)
        with local_path.open(encoding="utf-8") as f:
            local_data: dict[str, Any] = yaml.safe_load(f) or {}
        data = _deep_merge(data, local_data)

    defaults = _dict_to_config({})
    merged = _deep_merge(_config_to_dict(defaults), data)
    cfg = _dict_to_config(merged)
    validate_config(cfg)
    return cfg


def _config_to_dict(cfg: AppConfig) -> dict[str, Any]:
    return {
        "service": cfg.service.__dict__,
        "inference": cfg.inference.__dict__,
        "cats": {"min_refs": cfg.cats.min_refs, "seed": [s.__dict__ for s in cfg.cats.seed]},
        "cameras": [c.__dict__ for c in cfg.cameras],
        "session": cfg.session.__dict__,
        "preprocess": cfg.preprocess.__dict__,
        "recorder": cfg.recorder.__dict__,
        "mqtt": cfg.mqtt.__dict__,
        "waste": cfg.waste.__dict__,
    }
