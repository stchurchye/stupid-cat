from pathlib import Path

import pytest

from stupid_cat.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]
MINIMAL_CAMERAS = """
cameras:
  - id: cam1
    roi_polygon: [[0, 0], [100, 0], [100, 100], [0, 100]]
  - id: cam2
    roi_polygon: [[0, 0], [100, 0], [100, 100], [0, 100]]
recorder:
  primary_camera: cam1
"""


def test_load_config_merges_defaults(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        f"session:\n  roi_overlap_min: 0.25\n{MINIMAL_CAMERAS}",
        encoding="utf-8",
    )
    cfg = load_config(tmp_path / "config.yaml")
    assert cfg.session.roi_overlap_min == 0.25
    assert cfg.inference.fusion == "weighted_median"
    assert cfg.inference.fusion_max_frames == 64
    assert len(cfg.cats.seed) >= 1


def test_load_config_merges_local_override(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        f"service:\n  port: 8765\n{MINIMAL_CAMERAS}",
        encoding="utf-8",
    )
    (tmp_path / "local.yaml").write_text("service:\n  port: 9000\n", encoding="utf-8")
    cfg = load_config(tmp_path / "config.yaml", local_path=tmp_path / "local.yaml")
    assert cfg.service.port == 9000


def test_repo_config_yaml_loads() -> None:
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.session.cooldown_sec == 3.0
    assert cfg.recorder.primary_camera == "cam1"
    assert len(cfg.cats.seed) == 5
    assert cfg.inference.device == "cpu"


def test_invalid_primary_camera_raises(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        MINIMAL_CAMERAS.replace("primary_camera: cam1", "primary_camera: cam99"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="primary_camera"):
        load_config(tmp_path / "config.yaml")


def test_empty_cameras_raises(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("cameras: []\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="cameras"):
        load_config(tmp_path / "config.yaml")


def test_invalid_fusion_raises(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        f"inference:\n  fusion: bogus\n{MINIMAL_CAMERAS}",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="fusion"):
        load_config(tmp_path / "config.yaml")


def test_unknown_config_key_raises(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        f"service:\n  prot: 8765\n{MINIMAL_CAMERAS}",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown"):
        load_config(tmp_path / "config.yaml")


def test_concave_roi_raises(tmp_path: Path) -> None:
    yaml = """
cameras:
  - id: cam1
    roi_polygon: [[0, 0], [100, 0], [0, 100], [100, 100]]
  - id: cam2
    roi_polygon: [[0, 0], [100, 0], [100, 100], [0, 100]]
recorder:
  primary_camera: cam1
"""
    (tmp_path / "config.yaml").write_text(yaml, encoding="utf-8")
    with pytest.raises(ConfigError, match="convex"):
        load_config(tmp_path / "config.yaml")
