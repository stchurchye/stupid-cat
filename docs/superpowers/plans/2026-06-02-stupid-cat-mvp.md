# stupid-cat Phase 1 MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付可运行的 `stupid-cat` Phase 1：双路 RTSP（或样例视频）、YOLO 检猫、Re-ID 认五猫（含 unknown）、visit 时长、SQLite、本机网页时间线与纠错。

**Architecture:** 单进程 Python 服务：`ingest` 拉帧 → `preprocess` → `detector` → `session`（ROI FSM + 多帧融合）→ `reid` → `recorder` + `db`；FastAPI 同进程提供 REST/静态页。Mac 用 CPU/MPS + fixtures；Win1060 用 CUDA + 真 RTSP。

**Tech Stack:** Python 3.11, PyTorch, Ultralytics YOLO11s, torchvision (EfficientNet-B0), OpenCV, FastAPI, uvicorn, SQLite, pytest.

**Spec:** [2026-06-02-stupid-cat-litter-vision-design.md](../specs/2026-06-02-stupid-cat-litter-vision-design.md) v0.3

---

## File map (Phase 1)

| Path | Responsibility |
|------|----------------|
| `pyproject.toml` | 包元数据、pytest 入口 |
| `requirements-dev.txt` | Mac：torch CPU、ultralytics、opencv、fastapi、pytest |
| `requirements-win-cuda.txt` | Win：torch+cu121、同上 |
| `config.yaml` | 示例配置（含 `cats.seed`、`session.roi_overlap_min`） |
| `.gitignore` | `data/`、`config.local.yaml`、`models/*.pt`、`__pycache__` |
| `src/stupid_cat/__init__.py` | 版本号 |
| `src/stupid_cat/config.py` | 加载 yaml + local 覆盖 |
| `src/stupid_cat/db.py` | SQLite schema、migrations、CRUD |
| `src/stupid_cat/preprocess.py` | CLAHE、灰度模式 |
| `src/stupid_cat/detector.py` | YOLO 封装、主框选择 |
| `src/stupid_cat/reid.py` | backbone、embedding、质心、融合 |
| `src/stupid_cat/geometry.py` | ROI 多边形交集比 |
| `src/stupid_cat/session.py` | Visit FSM（idle/active/cooldown） |
| `src/stupid_cat/recorder.py` | visit 录像 max_seconds |
| `src/stupid_cat/ingest.py` | RTSP/视频文件、帧率限制、运动门控 |
| `src/stupid_cat/pipeline.py` | 主循环、暂停标志 |
| `src/stupid_cat/api/app.py` | FastAPI 路由 |
| `src/stupid_cat/web/static/` | 时间线 HTML/JS |
| `src/stupid_cat/__main__.py` | CLI 入口 |
| `scripts/build_embeddings.py` | 扫描 refs → centroid.npy |
| `scripts/install_win.ps1` | Win venv + pip |
| `tests/` | 单元测试（无 GPU 可跑） |
| `fixtures/README.md` | 说明如何放置 `sample_ir.mp4` |

**Out of scope (Phase 2):** `publisher.py`, `waste.py`, `firmware/esp32_oled/`

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `requirements-dev.txt`, `requirements-win-cuda.txt`, `.gitignore`, `src/stupid_cat/__init__.py`, `README.md`

- [x] **Step 1:** 创建包布局与 `pyproject.toml`（`[project] name = "stupid-cat"`，`packages = [{include = "stupid_cat", from = "src"}]`）

- [x] **Step 2:** `requirements-dev.txt` 固定版本范围示例：
  ```
  torch>=2.2
  torchvision>=0.17
  ultralytics>=8.3
  opencv-python-headless>=4.9
  fastapi>=0.110
  uvicorn[standard]>=0.27
  pyyaml>=6.0
  numpy>=1.26
  pytest>=8.0
  httpx>=0.27
  ```

- [x] **Step 3:** `README.md` 写 Mac 安装、`pytest`、fixtures 说明、Win 指向 `install_win.ps1`

- [x] **Step 4:** 验证
  ```bash
  cd "/Users/hongpengwang/硬件项目/stupid cat"
  python3.11 -m venv .venv && source .venv/bin/activate
  pip install -r requirements-dev.txt
  pytest --collect-only
  ```
  Expected: 0 tests collected（尚未写测试）

---

## Task 2: Configuration

**Files:**
- Create: `config.yaml`, `src/stupid_cat/config.py`, `tests/test_config.py`

- [x] **Step 1: 写失败测试**

```python
# tests/test_config.py
from stupid_cat.config import load_config

def test_load_config_merges_defaults(tmp_path):
    cfg = load_config(tmp_path / "config.yaml")
    assert cfg.session.roi_overlap_min == 0.25
    assert cfg.inference.fusion == "weighted_median"
    assert len(cfg.cats.seed) >= 1
```

- [x] **Step 2:** `pytest tests/test_config.py -v` → FAIL

- [x] **Step 3:** 实现 `config.py`（dataclass：`ServiceConfig`, `InferenceConfig`, `SessionConfig`, `CameraConfig`, `CatsConfig`；`load_config(path, local_path=None)`）

- [x] **Step 4:** 提交 `config.yaml`（与 spec §7.1 一致，含 `cats.seed` 五只占位、`cooldown_sec`、`roi_overlap_min`）

- [x] **Step 5:** `pytest tests/test_config.py -v` → PASS

---

## Task 3: Geometry & ROI overlap

**Files:**
- Create: `src/stupid_cat/geometry.py`, `tests/test_geometry.py`

- [x] **Step 1: 测试 bbox-ROI 重叠比**

```python
def test_bbox_roi_overlap_ratio():
    from stupid_cat.geometry import bbox_roi_overlap_ratio
    bbox = (100, 100, 200, 200)  # x1,y1,x2,y2
    roi = [(0, 0), (300, 0), (300, 300), (0, 300)]
    r = bbox_roi_overlap_ratio(bbox, roi)
    assert 0.0 < r <= 1.0
```

- [x] **Step 2–4:** 实现（shapely 可选；可用 OpenCV `cv2.intersectConvexConvex` 或栅格化，YAGNI 优先简单多边形相交）

- [x] **Step 5:** PASS

---

## Task 4: Database

**Files:**
- Create: `src/stupid_cat/db.py`, `tests/test_db.py`

- [x] **Step 1: 测试建表与 visit 写入**

```python
def test_insert_visit(tmp_path):
    from stupid_cat.db import Database
    db = Database(tmp_path / "test.db")
    db.init_schema()
    db.seed_cats([{"id": "mimi", "name": "咪咪"}])
    vid = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    db.end_visit(vid, ended_at="2026-06-02T10:02:00+08:00", duration_sec=120, confidence=0.0)
    row = db.get_visit(vid)
    assert row["duration_sec"] == 120
```

- [x] **Step 2–4:** 实现 `init_schema`（`cats`, `visits`, `corrections` 按 spec §9）、`seed_cats`、`create_visit`、`end_visit`、`list_visits`、`correct_visit`

- [x] **Step 5:** PASS

---

## Task 5: Preprocess

**Files:**
- Create: `src/stupid_cat/preprocess.py`, `tests/test_preprocess.py`

- [x] **Step 1:** 测试 CLAHE 不改变 shape，`input_mode=grayscale3` 输出 3 通道

- [x] **Step 2:** 实现 `preprocess_frame(frame_bgr, cfg) -> ndarray`

- [x] **Step 3:** PASS

---

## Task 6: Re-ID & fusion

**Files:**
- Create: `src/stupid_cat/reid.py`, `tests/test_reid.py`

- [x] **Step 1: 测试融合（无模型权重，用随机向量）**

```python
def test_weighted_median_fusion():
    from stupid_cat.reid import fuse_embeddings
    import numpy as np
    embs = [np.ones(8), np.ones(8) * 2]
    weights = [0.5, 1.0]
    v = fuse_embeddings(embs, weights, mode="weighted_median")
    assert v.shape == (8,)
```

- [x] **Step 2:** 实现 `Embedder`（懒加载 EfficientNet-B0、`embed(crop_bgr)`、L2 normalize）

- [x] **Step 3:** 实现 `load_centroid(path)`、`match_cat(visit_vector, centroids, threshold)` → `(cat_id, confidence)`

- [x] **Step 4:** 实现 `fuse_embeddings` 三种 mode（spec §8.3）

- [x] **Step 5:** PASS（融合测试；embedder 可用 `pytest.mark.skipif` 无 torch 时跳过）

---

## Task 7: Detector

**Files:**
- Create: `src/stupid_cat/detector.py`, `tests/test_detector.py`

- [x] **Step 1:** 实现 `select_primary_bbox(boxes)` → 面积最大 cat 框

- [x] **Step 2:** 测试 primary bbox 选择（纯 Python 列表，不加载 YOLO）

- [x] **Step 3:** `CatDetector` 类：`detect(frame) -> list[bbox]`，封装 ultralytics；`confirm_frames` 逻辑放 session 或 detector 内

- [x] **Step 4:** 集成测试标记 `@pytest.mark.gpu`，仅 Win/有权重时运行

---

## Task 8: Visit session FSM

**Files:**
- Create: `src/stupid_cat/session.py`, `tests/test_session.py`

- [x] **Step 1: 测试状态流转（mock 检测，不用 YOLO）**

```python
def test_visit_lifecycle():
    from stupid_cat.session import VisitSessionFSM
    fsm = VisitSessionFSM(enter_sec=0.1, exit_no_cat_sec=0.2, cooldown_sec=0.1, min_visit_sec=0.05, roi_overlap_min=0.25)
    # 模拟连续合格检测 → started
    # 模拟全路无检测 → ended
    # 模拟 cooldown 内不重启
```

- [x] **Step 2:** 实现 per-camera 检测缓存、`on_frame(camera_id, qualified: bool, timestamp)`

- [x] **Step 3:** 事件回调 `on_visit_start`, `on_visit_end(visit_id, buffers...)`

- [x] **Step 4:** PASS

---

## Task 9: Recorder

**Files:**
- Create: `src/stupid_cat/recorder.py`, `tests/test_recorder.py`

- [x] **Step 1:** 实现 `VisitRecorder`：`start(visit_id, camera_id, frame)`、`write_frame`、`stop`；写盘不超过 `max_seconds`

- [x] **Step 2:** 测试：写入 35s 模拟帧（加速时钟或 mock time），文件时长 ≤30s（spec §9.4）

- [x] **Step 3:** PASS

---

## Task 10: Ingest

**Files:**
- Create: `src/stupid_cat/ingest.py`, `fixtures/README.md`

- [x] **Step 1:** `FrameSource` 抽象：`VideoFileSource(path)`、`RtspSource(url)`（OpenCV `CAP_FFMPEG`）

- [x] **Step 2:** `rate_limit(fps)` + `motion_gate(prev, curr, threshold)` 生成器

- [x] **Step 3:** `MultiCameraIngest` 合并双路 `cameras[]` 为 `(camera_id, frame, timestamp)` 流

- [x] **Step 4:** 文档说明：用户自备 `fixtures/sample_ir.mp4` 或使用 RTSP

---

## Task 11: Pipeline 主循环

**Files:**
- Create: `src/stupid_cat/pipeline.py`, `src/stupid_cat/__main__.py`

- [x] **Step 1:** `Pipeline` 组装 config、db、detector、embedder、fsm、recorder

- [x] **Step 2:** 冷启动：无 centroid → `cat_id=unknown`（spec §6.5）

- [x] **Step 3:** `on_visit_end`：融合 → match → `db.end_visit` → 写 `recording_path`

- [x] **Step 4:** `pause_event` / `resume()` 供 API 调用

- [x] **Step 5:** CLI：`python -m stupid_cat --config config.yaml`（可选 `--video fixtures/sample_ir.mp4` 单路调试）

- [x] **Step 6:** Mac 冒烟：`python -m stupid_cat --config config.yaml --video fixtures/sample_ir.mp4`（无视频则 skip 并打印说明）

---

## Task 12: FastAPI

**Files:**
- Create: `src/stupid_cat/api/app.py`, `tests/test_api.py`

- [x] **Step 1:** `create_app(pipeline, db)` 注册路由（spec §10）

- [x] **Step 2:** `GET /health` 返回 `{cuda, cameras: [{id, connected, last_frame_at}], paused}`

- [x] **Step 3:** `httpx` 测试 `GET /cats`、`GET /visits`、`POST /visits/{id}/correct`

- [x] **Step 4:** 后台线程跑 pipeline 或 lifespan 启动（YAGNI：单进程 asyncio + 线程）

- [x] **Step 5:** `uvicorn stupid_cat.api.app:app` 或 `__main__` 子命令 `serve`

---

## Task 13: Web 时间线

**Files:**
- Create: `src/stupid_cat/web/static/index.html`, `app.js`, `style.css`

- [x] **Step 1:** 静态页拉取 `/api/v1/visits`，表格展示：时间、猫、时长、置信度、waste、播放链接

- [x] **Step 2:** 纠错下拉 → `POST /visits/{id}/correct`

- [x] **Step 3:** 手动浏览器验证 `http://127.0.0.1:8765/`

---

## Task 14: build_embeddings 脚本

**Files:**
- Create: `scripts/build_embeddings.py`

- [x] **Step 1:** 遍历 `data/cats/*/refs/*.jpg` → 计算质心 → `data/cats/{id}/centroid.npy`

- [x] **Step 2:** 文档写入 README；API `POST /cats/{id}/rebuild-embedding` 调用同一函数

---

## Task 15: Windows 部署文档

**Files:**
- Create: `scripts/install_win.ps1`

- [x] **Step 1:** venv、`pip install -r requirements-win-cuda.txt`、下载 `yolo11s.pt` 到 `models/`

- [x] **Step 2:** 示例 `config.local.yaml` 模板复制

- [x] **Step 3:** 任务计划程序注册说明（`schtasks` 或文档步骤）

- [x] **Step 4:** Win1060 验证清单：nvidia-smi、双 RTSP、health endpoint

---

## Task 16: Phase 1 集成验证

- [ ] Mac：`pytest tests/ -v` 全绿（GPU 测试 skip）

- [ ] Mac：样例视频跑通至少 1 次 visit 写入 DB

- [ ] Win：双 RTSP + CUDA；网页见 visit；纠错后 centroid 更新

- [ ] 对照 spec §13.1 验收表自测记录（可写在 `docs/validation-phase1.md`）

---

## Plan self-review (vs spec v0.3)

| Spec 要求 | Task |
|-----------|------|
| 方案 2 YOLO+Re-ID | 6–7, 11 |
| 双路 → 全局 FSM（§8.5） | 8, 10, 11 |
| ROI stream 坐标 + letterbox 反变换（§7.3） | 3, 7, 11 |
| roi_overlap_min, cooldown | 2–3, 8 |
| 多框最大面积 | 7, 11 |
| fusion_max_frames、confidence/frames_used | 6, 8, 11 |
| 录像 primary_camera + /recordings | 9, 12 |
| 录像 max_seconds vs duration | 9, 11 |
| 冷启动 unknown、min_refs 质心 | 4, 11, 14 |
| SQLite + API + 网页 + /health | 4, 12–13 |
| waste unknown Phase1 | 4（字段默认） |
| pause 立即 end visit（§8.2.7） | 11–12 |
| Mac/Win 双 requirements | 1, 15 |

**Phase 2**（MQTT ESP32、waste 启发式）另写 `2026-06-02-stupid-cat-phase2.md`，不在本计划内。

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-02-stupid-cat-mvp.md`.

**Two execution options:**

1. **Subagent-Driven（推荐）** — 每 Task 派生子 agent，任务间你做 review  
2. **Inline Execution** — 本会话按 Task 顺序直接实现，检查点分批提交  

你更想用哪种？回复 **「subagent」** 或 **「inline 开始 Task 1」** 即可。
