# stupid-cat 未完成事项交接（换机续作）

> 更新：2026-06-02  
> 用途：在另一台电脑（主要是 **Win1060 生产机**）继续时，按本清单推进。  
> 规格：`docs/superpowers/specs/2026-06-02-stupid-cat-litter-vision-design.md` v0.3  
> MVP 计划：`docs/superpowers/plans/2026-06-02-stupid-cat-mvp.md`（Task 1–15 已勾选）

---

## 0. 2026-06-15 加固更新（PR #1）

分支 **`harden/stability-correctness`** → **PR https://github.com/stchurchye/stupid-cat/pull/1**（已推送，未合并）。
对原 MVP 做了一轮稳定性 / 部署 / 正确性加固，经两轮多 agent 对抗审查，`pytest` 全绿（81 passed, 1 skipped）。**不需要硬件**，目标是「相机一到、Win 上 `git pull` 即可稳跑 24/7」。

已修（详见 PR 描述）：
- **24/7 稳定性**：双摄改每路独立线程 + 有界队列；FSM 墙钟看门狗（断流也能结束 visit）；`RLock` + centroids 写时复制（纠错不再崩）；逐帧异常隔离；visit 开始即写库 + 启动恢复孤儿 visit；慢操作（质心重建、ffmpeg）移出锁。
- **Pascal/CUDA 部署**：锁定 `torch==2.4.1+cu121`（防 sm_61 内核被砍）；启动自检（CUDA 冒烟 + ffmpeg 检查，失败即中止而非崩溃循环）；设备感知 FP16。
- **认猫准确率（仅框架）**：修 `weighted_median` 权重失效；纠错代表帧去偏 + 过质量门控；参考图质量门控按**动态范围**（黑猫红外图不再被误杀）；embed 改 cv2。
- **正确性**：ROI 坐标系按实际帧缩放（§7.3，支持子码流）；相机 `enabled` 开关；磁盘空间保护；录像按墙钟封顶。

**仍未做（本 PR 之外，见 PR「next steps」）**：认猫**模型升级**（微调 / 度量学习 / per-cat 阈值，**需你提供 IR 参考图**，是 ≥75% 目标的最大风险）；TensorRT；多向量画廊；打游戏自动降载；UI 增强；Phase 2（MQTT+ESP32、屎尿分类）。下文 §3–§10 仍有效。

> 换机注意：测试环境用 `python3.12` + `pip install -e ".[dev]"`；Win 生产用 `requirements-win-cuda.txt`（已钉 cu121）+ `pip install -e .`。

---

## 1. 项目在哪、怎么拉起来

| 项 | 说明 |
|----|------|
| 路径（Mac 开发机） | `/Users/hongpengwang/硬件项目/stupid cat` |
| 换机 | 用 **Git / U 盘 / 网盘** 同步整个目录（`data/`、`models/`、`config.local.yaml` 默认 gitignore，需单独拷） |
| Python | 3.11+（Mac 上用过 3.12 venv 也可） |
| 安装 | `python -m venv .venv` → `pip install -e ".[dev]"` 或 `requirements-dev.txt` |
| Win CUDA | `scripts/install_win.ps1` + `requirements-win-cuda.txt` |

**常用命令：**

```bash
# 测试
pytest tests/ -q

# 仅 API + 网页（看已有 DB，不拉 RTSP）
python -m stupid_cat serve --api-only

# 样例视频跑 pipeline + 网页
python -m stupid_cat serve --video fixtures/clip-01.mp4

# 只跑识别、无网页
python -m stupid_cat run --video fixtures/clip-01.mp4

# 生成五猫质心
python scripts/build_embeddings.py --config config.yaml
```

**样例视频：** `fixtures/clip-01.mp4` … `clip-04.mp4`（Mac 上已从 Downloads 拷入，换机需一并复制）

**模型：** `models/yolo11s.pt`（首次运行可自动下载，建议拷到新机避免重复下）

---

## 2. 已完成（不必重做）

- [x] Phase 1 代码骨架：config / db / geometry / preprocess / reid / detector / session / recorder / ingest / pipeline / API / 最小网页
- [x] `pytest`：**62 passed**（含 pipeline 集成测试）
- [x] Mac 样例视频冒烟：visit 写入 DB、录像 H.264、墙钟 `duration_sec`
- [x] `serve --api-only`、录像网页 `<video>` 播放
- [x] 纠错 crop 落盘 `data/correction_crops/`；重启后纠错仍可用
- [x] code-review 一批修复：Embedder 线程锁、RTSP 重连、health `ingest_active` 等

---

## 3. Task 16 — 集成验证（未勾完）

见 `docs/validation-phase1.md`，剩余：

- [ ] **Win1060**：CUDA + **双路 RTSP** + `GET /api/v1/health` 正常
- [ ] **浏览器**：`http://127.0.0.1:8765/` 时间线、播放、纠错自测并打勾
- [ ] **spec §13.1**：身份 / 时长 / 召回抽样记录（7 天、≥50 visit 等，可后补）

**Win 首次跑通建议顺序：**

1. `nvidia-smi` 确认 GPU  
2. `config.local.yaml`：RTSP 地址、密码、`device: cuda:0`、**按真机分辨率标 ROI**  
3. VLC 双路 RTSP 各稳定 60s+  
4. `python -m stupid_cat serve`（不要 `--api-only`）  
5. 等猫如厕或人为触发，看 DB / 网页

---

## 4. 训练数据 / 认猫（未做）

当前 visit 多为 **`cat_id=unknown`**（正常冷启动）。

| 步骤 | 状态 |
|------|------|
| 每猫 IR 参考图 `data/cats/{id}/refs/*.jpg` | [ ] 未收集（spec 验收目标 ≥30 张/猫，代码 `min_refs: 5` 即可建质心） |
| `python scripts/build_embeddings.py` | [ ] 未在真 IR 上跑 |
| 网页纠错 → 追加 refs + 重建质心 | [ ] 代码已有，需真 visit + 人工试 |
| 调 `similarity_threshold`（默认 0.55） | [ ] 现场未调 |

**五猫 ID（config.yaml seed）：** `mimi`, `cat2`…`cat5`（可改名，与 `refs` 目录名一致）

**注意：** refs 必须 **IR 域**，与 RTSP 画面一致；不要混 RGB 手机客厅照。

---

## 5. 硬件采购与安装（未做）

按 spec §5，**不需要录像机（NVR）**。

| 要买 | 建议 |
|------|------|
| 摄像头 | **2×** PoE **枪机或半球**（定焦 **2.8mm**、**1080p** 即可，不必 4MP） |
| 夜视 | **红外黑白**，关白光；**不要**「夜视全彩」当主方案 |
| 供电 | PoE 受电摄像头 + **PoE 交换机** 或 **2× PoE 注入器** |
| 网络 | 摄像头 + Win1060 **有线** 接路由器/交换机 |
| 补光 | **850nm IR 灯条** + 12V（盖内） |

**机位：** 盖内两角——cam1 斜俯视、cam2 斜侧视（单路可先装一台，准确率低于双路）。

**接线：**

```text
cam1、cam2 → PoE 交换机 → 路由器 ← Win1060（有线）
```

若暂只买 1 台：注入器 + 路由器亦可；`config.local.yaml` 里 **只保留一个 `cameras` 条目**。

**到货后：** 写 RTSP 进 `config.local.yaml` → VLC 测流 → 标 ROI（勿长期用 Mac 调试用的「全画面 ROI」）。

---

## 6. 软件 / UI 未完成（Phase 1 增强）

MVP **最小网页已有**（表格 + 录像 + 纠错下拉），以下 **未做**：

| 功能 | 说明 |
|------|------|
| 暂停 / 恢复按钮 | 仅有 API `POST /pause`、`/resume` |
| 日期 / 猫筛选 | API 支持 query，UI 无 |
| waste 列 | spec Task 13 提过，Phase 1 DB 恒 `unknown` |
| `POST /cats` 注册 UI | API 未实现路由 |
| 参考图上传 / 管理页 | 仅命令行 `build_embeddings.py` |
| 健康状态面板 | `/health` 有数据，网页未展示 |
| UI 美化 | 当前极简 HTML 表 |

Phase 2（**不在本阶段**）：MQTT + ESP32 OLED、屎尿分类 `waste` 启发式。

---

## 7. 代码层已知缺口（可后续迭代）

| 项 | 严重性 | 说明 |
|----|--------|------|
| **letterbox / stream 坐标反变换** | 中 | spec §7.3；真机 ROI 与标定分辨率不一致时 overlap 可能偏；上 RTSP 前需按 **实际分辨率** 重标 ROI |
| **子码流降分辨率** | 中 | spec 建议 Win 用 720p 子码流减解码压力；代码未自动选子码流 |
| **git 提交 / 远程仓库** | 低 | 换机前建议 `git init` + push，或整目录打包 |
| **fixtures / data / models** | — | 在 `.gitignore` 里，换机 **手动复制** |

---

## 8. 换机必拷文件清单

```
stupid cat/                    # 整个项目源码
config.local.yaml              # RTSP、ROI、CUDA（若已有）
data/stupid_cat.db             # visit 记录
data/recordings/               # 录像（可选）
data/cats/*/refs/              # 参考图（若有）
data/correction_crops/         # 纠错用 crop（可选）
models/yolo11s.pt              # ~18MB
fixtures/*.mp4                 # 样例视频（可选）
.venv/                         # 可不拷，新机重建
```

---

## 9. Win `config.local.yaml` 模板（新机填写）

```yaml
inference:
  device: "cuda:0"

cameras:
  - id: cam1
    rtsp_url: "rtsp://user:pass@192.168.1.101:554/stream1"
    weight: 0.5
    roi_polygon: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]   # 按 VLC 画面标
  - id: cam2
    rtsp_url: "rtsp://user:pass@192.168.1.102:554/stream1"
    weight: 0.5
    roi_polygon: [[...]]

session:
  enter_overlap_sec: 2.0
  exit_no_cat_sec: 8.0
  min_visit_sec: 3.0

recorder:
  primary_camera: cam1
```

单摄像头时删除 `cam2` 整段。

Mac 调试用过的「全画面 ROI」**不要**直接用于生产。

---

## 10. 建议优先级（另一台电脑先做啥）

1. **同步代码 + `models/yolo11s.pt` + `config.local.yaml` 模板**  
2. **硬件到货** → VLC → ROI → `serve` 双 RTSP  
3. **拍 IR refs** → `build_embeddings.py` → 再看 visit 是否还全是 unknown  
4. **浏览器验收** Task 16 勾选  
5. （可选）补 UI：暂停、筛选、health  
6. （可选）letterbox / 子码流优化  

---

## 11. 相关文档索引

| 文档 | 路径 |
|------|------|
| 设计规格 | `docs/superpowers/specs/2026-06-02-stupid-cat-litter-vision-design.md` |
| MVP 计划 | `docs/superpowers/plans/2026-06-02-stupid-cat-mvp.md` |
| 验证清单 | `docs/validation-phase1.md` |
| README | `README.md` |
| Agent / Issue | `docs/agents/`、`AGENTS.md` |

---

**一句话：** 软件 Phase 1 **能跑**；换机后重点是 **Win + 双 PoE IR 相机 + refs 质心 + Task 16 真机验收**；UI 与 letterbox 为增强项，可后做。
