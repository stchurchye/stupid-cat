# stupid-cat 设计规格书

| 字段 | 值 |
|------|-----|
| 项目 | stupid-cat（五猫共用猫砂盆视觉监控） |
| 版本 | 0.3 |
| 日期 | 2026-06-02 |
| 状态 | 已通过（v0.2 review 1–6；v0.3 实现澄清已合入） |

---

## 1. 背景与目标

### 1.1 场景

- 家中 **5 只猫** 共用 **有盖、超大开放式猫砂盆**（非封闭自动猫砂盆）。
- 盖内 **光线较暗**；单摄像头 **无法拍全整盆**。
- 希望 **无项圈**，以 **视觉** 为主记录如厕行为。

### 1.2 产品目标

| 优先级 | 目标 | 阶段 |
|--------|------|------|
| P0 | 记录每次如厕的 **开始时间、结束时间、时长** | Phase 1 |
| P0 | 识别 **哪一只猫**（允许 `unknown` + 人工纠错） | Phase 1 |
| P1 | 网页时间线查看历史、播放片段、纠错 | Phase 1 |
| P2 | **屎 / 尿** 启发式分类（非医疗级） | Phase 2B |
| P2 | 通过 **MQTT** 将最近一次 visit 摘要推到 **ESP32 + OLED** | Phase 2A |

### 1.3 非目标

- 不上传云端；不做手机 App（一期）。
- 不绑定 Home Assistant（可共用局域网 MQTT broker）。
- 不用 RFID 项圈、不用「五类 YOLO 单模型认猫」（已否决方案 1）。
- 不承诺 100% 认猫/认屎尿；不做腹泻/便血等医疗诊断。

---

## 2. 已锁定技术决策

| 项 | 决策 |
|----|------|
| 算法 | **方案 2**：YOLO 只检 `cat` + Re-ID embedding 认五猫 |
| 推理机 | **Windows 10/11** + **GTX 1060 6GB** |
| 摄像头 | **2× PoE IPC**，**RTSP**，**NoIR 或可关 IR-cut**，关闭白光补光 |
| 补光 | **850nm IR 灯条**，12V 本地供电（不走 PoE） |
| 网络 | **全网线**：IPC → PoE 交换机 → 路由器；1060 有线同网段 |
| 数据 | **本机 SQLite + 文件目录**；源数据在 Win，不在 ESP32 |
| 开发 | **Mac 写代码** → **Git** → **Win1060 部署**；两台均可装 Cursor |
| ESP32 | **Phase 2A**；**OLED 显示** 最近 visit 摘要 |

---

## 3. 系统上下文

```mermaid
flowchart TB
  subgraph litter [猫砂盆盖内]
    IPC1[PoE_IPC_cam1]
    IPC2[PoE_IPC_cam2]
    IR[850nm_IR灯条_12V]
  end
  subgraph lan [局域网]
    SW[PoE交换机]
    R[路由器]
    PC[Windows_1060_stupid-cat]
    IPC1 --> SW
    IPC2 --> SW
    SW --> R
    PC --> R
  end
  subgraph phase2 [Phase2]
    MQTT[MQTT_Broker]
    ESP[ESP32_OLED]
    PC -->|visit_ended| MQTT
    MQTT --> ESP
  end
  User[用户浏览器] -->|localhost| PC
```

---

## 4. 逻辑架构

```mermaid
sequenceDiagram
  participant RTSP as Dual_RTSP
  participant Ing as Ingest
  participant Det as YOLO11s
  participant ReID as ReID_Embedder
  participant Sess as VisitFSM
  participant Rec as Recorder
  participant DB as SQLite
  participant API as FastAPI
  participant Pub as MQTTPublisher

  RTSP->>Ing: frame
  Ing->>Ing: CLAHE_optional_motion_gate
  Ing->>Det: frame
  Det-->>Ing: cat_bboxes
  Ing->>ReID: crops
  ReID-->>Sess: embedding_per_frame
  Sess->>Sess: ROI_overlap_multiframe_fusion
  Sess->>DB: visit_start_update_end
  Sess->>Rec: clip_on_visit
  Note over Pub: Phase2_only
  Sess->>Pub: visit_ended_json
```

> 上图为单路简化；生产环境为 **双路并行 ingest → 单一全局 VisitFSM**（见 §4.2、§8.5）。

### 4.1 模块职责

| 模块 | 职责 |
|------|------|
| `ingest` | 双路 RTSP 各一线程/协程；帧差唤醒；活跃 5–6 fps/路，空闲 1 fps；断流重连；事件带 `camera_id` + 单调 `ts` |
| `preprocess` | IR 域统一：可选 CLAHE（L 或灰度）、轻度 denoise；`input_mode` 全链路一致 |
| `detector` | Ultralytics **YOLO11s**，COCO `cat`，`conf≥0.20`；**每路独立**连续 N 帧确认后才产出该路合格检测 |
| `identifier` | EfficientNet-B0 或 ResNet18 → L2 归一化 embedding；与五猫质心比 **余弦相似度** |
| `multi_cam` | 同 visit 内跨摄像头 embedding 融合（默认 **`weighted_median`**，见 §8.3） |
| `session_fsm` | ROI 重叠 ≥ 阈值且累计 `enter_overlap_sec`；全路无猫 `exit_no_cat_sec`；`cooldown` 防抖；多框取最大 |
| `waste_classifier` | Phase 2B：时长 + 后期帧差（刨砂）启发式 → `pee`/`poop`/`unknown` |
| `recorder` | **visit_start** 起录 **主摄像头** 一路；单文件最长 `max_seconds`（默认 30）；`duration_sec` 仍按整段 visit |
| `api` | REST + 静态时间线；纠错、暂停、ROI（可选） |
| `publisher` | Phase 2A：visit 结束发 MQTT JSON |

### 4.2 双路与全局 FSM

- **一个** `VisitFSM` 实例；`ingest` 每处理完一路一帧，向 FSM 投递 `FrameEvent{camera_id, ts, detections[]}`。
- FSM **不**要求两路帧时间对齐；以各事件上的 `ts`（本机 `time.monotonic()` 或帧到达时刻）做累计与超时。
- **进入 / 活跃 / 退出** 规则见 §8.5；Re-ID 与融合在 visit 结束时一次完成。

---

## 5. 硬件与安装

### 5.1 机位

| 摄像头 | 建议位置 | 目的 |
|--------|----------|------|
| **cam1** | 盖内侧一角 **斜俯视** | 常蹲砂区、猫背/腰花纹 |
| **cam2** | 另一角 **斜侧视或对侧俯视** | 补盲区（埋头、背对 cam1） |

- 不追求单路拍全盆；ROI 只圈 **各镜头内可见活动砂区**。
- 认猫 **不依赖猫脸**；以背纹、侧纹、体型为主。

### 5.2 采购要点（预算约 ¥700–1000，不含电脑）

- 2× **PoE IPC**：1080p 广角、**RTSP/ONVIF**、NoIR 或可关 IR-cut、可关白光。
- 1× **5 口 802.3af PoE 交换机**（或 2× PoE 注入器）。
- 2× 超五/六类网线；1× 850nm IR 灯条 + 12V 电源；支架、除雾辅材。

### 5.3 验收（Phase 0）

- [ ] VLC 双路 RTSP 各稳定 60s+
- [ ] 盖内 IR 下 **人眼** 能分清五猫（若不能，降算法预期或调机位）
- [ ] 每猫 IR 参考图 ≥30 张入 `data/cats/{id}/refs/`

---

## 6. 软件部署

### 6.1 运行环境

| 环境 | 用途 | Python | PyTorch |
|------|------|--------|---------|
| **Mac** | 开发、单元逻辑、样例视频冒烟 | 3.11 | CPU / MPS |
| **Windows 1060** | 生产推理、CUDA、真 RTSP、7×24 | 3.11 venv | **CUDA cu118/cu121** |

- 依赖文件：`requirements-dev.txt`（Mac）、`requirements-win-cuda.txt`（Win）。
- 敏感配置：`config.local.yaml`（gitignore），含 RTSP 密码。

### 6.2 默认推理参数

| 参数 | 值 |
|------|-----|
| 检测模型 | `yolo11s.pt` |
| `imgsz` | 640 |
| 活跃帧率 | **5–6 fps / 路** |
| 空闲帧率 | 1 fps（帧差唤醒） |
| `min_crop_px` | 80（短边小于则跳过 Re-ID） |
| Re-ID 精度 | FP32 |
| 相似度阈值 | 默认 0.55（可配置，现场调） |
| 低于阈值 | `cat_id = unknown` |
| 融合缓冲上限 | 每 visit 最多 **64** 条 embedding 参与融合（§8.6） |
| YOLO 确认 | **每摄像头独立** `yolo_confirm_frames`（默认 3） |

**Win1060 性能提示：** 双路 1080p 解码优先走 **子码流 / 降低解码分辨率**（如 720p），再 letterbox 到 `imgsz`；避免 CPU 解码成为瓶颈。

### 6.3 与日常使用共存

- 后台服务 + 可选系统托盘；`http://127.0.0.1:8765`（端口可配置）看时间线。
- 建议：**从不休眠**；打游戏前 **暂停监控**（`POST /api/pause` 或托盘）。
- 空闲 GPU 显存约 1–1.5GB；活跃推理 <2GB。

### 6.4 目录结构

```text
stupid-cat/
  config.yaml              # 示例配置（无密码）
  config.local.yaml        # 本地覆盖（gitignore）
  requirements-dev.txt
  requirements-win-cuda.txt
  README.md
  data/
    stupid_cat.db
    recordings/{visit_id}.mp4   # 主摄像头单路（见 §9.4）
    cats/{cat_id}/
      refs/*.jpg
      centroid.npy
  models/
    yolo11s.pt
  src/stupid_cat/
    ingest.py
    preprocess.py
    detector.py
    reid.py
    session.py
    waste.py               # Phase 2B
    recorder.py
    publisher.py           # Phase 2A
    api/
    web/static/
  firmware/esp32_oled/     # Phase 2A
  scripts/
    install_win.ps1
    build_embeddings.py
  fixtures/
    sample_ir.mp4          # Mac 无摄像头冒烟（可选）
```

### 6.5 冷启动与五猫注册

| 情况 | 行为 |
|------|------|
| **首次启动** | 允许启动；`config.yaml` 中 `cats.seed` 预填 5 只（`id` + `name`），启动时写入 `cats` 表（若不存在） |
| **`refs/` 为空或尚无 `centroid.npy`** | 正常记 **visit + 时长**；`cat_id` 恒为 **`unknown`** |
| **参考图就绪后** | 运行 `scripts/build_embeddings.py`（或 API `POST /cats/{id}/rebuild-embedding`）生成/更新质心 |
| **纠错后** | 优质 crop 追加到 `refs/` 并 **重建该猫质心** |

不在启动时强制阻断服务；避免「没拍完照就无法跑」。

### 6.6 质心构建（`build_embeddings.py`）

对每只猫 `data/cats/{id}/refs/*.jpg`：

1. 与推理相同 `preprocess` + Re-ID backbone 得到 embedding，**L2 归一化**。
2. 若有效图片 **< `cats.min_refs`（默认 5）**：跳过写 `centroid.npy`，打 WARN。
3. 否则 `centroid = normalize(mean(embeddings))`，写入 `data/cats/{id}/centroid.npy`。

参考图须为 IR 域；脚本不对 RGB 客厅照做自动拒绝（依赖用户规范），可选记录过亮/过暗文件名。

---

## 7. 配置规格

### 7.1 `config.yaml` 结构（示意）

```yaml
service:
  host: "127.0.0.1"
  port: 8765
  pause_on_start: false

inference:
  device: "cuda:0"          # Win；Mac 用 cpu / mps
  yolo_model: "models/yolo11s.pt"
  yolo_conf: 0.20
  yolo_confirm_frames: 3
  imgsz: 640
  active_fps: 6
  idle_fps: 1
  motion_threshold: 25      # 帧差唤醒
  min_crop_px: 80
  reid_backbone: "efficientnet_b0"
  similarity_threshold: 0.55
  fusion: "weighted_median" # weighted_median | weighted_mean | best_frame
  fusion_max_frames: 64     # 单 visit 参与融合的最大 embedding 条数

cats:
  min_refs: 5               # build_embeddings 最少有效参考图
  seed:                     # 首次启动写入 DB（可改）
    - { id: mimi, name: "咪咪" }
    - { id: cat2, name: "猫2" }
    # ... 共 5 只

cameras:
  - id: cam1
    rtsp_url: "rtsp://user:pass@192.168.1.101:554/stream1"
    weight: 0.5
    stream_width: 1920      # 标 ROI 用的分辨率（与拉流主画面一致）
    stream_height: 1080
    roi_polygon: [[100,80],[500,80],[500,400],[100,400]]  # stream 像素坐标
  - id: cam2
    rtsp_url: "rtsp://user:pass@192.168.1.102:554/stream1"
    weight: 0.5
    stream_width: 1920
    stream_height: 1080
    roi_polygon: [[...]]

session:
  roi_overlap_min: 0.25     # bbox 与 ROI 交集面积 / bbox 面积
  enter_overlap_sec: 2.0      # 满足重叠条件的累计时长 → 进入 visit
  exit_no_cat_sec: 8.0        # 所有摄像头均无合格检测的连续时长 → 结束
  cooldown_sec: 3.0           # visit 结束后此时间内不新开 visit
  min_visit_sec: 3.0          # 短于此次丢弃

preprocess:
  input_mode: "bgr"         # bgr | gray3（灰度复制三通道）
  clahe_enabled: true
  denoise_enabled: false

recorder:
  enabled: true
  primary_camera: "cam1"    # 录像来源；须为 cameras[].id 之一
  max_seconds: 30           # 单文件写入上限；visit 时长可 >30s

mqtt:                       # Phase 2A
  enabled: false
  broker_host: "192.168.1.1"
  broker_port: 1883
  topic_prefix: "stupid-cat/visit"
  username: ""
  password: ""

waste:                      # Phase 2B
  enabled: false
  pee_max_duration_sec: 90
  poop_min_duration_sec: 120
  dig_motion_threshold: 40
```

### 7.2 IR 图像约定

- 参考图与推理帧 **必须为同一 IR 域**（禁止 RGB 客厅照混入 `refs/`）。
- `preprocess.input_mode` 全链路固定：`bgr` 或 `gray3`，训练 / 推理 / `refs/` 一致。

### 7.3 ROI 与检测坐标系

| 步骤 | 坐标空间 |
|------|----------|
| 用户标 ROI | 各摄像头 **`stream_width` × `stream_height`**（与 `cameras[].stream_*` 一致，通常 1080p 主码流） |
| YOLO 推理 | letterbox 到 `imgsz`（640） |
| 重叠计算 | 将 YOLO 输出的 bbox **反变换回 stream 坐标**，再与 `roi_polygon` 算 §8.2 的 `overlap_ratio` |

禁止在 640 空间标 ROI 却在 1080p 空间算交集（或反之）。

---

## 8. Visit 状态机

### 8.1 状态

| 状态 | 含义 |
|------|------|
| `idle` | ROI 内无稳定猫 |
| `active` | 正在如厕会话 |
| `cooldown` | 结束后短暂防抖 |

### 8.2 规则

1. **进入（`idle` → `active`）**：在 **`cooldown` 之外** 的 `idle` 状态下，按 §8.5 累计 `enter_accumulator`（任一路合格检测即计时）≥ `enter_overlap_sec` → `visit_started`，并开始录像（§9.4 主摄）。
2. **`active` 期间**：每帧对 **主框**（见 §8.4）做 Re-ID；embedding 存入 visit 缓冲区（带 `camera_id`、时间戳、`weight`）。
3. **结束（`active` → `cooldown`）**：**所有摄像头** 在连续 `exit_no_cat_sec` 内均无合格检测（无猫或重叠不足）→ `visit_ended` → 进入 `cooldown` 持续 `cooldown_sec`。
4. **收尾**：按 §8.3 融合 embedding → 与质心比对 → 写库；停止录像；Phase 2A 可选 MQTT。
5. **时长**：`duration_sec = ended_at - started_at`（整段 visit 墙钟时间，**不受** `recorder.max_seconds` 截断）。
6. **过短丢弃**：若 `duration_sec < min_visit_sec`：**不**插入 `visits`；若已生成 mp4 则 `unlink`；不发送 MQTT。
7. **暂停**（`POST /api/v1/pause`）：若在 `active`，**立即** `visit_ended`（按当前缓冲正常融合写库，除非触发规则 6 丢弃）；FSM → `idle`（**不**进入 `cooldown`）。`resume` 后从 `idle` 重新累计进入条件。

**重叠判定（实现一致）：**

```text
overlap_ratio = area(bbox ∩ ROI_polygon) / area(bbox)
合格检测 = YOLO cat 且 overlap_ratio >= roi_overlap_min
```

### 8.3 身份融合（单次 visit）

配置项 `inference.fusion`：

| 模式 | 算法 |
|------|------|
| **`weighted_median`（默认）** | 将每条 embedding 的每一维视为一个样本；按 `camera.weight` **重复取整权重次**（至少 1 次）后，对该维取 **中位数**；全部维完成后 L2 归一化 |
| `weighted_mean` | 对 embedding 做 `camera.weight` 加权算术平均后 L2 归一化 |
| `best_frame` | 在缓冲中取与**任意**已知质心相似度最高的单帧 embedding（须已有质心） |

```text
per_frame: 主框 crop → embed (L2 norm) → 入缓冲（附 weight）
visit_vector = fuse(buffer, mode=inference.fusion)
score[cat_i] = cosine(visit_vector, centroid_i)
confidence = max(score)  # 无质心时为 0
cat_id = argmax(score) if confidence >= similarity_threshold else "unknown"
frames_used = len(buffer)  # 参与融合的最终条数（截断后，§8.6）
```

无质心时跳过比对，`cat_id = unknown`，`confidence = 0`。

### 8.4 多猫同时在 ROI（MVP）

- **单个 visit**，不拆分为两只猫。
- 同一帧、同一路若有多只 `cat` 框：**仅保留面积最大的一只** 用于 Re-ID 与重叠判定；其余框忽略。
- 若常出现两猫同蹲，身份易错 → 标 `unknown` 或依赖纠错；不在 MVP 做双猫跟踪。

### 8.5 双路事件与进入/退出累计

**每路检测状态（per-camera）：**

- `last_qualified_at[cam]`：该路最近一次合格检测的时刻；无则视为「很久未见到」。
- `enter_accumulator`：全局墙钟累计——在 `idle` 且非 `cooldown` 期间，**任意一路**出现合格检测即 **继续累加** `Δt`（两路同时有猫不双倍）；**所有路**均无合格检测则 **清零**。
- 当 `enter_accumulator ≥ enter_overlap_sec` → `visit_started`。

**退出：**

- 当 **对所有** `cameras[]`：`now - last_qualified_at[cam] > exit_no_cat_sec` → `visit_ended` → `cooldown` 持续 `cooldown_sec`。

**`active` 期间：** 任一路合格检测刷新该路 `last_qualified_at`；Re-ID 仅对该路当帧主框执行（见 §8.4）。

**运动门控：** `idle` 且帧差低于阈值时 ingest 可为 1 fps；`enter_accumulator` 仍按真实 `Δt` 累计（可能使进入略慢——若漏记短如厕，提高 `idle_fps` 或降低 `motion_threshold`）。

### 8.6 Visit 内 embedding 缓冲

- 每条：合格检测且 Re-ID 成功（crop 短边 ≥ `min_crop_px`）的 embedding + `camera_id` + `weight` + `ts` + `bbox_area`。
- 超过 `inference.fusion_max_frames`（默认 64）时：丢弃 **bbox 面积最小** 的旧条（同等面积丢最旧）。
- visit 结束只对缓冲做一次 §8.3 融合。

---

## 9. 数据模型（SQLite）

### 9.1 表 `cats`

| 列 | 类型 | 说明 |
|----|------|------|
| id | TEXT PK | 如 `mimi` |
| name | TEXT | 显示名 |
| created_at | TEXT ISO8601 | |

### 9.2 表 `visits`

| 列 | 类型 | 说明 |
|----|------|------|
| id | TEXT PK | UUID |
| cat_id | TEXT | 可 `unknown` |
| started_at | TEXT | |
| ended_at | TEXT | |
| duration_sec | INTEGER | |
| confidence | REAL | §8.3：`max(cosine)`；无质心为 0 |
| waste_type | TEXT | `pee` / `poop` / `unknown` |
| waste_confidence | REAL | Phase 2B |
| frames_used | INTEGER | §8.3 / §8.6：参与融合的 embedding 条数 |
| camera_ids | TEXT | JSON 数组 |
| recording_path | TEXT | 可空 |
| corrected | INTEGER | 0/1 是否人工改过 |
| created_at | TEXT | |

### 9.3 表 `corrections`

| 列 | 类型 | 说明 |
|----|------|------|
| id | INTEGER PK | |
| visit_id | TEXT FK | |
| old_cat_id | TEXT | |
| new_cat_id | TEXT | |
| created_at | TEXT | |

纠错时：更新 `visits.cat_id`，`corrected=1`，写 `corrections` 行。

- 若 `new_cat_id` 为 **已知猫**（存在于 `cats` 且 ≠ `unknown`）：将 visit 缓冲中 **得分最高的一帧** crop 追加到 `data/cats/{new_cat_id}/refs/`，并 **重建该猫质心**。
- 若改为 `unknown`：**不**追加 refs、不重建质心。

### 9.4 录像（`recorder`）

| 项 | 规则 |
|----|------|
| 来源 | **仅** `recorder.primary_camera`（默认 `cam1`）的预处理后画面 |
| 开始 | `visit_started` 时开始写 `data/recordings/{visit_id}.mp4` |
| 停止 | `visit_ended` 时关闭文件 |
| 长度上限 | 写入 **最多 `max_seconds`（默认 30）** 视频内容；超时后 **停止写盘** 但 visit 仍可为 `active` |
| 与时长字段 | `visits.duration_sec` = 墙钟 `ended - started`，**可大于 30** |
| 播放 URL | `GET /recordings/{visit_id}.mp4`（静态挂载，与 API 同端口） |
| 用途 | 纠错回放、训练样本；非完整监控录像 |

---

## 10. API 规格（Phase 1）

基址：`http://127.0.0.1:8765/api/v1`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/cats` | 列表 |
| POST | `/cats` | 注册猫 `{id, name}` |
| GET | `/visits?from=&to=&cat_id=` | 时间线 |
| GET | `/visits/{id}` | 详情；`recording_url` 为 `/recordings/{id}.mp4`（若存在） |
| POST | `/visits/{id}/correct` | `{cat_id}` 纠错（§9.3） |
| POST | `/cats/{id}/rebuild-embedding` | 从 `refs/` 重建质心 |
| POST | `/pause` | 暂停推理（§8.2 规则 7） |
| POST | `/resume` | 恢复 |
| GET | `/health` | 见下表 |

**`GET /health` 响应（JSON）：**

| 字段 | 说明 |
|------|------|
| `status` | `ok` \| `degraded`（任一路断流 >30s） |
| `paused` | bool |
| `device` | 如 `cuda:0` |
| `cuda_available` | bool |
| `cameras` | `[{id, connected, last_frame_at, fps_actual}]` |
| `active_visit_id` | 可 null |
| `db_ok` | bool |

静态页：`GET /` → 时间线 UI（visit 列表、播放 mp4、纠错下拉）。

**Phase 2B（待增）：** `PATCH /visits/{id}` body `{waste_type}` — 见 §12。

---

## 11. MQTT 载荷（Phase 2A）

Topic：**`{topic_prefix}/latest`**（固定；ESP32 只订阅一条；retain=true 推荐）

```json
{
  "event": "visit_ended",
  "visit_id": "550e8400-e29b-41d4-a716-446655440000",
  "cat_id": "mimi",
  "cat_name": "咪咪",
  "started_at": "2026-06-02T14:30:00+08:00",
  "ended_at": "2026-06-02T14:32:10+08:00",
  "duration_sec": 130,
  "waste_type": "unknown",
  "confidence": 0.82,
  "camera_ids": ["cam1", "cam2"]
}
```

ESP32：订阅后刷新 OLED（128×64 SSD1306 类，Phase 2 定板）；RAM 保留最近 1～5 条轮播。

---

## 12. 屎尿分类（Phase 2B）

| 规则 | 条件 | 输出 |
|------|------|------|
| 短停留 | `duration_sec < pee_max_duration_sec` 且后期帧差低 | `pee`（低置信） |
| 长停留 + 刨砂 | `duration_sec ≥ poop_min_duration_sec` 且后期帧差高 | `poop` |
| 其余 | — | `unknown` |

- Phase 1 字段恒为 `unknown`。
- 目标准确率 **≥70%**（居家记录）；错误可人工改 `waste_type`（Phase 2B：`PATCH /visits/{id}`  body `{waste_type}` 或扩展 correct API）。

---

## 13. 验收标准

### 13.1 Phase 1 MVP

| 项 | 标准 |
|----|------|
| 时长 | 起止与人工观察误差 **≤30s**（猫曾出现在 ROI 内） |
| 身份 | 自动识别 **≥75%** visit 正确：连续 **7 天**、至少 **50** 次有效 visit 抽样；仅统计猫曾进入 ROI 的 visit；**纠错前**计数；可附五猫混淆矩阵 |
| 召回 | 双机位覆盖区内 **≥90%** 如厕产生 visit |
| 可用性 | Win 后台 24h 运行；断流自动重连；网页可纠错 |
| 隐私 | 录像与 DB 仅本机 |

### 13.2 Phase 2A

- visit 结束后 5s 内 OLED 显示本次摘要。
- ESP32 断线不影响 Win 记录。

### 13.3 Phase 2B

- 屎尿启发式在标注测试集上 **≥70%**（用户自选 30+ 样本）。

---

## 14. 分阶段交付

| 阶段 | 交付物 | 负责人 |
|------|--------|--------|
| **0** | 双 RTSP 通、IR 参考图、ROI 坐标、Win CUDA 冒烟 | 用户 + Agent 文档 |
| **1a** | ingest/detector/reid/session/recorder 可跑 | Agent |
| **1b** | SQLite、API、网页、纠错 | Agent |
| **1c** | 现场调阈值、权重（2–4 周） | 用户反馈 + Agent |
| **2A** | MQTT + ESP32 OLED 固件 | Agent + 用户接线 |
| **2B** | waste 启发式 + 参考图自动入库脚本 | Agent |

---

## 15. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 五猫太像 | 多参考图、unknown、纠错入库、加 cam2 |
| 双猫同蹲 | MVP 只跟踪最大框；易 unknown / 纠错 |
| IR 过曝/起雾 | 调 IR 角度、除雾、清洁镜头 |
| RTSP 断流 | 重连退避；日志告警 |
| 盲区漏记 | 调整 ROI/机位 |
| RGB/IR 混训 | 规范 `refs/` 仅 IR |
| 游戏抢 GPU | 暂停服务 |
| 双路 1080p 解码吃满 CPU | 子码流 / 降解码分辨率（§6.2） |
| 长 visit 内存涨 | `fusion_max_frames` 截断（§8.6） |
| 身份 75% 偏乐观 | 1c 用混淆矩阵调阈值；必要时以「纠错后」作辅助指标 |

---

## 16. 开源参考

| 项目 | 用途 |
|------|------|
| [hsc-reident](https://github.com/TobiasTrein/hsc-reident) | Re-ID / 阈值 |
| [Yolo2Mqtt](https://github.com/cobryan05/Yolo2Mqtt) | ROI 重叠 + 时长 FSM |
| [katzenschreck-fcats](https://github.com/andremotz/katzenschreck-fcats) | RTSP + MQTT 结构 |
| [Estefannie IR 猫厕](https://www.estefannie.com/blog/f63kkhwgq56u76hjsysy1moy5ttuuf) | IR + 屎尿时长启发式 |
| [PMC 多猫猫砂论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10648833/) | 嵌入 + 注册 + 漂移再训 |

---

## 17. 审阅记录

| 日期 | 审阅人 | 结论 | 备注 |
|------|--------|------|------|
| 2026-06-02 | Agent review | 修订 1–6 | → v0.2 |
| 2026-06-02 | 用户 | **通过** | 「spec 通过，按 review 改 1–6」 |
| 2026-06-02 | Agent spec review | **修订 → v0.3** | 双路 FSM、ROI 坐标、录像主摄、融合/缓冲、pause/health、质心构建 |

---

## 附录 A：用户分工

**用户必须：** 安装 PoE/IPC/IR；拍摄 IR 参考图；填写 RTSP；偶尔网页纠错；1060 常开。

**Agent 实现：** 除上述外全部软件（Mac 开发 → Win 部署）。

## 附录 B：Mac / Win 协作

1. Mac：实现 + 样例视频冒烟。  
2. Git 同步。  
3. Win1060：CUDA、`config.local.yaml`、双路真流、任务计划程序自启。  
4. 调参阶段优先在 **Win Cursor** 会话进行。
