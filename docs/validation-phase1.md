# Phase 1 集成验证清单（Task 16）

- [x] Mac: `pytest tests/ -v` 全绿（GPU 测试可 skip）— 2026-06-02，58 passed / 1 skipped
- [x] Mac: `python -m stupid_cat run --video fixtures/clip-01.mp4` 产生 ≥1 visit — 2026-06-02，`config.local.yaml` 全画面 ROI；visit `1e949695…` duration 9s，`cat_id=unknown`（无 centroid），录像 6.1MB
- [ ] Win1060: CUDA + 双 RTSP + `GET /api/v1/health`
- [ ] 网页 `http://127.0.0.1:8765/` 时间线与纠错（本机浏览器自测）
- [ ] 对照 spec §13.1 记录身份/时长/召回抽样结果

**样例视频：** `fixtures/clip-01.mp4` … `clip-04.mp4`（来自 Downloads 四段 MP4）
