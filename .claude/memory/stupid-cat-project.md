---
name: stupid-cat-project
description: "stupid-cat repo identity, local clone, and the hardening work (PR"
metadata: 
  node_type: memory
  type: project
  originSessionId: 41f22c11-62f1-444b-9bb5-b950376a79ba
---

**stupid-cat** = vision monitor for 5 cats sharing one large lidded litter box (who/when/duration), 2 PoE IR cameras (RTSP), inference on Windows + GTX 1060 (CUDA), Mac dev-only. Stack: FastAPI+SQLite+OpenCV+ultralytics YOLO11s + EfficientNet-B0 Re-ID. Phase-1 MVP; hardware not yet purchased; user runs/pulls it on a separate Windows machine.

- GitHub owner is **stchurchye** (not `churchye`): https://github.com/stchurchye/stupid-cat
- Local clone: `/Users/church/claude/stupidcat/stupid-cat` (the clone is the git repo; parent `stupidcat/` is not).
- Test venv: `.venv` (python3.12) with FULL deps incl. torch 2.12 / torchvision 0.27 / ultralytics 8.4 (CPU). Full suite = **81 passed, 1 skipped** (the skipped one needs yolo weights). Production is PINNED lower (torch 2.4.1+cu121) for Pascal — embed path verified compatible with both.

On 2026-06-15, did an autonomous multi-track hardening pass → **PR #1 (MERGED to main)**: 24/7 stability/thread-safety, Pascal/CUDA deploy+perf, Re-ID accuracy code bugs, correctness; validated by adversarial-review workflows.

Then the **Windows machine** added a big update (now on main): per-camera dual-cam recordings, "unified correction learning" (per-camera correction crops), live preview + stats web pages (`live.html/js`, `stats.html/js`), `timeutil.py`, and many Windows scripts (`setup_hikvision.py`, `prune_recordings`, `install_scheduled_tasks.ps1`, `start.ps1`). A 5th /code-review of that update found 4 HIGH + 8 MEDIUM (all confirmed) → fixed in branch **`fix/dualcam-review-fixes`** → **PR [#2](https://github.com/stchurchye/stupid-cat/pull/2)**: recording-race re-check, single correction crop (no double-count), orphan recovery of secondary clips, offset-aware time filter, re-correction ref-move, preview ROI-scale+staleness, recording_path fallback, /recordings always-mount + /visits limit, hikvision URL-decode+re.sub-escape. Tests 111 passed.

Key facts for next time:
- Locking model: `Pipeline._lock` (RLock) guards FSM/recorder/buffer/_last_frame_at; centroids are copy-on-write; slow work (embed in correct_visit/rebuild, ffmpeg reencode) runs OFF the lock. Don't reintroduce slow work under the lock.
- torch is pinned for sm_61 in requirements-win-cuda.txt — do NOT bump past <2.6 / cu121 without checking Pascal kernel support.
- Re-ID ref quality gate keys on dynamic range (not brightness) so dark IR cats pass.

NOT done (needs user materials or is deferred), see PR "next steps":
- **Cat-ID accuracy model upgrade** (fine-tune / metric learning / per-cat thresholds) — BLOCKED on the IR reference images the user will provide. This is the #1 risk to the ≥75% identity goal. [[stupid-cat-review-findings]]
- TensorRT export, multi-vector gallery, GPU auto-pause while gaming, UI enhancements (pause/filter/health/upload pages), Phase 2 (MQTT+ESP32, waste classification).
