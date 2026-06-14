# stupid-cat

Vision monitor for five cats sharing a large lidded litter box (who / when / duration).

**Spec:** `docs/superpowers/specs/2026-06-02-stupid-cat-litter-vision-design.md` (v0.3)

## Mac (development)

```bash
cd "/Users/hongpengwang/硬件项目/stupid cat"
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```

## Fixtures

Place a short IR sample clip at `fixtures/sample_ir.mp4` for pipeline smoke tests (see `fixtures/README.md`).

## Windows (production, GTX 1060)

1. Run `scripts/install_win.ps1` (creates venv, installs CUDA PyTorch + deps).
2. Copy `config.yaml` → `config.local.yaml` and set RTSP URLs.
3. Run the service via `python -m stupid_cat` (CLI added in later tasks).
