"""CLI entry: python -m stupid_cat [run|serve]"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from stupid_cat.pipeline import build_pipeline


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--local-config",
        type=Path,
        default=Path("config.local.yaml"),
        help="Optional local override (ignored if missing)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Single-camera debug video (uses primary_camera id)",
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="serve: start API/static only (no RTSP or video ingest)",
    )


def cmd_run(args: argparse.Namespace) -> int:
    local = args.local_config if args.local_config.exists() else None
    if args.video and not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        print("Place a clip at fixtures/sample_ir.mp4 or pass --video PATH", file=sys.stderr)
        return 1

    pipeline, sources = build_pipeline(
        args.config,
        local_config_path=local,
        video_path=args.video,
    )
    pipeline.start_background(sources)

    print("Pipeline running. Ctrl+C to stop.")
    try:
        if pipeline._thread:
            pipeline._thread.join()
    except KeyboardInterrupt:
        pipeline.stop()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from stupid_cat.api.app import create_app

    local = args.local_config if args.local_config.exists() else None
    if args.video and not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 1
    if args.api_only and args.video:
        print("Use either --api-only or --video, not both.", file=sys.stderr)
        return 1

    pipeline, sources = build_pipeline(
        args.config,
        local_config_path=local,
        video_path=None if args.api_only else args.video,
    )
    if not args.api_only:
        if pipeline.cfg.service.pause_on_start:
            pipeline.pause()
        pipeline.start_background(sources)
    else:
        print("API-only mode: reading existing data/, no live ingest.")

    app = create_app(pipeline, pipeline.db)
    host = pipeline.cfg.service.host
    port = pipeline.cfg.service.port
    print(f"Serving http://{host}:{port}/ (API + static)")
    uvicorn.run(app, host=host, port=port, log_level="info")
    pipeline.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="stupid-cat litter vision monitor")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "serve"],
        help="run: pipeline only; serve: pipeline + API (default: run)",
    )
    _add_common_args(parser)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "serve":
        return cmd_serve(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
