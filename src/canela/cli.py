from __future__ import annotations

import argparse
import asyncio
import logging

from .config import AppConfig, StreamConfig, load_config
from .detector import MotionDetectorService

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canela motion detection service")
    parser.add_argument(
        "--settings",
        action="append",
        help="Path to Dynaconf settings file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Set verbosity level (default: INFO).",
    )
    return parser


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _describe_stream(stream: StreamConfig) -> str:
    feeds = []
    for resolution in stream.resolutions:
        label = resolution.name or "feed"
        source = resolution.source or stream.source or "unset-source"
        feeds.append(f"{label}={source}")
    return f"{stream.name} ({len(stream.resolutions)} feed(s): {', '.join(feeds)})"


def log_startup_summary(config: AppConfig, settings_files: list[str] | None, log_level: str) -> None:
    logger.info("👋 Canela is starting.")
    logger.info("Using settings: %s", ", ".join(settings_files or ["settings.toml", ".secrets.toml"]))
    logger.info("Log level: %s", log_level.upper())
    logger.info("Configured streams: %d", len(config.streams))
    for stream in config.streams:
        logger.info("  - %s", _describe_stream(stream))
    logger.info(
        "Warm-up: %d frames | Evidence window: %.1fs before / %.1fs after",
        config.motion.warmup_frames,
        config.evidence.pre_seconds,
        config.evidence.post_seconds,
    )
    logger.info("Ready. Waiting for motion events...")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    config = load_config(settings_files=args.settings)
    log_startup_summary(config, args.settings, args.log_level)
    service = MotionDetectorService(config)
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user. Exiting cleanly.")


if __name__ == "__main__":
    main()
