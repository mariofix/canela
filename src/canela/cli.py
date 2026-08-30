from __future__ import annotations

import argparse
import asyncio
import logging

from .config import load_config
from .detector import MotionDetectorService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canela motion detection service")
    parser.add_argument(
        "--settings",
        action="append",
        help="Path to Dynaconf settings file. Can be passed multiple times.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    config = load_config(settings_files=args.settings)
    service = MotionDetectorService(config)
    asyncio.run(service.run())


if __name__ == "__main__":
    main()
