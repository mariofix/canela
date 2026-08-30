import argparse
import logging
import signal
import sys
import threading
import time

from . import notify
from .config import build_stream_configs, settings
from .detectors import base as detector_base
from .detectors import build_detectors
from .signals import event_saved


def parse_args():
    parser = argparse.ArgumentParser(
        prog="motion",
        description="OpenCV MOG2 motion + audio detector for RTSP access control",
    )
    parser.add_argument("--config", help="path to an additional settings file to merge in")
    parser.add_argument("--only", help="comma-separated stream names to run (default: all configured)")
    parser.add_argument("--rtsp-url", help="ad-hoc mode: run a single stream not defined in settings.toml")
    parser.add_argument("--name", default="adhoc", help="name for --rtsp-url ad-hoc stream (default: adhoc)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        settings.load_file(path=args.config)

    log_level = args.log_level or settings.get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
        stream=sys.stdout,
    )

    signal.signal(signal.SIGTERM, detector_base.handle_signal)
    signal.signal(signal.SIGINT, detector_base.handle_signal)

    # Wire up built-in receivers. Add more the same way, anywhere - this is
    # the only place detector code and notification code need to meet.
    event_saved.connect(notify.log_event_jsonl)

    stream_configs = build_stream_configs(args)
    detectors = build_detectors(stream_configs)

    threads = [threading.Thread(target=d.run, name=d.label, daemon=True) for d in detectors]

    logging.info(
        "detector started with %d worker(s): %s",
        len(threads),
        ", ".join(t.name for t in threads),
    )

    for t in threads:
        t.start()

    while detector_base.RUNNING and any(t.is_alive() for t in threads):
        time.sleep(1)

    for t in threads:
        t.join(timeout=5)

    logging.info("detector stopped")


if __name__ == "__main__":
    main()
