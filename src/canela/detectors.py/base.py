import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime

from ..signals import (
    detection_event,
    detector_started,
    detector_stopped,
    event_saved,
    stream_connected,
    stream_disconnected,
)

# Process-lifecycle flag every detector's run() loop checks. Lives here (not
# in cli.py) because it must stay a live reference for every thread - a
# `from ... import RUNNING` elsewhere would copy the value at import time
# and never see handle_signal's mutation.
RUNNING = True


def handle_signal(signum, frame):
    global RUNNING
    RUNNING = False


class StreamDetector(ABC):
    """
    Template for a connect / reconnect / detect / save loop against an
    RTSP-backed source. Subclasses only implement the parts that differ
    (how to open the source, how to read a chunk, how to detect an event,
    how to save one). The connect/reconnect/cooldown scaffolding - and now
    the signal firing - lives here once, instead of being duplicated per
    detector.
    """

    def __init__(self, cfg, cooldown: float):
        self.cfg = cfg
        self.cooldown = cooldown
        self.last_save_time = 0.0

    @property
    @abstractmethod
    def label(self) -> str:
        """Log-line prefix, e.g. 'mudroom' or 'mudroom-audio'."""

    @abstractmethod
    def open(self):
        """Open the underlying source. Return a handle, or None on failure."""

    @abstractmethod
    def read(self, handle):
        """Read one chunk from handle. Return None/falsy on failure."""

    @abstractmethod
    def close(self, handle):
        """Release the handle."""

    @abstractmethod
    def detect(self, chunk):
        """Return a truthy detection result, or None if nothing detected."""

    @abstractmethod
    def save_event(self, chunk, result, timestamp: str):
        """Persist whatever the event needs (images, wav, etc.)."""

    def on_connect(self):
        """Optional hook, called once right after a successful open()."""

    def run(self):
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        handle = None

        logging.info("[%s] starting, source=%s", self.label, self.cfg.rtsp_url)
        detector_started.send(self)

        while RUNNING:
            if handle is None:
                handle = self.open()

                if handle is None:
                    logging.error("[%s] failed to open source", self.label)
                    time.sleep(5)
                    continue

                self.on_connect()
                logging.info("[%s] source connected", self.label)
                stream_connected.send(self)

            chunk = self.read(handle)

            if not chunk:
                logging.warning("[%s] read failed, reconnecting", self.label)
                stream_disconnected.send(self)
                self.close(handle)
                handle = None
                time.sleep(2)
                continue

            result = self.detect(chunk)

            if result is not None:
                detection_event.send(self, result=result)

                if time.time() - self.last_save_time > self.cooldown:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    logging.info("[%s] event detected, saving: %s", self.label, timestamp)
                    self.save_event(chunk, result, timestamp)
                    event_saved.send(self, result=result, timestamp=timestamp)
                    self.last_save_time = time.time()

        if handle is not None:
            self.close(handle)

        logging.info("[%s] stopped", self.label)
        detector_stopped.send(self)
