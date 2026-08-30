from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ._cv2 import import_cv2
from .alerts import AlertPipeline
from .config import AppConfig, Resolution, StreamConfig
from .evidence import EvidenceWriter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FrameSample:
    timestamp: datetime
    frame: np.ndarray


class MotionDetectorService:
    def __init__(self, config: AppConfig):
        self._config = config

    async def run(self) -> None:
        if not self._config.streams:
            raise ValueError("No streams configured")
        pipeline = AlertPipeline(self._config.alerts)
        tasks = [
            asyncio.create_task(self._run_stream(stream, pipeline), name=f"stream:{stream.name}")
            for stream in self._config.streams
        ]
        await asyncio.gather(*tasks)

    async def _run_stream(self, stream: StreamConfig, pipeline: AlertPipeline) -> None:
        cv2 = import_cv2()
        capture = cv2.VideoCapture(_parse_source(stream.source))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open stream: {stream.name} ({stream.source})")

        frame_interval = 1.0 / max(stream.fps, 0.1)
        max_pre_frames = max(1, int(stream.fps * self._config.evidence.pre_seconds))
        pre_buffer: deque[FrameSample] = deque(maxlen=max_pre_frames)

        evidence_writer = EvidenceWriter(
            root_dir=self._resolve_root_dir(),
            output_fps=self._config.evidence.output_fps,
        )

        last_detection_at: datetime | None = None
        previous_frames: dict[tuple[int, int], np.ndarray] = {}

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    await asyncio.sleep(frame_interval)
                    continue

                now = datetime.now(UTC)
                pre_buffer.append(FrameSample(timestamp=now, frame=frame.copy()))

                triggered_resolution, score = _detect_motion(
                    frame,
                    stream.resolutions,
                    previous_frames,
                    self._config.motion.delta_threshold,
                    self._config.motion.motion_ratio_threshold,
                    cv2,
                )
                if not triggered_resolution:
                    await asyncio.sleep(frame_interval)
                    continue

                if last_detection_at and now - last_detection_at < timedelta(seconds=self._config.motion.cooldown_seconds):
                    await asyncio.sleep(frame_interval)
                    continue

                last_detection_at = now
                event_frames = [sample.frame for sample in pre_buffer]
                post_frames = await self._collect_post_frames(capture, frame_interval, stream.fps)
                event_frames.extend(post_frames)

                event_dir = evidence_writer.write_event(
                    stream_name=stream.name,
                    detected_at=now,
                    metadata={
                        "trigger_resolution": {
                            "width": triggered_resolution.width,
                            "height": triggered_resolution.height,
                        },
                        "motion_score": score,
                        "pre_seconds": self._config.evidence.pre_seconds,
                        "post_seconds": self._config.evidence.post_seconds,
                    },
                    frames=event_frames,
                )
                payload: dict[str, Any] = {
                    "stream": stream.name,
                    "detected_at": now.isoformat(),
                    "event_dir": str(event_dir),
                    "motion_score": score,
                }
                await pipeline.run(payload)
                await asyncio.sleep(frame_interval)
        finally:
            capture.release()

    async def _collect_post_frames(self, capture: Any, frame_interval: float, fps: float) -> list[np.ndarray]:
        post_frame_count = max(0, int(fps * self._config.evidence.post_seconds))
        post_frames: list[np.ndarray] = []
        for _ in range(post_frame_count):
            ok, frame = capture.read()
            if ok:
                post_frames.append(frame.copy())
            await asyncio.sleep(frame_interval)
        return post_frames

    def _resolve_root_dir(self) -> Path:
        from .config import resolve_root

        return resolve_root(self._config.evidence.root_dir)


def _detect_motion(
    frame: np.ndarray,
    resolutions: list[Resolution],
    previous_frames: dict[tuple[int, int], np.ndarray],
    delta_threshold: float,
    motion_ratio_threshold: float,
    cv2: Any,
) -> tuple[Resolution | None, float]:
    for resolution in resolutions:
        resized = cv2.resize(frame, (resolution.width, resolution.height))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        key = (resolution.width, resolution.height)

        prev_gray = previous_frames.get(key)
        previous_frames[key] = gray
        if prev_gray is None:
            continue

        diff = cv2.absdiff(prev_gray, gray)
        changed = (diff >= delta_threshold).sum()
        ratio = float(changed / diff.size)
        if ratio >= motion_ratio_threshold:
            return resolution, ratio
    return None, 0.0


def _parse_source(source: str) -> str | int:
    return int(source) if source.isdigit() else source
