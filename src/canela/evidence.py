from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ._cv2 import import_cv2


class EvidenceWriter:
    def __init__(self, root_dir: Path, output_fps: float = 10.0):
        self._root_dir = root_dir
        self._output_fps = output_fps
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def write_event(
        self,
        *,
        stream_name: str,
        detected_at: datetime,
        metadata: dict[str, Any],
        frames: list[np.ndarray],
    ) -> Path:
        event_id = detected_at.strftime("%Y%m%dT%H%M%S.%fZ")
        event_dir = self._root_dir / stream_name / event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        if not frames:
            raise ValueError("No frames were provided for evidence generation")

        snapshot_path = event_dir / "snapshot.jpg"
        self._write_snapshot(snapshot_path, frames[-1])

        clip_path = event_dir / "clip.mp4"
        self._write_clip(clip_path, frames)

        motion_data = {
            "stream": stream_name,
            "detected_at": detected_at.astimezone(UTC).isoformat(),
            "snapshot": str(snapshot_path),
            "clip": str(clip_path),
            **metadata,
        }
        (event_dir / "motion.json").write_text(json.dumps(motion_data, indent=2), encoding="utf-8")
        return event_dir

    def _write_snapshot(self, snapshot_path: Path, frame: np.ndarray) -> None:
        cv2 = import_cv2()
        ok = cv2.imwrite(str(snapshot_path), frame)
        if not ok:
            raise RuntimeError(f"Failed to write snapshot at {snapshot_path}")

    def _write_clip(self, clip_path: Path, frames: list[np.ndarray]) -> None:
        cv2 = import_cv2()
        first = frames[0]
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._output_fps,
            (width, height),
        )
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
