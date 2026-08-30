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
        detection_frame: np.ndarray | None = None,
        motion_boxes: list[tuple[int, int, int, int]] | None = None,
        prev_frame: np.ndarray | None = None,
    ) -> Path:
        detected_at_utc = detected_at.astimezone(UTC)
        event_id = detected_at_utc.strftime("%Y%m%dT%H%M%S.%fZ")
        event_dir = self._root_dir / stream_name / event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        if not frames:
            raise ValueError("No frames were provided for evidence generation")

        # 0_prev.jpg — the frame immediately before motion was detected
        if prev_frame is not None:
            self._write_snapshot(event_dir / "0_prev.jpg", prev_frame)

        # 1_raw.jpg — the raw detection frame (high-res, no annotations)
        raw_frame = detection_frame if detection_frame is not None else frames[-1]
        self._write_snapshot(event_dir / "1_raw.jpg", raw_frame)

        # 2_box.jpg — the detection frame annotated with motion bounding boxes
        annotated = self._draw_boxes(raw_frame, motion_boxes or [])
        self._write_snapshot(event_dir / "2_box.jpg", annotated)

        # snapshot.jpg — alias pointing to the annotated image (used by alert pipeline)
        self._write_snapshot(event_dir / "snapshot.jpg", annotated)

        clip_path = event_dir / "clip.mp4"
        self._write_clip(clip_path, frames)

        motion_data = {
            "stream": stream_name,
            "detected_at": detected_at_utc.isoformat(),
            "snapshot": str(event_dir / "snapshot.jpg"),
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

    def _draw_boxes(
        self,
        frame: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        if not boxes:
            return frame.copy()
        cv2 = import_cv2()
        annotated = frame.copy()
        h, w = frame.shape[:2]
        thickness = max(2, int(2 * w / 640))
        for x, y, bw, bh in boxes:
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), thickness)
        return annotated

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
