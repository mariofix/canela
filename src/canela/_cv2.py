from __future__ import annotations

from typing import Any


def import_cv2() -> Any:
    """Import OpenCV lazily for runtime-only camera and video operations."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "OpenCV is required for motion detection evidence capture. Install opencv-python-headless."
        ) from exc
    return cv2
