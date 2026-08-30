from __future__ import annotations


def import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "OpenCV is required for motion detection evidence capture. Install opencv-python-headless."
        ) from exc
    return cv2
