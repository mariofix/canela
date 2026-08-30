import logging
import os
from dataclasses import dataclass

import cv2

from ..config import StreamConfig
from .base import StreamDetector


@dataclass
class MotionResult:
    boxes: list
    scale_x: float
    prev_frame: object


class MotionDetector(StreamDetector):
    def __init__(self, cfg: StreamConfig):
        super().__init__(cfg, cooldown=cfg.cooldown)
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=cfg.history,
            varThreshold=cfg.var_threshold,
            detectShadows=False,
        )
        self.prev_frame = None
        self.frame_count = 0

    @property
    def label(self):
        return self.cfg.name

    def on_connect(self):
        # Reset warm-up on (re)connect so the subtractor re-settles.
        self.frame_count = 0
        logging.info("[%s] warming up background subtractor", self.label)

    def open(self):
        cap = cv2.VideoCapture(self.cfg.rtsp_url)

        if not cap.isOpened():
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def read(self, handle):
        ok, frame = handle.read()
        return frame if ok else None

    def close(self, handle):
        handle.release()

    def detect(self, frame):
        self.frame_count += 1

        # Get original dimensions for bounding box scaling
        high_h, high_w = frame.shape[:2]
        scale_x = high_w / self.cfg.detect_width
        scale_y = high_h / self.cfg.detect_height

        # Downscale a copy for detection to save CPU
        frame_detect = cv2.resize(
            frame,
            (self.cfg.detect_width, self.cfg.detect_height),
            interpolation=cv2.INTER_AREA,
        )

        mask = self.subtractor.apply(frame_detect)

        # Skip detection logic until the background model has stabilized
        if self.frame_count < self.cfg.warmup_frames:
            self.prev_frame = frame.copy()
            return None

        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < self.cfg.min_contour_area:
                continue

            # Get bounding box on the low-res detection frame, scale to high-res
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append(
                (
                    int(x * scale_x),
                    int(y * scale_y),
                    int(w * scale_x),
                    int(h * scale_y),
                )
            )

        result = MotionResult(boxes=boxes, scale_x=scale_x, prev_frame=self.prev_frame) if boxes else None

        # Store the clean, high-res frame for the next iteration
        self.prev_frame = frame.copy()
        return result

    def save_event(self, frame, result: MotionResult, timestamp):
        # 1. The high-res motion detected image, no modifications
        cv2.imwrite(os.path.join(self.cfg.save_dir, f"{timestamp}_1_raw.jpg"), frame)

        # 2. The high-res motion detected image with scaled squares
        frame_with_box = frame.copy()
        for hx, hy, hw, hh in result.boxes:
            cv2.rectangle(
                frame_with_box,
                (hx, hy),
                (hx + hw, hy + hh),
                (0, 255, 0),
                max(2, int(2 * result.scale_x)),
            )
        cv2.imwrite(os.path.join(self.cfg.save_dir, f"{timestamp}_2_box.jpg"), frame_with_box)

        # 3. The high-res previous image before the motion
        if result.prev_frame is not None:
            cv2.imwrite(os.path.join(self.cfg.save_dir, f"{timestamp}_0_prev.jpg"), result.prev_frame)
