from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any
from urllib import request

logger = logging.getLogger(__name__)


def send_telegram_alert(
    payload: dict[str, Any],
    *,
    chat_id: str | None = None,
    token: str | None = None,
    photo_path: str | None = None,
    timeout: int = 30,
) -> None:
    """Send the motion snapshot and detection metadata to a Telegram chat."""
    resolved_chat_id = _resolve_chat_id(chat_id)
    resolved_token = _resolve_token(token)
    if not resolved_chat_id or not resolved_token:
        raise ValueError("Telegram alert requires both TELEGRAM_CHAT_ID and TELEGRAM_BOT_TOKEN")

    snapshot = _resolve_image_path(payload, photo_path)
    caption = _build_caption(payload)

    logger.info("Sending Telegram alert for '%s' to chat %s", payload.get("stream", "unknown"), resolved_chat_id)
    _send_photo(
        token=resolved_token,
        chat_id=resolved_chat_id,
        photo_path=snapshot,
        caption=caption,
        timeout=timeout,
    )


def main() -> None:
    raw_payload = os.environ.get("CANELA_ALERT_PAYLOAD")
    if not raw_payload:
        raise ValueError("CANELA_ALERT_PAYLOAD is required")

    payload = json.loads(raw_payload)
    args_json = os.environ.get("CANELA_ALERT_ARGS") or "{}"
    args = json.loads(args_json)
    send_telegram_alert(payload, **args)


def _resolve_chat_id(chat_id: str | None) -> str:
    return str(chat_id or os.environ.get("TELEGRAM_CHAT_ID") or "").strip()


def _resolve_token(token: str | None) -> str:
    return str(token or os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()


def _resolve_image_path(payload: dict[str, Any], photo_path: str | None) -> Path:
    if photo_path:
        candidate = Path(photo_path)
        if candidate.exists():
            return candidate

    event_dir = payload.get("event_dir")
    if isinstance(event_dir, str):
        candidate = Path(event_dir) / "snapshot.jpg"
        if candidate.exists():
            return candidate

    motion_path = payload.get("snapshot")
    if isinstance(motion_path, str):
        candidate = Path(motion_path)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Telegram alert requires a snapshot image; set photo_path or ensure the event_dir contains snapshot.jpg"
    )


def _build_caption(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "🚨 Motion detected",
        f"Stream: {payload.get('stream', 'unknown')}",
        f"Detected at: {payload.get('detected_at', 'unknown')}",
        f"Motion score: {payload.get('motion_score', 'n/a')}",
    ]

    event_dir = payload.get("event_dir")
    if event_dir:
        lines.append(f"Evidence: {event_dir}")

    metadata = _load_event_metadata(payload)
    if isinstance(metadata, dict):
        trigger_source = metadata.get("trigger_source")
        if trigger_source:
            lines.append(f"Source: {trigger_source}")
        trigger_resolution = metadata.get("trigger_resolution")
        if isinstance(trigger_resolution, dict):
            width = trigger_resolution.get("width")
            height = trigger_resolution.get("height")
            if width is not None and height is not None:
                lines.append(f"Resolution: {width}x{height}")

    return "\n".join(lines)


def _load_event_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    event_dir = payload.get("event_dir")
    if not isinstance(event_dir, str):
        return {}

    metadata_path = Path(event_dir) / "motion.json"
    if not metadata_path.exists():
        return {}

    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def _send_photo(*, token: str, chat_id: str, photo_path: Path, caption: str, timeout: int) -> None:
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with photo_path.open("rb") as image_file:
        file_data = image_file.read()

    body, boundary = _encode_multipart(
        fields={
            "chat_id": str(chat_id),
            "caption": caption,
        },
        file_field_name="photo",
        file_name=photo_path.name,
        file_data=file_data,
        content_type="image/jpeg",
    )

    request_obj = request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with request.urlopen(request_obj, timeout=timeout) as response:
        response_data = response.read()

    if not response_data:
        raise RuntimeError("Telegram API returned an empty response")

    parsed = json.loads(response_data.decode("utf-8"))
    if not parsed.get("ok", False):
        raise RuntimeError(f"Telegram API error: {parsed}")


def _encode_multipart(
    *,
    fields: dict[str, str],
    file_field_name: str,
    file_name: str,
    file_data: bytes,
    content_type: str,
) -> tuple[bytes, str]:
    boundary = f"----canela-telegram-{uuid.uuid4().hex}"
    parts: list[bytes] = []

    for key, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        parts.append(f"{value}\r\n".encode("utf-8"))

    parts.append(f"--{boundary}\r\n".encode("utf-8"))
    parts.append(
        (
            f'Content-Disposition: form-data; name="{file_field_name}"; filename="{file_name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
    )
    parts.append(file_data)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), boundary


if __name__ == "__main__":
    main()
