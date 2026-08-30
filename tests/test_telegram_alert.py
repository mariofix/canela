from __future__ import annotations

import json
from pathlib import Path

from canela.telegram_alert import send_telegram_alert


def test_send_telegram_alert_sends_snapshot_and_caption(tmp_path: Path, monkeypatch) -> None:
    event_dir = tmp_path / "cam-1" / "20240101T000000.000000Z"
    event_dir.mkdir(parents=True)
    snapshot = event_dir / "snapshot.jpg"
    snapshot.write_bytes(b"fake-jpeg-data")
    payload = {
        "stream": "cam-1",
        "detected_at": "2024-01-01T00:00:00+00:00",
        "event_dir": str(event_dir),
        "motion_score": 0.1234,
    }

    seen: dict[str, object] = {}

    class FakeResponse:
        def __init__(self, body: bytes):
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["headers"] = dict(req.headers)
        body = req.data.decode("utf-8", errors="replace")
        seen["body"] = body
        return FakeResponse(b'{"ok": true}')

    monkeypatch.setattr("canela.telegram_alert.request.urlopen", fake_urlopen)

    send_telegram_alert(payload, chat_id="123456", token="abc123", timeout=10)

    assert seen["url"] == "https://api.telegram.org/botabc123/sendPhoto"
    assert seen["method"] == "POST"
    assert "chat_id" in str(seen["body"])
    assert "cam-1" in str(seen["body"])
    assert "Motion detected" in str(seen["body"])
    assert "snapshot.jpg" in str(seen["body"])
