from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def record_payload(payload: dict[str, Any], output_path: str) -> None:
    Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
