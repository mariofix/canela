from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def print_alert(payload: dict[str, Any], prefix: str = "canela") -> None:
    logger.warning("%s motion alert: %s", prefix, json.dumps(payload, sort_keys=True))
