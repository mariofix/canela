from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
from collections.abc import Awaitable
from typing import Any

from .config import AlertStep

logger = logging.getLogger(__name__)


class AlertPipeline:
    def __init__(self, steps: list[AlertStep]):
        self._steps = steps

    async def run(self, payload: dict[str, Any]) -> None:
        pending: list[asyncio.Task[None]] = []
        for step in self._steps:
            if step.async_step:
                pending.append(asyncio.create_task(self._run_step(step, payload)))
                continue
            await self._run_step(step, payload)
        if pending:
            await asyncio.gather(*pending)

    async def _run_step(self, step: AlertStep, payload: dict[str, Any]) -> None:
        if "::" in step.run:
            await _run_python_callable(step.run, payload, step.args)
            return
        await _run_shell_command(step.run, payload, step.args)


async def _run_python_callable(spec: str, payload: dict[str, Any], args: dict[str, Any]) -> None:
    module_name, callable_name = spec.split("::", 1)
    module = importlib.import_module(module_name)
    target = getattr(module, callable_name)
    result = target(payload, **args)
    if isinstance(result, Awaitable):
        await result


async def _run_shell_command(command: str, payload: dict[str, Any], args: dict[str, Any]) -> None:
    env = {
        "CANELA_ALERT_PAYLOAD": json.dumps(payload),
        "CANELA_ALERT_ARGS": json.dumps(args),
    }
    process = await asyncio.create_subprocess_shell(
        command,
        env={**os.environ, **env},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        error_output = stderr.decode().strip()
        raise RuntimeError(f"alert command failed ({process.returncode}): {error_output}")
    if stdout:
        logger.info("alert command output: %s", stdout.decode().strip())
