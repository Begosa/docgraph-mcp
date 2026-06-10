from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Any

from .config import cfg_get

TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")
_LEVELS = {"off": logging.CRITICAL + 10, "error": logging.ERROR, "warning": logging.WARNING, "warn": logging.WARNING, "info": logging.INFO, "debug": logging.DEBUG, "trace": TRACE_LEVEL}


def level_from_name(value: str | int | None, default: int = logging.INFO) -> int:
    if isinstance(value, int):
        return value
    if value is None:
        return default
    return _LEVELS.get(str(value).strip().lower(), default)


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)), "level": record.levelname, "logger": record.name}
        try:
            payload = json.loads(record.getMessage())
            if isinstance(payload, dict):
                base.update(payload)
            else:
                base["message"] = payload
        except Exception:
            base["message"] = record.getMessage()
        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False, sort_keys=True)


def _resolve_log_path(path_value: str | None, root: Path) -> Path:
    path = Path(path_value or "docs/logs/docgraph-mcp.log")
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def configure_logging(config: dict[str, Any], root: str | Path, *, component: str = "backend") -> tuple[logging.Logger, dict[str, Any]]:
    root_path = Path(root).resolve()
    enabled = bool(cfg_get(config, "logging.enabled", True))
    level_name = os.environ.get("DOCGRAPH_LOG_LEVEL") or str(cfg_get(config, "logging.level", "info"))
    level = level_from_name(level_name)
    if not enabled:
        level = _LEVELS["off"]
    log_file = os.environ.get("DOCGRAPH_LOG_FILE") or cfg_get(config, "logging.file", "docs/logs/docgraph-mcp.log")
    max_bytes = int(cfg_get(config, "logging.max_bytes", 5_000_000))
    backup_count = int(cfg_get(config, "logging.backup_count", 3))
    to_stderr = bool(cfg_get(config, "logging.stderr", False))
    logger = logging.getLogger("docgraph_mcp")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()
    if not enabled or level > logging.CRITICAL:
        logger.addHandler(logging.NullHandler())
    else:
        path = _resolve_log_path(str(log_file), root_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        handler.setFormatter(JsonLineFormatter())
        handler.setLevel(level)
        logger.addHandler(handler)
        if to_stderr:
            stderr = logging.StreamHandler(sys.stderr)
            stderr.setFormatter(JsonLineFormatter())
            stderr.setLevel(level)
            logger.addHandler(stderr)
    settings = {"enabled": enabled, "configured_level": logging.getLevelName(level), "level_name": str(level_name).lower(), "file": str(_resolve_log_path(str(log_file), root_path)), "component": component, "include_payloads": bool(cfg_get(config, "logging.include_payloads", False)), "payload_preview_chars": int(cfg_get(config, "logging.payload_preview_chars", 500))}
    log_event(logger, logging.INFO, "logging.configured", **settings)
    return logger, settings


def sanitize_for_log(value: Any, *, include_payloads: bool, preview_chars: int, depth: int = 0) -> Any:
    if depth > 4:
        return "<max-depth>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if include_payloads:
            return value if len(value) <= preview_chars else value[:preview_chars] + f"...<truncated {len(value)} chars>"
        return {"type": "str", "chars": len(value), "preview": value[:80] + ("..." if len(value) > 80 else "")}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): sanitize_for_log(v, include_payloads=include_payloads, preview_chars=preview_chars, depth=depth + 1) for k, v in list(value.items())[:30]}
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        limit = len(seq) if include_payloads else min(len(seq), 30)
        out = [sanitize_for_log(v, include_payloads=include_payloads, preview_chars=preview_chars, depth=depth + 1) for v in seq[:limit]]
        if len(seq) > limit:
            out.append(f"<truncated {len(seq) - limit} items>")
        return out
    return repr(value)


def log_event(logger: logging.Logger, level: int | str, event: str, **fields: Any) -> None:
    numeric = level_from_name(level, logging.INFO) if isinstance(level, str) else level
    if not logger.isEnabledFor(numeric):
        return
    logger.log(numeric, json.dumps({"event": event, **fields}, ensure_ascii=False, sort_keys=True))
