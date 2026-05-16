"""
Structured logging configuration for the Email Timing Response project.

Usage:
    from monitoring.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Training started", extra={"episode": 1})
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON for ingestion by log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        # Merge any extra fields passed via extra={}
        for key, val in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                payload[key] = val

        return json.dumps(payload, default=str)


def _build_logger(name: str, level: int, log_dir: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)

    # ── console handler (human-readable) ─────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
    )
    logger.addHandler(ch)

    # ── rotating file handler (JSON) ──────────────────────────────────────────
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "app.log"),
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JSONFormatter())
    logger.addHandler(fh)

    logger.propagate = False
    return logger


# Global defaults — override via env vars
_LOG_LEVEL = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
_LOG_DIR = os.environ.get("LOG_DIR", "logs")


def get_logger(name: str, level: int = None, log_dir: str = None) -> logging.Logger:
    """
    Return (or create) a project logger.

    Args:
        name:    Usually __name__ of the calling module.
        level:   Override default log level.
        log_dir: Directory where rotating log files are written.
    """
    return _build_logger(
        name=name,
        level=level or _LOG_LEVEL,
        log_dir=log_dir or _LOG_DIR,
    )
