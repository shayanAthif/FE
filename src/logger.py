"""Structured logging and memory monitoring module."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
import psutil

from src.config import PROJECT_ROOT, get_config

_initialized_loggers = {}


def setup_logger(
    name: str = "hidden_risk",
    log_dir: Optional[Path | str] = None,
    log_file: Optional[str] = None,
    level: Optional[str] = None,
) -> logging.Logger:
    """Set up and return a structured logger with both console and rotating file output."""
    if name in _initialized_loggers:
        return _initialized_loggers[name]

    config = get_config()
    target_dir = Path(log_dir) if log_dir else PROJECT_ROOT / config.logging.log_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = log_file or config.logging.log_file
    log_path = target_dir / filename

    log_level = getattr(logging, (level or config.logging.level).upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    # Clear existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File Handler (5 MB max, 5 backups)
    fh = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(log_level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    _initialized_loggers[name] = logger
    return logger


def get_logger(name: str = "hidden_risk") -> logging.Logger:
    """Get or initialize the logger."""
    if name not in _initialized_loggers:
        return setup_logger(name)
    return _initialized_loggers[name]


def get_current_ram_gb() -> float:
    """Return the current resident set size (RSS) RAM in GB used by the current process."""
    process = psutil.Process()
    mem_bytes = process.memory_info().rss
    return round(mem_bytes / (1024 ** 3), 3)


def check_ram_headroom(logger: Optional[logging.Logger] = None) -> float:
    """Check current RAM usage against configured thresholds (10 GB warning, 12 GB max)."""
    current_gb = get_current_ram_gb()
    config = get_config()
    active_logger = logger or get_logger()

    if current_gb >= config.memory.max_ram_gb:
        active_logger.error(
            f"[Memory] CRITICAL: Current RAM usage {current_gb:.2f} GB exceeded hard limit of {config.memory.max_ram_gb} GB!"
        )
    elif current_gb >= config.memory.warning_ram_gb:
        active_logger.warning(
            f"[Memory] WARNING: Current RAM usage {current_gb:.2f} GB approaching warning threshold of {config.memory.warning_ram_gb} GB."
        )

    return current_gb

