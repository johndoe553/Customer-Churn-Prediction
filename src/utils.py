"""
utils.py — Shared utility functions for logging, file I/O, and validation.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import joblib


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with console output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_directory(path: Path) -> None:
    """Create directory (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_pickle(obj: Any, path: Path) -> None:
    """Persist a Python object to disk with joblib."""
    ensure_directory(path.parent)
    joblib.dump(obj, path)


def load_pickle(path: Path) -> Any:
    """Load a joblib-persisted object."""
    if not path.exists():
        raise FileNotFoundError(f"Artefact not found: {path}")
    return joblib.load(path)


def save_json(data: Dict, path: Path) -> None:
    """Write a dictionary to a JSON file."""
    ensure_directory(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def load_json(path: Path) -> Dict:
    """Read a JSON file into a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_positive(value: float, name: str) -> None:
    """Raise ValueError if value is negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_range(value: float, low: float, high: float, name: str) -> None:
    """Raise ValueError if value is outside [low, high]."""
    if not (low <= value <= high):
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")
