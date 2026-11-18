"""Platform helpers with no third-party dependencies."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def expand_path(value: str | Path) -> Path:
    """Expand Windows environment variables and return an absolute path."""

    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    return Path(expanded).expanduser().resolve()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iso_from_timestamp(timestamp: float | int | None) -> str | None:
    if timestamp is None:
        return None
    try:
        value = float(timestamp)
        if value > 100_000_000_000:
            value /= 1000
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def iso_from_stat(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def compact_path(path: str, max_length: int = 76) -> str:
    if len(path) <= max_length:
        return path
    head = max_length // 2 - 2
    return f"{path[:head]}…{path[-(max_length - head - 1):]}"


def clean_title(value: str | None, fallback: str = "未命名会话") -> str:
    text = " ".join((value or "").split()).strip()
    if not text:
        return fallback
    return text[:180]


def slug(value: str, fallback: str = "conversation") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return text[:80] or fallback


def copy_to_clipboard(value: str) -> bool:
    """Copy text using the Windows clipboard when available."""

    if os.name != "nt":
        return False
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "$input | Set-Clipboard"],
            input=value,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        return completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
