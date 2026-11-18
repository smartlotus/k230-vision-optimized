"""Configuration loading and first-run setup."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .models import HubConfig, SourceConfig
from .utils import expand_path


def default_data_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return expand_path(Path(local_app_data) / "CodexConversationHub")
    return expand_path(Path.home() / ".codex-conversation-hub")


def default_config() -> HubConfig:
    user_profile = os.environ.get("USERPROFILE", str(Path.home()))
    app_data = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
    return HubConfig(
        data_dir=default_data_dir(),
        sources=[
            SourceConfig(
                id="codex-sessions",
                kind="codex_jsonl",
                path=str(Path(user_profile) / ".codex" / "sessions"),
            ),
            SourceConfig(
                id="vscode-chat-sessions",
                kind="vscode_chat_jsonl",
                path=str(Path(app_data) / "Code" / "User" / "workspaceStorage"),
                enabled=False,
            ),
        ],
    )


def load_config(path: Path | None = None) -> HubConfig:
    if path is None or not path.exists():
        return default_config()
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = [
        SourceConfig(
            id=str(item["id"]),
            kind=str(item.get("kind", "codex_jsonl")),
            path=str(item["path"]),
            enabled=bool(item.get("enabled", True)),
        )
        for item in payload.get("sources", [])
    ]
    if not sources:
        sources = default_config().sources
    return HubConfig(
        data_dir=expand_path(payload.get("data_dir", default_data_dir())),
        archive_originals=bool(payload.get("archive_originals", True)),
        max_file_bytes=int(payload.get("max_file_bytes", 250_000_000)),
        preview_chars=int(payload.get("preview_chars", 12_000)),
        sources=sources,
    )


def write_config(path: Path, config: HubConfig | None = None) -> Path:
    config = config or default_config()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
