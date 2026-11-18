"""Incremental scanners for Codex rollout JSONL and VS Code chat JSONL."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .models import HubConfig, SourceConfig
from .storage import Repository
from .utils import clean_title, expand_path, iso_from_stat, iso_from_timestamp, sha256_file


@dataclass
class ParsedConversation:
    conversation: dict[str, Any]
    messages: list[dict[str, Any]]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(part for part in parts if part).strip()


def _append_message(
    messages: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    role: str,
    text: str,
    event_type: str,
    raw: dict[str, Any],
    created_at: str | None = None,
) -> None:
    text = text.strip()
    if not text or role not in {"user", "assistant"}:
        return
    if role == "user" and _is_context_payload(text):
        return
    key = (role, text)
    if key in seen:
        return
    seen.add(key)
    messages.append(
        {
            "ordinal": len(messages),
            "role": role,
            "text": text,
            "event_type": event_type,
            "created_at": created_at,
            "raw": raw,
        }
    )


def _is_context_payload(text: str) -> bool:
    return (
        text.startswith("# AGENTS.md instructions")
        or text.startswith("<environment_context>")
        or text.startswith("<INSTRUCTIONS>")
    )


def parse_codex_jsonl(path: Path, source: SourceConfig, preview_chars: int) -> ParsedConversation:
    metadata: dict[str, Any] = {}
    messages: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    model: str | None = None
    reasoning_effort: str | None = None
    first_user: str | None = None
    session_id: str | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            item_type = item.get("type")
            payload = item.get("payload") or {}
            if item_type == "session_meta":
                metadata = payload if isinstance(payload, dict) else {}
                session_id = metadata.get("id") or metadata.get("session_id")
            elif item_type == "event_msg" and isinstance(payload, dict):
                event_type = str(payload.get("type", ""))
                if event_type == "user_message":
                    text = str(payload.get("message", ""))
                    _append_message(messages, seen, "user", text, event_type, payload)
                    if text.strip():
                        first_user = first_user or text
                elif event_type == "agent_message":
                    _append_message(
                        messages,
                        seen,
                        "assistant",
                        str(payload.get("message", "")),
                        event_type,
                        payload,
                    )
                elif event_type == "thread_settings_applied":
                    settings = payload.get("thread_settings") or {}
                    model = model or settings.get("model")
                    reasoning_effort = reasoning_effort or settings.get("reasoning_effort")
            elif item_type == "response_item" and isinstance(payload, dict):
                role = payload.get("role")
                if role in {"user", "assistant"} and payload.get("type") == "message":
                    text = _text_from_content(payload.get("content"))
                    _append_message(messages, seen, role, text, "response_item", payload)
                    if role == "user" and text.strip() and not _is_context_payload(text):
                        first_user = first_user or text

    if not session_id:
        session_id = path.stem
    updated_at = iso_from_stat(path)
    preview = "\n\n".join(message["text"] for message in messages[-3:])[:preview_chars]
    conversation_id = str(session_id)
    account_id = metadata.get("creator_account_id")
    conversation = {
        "id": conversation_id,
        "source_file_id": None,
        "source_id": source.id,
        "source_kind": source.kind,
        "original_path": str(path),
        "archive_path": None,
        "account_id": account_id,
        "source_label": metadata.get("originator") or metadata.get("source") or "Codex",
        "originator": metadata.get("originator"),
        "model_provider": metadata.get("model_provider"),
        "model": model or metadata.get("model"),
        "reasoning_effort": reasoning_effort,
        "cwd": metadata.get("cwd"),
        "git_branch": metadata.get("git_branch"),
        "title": clean_title(metadata.get("title") or first_user),
        "created_at": metadata.get("timestamp") or iso_from_timestamp(metadata.get("created_at")),
        "updated_at": updated_at,
        "message_count": len(messages),
        "user_message_count": sum(1 for message in messages if message["role"] == "user"),
        "assistant_message_count": sum(1 for message in messages if message["role"] == "assistant"),
        "preview": preview,
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
        "content_hash": None,
    }
    return ParsedConversation(conversation, messages)


def parse_vscode_chat_jsonl(path: Path, source: SourceConfig, preview_chars: int) -> ParsedConversation:
    metadata: dict[str, Any] = {}
    messages: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("kind") == 0 and isinstance(item.get("v"), dict):
                metadata = item["v"]
            elif item.get("kind") == 1:
                value = item.get("v")
                if isinstance(value, str):
                    _append_message(messages, seen, "user", value, "vscode_chat", item)

    input_state = metadata.get("inputState") or {}
    selected_model = input_state.get("selectedModel") or {}
    auth = selected_model.get("auth") or {}
    session_id = metadata.get("sessionId") or path.stem
    first_user = messages[0]["text"] if messages else None
    created_at = iso_from_timestamp(metadata.get("creationDate"))
    conversation_id = f"vscode:{session_id}"
    preview = "\n\n".join(message["text"] for message in messages[-3:])[:preview_chars]
    conversation = {
        "id": conversation_id,
        "source_file_id": None,
        "source_id": source.id,
        "source_kind": source.kind,
        "original_path": str(path),
        "archive_path": None,
        "account_id": auth.get("accountLabel"),
        "source_label": "VS Code Chat",
        "originator": "vscode",
        "model_provider": selected_model.get("vendor"),
        "model": selected_model.get("identifier") or selected_model.get("name"),
        "reasoning_effort": None,
        "cwd": None,
        "git_branch": None,
        "title": clean_title(first_user, "VS Code Chat 会话"),
        "created_at": created_at,
        "updated_at": iso_from_stat(path),
        "message_count": len(messages),
        "user_message_count": len(messages),
        "assistant_message_count": 0,
        "preview": preview,
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
        "content_hash": None,
    }
    return ParsedConversation(conversation, messages)


class Scanner:
    """Discover, fingerprint, archive, and index configured conversation files."""

    def __init__(self, config: HubConfig, repository: Repository):
        self.config = config
        self.repository = repository
        self.archive_root = config.data_dir / "archive"
        self.archive_root.mkdir(parents=True, exist_ok=True)

    def discover(self, source: SourceConfig) -> Iterator[Path]:
        root = expand_path(source.path)
        if not root.exists():
            return
        if source.kind == "codex_jsonl":
            candidates = root.rglob("rollout-*.jsonl")
        elif source.kind == "vscode_chat_jsonl":
            candidates = root.glob("*/chatSessions/*.jsonl")
        else:
            candidates = root.rglob("*.jsonl")
        for path in candidates:
            if path.is_file() and not path.is_symlink():
                yield path

    def _archive_snapshot(self, path: Path, digest: str) -> Path:
        destination = self.archive_root / digest[:2] / f"{digest}.jsonl"
        if destination.exists():
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f"{digest}.", suffix=".part", dir=destination.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            shutil.copyfile(path, temporary_path)
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        return destination

    def _parse(self, path: Path, source: SourceConfig) -> ParsedConversation:
        if source.kind == "vscode_chat_jsonl":
            return parse_vscode_chat_jsonl(path, source, self.config.preview_chars)
        return parse_codex_jsonl(path, source, self.config.preview_chars)

    def scan(self) -> dict[str, Any]:
        started_at = now_iso()
        run_id = self.repository.start_scan(started_at)
        stats: dict[str, Any] = {
            "files_seen": 0,
            "files_indexed": 0,
            "files_skipped": 0,
            "conversations_indexed": 0,
            "messages_indexed": 0,
            "errors": [],
            "sources": [],
        }
        try:
            for source in self.config.sources:
                if not source.enabled:
                    continue
                source_path = expand_path(source.path)
                self.repository.upsert_source(source.id, source.kind, str(source_path), source.enabled)
                source_error: str | None = None
                if not source_path.exists():
                    source_error = f"source path does not exist: {source_path}"
                    stats["errors"].append({"source": source.id, "error": source_error})
                    self.repository.mark_source_scan(source.id, now_iso(), source_error)
                    stats["sources"].append({"id": source.id, "files": 0, "error": source_error})
                    continue
                source_files = 0
                for path in self.discover(source):
                    stats["files_seen"] += 1
                    source_files += 1
                    try:
                        stat = path.stat()
                        if stat.st_size > self.config.max_file_bytes:
                            raise ValueError(
                                f"file exceeds max_file_bytes={self.config.max_file_bytes}: {stat.st_size}"
                            )
                        original_path = str(path)
                        old = self.repository.existing_file(source.id, original_path)
                        if (
                            old
                            and old["size_bytes"] == stat.st_size
                            and old["mtime_ns"] == stat.st_mtime_ns
                            and old["parse_status"] == "ok"
                            and (
                                not self.config.archive_originals
                                or old["archive_path"] and Path(old["archive_path"]).exists()
                            )
                        ):
                            stats["files_skipped"] += 1
                            continue
                        digest = sha256_file(path)
                        parsed = self._parse(path, source)
                        archive_path = (
                            self._archive_snapshot(path, digest) if self.config.archive_originals else None
                        )
                        parsed.conversation["archive_path"] = str(archive_path) if archive_path else None
                        parsed.conversation["content_hash"] = digest
                        file_id = self.repository.upsert_source_file(
                            {
                                "source_id": source.id,
                                "original_path": original_path,
                                "size_bytes": stat.st_size,
                                "mtime_ns": stat.st_mtime_ns,
                                "sha256": digest,
                                "archive_path": str(archive_path) if archive_path else None,
                                "parse_status": "ok",
                                "error": None,
                                "conversation_id": parsed.conversation["id"],
                                "indexed_at": now_iso(),
                            }
                        )
                        parsed.conversation["source_file_id"] = file_id
                        self.repository.replace_conversation(parsed.conversation, parsed.messages)
                        stats["files_indexed"] += 1
                        stats["conversations_indexed"] += 1
                        stats["messages_indexed"] += len(parsed.messages)
                    except Exception as exc:  # preserve one bad source and continue scanning
                        error = f"{type(exc).__name__}: {exc}"
                        stats["errors"].append({"source": source.id, "path": str(path), "error": error})
                        try:
                            stat = path.stat()
                            self.repository.upsert_source_file(
                                {
                                    "source_id": source.id,
                                    "original_path": str(path),
                                    "size_bytes": stat.st_size,
                                    "mtime_ns": stat.st_mtime_ns,
                                    "sha256": "",
                                    "archive_path": None,
                                    "parse_status": "error",
                                    "error": error,
                                    "conversation_id": None,
                                    "indexed_at": now_iso(),
                                }
                            )
                        except OSError:
                            pass
                self.repository.mark_source_scan(source.id, now_iso(), source_error)
                stats["sources"].append({"id": source.id, "files": source_files, "error": source_error})
            status = "partial" if stats["errors"] else "ok"
            self.repository.finish_scan(run_id, stats, now_iso(), status)
            stats["run_id"] = run_id
            stats["status"] = status
            return stats
        except Exception as exc:
            stats["errors"].append({"error": f"{type(exc).__name__}: {exc}"})
            self.repository.finish_scan(run_id, stats, now_iso(), "error")
            stats["run_id"] = run_id
            stats["status"] = "error"
            return stats
