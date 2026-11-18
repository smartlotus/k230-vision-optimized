"""Local HTTP API and browser UI server."""

from __future__ import annotations

import json
import os
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .exporter import export_bundle
from .scanner import Scanner
from .utils import copy_to_clipboard


class HubApplication:
    def __init__(self, config, repository):
        self.config = config
        self.repository = repository
        self.scanner = Scanner(config, repository)
        self.scan_lock = threading.Lock()
        self.last_scan: dict[str, Any] | None = None

    def scan(self) -> dict[str, Any]:
        with self.scan_lock:
            self.last_scan = self.scanner.scan()
            return self.last_scan

    def export(self, conversation_id: str) -> Path:
        return export_bundle(self.repository, conversation_id, self.config.data_dir / "continuations")


class HubHandler(BaseHTTPRequestHandler):
    server_version = "CodexConversationHub/0.1"

    @property
    def app(self) -> HubApplication:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[hub] {format % args}")

    def _send_json(self, value: Any, status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_text(self, value: str, content_type: str = "text/plain; charset=utf-8") -> None:
        payload = value.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        self._send_json({"error": message}, status)

    def _path_parts(self) -> list[str]:
        return [urllib.parse.unquote(part) for part in self.path.split("?", 1)[0].split("/") if part]

    def do_GET(self) -> None:
        parts = self._path_parts()
        if not parts:
            index_path = Path(__file__).with_name("static") / "index.html"
            self._send_text(index_path.read_text(encoding="utf-8"), "text/html; charset=utf-8")
            return
        if parts[0] != "api":
            self._error("not found", HTTPStatus.NOT_FOUND)
            return
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if parts[1:] == ["summary"]:
                self._send_json({**self.app.repository.summary(), "filters": self.app.repository.filters()})
                return
            if parts[1:] == ["runs"]:
                self._send_json(self.app.repository.latest_runs())
                return
            if parts[1:] == ["conversations"]:
                self._send_json(
                    {
                        "items": self.app.repository.list_conversations(
                            query.get("q", [""])[0],
                            query.get("account", [""])[0],
                            query.get("source", [""])[0],
                            query.get("cwd", [""])[0],
                            int(query.get("limit", [60])[0]),
                            int(query.get("offset", [0])[0]),
                        )
                    }
                )
                return
            if len(parts) == 3 and parts[1] == "conversation":
                conversation = self.app.repository.get_conversation(parts[2])
                if conversation is None:
                    self._error("conversation not found", HTTPStatus.NOT_FOUND)
                else:
                    self._send_json(conversation)
                return
            self._error("not found", HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError) as exc:
            self._error(str(exc))

    def do_POST(self) -> None:
        parts = self._path_parts()
        try:
            if parts[0:2] == ["api", "scan"]:
                self._send_json(self.app.scan())
                return
            if len(parts) == 3 and parts[0:2] == ["api", "export"]:
                path = self.app.export(parts[2])
                self._send_json({"path": str(path), "copied": copy_to_clipboard(str(path))})
                return
            if len(parts) == 3 and parts[0:2] == ["api", "copy"]:
                conversation = self.app.repository.get_conversation(parts[2])
                if conversation is None:
                    self._error("conversation not found", HTTPStatus.NOT_FOUND)
                    return
                parsed = urllib.parse.urlparse(self.path)
                query = urllib.parse.parse_qs(parsed.query)
                kind = query.get("kind", ["original"])[0]
                values = {
                    "original": conversation.get("original_path"),
                    "archive": conversation.get("archive_path"),
                }
                if kind == "bundle":
                    values["bundle"] = str(self.app.export(parts[2]))
                value = values.get(kind)
                if not value:
                    self._error(f"path unavailable: {kind}")
                    return
                self._send_json({"path": value, "copied": copy_to_clipboard(value)})
                return
            if len(parts) == 3 and parts[0:2] == ["api", "open"]:
                conversation = self.app.repository.get_conversation(parts[2])
                if conversation is None:
                    self._error("conversation not found", HTTPStatus.NOT_FOUND)
                    return
                parsed = urllib.parse.urlparse(self.path)
                kind = urllib.parse.parse_qs(parsed.query).get("kind", ["original"])[0]
                value = conversation.get("archive_path" if kind == "archive" else "original_path")
                if not value or not Path(value).exists():
                    self._error("path unavailable")
                    return
                if os.name == "nt":
                    os.startfile(value)  # type: ignore[attr-defined]
                else:
                    webbrowser.open(Path(value).as_uri())
                self._send_json({"opened": value})
                return
            self._error("not found", HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError, OSError) as exc:
            self._error(str(exc))


def serve(app: HubApplication, host: str, port: int, open_browser: bool = False) -> None:
    server = ThreadingHTTPServer((host, port), HubHandler)
    server.app = app  # type: ignore[attr-defined]
    actual_port = server.server_address[1]
    url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{actual_port}/"
    print(f"Codex Conversation Hub: {url}")
    print(f"Database: {app.repository.db_path}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    finally:
        server.server_close()
