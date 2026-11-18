from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.codex_conversation_hub.exporter import export_bundle
from tools.codex_conversation_hub.models import HubConfig, SourceConfig
from tools.codex_conversation_hub.scanner import Scanner
from tools.codex_conversation_hub.storage import Repository


class HubTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "sessions"
        self.source.mkdir()
        self.config = HubConfig(
            data_dir=self.root / "data",
            sources=[SourceConfig("fixture", "codex_jsonl", str(self.source))],
        )
        self.repository = Repository(self.config.data_dir / "hub.sqlite3")

    def tearDown(self) -> None:
        self.repository.close()
        self.temp_dir.cleanup()

    def write_session(self, suffix: str = "") -> Path:
        path = self.source / f"rollout-2026-11-18-test{suffix}.jsonl"
        lines = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "session-fixture",
                    "creator_account_id": "account-fixture",
                    "originator": "codex_vscode",
                    "source": "vscode",
                    "cwd": r"C:\fixture\project",
                    "timestamp": "2026-11-18T00:00:00+00:00",
                },
            },
            {"type": "event_msg", "payload": {"type": "user_message", "message": "请继续读取相机项目"}},
            {"type": "event_msg", "payload": {"type": "agent_message", "message": "我会先读取项目交接和验证证据。"}},
            {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "# AGENTS.md instructions\n隐藏运行上下文"}]}},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "我会先读取项目交接和验证证据。"}]}},
        ]
        path.write_text("\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n", encoding="utf-8")
        return path

    def test_scan_archive_search_export_and_incremental_skip(self) -> None:
        path = self.write_session()
        scanner = Scanner(self.config, self.repository)
        first = scanner.scan()
        self.assertEqual(first["files_indexed"], 1)
        self.assertEqual(first["messages_indexed"], 2)
        conversation = self.repository.get_conversation("session-fixture")
        self.assertIsNotNone(conversation)
        assert conversation is not None
        self.assertTrue(Path(conversation["archive_path"]).exists())
        self.assertTrue(conversation["title"].startswith("请继续读取相机项目"))
        self.assertEqual(self.repository.list_conversations("相机")[0]["id"], "session-fixture")
        bundle = export_bundle(self.repository, "session-fixture", self.root / "continuations")
        self.assertTrue(bundle.exists())
        self.assertIn(str(path), bundle.read_text(encoding="utf-8"))
        second = scanner.scan()
        self.assertEqual(second["files_indexed"], 0)
        self.assertEqual(second["files_skipped"], 1)

    def test_changed_file_is_reindexed(self) -> None:
        path = self.write_session()
        scanner = Scanner(self.config, self.repository)
        scanner.scan()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"type": "event_msg", "payload": {"type": "user_message", "message": "新增请求"}}, ensure_ascii=False) + "\n")
        result = scanner.scan()
        self.assertEqual(result["files_indexed"], 1)
        self.assertEqual(self.repository.get_conversation("session-fixture")["user_message_count"], 2)


if __name__ == "__main__":
    unittest.main()
