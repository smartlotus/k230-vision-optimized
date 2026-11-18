"""Command-line entry point for the local Codex Conversation Hub."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import default_config, load_config, write_config
from .exporter import export_bundle
from .scanner import Scanner
from .storage import Repository
from .utils import copy_to_clipboard
from .web import HubApplication, serve


def _config_path(args: argparse.Namespace) -> Path | None:
    return Path(args.config).expanduser().resolve() if args.config else None


def _load(args: argparse.Namespace):
    config = load_config(_config_path(args))
    if args.data_dir:
        config.data_dir = Path(args.data_dir).expanduser().resolve()
    if args.no_archive:
        config.archive_originals = False
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codex-hub",
        description="本地优先的 Codex 多账号对话归档、检索和续接工作台",
    )
    parser.add_argument("--config", help="配置 JSON 路径")
    parser.add_argument("--data-dir", help="覆盖数据库和归档目录")
    parser.add_argument("--no-archive", action="store_true", help="只索引，不复制原始 JSONL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="生成首个配置文件")
    init.add_argument("--output", default="codex-hub.config.json")

    subparsers.add_parser("scan", help="执行一次增量扫描")

    serve_parser = subparsers.add_parser("serve", help="启动本地网页工作台")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--open", action="store_true", help="启动后打开浏览器")

    export = subparsers.add_parser("export", help="导出指定会话的续接包")
    export.add_argument("conversation_id")
    export.add_argument("--output-dir")
    export.add_argument("--copy", action="store_true", help="同时复制续接包路径")

    path = subparsers.add_parser("path", help="打印指定会话的路径")
    path.add_argument("conversation_id")
    path.add_argument("--kind", choices=["original", "archive"], default="original")
    path.add_argument("--copy", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.command == "init":
        path = write_config(Path(args.output).expanduser().resolve())
        print(json.dumps({"config": str(path), "data_dir": str(default_config().data_dir)}, ensure_ascii=False))
        return 0

    config = _load(args)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    repository = Repository(config.data_dir / "hub.sqlite3")
    try:
        if args.command == "scan":
            print(json.dumps(Scanner(config, repository).scan(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "serve":
            app = HubApplication(config, repository)
            result = app.scan()
            print(json.dumps(result, ensure_ascii=False, indent=2))
            serve(app, args.host, args.port, args.open)
            return 0
        if args.command == "export":
            output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else config.data_dir / "continuations"
            path = export_bundle(repository, args.conversation_id, output_dir)
            copied = copy_to_clipboard(str(path)) if args.copy else False
            print(json.dumps({"path": str(path), "copied": copied}, ensure_ascii=False))
            return 0
        if args.command == "path":
            conversation = repository.get_conversation(args.conversation_id)
            if conversation is None:
                print(f"conversation not found: {args.conversation_id}", file=sys.stderr)
                return 2
            value = conversation.get("original_path" if args.kind == "original" else "archive_path")
            if not value:
                print(f"path unavailable: {args.kind}", file=sys.stderr)
                return 2
            if args.copy:
                copy_to_clipboard(value)
            print(value)
            return 0
    finally:
        if args.command != "serve":
            repository.close()
    return 2


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
