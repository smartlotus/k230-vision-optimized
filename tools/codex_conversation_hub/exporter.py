"""Create portable continuation bundles for a selected conversation."""

from __future__ import annotations

from pathlib import Path

from .storage import Repository
from .utils import clean_title, slug


def _format_message(role: str, text: str) -> str:
    label = "用户" if role == "user" else "助手"
    return f"### {label}\n\n{text.strip()}\n"


def build_bundle(conversation: dict, max_messages: int = 36, max_chars: int = 60_000) -> str:
    messages = conversation.get("messages", [])[-max_messages:]
    body_parts = [
        "# Codex 对话续接包",
        "",
        f"- 会话标题：{clean_title(conversation.get('title'))}",
        f"- 会话 ID：`{conversation.get('id', '')}`",
        f"- 来源：{conversation.get('source_label') or conversation.get('source_kind') or '未知'}",
        f"- 账号标识：`{conversation.get('account_id') or '未知'}`",
        f"- 项目目录：`{conversation.get('cwd') or '未知'}`",
        f"- 原始文件：`{conversation.get('original_path') or ''}`",
        f"- 归档副本：`{conversation.get('archive_path') or '未启用归档'}`",
        "",
        "## 在另一个 Codex 客户端中继续",
        "",
        "请先读取本文件和上面的原始对话文件，再继续处理最后一个用户请求。保留项目路径、已完成事项、未完成事项和已知限制。不要把工具回执或本地索引状态当成任务完成证明。",
        "",
        "## 对话摘录",
        "",
    ]
    for message in messages:
        body_parts.append(_format_message(message.get("role", "assistant"), message.get("text", "")))
    text = "\n".join(body_parts)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n\n> 摘录达到长度上限。完整内容请读取原始文件或归档副本。\n"
    return text


def export_bundle(repository: Repository, conversation_id: str, output_dir: Path) -> Path:
    conversation = repository.get_conversation(conversation_id)
    if conversation is None:
        raise KeyError(f"conversation not found: {conversation_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{slug(conversation_id)}.md"
    path = output_dir / filename
    path.write_text(build_bundle(conversation), encoding="utf-8")
    return path
