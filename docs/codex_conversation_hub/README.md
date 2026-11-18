# Codex Conversation Hub

这是一个本地优先的 Codex 对话资料库。它把不同账号、Codex Desktop 和 VS Code Codex 产生的本地会话统一收录到 SQLite，并且保留原始 JSONL 的不可变副本。

## 能做什么

- 每次 `serve` 或 `scan` 先扫描配置的源目录。已处理且大小、修改时间没有变化的文件会跳过。
- 默认读取 `%USERPROFILE%/.codex/sessions` 下的 `rollout-*.jsonl`。
- 可选读取 VS Code `workspaceStorage/*/chatSessions/*.jsonl` 的可读会话元数据。它通常包含 Copilot Chat 历史，默认关闭，避免把非 Codex 对话混入资料库。
- 从 Codex `session_meta` 中记录账号标识、来源、模型、工作目录、分支和会话 ID。
- 对用户消息和助手消息建立 SQLite 索引，网页工作台支持关键词、账号和来源筛选。
- 为每个源文件生成 SHA-256 归档副本，源文件变化时产生新的快照，不覆盖历史副本。
- 生成续接包 Markdown。续接包包含原始路径、归档路径、项目目录、账号标识和最近消息摘录，可在另一个账号的 Codex 客户端或 VS Code 中复制路径后继续。

## 启动

在仓库根目录执行：

```powershell
python -m tools.codex_conversation_hub init --output codex-hub.config.json
python -m tools.codex_conversation_hub serve --config codex-hub.config.json --open
```

也可以使用 PowerShell 快捷入口：

```powershell
.\tools\codex_conversation_hub\start_hub.ps1
```

默认数据库和归档目录为 `%LOCALAPPDATA%\CodexConversationHub`。使用 `--data-dir` 可以把它改到另一块磁盘或同步目录。

## 续接流程

1. 在网页左侧搜索标题、项目路径或消息内容。
2. 选择会话后点击“生成并复制续接包”。
3. 在目标 Codex 客户端中粘贴续接包路径，并要求它先读取文件，再继续最后一个用户请求。
4. 如果只需要原始完整记录，可以点击“复制原始路径”或“复制归档路径”。

续接包是文本上下文桥接层。它不会替换目标客户端的登录状态，也不会自动把一个账号的会话注入另一个账号。

## 目录

```text
tools/codex_conversation_hub/
  app.py                 CLI 入口
  config.py              默认路径和配置文件
  scanner.py             增量扫描、解析和归档
  storage.py             SQLite 数据库和查询
  exporter.py            续接包生成
  web.py                 本地 API 和 HTTP 服务
  static/index.html      工作台界面
  tests/test_hub.py      隔离测试夹具
```

## 数据边界

系统只访问配置的会话 JSONL 和可读元数据。它不会读取或修改 `auth.json`、密码、令牌或账号设置，也不会通过浏览器自动登录。当前版本对新版 Codex JSONL 的用户和助手消息、账号元数据、工作目录和模型字段进行结构化索引。遇到未知事件会保留原文件并继续扫描，错误会显示在扫描结果中。

本地数据库和归档副本可能包含完整对话内容，请把 `data_dir` 视为私有资料目录。不要把它加入公共 Git 提交或公开同步。
