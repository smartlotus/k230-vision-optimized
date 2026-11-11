#!/usr/bin/env python3
"""Portable OPS development coordination. Python standard library only."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SCHEMA = 1
STATES = {"planned", "active", "paused", "blocked", "ready", "integrated"}
MARKER = "<!-- ops-pipeline -->"


class OpsError(Exception):
    pass


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def run(root, *cmd, check=True):
    try:
        p = subprocess.run(cmd, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    except subprocess.TimeoutExpired:
        raise OpsError(f"{cmd[0]} operation timed out; inspect local/remote outcome before retrying")
    if check and p.returncode:
        # Avoid echoing credential-bearing command arguments/URLs.
        raise OpsError(f"{cmd[0]} {cmd[1] if len(cmd) > 1 else ''} failed ({p.returncode}): {redact(p.stderr.strip())}")
    return p


def redact(text):
    text = re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[redacted]@", text)
    return re.sub(r"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,})", "[redacted-token]", text)


def git(root, *args, check=True):
    return run(root, "git", "--literal-pathspecs", *args, check=check).stdout.strip()


def ident(value):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", value):
        raise OpsError("IDs must use 1-64 lowercase letters, digits or hyphens")
    return value


def relpath(root, value):
    value = value.replace("\\", "/").rstrip("/")
    p = PurePosixPath(value)
    if not value or p.is_absolute() or any(x in value for x in (":", "*", "?", "[", "]")) or any(x in {"..", "."} for x in value.split("/")):
        raise OpsError(f"Expected literal project-relative file/directory: {value}")
    value = str(p)
    if value.casefold() == ".git" or value.casefold().startswith(".git/"):
        raise OpsError("Git metadata is not an editable task scope")
    target = (root / value).resolve()
    if not target.is_relative_to(root):
        raise OpsError("Path escapes project (including symlink)")
    if target != (root / value).absolute():
        raise OpsError("Symlink/junction aliases cannot be claimed as separate paths")
    return value


def overlap(a, b):
    # Conservative on case-sensitive systems; consistent across Windows/Linux.
    a, b = a.casefold(), b.casefold()
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def owned(task, value):
    return any(value.casefold() == p.casefold() or value.casefold().startswith(p.casefold() + "/") for p in task["paths"])


def fingerprint(root, files):
    result = {}
    for f in files:
        f = relpath(root, f)
        p = root / f
        if p.is_dir():
            raise OpsError("Checkpoint --files must list exact files, not directories")
        result[f] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    return result


def scoped_files(w, task, extra=()):
    tracked = git(w.root, "ls-files", "-z").split("\0")
    untracked = git(w.root, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
    return sorted({p for p in tracked + untracked + list(extra) if p and owned(task, p)
                   and not p.startswith((".ops/tasks/", ".ops/handoffs/"))})


def append_block(path, block):
    old = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    if MARKER not in old:
        path.write_text(old + ("\n\n" if old else "") + MARKER + "\n" + block.rstrip() + "\n<!-- /ops-pipeline -->\n", encoding="utf-8")


class Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.config = read(self.root / ".ops/config.json")
        if self.config["schema_version"] != SCHEMA:
            raise OpsError("Unsupported config schema; use matching tool version")
        top = Path(git(self.root, "rev-parse", "--show-toplevel")).resolve()
        if top != self.root:
            raise OpsError("--root must be the Git worktree root")
        common = Path(git(self.root, "rev-parse", "--git-common-dir"))
        self.common = (self.root / common).resolve()
        self.runtime = self.common / "ops-runtime" / self.config["project_id"]
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.dbpath = self.runtime / "coordination.sqlite3"
        self.db = sqlite3.connect(self.dbpath, timeout=5, isolation_level=None)
        self.db.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, record TEXT NOT NULL, checkout TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        with self.transaction():
            if not self.db.execute("SELECT value FROM meta WHERE key='initialized'").fetchone():
                for path in sorted((self.root / ".ops/tasks").glob("*.json")):
                    task = read(path)
                    if task["schema_version"] != SCHEMA or task["project_id"] != self.config["project_id"]:
                        raise OpsError("Foreign/incompatible task snapshot; reconciliation required")
                    # A cloned reservation is a report, never a live local lock.
                    task["claimed"] = False
                    if task["state"] == "active":
                        task["state"] = "paused"
                        task["reconcile_required"] = "Imported active task; confirm original owner before writing"
                    self.db.execute("INSERT INTO tasks VALUES (?,?,?)", (task["id"], json.dumps(task), str(self.root)))
                self.db.execute("INSERT INTO meta VALUES ('initialized',?)", (now(),))

    @contextlib.contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def all(self):
        return [(json.loads(t), Path(p)) for t, p in self.db.execute("SELECT record,checkout FROM tasks ORDER BY id")]

    def task(self, task_id):
        row = self.db.execute("SELECT record,checkout FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            raise OpsError("Unknown task; coordinator must register it first")
        return json.loads(row[0]), Path(row[1])

    def coordinator(self, agent):
        if agent != self.config["coordinator"]:
            raise OpsError("Only the configured coordinator may change assignments/integration")

    def check_owner(self, task, args):
        if not task.get("claimed") or task.get("agent") != args.agent or task.get("session") != args.session:
            raise OpsError("Task is not claimed by this Agent session")
        _, checkout = self.task(task["id"])
        if checkout != self.root:
            raise OpsError("Operate from the registered task checkout")
        if git(self.root, "branch", "--show-current") != task["branch"]:
            raise OpsError("Branch changed since claim")
        self.check_snapshot(task, checkout)

    def check_snapshot(self, task, checkout):
        path = checkout / ".ops/tasks" / (task["id"] + ".json")
        if not path.exists() or read(path) != task:
            raise OpsError(f"Runtime/snapshot disagree for {task['id']}; preserve both and reconcile")

    def save(self, task):
        task["revision"] += 1
        task["updated_at"] = now()
        # Snapshot first, then transactional DB. A crash gap is detected on entry.
        atomic(self.root / ".ops/tasks" / (task["id"] + ".json"), task)
        self.db.execute("INSERT INTO tasks VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET record=excluded.record,checkout=excluded.checkout", (task["id"], json.dumps(task, ensure_ascii=False), str(self.root)))


def init(args):
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise OpsError("Project root must already exist")
    top = run(root, "git", "rev-parse", "--show-toplevel", check=False)
    if top.returncode:
        run(root, "git", "init", "-b", "main")
    elif Path(top.stdout.strip()).resolve() != root:
        raise OpsError("Refusing to initialize a subdirectory of another Git repository")
    config_path = root / ".ops/config.json"
    if not config_path.exists():
        if args.github_auto and not args.authorization:
            raise OpsError("--github-auto requires an authorization note from the user instruction")
        config = {"schema_version": SCHEMA, "project_id": uuid.uuid4().hex, "coordinator": args.coordinator,
                  "main_branch": git(root, "branch", "--show-current") or "main",
                  "sync": {"auto": args.github_auto, "authorization": args.authorization or "", "remote": args.remote,
                           "create_private": args.github_auto, "repo": args.repo}}
        atomic(config_path, config)
    elif args.github_auto:
        raise OpsError("Existing config preserved; configure sync after inspecting existing settings")
    for directory in ("tasks", "handoffs", "tools"):
        (root / ".ops" / directory).mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    protocol = source.parent.parent / "references/development-protocol.md"
    if not protocol.exists():
        protocol = source.parent.parent / "PROTOCOL.md"
    for src, target in ((source, root / ".ops/tools/ops.py"), (protocol, root / ".ops/PROTOCOL.md")):
        if not target.exists():
            shutil.copyfile(src, target)
    project = root / ".ops/PROJECT.md"
    if not project.exists():
        project.write_text("# 项目交接入口\n\n初始化仅建立结构；以下内容由协调者依据项目事实填写。\n\n## 目标与明确不做的范围\n\n待填写。\n\n## 目录与关键接口\n\n待填写。\n\n## 启动、构建与验收命令\n\n待填写。\n\n## 约束、用户授权与主线集成规则\n\n待填写。\n\n## 当前阶段与任务入口\n\n运行 `python .ops/tools/ops.py status`，再读任务最新 handoff。任务状态以同机共享运行库为当前登记，仓库 JSON 是持久快照。\n", encoding="utf-8")
    block = "## OPS Pipeline 开发交接\n\n开发前读 `.ops/PROJECT.md`、`.ops/PROTOCOL.md`，运行 `python .ops/tools/ops.py status`。\n写入前按协议认领任务；接手核验最新交接；模块结束/暂停/换 Agent 时写 checkpoint 并按已授权配置同步。\n保留项目原有协议与运行状态；配置不能覆盖宿主权限和明确项目限制。"
    append_block(root / "AGENTS.md", block)
    # Only touch other tool entrypoints when they already exist.
    for name in ("CLAUDE.md", "GEMINI.md"):
        if (root / name).exists():
            append_block(root / name, block)
    gi = root / ".gitignore"
    if not gi.exists():
        gi.write_text("", encoding="utf-8")
    old_ignore = gi.read_text(encoding="utf-8-sig")
    if "# ops-pipeline local exclusions" not in old_ignore:
        gi.write_text(old_ignore + "\n# ops-pipeline local exclusions\n.ops/local/\n.env\n.env.*\n!.env.example\n*.pem\n*.key\n__pycache__/\n", encoding="utf-8")
    w = Workspace(root)
    w.db.close()
    return {"initialized": str(root), "next": "Fill PROJECT.md; commit exact bootstrap files locally; task-add; use an isolated task branch/worktree"}


def task_add(w, a):
    w.coordinator(a.agent)
    task_id = ident(a.task)
    paths = [relpath(w.root, p) for p in a.paths]
    if any(overlap(p, ".ops") or overlap(p, "AGENTS.md") for p in paths):
        if a.lane != "serial":
            raise OpsError("Protocol/coordination changes require the serial lane")
    if w.db.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone():
        raise OpsError("Task ID already exists")
    for dep in a.depends:
        w.task(dep)
    task = {"schema_version": SCHEMA, "project_id": w.config["project_id"], "id": task_id, "title": a.title,
            "paths": paths, "lane": a.lane, "resources": a.resources, "depends": a.depends,
            "acceptance": a.acceptance, "branch": a.branch or "ops/" + task_id, "state": "planned",
            "claimed": False, "agent": None, "session": None, "revision": 0, "handoff": None}
    git(w.root, "check-ref-format", "--branch", task["branch"])
    if task["branch"] in {w.config["main_branch"], "main", "master", "trunk"}:
        raise OpsError("Task branch must differ from the protected main branch")
    w.save(task)
    return task


def claim(w, a):
    task, checkout = w.task(a.task)
    if task.get("reconcile_required"):
        raise OpsError(task["reconcile_required"])
    w.check_snapshot(task, checkout)
    if task["state"] == "integrated":
        raise OpsError("Integrated task is immutable; create a new task")
    if task["claimed"]:
        raise OpsError("Task already claimed (including by an earlier session)")
    if not a.agent.strip() or not a.session.strip():
        raise OpsError("Agent and session identities must be nonempty")
    if git(w.root, "branch", "--show-current") != task["branch"]:
        raise OpsError(f"Use the assigned branch {task['branch']} in this checkout first")
    for dep in task["depends"]:
        if w.task(dep)[0]["state"] != "integrated":
            raise OpsError(f"Dependency {dep} is not integrated")
    for other, location in w.all():
        if not other["claimed"]:
            continue
        conflict = task["lane"] == "serial" or other["lane"] == "serial" or location == w.root or task["branch"] == other["branch"]
        conflict |= bool(set(task["resources"]) & set(other["resources"]))
        conflict |= any(overlap(p, q) for p in task["paths"] for q in other["paths"])
        if conflict:
            raise OpsError(f"Claim conflicts with {other['id']} owned by {other['agent']}/{other['session']}")
    if task["handoff"]:
        # Cross-worktree takeover must carry the exact prior snapshot first.
        if not (w.root / task["handoff"]).exists():
            raise OpsError("Latest handoff missing here; fetch/switch exact task branch or copy preserved local handoff first")
        if a.reconcile:
            w.coordinator(a.agent)
            if not a.reason:
                raise OpsError("Reconciliation claim requires an evidence-based --reason")
            task["reconciliation_note"] = {"reason": a.reason, "at": now()}
        else:
            verify(w, a)
    task.update(state="active", claimed=True, agent=a.agent, session=a.session)
    w.save(task)
    return task


def checkpoint(w, a):
    task, _ = w.task(a.task)
    w.check_owner(task, a)
    files = sorted(set(relpath(w.root, p) for p in a.files))
    if any(not owned(task, p) for p in files):
        raise OpsError("Checkpoint includes files outside the task scope")
    changed = git(w.root, "diff", "HEAD", "--no-renames", "--name-only", "-z").split("\0")
    changed += git(w.root, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
    missing = [p for p in changed if p and owned(task, p) and p not in files
               and not p.startswith((".ops/tasks/", ".ops/handoffs/"))]
    if missing:
        raise OpsError("Checkpoint omits modified owned files: " + ", ".join(sorted(set(missing))))
    report = read(Path(a.report).resolve())
    for key in ("completed", "remaining", "decisions", "risks", "next_steps", "verification"):
        if key not in report or not isinstance(report[key], list):
            raise OpsError(f"Report needs list field: {key}")
    if not report["next_steps"]:
        raise OpsError("Report needs a concrete next step (including review/integration)")
    for v in report["verification"]:
        if not all(k in v for k in ("command", "exit_code", "summary", "evidence")) or not isinstance(v["exit_code"], int):
            raise OpsError("Verification needs command, integer exit_code, summary and evidence list")
        for p in v["evidence"]:
            if not (w.root / relpath(w.root, p)).is_file():
                raise OpsError("Verification evidence file missing")
    if a.state == "ready" and (not report["verification"] or any(v["exit_code"] != 0 for v in report["verification"])):
        raise OpsError("Ready needs actual successful verification; use paused/blocked for incomplete work")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    base = ".ops/handoffs/" + task["id"] + "/" + stamp
    data = {"schema_version": SCHEMA, "project_id": w.config["project_id"], "task": task["id"], "created_at": now(),
            "agent": a.agent, "session": a.session, "branch": task["branch"], "head": git(w.root, "rev-parse", "HEAD"),
            "state": a.state, "files": fingerprint(w.root, files),
            "git_blob_oids": {p: git(w.root, "hash-object", "--path=" + p, p) if (w.root / p).exists() else None for p in files},
            "scope_git_blob_oids": {p: git(w.root, "hash-object", "--path=" + p, p) if (w.root / p).exists() else None for p in scoped_files(w, task, files)},
            "scope_fingerprints": fingerprint(w.root, scoped_files(w, task, files)), "report": report}
    serialized = json.dumps(data, ensure_ascii=False)
    if redact(serialized) != serialized:
        raise OpsError("Report appears to contain a credential; redact at source")
    atomic(w.root / (base + ".json"), data)
    lines = [f"# 交接：{task['title']}", "", f"任务 `{task['id']}` · `{a.state}` · `{data['created_at']}`",
             f"分支 `{task['branch']}`；观察到的 HEAD `{data['head']}`（提交前快照，不代表已云备份）。", "",
             "机器可核验记录：`" + base + ".json`；同步回执用 `status` 查看。", ""]
    for key, label in (("completed", "已完成"), ("remaining", "未完成"), ("decisions", "决定与理由"), ("risks", "已知问题"), ("verification", "验证证据"), ("next_steps", "下一步")):
        lines += ["## " + label, ""] + ["- " + (json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)) for x in report[key]] + [""]
    (w.root / (base + ".md")).write_text("\n".join(lines), encoding="utf-8")
    task.update(state=a.state, handoff=base + ".json")
    w.save(task)
    return {"task": a.task, "handoff": base + ".md", "sync_files": files + [base + ".json", base + ".md", ".ops/tasks/" + a.task + ".json"]}


def verify(w, a):
    task, checkout = w.task(a.task)
    w.check_snapshot(task, checkout)
    w.check_snapshot(task, w.root)
    if not task["handoff"]:
        raise OpsError("No checkpoint yet; inspect the planned task")
    data = read(w.root / task["handoff"])
    if data["project_id"] != w.config["project_id"] or data["task"] != task["id"]:
        raise OpsError("Foreign handoff")
    if git(w.root, "branch", "--show-current") != data["branch"]:
        raise OpsError("Wrong branch for this handoff")
    head = git(w.root, "rev-parse", "HEAD")
    if run(w.root, "git", "merge-base", "--is-ancestor", data["head"], head, check=False).returncode:
        raise OpsError("Current HEAD is not descended from the observed handoff HEAD")
    if fingerprint(w.root, data["files"]) != data["files"]:
        raise OpsError("Checkpoint files changed/missing; investigate and re-verify affected work")
    current_scope = fingerprint(w.root, scoped_files(w, task, data["files"]))
    if current_scope != data["scope_fingerprints"]:
        raise OpsError("Owned scope changed since checkpoint, including unlisted files")
    return {"verified": a.task, "observed_head": data["head"], "current_head": head,
            "note": "Fingerprint/lineage verified; read remaining work and examine new commits/unlisted changes separately"}


def release(w, a):
    task, _ = w.task(a.task)
    w.check_owner(task, a)
    if task["state"] == "active" or not task["handoff"]:
        raise OpsError("Checkpoint paused/blocked/ready before releasing ownership")
    verify(w, a)
    task["claimed"] = False
    w.save(task)
    return {"released": a.task, "note": "Release changes the snapshot; sync it before transfer when automatic backup is enabled"}


def recover(w, a):
    w.coordinator(a.agent)
    task, location = w.task(a.task)
    if not a.reason or not a.owner_stopped:
        raise OpsError("Recovery requires --owner-stopped and an evidence-based --reason")
    # Explicitly preserve the old snapshot, including crash-ahead versions.
    old = location / ".ops/tasks" / (a.task + ".json")
    evidence = {"runtime": task, "snapshot": read(old) if old.exists() else None, "reason": a.reason, "at": now()}
    path = w.root / ".ops/handoffs" / a.task / ("recovery-" + uuid.uuid4().hex + ".json")
    atomic(path, evidence)
    task.update(claimed=False, state="paused", reconcile_required=None, recovery=str(path.relative_to(w.root)).replace("\\", "/"))
    w.save(task)
    return {"recovered": a.task, "evidence": task["recovery"], "next": "Reconcile preserved work/checkpoint; recover does not declare work ready"}


def receipt_path(w, task_id):
    return w.runtime / ("sync-" + task_id + ".json")


def github_repo(url):
    m = re.fullmatch(r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?", url)
    if not m:
        raise OpsError("Remote must be an uncredentialed github.com HTTPS/SSH URL; keep non-GitHub remotes unchanged")
    return m.group(1)


def remote_for_sync(w):
    cfg = w.config["sync"]
    remotes = git(w.root, "remote").splitlines()
    name = cfg["remote"]
    if remotes:
        if name not in remotes:
            raise OpsError("Configured remote absent; inspect multiple/existing remotes rather than creating another repository")
        url = git(w.root, "remote", "get-url", name)
        repo = github_repo(url)
        pushurls = git(w.root, "remote", "get-url", "--push", "--all", name).splitlines()
        if len(pushurls) != 1 or github_repo(pushurls[0]).casefold() != repo.casefold():
            raise OpsError("Push destination differs from the configured fetch repository")
        if cfg.get("repo") and cfg["repo"].casefold() != repo.casefold():
            raise OpsError("Pinned repository differs from existing remote")
        pin_target(w, name, repo)
        return name, repo
    if not cfg["create_private"]:
        raise OpsError("No remote and private repository creation was not authorized")
    cloud = w.runtime / "cloud-intent.json"
    intent = read(cloud) if cloud.exists() else None
    if intent:
        repo = intent["repo"]
    else:
        login = run(w.root, "gh", "api", "user", "--jq", ".login").stdout.strip()
        slug = re.sub(r"[^a-z0-9-]+", "-", w.root.name.lower()).strip("-")[:60] or "project"
        repo = cfg.get("repo") or login + "/" + slug + "-" + w.config["project_id"][:8]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise OpsError("New repository needs OWNER/NAME")
        exists = run(w.root, "gh", "repo", "view", repo, "--json", "nameWithOwner,isPrivate", check=False)
        if exists.returncode == 0:
            raise OpsError("Requested repository already exists; explicitly connect it after inspection")
        # Verify authentication before recording a creation intent.
        run(w.root, "gh", "auth", "status")
        intent = {"repo": repo, "requested_private": True, "at": now()}
        atomic(cloud, intent)
        created = run(w.root, "gh", "repo", "create", repo, "--private", check=False)
        view = run(w.root, "gh", "repo", "view", repo, "--json", "nameWithOwner,isPrivate", check=False)
        if view.returncode or not json.loads(view.stdout)["isPrivate"]:
            raise OpsError("Private repository creation not confirmed; keep the intent and retry after inspection")
        if created.returncode:
            intent["recovered_unknown_outcome"] = True
            atomic(cloud, intent)
    view = run(w.root, "gh", "repo", "view", repo, "--json", "nameWithOwner,isPrivate", check=False)
    if view.returncode:
        if "404" not in view.stderr:
            raise OpsError("Creation outcome still unconfirmed; inspect authentication/network before retrying")
        # Confirmed absent: retry the same recorded target once, never choose a new name.
        run(w.root, "gh", "repo", "create", repo, "--private", check=False)
        view = run(w.root, "gh", "repo", "view", repo, "--json", "nameWithOwner,isPrivate")
    if not json.loads(view.stdout)["isPrivate"]:
        raise OpsError("Created repository is not private; inspect before publication")
    git(w.root, "remote", "add", name, "https://github.com/" + repo + ".git")
    pin_target(w, name, repo)
    return name, repo


def pin_target(w, name, repo):
    path = w.runtime / "cloud-target.json"
    target = {"remote": name, "repo": repo.casefold()}
    if path.exists() and read(path) != target:
        raise OpsError("Remote changed after sync target was pinned; coordinator must inspect and explicitly reconfigure")
    if not path.exists():
        atomic(path, target)


def scan_committed_history(w):
    """Check actual reachable Git blobs, including old versions in a first backup."""
    objects = git(w.root, "rev-list", "--objects", "HEAD").splitlines()
    ids = []
    for entry in objects:
        oid, _, path = entry.partition(" ")
        ids.append(oid)
        base = PurePosixPath(path).name.casefold()
        if base == ".env" or (base.startswith(".env.") and base != ".env.example") or base.endswith((".pem", ".key")):
            raise OpsError("Credential filename exists in outgoing Git history; inspect history before publication")
    query = "\n".join(ids) + "\n"
    checked = subprocess.run(["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
                             cwd=w.root, input=query, capture_output=True, text=True, encoding="utf-8", timeout=60)
    if checked.returncode:
        raise OpsError("Cannot inspect committed objects")
    blobs = []
    total = 0
    for line in checked.stdout.splitlines():
        oid, kind, size = line.split()
        if kind == "blob":
            size = int(size)
            if size > 20 * 1024 * 1024:
                raise OpsError("Large blob in outgoing Git history; use reviewed artifact storage")
            total += size
            blobs.append(oid)
    if total > 250 * 1024 * 1024:
        raise OpsError("History exceeds the automatic scan budget; coordinator must arrange a reviewed publication path")
    data = subprocess.run(["git", "cat-file", "--batch"], cwd=w.root,
                          input=("\n".join(blobs) + "\n").encode(), capture_output=True, timeout=60)
    if data.returncode:
        raise OpsError("Cannot read committed objects for credential checks")
    offset = 0
    for _ in blobs:
        end = data.stdout.index(b"\n", offset)
        header = data.stdout[offset:end].split()
        size = int(header[2])
        content = data.stdout[end + 1:end + 1 + size]
        offset = end + 1 + size + 1
        if b"\0" not in content:
            value = content.decode("utf-8", "replace")
            if redact(value) != value:
                raise OpsError("Possible credential in actual outgoing Git history; no push performed")


def sync(w, a):
    cfg = w.config["sync"]
    if not cfg["auto"] or not cfg["authorization"]:
        raise OpsError("Automatic GitHub sync is not enabled/authorized for this project")
    task, _ = w.task(a.task)
    if task["state"] == "integrated":
        raise OpsError("Integrated tasks cannot be backed up from an old task branch; use authorized main delivery")
    w.check_snapshot(task, w.root)
    if task["claimed"]:
        w.check_owner(task, a)
    elif task["agent"] != a.agent or task["session"] != a.session:
        w.coordinator(a.agent)
    verify(w, a)
    data = read(w.root / task["handoff"])
    files = list(data["files"]) + [task["handoff"], task["handoff"].removesuffix(".json") + ".md", ".ops/tasks/" + a.task + ".json"]
    files += a.files or []
    files = sorted(set(relpath(w.root, p) for p in files))
    allowed_metadata = {task["handoff"], task["handoff"].removesuffix(".json") + ".md", ".ops/tasks/" + a.task + ".json"}
    if any(not owned(task, p) and p not in allowed_metadata for p in files):
        raise OpsError("Sync file list exceeds task ownership")
    staged = git(w.root, "diff", "--cached", "--name-only", "-z").split("\0")
    if any(p and p not in files for p in staged):
        raise OpsError("Unrelated staged changes present; preserved. Use an isolated checkout or resolve index ownership")
    plan = {"task": a.task, "files": files, "target_branch": task["branch"], "remote": cfg["remote"]}
    if not a.execute:
        return {"dry_run": plan}
    # Preflight selected working files; postcommit checks inspect actual Git history.
    for p in files:
        basename = PurePosixPath(p).name.casefold()
        if (basename == ".env" or (basename.startswith(".env.") and basename != ".env.example") or basename.endswith((".pem", ".key"))):
            raise OpsError("Tracked/selected credential file detected; remove from publication scope")
        local = w.root / p
        if local.is_file():
            if local.stat().st_size > 20 * 1024 * 1024:
                raise OpsError("Large file detected; configure artifact storage rather than automatic source backup")
            content = local.read_bytes()
            if b"\0" not in content and redact(content.decode("utf-8", "replace")) != content.decode("utf-8", "replace"):
                raise OpsError("Possible credential detected; inspect before publication")
    # Commit exact paths; --only prevents unrelated hook-staged files entering the commit.
    existing = [p for p in files if (w.root / p).exists() or git(w.root, "ls-files", "--", p)]
    git(w.root, "add", "--", *existing)
    changed = git(w.root, "diff", "--cached", "--name-only", "--", *files)
    if changed:
        git(w.root, "commit", "--only", "-m", "ops: checkpoint " + a.task, "--", *existing)
    head = git(w.root, "rev-parse", "HEAD")
    record = {**plan, "head": head, "snapshot_revision": task["revision"], "at": now(), "status": "pending"}
    atomic(receipt_path(w, a.task), record)
    try:
        scan_committed_history(w)
        remote, repo = remote_for_sync(w)
        record["repo"] = repo
        # Push HEAD to the task branch; never force and never update main implicitly.
        git(w.root, "push", remote, "HEAD:refs/heads/" + task["branch"])
        actual = git(w.root, "ls-remote", remote, "refs/heads/" + task["branch"]).split()
        if not actual or actual[0] != head:
            raise OpsError("Remote SHA differs; cloud backup not confirmed")
        record.update(status="synced", remote_head=actual[0], confirmed_at=now())
        atomic(receipt_path(w, a.task), record)
        return record
    except (OpsError, OSError) as exc:
        record["error"] = redact(str(exc))
        atomic(receipt_path(w, a.task), record)
        raise


def finish(w, a):
    w.coordinator(a.agent)
    task, location = w.task(a.task)
    w.check_snapshot(task, location)
    if task["claimed"] or task["state"] != "ready":
        raise OpsError("Only released, verified ready tasks can be integrated")
    if git(w.root, "branch", "--show-current") != w.config["main_branch"]:
        raise OpsError("Finish from the configured integration/main branch")
    head = git(w.root, "rev-parse", "HEAD")
    source = git(w.root, "rev-parse", "--verify", a.source_commit + "^{commit}")
    if run(w.root, "git", "merge-base", "--is-ancestor", source, head, check=False).returncode:
        raise OpsError("Source commit has not been integrated into current main HEAD")
    data = read(location / task["handoff"])
    for p, expected in data["git_blob_oids"].items():
        blob = run(w.root, "git", "rev-parse", "--verify", source + ":" + p, check=False)
        actual = blob.stdout.strip() if blob.returncode == 0 else None
        if actual != expected:
            raise OpsError("Source commit does not contain checkpointed work")
    main_paths = {p for p in git(w.root, "ls-tree", "-r", "--name-only", "-z", head).split("\0")
                  if p and owned(task, p) and not p.startswith((".ops/tasks/", ".ops/handoffs/"))}
    expected_paths = {p for p, oid in data["scope_git_blob_oids"].items() if oid is not None}
    if main_paths != expected_paths:
        raise OpsError("Current main task scope has added/removed files since checkpoint")
    for p, expected in data["scope_git_blob_oids"].items():
        blob = run(w.root, "git", "rev-parse", "--verify", head + ":" + p, check=False)
        actual = blob.stdout.strip() if blob.returncode == 0 else None
        if actual != expected:
            raise OpsError("Current main task scope differs from checkpoint; reconcile and verify the integration result")
        current = git(w.root, "hash-object", "--path=" + p, p) if (w.root / p).exists() else None
        if current != actual:
            raise OpsError("Uncommitted main changes affect the task; integration must be committed and verified")
    if set(scoped_files(w, task, data["files"])) != set(data["scope_git_blob_oids"]):
        raise OpsError("Uncommitted main additions affect the task; reconcile before integration")
    task.update(state="integrated", integrated_commit=head, source_commit=source, integration_evidence=a.evidence)
    if not (w.root / relpath(w.root, a.evidence)).is_file():
        raise OpsError("Integration validation evidence missing")
    w.save(task)
    return task


def status(w, a):
    result = []
    for task, location in w.all():
        warnings = []
        try:
            w.check_snapshot(task, location)
        except (OpsError, OSError, ValueError) as e:
            warnings.append(str(e))
        if task.get("reconcile_required"):
            warnings.append(task["reconcile_required"])
        r = receipt_path(w, task["id"])
        receipt = read(r) if r.exists() else None
        result.append({"task": task, "local_checkout": str(location), "sync": receipt,
                       "sync_current": bool(receipt and receipt["status"] == "synced" and receipt.get("snapshot_revision") == task["revision"]),
                       "warnings": warnings})
    return {"schema_version": SCHEMA, "project_id": w.config["project_id"], "scope": "this Git common directory only",
            "tasks": result, "healthy": not any(r["warnings"] for r in result)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("init")
    q.add_argument("--coordinator", default="coordinator")
    q.add_argument("--github-auto", action="store_true")
    q.add_argument("--authorization")
    q.add_argument("--remote", default="origin")
    q.add_argument("--repo", help="OWNER/NAME for a new private repo; existing remotes take precedence")
    q = sub.add_parser("task-add")
    q.add_argument("--agent", required=True)
    q.add_argument("--task", required=True)
    q.add_argument("--title", required=True)
    q.add_argument("--paths", nargs="+", required=True)
    q.add_argument("--acceptance", nargs="+", required=True)
    q.add_argument("--lane", choices=["serial", "parallel"], default="serial")
    q.add_argument("--resources", nargs="*", default=[])
    q.add_argument("--depends", nargs="*", default=[])
    q.add_argument("--branch")
    for name in ("claim", "checkpoint", "release", "sync"):
        q = sub.add_parser(name)
        q.add_argument("--task", required=True)
        q.add_argument("--agent", required=True)
        q.add_argument("--session", required=True)
        if name == "claim":
            q.add_argument("--reconcile", action="store_true")
            q.add_argument("--reason")
        if name == "checkpoint":
            q.add_argument("--files", nargs="+", required=True)
            q.add_argument("--report", required=True)
            q.add_argument("--state", choices=["paused", "blocked", "ready"], required=True)
        if name == "sync":
            q.add_argument("--files", nargs="*", default=[])
            q.add_argument("--execute", action="store_true")
    for name in ("status", "doctor"):
        sub.add_parser(name)
    q = sub.add_parser("verify")
    q.add_argument("--task", required=True)
    q = sub.add_parser("recover")
    q.add_argument("--task", required=True)
    q.add_argument("--agent", required=True)
    q.add_argument("--reason", required=True)
    q.add_argument("--owner-stopped", action="store_true")
    q = sub.add_parser("finish")
    q.add_argument("--task", required=True)
    q.add_argument("--agent", required=True)
    q.add_argument("--source-commit", required=True)
    q.add_argument("--evidence", required=True)
    return p


def main(argv=None):
    # Machine-readable CLI output is UTF-8, even when Windows redirects under GBK.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    a = parser().parse_args(argv)
    try:
        if a.command == "init":
            result = init(a)
        else:
            w = Workspace(a.root)
            try:
                if a.command in ("status", "doctor", "verify"):
                    result = verify(w, a) if a.command == "verify" else status(w, a)
                else:
                    function = {"task-add": task_add, "claim": claim, "checkpoint": checkpoint,
                                "release": release, "recover": recover, "sync": sync, "finish": finish}[a.command]
                    with w.transaction():
                        result = function(w, a)
                    if a.command in ("checkpoint", "release") and w.config["sync"]["auto"]:
                        a.files, a.execute = [], True
                        with w.transaction():
                            result["sync"] = sync(w, a)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                if a.command == "doctor" and not result["healthy"]:
                    return 2
            finally:
                w.db.close()
            return 0
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OpsError, OSError, ValueError, KeyError, sqlite3.Error, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"error": redact(str(exc)), "local_work_preserved": True}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
