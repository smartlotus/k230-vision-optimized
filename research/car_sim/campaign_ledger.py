"""Durable local task attempts and atomic publication for the campaign runner."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4


TERMINAL = {"PASS", "FAIL", "BLOCKED"}


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def supervisor_lock(root):
    """The OS releases this lock even when the supervisor is killed."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "supervisor.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if not handle.tell():
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            lock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            lock = lambda: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(handle, fcntl.LOCK_UN)
        try:
            lock()
        except OSError as exc:
            raise RuntimeError(f"Campaign supervisor already holds {root}") from exc
        try:
            yield
        finally:
            unlock()


class Ledger:
    def __init__(self, root):
        self.root = root
        self.db = sqlite3.connect(root / "queue.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA foreign_keys = ON;
            PRAGMA synchronous = FULL;
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, specification TEXT NOT NULL,
                status TEXT NOT NULL, attempt TEXT, reason TEXT, updated REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY, job TEXT NOT NULL REFERENCES jobs(id),
                status TEXT NOT NULL, started REAL NOT NULL, updated REAL NOT NULL,
                pid INTEGER, lease_until REAL, detail TEXT
            );
        """)

    def register(self, specs, manifest_id):
        for spec in specs:
            fingerprint = identity([manifest_id, spec])
            with self.db:
                self.db.execute(
                    "INSERT OR IGNORE INTO jobs VALUES (?, ?, ?, 'PENDING', NULL, NULL, ?)",
                    (spec["key"], fingerprint, canonical(spec), time.time()),
                )
            if self.job(spec["key"])["fingerprint"] != fingerprint:
                raise ValueError(f"Job identity changed: {spec['key']}")

    def job(self, key):
        return dict(self.db.execute("SELECT * FROM jobs WHERE id=?", (key,)).fetchone())

    def jobs(self):
        return [dict(row) for row in self.db.execute("SELECT * FROM jobs ORDER BY rowid")]

    def attempts(self, key):
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM attempts WHERE job=? ORDER BY started", (key,)
        )]

    def stage(self, key, lease_seconds):
        attempt = uuid4().hex
        directory = self.root / "attempts" / key / attempt / "work"
        directory.mkdir(parents=True)
        now = time.time()
        with self.db:
            updated = self.db.execute(
                "UPDATE jobs SET status='RUNNING', attempt=?, reason=NULL, updated=? "
                "WHERE id=? AND status='PENDING'", (attempt, now, key),
            )
            if updated.rowcount != 1:
                raise RuntimeError(f"Job is not available: {key}")
            self.db.execute(
                "INSERT INTO attempts VALUES (?, ?, 'RUNNING', ?, ?, NULL, ?, NULL)",
                (attempt, key, now, now, now + lease_seconds),
            )
        return attempt, directory

    def heartbeat(self, attempt, pid, lease_seconds):
        now = time.time()
        with self.db:
            self.db.execute(
                "UPDATE attempts SET pid=?, updated=?, lease_until=? WHERE id=? AND status='RUNNING'",
                (pid, now, now + lease_seconds, attempt),
            )

    def mark(self, key, status, reason=None, attempt_status=None):
        now = time.time()
        with self.db:
            row = self.job(key)
            if attempt_status and row["attempt"]:
                self.db.execute(
                    "UPDATE attempts SET status=?, updated=?, lease_until=NULL, detail=? WHERE id=?",
                    (attempt_status, now, reason, row["attempt"]),
                )
            self.db.execute(
                "UPDATE jobs SET status=?, reason=?, updated=? WHERE id=?",
                (status, reason, now, key),
            )

    def _verify(self, directory, row, required):
        try:
            receipt = read_json(directory / "receipt.json")
            files = receipt["files"]
            if (receipt["fingerprint"] != row["fingerprint"] or
                    receipt["attempt"] != row["attempt"] or
                    receipt["key"] != row["id"] or
                    receipt["status"] not in {"PASS", "FAIL"} or
                    not set(required).issubset(files)):
                return None
            if any(Path(name).is_absolute() or ".." in Path(name).parts or
                   digest(directory / name) != expected for name, expected in files.items()):
                return None
            return receipt
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return None

    def recover(self, specs):
        changed = set()
        for row in self.jobs():
            key = row["id"]
            destination = self.root / "episodes" / key
            if destination.exists():
                receipt = self._verify(destination, row, specs[key]["artifacts"])
                if receipt:
                    if row["status"] != receipt["status"]:
                        self.mark(key, receipt["status"], receipt.get("reason"), receipt["status"])
                    continue
                self._archive(key, destination, "invalid")
                self.mark(key, "PENDING", "Published receipt failed readback", "INVALID")
                changed.add(key)
            elif row["status"] in {"RUNNING", "PASS", "FAIL"}:
                self.mark(key, "PENDING", "Missing publication after interruption", "INTERRUPTED")
                changed.add(key)

        while changed:
            downstream = set()
            for row in self.jobs():
                key = row["id"]
                spec = specs[key]
                depends = set(spec["depends"])
                if spec["worker"] == "delivery":
                    depends.update(specs.keys() - {key})
                if row["status"] not in TERMINAL or not (depends & changed):
                    continue
                destination = self.root / "episodes" / key
                if destination.exists():
                    self._archive(key, destination, "superseded")
                self.mark(key, "PENDING", "Upstream evidence will be regenerated", "SUPERSEDED")
                downstream.add(key)
            changed = downstream

    def _archive(self, key, path, reason):
        archive = self.root / "attempts" / key / f"{reason}-{uuid4().hex}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        path.rename(archive)

    def publish(self, spec, attempt, directory):
        key = spec["key"]
        row = self.job(key)
        result = read_json(directory / "result.json")
        if row["attempt"] != attempt or row["status"] != "RUNNING":
            raise ValueError("Attempt no longer owns the job")
        if result["status"] not in {"PASS", "FAIL"}:
            raise ValueError("Worker must supply a case verdict")
        if any(not (directory / name).is_file() for name in spec["artifacts"]):
            raise ValueError("Required artifacts missing")
        receipt = {
            "key": key, "fingerprint": row["fingerprint"], "attempt": attempt,
            "status": result["status"], "reason": result.get("reason"),
            "files": {p.relative_to(directory).as_posix(): digest(p)
                      for p in sorted(directory.rglob("*")) if p.is_file()},
        }
        write_json(directory / "receipt.json", receipt)
        destination = self.root / "episodes" / key
        destination.parent.mkdir(exist_ok=True)
        directory.rename(destination)
        self.mark(key, result["status"], result.get("reason"), result["status"])
