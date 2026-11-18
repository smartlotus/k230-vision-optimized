"""Build a checked offline ZIP from the Linux handoff worktree."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

from verify_payload import verify_inventory


def content(source):
    return {
        "bytes": len(source),
        "sha256": hashlib.sha256(source).hexdigest(),
    }


def select(root):
    paths = {Path("AGENTS.md")}
    paths.update(path.relative_to(root) for path in (root / "docs/linux-handoff").glob("*.md"))
    paths.update(Path("linux_handoff") / name for name in (
        "PAYLOAD_MANIFEST.json", "verify_payload.py", "build_bundle.py",
    ))
    payload = json.loads((root / "linux_handoff/PAYLOAD_MANIFEST.json").read_text(encoding="utf-8"))
    verify_inventory(root / "linux_handoff/payload", payload)
    paths.update(Path("linux_handoff/payload") / row["path"] for row in payload["files"])
    tracked = subprocess.check_output([
        "git", "-C", str(root), "ls-files", "-z", "--", "k230_transport_mvp",
    ]).decode("utf-8").split("\0")
    folders = {"core", "vision", "k230", "ports", "stm32", "tests"}
    suffixes = {".py", ".json", ".md", ".c", ".h", ".txt"}
    for name in filter(None, tracked):
        path = Path(name)
        if ((path.parts[1] in folders or path.parent.as_posix() == "k230_transport_mvp")
                and path.suffix in suffixes):
            paths.add(path)
    return {path.as_posix(): (root / path).read_bytes() for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = select(root)
    files["START_HERE.md"] = """# 庐山派 K230 Linux 交接包

先读 [迁移交接](docs/linux-handoff/README.md) 和 [执行提示词](docs/linux-handoff/EXECUTION_PROMPT.md)。

用 Python 3.11+ 在解压目录执行 `python linux_handoff/verify_payload.py --package` 核对每个文件。Linux 镜像从平台指引中的官方入口获取。
""".encode("utf-8")
    manifest = {"schema": 1, "files": [
        {"path": name, **content(files[name])} for name in sorted(files)
    ]}
    files["PACKAGE_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    arguments = dict(compression=ZIP_DEFLATED, compresslevel=6)
    with ZipFile(args.output, "x", **arguments) as archive:
        for name in sorted(files):
            archive.writestr(name, files[name])
    with ZipFile(args.output) as archive:
        if set(archive.namelist()) != files.keys() or archive.testzip():
            raise ValueError("ZIP inventory or CRC mismatch")
        for item in manifest["files"]:
            if content(archive.read(item["path"])) != {key: item[key] for key in ("bytes", "sha256")}:
                raise ValueError(f"ZIP readback mismatch: {item['path']}")
    with args.output.open("rb") as stream:
        archive_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt = {"file": args.output.name, "sha256": archive_sha,
               "bytes": args.output.stat().st_size, "members": len(files)}
    args.output.with_suffix(".sha256.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
