"""Verify the payload manifest or the complete extracted Linux handoff ZIP."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify_inventory(base, manifest):
    rows = manifest["files"]
    expected = {}
    for row in rows:
        name = PurePosixPath(row["path"])
        if name.is_absolute() or ".." in name.parts or str(name) in expected:
            raise ValueError(f"Invalid or repeated manifest path: {name}")
        expected[str(name)] = row
    total = 0
    for name, row in expected.items():
        path = base / name
        if not path.resolve().is_relative_to(base.resolve()) or path.is_symlink():
            raise ValueError(f"Unexpected link or escaped path: {name}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        size = path.stat().st_size
        if digest != row["sha256"] or size != row["bytes"]:
            raise ValueError(f"Checksum or size mismatch: {name}")
        total += size
    print(f"Verified {len(expected)} files, {total:,} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="store_true", help="verify all ZIP files after extraction")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    if args.package:
        base = here.parent
        inventory = base / "PACKAGE_MANIFEST.json"
    else:
        base = here / "payload"
        inventory = here / "PAYLOAD_MANIFEST.json"
    manifest = json.loads(inventory.read_text(encoding="utf-8"))
    verify_inventory(base, manifest)
    if not args.package:
        models = manifest["models"]
        listed = {entry["path"]: entry for entry in manifest["files"]}
        for model in models:
            raw = listed[model["file"]]
            if raw["sha256"] != model["sha256"] or raw["bytes"] != model["bytes"]:
                raise ValueError(f"Inconsistent model identity: {model['id']}")
        print(f"Model identities verified: {len(models)}")


if __name__ == "__main__":
    main()
