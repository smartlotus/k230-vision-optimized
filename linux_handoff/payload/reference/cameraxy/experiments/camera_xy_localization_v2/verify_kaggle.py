"""Verify a size-bounded archive and compact outputs read from a saved Kaggle version."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from contract import TARGETS, digest

MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_BYTES = 96 * 1024 * 1024


def verify(spec, output, source, version):
    expected = spec["notebook_ref"]
    if digest(source) != spec["entry_sha256"]:
        raise ValueError(f"Saved source bytes differ for {expected}/{version}")
    receipt = json.loads((output / "RECEIPT.json").read_text(encoding="utf8"))
    fields = dict(status="COMPUTE_COMPLETE", run_id=spec["run_id"], input_sha256=spec["input_sha256"],
                  input_inventory_sha256=spec["input_inventory_sha256"],
                  dataset_manifest_sha256=spec["manifest_sha256"], automatic_replay=False)
    if any(receipt.get(key) != value for key, value in fields.items()):
        raise ValueError("Kaggle compute receipt does not match the prepared job")
    result = json.loads((output / "result.json").read_text(encoding="utf8"))
    if result["targets"] != list(TARGETS) or result["dataset_manifest_sha256"] != spec["manifest_sha256"]:
        raise ValueError("Returned result lineage differs from submission")
    for item in receipt["compact_results"]:
        if Path(item["name"]).name != item["name"]:
            raise ValueError("Unexpected compact output path")
        file = output / item["name"]
        if file.stat().st_size != item["bytes"] or digest(file) != item["sha256"]:
            raise ValueError(f"Compact output checksum mismatch: {file.name}")
    archive = receipt["archive"]
    name = f"{spec['run_id']}-{archive['sha256'][:16]}.zip"
    if archive["name"] != name or not 0 < archive["bytes"] < MAX_ARCHIVE_BYTES:
        raise ValueError("Archive name or size drift")
    file = output / name
    if file.stat().st_size != archive["bytes"] or digest(file) != archive["sha256"]:
        raise ValueError("Full Kaggle archive readback SHA256 mismatch")
    with zipfile.ZipFile(file) as package:
        inventory = json.loads(package.read("ARTIFACT_INVENTORY.json"))
        names = [item["name"] for item in inventory]
        if (len(set(names)) != len(names) or len(package.namelist()) != len(names)+1 or
                set(package.namelist()) != set(names) | {"ARTIFACT_INVENTORY.json"}):
            raise ValueError("Archive inventory changed")
        if sum(item["bytes"] for item in inventory) > MAX_EXPANDED_BYTES:
            raise ValueError("Archive expanded size out of bounds")
        for item in inventory:
            name = item["name"]
            if Path(name).name != name or package.getinfo(name).file_size != item["bytes"]:
                raise ValueError("Unsafe archive member path or size")
            with package.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                    raise ValueError(f"Saved member hash differs: {name}")
        source_run = json.loads(package.read("run.json"))
        for key, actual in source_run.items():
            if key in spec and spec[key] != actual:
                raise ValueError(f"Run configuration disagrees with source: {key}")
    return dict(
        schema="cloud-direct-write-v1", status="FULL_BYTE_SHA256_PASS", provider="Kaggle",
        run_id=spec["run_id"], kernel_ref=expected, version=version,
        source_sha256=digest(source), dataset_ref=spec["dataset_ref"],
        dataset_manifest_sha256=spec["manifest_sha256"], archive=archive,
        uploaded=dict(archive, file_id=f"{expected}/{version}/{archive['name']}",
                      provider_readback="FULL_BYTE_SHA256_PASS"),
        compact_results=receipt["compact_results"], verified_archive_members=inventory,
        automatic_replay=False, scientific_full_gate=result["full_gate"],
    )


def main():
    p = argparse.ArgumentParser()
    for key in ("spec", "output", "source", "receipt"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--version", type=int, required=True)
    a = p.parse_args()
    source = json.loads(a.spec.read_text(encoding="utf8"))
    record = verify(source, a.output, a.source, a.version)
    with a.receipt.open("x", encoding="utf8") as f:
        f.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(f'{record["kernel_ref"]}/{record["version"]} {record["status"]} {record["archive"]["sha256"]}')


if __name__ == "__main__":
    main()
