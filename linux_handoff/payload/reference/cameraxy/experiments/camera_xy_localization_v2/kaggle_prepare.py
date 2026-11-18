"""Prepare an immutable private Kaggle input and the matching saved run."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

from contract import COUNTS, TARGETS, digest, load_rows, xy

ROOT = Path(__file__).resolve().parent
CODE = ("contract.py", "model.py", "baselines.py", "train.py")
MAP_CODE = ("map_matcher.py", "train_map.py")
RECIPE = ("pilot-v1", "xy-map-v1")


def freeze_notebook(root, ref, config, ort_dataset):
    root.mkdir(parents=True)
    title = ref.split("/")[1].replace("-", " ").title()
    code = "CONFIG = " + repr(config) + "\n" + (ROOT / "kaggle_entry.py").read_text(encoding="utf8")
    (root / "entry.py").write_text(code, encoding="utf8")
    (root / "kernel-metadata.json").write_text(json.dumps({
        "id": ref, "title": title, "code_file": "entry.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": config["device"] == "cuda",
        "enable_internet": False, "dataset_sources": [config["dataset_ref"], ort_dataset],
        "kernel_sources": [], "competition_sources": [],
    }, indent=2), encoding="utf8")
    frozen = dict(config, notebook_ref=ref, entry_sha256=digest(root / "entry.py"),
                  kernel_metadata_sha256=digest(root / "kernel-metadata.json"),
                  ort_dataset=ort_dataset)
    (root.parent / "SUBMISSION.json").write_text(json.dumps(frozen, indent=2), encoding="utf8")
    print(json.dumps(frozen, indent=2))


def main():
    p = argparse.ArgumentParser()
    for key in ("data", "layout", "roadnet"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reuse-input", type=Path, help="Existing prepared job spec; retain original private Dataset")
    p.add_argument("--recipe", choices=RECIPE, default="pilot-v1")
    p.add_argument("--ref", required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--ort-wheel", type=Path)
    p.add_argument("--ort-dataset", default="lotus30/k230-camera-xy-ort-cp313-a01")
    a = p.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]{8,42}", a.ref):
        p.error("expected owner/unique-run-slug")
    if a.out.exists():
        p.error("choose a new immutable job output")
    if a.reuse_input:
        prior = json.loads(a.reuse_input.read_text(encoding="utf8"))
        for key in ("notebook_ref", "entry_sha256", "kernel_metadata_sha256"):
            prior.pop(key, None)
        ort_dataset = prior.pop("ort_dataset", a.ort_dataset)
        if a.epochs is not None and a.epochs != prior["epochs"]:
            p.error("use the original epoch budget when replacing a refused session")
        prior.update(run_id=a.ref.split("/")[1], device=a.device)
        freeze_notebook(a.out / "kernel", a.ref, prior, ort_dataset)
        return
    if not (a.data and a.layout and a.roadnet and a.ort_wheel and a.epochs and a.epochs > 0):
        p.error("new input requires data, layout, RoadNet ONNX, ORT wheel, and positive epochs")
    qa = json.loads((a.data / "qa.json").read_text(encoding="utf-8"))
    run = json.loads((a.data / "run.json").read_text(encoding="utf8"))
    manifest = a.data / "manifest.jsonl"
    if (qa["status"] != "PASS" or qa["manifest_sha256"] != digest(manifest)
            or qa["targets"] != list(TARGETS) or qa["pose_groups"] != COUNTS[run["stage"]]):
        raise ValueError("Cannot stage a dataset without verified label QA")
    if digest(a.roadnet) != "00eaaa35f36a0cfd6655531115d2689d501fdaa774f7b79c5cc61e6889099486":
        raise ValueError("RoadNet reference model differs from audited full-v3 ONNX")
    source = CODE + (MAP_CODE if a.recipe == "xy-map-v1" else ())
    members = {p: ROOT / p for p in source}
    members.update({"layout.json": a.layout, "roadnet.onnx": a.roadnet})
    for name in ("run.json", "qa.json", "manifest.jsonl", "field_mask.png"):
        members[f"data/{name}"] = a.data / name
    rows = load_rows(manifest)
    if (len(rows) != qa["frames"] or len({r["pose_group_id"] for r in rows}) != qa["pose_groups"]
            or len({r["sample_id"] for r in rows}) != len(rows)):
        raise ValueError("CameraXY manifest sample/pose counts do not match QA")
    for row in rows:
        xy(row)
        for field in ("image", "mask"):
            name = row[field]
            path = a.data / name
            if not path.resolve().is_relative_to(a.data.resolve()):
                raise ValueError(f"Path escapes dataset: {name}")
            if digest(path) != row[field + "_sha256"]:
                raise ValueError(f"Dataset bytes changed after QA: {path}")
            members[f"data/{name}"] = path
    a.out.mkdir()
    dataset = a.out / "input"
    kernel = a.out / "kernel"
    dataset.mkdir()
    archive_name = "camera_xy.bin" if a.recipe == "xy-map-v1" else "camera_xy.zip"
    archive = dataset / archive_name
    inventory = {name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                 for name, path in members.items()}
    inventory_json = json.dumps(inventory, sort_keys=True).encode("utf8")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as package:
        for name, path in members.items():
            package.write(path, name)
        package.writestr("INPUT_INVENTORY.json", inventory_json)
    ref = a.ref
    slug = ref.split("/")[1]
    title = slug.replace("-", " ").title()
    dataset_ref = ref + "-input"
    (dataset / "dataset-metadata.json").write_text(json.dumps({
        "title": title + " Input", "id": dataset_ref, "licenses": [{"name": "other"}],
    }), encoding="utf8")
    config = dict(run_id=slug, dataset_ref=dataset_ref, intended_input_version=1,
                  recipe=a.recipe, archive_name=archive_name,
                  input_bytes=archive.stat().st_size, input_sha256=digest(archive),
                  input_inventory_sha256=hashlib.sha256(inventory_json).hexdigest(),
                  manifest_sha256=qa["manifest_sha256"], device=a.device, epochs=a.epochs, seed=37121,
                  onnxruntime_wheel=dict(name=a.ort_wheel.name, bytes=a.ort_wheel.stat().st_size,
                                        sha256=digest(a.ort_wheel)))
    freeze_notebook(kernel, a.ref, config, a.ort_dataset)


if __name__ == "__main__":
    main()
