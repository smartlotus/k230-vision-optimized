"""Executed by Kaggle from the exact prepared version with injected CONFIG."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import zipfile


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def metadata(path):
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}


def verified_inventory(document, cfg):
    if hashlib.sha256(document).hexdigest() != cfg["input_inventory_sha256"]:
        raise ValueError("Kaggle mounted inventory differs from prepared bytes")
    inventory = json.loads(document)
    for name in inventory:
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or "\\" in name:
            raise ValueError("Unsafe mounted input path")
    if sum(item["bytes"] for item in inventory.values()) > cfg["input_bytes"]:
        raise ValueError("Mounted inventory exceeds input budget")
    return inventory


def accept_input(root, name, info, source):
    path = root / PurePosixPath(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        shutil.copyfileobj(source, target)
    if path.stat().st_size != info["bytes"] or sha(path) != info["sha256"]:
        raise ValueError(f"Input member drift: {name}")


def main(cfg):
    work = Path("/kaggle/working")
    assert work.is_dir()
    if shutil.disk_usage(work).free < 3 * cfg["input_bytes"] + 200_000_000:
        raise RuntimeError("Insufficient Kaggle run volume")
    inputs = Path("/kaggle/input")
    dependencies = Path(tempfile.mkdtemp(prefix="camera_xy_deps_"))
    wheel_spec = cfg["onnxruntime_wheel"]
    wheels = list(inputs.rglob(wheel_spec["name"]))
    if len(wheels) != 1 or metadata(wheels[0]) != wheel_spec:
        raise ValueError("Kaggle ORT wheel mount does not match PyPI bytes")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                    "--disable-pip-version-check", "--target", str(dependencies), str(wheels[0])],
                   check=True)
    sys.path.insert(0, str(dependencies))
    import onnxruntime
    import torch
    if cfg["device"] == "cuda":
        assert torch.cuda.is_available(), "Kaggle GPU was not assigned"
        assert torch.mm(torch.ones((2, 2), device="cuda"), torch.ones((2, 2), device="cuda")).sum().item() == 8
    temp_root = Path(tempfile.gettempdir())
    temp_free = shutil.disk_usage(temp_root).free
    if temp_free < cfg["input_bytes"] + 100_000_000:
        raise RuntimeError("Insufficient space on the actual Kaggle input staging volume")
    root = Path(tempfile.mkdtemp(prefix="camera_xy_run_", dir=temp_root))
    archive_name = cfg.get("archive_name", "camera_xy.zip")
    if archive_name not in ("camera_xy.zip", "camera_xy.bin"):
        raise ValueError("Unknown immutable dataset container")
    archives = list(inputs.rglob(archive_name))
    inventories = list(inputs.rglob("INPUT_INVENTORY.json"))
    if len(archives) == 1 and not inventories:
        if archives[0].stat().st_size != cfg["input_bytes"] or sha(archives[0]) != cfg["input_sha256"]:
            raise ValueError("Kaggle mounted archive differs from frozen bytes")
        with zipfile.ZipFile(archives[0]) as package:
            document = package.read("INPUT_INVENTORY.json")
            inventory = verified_inventory(document, cfg)
            if (len(package.namelist()) != len(inventory) + 1 or
                    set(package.namelist()) != set(inventory) | {"INPUT_INVENTORY.json"}):
                raise ValueError("Unexpected input archive members")
            for name, info in inventory.items():
                if package.getinfo(name).file_size != info["bytes"]:
                    raise ValueError("Unexpected archived member size")
                with package.open(name) as source:
                    accept_input(root, name, info, source)
    elif len(inventories) == 1 and not archives:
        # Kaggle unpacks uploaded nested ZIPs while creating a Dataset.
        manifest = inventories[0]
        inventory = verified_inventory(manifest.read_bytes(), cfg)
        for name, info in inventory.items():
            with (manifest.parent / name).open("rb") as source:
                accept_input(root, name, info, source)
    else:
        raise ValueError("Cannot identify a unique frozen CameraXY input")
    if sha(root / "data/manifest.jsonl") != cfg["manifest_sha256"]:
        raise ValueError("Unexpected dataset")
    recipe = cfg.get("recipe", "pilot-v1")
    trainer = {"pilot-v1": "train.py", "xy-map-v1": "train_map.py"}[recipe]
    # Keep epoch checkpoints as saved working files during the larger run.
    outputs = work / "progress" if recipe == "xy-map-v1" else root / "outputs"
    log = root / "train.log"
    args = [sys.executable, "-u", str(root / trainer), "--data", str(root / "data"),
            "--out", str(outputs), "--layout", str(root / "layout.json"), "--roadnet",
            str(root / "roadnet.onnx"), "--seed", str(cfg["seed"]), "--epochs",
            str(cfg["epochs"]), "--device", cfg["device"]]
    print(json.dumps({"run_id": cfg["run_id"], "input_sha256": cfg["input_sha256"],
                      "manifest_sha256": cfg["manifest_sha256"], "onnxruntime": onnxruntime.__version__,
                      "staging_free_bytes_before": temp_free}), flush=True)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 2
    witness = onnxruntime.InferenceSession(str(root / "roadnet.onnx"), sess_options=options,
                                           providers=["CPUExecutionProvider"])
    import numpy as np
    if witness.run(None, {"rgb": np.zeros((1, 3, 192, 320), dtype=np.float32)})[0].shape != (1, 7, 48, 80):
        raise ValueError("ORT runtime failed RoadNet shape witness")
    del witness
    env = dict(os.environ, PYTHONPATH=str(dependencies))
    with log.open("x", encoding="utf8") as f, subprocess.Popen(
            args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf8") as proc:
        for line in proc.stdout:
            f.write(line)
            print(line, end="", flush=True)
        exit_code = proc.wait()
    if exit_code != 0:
        raise RuntimeError(f"CameraXY workload failed ({exit_code})")
    shutil.copyfile(log, outputs / "train.log")
    run = dict(cfg, independent_replay=False)
    (outputs / "run.json").write_text(json.dumps(run, indent=2), encoding="utf8")
    members = [metadata(path) for path in sorted(outputs.iterdir())]
    source = work / "archive.pending"
    with zipfile.ZipFile(source, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as package:
        for item in members:
            package.write(outputs / item["name"], item["name"])
        package.writestr("ARTIFACT_INVENTORY.json", json.dumps(members, sort_keys=True))
    with zipfile.ZipFile(source) as package:
        for item in members:
            b = package.read(item["name"])
            assert len(b) == item["bytes"] and hashlib.sha256(b).hexdigest() == item["sha256"]
    assert source.stat().st_size < 64 * 1024 * 1024
    archive = work / (cfg["run_id"] + "-" + sha(source)[:16] + ".zip")
    source.rename(archive)
    small = ("result.json", "metrics.json", "environment.json", "ambiguity_pairs.json", "train.log")
    for name in small:
        shutil.copyfile(outputs / name, work / name)
    receipt = dict(schema="camera-xy-kaggle-compute-v1", status="COMPUTE_COMPLETE",
                   run_id=cfg["run_id"], input_sha256=cfg["input_sha256"],
                   dataset_manifest_sha256=cfg["manifest_sha256"], archive=metadata(archive),
                   input_inventory_sha256=cfg["input_inventory_sha256"],
                   compact_results=[metadata(work / name) for name in small],
                   automatic_replay=False)
    tmp = work / "RECEIPT.tmp"
    tmp.write_text(json.dumps(receipt, indent=2), encoding="utf8")
    tmp.rename(work / "RECEIPT.json")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main(CONFIG)
