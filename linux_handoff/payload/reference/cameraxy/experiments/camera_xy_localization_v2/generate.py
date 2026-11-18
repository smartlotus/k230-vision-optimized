"""Render resumable CameraXY shards, validate coordinates/labels, and freeze a dataset."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image

from contract import (COUNTS, FIELDS, PROFILE, TARGETS,
                      digest, load_rows, map_to_pixel, schedule, xy)

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def verify_frame(base, row, expected, run, classes):
    name = f'{expected["group"]}-{expected["variant"]}'
    match = (row["sample_id"], row["scene_id"], row["pose_group_id"], row["split"],
             row["spatial_block_id"], row["variant"], row["ambiguity_region"], xy(row))
    wanted = (name, name, expected["group"], expected["split"], expected["block"],
              expected["variant"], expected["region"], (expected["x"], expected["y"]))
    if set(row) != FIELDS or match != wanted:
        raise ValueError(f"Manifest contract or deterministic schedule drift at {name}")
    m = row["camera_metadata"]
    t = np.array(m["camera_to_world_4x4"])
    k = np.array(m["K"])
    if not (t.shape == (4, 4) and k.shape == (3, 3) and np.isfinite(t).all() and
            np.isfinite(k).all() and np.allclose(t[3], (0, 0, 0, 1)) and
            abs(np.linalg.det(t[:3, :3]) - 1) < 1e-6 and (t[2, 2] < -.99) and
            np.allclose(t[:2, 3], xy(row), rtol=0, atol=.001) and
            abs(t[2, 3] - m["height_mm"]) < .001 and
            0 <= m["projection_check_max_px"] < .005):
        raise ValueError(f"Projection/optical center contract failed at {name}")
    w, h = run["width"], run["width"] * 9 // 16
    if m["resolution"] != [w, h] or not np.allclose(k[:2, 2], [(w-1)/2, (h-1)/2]):
        raise ValueError(f"K/resolution mismatch at {name}")
    config = row["render_config"]
    if any(config.get(k) != v for k, v in run["render_inputs_sha256"].items()):
        raise ValueError(f"Source bytes drift at {name}")
    if config["lighting_id"] not in (("shadow",) if row["variant"] == 2 else PROFILE):
        raise ValueError(f"Appearance split drift at {name}")
    for field, directory, mode in (("image", "rgb", "RGB"), ("mask", "masks", "L")):
        expected_path = f"{directory}/{name}.png"
        path = base / expected_path
        if row[field] != expected_path or digest(path) != row[field + "_sha256"]:
            raise ValueError(f"Image checksum/path drift at {path}")
        with Image.open(path) as img:
            arr = np.array(img)
            if img.mode != mode or img.size != (w, h) or np.ptp(arr) == 0:
                # A valid pure-class mask can be constant (e.g. blue corners).
                if field == "image" or img.mode != mode or img.size != (w, h):
                    raise ValueError(f"Invalid or flat image at {path}")
            if field == "mask":
                ids = set(np.unique(arr).tolist())
                if not ids <= {0, 1, 2, 3, 4, 5, 6, 255}:
                    raise ValueError(f"Unknown RoadNet IDs at {path}")
                classes.update(ids)


def verify_frozen(path, stage):
    report = json.loads((path / "qa.json").read_text(encoding="utf8"))
    run = json.loads((path / "run.json").read_text(encoding="utf8"))
    if (report["status"], run["stage"], report["pose_groups"]) != ("PASS", stage, COUNTS[stage]):
        raise ValueError(f"{stage} predecessor QA failed")
    if digest(path / "manifest.jsonl") != report["manifest_sha256"]:
        raise ValueError("Predecessor manifest changed after QA")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=COUNTS)
    for key in ("blender", "blend", "roadnet", "cad", "out"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--seed", type=int)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--margin", type=float, default=50)
    p.add_argument("--shard", type=int, default=100)
    p.add_argument("--predecessor", type=Path)
    p.add_argument("--pilot-selection", type=Path)
    a = p.parse_args()
    if a.width < 160 or a.width % 16 or not 0 <= a.margin <= 150 or a.shard < 1:
        p.error("width >= 160 and divisible by 16, margin in [0, 150], positive shard size")
    stage = a.stage
    if stage != "smoke":
        if a.predecessor is None:
            p.error("pass the preceding accepted dataset directory")
        preceding = "smoke" if stage == "pilot" else "pilot"
        prev = verify_frozen(a.predecessor, preceding)
        if stage == "full":
            if a.pilot_selection is None:
                p.error("full render requires an accepted pilot experiment and method selection")
            selection = json.loads(a.pilot_selection.read_text(encoding="utf8"))
            if (selection["dataset_manifest_sha256"] != prev["manifest_sha256"]
                    or selection["pilot_qa_sha256"] != digest(a.predecessor / "qa.json")
                    or selection["synthetic_full_render"] != "PASS"
                    or not selection["selected_method"]):
                p.error("pilot selection does not authorize expansion for this dataset")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = {"scene_sha256": a.blend, "renderer_sha256": a.roadnet / "blender_scene.py",
               "layout_sha256": a.roadnet / "configs/field_layout.json", "cad_sha256": a.cad,
               "sampler_sha256": a.roadnet / "sim.py", "camera_xy_renderer_sha256": ROOT / "render.py",
               "camera_xy_contract_sha256": ROOT / "contract.py"}
    hashes = {name: digest(path) for name, path in sources.items()}
    pinned = json.loads((ROOT / "assets.json").read_text(encoding="utf8"))["asset_mix"]
    source_keys = dict(scene="roadnet_blend", renderer="roadnet_renderer", sampler="roadnet_sampler",
                       layout="roadnet_layout", cad="topface_glb")
    if any(hashes[name + "_sha256"] != pinned[key] for name, key in source_keys.items()):
        raise ValueError("Input geometry, CAD or RoadNet sources do not match G0 audit")
    seed = a.seed if a.seed is not None else 20261117 + 1009 * list(COUNTS).index(stage)
    run = dict(stage=stage, count=COUNTS[stage], seed=seed, width=a.width, margin=a.margin,
               shard=a.shard, render_inputs_sha256=hashes, blender_exe_sha256=digest(a.blender))
    if stage == "full":
        run["pilot_selection_sha256"] = digest(a.pilot_selection)
    record = out / "run.json"
    if record.exists() and json.loads(record.read_text(encoding="utf8")) != run:
        p.error("existing run parameters changed, choose a new output directory")
    write_json(record, run)
    layout = json.loads(sources["layout_sha256"].read_text(encoding="utf8"))
    planned = list(schedule(stage, layout, seed, a.margin, run["count"]))
    combined = []
    classes = set()
    for start in range(0, len(planned), a.shard):
        stop = min(start + a.shard, len(planned))
        part = out / f"shard-{start:05}"
        if not (part / "manifest.jsonl").is_file():
            if part.exists():
                # Preserve interrupted evidence and repeat only this partial shard.
                attempt = len(list(out.glob(part.name + "-interrupted-*")))
                part.rename(out / f"{part.name}-interrupted-{attempt}")
            log = out / f"{part.name}-attempt-{len(list(out.glob(part.name + '-attempt-*.log'))):02}.log"
            cmd = [str(a.blender.resolve()), "-b", str(a.blend.resolve()), "-t", "4",
                   "--python-exit-code", "1", "--python", str(ROOT / "render.py"), "--",
                   "--roadnet", str(a.roadnet.resolve()), "--cad", str(a.cad.resolve()),
                   "--out", str(part), "--stage", stage, "--count", str(run["count"]),
                   "--width", str(a.width), "--seed", str(seed), "--margin", str(a.margin),
                   "--start", str(start), "--stop", str(stop)]
            with log.open("xb") as stream:
                subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True)
        rows = load_rows(part / "manifest.jsonl")
        for row, item in zip(rows, planned[start:stop], strict=True):
            verify_frame(part, row, item, run, classes)
            for key in ("image", "mask"):
                row[key] = f"{part.name}/{row[key]}"
            combined.append(row)
        print(f"{stage}: checked {stop}/{len(planned)} frames", flush=True)

    static = out / "shard-00000/field_mask.png"
    with Image.open(static) as image:
        anchors = [(0, 0, 0), (0, 800, 0), (800, 0, 0), (425, 425, 1),
                   (-425, 425, 1), (425, -425, 1), (-425, -425, 1),
                   (1050, 1050, 3), (1050, -1050, 3), (-1050, 1050, 2), (0, 1300, 255)]
        for x, y, expected in anchors:
            u, v = map_to_pixel(x, y, *image.size, 3000)
            if image.getpixel((round(u), round(v))) != expected:
                raise ValueError(f"Rendered field orientation anchor failed: {x}, {y}")
    for key in ("field_mask.png", "field_preview.png"):
        shutil.copyfile(out / "shard-00000" / key, out / key)
    manifest = out / "manifest.jsonl"
    scratch = manifest.with_suffix(".tmp")
    with scratch.open("w", encoding="utf8") as stream:
        for row in combined:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    scratch.replace(manifest)
    groups = {row["pose_group_id"]: row for row in combined}
    if len(groups) != run["count"] or len({xy(row) for row in groups.values()}) != len(groups):
        raise ValueError("Duplicate camera positions")
    appearance_pairs = 0
    by_id = {row["sample_id"]: row for row in combined}
    for row in combined:
        if row["variant"] != 2:
            continue
        r = by_id[row["pose_group_id"] + "-0"]
        if r["mask_sha256"] != row["mask_sha256"] or r["camera_metadata"] != row["camera_metadata"]:
            raise ValueError("Paired appearance holdout changed geometry")
        appearance_pairs += 1
    report = dict(
        status="PASS", manifest_sha256=digest(manifest), field_mask_sha256=digest(out / "field_mask.png"),
        targets=TARGETS, pose_groups=len(groups), frames=len(combined),
        positions_by_split=Counter(r["split"] for r in groups.values()),
        frames_by_split=Counter(r["split"] for r in combined),
        region_positions=Counter(r["ambiguity_region"] for r in groups.values()),
        appearance_pairs=appearance_pairs, semantic_class_ids=sorted(classes),
        north_up_anchor_count=len(anchors), renderer_log_sha256={log.name:digest(log) for log in out.glob("*.log")},
        max_projection_error_px=max(r["camera_metadata"]["projection_check_max_px"] for r in combined),
    )
    write_json(out / "qa.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
