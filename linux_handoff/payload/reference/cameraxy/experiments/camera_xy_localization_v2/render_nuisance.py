"""Render pairwise nuisance checks in the frozen RoadNet Blender scene."""
import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from contract import TARGETS, block_and_split, digest
from render import add_qr_board, apply, camera_row, existing_scene, render_record, use_cad

SEED = 391284
N_PER_REGION = 16
FACTOR = ("control", "yaw90", "yaw180", "appearance", "obstacle", "material", "station")


def challenge_positions(layout):
    rng = random.Random(SEED)
    center = layout["blue_start_zones"]["centers"][0][0] * 1000
    for region in ("intersection", "straight_x", "straight_y", "blue", "storage", "rough"):
        for i in range(N_PER_REGION):
            sign = 1 if i % 2 == 0 else -1
            if region == "intersection":
                x, y = rng.uniform(-120, 120), rng.uniform(-120, 120)
            elif region.startswith("straight"):
                along, across = sign * rng.uniform(470, 790), rng.uniform(-130, 130)
                x, y = (along, across) if region == "straight_x" else (across, along)
            elif region == "blue":
                x, y = center + rng.uniform(-65, 65), sign * (center + rng.uniform(-65, 65))
            elif region == "storage":
                x, y = rng.uniform(-1130, -1010), rng.uniform(-70, 70)
            else:
                x, y = rng.uniform(-70, 70), rng.uniform(-1130, -1010)
            x, y = round(x, 4), round(y, 4)
            block, _ = block_and_split(x, y)
            yield dict(group=f"stress-{region}-{i:02}", provenance=rng.randrange(2**31),
                       x=x, y=y, block=block, split="synthetic_challenge", variant=0, region=region)


def changed_rec(base, factor):
    rec = deepcopy(base)
    x, y, _ = rec["center"]
    extra_material = None
    if factor in ("yaw90", "yaw180"):
        rec["camera_yaw_deg"] = (rec["camera_yaw_deg"] + int(factor[3:])) % 360
    elif factor == "appearance":
        rec["lighting_id"] = "shadow"
        rec["palette_linear"] = [tuple(min(.99, c * f) for c, f in zip(color, (.80, .85, 1.16)))
                                 for color in rec["palette_linear"]]
    elif factor == "obstacle":
        rec["obstacle_instances"] = [dict(center_xy=[x-.05, y])]
    elif factor == "material":
        extra_material = [x-.05, y, .004]
    elif factor == "station":
        rec["station_jitter_m"] = [.04, -.04]
    return rec, extra_material


def frozen_identity(a, bpy):
    audit = json.loads((ROOT / "assets.json").read_text(encoding="utf8"))["asset_mix"]
    identity = dict(roadnet_blend=digest(bpy.data.filepath),
                    roadnet_renderer=digest(a.roadnet / "blender_scene.py"),
                    roadnet_sampler=digest(a.roadnet / "sim.py"),
                    roadnet_layout=digest(a.roadnet / "configs/field_layout.json"),
                    topface_glb=digest(a.cad))
    if identity != audit:
        raise ValueError("Challenge must use the same audited map and CAD")
    return dict(**identity, renderer_sha256=digest(ROOT / "render.py"), challenge_sha256=digest(__file__))


def main():
    import bpy

    parser = argparse.ArgumentParser()
    for key in ("roadnet", "cad", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    a = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    identity = frozen_identity(a, bpy)
    sys.path.insert(0, str(a.roadnet.resolve()))
    road = importlib.import_module("blender_scene")
    sim = importlib.import_module("sim")
    size = (320, 180)
    scene = existing_scene(road, size)
    # Blender writes these PNG text chunks even with the visible stamp off.
    scene.scene.render.use_stamp_date = False
    scene.scene.render.use_stamp_render_time = False
    scene.scene.render.use_stamp_filename = False
    use_cad(a.cad, scene.blocks)
    board = add_qr_board(road, scene)
    out = a.out.resolve()
    out.mkdir()
    for folder in ("rgb", "masks"):
        (out / folder).mkdir()
    with (out / "manifest.incomplete").open("x", encoding="utf8") as stream:
        for item in challenge_positions(sim.LAYOUT):
            base, seed, _ = render_record(item, sim)
            base.update(lighting_id="nominal", obstacle_instances=[], station_item_color=None,
                        station_jitter_m=[0., 0.])
            for i, factor in enumerate(FACTOR):
                if factor == "station" and item["region"] not in {"storage", "rough"}:
                    continue
                rec, extra_material = changed_rec(base, factor)
                name = f"{item['group']}-{factor}"
                rec.update(image_path=f"rgb/{name}.png", mask_path=f"masks/{name}.png")
                apply(road, scene, rec, size, board)
                if extra_material is not None:
                    obj = scene.blocks[3][0]
                    obj.hide_render = False
                    obj.location = extra_material
                    obj.data.materials[0] = scene.stock[0]
                row = camera_row(scene.scene, scene.camera, rec, dict(item, variant=i), seed,
                                 name, size, identity)
                road.render(scene, rec, out, size)
                row.update(factor=factor, control_id=f"{item['group']}-control",
                           extra_material_base_m=extra_material,
                           image_sha256=rec["image_sha256"], mask_sha256=rec["mask_sha256"])
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
            print(f"CAMXY_STRESS_COMPLETE {item['group']}", flush=True)
    (out / "manifest.incomplete").rename(out / "manifest.jsonl")
    print(f"CAMXY_STRESS_PASS groups={6*N_PER_REGION} targets={TARGETS}", flush=True)


if __name__ == "__main__":
    main()
