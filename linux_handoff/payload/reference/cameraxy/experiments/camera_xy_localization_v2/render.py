"""Apply CameraXY sampling to the existing RoadNet .blend and its two render passes."""
import argparse
import importlib
import json
import math
from pathlib import Path
import random
import sys
from collections import namedtuple

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from contract import PROFILE, TARGETS, digest, schedule


Parts = namedtuple("Parts", "scene labels camera mats stock table obstacles blocks key fill")


def existing_scene(renderer, size):
    scene = bpy.data.scenes["Scene"]
    labels = bpy.data.scenes["Semantic object IDs"]
    assert scene.camera == labels.camera
    objects = bpy.data.objects
    mats = bpy.data.materials
    renderer.configure(scene, size, "BLENDER_EEVEE")
    renderer.configure(labels, size, "BLENDER_WORKBENCH")
    scene.eevee.taa_render_samples = 4
    def saved_material(name, template, color):
        mat = mats.get(name)
        if mat is None:
            mat = mats[template].copy()
            mat.name = name
            mat.diffuse_color = (*color, 1)
            mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*color, 1)
        return mat

    palette = [saved_material(f"Class {i}", "Class 4", color)
               for i, color in enumerate(renderer.PALETTE)]
    stock = [saved_material(f"Stock {i}", "Stock 1", color)
             for i, color in enumerate(renderer.BLOCK_COLORS, 1)]
    return Parts(scene, labels, scene.camera, palette, stock, objects["Turntable"],
            [objects[f"Obstacle {i}"] for i in range(4)],
            [[objects[f"Block {i} / {j}"] for j in range(3)] for i in range(4)],
            objects["Area"], objects["Area.001"])


def use_cad(path, pieces):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = set(bpy.data.objects) - before
    model, = [o for o in imported if o.type == "MESH"]
    mesh = model.data.copy()
    bpy.context.view_layer.update()
    mesh.transform(Matrix.Rotation(-math.pi / 2, 4, "X") @ model.matrix_world)
    xyz = np.array([v.co[:] for v in mesh.vertices])
    extent = np.ptp(xyz, axis=0)
    if not np.allclose(extent, [.05, .05, .06], atol=.001):
        raise ValueError(f"Unexpected CAD unit/axis convention: {extent}")
    for q, expected in ((xyz[:, 2].min(), .05), (xyz[:, 2].max(), .03)):
        width = np.ptp(xyz[abs(xyz[:, 2] - q) < .0001, :2], axis=0)
        if not np.allclose(width, expected, atol=.001):
            raise ValueError(f"Incorrect CAD upright orientation: {width}")
    for v in mesh.vertices:
        v.co.z -= float(xyz[:, 2].min())
    for o in imported:
        bpy.data.objects.remove(o, do_unlink=True)
    for block in pieces:
        block[0].data = mesh.copy()
        block[0].data.materials.clear()
        block[0].data.materials.append(bpy.data.materials["Stock 1"])


def render_record(item, sim):
    variant = item["variant"]
    seed = item["provenance"] * 2 + variant % 2
    rng = random.Random(seed)
    pose_rng = random.Random(item["provenance"])
    x, y = item["x"] / 1000, item["y"] / 1000
    # Variant 2 is a paired lighting/material holdout with variant 0 pose and geometry.
    yaw = (pose_rng.uniform(0, 360) + 90 * (variant % 2) + rng.uniform(-20, 20)) % 360
    station = rng.choice(("storage", "processing"))
    period = rng.uniform(6, 10)
    rec = dict(
        center=[x, y, rng.uniform(.37, .43)],
        camera_yaw_deg=yaw, camera_pitch_deg=rng.uniform(-2, 2),
        camera_roll_deg=rng.uniform(-2, 2), diag_fov_deg=rng.uniform(92, 100),
        lighting_id=rng.choice(PROFILE), lamp_xy=[rng.uniform(-.9, .9) for _ in range(2)],
        obstacle_instances=[dict(center_xy=p) for p in sim.obstacles(rng, station, x, y)],
        turntable_height_m=rng.uniform(.08, .10),
        turntable_phase_rad=rng.uniform(0, math.tau),
        turntable_period_s=period, turntable_direction=rng.choice((-1, 1)),
        turntable_color_ids=rng.sample(range(6), 3),
        stratum=station, station_item_color=rng.choice([None, *range(6)]),
        palette_linear=[tuple(min(.99, c * rng.uniform(.9, 1.10)) for c in color)
                        for color in sim.PALETTE],
    )
    rec["turntable_xy_m"] = [rng.uniform(-.10, .10), rng.uniform(1.265, 1.275)]
    rec["station_jitter_m"] = [rng.uniform(-.02, .02) for _ in range(2)]
    rec["qr_board_center_m"] = [1.197, rng.uniform(-.10, .10), rng.uniform(.20, .42)]
    rec["material_yaws_rad"] = [rng.uniform(0, math.tau) for _ in range(4)]
    if variant == 2:
        rec["lighting_id"] = "shadow"
        rec["palette_linear"] = [tuple(min(.99, c * f) for c, f in zip(color, (.80, .85, 1.16)))
                                 for color in rec["palette_linear"]]
    name = f'{item["group"]}-{variant}'
    rec.update(image_path=f"rgb/{name}.png", mask_path=f"masks/{name}.png")
    return rec, seed, name


def add_qr_board(road, parts):
    # Figure 1 places the vertical board on the right edge. Payload content
    # is immaterial to localization, so this is an A4 geometry placeholder.
    board = road.rect("CameraXY A4 QR board", (0, 0), (.297, .21), 0)
    road.identify(board, 6, parts.mats[2])
    board.rotation_euler = (math.pi / 2, 0, math.pi / 2)
    square = road.rect("CameraXY QR placeholder", (0, 0), (.08, .08), .001)
    road.identify(square, 6, parts.mats[4])
    square.location.z = -.002  # inward-facing side
    square.parent = board
    return board


def apply(renderer, parts, rec, size, board):
    board.location = rec["qr_board_center_m"]
    base = renderer.LAYOUT["raw_material_turntable"]["center"]
    renderer.LAYOUT["raw_material_turntable"]["center"] = rec["turntable_xy_m"]
    try:
        renderer.apply_record(parts, rec, size)
    finally:
        renderer.LAYOUT["raw_material_turntable"]["center"] = base
    for i, block in enumerate(parts.blocks):
        if not block[0].hide_render:
            block[0].location.z -= .004
            block[0].rotation_euler.z = rec["material_yaws_rad"][i]
        for component in block[1:]:
            component.hide_render = True
    for zone, shift in zip(("temporary_storage_zone", "rough_processing_zone"),
                           rec["station_jitter_m"]):
        axis = "y" if zone.startswith("temporary") else "x"
        for suffix in ("", " rings"):
            setattr(bpy.data.objects[zone + suffix].location, axis, shift)
        if rec["station_item_color"] is not None and zone.startswith(
                "temporary" if rec["stratum"] == "storage" else "rough"):
            setattr(parts.blocks[3][0].location, axis,
                    getattr(parts.blocks[3][0].location, axis) + shift)


def camera_row(scene, camera, rec, item, seed, name, size, provenance):
    t = np.asarray(rec["camera_to_world_blender"], dtype=float)
    f = rec["intrinsics_pinhole"]
    k = [[f["fx"], 0., f["cx"]], [0., f["fy"], f["cy"]], [0., 0., 1.]]
    w, h = size
    cv_to_world = t @ np.diag([1, -1, -1, 1])
    world_to_cv = np.linalg.inv(cv_to_world)
    x, y, z = t[:3, 3]
    # Compare with Blender projection at several field points, including the
    # vertical projection of the optical center (tilt may move its pixel).
    errors = []
    for dx, dy in ((0, 0), (.10, 0), (0, .10), (-.10, -.06)):
        ground = (x + dx, y + dy, 0)
        xyz = (world_to_cv @ [*ground, 1])[:3]
        u, v, _ = np.asarray(k) @ (xyz / xyz[2])
        projected = world_to_camera_view(scene, camera, Vector(ground))
        errors.append(math.hypot(u - (w * projected.x - .5), v - (h * (1 - projected.y) - .5)))
    if max(errors) >= .005 or max(abs(1000*x-item["x"]), abs(1000*y-item["y"])) > .001:
        raise ValueError(f"Projection or camera optical center mismatch: {name}, {errors}")
    matrix = cv_to_world.copy()
    matrix[:3, 3] *= 1000
    meta = dict(
        camera_to_world_4x4=matrix.tolist(), matrix_convention="OpenCV_camera_to_FIELD_MAP_mm",
        resolution=list(size), K=k, distortion=[0.0] * 5,
        calibration_status="SYNTHETIC_PINHOLE_PRIOR_REAL_CALIBRATION_UNKNOWN",
        yaw_deg=rec["camera_yaw_deg"], pitch_deg=rec["camera_pitch_deg"],
        roll_deg=rec["camera_roll_deg"], height_mm=1000*z,
        orientation_convention="Blender XYZ Euler perturbation about nadir, local X/Y then world Z",
        projection_check_max_px=max(errors),
    )
    config = {k: rec[k] for k in (
        "lighting_id", "lamp_xy", "palette_linear", "diag_fov_deg", "obstacle_instances",
        "turntable_height_m", "turntable_phase_rad", "turntable_period_s", "turntable_direction",
        "turntable_color_ids", "turntable_xy_m", "station_item_color", "stratum",
        "station_jitter_m", "material_yaws_rad", "qr_board_center_m")}
    config.update(provenance, engine=scene.render.engine, render_samples=scene.eevee.taa_render_samples,
                  exposure=scene.view_settings.exposure, blender=bpy.app.version_string)
    # The exact sampled pair is shared across variants after sub-micron
    # verification against the matrix written by Blender's float32 transform.
    return dict(
        sample_id=name, scene_id=name, pose_group_id=item["group"], spatial_block_id=item["block"],
        split=item["split"], variant=item["variant"], seed=seed, ambiguity_region=item["region"],
        camera_xy_target=dict(zip(TARGETS, (item["x"], item["y"]))),
        camera_metadata=meta, render_config=config, image=rec["image_path"], mask=rec["mask_path"],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roadnet", type=Path, required=True)
    p.add_argument("--cad", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--margin", type=float, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--stop", type=int, required=True)
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])
    sys.path.insert(0, str(a.roadnet.resolve()))
    road = importlib.import_module("blender_scene")
    sim = importlib.import_module("sim")
    size = a.width, a.width * 9 // 16
    parts = existing_scene(road, size)
    use_cad(a.cad, parts.blocks)
    board = add_qr_board(road, parts)
    files = {"scene_sha256": Path(bpy.data.filepath), "renderer_sha256": a.roadnet / "blender_scene.py",
             "layout_sha256": a.roadnet / "configs/field_layout.json", "cad_sha256": a.cad,
             "sampler_sha256": a.roadnet / "sim.py", "camera_xy_renderer_sha256": Path(__file__),
             "camera_xy_contract_sha256": ROOT / "contract.py"}
    provenance = {name: digest(path) for name, path in files.items()}
    rows = list(schedule(a.stage, sim.LAYOUT, a.seed, a.margin, a.count))[a.start:a.stop]
    out = a.out.resolve()
    out.mkdir()
    for d in ("rgb", "masks"):
        (out / d).mkdir()
    with (out / "manifest.incomplete").open("x", encoding="utf8") as stream:
        for item in rows:
            rec, seed, name = render_record(item, sim)
            apply(road, parts, rec, size, board)
            row = camera_row(parts.scene, parts.camera, rec, item, seed, name, size, provenance)
            road.render(parts, rec, out, size)
            for key in ("image", "mask"):
                row[key + "_sha256"] = rec[key + "_sha256"]
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            print(f"CAMERA_XY_FRAME {name}", flush=True)
    if a.start == 0:
        camera = parts.camera
        camera.location, camera.rotation_euler = (0, 0, 4), (0, 0, 0)
        camera.data.type, camera.data.ortho_scale = "ORTHO", 3.0
        for obj in (parts.table, *parts.obstacles, *(o for block in parts.blocks for o in block),
                    board, *board.children):
            obj.hide_render = True
        for obj in bpy.data.objects:
            if "zone" in obj.name:
                obj.location.x = obj.location.y = 0.0
        for scene in (parts.scene, parts.labels):
            scene.render.resolution_x = scene.render.resolution_y = 800
        road.render(parts, dict(image_path="field_preview.png", mask_path="field_mask.png"),
                    out, (800, 800))
    (out / "manifest.incomplete").rename(out / "manifest.jsonl")


if __name__ == "__main__":
    main()
