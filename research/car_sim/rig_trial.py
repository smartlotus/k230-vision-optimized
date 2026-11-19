"""Render and judge the three-axis R3 tool assembly in Blender."""

import argparse
from dataclasses import asdict, replace
import itertools
import random
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_scene import prepare_round, shoot, write_json
from gripper_rig import JointPose, ToolDimensions, feedback, set_joint_pose, tool_contract, validate_pose


def extents(rig):
    world_to_body = rig["root"].matrix_world.inverted()
    coordinates = [world_to_body @ obj.matrix_world @ Vector(corner)
                   for obj in rig["collection"].objects if obj.type == "MESH" for corner in obj.bound_box]
    mins, maxs = [[1000 * bound(point[axis] for point in coordinates) for axis in range(3)]
                  for bound in (min, max)]
    return {"min_mm": mins, "max_mm": maxs, "span_mm": [b - a for a, b in zip(mins, maxs)]}


def disk_samples(center):
    for x, y in itertools.product(range(-5, 6), repeat=2):
        if x * x + y * y <= 25:
            yield center + Vector((.003 * x, .003 * y, 0))


def count_occlusion(scene, cam, center, points):
    origin = cam.matrix_world.translation
    deps = bpy.context.evaluated_depsgraph_get()
    tool, other, outside = 0, 0, 0
    for dest in points:
        p = world_to_camera_view(scene, cam, dest)
        if p.z <= 0 or not (0 < p.x < 1 and 0 < p.y < 1):
            outside += 1
            continue
        ray = dest - origin
        hit, _, _, _, obj, _ = scene.ray_cast(deps, origin, ray, distance=ray.length - .0002)
        if hit:
            if obj.name.startswith("GRIPPER |"):
                tool += 1
            else:
                other += 1
    return {"occluded_by_gripper": tool, "occluded_other": other, "outside_image": outside,
            "samples": len(points), "ratio": tool / len(points), "center_world_m": list(center)}


def occlusion_witness(scene, cam, rig, nominal, block, out):
    bpy.context.view_layer.update()
    lens = cam.matrix_world.translation.copy()
    candidates = []
    for x, y in itertools.product(range(-80, 91, 10), range(-105, 106, 15)):
        center = Vector((lens.x + x / 1000, lens.y + y / 1000, .063))
        candidates.append((center, tuple(disk_samples(center))))
    ratings = [[] for _ in candidates]
    slides = (-25, -15, -5, 5, 15, 25, 35)
    for slide in slides:
        set_joint_pose(rig, replace(nominal, slide_mm=slide, gap_mm=16))
        bpy.context.view_layer.update()
        for index, (center, points) in enumerate(candidates):
            ratings[index].append(count_occlusion(scene, cam, center, points))
    for index, rows in enumerate(ratings):
        if any(row["occluded_other"] or row["outside_image"] for row in rows):
            continue
        clear = next((i for i, row in enumerate(rows) if row["ratio"] == 0), None)
        part = next((i for i, row in enumerate(rows) if .2 <= row["ratio"] <= .8), None)
        covered = next((i for i, row in enumerate(rows) if row["ratio"] == 1), None)
        if None in (clear, part, covered):
            continue
        center, points = candidates[index]
        block.location = (center.x, center.y, center.z - .060)
        block.hide_render = block.hide_viewport = False
        results = {}
        for label, slot in (("clear", clear), ("partial", part), ("covered", covered)):
            slide = slides[slot]
            set_joint_pose(rig, replace(nominal, slide_mm=slide, gap_mm=16))
            bpy.context.view_layer.update()
            observed = count_occlusion(scene, cam, center, points)
            shoot(scene, cam, out / f"occlusion-{label}.png")
            projection = world_to_camera_view(scene, cam, center)
            results[label] = {
                **observed, "rgb": f"occlusion-{label}.png", "slide_mm": slide,
                "target_uv": [640 * projection.x, 360 * (1 - projection.y)],
                "camera_world_matrix": [list(row) for row in cam.matrix_world],
            }
        return results
    return {}


def run(hub, out):
    contract = tool_contract()
    dimensions = ToolDimensions(**contract["nominal"])
    scene, rig, _, _, blocks, _, _, _, _ = prepare_round(hub, gripper=dimensions)
    scene.frame_set(1)
    root, turret, lift, cam, tcp = (rig[key] for key in ("root", "turret", "lift", "camera", "tcp"))
    for obj in (root, turret, lift, *blocks.values()):
        obj.animation_data_clear()
    for block in blocks.values():
        block.hide_render = block.hide_viewport = True
    root.location = (0, .4, 0)
    root.rotation_euler.z = 0
    nominal = JointPose(-90, 330, 0, 60)
    trace = {}
    rng = random.Random(3101)
    for key, pose in {
        "reference": nominal,
        "extended": replace(nominal, slide_mm=35),
        "lowered": replace(nominal, optical_height_mm=220),
        "rotated": replace(nominal, theta_deg=-55),
        "stowed": JointPose(90, 330, -25, 0),
        "minimum": JointPose(-90, 100, -25, 60),
        "drive": JointPose(-90, 400, -25, 60),
    }.items():
        set_joint_pose(rig, pose)
        bpy.context.view_layer.update()
        trace[key] = {
            "pose": asdict(pose), "camera": cam.matrix_world.copy(), "tcp": tcp.matrix_world.copy(),
            "turret_axis_m": list(turret.matrix_world.translation), "footprint": extents(rig),
            "jaw_center_spacing_mm": 1000 * (rig["fingers"][1].matrix_world.translation -
                                            rig["fingers"][0].matrix_world.translation).length,
            "feedback": [feedback(pose, mode, 1000, 45, dimensions, rng) for mode in ("exact", "preset")],
        }
    ref, ext, lower, rot, stowed, minimum, drive = (
        trace[key] for key in ("reference", "extended", "lowered", "rotated", "stowed", "minimum", "drive")
    )
    axis = Vector(ref["turret_axis_m"])
    rotation = Matrix.Rotation(35 * 3.141592653589793 / 180, 3, "Z")
    rotate = lambda p: axis + rotation @ (p - axis)
    vdiff = lambda a, b: (a.translation - b.translation)
    checks = {
        "slide_keeps_entire_camera_transform": ref["camera"] == ext["camera"],
        "slide_tcp_toward_body_right_exact_mm": (vdiff(ext["tcp"], ref["tcp"]) - Vector((.035, 0, 0))).length < 1e-6,
        "lift_matches_for_both_parts": all((vdiff(lower[key], ref[key]) - Vector((0, 0, -.110))).length < 1e-6
                                           for key in ("camera", "tcp")),
        "rotation_matches_shared_axis": all((rot[key].translation - rotate(ref[key].translation)).length < 1e-6
                                            for key in ("camera", "tcp")),
        "retracted_stowed_envelope": all(value <= limit for value, limit
                                        in zip(stowed["footprint"]["span_mm"], (300, 300, 400))),
        "nominal_fingers_clear_ground_at_low_stop": abs(minimum["tcp"].translation.z - .020) < 1e-6
                                                   and minimum["footprint"]["min_mm"][2] >= -.001,
        "drive_height": abs(drive["camera"].translation.z - .4) < 1e-6,
        "jaw_spacing_moves_about_fixed_center": abs(ref["jaw_center_spacing_mm"] - 64) < .001
                                                and abs(stowed["jaw_center_spacing_mm"] - 4) < .001,
        "preset_mode_has_no_continuous_joint_values": all("joints" not in value["feedback"][1] for value in trace.values()),
    }
    limits = []
    for dz, outward in itertools.product(*contract["sweep"].values()):
        alternate = replace(dimensions, tcp_below_lens_mm=dz, tcp_outward_mm=outward)
        for slide, z in itertools.product((-25, 35), (100, 120, 220, 285, 400)):
            pose = JointPose(-90, z, slide, 60)
            try:
                validate_pose(pose, alternate, rig["parameters"]["confirmed"], rig["parameters"]["assumed"])
            except ValueError as exc:
                issue = str(exc)
            else:
                issue = None
            limits.append({
                "tcp_offset_mm": [outward, -dz], "pose": asdict(pose),
                "fingertip_clearance_mm": z - dz - alternate.finger_height_mm / 2, "rejected": issue,
            })
    checks["unsafe_offset_range_limits_reject_ground_contact"] = all(
        bool(case["rejected"]) == (case["fingertip_clearance_mm"] < 0) for case in limits
    ) and any(case["rejected"] for case in limits)
    for row in trace.values():
        for key in ("camera", "tcp"):
            row[key] = [list(line) for line in row[key]]
    set_joint_pose(rig, nominal)
    witnesses = occlusion_witness(scene, cam, rig, nominal, blocks[1, 1], out)
    checks["same_target_clear_partial_covered"] = (
        bool(witnesses) and witnesses["clear"]["ratio"] == 0 and .2 <= witnesses["partial"]["ratio"] <= .8
        and witnesses["covered"]["ratio"] == 1
        and witnesses["clear"]["camera_world_matrix"] == witnesses["covered"]["camera_world_matrix"]
    )
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "rig-r3.blend"), compress=True)
    write_json(out / "rig.json", {
        "checks": checks, "trace": trace, "gripper_contract": contract, "limit_scan": limits,
        "occlusion": witnesses, "geometry_scope": "one frozen field and labeled gripper proxy",
        "render_device": scene.cycles.device, "gate_pass": all(checks.values()),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    run(args.hub, args.out)
