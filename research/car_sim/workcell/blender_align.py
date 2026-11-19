"""Camera-to-controller-to-tool alignment bench on the rendered material tray."""

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from actuator_loop import ActuatorLoop, decode, encode
from build_scene import prepare_round, shoot, write_json
from gripper_rig import JointPose, ToolDimensions, feedback, set_joint_pose, tool_contract
from workcell.alignment import AlignmentSession, NUDGE_DURATION_S
from workcell.geometry import Calibration, measured_pose, rotate


CASE_MATRIX = (
    {"name": "slide-out", "mode": "exact", "offset_mm": [8, 0], "color": 1, "expected": "ALIGNED"},
    {"name": "slide-in", "mode": "exact", "offset_mm": [-7, 0], "color": 1, "expected": "ALIGNED"},
    {"name": "cross-rail", "mode": "exact", "offset_mm": [0, 7], "color": 1, "expected": "ALIGNED"},
    {"name": "preset", "mode": "preset", "offset_mm": [-6, -4], "color": 1, "expected": "ALIGNED"},
    {"name": "wrong-color", "mode": "exact", "offset_mm": [1, 0], "color": 6, "expected": "HOLD"},
    {"name": "same-color-neighbor", "mode": "exact", "offset_mm": [3, 0], "color": 1,
     "neighbor": True, "expected": "HOLD"},
    {"name": "out-of-reach", "mode": "exact", "offset_mm": [38, 0], "color": 1, "expected": "HOLD",
     "expected_reason": "BOUNDED_REPARK_REQUIRED"},
)


def initial_case(scene, rig, blocks, case, dimensions):
    for obj in blocks.values():
        obj.hide_render = obj.hide_viewport = True
    root = rig["root"]
    root.location = (0, .893, 0)
    root.rotation_euler.z = math.pi / 2
    block = blocks[1, case["color"]]
    initial = JointPose(-90, 330, dimensions.slide_max_mm if case["mode"] == "preset" else 24,
                        dimensions.open_gap_mm)
    set_joint_pose(rig, initial)
    bpy.context.view_layer.update()
    table = scene.objects["Turntable"]
    table_top = table.location.z + table.dimensions.z / 2
    tcp = rig["tcp"].matrix_world.translation.copy()
    dx, dy = rotate(root.rotation_euler.z, *case["offset_mm"])
    block.location = (tcp.x + dx / 1000, tcp.y + dy / 1000, table_top)
    block.hide_render = block.hide_viewport = False
    if case.get("neighbor"):
        neighbor = blocks[2, case["color"]]
        neighbor.location = (block.location.x + .057, block.location.y, table_top)
        neighbor.hide_render = neighbor.hide_viewport = False
    bpy.context.view_layer.update()
    return initial, block, 1000 * (table_top + .06)


def case_report(scene, rig, blocks, case, dimensions, vision, out):
    folder = out / case["name"]
    folder.mkdir()
    pose, block, plane_mm = initial_case(scene, rig, blocks, case, dimensions)
    root, cam, tcp = (rig[key] for key in ("root", "camera", "tcp"))
    align = AlignmentSession(dimensions, plane_mm)
    device = ActuatorLoop()
    rng = random.Random(3500)
    ambient = scene.view_settings.exposure
    trace = []
    for index in range(14):
        stamp = 200 * (index + 1)
        sensor = feedback(pose, case["mode"], stamp, 45, dimensions, rng)
        sight_pose = measured_pose(sensor, dimensions)
        if sight_pose is None:
            raise RuntimeError("Fixture presented an inadmissible observation pose")
        taught = Calibration.from_vehicle().tcp_xy(sight_pose, dimensions)
        u, v = align.calibration.project(sight_pose, (*taught, plane_mm))
        window = [max(0, round(u - 112)), max(0, round(v - 104)),
                  min(640, round(u + 112)), min(360, round(v + 104))]
        rgb_path = folder / f"{index:02d}.png"
        scene.view_settings.exposure = ambient + (0., .4, -.35)[index % 3]
        shoot(scene, cam, rgb_path)
        detected = vision({"image": str(rgb_path), "description": ["RAW", "pick_top", 1], "window": window})
        detected.update(camera_id="ROAD_SIDE", frame_id=index + 1, captured_ms=stamp)
        decision = align.observe(detected, sensor, stamp + 45)
        tool_before = tcp.matrix_world.translation.copy()
        target_uv = world_to_camera_view(scene, cam, Vector((block.location.x, block.location.y, plane_mm / 1000)))
        truth_xy_error = 1000 * math.dist(tuple(tool_before)[:2], tuple(block.location)[:2])
        row = {
            "rgb": rgb_path.relative_to(out).as_posix(), "frame_id": index + 1,
            "frame_time_ms": stamp, "feedback": sensor, "detection": detected,
            "decision": decision, "truth": {"tcp_xy_error_mm": truth_xy_error,
                                            "center_uv": [640 * target_uv.x, 360 * (1 - target_uv.y)]},
        }
        trace.append(row)
        if decision["status"] == "ALIGNED":
            break
        if decision["status"] != "ADJUST":
            if index >= 3 and all(item["decision"]["status"] == "HOLD" for item in trace[-3:]):
                break
            continue
        action = {
            "type": "COMMAND", "axis": decision["axis"], "command": decision["command"],
            "action_id": f"{case['name']}-{index + 1}", "sequence": index + 1,
            "issued_ms": stamp + 45, "valid_until_ms": stamp + 140,
            "observation_id": index + 1, "values": decision["values"],
        }

        def apply(values):
            nonlocal pose
            if action["axis"] == "BASE":
                x, y = rotate(root.rotation_euler.z, values["vx_mps"], values["vy_mps"])
                root.location.x += x * NUDGE_DURATION_S
                root.location.y += y * NUDGE_DURATION_S
            else:
                pose = replace(pose, **values)
                set_joint_pose(rig, pose)
            bpy.context.view_layer.update()
            return {"moving": False, "axis": action["axis"], "accepted": True}

        wire = encode(action)
        row["action_wire"] = wire.decode("ascii").strip()
        row["receipts"] = [decode(item, "RECEIPT") for item in device.receive(wire, stamp + 45, apply)]
        row["post_action_pose"] = asdict(pose)
    scene.view_settings.exposure = ambient
    final = trace[-1]
    status = final["decision"]["status"]
    expected_aligned = case["expected"] == "ALIGNED"
    tolerance = final["truth"]["tcp_xy_error_mm"] <= 4.0
    checks = {
        "correct_terminal_behavior": status == case["expected"] and (not expected_aligned or tolerance),
        "observed_3_consecutive_close_frames": not expected_aligned or (
            len(trace) >= 3 and all(math.hypot(*row["decision"].get("tcp_error_px", (float("inf"), 0))) <= 2.5
                                    for row in trace[-3:])
        ),
        "wrong_color_never_moves": case["name"] != "wrong-color" or device.executed == 0,
        "matches_required_hold_reason": not case.get("expected_reason")
                                       or final["decision"]["reason"] == case["expected_reason"],
    }
    return {"case": case, "status": status, "checks": checks, "gate_pass": all(checks.values()),
            "frames": trace, "final_truth_error_mm": final["truth"]["tcp_xy_error_mm"]}


def run(args):
    dimensions = ToolDimensions(**tool_contract()["nominal"])
    scene, rig, _, _, blocks, _, _, _, _ = prepare_round(args.hub, gripper=dimensions)
    scene.frame_set(1)
    for obj in (rig["root"], rig["turret"], rig["lift"], *blocks.values()):
        obj.animation_data_clear()
    command = [str(args.python), "-B", "-u", str(SIM / "servo_trial.py"), "--hub", str(args.hub), "--worker"]
    outcomes = []
    with (args.out / "vision.log").open("wb") as log:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=log, text=True, encoding="utf-8") as proc:
            def vision(inputs):
                proc.stdin.write(json.dumps(inputs) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError("Frozen TopFace subprocess exited")
                return json.loads(line)

            try:
                for case in CASE_MATRIX:
                    if args.one is None or args.one == case["name"]:
                        outcomes.append(case_report(scene, rig, blocks, case, dimensions, vision, args.out))
            finally:
                proc.stdin.close()
            if proc.wait(timeout=15) != 0:
                raise RuntimeError("Frozen TopFace subprocess failed")
    write_json(args.out / "alignment.json", {
        "cases": outcomes, "gate_pass": all(case["gate_pass"] for case in outcomes) and bool(outcomes),
        "scope": "Image-driven workcell alignment in the fixed R2 tray; no grasp was issued",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--one", choices=[case["name"] for case in CASE_MATRIX])
    run(parser.parse_args(sys.argv[sys.argv.index("--") + 1:]))
