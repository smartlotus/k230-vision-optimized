"""Render the RAW workcell, descend, grip and verify a bounded trial lift."""

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import bpy

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from actuator_loop import ActuatorLoop, decode, encode
from build_scene import prepare_round, shoot, write_json
from gripper_rig import JointPose, ToolDimensions, feedback, set_joint_pose, tool_contract
from workcell.geometry import Calibration, measured_pose
from workcell.kinematic_grasp import MaterialBody, opposed_contact, test_lift


SCENARIOS = {
    "trial-lift": "HELD_AFTER_TRIAL_LIFT",
    "wrong-color": "REJECT_HIGH_OBSERVATION",
    "empty-after-observation": "EMPTY_CONTACT",
    "slip": "SLIPPED_DURING_TRIAL_LIFT",
    "stale-low-observation": "REJECT_LOW_OBSERVATION",
    "lift-not-in-position": "LIFT_POSITION_UNCONFIRMED",
}
WORK_SLIDE_MM = 18.
APPROACH_HEIGHT_MM = 330.
GRASP_HEIGHT_MM = 225.


def evaluate(scene, rig, blocks, dimensions, cad, vision, out, name):
    directory = out / name
    directory.mkdir()
    block = blocks[1, 6 if name == "wrong-color" else 1]
    for item in blocks.values():
        item.hide_render = item.hide_viewport = item != block
    root, tcp, cam, turret = (rig[key] for key in ("root", "tcp", "camera", "turret"))
    root.location = (0., .924, 0.)
    root.rotation_euler.z = math.pi / 2
    table = scene.objects["Turntable"]
    table_top = table.location.z + table.dimensions.z / 2
    plane_mm = 1000 * (table_top + .06)
    expected_pose = JointPose(-90., APPROACH_HEIGHT_MM, WORK_SLIDE_MM, dimensions.open_gap_mm)
    # Fixture setup chooses a reachable tray stop. Controller inputs are RGB,
    # the taught fixture profile, and the independent joint feedback below.
    set_joint_pose(rig, expected_pose)
    bpy.context.view_layer.update()
    block.location = (tcp.matrix_world.translation.x, tcp.matrix_world.translation.y, table_top)
    pose = replace(expected_pose, slide_mm=dimensions.slide_min_mm)
    set_joint_pose(rig, pose)
    bpy.context.view_layer.update()
    runner = ActuatorLoop()
    calibration = Calibration.from_vehicle()
    trace, counter = [], 0
    rng = random.Random(3601)

    def observe(blend_name, limit_px, obsolete=False):
        nonlocal counter
        counter += 1
        captured = counter * 200
        sensor = feedback(pose, "exact", captured, 45, dimensions, rng)
        status = measured_pose(sensor, dimensions)
        reference = calibration.project(status, (*calibration.tcp_xy(status, dimensions), plane_mm))
        u, v = reference
        window = [max(0, round(u - 75)), max(0, round(v - 80)),
                  min(640, round(u + 75)), min(360, round(v + 80))]
        rgb = directory / f"{blend_name}.png"
        shoot(scene, cam, rgb)
        hit = vision({"image": str(rgb), "description": ["RAW", "pick_top", 1], "window": window})
        gap = math.dist(reference, hit["center_xy"]) if hit["center_xy"] else None
        age = 320 if obsolete else 45
        passed = hit["status"] == "ACQUIRED" and gap is not None and gap <= limit_px and age <= 175
        trace.append({"event": blend_name, "frame_id": counter, "rgb": rgb.relative_to(out).as_posix(),
                      "captured_ms": captured, "age_ms": age, "sensor": sensor,
                      "detection": hit, "pixel_residual": gap, "limit_px": limit_px, "accepted": passed})
        return counter if passed else None

    def action(axis, command, values, apply=None, observation=None):
        nonlocal counter
        counter += 1
        issued = 200 * counter if observation is None else 200 * observation + 45
        expires = issued + 120 if observation is None else 200 * observation + 175
        command_frame = encode({"type": "COMMAND", "action_id": f"{name}-{counter}", "axis": axis,
                                "command": command, "sequence": counter, "observation_id": observation,
                                "issued_ms": issued, "valid_until_ms": expires, "values": values})

        def move(proposed):
            nonlocal pose
            pose = replace(pose, **proposed)
            set_joint_pose(rig, pose)
            bpy.context.view_layer.update()
            return {"moving": False, "joint": asdict(pose), "confirmed": True}

        receipts = runner.receive(command_frame, issued + 20, apply or move)
        decoded = [decode(item, "RECEIPT") for item in receipts]
        trace.append({"event": command, "wire": command_frame.decode("ascii").strip(), "receipts": decoded})
        return decoded[-1]["feedback"]

    def outcome():
        nonlocal pose
        action("SLIDE", "SLIDE_MOVE", {"slide_mm": WORK_SLIDE_MM})
        approach = observe("approach", 6.5)
        if approach is None:
            return "REJECT_HIGH_OBSERVATION"
        no_position = (lambda _: {"moving": True, "confirmed": False}) if name == "lift-not-in-position" else None
        lowered = action("LIFT", "LIFT_MOVE", {"optical_height_mm": GRASP_HEIGHT_MM}, no_position,
                         observation=approach)
        if not lowered["confirmed"]:
            return "LIFT_POSITION_UNCONFIRMED"
        observed = observe("after-descent", 2.5, obsolete=name == "stale-low-observation")
        if observed is None:
            return "REJECT_LOW_OBSERVATION"
        if name == "empty-after-observation":
            block.location.x += .065
            bpy.context.view_layer.update()
            trace.append({"event": "TRAY_DISPLACEMENT_AFTER_EXPOSURE", "shift_mm": 65})
        before = tuple(1000 * v for v in tcp.matrix_world.translation)
        contact = None

        def close(values):
            nonlocal contact, pose
            material = MaterialBody(int(block["batch"]), int(block["color_id"]),
                                    tuple(1000 * v for v in block.location))
            angle = root.rotation_euler.z + turret.rotation_euler.z
            contact, reason = opposed_contact(before, angle, pose.gap_mm, values["gap_mm"], dimensions, cad, [material])
            actual_gap = contact.supported_diameter_mm if contact else values["gap_mm"]
            pose = replace(pose, gap_mm=actual_gap)
            set_joint_pose(rig, pose)
            bpy.context.view_layer.update()
            return {"clamped": contact is not None, "contact": reason, "gap_mm": actual_gap}

        clamped = action("JAW", "JAW_MOVE", {"gap_mm": 0.}, close, observation=observed)
        shoot(scene, cam, directory / "closed.png")
        if not clamped["clamped"]:
            return clamped["contact"]
        start_z = block.location.z

        def try_lift(values):
            nonlocal pose
            pose = replace(pose, **values)
            set_joint_pose(rig, pose)
            bpy.context.view_layer.update()
            after = tuple(1000 * val for val in tcp.matrix_world.translation)
            response = test_lift(contact, before, after, slips=name == "slip")
            if response["holding"]:
                block.location.z = start_z + response["lift_mm"] / 1000
            bpy.context.view_layer.update()
            return response

        lifted = action("LIFT", "LIFT_MOVE", {"optical_height_mm": GRASP_HEIGHT_MM + 10}, try_lift)
        shoot(scene, cam, directory / "after-lift.png")
        trace.append({"event": "INDEPENDENT_LIFT_WITNESS",
                      "block_height_change_mm": 1000 * (block.location.z - start_z),
                      "contact_identity": list(contact.identity)})
        return lifted["reason"]

    reason = outcome()
    off_limits = name in ("wrong-color", "stale-low-observation", "lift-not-in-position")
    closed = any(row["event"] == "JAW_MOVE" for row in trace)
    moved_mm = next((row["block_height_change_mm"] for row in reversed(trace)
                     if row["event"] == "INDEPENDENT_LIFT_WITNESS"), 0)
    expected_hold = name == "trial-lift"
    passed = reason == SCENARIOS[name] and (not off_limits or not closed) and (
        moved_mm > 9 if expected_hold else moved_mm == 0
    )
    return {"case": name, "reason": reason, "expected": SCENARIOS[name], "gate_pass": passed,
            "holding_after_trial": expected_hold and passed, "carrier_picks": 0, "trace": trace}


def run(args):
    dimensions = ToolDimensions(**tool_contract()["nominal"])
    scene, rig, _, _, blocks, _, _, _, _ = prepare_round(args.hub, gripper=dimensions)
    scene.frame_set(1)
    for obj in (rig["root"], rig["turret"], rig["lift"], *blocks.values()):
        obj.animation_data_clear()
    source = args.hub / ("webhelp/camera_xy_v2_audit_20261117/k230_material_block_model_bundle/"
                         "k230_material_block_model/model_validation.json")
    cad = json.loads(source.read_text(encoding="utf-8"))
    command = [str(args.python), "-B", "-u", str(SIM / "servo_trial.py"), "--worker", "--hub", str(args.hub)]
    scenarios = [args.one] if args.one else SCENARIOS
    with (args.out / "vision.log").open("wb") as log:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=log, text=True, encoding="utf-8") as proc:
            def vision(request):
                proc.stdin.write(json.dumps(request) + "\n")
                proc.stdin.flush()
                if not (row := proc.stdout.readline()):
                    raise RuntimeError("Frozen perception subprocess stopped")
                return json.loads(row)

            try:
                outcomes = [evaluate(scene, rig, blocks, dimensions, cad, vision, args.out, name)
                            for name in scenarios]
            finally:
                proc.stdin.close()
            if proc.wait(timeout=15):
                raise RuntimeError("Frozen perception subprocess failed")
    write_json(args.out / "grasp.json", {
        "cases": outcomes, "gate_pass": all(row["gate_pass"] for row in outcomes),
        "scope": "Rendered approach and geometric opposing-jaw contact, independent of selected color",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--one", choices=SCENARIOS)
    run(parser.parse_args(sys.argv[sys.argv.index("--") + 1:]))
