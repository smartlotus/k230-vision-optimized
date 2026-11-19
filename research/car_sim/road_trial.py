"""Blender road camera and kinematic plant for paired development rollouts."""

import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actuator_loop import ActuatorLoop, decode, encode
from build_scene import prepare_round, shoot, write_json
from gripper_rig import JointPose, ToolDimensions, set_joint_pose, tool_contract


SIM = Path(__file__).resolve().parent
DT_S = .12
BASE_FORWARD_MPS = .17
START_Y_M = .23
END_Y_M = .55
MAX_STEPS = 21


def sample_frame(scene, rig, cam, path, camera_id, frame_id, stamp_ms, at_travel=True):
    shoot(scene, cam, path)
    return {
        "image": str(path), "frame_id": frame_id, "captured_ms": stamp_ms,
        "camera_id": camera_id, "rig_at_travel": at_travel, "is_simulated": True,
        "camera_from_body": [list(row) for row in
                             (cam.matrix_world.inverted() @ rig["root"].matrix_world)],
        "focal_px_640": cam.data.lens * 640 / cam.data.sensor_width,
    }


def set_body(rig, x, y, yaw):
    body = rig["root"]
    body.location = (x, y, 0)
    body.rotation_euler.z = yaw
    bpy.context.view_layer.update()


def cases(scene, rig, front, out, ask):
    cam = rig["camera"]
    nominal = scene.view_settings.exposure
    rows = []
    near = False
    for frame_id, (name, pose, age, exposure, expected) in enumerate((
        ("nominal", (0, .35, 0), 0, 0, "CLEAR"),
        ("right-near", (.04, .35, 0), 0, 0, "NEAR"),
        ("right-still-near", (.028, .35, 0), 0, 0, "NEAR"),
        ("recovered", (.005, .35, 0), 0, 0, "CLEAR"),
        ("left-away", (-.04, .35, 0), 0, 0, "CLEAR"),
        ("crossing", (0, 0, 0), 0, 0, "UNKNOWN"),
        ("wrong-camera", (0, .35, 0), 0, 0, "UNKNOWN"),
        ("expired", (0, .35, 0), 240, 0, "UNKNOWN"),
        ("shadow", (0, .35, 0), 0, -.37, "CLEAR"),
        ("occluded", (0, .35, 0), 0, 0, "UNKNOWN"),
    ), 1):
        set_body(rig, *pose)
        scene.view_settings.exposure = nominal + exposure
        sensor = front if name == "wrong-camera" else cam
        blocker = None
        if name == "occluded":
            location = cam.matrix_world.translation.copy()
            location.z -= .018
            bpy.ops.mesh.primitive_cube_add(size=.14, location=location)
            blocker = bpy.context.object
            blocker.scale.z = .03
            material = bpy.data.materials.new("Camera test cover")
            material.diffuse_color = (.002, .002, .002, 1.0)
            material.use_nodes = True
            next(node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeBsdfPrincipled").inputs[
                "Base Color"
            ].default_value = (.002, .002, .002, 1.0)
            blocker.data.materials.append(material)
        sample = sample_frame(scene, rig, sensor, out / f"{name}.png",
                              "QR_FRONT" if sensor == front else "ROAD_SIDE", frame_id, frame_id * 400)
        result = ask({
            **sample, "received_ms": sample["captured_ms"] + age, "previously_near": near,
            "prediction_path": str(out / f"{name}-labels.png"),
        })
        near = result["status"] == "NEAR"
        rows.append({"name": name, "pose": list(pose), "expected": expected,
                     "observation": result, "capture": sample,
                     "rgb": f"{name}.png", "labels": f"{name}-labels.png"})
        if blocker is not None:
            bpy.data.objects.remove(blocker, do_unlink=True)
    write_json(out / "road_observations.json", rows)


def drive_pair(scene, rig, out, ask, case):
    cam = rig["camera"]
    traces = {}
    for tag in ("open", "closed"):
        folder = out / tag
        folder.mkdir()
        dropped = set(case.get("drop_frames", []))
        seed = random.Random(case["seed"])
        x = case["offset_mm"] / 1000
        y = START_Y_M
        yaw = math.radians(case["yaw_deg"])
        traces[tag] = []
        history = []
        near = False
        controller = ActuatorLoop()
        for frame_id in range(MAX_STEPS):
            now_ms = round(frame_id * DT_S * 1000)
            set_body(rig, x, y, yaw)
            actual_pose = [x, y, yaw]
            # The controller receives only a noisy progress estimate in this segment.
            measured_y = y + seed.uniform(-.0015, .0015)
            at_stop = measured_y >= END_Y_M - .004
            remaining = max(0., END_Y_M - measured_y)
            forward_speed = min(BASE_FORWARD_MPS, remaining / DT_S)
            if frame_id not in dropped:
                image = folder / f"{frame_id:03d}.png"
                history.append(sample_frame(scene, rig, cam, image, "ROAD_SIDE", frame_id, now_ms))
            index = max(0, len(history) - 1 - case.get("delay_frames", 0))
            sample = history[index]
            message = {**sample, "received_ms": now_ms, "previously_near": near, "forward_mps": forward_speed}
            result = ask(message)
            near = result["status"] == "NEAR"
            command = result["command"] if tag == "closed" else {
                "vx_mps": 0., "vy_mps": forward_speed, "yaw_rate_rad_s": 0., "mode": "OPEN_REFERENCE",
            }
            if at_stop:
                command = {"vx_mps": 0., "vy_mps": 0., "yaw_rate_rad_s": 0., "mode": "STOP"}
            action_id = f"{tag}-{frame_id}"
            wire = encode({"type": "COMMAND", "action_id": action_id, "axis": "BASE",
                           "command": "BASE_VELOCITY", "sequence": frame_id,
                           "observation_id": sample["frame_id"], "issued_ms": now_ms,
                           "valid_until_ms": now_ms + 120,
                           "values": {key: command[key] for key in ("vx_mps", "vy_mps", "yaw_rate_rad_s")}})

            def execute(values):
                nonlocal x, y, yaw
                vx, vy, rate = (values[key] for key in ("vx_mps", "vy_mps", "yaw_rate_rad_s"))
                slip = case.get("lateral_slip_mm_per_s", 0) * .001 if vy else 0
                gain = case.get("lateral_response", 1.)
                x += (math.cos(yaw) * vx * gain - math.sin(yaw) * vy + slip) * DT_S
                y += (math.sin(yaw) * vx * gain + math.cos(yaw) * vy) * DT_S
                yaw += rate * DT_S
                return {"moving": bool(vx or vy or rate), "applied": values}

            receipts = controller.receive(wire, now_ms, execute)
            statuses = [decode(receipt, "RECEIPT") for receipt in receipts]
            if [item["status"] for item in statuses] != ["ACK", "DONE"]:
                raise RuntimeError("Expected a bounded virtual drive action")
            relative_rgb = Path(sample["image"]).relative_to(out).as_posix()
            traces[tag].append({
                "frame_id": frame_id, "action_id": action_id,
                "observed_frame_id": sample["frame_id"], "observation_age_ms": now_ms - sample["captured_ms"],
                "source_rgb": relative_rgb, "rendered_this_step": frame_id not in dropped,
                "capture": {**sample, "image": relative_rgb}, "observed_odometry_y_m": measured_y,
                "pose_at_exposure_m_rad": actual_pose, "integrated_pose_m_rad": [x, y, yaw],
                "road": result, "command": command, "dt_s": DT_S,
                "wire": wire.decode("ascii").strip(), "receipts": statuses,
            })
            if at_stop:
                break
    write_json(out / "road_drive.json", {"case": case, "traces": traces,
                                         "stop_target_y_m": END_Y_M, "sim_step_s": DT_S})


def run(args):
    dimensions = ToolDimensions(**tool_contract()["nominal"])
    scene, rig, front, _, blocks, _, _, _, _ = prepare_round(args.hub, gripper=dimensions)
    scene.frame_set(1)
    for obj in (rig["root"], rig["turret"], rig["lift"], *blocks.values()):
        obj.animation_data_clear()
    for obj in blocks.values():
        obj.hide_render = obj.hide_viewport = True
    # Clear the ground view below the lens before moving the chassis.
    pose = JointPose(-90, 400, dimensions.slide_max_mm, dimensions.gap_min_mm)
    set_joint_pose(rig, pose)
    command = [str(args.python), "-B", "-u", str(SIM / "road_follow.py"), "--hub", str(args.hub)]
    with (args.out / "inference.log").open("wb") as log:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=log, text=True, encoding="utf-8") as process:
            def ask(request):
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
                response = process.stdout.readline()
                if not response:
                    raise RuntimeError("RoadNet host worker exited")
                return json.loads(response)

            try:
                if args.mode == "observe":
                    cases(scene, rig, front, args.out, ask)
                else:
                    case = json.loads(args.case.read_text(encoding="utf-8"))
                    drive_pair(scene, rig, args.out, ask, case)
            finally:
                process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("RoadNet host worker failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--mode", choices=("observe", "drive"), required=True)
    parser.add_argument("--case", type=Path)
    run(parser.parse_args(sys.argv[sys.argv.index("--") + 1:]))
