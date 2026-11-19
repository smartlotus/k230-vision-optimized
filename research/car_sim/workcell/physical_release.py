"""Step a stacked release with moving gripper fingers in the MuJoCo world."""

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from campaign_ledger import digest, read_json, write_json
from gripper_rig import ToolDimensions, tool_contract
from stack_contact import CHECK_WINDOW_S, HORIZON_S, TIMESTEP, scene as load_pair


OPENING_S = .110
WITHDRAWAL_S = .260


@dataclass(frozen=True)
class ReleaseSetup:
    lower_bottom_xyz_m: tuple
    held_bottom_xyz_m: tuple
    tcp_xyz_m: tuple
    tool_yaw_rad: float
    closed_gap_mm: float
    friction: float = .55


def yaw_quaternion(yaw):
    return (math.cos(yaw / 2), 0., 0., math.sin(yaw / 2))


def fixture_xml(tool):
    w, d, h = (getattr(tool, f"finger_{axis}_mm") / 2000 for axis in ("thickness", "depth", "height"))
    return "".join(f'<body name="{name}" mocap="true" pos="0 0 1">'
                   f'<geom type="box" size="{w} {d} {h}" friction="0.55 0.0001 0.00001"/>'
                   "</body>" for name in ("finger_left", "finger_right"))


def withdraw_positions(setup, tool, now_s):
    progress = min(1., now_s / OPENING_S)
    gap = setup.closed_gap_mm + progress * (tool.open_gap_mm - setup.closed_gap_mm)
    withdraw = min(1., max(0., (now_s - OPENING_S) / WITHDRAWAL_S))
    x, y, z = setup.tcp_xyz_m
    c, s = math.cos(setup.tool_yaw_rad), math.sin(setup.tool_yaw_rad)
    for sign in (-1., 1.):
        local_x = sign * (gap + tool.finger_thickness_mm) / 2000
        local_y, lifted = -.024 * withdraw, .032 * withdraw
        yield (x + c * local_x - s * local_y, y + s * local_x + c * local_y, z + lifted)


def probe(cad, setup, tool):
    if not tool.gap_min_mm <= setup.closed_gap_mm <= tool.open_gap_mm or not 0 < setup.friction <= 1:
        raise ValueError("Invalid release gap or friction")
    model, height = load_pair(cad, setup.friction, fixture_xml(tool))
    data = mujoco.MjData(model)
    lower, upper = (model.body(name).id for name in ("lower", "upper"))
    fingers = [model.body(name).id for name in ("finger_left", "finger_right")]
    for body, location in ((lower, setup.lower_bottom_xyz_m), (upper, setup.held_bottom_xyz_m)):
        data.qpos[model.jnt_qposadr[model.body_jntadr[body]]:][:7] = (*location, 1., 0., 0., 0.)
    ids = [model.body_mocapid[body] for body in fingers]
    data.mocap_quat[ids] = yaw_quaternion(setup.tool_yaw_rad)
    frame_count = round(HORIZON_S / TIMESTEP)
    sample_window = round(CHECK_WINDOW_S / TIMESTEP)
    samples, finger_steps, support_steps = [], 0, 0
    min_height, max_height = math.inf, -math.inf
    max_tilt, max_speed, max_spin = 0., 0., 0.
    for i in range(frame_count):
        now_s = i * TIMESTEP
        data.mocap_pos[ids] = tuple(withdraw_positions(setup, tool, now_s))
        mujoco.mj_step(model, data)
        contacts = [
            {int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])}
            for c in data.contact if c.efc_address >= 0
        ]
        tool_contact = any(upper in pair and any(finger in pair for finger in fingers) for pair in contacts)
        supported = {lower, upper} in contacts
        finger_steps += int(tool_contact)
        if i % 50 == 0 or i == frame_count - 1:
            samples.append({"time_s": now_s, "support_contact": supported, "finger_contact": tool_contact,
                            "lower_pose_xyz_quat": list(map(float, data.qpos[:7])),
                            "upper_pose_xyz_quat": list(map(float, data.qpos[7:])),
                            "finger_xyz": data.mocap_pos[ids].tolist()})
        if i >= frame_count - sample_window:
            support_steps += int(supported)
            separation = 1000 * float(data.xpos[upper, 2] - data.xpos[lower, 2])
            min_height, max_height = min(min_height, separation), max(max_height, separation)
            max_tilt = max(max_tilt, float(np.rad2deg(np.arccos(np.clip(data.xmat[[lower, upper], 8], -1, 1))).max()))
            speed = data.qvel.reshape(2, 6)
            max_speed = max(max_speed, float(np.linalg.norm(speed[:, :3], axis=1).max()) * 1000)
            max_spin = max(max_spin, math.degrees(float(np.linalg.norm(speed[:, 3:], axis=1).max())))
    stable = (support_steps == sample_window and max_tilt <= 5 and max_speed <= 2 and max_spin <= 3
              and height * 1000 - 2 <= min_height <= max_height <= height * 1000 + 2)
    return {
        "setup": asdict(setup), "stable": stable, "contact_fraction_final": support_steps / sample_window,
        "finger_contact_steps": finger_steps, "final_min_max_height_mm": [min_height, max_height],
        "final_max_tilt_deg": max_tilt, "final_max_speed_mm_s": max_speed,
        "final_max_spin_deg_s": max_spin, "lower_pose_xyz_quat": samples[-1]["lower_pose_xyz_quat"],
        "upper_pose_xyz_quat": samples[-1]["upper_pose_xyz_quat"], "trace": samples,
        "geometry_scope": "Rigid CAD proxy plus opening and withdrawing kinematic finger contacts",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cad-receipt", type=Path, required=True)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cad = read_json(args.cad_receipt)
    tool = ToolDimensions(**tool_contract()["nominal"])
    if args.case:
        setup = ReleaseSetup(**read_json(args.case))
        result = {"release": probe(cad, setup, tool)}
    else:
        controls = []
        for dx_mm in (0, 6, 25):
            setup = ReleaseSetup((0., 0., 0.), (dx_mm / 1000, 0., .061),
                                 (dx_mm / 1000, 0., .108), 0., 34.)
            row = probe(cad, setup, tool)
            row["expected_stable"] = dx_mm < 15
            controls.append(row)
        result = {"controls": controls, "gate_pass": all(r["stable"] == r["expected_stable"] for r in controls)}
    write_json(args.out, {**result, "cad_sha256": digest(args.cad_receipt),
                          "mujoco_version": mujoco.__version__})


if __name__ == "__main__":
    main()
