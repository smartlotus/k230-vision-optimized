"""R3 gripper geometry and simulated joint feedback, in millimetres."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


MM = .001


@dataclass(frozen=True)
class ToolDimensions:
    tcp_below_lens_mm: float
    tcp_outward_mm: float
    slide_min_mm: float
    slide_max_mm: float
    gap_min_mm: float
    open_gap_mm: float
    finger_thickness_mm: float
    finger_depth_mm: float
    finger_height_mm: float
    rail_x_offset_mm: float
    rail_top_below_lens_mm: float
    rail_length_mm: float


def tool_contract():
    return json.loads(Path(__file__).with_name("gripper_dimensions.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class JointPose:
    theta_deg: float
    optical_height_mm: float
    slide_mm: float
    gap_mm: float


HEIGHTS = {"FLOOR_CLEAR": 120.0, "PICK": 220.0, "STACK": 285.0,
           "STOW": 330.0, "TRAVEL": 400.0}
ANGLES = {"STOWED": 90.0, "SIDE": -90.0}
HEIGHT_TOLERANCE_MM = 3.0
SLIDE_TOLERANCE_MM = 2.0
ANGLE_TOLERANCE_DEG = .75


def validate_pose(pose, dimensions, confirmed, angles):
    ranges = (
        ("theta_deg", *sorted((angles["stow_yaw_deg"], angles["deploy_yaw_deg"]))),
        ("optical_height_mm", confirmed["optical_height_min_mm"], confirmed["optical_height_max_mm"]),
        ("slide_mm", dimensions.slide_min_mm, dimensions.slide_max_mm),
        ("gap_mm", dimensions.gap_min_mm, dimensions.open_gap_mm),
    )
    for field, low, high in ranges:
        actual = getattr(pose, field)
        if not math.isfinite(actual) or not low <= actual <= high:
            raise ValueError(f"{field} outside [{low}, {high}]: {actual}")
    tip_height = pose.optical_height_mm - dimensions.tcp_below_lens_mm - dimensions.finger_height_mm / 2
    if tip_height < 0:
        raise ValueError(f"Fingertip would penetrate the ground by {-tip_height:g} mm")


def add_gripper(builder, lift, camera, dimensions):
    camera_x, camera_y, _ = (p / MM for p in camera.location)
    rail_x = camera_x + dimensions.rail_x_offset_mm
    rail_z = -dimensions.rail_top_below_lens_mm - 5.0
    rail_end_y = camera_y + 3.0
    rail_y = rail_end_y - dimensions.rail_length_mm / 2
    builder.box("GRIPPER | rail support", lift, (rail_x, rail_y - 25, -6),
                (17, 14, 2 * (28 + 6)), builder.blue)
    builder.box("GRIPPER | rail cross bracket", lift, ((rail_x + camera_x) / 2, rail_y - 25, 28),
                (abs(rail_x - camera_x) + 15, 16, 8), builder.blue)
    rail = builder.box("GRIPPER | fixed rail", lift, (rail_x, rail_y, rail_z),
                       (12, dimensions.rail_length_mm, 10), builder.steel)
    slide = builder.obj("GRIPPER | slide", lift, (camera_x, camera_y, -dimensions.tcp_below_lens_mm))
    top = dimensions.tcp_below_lens_mm + rail_z
    builder.box("GRIPPER | sliding runner", slide,
                (dimensions.rail_x_offset_mm, -36, top),
                (9, 76, 6), builder.darkblue)
    builder.box("GRIPPER | crosshead", slide, (dimensions.rail_x_offset_mm / 2, 0, top - 7),
                (abs(dimensions.rail_x_offset_mm) + 10, 14, 8), builder.blue)
    fingers = [builder.box(f"GRIPPER | {'left' if sign < 0 else 'right'} finger", slide,
                           (0, 0, 0),
                           (dimensions.finger_thickness_mm, dimensions.finger_depth_mm,
                            dimensions.finger_height_mm), builder.black)
               for sign in (-1, 1)]
    tcp = builder.obj("GRIPPER | TCP", slide, (0, 0, 0))
    return {"tool_dimensions": dimensions, "rail": rail, "slide": slide,
            "fingers": fingers, "tcp": tcp}


def set_joint_pose(rig, pose):
    """Apply absolute simulated joints. Validate all axes before changing any."""
    dimensions = rig["tool_dimensions"]
    config = rig["parameters"]
    validate_pose(pose, dimensions, config["confirmed"], config["assumed"])
    rig["turret"].rotation_euler.z = math.radians(pose.theta_deg)
    rig["lift"].location.z = (pose.optical_height_mm - config["drawing_mm"]["chassis_top_height"]) * MM
    rig["slide"].location.y = rig["camera"].location.y + (dimensions.tcp_outward_mm + pose.slide_mm) * MM
    half_spacing = (pose.gap_mm + dimensions.finger_thickness_mm) * MM / 2
    for sign, finger in zip((-1, 1), rig["fingers"]):
        finger.location.x = sign * half_spacing


def feedback(pose, mode, capture_ms, delay_ms, dimensions, rng, moving=False):
    """Return only sensor channels available to the PC controller at exposure."""
    envelope = {"captured_ms": capture_ms, "received_ms": capture_ms + delay_ms,
                "is_simulated": True, "mode": mode, "moving": moving}
    if mode == "exact":
        measured = asdict(pose)
        for key, noise in (("theta_deg", .15), ("optical_height_mm", .5), ("slide_mm", .4)):
            measured[key] += rng.uniform(-noise, noise)
        return {**envelope, "joints": measured, "at_limit": None}
    if mode != "preset":
        raise ValueError(f"Unknown feedback mode: {mode}")
    slide_values = {"RETRACTED": dimensions.slide_min_mm, "WORK": 0.0,
                    "EXTENDED": dimensions.slide_max_mm}

    def classify(value, presets, tolerance):
        return next((name for name, reference in presets.items()
                     if abs(value - reference) <= tolerance), None)

    presets = {
        "theta": classify(pose.theta_deg, ANGLES, ANGLE_TOLERANCE_DEG),
        "height": classify(pose.optical_height_mm, HEIGHTS, HEIGHT_TOLERANCE_MM),
        "slide": classify(pose.slide_mm, slide_values, SLIDE_TOLERANCE_MM),
        "jaw": "OPEN" if pose.gap_mm == dimensions.open_gap_mm else
               "CLOSED" if pose.gap_mm == dimensions.gap_min_mm else None,
    }
    return {**envelope, "presets": presets, "homed": not moving and presets["theta"] == "STOWED"
            and presets["slide"] == "RETRACTED", "at_position": not moving and all(presets.values()),
            "at_limit": presets["slide"] in {"RETRACTED", "EXTENDED"}}
