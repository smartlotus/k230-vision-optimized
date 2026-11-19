"""Camera/TCP geometry from the declared vehicle and current feedback."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

from gripper_rig import ANGLES, HEIGHTS, JointPose


SIM = Path(__file__).resolve().parents[1]


def rotate(theta, x, y):
    c, s = math.cos(theta), math.sin(theta)
    return c * x - s * y, s * x + c * y


@dataclass(frozen=True)
class Calibration:
    turret_x_mm: float
    turret_y_mm: float
    camera_local_x_mm: float
    camera_local_y_mm: float
    focal_px: float

    @classmethod
    def from_vehicle(cls):
        config = json.loads((SIM / "vehicle_dimensions.json").read_text(encoding="utf-8"))
        assumed, confirmed = config["assumed"], config["confirmed"]
        if confirmed["camera_tilt_from_down_deg"] != 0 or assumed["deploy_yaw_deg"] != -90:
            raise ValueError("Expected the declared vertical right-facing rig")
        side = assumed["turret_y"]
        reach = math.sqrt(confirmed["pivot_to_lens_horizontal_mm"] ** 2 - side ** 2)
        driving_x = config["drawing_mm"]["chassis_width"] / 2 + assumed["driving_overhang_mm"]
        focal = math.hypot(640, 360) / (2 * math.tan(math.radians(assumed["camera_diagonal_fov_deg"]) / 2))
        return cls(driving_x - reach, side, side, reach, focal)

    def lens_xy(self, pose):
        dx, dy = rotate(math.radians(pose.theta_deg), self.camera_local_x_mm, self.camera_local_y_mm)
        return self.turret_x_mm + dx, self.turret_y_mm + dy

    def tcp_xy(self, pose, dimensions):
        cx, cy = self.lens_xy(pose)
        dx, dy = rotate(math.radians(pose.theta_deg), 0, dimensions.tcp_outward_mm + pose.slide_mm)
        return cx + dx, cy + dy

    def project(self, pose, point):
        x, y, z = point
        depth = pose.optical_height_mm - z
        if depth <= 0:
            raise ValueError("Planar work surface is above the optical center")
        cx, cy = self.lens_xy(pose)
        u, v = rotate(-math.radians(pose.theta_deg), x - cx, y - cy)
        return 320 + self.focal_px * u / depth, 180 - self.focal_px * v / depth

    def unproject(self, pose, uv, plane_mm):
        depth = pose.optical_height_mm - plane_mm
        if depth <= 0:
            raise ValueError("Planar work surface is above the optical center")
        u, v = uv
        cx, cy = self.lens_xy(pose)
        x, y = rotate(math.radians(pose.theta_deg), (u - 320) * depth / self.focal_px,
                      (180 - v) * depth / self.focal_px)
        return cx + x, cy + y


def measured_pose(packet, dimensions):
    if packet["moving"]:
        return None
    if packet["mode"] == "exact":
        return JointPose(**packet["joints"])
    if packet["mode"] != "preset" or not packet["at_position"]:
        return None
    preset = packet["presets"]
    slide = {"RETRACTED": dimensions.slide_min_mm, "WORK": 0., "EXTENDED": dimensions.slide_max_mm}
    gap = {"OPEN": dimensions.open_gap_mm, "CLOSED": dimensions.gap_min_mm}
    return JointPose(ANGLES[preset["theta"]], HEIGHTS[preset["height"]], slide[preset["slide"]],
                     gap[preset["jaw"]])
