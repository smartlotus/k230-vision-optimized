"""Fresh RGB alignment with separate commands for the base, turret and slide."""

import math

from workcell.geometry import Calibration, measured_pose, rotate


ALIGN_LIMIT_PX = 2.5
REQUIRED_STABLE_FRAMES = 3
MAX_AGE_MS = 175
MAX_BASE_NUDGE_MM = 19.
NUDGE_DURATION_S = .12


def bound(value, low, high):
    return min(high, max(low, value))


class AlignmentSession:
    def __init__(self, dimensions, plane_mm, camera_id="ROAD_SIDE"):
        self.dimensions = dimensions
        self.plane_mm = plane_mm
        self.camera_id = camera_id
        self.calibration = Calibration.from_vehicle()
        self.last_frame = -1
        self.stable = 0
        self.remaining_base_mm = MAX_BASE_NUDGE_MM
        self.last_joint_stamp = -1

    def observe(self, image, sensor, received_ms):
        if image["camera_id"] != self.camera_id or image["frame_id"] <= self.last_frame:
            self.stable = 0
            return {"status": "HOLD", "reason": "CAMERA_OR_FRAME_ID"}
        self.last_frame = image["frame_id"]
        capture, joint_time = image["captured_ms"], sensor["captured_ms"]
        if (not 0 <= received_ms - capture <= MAX_AGE_MS or
                not 0 <= received_ms - sensor["received_ms"] <= MAX_AGE_MS or
                not 0 <= capture - joint_time <= MAX_AGE_MS or joint_time < self.last_joint_stamp):
            self.stable = 0
            return {"status": "HOLD", "reason": "EXPIRED_IMAGE_OR_JOINT_FEEDBACK"}
        self.last_joint_stamp = joint_time
        pose = measured_pose(sensor, self.dimensions)
        if pose is None:
            self.stable = 0
            return {"status": "HOLD", "reason": "JOINT_POSITION_UNKNOWN"}
        if image["status"] != "ACQUIRED" or image.get("center_xy") is None:
            self.stable = 0
            return {"status": "HOLD", "reason": image["reason"]}
        target_uv = tuple(image["center_xy"])
        target = self.calibration.unproject(pose, target_uv, self.plane_mm)
        tcp = self.calibration.tcp_xy(pose, self.dimensions)
        reference = self.calibration.project(pose, (*tcp, self.plane_mm))
        dx, dy = (target[i] - tcp[i] for i in range(2))
        cross, along = rotate(-math.radians(pose.theta_deg), dx, dy)
        pixels = (target_uv[0] - reference[0], reference[1] - target_uv[1])
        trace = {
            "tcp_ref_px": list(reference), "target_px": list(target_uv),
            "image_center_error_px": [target_uv[0] - 320, 180 - target_uv[1]],
            "tcp_error_px": list(pixels), "cross_rail_mm": cross, "along_rail_mm": along,
            "status": "OBSERVE", "reason": "WITHIN_PIXEL_BAND",
        }
        if math.hypot(*pixels) <= ALIGN_LIMIT_PX:
            self.stable += 1
            if self.stable >= REQUIRED_STABLE_FRAMES:
                trace["status"] = "ALIGNED"
            return trace
        self.stable = 0
        axis_action = self.choose_axis(pose, sensor["mode"], dx, dy, cross, along)
        if axis_action is None:
            return {**trace, "status": "HOLD", "reason": "BOUNDED_REPARK_REQUIRED"}
        axis, command, values = axis_action
        return {**trace, "status": "ADJUST", "reason": "FRESH_PIXEL_RESIDUAL", "axis": axis,
                "command": command, "values": values}

    def choose_axis(self, pose, mode, dx, dy, cross, along):
        if mode == "exact":
            # Positive yaw sends local +Y toward local -X. Cross-rail correction
            # therefore uses the opposite sign from the local X residual.
            radius = self.calibration.camera_local_y_mm + self.dimensions.tcp_outward_mm + pose.slide_mm
            yaw_step = bound(math.degrees(-cross / radius), -4, 4)
            candidate_yaw = bound(pose.theta_deg + yaw_step, -90, 90)
            if abs(cross) > 1.4 and abs(candidate_yaw - pose.theta_deg) > .40:
                return "TURRET", "TURRET_MOVE", {"theta_deg": candidate_yaw}
            if abs(along) > 1.25 and abs(cross) <= max(1.4, abs(along) / 2):
                position = bound(pose.slide_mm + bound(along, -8, 8),
                                 self.dimensions.slide_min_mm, self.dimensions.slide_max_mm)
                if abs(position - pose.slide_mm) >= .8:
                    return "SLIDE", "SLIDE_MOVE", {"slide_mm": position}
        distance = math.hypot(dx, dy)
        if distance > self.remaining_base_mm + 4 or self.remaining_base_mm < 1:
            return None
        request = min(4., self.remaining_base_mm, distance)
        velocity = request / distance / (1000 * NUDGE_DURATION_S)
        self.remaining_base_mm -= request
        return "BASE", "BASE_VELOCITY", {"vx_mps": dx * velocity, "vy_mps": dy * velocity,
                                         "yaw_rate_rad_s": 0.}
