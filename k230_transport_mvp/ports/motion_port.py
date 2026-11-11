"""Motion receipts and a separate, default-disabled LIVE authorization gate.

This module never translates pixels into motion and never retries a submission.
An external controller owns motor interlocks and emergency stop hardware.
"""
import math
import time


def monotonic_ms():
    if hasattr(time, "ticks_ms"):
        return time.ticks_ms()
    return int(time.monotonic() * 1000)


def _elapsed(now, then):
    return time.ticks_diff(now, then) if hasattr(time, "ticks_diff") else now-then


class MotionReceipt(dict):
    def __init__(self, status, command_id, reason="", **details):
        if status not in ("ACK", "DONE", "BUSY", "ERROR", "UNKNOWN"):
            raise ValueError("invalid motion status")
        dict.__init__(self, status=status, command_id=command_id, reason=reason, **details)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class SafetyGate:
    def check_navigation(self, observation, now_ms, context, destination):
        """Authorize a separately verified route; QR supplies no mechanical pose."""
        if context.get("motion_enabled") is not True or context.get("motion_authorization") != "EXPLICIT_ENABLED":
            return False, "MOTION_NOT_AUTHORIZED"
        if context.get("navigation_authorization") != "EXPLICIT_ENABLED":
            return False, "NAVIGATION_NOT_AUTHORIZED"
        obs = observation or {}
        if obs.get("is_simulated") is not False or obs.get("status") != "OK" or obs.get("stable") is not True:
            return False, "NAVIGATION_REAL_STABLE_SOURCE_REQUIRED"
        if type(obs.get("frame_id")) is not int or not 0 <= obs["frame_id"] <= 4294967295:
            return False, "FRAME_ID_UNKNOWN"
        camera_id = obs.get("camera_id")
        if type(camera_id) not in (str, int) or (type(camera_id) is int and camera_id < 0) or (type(camera_id) is str and not camera_id):
            return False, "CAMERA_ID_UNKNOWN"
        if obs.get("kind") not in ("QR", "LANE", "COLOR", "RING", "TOP"):
            return False, "NAVIGATION_OBSERVATION_UNKNOWN"
        age_limit = context.get("max_observation_age_ms", 250)
        if type(age_limit) is not int or not 1 <= age_limit <= 5000:
            return False, "FRESHNESS_CONFIG_INVALID"
        for timestamp in (obs.get("timestamp_ms"), context.get("received_at_ms"), context.get("navigation_timestamp_ms")):
            if type(timestamp) is not int or not 0 <= _elapsed(now_ms, timestamp) <= age_limit:
                return False, "NAVIGATION_STATE_STALE_OR_UNKNOWN"
        binding = context.get("active_binding")
        if not isinstance(binding, dict):
            return False, "NAVIGATION_BINDING_UNKNOWN"
        for key in ("camera_id", "station", "image_profile"):
            if obs.get(key) is None or obs.get(key) != binding.get(key):
                return False, "NAVIGATION_BINDING_MISMATCH:" + key
        profile = obs.get("image_profile")
        if not isinstance(profile, dict) or not profile.get("pose_id"):
            return False, "PARKING_POSE_UNKNOWN"
        if context.get("current_pose_verified") is not True or not context.get("current_station") or context["current_station"] != obs.get("station"):
            return False, "NAVIGATION_CURRENT_POSE_UNKNOWN"
        if context.get("route_verified") is not True or context.get("route_destination") != destination:
            return False, "NAVIGATION_ROUTE_NOT_VERIFIED"
        if context.get("current_observation_identity") != (obs.get("camera_id"), obs.get("frame_id"), obs.get("target_id")):
            return False, "OBSERVATION_IDENTITY_MISMATCH"
        if context.get("lane_safe") is not True or context.get("obstacle_clear") is not True:
            return False, "PATH_CLEARANCE_UNKNOWN"
        return True, "NAVIGATION_AUTHORIZED"

    def check(self, observation, now_ms, context):
        if context.get("motion_enabled") is not True or context.get("motion_authorization") != "EXPLICIT_ENABLED":
            return False, "MOTION_NOT_AUTHORIZED"
        obs = observation or {}
        if obs.get("is_simulated") is not False:
            return False, "REAL_SOURCE_REQUIRED"
        if obs.get("status") != "OK" or obs.get("stable") is not True:
            return False, "TARGET_NOT_STABLE"
        if type(obs.get("frame_id")) is not int or not 0 <= obs["frame_id"] <= 4294967295:
            return False, "FRAME_ID_UNKNOWN"
        camera_id = obs.get("camera_id")
        if type(camera_id) not in (str, int) or (type(camera_id) is int and camera_id < 0) or (type(camera_id) is str and not camera_id):
            return False, "CAMERA_ID_UNKNOWN"
        if obs.get("coord_frame") != "ROBOT_MM":
            return False, "ROBOT_MM_REQUIRED"
        if obs.get("kind") == "QR" or obs.get("target_pose_valid") is not True:
            return False, "TARGET_POSE_UNKNOWN"
        timestamp = obs.get("timestamp_ms")
        received = context.get("received_at_ms")
        age_limit = context.get("max_observation_age_ms", 250)
        if not isinstance(age_limit, int) or isinstance(age_limit, bool) or not 1 <= age_limit <= 5000:
            return False, "FRESHNESS_CONFIG_INVALID"
        if not isinstance(timestamp, int) or isinstance(timestamp, bool) or not isinstance(received, int):
            return False, "FRESHNESS_UNKNOWN"
        # Local observation timestamps only. A remote sender must first replace
        # timestamp_ms with its local receive timestamp and retain sender time separately.
        if not 0 <= _elapsed(now_ms, timestamp) <= age_limit or not 0 <= _elapsed(now_ms, received) <= age_limit:
            return False, "OBSERVATION_STALE"
        binding = context.get("active_binding")
        calibration = context.get("calibration")
        if not isinstance(binding, dict) or not isinstance(calibration, dict):
            return False, "CALIBRATION_UNKNOWN"
        if calibration.get("status") != "VERIFIED" or calibration.get("is_simulated") is not False:
            return False, "CALIBRATION_NOT_VERIFIED"
        matrix = calibration.get("H")
        if not isinstance(matrix, (list, tuple)) or len(matrix) != 3:
            return False, "H_UNKNOWN"
        try:
            if any(len(row) != 3 for row in matrix) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for row in matrix for v in row):
                return False, "H_INVALID"
            a, b, c = matrix
            determinant = a[0] * (b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0])
            if abs(determinant) < 1e-12:
                return False, "H_SINGULAR"
        except (ValueError, TypeError):
            return False, "H_INVALID"
        for key in ("calibration_id", "camera_id", "station", "plane", "image_profile"):
            value = obs.get(key)
            if value is None or value == "" or value != binding.get(key) or value != calibration.get(key):
                return False, "BINDING_MISMATCH:" + key
        profile = obs.get("image_profile")
        pose_id = profile.get("pose_id") if isinstance(profile, dict) else None
        if not pose_id or calibration.get("pose_id") != pose_id:
            return False, "PARKING_POSE_UNKNOWN"
        identity = context.get("current_observation_identity")
        if identity != (obs.get("camera_id"), obs.get("frame_id"), obs.get("target_id")):
            return False, "OBSERVATION_IDENTITY_MISMATCH"
        if context.get("required_target_id") is not None and str(obs.get("target_id")) != str(context["required_target_id"]):
            return False, "TARGET_ID_MISMATCH"
        if context.get("lane_safe") is not True or context.get("obstacle_clear") is not True:
            return False, "PATH_CLEARANCE_UNKNOWN"
        workspace = context.get("workspace_mm")
        if not isinstance(workspace, dict):
            return False, "WORKSPACE_UNKNOWN"
        try:
            for axis in ("x", "y"):
                value = obs.get(axis + "_mm")
                bounds = workspace.get(axis)
                if isinstance(value, bool) or value is None or not math.isfinite(value):
                    return False, "POSE_INVALID"
                if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
                    return False, "WORKSPACE_INVALID"
                if not bounds[0] <= value <= bounds[1]:
                    return False, "TARGET_UNREACHABLE"
        except (ValueError, TypeError):
            return False, "POSE_OR_WORKSPACE_INVALID"
        return True, "AUTHORIZED"


class LiveMotionPort:
    def __init__(self, config=None, transport=None, observation_provider=None, clock_ms=None):
        self.config = config if config is not None else {}
        self.transport = transport
        self.observation_provider = observation_provider
        self.clock_ms = clock_ms or monotonic_ms
        self.gate = SafetyGate()
        self.observation = None
        self.received_at_ms = None
        self.counter = 0
        self.pending = None
        self.consumed_frames = {}

    def update_observation(self, observation, received_at_ms=None):
        self.observation = dict(observation)
        self.received_at_ms = self.clock_ms() if received_at_ms is None else received_at_ms

    def disable(self):
        self.config["motion_enabled"] = False

    def _submit(self, action, arguments, target_id=None):
        self.counter += 1
        command_id = "live-%d" % self.counter
        if self.pending is not None:
            return MotionReceipt("UNKNOWN", command_id, "PREVIOUS_MOTION_UNRESOLVED")
        if self.observation_provider is not None:
            try:
                self.update_observation(self.observation_provider())
            except Exception:
                return MotionReceipt("ERROR", command_id, "PERCEPTION_UNAVAILABLE")
        if action in ("pick_to_carrier", "place_from_carrier"):
            color = arguments.get("color_id")
            slot = arguments.get("slot")
            if type(color) is not int or not 1 <= color <= 6 or (slot is not None and (type(slot) is not int or not 1 <= slot <= 3)):
                return MotionReceipt("ERROR", command_id, "ACTION_ARGUMENT_INVALID")
            station = arguments.get("source", arguments.get("dest"))
            if station not in ("RAW", "PROCESS", "BUFFER") or station != (self.observation or {}).get("station"):
                return MotionReceipt("ERROR", command_id, "ACTION_STATION_MISMATCH")
            if action == "pick_to_carrier" and (station not in ("RAW", "PROCESS") or (station == "PROCESS" and slot is None)):
                return MotionReceipt("ERROR", command_id, "PICK_SOURCE_OR_SLOT_INVALID")
            if action == "place_from_carrier" and (type(arguments.get("layer")) is not int or not 1 <= arguments["layer"] <= 2 or slot is None):
                return MotionReceipt("ERROR", command_id, "ACTION_ARGUMENT_INVALID")
            if action == "place_from_carrier" and station not in ("PROCESS", "BUFFER"):
                return MotionReceipt("ERROR", command_id, "PLACEMENT_STATION_INVALID")
            if action == "place_from_carrier" and station == "PROCESS" and arguments["layer"] != 1:
                return MotionReceipt("ERROR", command_id, "PROCESS_LAYER_INVALID")
            kind = (self.observation or {}).get("kind")
            if action == "pick_to_carrier" and kind != "COLOR":
                return MotionReceipt("ERROR", command_id, "COLOR_OBSERVATION_REQUIRED")
            if action == "place_from_carrier":
                expected = "TOP" if arguments["layer"] == 2 else "RING"
                if kind != expected:
                    return MotionReceipt("ERROR", command_id, expected + "_OBSERVATION_REQUIRED")
                target_id = color if arguments["layer"] == 2 else slot
        if action == "drive_to" and arguments.get("station") not in ("QR", "RAW", "PROCESS", "BUFFER", "START_1", "START_2"):
            return MotionReceipt("ERROR", command_id, "ACTION_ARGUMENT_INVALID")
        if action == "return_to_start" and arguments.get("start_id") not in ("START_1", "START_2"):
            return MotionReceipt("ERROR", command_id, "ACTION_ARGUMENT_INVALID")
        context = dict(self.config)
        context["received_at_ms"] = self.received_at_ms
        context["required_target_id"] = target_id
        if action in ("drive_to", "return_to_start"):
            destination = arguments.get("station", arguments.get("start_id"))
            allowed, reason = self.gate.check_navigation(self.observation, self.clock_ms(), context, destination)
        else:
            allowed, reason = self.gate.check(self.observation, self.clock_ms(), context)
        if not allowed:
            return MotionReceipt("ERROR", command_id, reason)
        camera_id = self.observation["camera_id"]
        frame_id = self.observation["frame_id"]
        if frame_id <= self.consumed_frames.get(camera_id, -1):
            return MotionReceipt("ERROR", command_id, "OBSERVATION_ALREADY_CONSUMED")
        if self.transport is None:
            return MotionReceipt("ERROR", command_id, "MOTION_TRANSPORT_UNAVAILABLE")
        # Record uncertainty before calling external code; exceptions may occur
        # after physical acceptance. No timeout or reconnect clears this state.
        self.pending = command_id
        self.consumed_frames[camera_id] = frame_id
        try:
            response = self.transport.submit(command_id, action, arguments)
            status = response.get("status", "UNKNOWN")
            if status not in ("ACK", "BUSY", "DONE", "ERROR", "UNKNOWN"):
                status = "UNKNOWN"
            if response.get("command_id") != command_id:
                return MotionReceipt("UNKNOWN", command_id, "RESPONSE_ID_MISMATCH")
            if status in ("DONE", "ERROR"):
                self.pending = None
            return MotionReceipt(status, command_id, response.get("reason", ""))
        except Exception as exc:
            return MotionReceipt("UNKNOWN", command_id, "TRANSPORT_UNCERTAIN:" + str(exc))

    def resolve(self, receipt):
        """Consume controller completion for the exact outstanding command."""
        if self.pending is None or receipt.get("command_id") != self.pending:
            raise ValueError("no matching pending command")
        if receipt.get("status") not in ("DONE", "ERROR"):
            raise ValueError("completion required")
        self.pending = None

    def drive_to(self, station):
        return self._submit("drive_to", {"station": station})

    def pick_to_carrier(self, color_id, source, slot):
        return self._submit("pick_to_carrier", {"color_id": color_id, "source": source, "slot": slot}, color_id)

    def place_from_carrier(self, color_id, dest, slot, layer):
        return self._submit("place_from_carrier", {"color_id": color_id, "dest": dest, "slot": slot, "layer": layer}, color_id)

    def return_to_start(self, start_id):
        return self._submit("return_to_start", {"start_id": start_id})
