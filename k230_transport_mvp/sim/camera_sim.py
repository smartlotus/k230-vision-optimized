"""Synthetic observations; no real image or measured calibration is implied."""


class CameraSim:
    def __init__(self, payload, inject=None):
        self.payload, self.inject = payload, inject
        self.frame_id, self.now_ms = 0, 0
        self.injected = False

    def get_frame(self, camera_id):
        self.frame_id += 1
        self.now_ms += 40
        return {"camera_id": camera_id, "frame_id": self.frame_id,
                "timestamp_ms": self.now_ms, "synthetic": True}

    def detect(self, mode, frame, target_id=None, station=None):
        observation = {
            "status": "OK", "kind": mode, "camera_id": frame["camera_id"],
            "frame_id": frame["frame_id"], "timestamp_ms": frame["timestamp_ms"],
            "target_id": target_id, "coord_frame": "NONE" if mode == "QR" else "ROBOT_MM",
            "u_px": None if mode == "QR" else 100 + (target_id or 0) * 10,
            "v_px": None if mode == "QR" else 120,
            "x_mm": None if mode == "QR" else float((target_id or 0) * 10),
            "y_mm": None if mode == "QR" else 20.0,
            "angle_deg": None, "score_hint": 100, "stable": True,
            "is_simulated": True, "calibration_id": None if mode == "QR" else "SIM_ONLY_V1",
            "raw_payload": self.payload if mode == "QR" else None,
            "station": station, "target_pose_valid": mode != "QR", "stale_after_ms": 200}
        if mode != "QR" and not self.injected:
            if self.inject == "dropped-frame":
                observation.update(status="UNAVAILABLE", reason="DROPPED_FRAME")
                self.injected = True
            elif mode == "COLOR" and self.inject == "missing-color":
                observation.update(status="NOT_FOUND", reason="MISSING_COLOR")
                self.injected = True
            elif mode == "COLOR" and self.inject == "wrong-color":
                observation["target_id"] = target_id % 6 + 1
                self.injected = True
            elif mode in ("RING", "TOP") and self.inject == "ring-not-found":
                observation.update(status="NOT_FOUND", reason="ID_UNKNOWN")
                self.injected = True
        if observation["status"] != "OK":
            observation.update(x_mm=None, y_mm=None, target_pose_valid=False)
        return observation
