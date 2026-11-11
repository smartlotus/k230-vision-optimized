"""Small shared helpers; no board or desktop imaging imports."""
import math
import time


def now_ms():
    if hasattr(time, "ticks_ms"):
        return time.ticks_ms()
    return int(time.monotonic() * 1000)


def elapsed_ms(now, then):
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(now, then)
    return now - then


def setting(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def in_roi(x, y, roi):
    return roi is not None and roi[0] <= x < roi[0] + roi[2] and roi[1] <= y < roi[1] + roi[3]


def observation(kind, frame=None, target_id=None, station=None, status="NOT_FOUND", reason=None):
    frame = frame or {}
    return {
        "status": status, "kind": kind,
        "camera_id": frame.get("camera_id"), "frame_id": frame.get("frame_id"),
        "timestamp_ms": frame.get("timestamp_ms", now_ms()), "target_id": target_id,
        "coord_frame": "NONE", "u_px": None, "v_px": None,
        "x_mm": None, "y_mm": None, "angle_deg": None,
        "score_hint": 0, "stable": False,
        "is_simulated": frame.get("is_simulated", False),
        "calibration_id": None, "raw_payload": None,
        "station": station, "image_profile": frame.get("image_profile"),
        "target_pose_valid": False, "reason": reason,
    }


def pixels(result, u, v, score=0):
    result.update({"coord_frame": "IMAGE_PX", "u_px": u, "v_px": v, "score_hint": score})
    return result
