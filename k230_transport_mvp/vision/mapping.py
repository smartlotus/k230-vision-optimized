"""Measured homography application with exact source and plane binding."""
from .common import finite


def transform_point(H, u, v):
    if not H or len(H) != 3 or any(len(row) != 3 for row in H):
        raise ValueError("H must be 3 by 3")
    if not all(finite(n) for row in H for n in row) or not finite(u) or not finite(v):
        raise ValueError("non-finite homography input")
    det = (H[0][0]*(H[1][1]*H[2][2]-H[1][2]*H[2][1])
           - H[0][1]*(H[1][0]*H[2][2]-H[1][2]*H[2][0])
           + H[0][2]*(H[1][0]*H[2][1]-H[1][1]*H[2][0]))
    if abs(det) < 1e-12:
        raise ValueError("singular homography")
    denominator = H[2][0]*u+H[2][1]*v+H[2][2]
    if abs(denominator) < 1e-9:
        raise ValueError("projective denominator is zero")
    x = (H[0][0]*u+H[0][1]*v+H[0][2])/denominator
    y = (H[1][0]*u+H[1][1]*v+H[1][2])/denominator
    if not finite(x) or not finite(y):
        raise ValueError("non-finite projected point")
    return x, y


def apply_calibration(result, calibration, plane):
    result.update({"x_mm": None, "y_mm": None, "calibration_id": None, "target_pose_valid": False, "plane": plane})
    result["coord_frame"] = "IMAGE_PX" if result.get("u_px") is not None and result.get("v_px") is not None else "NONE"
    if result["status"] != "OK":
        return result
    if not calibration:
        result.update({"status": "UNCALIBRATED", "reason": "CALIBRATION_REQUIRED"})
        return result
    expected = {"camera_id": result["camera_id"], "plane": plane, "station": result.get("station"),
                "image_profile": result.get("image_profile"), "is_simulated": result["is_simulated"]}
    calibration_id = calibration.get("id") or calibration.get("calibration_id")
    profile = result.get("image_profile")
    pose = profile.get("pose_id") if isinstance(profile, dict) else None
    if (calibration.get("validated") is not True or not calibration_id
            or (calibration.get("id") and calibration.get("calibration_id") and calibration["id"] != calibration["calibration_id"])
            or any(calibration.get(k) != v for k, v in expected.items())
            or (not result["is_simulated"] and (not pose or calibration.get("pose_id") != pose))):
        result.update({"status": "UNCALIBRATED", "reason": "CALIBRATION_BINDING_MISMATCH"})
        return result
    try:
        x, y = transform_point(calibration.get("H"), result["u_px"], result["v_px"])
    except (ValueError, TypeError, IndexError):
        result.update({"status": "UNCALIBRATED", "reason": "INVALID_HOMOGRAPHY"})
        return result
    result.update({"coord_frame": "ROBOT_MM", "x_mm": x, "y_mm": y,
                   "calibration_id": calibration_id,
                   "target_pose_valid": bool(result["stable"] and calibration.get("target_pose_valid") is True)})
    return result
