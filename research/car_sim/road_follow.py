"""Estimate the right road boundary in the body frame from RoadNet and feedback."""

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

from perception import ROAD_DIR, ROAD_SHA, camera_source, load_session, road_outputs


TARGET_GAP_M = .060
NEAR_ENTER_M = .025
NEAR_EXIT_M = .035
VEHICLE_RIGHT_EDGE_M = .140
MAX_OBSERVATION_AGE_MS = 175
X_SAMPLES_M = np.arange(.075, .426, .0025)
Y_SAMPLES_M = np.linspace(-.16, .16, 17)


def road_grid(labels, camera):
    height, width = labels.shape
    x, y = np.meshgrid(X_SAMPLES_M, Y_SAMPLES_M)
    body = np.stack((x, y, np.zeros_like(x), np.ones_like(x)), axis=-1)
    points = body @ np.asarray(camera["camera_from_body"], dtype=float).T
    depth = -points[..., 2]
    fx = camera["focal_px_640"] * width / 640
    fy = camera["focal_px_640"] * height / 360
    with np.errstate(divide="ignore", invalid="ignore"):
        u = width * .5 + fx * points[..., 0] / depth
        v = height * .5 - fy * points[..., 1] / depth
    valid = (depth > 0) & (u >= 0) & (v >= 0) & (u < width) & (v < height)
    classes = np.full(x.shape, 255, dtype=np.uint8)
    classes[valid] = labels[v[valid].astype(int), u[valid].astype(int)]
    return classes


def edge_samples(classes):
    rows = []
    for y, row in zip(Y_SAMPLES_M, classes):
        candidates = []
        for index in range(12, len(row) - 15):
            if row[index - 1] != 0 or row[index] not in (1, 2, 3):
                continue
            if (np.count_nonzero(row[index - 12:index] == 0) >= 10 and
                    np.count_nonzero(np.isin(row[index:index + 15], (1, 2, 3))) >= 13):
                candidates.append(float((X_SAMPLES_M[index - 1] + X_SAMPLES_M[index]) * .5))
        if len(candidates) == 1:
            rows.append((float(y), candidates[0]))
    return rows


def fit_boundary(rows):
    """Ignore isolated branch edges from the mouth of an intersection."""
    points = np.asarray(rows)
    longitudinal, lateral = points.T
    separation = longitudinal[:, None] - longitudinal
    delta = lateral[:, None] - lateral
    long_baseline = separation >= .059
    if not long_baseline.any():
        return None
    slope = np.median(delta[long_baseline] / separation[long_baseline])
    center = np.median(lateral - slope * longitudinal)
    good = np.abs(lateral - (slope * longitudinal + center)) <= .009
    count = int(good.sum())
    if count < max(5, math.ceil(.60 * len(rows))) or np.ptp(longitudinal[good]) < .080:
        return None
    slope, center = np.polyfit(longitudinal[good], lateral[good], 1)
    return (float(slope), float(center), count) if abs(slope) <= .40 else None


def observe(labels, sample, previously_near=False):
    now = sample["received_ms"]
    capture = sample["captured_ms"]
    if sample["camera_id"] != "ROAD_SIDE" or not sample["rig_at_travel"]:
        return {"status": "UNKNOWN", "reason": "WRONG_CAMERA_OR_RIG"}
    if now - capture < 0 or now - capture > MAX_OBSERVATION_AGE_MS:
        return {"status": "UNKNOWN", "reason": "STALE_OBSERVATION"}
    rows = edge_samples(road_grid(labels, sample))
    if len(rows) < 5:
        return {"status": "UNKNOWN", "reason": "RIGHT_BOUNDARY_NOT_OBSERVABLE", "rows": rows}
    fit = fit_boundary(rows)
    if fit is None:
        return {"status": "UNKNOWN", "reason": "AMBIGUOUS_RIGHT_BOUNDARY", "rows": rows}
    slope, center, inliers = fit
    gap = center - VEHICLE_RIGHT_EDGE_M
    warning_gap = gap - abs(slope) * .1475
    near = warning_gap < (NEAR_EXIT_M if previously_near else NEAR_ENTER_M)
    return {"status": "NEAR" if near else "CLEAR", "reason": "RIGHT_GRAY_EDGE",
            "gap_mm": 1000 * gap, "predicted_corner_gap_mm": 1000 * warning_gap,
            "body_heading_error_deg": math.degrees(math.atan(slope)), "inlier_rows": inliers,
            "rows": rows, "right_boundary_x_body_m": center,
            "recommendation": {"actuator": "BASE", "direction": "BODY_LEFT"} if near else None}


def tracking_command(observation, forward_mps):
    if observation["status"] == "UNKNOWN":
        return {"vx_mps": 0., "vy_mps": 0., "yaw_rate_rad_s": 0., "mode": "HOLD"}
    gap = observation["gap_mm"] / 1000
    lateral = max(-.115, min(.115, 4.0 * (gap - TARGET_GAP_M)))
    yaw_error = math.radians(observation["body_heading_error_deg"])
    yaw_rate = max(-.65, min(.65, -3.2 * yaw_error))
    return {"vx_mps": lateral, "vy_mps": forward_mps,
            "yaw_rate_rad_s": yaw_rate, "mode": "TRACK"}


def serve(hub):
    frozen = load_session(hub / ROAD_DIR / "roadnet.onnx", ROAD_SHA)
    for line in sys.stdin:
        inputs = json.loads(line)
        path = Path(inputs["image"])
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to decode {path}")
        _, logits = road_outputs(frozen, camera_source(bgr))
        labels = logits.argmax(axis=2).astype(np.uint8)
        if inputs.get("prediction_path"):
            path = Path(inputs["prediction_path"])
            path.write_bytes(cv2.imencode(".png", labels)[1].tobytes())
        result = observe(labels, inputs, inputs.get("previously_near", False))
        if "forward_mps" in inputs:
            result["command"] = tracking_command(result, inputs["forward_mps"])
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, required=True)
    serve(parser.parse_args().hub)
