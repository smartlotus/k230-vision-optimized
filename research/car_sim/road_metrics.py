"""Independent road judge. The controller never imports this module."""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from campaign_ledger import digest, read_json
from round_plan import FLOOR_SPAN_M, MAP_MASK_SHA256, OBSTACLES, _sweep


MASK = Path("worktrees/ops-temporal-camera-v1/docs/temporal_camera/map-correction/field_mask.png")
BODY_WIDTH, BODY_LENGTH = .280, .295


class RoadJudge:
    def __init__(self, hub):
        source = hub / MASK
        if digest(source) != MAP_MASK_SHA256:
            raise ValueError("Independent gray-field mask failed fingerprint")
        with Image.open(source) as img:
            self.field = np.asarray(img.convert("L")).copy()
        self.height, self.width = self.field.shape

    def pixel(self, x, y):
        return int((.5 + x / FLOOR_SPAN_M) * self.width), int((.5 - y / FLOOR_SPAN_M) * self.height)

    def truth_gap(self, x, y, heading):
        values = []
        for offset in np.arange(.075, .39, .001):
            u, v = self.pixel(x + math.cos(heading) * offset, y + math.sin(heading) * offset)
            values.append((offset, int(self.field[v, u]) if 0 <= u < self.width and 0 <= v < self.height else 255))
        for (before, left), (after, right) in zip(values, values[1:]):
            if left == 0 and right == 1:
                return float(1000 * ((before + after) / 2 - BODY_WIDTH / 2))
        return None

    def footprint(self, pose):
        x, y, angle = pose
        c, s = math.cos(angle), math.sin(angle)
        points = [self.pixel(x + c * lx - s * ly, y + s * lx + c * ly)
                  for lx, ly in ((-BODY_WIDTH / 2, -BODY_LENGTH / 2), (BODY_WIDTH / 2, -BODY_LENGTH / 2),
                                 (BODY_WIDTH / 2, BODY_LENGTH / 2), (-BODY_WIDTH / 2, BODY_LENGTH / 2))]
        xs, ys = zip(*points)
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        if left < 0 or top < 0 or right >= self.width or bottom >= self.height:
            return {"forbidden_pixels": None, "out_of_mask": True, "collision": None}
        raster = Image.new("1", (right - left + 1, bottom - top + 1))
        ImageDraw.Draw(raster).polygon([(u - left, v - top) for u, v in points], fill=1)
        pixels = self.field[top:bottom + 1, left:right + 1][np.asarray(raster)]
        forbidden = int(np.count_nonzero(pixels != 0))
        gaps = {}
        for name, ox, oy, radius in OBSTACLES:
            dx, dy = ox - x, oy - y
            outside_x = max(0., abs(c * dx + s * dy) - BODY_WIDTH / 2)
            outside_y = max(0., abs(-s * dx + c * dy) - BODY_LENGTH / 2)
            gaps[name] = math.hypot(outside_x, outside_y) - radius
        return {"forbidden_pixels": forbidden, "out_of_mask": False,
                "collision": any(gap <= 0 for gap in gaps.values())}


def observation_report(hub, source):
    rows = read_json(source)
    judge = RoadJudge(hub)
    scored = []
    for case in rows:
        seen = case["observation"]
        expected = case["expected"]
        truth = judge.truth_gap(*case["pose"])
        error = seen["gap_mm"] - truth if truth is not None and seen.get("gap_mm") is not None else None
        correct_direction = seen.get("recommendation") == {"actuator": "BASE", "direction": "BODY_LEFT"}
        passed = (seen["status"] == expected and correct_direction == (expected == "NEAR")
                  and (expected == "UNKNOWN" or error is not None and abs(error) <= 12))
        scored.append({
            "name": case["name"], "expected": expected, "actual": seen["status"], "reason": seen["reason"],
            "right_gap_truth_mm": truth, "right_gap_estimate_mm": seen.get("gap_mm"), "gap_error_mm": error,
            "correct_recommendation": correct_direction == (expected == "NEAR"), "pass": passed,
        })
    return {"verdicts": scored, "denominator": len(scored), "passed": sum(r["pass"] for r in scored),
            "map_sha256": MAP_MASK_SHA256, "gate_pass": all(r["pass"] for r in scored)}


def summarize_drive(traces, judge, target, dt):
    result = {}
    for tag, trace in traces.items():
        poses = [trace[0]["pose_at_exposure_m_rad"]] + [row["integrated_pose_m_rad"] for row in trace]
        error = [abs(row["integrated_pose_m_rad"][0]) * 1000 for row in trace]
        first_failure = None
        total_samples = 0
        for index, interval in enumerate(zip(poses, poses[1:])):
            for pose in _sweep([interval], BODY_WIDTH, BODY_LENGTH):
                total_samples += 1
                footprint = judge.footprint(pose)
                if first_failure is None and (footprint["collision"] or footprint["out_of_mask"]
                                             or footprint["forbidden_pixels"]):
                    first_failure = {"step": index, "pose_m_rad": list(pose), **footprint}
        converged = next((i for i in range(len(error) - 2) if all(v <= 8 for v in error[i:i + 3])), None)
        applied = set(row["action_id"] for row in trace)
        result[tag] = {
            "frames": len(trace), "distinct_observed_frames": len({row["source_rgb"] for row in trace}),
            "initial_lateral_error_mm": poses[0][0] * 1000, "final_lateral_error_mm": poses[-1][0] * 1000,
            "final_heading_error_deg": math.degrees(poses[-1][2]),
            "stop_error_mm": abs(poses[-1][1] - target) * 1000,
            "stopped": trace[-1]["command"]["mode"] == "STOP",
            "converged_at_sim_s": None if converged is None else (converged + 1) * dt,
            "first_safety_failure": first_failure, "footprint_sample_count": total_samples,
            "p95_lateral_error_mm": float(np.percentile(error, 95)),
            "unknown_frames": sum(row["road"]["status"] == "UNKNOWN" for row in trace),
            "stale_frames": sum(row["road"]["reason"] == "STALE_OBSERVATION" for row in trace),
            "visual_holds": sum(row["command"]["mode"] == "HOLD" for row in trace),
            "duplicate_action_ids": len(trace) - len(applied), "trace_lateral_error_mm": error,
        }
    return result


def drive_report(hub, source):
    data = read_json(source)
    result = summarize_drive(data["traces"], RoadJudge(hub), data["stop_target_y_m"], data["sim_step_s"])
    open_, closed = (result[tag] for tag in ("open", "closed"))
    count = min(open_["frames"], closed["frames"])
    baseline, assisted = (float(np.percentile(side["trace_lateral_error_mm"][:count], 95))
                          for side in (open_, closed))
    improvement = 1 - assisted / baseline if baseline > 0 else None
    gates = {
        "matched_p95_reduced_by_30_percent": improvement is not None and improvement >= .30,
        "closed_loop_stopped_within_8_mm": closed["stopped"] and closed["stop_error_mm"] <= 8,
        "closed_loop_chassis_safe": closed["first_safety_failure"] is None,
        "closed_loop_lateral_settled": closed["converged_at_sim_s"] is not None
                                       and abs(closed["final_lateral_error_mm"]) <= 8,
        "no_repeated_action_id": all(side["duplicate_action_ids"] == 0 for side in result.values()),
        "stale_is_held": not data["case"].get("drop_frames") or closed["stale_frames"] > 0 and closed["visual_holds"] > 0,
    }
    return {"case": data["case"], "sides": result, "paired_sample_count": count,
            "matched_p95_mm": {"open": baseline, "closed": assisted},
            "matched_reduction": improvement, "checks": gates, "gate_pass": all(gates.values()),
            "scope": "One straight fixed-field segment, seed-matched simulated odometry and disturbance",
            "truth_mask_sha256": MAP_MASK_SHA256}
