"""Offline acquisition checks on each pre-action robot view at its prescribed dock."""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from perception import FrozenPerception, TOP_SHA, digest
from round_plan import build_round

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "k230_transport_mvp"))
from vision.ring_native import group_concentric

# At the prescribed slot-specific dock the lens is forward-offset relative to
# the work point. The next ground ring is about 125 px away along the image X
# axis at the nominal height. Keep the window narrower than one slot spacing.
# The elevated turntable top has a different plane and projected work point.
WINDOWS = {"RAW": (145, 174, 245, 268), "PAPER": (185, 162, 279, 236)}
MATCH_RADIUS = 8.0  # reporting threshold only, measured in 640x360 source pixels


def read_png(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != (360, 640, 3):
        raise ValueError(f"Expected BGR 640x360: {path}")
    return image


def inside(point, window):
    u, v = point
    left, top, right, bottom = window
    return left <= u <= right and top <= v <= bottom


def printed_levels(gray, center, outer):
    # Downward, level lens: circles remain circles on each horizontal plane.
    # Verify narrow dark strokes have light paper on BOTH sides for most of
    # the circumference. This rejects solid disks and single thick outlines.
    angles = np.arange(48) * (2 * math.pi / 48)
    radii = np.arange(1, math.ceil(outer) + 6)
    x = (center[0] + radii[:, None] * np.cos(angles)).astype(np.float32)
    y = (center[1] + radii[:, None] * np.sin(angles)).astype(np.float32)
    samples = cv2.remap(gray, x, y, cv2.INTER_LINEAR).astype(np.float32)
    contrast = np.minimum(samples[:-4], samples[4:]) - samples[2:-2]
    support = (contrast >= 8).mean(axis=1)
    peaks = [i for i in range(1, len(support)-1) if support[i] >= .65 and
             support[i-1] <= support[i] > support[i+1] and radii[i+2] > outer*.38]
    levels = []
    for i in sorted(peaks, key=lambda i: -support[i]):
        radius = int(radii[i+2])
        if all(abs(radius - level) >= 3 for level in levels):
            levels.append(radius)
    return sorted(levels)


def glyph_mask(mask):
    y, x = np.nonzero(mask)
    ink = mask[y.min():y.max()+1, x.min():x.max()+1]
    scale = min(36/ink.shape[0], 30/ink.shape[1])
    w, h = (max(1, round(s*scale)) for s in ink.shape[::-1])
    sample = np.zeros((44, 36), np.uint8)
    x, y = (36-w)//2, (44-h)//2
    sample[y:y+h, x:x+w] = (cv2.resize(ink, (w,h)) > 127)*255
    distance = cv2.distanceTransform(255-sample, cv2.DIST_L2, 3)
    return sample > 0, distance


@lru_cache(maxsize=1)
def digit_templates():
    # Generic font templates generated independently of every Blender frame.
    templates = {}
    for digit in (1, 2, 3):
        shapes = []
        for font in (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX):
            mask = np.zeros((100,90),np.uint8)
            cv2.putText(mask, str(digit), (5,76), font, 2.5, 255, 4, cv2.LINE_AA)
            shapes.append(glyph_mask(mask))
        mask = Image.new("L",(80,80))
        ImageDraw.Draw(mask).text((6,0),str(digit),fill=255,font=ImageFont.load_default(size=50))
        shapes.append(glyph_mask(np.asarray(mask)))
        templates[digit] = shapes
    return templates


def read_slot_digit(gray, center, outer):
    x, y = (round(v) for v in center)
    radius = round(.4*outer)
    crop = gray[y-radius:y+radius+1, x-radius:x+radius+1]
    _, mask = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    possible = [i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA] >= 7 and
                radius*.4 <= stats[i,cv2.CC_STAT_HEIGHT] <= radius*1.3 and
                math.dist(centroids[i], (radius,radius)) <= .4*radius]
    if len(possible) != 1:
        return None
    ink, distance = glyph_mask((labels==possible[0]).astype(np.uint8)*255)
    ranked = sorted((min(float(d[ink].mean()+distance[m].mean())*.5 for m,d in shapes), digit)
                    for digit, shapes in digit_templates().items())
    best, runner = ranked[:2]
    return best[1] if best[0] <= 1.0 and runner[0] - best[0] >= .25 else None


def ring_candidates(image, window):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    left, top, right, bottom = window
    margin = 54
    x0, y0 = max(0, left-margin), max(0, top-margin)
    patch = gray[y0:bottom+margin, x0:right+margin]
    edges = cv2.Canny(patch, 25, 75)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    fits = []
    for points in contours:
        if len(points) < 60:
            continue
        (u, v), (a, b), _ = cv2.fitEllipse(points)
        center = (u+x0, v+y0)
        if not (32 <= a <= b <= 100 and a/b >= .90 and inside(center, window)):
            continue
        dx, dy = points[:, 0, 0]-u, points[:, 0, 1]-v
        sectors = (np.arctan2(dy, dx) + math.pi) * (24 / (2 * math.pi))
        if len(np.unique(np.minimum(23, sectors.astype(int)))) < 22:
            continue
        fits.append({"center": list(center), "semiaxes": [b/2, a/2],
                     "rms_px": float(np.sqrt(np.mean((np.hypot(dx,dy)-(a+b)/4)**2)))})
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    rows, cols = np.ogrid[:gray.shape[0], :gray.shape[1]]
    candidates = []
    for group in group_concentric(fits, center_tolerance=4, radius_gap=8, min_levels=2):
        outer = max(fit["semiaxes"][0] for fit in group["boundaries"])
        center = group["center"]
        levels = printed_levels(gray, center, outer)
        if len(levels) < 3:
            continue
        r = np.hypot(cols-center[0], rows-center[1])
        # A projected outline can still surround a placed block. Compare the
        # interior appearance to clean paper immediately outside the rings.
        paper = np.median(lab[(r >= outer+3) & (r <= outer+7)], axis=0)
        delta = np.linalg.norm(lab[r < .43*outer] - paper, axis=1)
        fraction = float(np.mean(delta < 20))
        candidates.append({"center": center, "stroke_radii_px": levels,
                           "read_slot": read_slot_digit(gray, center, outer),
                           "interior_paper_fraction": fraction, "appears_empty": fraction >= .85})
    return candidates


def acquisition(description, image, decoded, slot=None, window=None):
    station, kind, expected = description
    window = window or WINDOWS["RAW" if station == "RAW" else "PAPER"]
    detections = decoded["detections"]
    in_window = [d for d in detections if inside(d["center_xy_projected_3d"], window)]
    details = {"station": station, "feature": kind, "expected_observed_color": expected,
               "capture_window_xyxy": window, "detections_in_window": [
                   {"color": d["color_id"], "score": d["score"],
                    "center": d["center_xy_projected_3d"]} for d in in_window]}
    center, reason = None, "REQUIRE_UNIQUE_COLOR_TOP"
    if kind == "empty_rings":
        candidates = ring_candidates(image, window)
        details["rings_in_window"] = candidates
        reason = "REQUIRE_UNIQUE_EMPTY_CONCENTRIC_MARK"
        if in_window:
            reason = "MATERIAL_PREDICTED_IN_SLOT"
        elif len(candidates) == 1 and candidates[0]["appears_empty"]:
            if slot in (1,2,3) and candidates[0]["read_slot"] == slot:
                center = candidates[0]["center"]
            else:
                reason = "SLOT_DIGIT_UNREADABLE_OR_WRONG"
    elif len(in_window) == 1:
        d = in_window[0]
        if d["color_id"] != expected:
            reason = "OBSERVED_COLOR_MISMATCH"
        elif (d["score"] < .5 or d["center_valid_probability"] < .8 or
              d["center_visible_probability"] < .8 or d["boundary_candidate"]):
            reason = "INSUFFICIENT_TOP_CONFIDENCE"
        else:
            center = d["center_xy_projected_3d"]
    return {**details, "status": "ACQUIRED" if center else "HOLD",
            "reason": "UNIQUE_IMAGE_FEATURE" if center else reason, "center_xy": center,
            "offset_from_image_center_px": [center[0]-320, center[1]-180] if center else None}


def score_reference(observation, ref):
    if observation["feature"] == "support_top" and observation["expected_observed_color"] != ref["support_color_id"]:
        raise ValueError("Reference support color disagrees with the recorded mission")
    center = observation["center_xy"]
    visible = ref["visible_target"]
    error = math.dist(center, ref["target_center_xy"]) if center and visible else None
    result = "HOLD" if center is None else "ON_TARGET" if error is not None and error <= MATCH_RADIUS else "OFF_TARGET"
    return {"result": result, "error_px": error, "visible": visible, "reference_xy": ref["target_center_xy"]}


def counts(scores):
    return {k: sum(s["result"] == k for s in scores) for k in ("ON_TARGET", "OFF_TARGET", "HOLD")}


def run(hub, folder):
    mission = json.loads((folder / "round.json").read_text(encoding="utf-8"))
    qr = cv2.QRCodeDetector().detectAndDecode(read_png(folder / "qr.png"))[0]
    # A decode mismatch invalidates the sequence. No action is eligible.
    if qr != mission["plan"]["task"]:
        return {"status": "HOLD_QR", "decoded_qr": qr, "steps_evaluated": 0}
    steps = build_round(qr, mission["plan"]["start_id"])["steps"]
    models = FrozenPerception(hub)
    observations, support_for_slot = [], {}
    if len(steps) != len(mission["timeline"]):
        raise ValueError("QR plan and camera replay length differ")
    for index, (step, event) in enumerate(zip(steps, mission["timeline"])):
        action, slot = step["action"], step["target_slot"]
        wanted = step["target_color"]
        if (index, action, wanted, slot, step["batch"], step["layer"]) != (
            event["index"], event["action"], event["color"], event["slot"], event["batch"], event["layer"]
        ):
            raise ValueError(f"QR plan and render differ at step {index}")
        if action not in ("PICK", "PLACE"):
            continue
        if action == "PICK":
            description = (step["source_station"], "pick_top", wanted)
        elif step["layer"] == 2:
            description = (step["dest_station"], "support_top", support_for_slot[slot])
        else:
            description = (step["dest_station"], "empty_rings", None)
        name = f"{index:02}.png"
        path = folder / "mission_camera" / name
        image = read_png(path)
        _, decoded = models.observe(image)
        observation = acquisition(description, image, decoded, slot)
        observations.append({"index": index, "file": name, "sha256": digest(path),
                             "action": action, "slot": slot, "carried_or_pick_color": wanted, **observation})
        # The recorded timeline executes independently of these observations.
        if action == "PLACE" and step["dest_station"] == "BUFFER" and step["layer"] == 1:
            support_for_slot[slot] = wanted

    # The renderer can supply one fresh exposure for a rejected initial
    # observation. Never grant a retry based on the reference residual.
    rows = {o["file"]: o for o in observations}
    retry_images = sorted((folder/"mission_retry").glob("*.png"))
    for path in retry_images:
        obs = rows.get(path.name)
        if obs is None or obs["status"] != "HOLD":
            raise ValueError(f"Unexpected retry without initial HOLD: {path.name}")
        retry_hash = digest(path)
        if retry_hash == obs["sha256"]:
            raise ValueError(f"Retry duplicates first observation: {path.name}")
        description = (obs["station"], obs["feature"], obs["expected_observed_color"])
        retry_image = read_png(path)
        _, retry_decoded = models.observe(retry_image)
        obs["retry"] = {"file": f"mission_retry/{path.name}", "sha256": retry_hash,
                        **acquisition(description, retry_image, retry_decoded, obs["slot"])}

    # Offline scoring starts here, after every image decision is frozen.
    truth = json.loads((folder / "mission_truth.json").read_text(encoding="utf-8"))["frames"]
    retry_ref_path = folder / "mission_retry_truth.json"
    retry_truth = json.loads(retry_ref_path.read_text(encoding="utf-8"))["frames"] if retry_images else {}
    if set(truth) != set(rows) or set(retry_truth) != {p.name for p in retry_images}:
        raise ValueError("Camera acquisition set differs from the reference")
    post_retry = []
    for name, obs in rows.items():
        obs["offline_score"] = score_reference(obs, truth[name])
        if name in retry_truth:
            ref = retry_truth[name]
            if ref.get("primary_sha256", obs["sha256"]) != obs["sha256"]:
                raise ValueError(f"Retry reference belongs to a different initial capture: {name}")
            obs["retry"]["offline_score"] = score_reference(obs["retry"], ref)
        post_retry.append(obs.get("retry", obs)["offline_score"])
    primary_counts = counts([o["offline_score"] for o in observations])
    final_counts = counts(post_retry)
    return {"status": "ALL_FEATURES_ACQUIRED" if final_counts["ON_TARGET"] == len(observations) else "INCOMPLETE_ACQUISITION",
            "scope": "Single-frame acquisition and rejection on independently prescribed operation frames. "
                     "Empty-slot digits checked against generic templates. Occupied slots use expected color and "
                     "prescribed docking. No gripper alignment, "
                     "serial command, scene feedback, or closed-loop mission success evaluated.",
            "decoded_qr": qr, "input_wh": [640, 360], "topface_onnx_sha256": TOP_SHA,
            "source_origin": "upper left, +u right, +v down", "steps_evaluated": len(observations),
            "capture_windows_xyxy": WINDOWS, "offline_match_radius_px": MATCH_RADIUS,
            "scores": primary_counts, "scores_after_retry": final_counts, "reobserved": len(retry_images),
            "observations": observations}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, required=True)
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.hub, args.dir)
    (args.dir / "mission_vision.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(report["status"], "primary", report.get("scores", {}),
          "after_retry", report.get("scores_after_retry", {}))
