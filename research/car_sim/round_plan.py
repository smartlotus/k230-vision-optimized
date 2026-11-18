"""Kinematic routes for the frozen gray-field scene, in metres, Z up.

Yaw is measured counterclockwise about +Z from the chassis forward axis +Y.
The example sites and robot drives are simulated. The workflow engine supplies
the manipulation order, color, cargo constraints and completion bookkeeping.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


MAP_MASK_SHA256 = "fa5f526d741e27af830cc4b5a6c3e945ef48442670db3aabd12bd642def6f8d2"
FLOOR_SPAN_M = 3.0  # overhead mask spans a 3 m square around the 2.4 m field
HUB = (0.0, 0.0, 0.0)
NORTH = (0.0, 0.86, 0.0)
NE = (0.90, 0.86, 0.0)
SITE_PATHS = {
    "START_1": [HUB, NORTH, NE, (1.05, 0.86, 0.0), (1.05, 1.05, 0.0)],
    "START_2": [HUB, (0.0, -0.86, 0.0), (1.05, -0.86, 0.0), (1.05, -1.05, 0.0)],
    "QR": [HUB, NORTH, NE, (0.90, 0.86, -math.pi/2),
           (0.95, 0.86, -math.pi/2), (0.95, 0.08, -math.pi/2)],
    "RAW": [HUB, (0.,0.,math.pi/2), (0., .86, math.pi/2), (0.0, 0.965, math.pi/2)],
    "PROCESS": [HUB, (0.0, 0.0, -math.pi/2), (0.0, -0.89, -math.pi/2)],
    "BUFFER": [HUB, (0.0, 0.0, math.pi), (-0.89, 0.0, math.pi)],
}
# Mesh world centres in field_gray_v1.blend. The frozen render shows only
# Obstacle 0. The material tray and obstructing cylinder are absent from
# the static class raster, so both need their own geometry checks.
OBSTACLES = (("Obstacle 0", 0.884112716, -0.115080357, 0.025000001),
             ("Turntable", 0.0, 1.269999981, 0.150000006))


def _site_path(site, slot):
    path = list(SITE_PATHS[site])
    if site not in ("PROCESS", "BUFFER") or slot in (None, 2):
        return path
    if slot not in (1, 3):
        raise ValueError("Unexpected station slot")
    x, y, yaw = path[-1]
    offset = (slot - 2) * 0.15
    path.append((x - offset if site == "PROCESS" else x,
                 y + offset if site == "BUFFER" else y, yaw))
    return path


def _connect(site, slot, next_site, next_slot=None):
    source = _site_path(site, slot)
    target = _site_path(next_site, next_slot)
    if site == next_site:
        return [source[-1]] if source[-1] == target[-1] else [source[-1], target[-1]]
    common = 0
    while common < min(len(source), len(target)) and source[common] == target[common]:
        common += 1
    return source[common - 1:][::-1] + target[common:]


def build_round(payload="156+123+516+231", start_id="START_1"):
    """Return QR approach and the existing workflow's 31 PICK/PLACE/travel steps."""
    if start_id not in ("START_1", "START_2"):
        raise ValueError("Select an existing blue starting zone")
    project = Path(__file__).resolve().parents[2] / "k230_transport_mvp"
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from sim.run_workflow import run_workflow

    mission = run_workflow(payload, capacity=3, start_id=start_id)
    if mission["summary"]["status"] != "DONE" or not mission["field_complete"]:
        raise ValueError("Underlying simulated workflow did not complete")
    events = [dict(action="QR", source_station=start_id, dest_station="QR",
                   batch=0, layer=0, target_color=None, target_slot=None, pick_index=None,
                   poses=_connect(start_id, None, "QR"))]
    site, slot = "QR", None
    for entry in mission["journal"]:
        step = dict(entry["intent"]["step"])
        action = step["action"]
        if action == "PICK":
            next_site = step["source_station"]
        elif action == "RETURN":
            next_site = start_id
        else:
            next_site = step["dest_station"]
        next_slot = step["target_slot"] if action in ("PICK", "PLACE") else None
        step["poses"] = _connect(site, slot, next_site, next_slot)
        if action in ("PICK", "PLACE"):
            # Last 6 mm from the lane towards the dock is an observation
            # approach, including repeated visits to the same turntable pose.
            x, y, yaw = step["poses"][-1]
            step["poses"] += [(x-.006*math.cos(yaw), y-.006*math.sin(yaw), yaw), (x,y,yaw)]
        events.append(step)
        site, slot = next_site, next_slot
    dims = json.loads(Path(__file__).with_name("vehicle_dimensions.json").read_text(encoding="utf-8"))
    footprint = [dims["drawing_mm"][key]/1000 for key in ("chassis_width", "chassis_length")]
    return {"task": payload, "start_id": start_id, "workflow_summary": mission["summary"],
            "qr_board_center_xy_m": [1.20, 0.08], "footprint_width_length_m": footprint,
            "slot_order": {"PROCESS": "decreasing X", "BUFFER": "increasing Y"},
            "steps": events}


def _sweep(segments, width, length):
    """Use poses no more than 2 mm apart when measured at the farthest corner."""
    radius = math.hypot(width, length) / 2
    for poses in segments:
        yield poses[0]
        for a, b in zip(poses, poses[1:]):
            corner_bound = math.dist(a[:2], b[:2]) + abs(b[2] - a[2]) * radius
            steps = max(1, math.ceil(corner_bound / 0.002))
            for index in range(1, steps + 1):
                t = index / steps
                yield tuple(v + (w - v) * t for v, w in zip(a, b))


def validate_mask(plan, mask_path, footprint=None):
    """Sample the entire nominal chassis rectangle against the authoritative raster."""
    from PIL import Image, ImageChops, ImageDraw

    mask_path = Path(mask_path)
    if hashlib.sha256(mask_path.read_bytes()).hexdigest() != MAP_MASK_SHA256:
        raise ValueError("The floor mask differs from the verified gray-field map")
    with Image.open(mask_path) as src:
        field = src.convert("L")
    visited = Image.new("L", field.size)
    painter = ImageDraw.Draw(visited)
    width, length = footprint or plan["footprint_width_length_m"]
    if not all(math.isfinite(v) and v > 0 for v in (width, length)):
        raise ValueError("Footprint must be finite and positive")
    closest = {name: math.inf for name, *_ in OBSTACLES}
    count = 0
    for x, y, yaw in _sweep([s["poses"] for s in plan["steps"]], width, length):
        count += 1
        c, s = math.cos(yaw), math.sin(yaw)
        corners = []
        for ax, ay in ((-1,-1), (1,-1), (1,1), (-1,1)):
            lx, ly = ax * width / 2, ay * length / 2
            wx, wy = x + c * lx - s * ly, y + s * lx + c * ly
            if max(abs(wx), abs(wy)) > 1.2:
                raise ValueError(f"Chassis leaves the map near {(x, y)}")
            corners.append((int((wx/FLOOR_SPAN_M + .5) * field.width),
                            int((.5 - wy/FLOOR_SPAN_M) * field.height)))
        painter.polygon(corners, fill=255)
        for name, ox, oy, radius in OBSTACLES:
            rx, ry = ox - x, oy - y
            dx = max(0., abs(c * rx + s * ry) - width/2)
            dy = max(0., abs(-s * rx + c * ry) - length/2)
            gap = math.hypot(dx, dy) - radius
            closest[name] = min(closest[name], gap)
            if gap <= 0:
                raise ValueError(f"Chassis collision with {name} near {(x, y)}")
    forbidden = field.point([0 if v in (0, 3) else 255 for v in range(256)])
    overlap = ImageChops.multiply(visited, forbidden)
    if overlap.getbbox():
        raise ValueError(f"Chassis crosses a blocked floor pixel inside {overlap.getbbox()}")
    return {"sampled_poses": count, "footprint_width_length_m": [width, length],
            "floor_mask_sha256": MAP_MASK_SHA256, "blocked_floor_pixels": 0,
            "min_chassis_gaps_m": closest, "max_corner_step_m": .002,
            "scope": "Kinematic chassis footprint and frozen obstacle layout"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="156+123+516+231")
    parser.add_argument("--start", choices=("START_1", "START_2"), default="START_1")
    parser.add_argument("--mask", type=Path)
    args = parser.parse_args()
    plan = build_round(args.task, args.start)
    if args.mask:
        plan["route_validation"] = validate_mask(plan, args.mask)
    print(json.dumps(plan, indent=2))
