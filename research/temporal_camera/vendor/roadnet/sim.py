import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAYOUT = json.loads((ROOT / "configs/field_layout.json").read_text(encoding="utf8"))
# Figure 1 places the blue squares flush with the right corners and the
# turntable 75–85 mm inside the upper line at its southern edge.
STRATA = (
    "vertical", "horizontal", "intersection", "yellow_edge", "white_edge",
    "blue_edge", "obstacle", "storage", "processing", "turntable",
)
STAGE = {"smoke": (10000, 10, 1), "pilot": (20000, 10, 10),
         "full": (30000, 50, 10), "hardcase": (40000, 10, 2),
         "refine": (50000, 100, 2), "road_acceptance": (60000, 40, 1)}
PALETTE = [
    (0.31, 0.31, 0.31), (0.77, 0.73, 0.42), (0.81, 0.81, 0.78),
    (0.035, 0.12, 0.63), (0.015, 0.015, 0.015), (0.80, 0.045, 0.035),
    (0.07, 0.07, 0.07),
]
BLOCK_COLORS = ((0.74,.043,.03), (.91,.68,.03), (.035,.14,.68),
                (.03,.48,.11), (.024,.025,.023), (.08,.54,.76))
# Separate raw integer channels make exact mask validation independent of PNG color management.
MASK_COLORS = [(1.0,0,0), (0,1.0,0), (0,0,1.0), (1.0,1.0,0),
               (1.0,0,1.0), (0,1.0,1.0), (1.0,1.0,1.0)]
RINGS = (.053, .058, .065, .075, .085, .095)


def anchor(kind, rng):
    side = rng.choice((-1, 1))
    yaw = rng.choice((0, 180))
    along = side * rng.uniform(.42, .81)
    return {
        "vertical": (rng.uniform(-.07,.07), along, yaw),
        "horizontal": (along, rng.uniform(-.07,.07), yaw + 90),
        "intersection": (rng.uniform(-.16,.16), rng.uniform(-.16,.16),
                         yaw + rng.choice((0,90))),
        "yellow_edge": (side * rng.uniform(.14,.29), rng.choice((-1,1)) *
                         rng.uniform(.33,.60), yaw),
        "white_edge": (side * rng.uniform(.21,.41), side * rng.uniform(.73,.92), yaw),
        "blue_edge": (rng.uniform(.80,.99), side * rng.uniform(.82,1.0), yaw),
        "obstacle": (rng.uniform(-.09,.09), along, yaw),
        "storage": (-1.04, rng.uniform(-.12,.12), 90),
        "processing": (rng.uniform(-.12,.12), -1.04, 180),
        "turntable": (rng.uniform(-.07,.07), 1.05, 0),
    }[kind]


def obstacles(rng, kind, x, y):
    count = rng.randrange(5)
    positions = []
    if kind == "obstacle":
        positions.append((x + rng.uniform(-.09,.09), y + rng.uniform(-.13,.13)))
        count = max(1, count)
    for _ in range(80):
        if len(positions) == count:
            break
        cross = rng.uniform(-.16,.16)
        along = rng.uniform(-.96,.96)
        px, py = (cross, along) if rng.randrange(2) else (along, cross)
        if all(math.hypot(px-a, py-b) >= .10 for a,b in positions):
            positions.append((px, py))
    return positions


def scene_groups(stage, stratum):
    groups = STAGE[stage][1]
    return groups * 2 if stage == "hardcase" and stratum in ("white_edge", "obstacle") else groups


def capture_records(stage):
    base, _, views = STAGE[stage]
    partition = random.Random(base)
    for order, kind in enumerate(STRATA):
        groups = scene_groups(stage, kind)
        splits = ["train"] * (groups * 8 // 10) + ["val"] * (groups // 10) + ["test"] * (groups // 10)
        partition.shuffle(splits)
        for group, split in enumerate(splits):
            seed = base + order * 100 + group
            rng = random.Random(seed)
            x, y, yaw = anchor(kind, rng)
            lights = ("nominal", "low_light", "shadow", "reflection")[group % 4]
            if stage == "hardcase" and kind in ("white_edge", "obstacle"):
                lights = ("nominal", "low_light", "low_light", "shadow", "reflection")[group % 5]
            obs = obstacles(rng, kind, x, y)
            phase = rng.uniform(0, 2*math.pi)
            table_height = rng.uniform(*LAYOUT["raw_material_turntable"]["height_range"])
            period = rng.uniform(*LAYOUT["raw_material_turntable"]["rotation_period_s"])
            palette = [tuple(min(.99, channel * rng.uniform(.87,1.13)) for channel in c) for c in PALETTE]
            stock_colors = rng.sample(range(6), 3)
            station_item = rng.randrange(6) if kind in ("storage","processing") and group % 3 else None
            lamp_xy = (rng.uniform(-.9,.9),rng.uniform(-.9,.9))
            height = rng.uniform(.369,.431)
            pitch = rng.uniform(-2.5,2.5)
            roll = rng.uniform(-2.5,2.5)
            heading = yaw + rng.uniform(-6,6)
            fov = rng.uniform(94,98)
            direction = (-math.sin(math.radians(heading)), math.cos(math.radians(heading)))
            for view in range(views):
                seed_frame = seed * 100 + view
                frame_rng = random.Random(seed_frame)
                advance = (.018 * (view - (views-1)/2))
                px = min(1.14,max(-1.14,x + direction[0]*advance + frame_rng.uniform(-.016,.016)))
                py = min(1.14,max(-1.14,y + direction[1]*advance + frame_rng.uniform(-.016,.016)))
                scene_id = f"{stage}-{seed}"
                identifier = f"{scene_id}-{view:02d}"
                yield {
                    "id": identifier, "scene_id": scene_id,
                    "trajectory_id": scene_id, "camera_profile_seed": seed,
                    "scene_seed": seed, "seed": seed_frame, "split": split,
                    "source": "synthetic", "stratum": kind, "lighting_id": lights,
                    "surface_variant_id": seed, "palette_linear": palette,
                    "center": [px, py, height + frame_rng.uniform(-.003,.003)],
                    "camera_yaw_deg": heading + frame_rng.uniform(-.85,.85),
                    "camera_roll_deg": roll + frame_rng.uniform(-.4,.4),
                    "camera_pitch_deg": pitch + frame_rng.uniform(-.4,.4),
                    "diag_fov_deg": fov, "field_pose_xy_m": [0,0],
                    "obstacle_instances": [{"center_xy": p, "class_id": 4,
                                           "diameter_m": .05, "height_m": .10} for p in obs],
                    "turntable_height_m":table_height, "turntable_period_s":period,
                    "turntable_phase_rad": phase + (2*math.pi * view / 30 / period),
                    "turntable_color_ids": stock_colors, "station_item_color": station_item,
                    "lamp_xy": lamp_xy, "image_path": f"rgb/{identifier}.png",
                    "mask_path": f"masks/{identifier}.png",
                }

