"""CameraXY manifest contract, deterministic sampling and coordinate conventions."""
import hashlib
import json
import math
import random
from pathlib import Path

TARGETS = ("camera_x_mm", "camera_y_mm")
HALF_MM = 1200
COUNTS = {"smoke": 100, "pilot": 1000, "full": 10000}
FOLDS = ("train", "val", "test")
PROFILE = ("nominal", "low_light", "reflection")
FIELDS = {
    "sample_id", "scene_id", "pose_group_id", "spatial_block_id", "seed", "split", "variant",
    "camera_xy_target", "camera_metadata", "render_config", "image", "image_sha256",
    "mask", "mask_sha256", "ambiguity_region",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def xy(row):
    target = row["camera_xy_target"]
    if set(target) != set(TARGETS):
        raise ValueError(f"Unexpected target keys: {sorted(target)}")
    pair = tuple(target[name] for name in TARGETS)
    if not all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= HALF_MM for v in pair):
        raise ValueError("Camera optical center is outside FIELD_MAP")
    return pair


def map_to_pixel(x, y, width, height, span_mm):
    """North-up metric map to pixel center coordinates."""
    return (x / span_mm + .5) * width - .5, (.5 - y / span_mm) * height - .5


def block_and_split(x, y):
    bx = min(7, int((x + HALF_MM) // 300))
    by = min(7, int((y + HALF_MM) // 300))
    if by == 6 or (bx == 7 and by == 0):
        split = "test"
    elif bx == 1:
        split = "val"
    elif 550 < y < 950 or -950 < x < -550:
        split = "buffer"
    else:
        split = "train"
    return f"{bx}:{by}", split


def classify(x, y, layout, margin):
    for center in layout["blue_start_zones"]["centers"]:
        if all(abs(v - 1000 * c) <= 500 * s for v, c, s in zip(
                (x, y), center, layout["blue_start_zones"]["size"])):
            return "blue"
    arm = 500 * layout["road"]["arm_width"]
    if min(abs(x), abs(y)) <= arm + margin:
        if 450 <= max(abs(x), abs(y)) < 920 and min(abs(x), abs(y)) < arm:
            return "repeated_straight"
        return "road"
    return None


def schedule(stage, layout, seed, margin, count):
    rng = random.Random(seed)
    for split, quota in zip(FOLDS, (count * 8 // 10, count // 10, count // 10)):
        for index in range(quota):
            for _ in range(10000):
                x, y = (round(rng.uniform(-HALF_MM, HALF_MM), 4) for _ in range(2))
                block, selected = block_and_split(x, y)
                region = classify(x, y, layout, margin)
                if selected == split and region:
                    break
            else:
                raise RuntimeError(f"Cannot sample {split} from this camera envelope")
            group = f"{stage}-{split}-{index:05d}"
            pose_seed = rng.randrange(2**31)
            for variant in range(3 if split == "test" else 2):
                yield dict(group=group, provenance=pose_seed, x=x, y=y, block=block,
                           split=split, variant=variant, region=region)
