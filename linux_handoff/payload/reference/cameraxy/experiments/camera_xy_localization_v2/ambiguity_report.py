"""Map the distant look-alikes and summarize invariance on matched CameraXY views."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from baselines import rgb_descriptors, template_map
from contract import digest, load_rows, map_to_pixel, xy
from model import pixels


def summarize(values):
    return {"n": len(values), "median": float(np.median(values)) if len(values) else None,
            "p05": float(np.quantile(values, .05)) if len(values) else None,
            "p95": float(np.quantile(values, .95)) if len(values) else None}


def exact_aliases(mask, layout, margin):
    centers, bank = template_map(mask, layout, margin)
    patterns = defaultdict(dict)
    for j, templates in enumerate(bank):
        for k, view in enumerate(templates):
            signature = hashlib.sha256(view.tobytes()).hexdigest()
            patterns[signature].setdefault(j, k)
    ambiguous, covered = [], set()
    for sig, group in patterns.items():
        if len(group) < 2:
            continue
        pairs = np.array(list(group))
        points = centers[pairs]
        distance = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        q, r = np.unravel_index(distance.argmax(), distance.shape)
        if distance[q, r] < 250:
            continue
        covered.update(group)
        ambiguous.append(dict(pattern_sha256=sig, unique_centers=len(group),
                              separation_mm=float(distance[q, r]), camera_xy_examples_mm=points[[q, r]].tolist(),
                              rotation_hypotheses_deg=[15*group[pairs[q]], 15*group[pairs[r]]]))
    ambiguous.sort(key=lambda r: r["separation_mm"], reverse=True)
    return dict(
        scope="32x18 synthetic static RoadNet semantic template at 400 mm and 96 deg, "
              "50 mm candidate and 15 deg rotation spacing",
        candidate_positions=len(centers), candidate_views=len(centers)*bank.shape[1],
        unique_pattern_count=len(patterns),
        exact_patterns_with_at_least_two_positions_separated_250_mm=len(ambiguous),
        candidate_positions_in_these_patterns=len(covered), examples=ambiguous[:3],
    )


def grid_exposure(rows):
    coordinates = {split: np.array([xy(row) for row in rows if row["split"] == split and row["variant"] == 0])
                   for split in ("train", "test")}
    exposure = {}
    for step in (50, 100, 150):
        grid = {split: [tuple(v) for v in np.minimum(2400//step-1, (sample+1200)//step)]
                for split, sample in coordinates.items()}
        training = set(grid["train"])
        exposure[str(step)] = dict(train_occupied_cells=len(training),
                                    test_position_cells_seen_in_train=sum(x in training for x in grid["test"]),
                                    test_positions=len(grid["test"]))
    return exposure


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--layout", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    manifest = a.data / "manifest.jsonl"
    rows = {r["sample_id"]: r for r in load_rows(manifest)}
    pair = json.loads(a.pairs.read_text(encoding="utf8"))
    test = [r for r in pair if r["split"] == "test"]
    positions = [r for r in rows.values() if r["split"] == "test" and r["variant"] == 0]
    f = {}
    for row in positions:
        group = row["pose_group_id"]
        frames = []
        for i in (0, 1, 2):
            with Image.open(a.data / rows[f"{group}-{i}"]["image"]) as image:
                frames.append(pixels(image))
        f[group] = rgb_descriptors(frames)
    same_position = {label: [float(np.mean((v[0] - v[i])**2)) for v in f.values()]
                     for label, i in (("different_yaw_and_scene_nuisances", 1),
                                      ("same_pose_and_geometry_different_appearance", 2))}
    closest = sorted(test, key=lambda row: row["descriptor_mse"])[:20]
    main_map = Image.open(a.data / "field_preview.png").convert("RGB")
    draw = ImageDraw.Draw(main_map)
    for rec in reversed(closest):
        a_xy = map_to_pixel(*rec["xy_mm"], *main_map.size, 3000)
        b_xy = map_to_pixel(*rec["reference_xy_mm"], *main_map.size, 3000)
        draw.line((*a_xy, *b_xy), fill="#e65334", width=2)
        for center, color in ((a_xy, "#e65334"), (b_xy, "#2852a0")):
            u, v = center
            draw.ellipse((u-4, v-4, u+4, v+4), fill=color, outline="#fff4e0", width=1)
    draw.rectangle((0, 0, main_map.width, 58), fill="#202126")
    draw.text((20, 14), "20 closest visual pairs at different map locations", fill="#ffffff")
    draw.text((20, 34), "Orange: held-out frame     Blue: training reference     Separation >= 250 mm",
              fill="#ebe4da")
    a.out.mkdir()
    main_map.save(a.out / "ambiguity_map.png")
    summary = dict(
        dataset_manifest_sha256=digest(manifest), pairs_sha256=digest(a.pairs),
        closest_pair_map_sha256=digest(a.out / "ambiguity_map.png"),
        test_distant_xy_mm=summarize([r["distance_mm"] for r in test]),
        test_distant_descriptor_mse=summarize([r["descriptor_mse"] for r in test]),
        same_position_descriptor_mse={k: summarize(v) for k, v in same_position.items()},
        closest_twenty_regions=Counter(r["region"] for r in closest), closest=closest[:5])
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    (a.out / "grid-exposure.json").write_text(json.dumps(grid_exposure(list(rows.values())), indent=2), encoding="utf8")
    if a.layout:
        layout = json.loads(a.layout.read_text(encoding="utf8"))
        margin = json.loads((a.data / "run.json").read_text(encoding="utf8"))["margin"]
        with Image.open(a.data / "field_mask.png") as mask:
            alias = exact_aliases(np.asarray(mask), layout, margin)
        (a.out / "exact-static-aliases.json").write_text(json.dumps(alias, indent=2), encoding="utf8")
    if closest:
        nearest = closest[0]
        preview = Image.new("RGB", (664, 248), "#272525")
        labels = ImageDraw.Draw(preview)
        gap = nearest["distance_mm"]
        labels.text((8, 8), f"Closest distant RGB descriptor pair ({gap:.1f} mm apart)", fill="white")
        for x, key, label in ((8, "sample_id", "Held out"), (336, "reference_id", "Training reference")):
            row = rows[nearest[key]]
            with Image.open(a.data / row["image"]) as image:
                preview.paste(image.resize((320, 180)), (x, 30))
            px, py = xy(row)
            labels.text((x, 217), f"{label}  X {px:.1f} mm   Y {py:.1f} mm", fill="white")
        preview.save(a.out / "closest-rgb-pair.png")


if __name__ == "__main__":
    main()
