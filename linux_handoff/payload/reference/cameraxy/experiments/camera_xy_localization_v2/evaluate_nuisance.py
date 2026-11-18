"""Validate matched perturbations and rescore the unchanged Kaggle CameraXY model."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from baselines import mask_inputs
from contract import TARGETS, digest, load_rows, xy
from infer_map import LIVE_METHODS, MapLocalizer
from map_matcher import analytic_masks_scores, decode_scores
from model import pixels
from train import calibrated
from train_map import SEMANTIC_LOGIT_SCALE, fused_scores

CHANGES = {
    "yaw90": ({"yaw_deg", "camera_to_world_4x4"}, set()),
    "yaw180": ({"yaw_deg", "camera_to_world_4x4"}, set()),
    "appearance": (set(), {"lighting_id", "palette_linear"}),
    "obstacle": (set(), {"obstacle_instances"}),
    "material": (set(), set()),
    "station": (set(), {"station_jitter_m"}),
}
STATION_REGIONS = {"storage", "rough"}


def differences(a, b):
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def read_checked_frames(data, rows, previously_sampled):
    groups = defaultdict(dict)
    images, masks = [], []
    source_images = set()
    for i, row in enumerate(rows):
        factor = row["factor"]
        if factor not in ("control", *CHANGES) or row["split"] != "synthetic_challenge":
            raise ValueError("Unexpected challenge factor or split")
        group = groups[row["pose_group_id"]]
        if factor in group or xy(row) in previously_sampled or set(row["camera_xy_target"]) != set(TARGETS):
            raise ValueError("Challenge overlaps existing positions or XY label contract drifted")
        group[factor] = i
        dims = row["camera_metadata"]["resolution"]
        rgb, mask = data / row["image"], data / row["mask"]
        for file, key in ((rgb, "image"), (mask, "mask")):
            if not file.resolve().is_relative_to(data.resolve()) or digest(file) != row[key + "_sha256"]:
                raise ValueError(f"Image integrity mismatch: {file}")
        if row["camera_metadata"]["projection_check_max_px"] >= .005:
            raise ValueError("Camera projection QA failed")
        source_images.add(row["image_sha256"])
        with Image.open(rgb) as im, Image.open(mask) as semantic:
            if im.mode != "RGB" or semantic.mode != "L" or list(im.size) != dims or im.size != semantic.size:
                raise ValueError(f"RGB and masks do not have the expected geometry: {rgb}")
            images.append(pixels(im))
            mask_array = np.asarray(semantic)
            if not set(np.unique(mask_array)) <= {0, 1, 2, 3, 4, 5, 6, 255}:
                raise ValueError("Unknown RoadNet mask class")
            masks.append(mask_array)
    indices = defaultdict(list)
    for group in groups.values():
        base = rows[group["control"]]
        expected = {"control", *CHANGES}
        if base["ambiguity_region"] not in STATION_REGIONS:
            expected.remove("station")
        if set(group) != expected:
            raise ValueError("Challenge group count/factors differ from schedule")
        if base["extra_material_base_m"] is not None or np.any(masks[group["control"]] == 4):
            raise ValueError("Control does not have clean dynamic insertion conditions")
        for factor, i in group.items():
            if factor == "control":
                continue
            v = rows[i]
            allowed_camera, changes_render = CHANGES[factor]
            camera_delta = differences(base["camera_metadata"], v["camera_metadata"])
            render_delta = differences(base["render_config"], v["render_config"])
            if allowed_camera:
                camera_delta.discard("projection_check_max_px")
                delta = (v["camera_metadata"]["yaw_deg"] - base["camera_metadata"]["yaw_deg"]) % 360
                if abs(delta - int(factor[3:])) > 1e-10:
                    raise ValueError("Unexpected paired camera yaw perturbation")
            if (camera_delta != allowed_camera or render_delta != changes_render
                    or xy(base) != xy(v) or base["sample_id"] != v["control_id"]
                    or v["seed"] != base["seed"] or v["ambiguity_region"] != base["ambiguity_region"]):
                raise ValueError(f"Unexpected paired change: {factor}, {v['sample_id']}")
            if (v["extra_material_base_m"] is None) == (factor == "material"):
                raise ValueError("Unexpected extra material in a controlled pair")
            m0, m1 = masks[group["control"]], masks[i]
            if factor == "appearance" and (not np.array_equal(m0, m1) or v["image_sha256"] == base["image_sha256"]):
                raise ValueError("Appearance pair must change RGB while keeping the same mask")
            if factor in ("material", "obstacle"):
                cls = 5 if factor == "material" else 4
                if not ((m1 == cls) & (m0 != cls)).any():
                    raise ValueError(f"Rendered {factor} does not add visible matching class pixels")
            if factor == "station" and not (((m1 == 6) | (m0 == 6)) & (m1 != m0)).any():
                raise ValueError("Station shift did not move any visible markings")
            indices[factor].append((group["control"], i))
    n_groups = {k: len(v) for k, v in indices.items()}
    if (len(groups) != 96 or len(rows) != 608 or len(source_images) != len(rows)
            or any(n_groups[k] != (32 if k == "station" else 96) for k in CHANGES)):
        raise ValueError("Unexpected challenge sample counts")
    return np.stack(images), indices


@torch.inference_mode()
def score_methods(net, templates, data, roadnet, rows, images):
    learned = np.concatenate([net(torch.from_numpy(x).float() / 255).numpy()
                              for x in np.array_split(images, (len(images)+63)//64)])
    geom = SEMANTIC_LOGIT_SCALE * analytic_masks_scores(
        templates, mask_inputs(data, rows, range(len(rows)), roadnet), "cpu")
    return dict(learned_map=learned, semantic_roadnet=geom, fused_map=fused_scores(learned, geom))


def paired_metrics(d, first, p, pairs):
    base, trial = np.asarray(pairs).T
    errors_first = d[:, 0]
    best = d.min(axis=1)
    ok = best <= 50
    return dict(
        pairs=len(base), control_top5_within50=int(ok[base].sum()),
        trial_top5_within50=int(ok[trial].sum()),
        candidate_regressions=int((ok[base] & ~ok[trial]).sum()), candidate_rescues=int((~ok[base] & ok[trial]).sum()),
        control_top1_within50=int((errors_first[base] <= 50).sum()), trial_top1_within50=int((errors_first[trial] <= 50).sum()),
        control_min_error_median_mm=float(np.median(best[base])), trial_min_error_median_mm=float(np.median(best[trial])),
        trial_min_error_p95_mm=float(np.quantile(best[trial], .95)),
        alternate_mean_top5_probability=float(np.mean(p[trial])),
        alternate_top5_brier=float(np.mean((p[trial] - ok[trial])**2)),
        control_first_probability=float(np.mean(first[base])), trial_first_probability=float(np.mean(first[trial])),
        rate_delta_top5=float(ok[trial].mean() - ok[base].mean()),
    )


def render_example(data, rows, example, output):
    files = {r["sample_id"]: data / r["image"] for r in rows}
    canvas = Image.new("RGB", (690, 278), "#202428")
    label = ImageDraw.Draw(canvas)
    label.text((18, 14), "Same simulated camera XY, different lighting and surface color", fill="white")
    x_mm, y_mm = example["camera_xy_mm"]
    label.text((18, 36), f"Field optical center  X {x_mm:.1f} mm   Y {y_mm:.1f} mm", fill="#c9cdd2")
    for x, key, caption, error in (
            (18, "control_id", "Control", "control_nearest_mm"),
            (352, "perturbed_id", "Appearance change", "perturbed_nearest_mm")):
        with Image.open(files[example[key]]) as pic:
            canvas.paste(pic, (x, 58))
        label.text((x, 248), f"{caption}   best of 5 error: {example[error]:.1f} mm", fill="#e2e3e4")
    canvas.save(output)


def main():
    p = argparse.ArgumentParser()
    for key in ("data", "full", "layout", "roadnet", "verified-output", "out", "render-log"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    archive = a.verified_output
    result = json.loads((archive / "result.json").read_text(encoding="utf8"))
    loc = MapLocalizer(archive / "map-inference.pt", archive / "metrics.json", result,
                       data=a.full, layout=a.layout, roadnet=a.roadnet)
    model_metrics = json.loads((archive / "metrics.json").read_text(encoding="utf8"))
    manifest = a.data / "manifest.jsonl"
    full_manifest = a.full / "manifest.jsonl"
    if digest(full_manifest) != loc.manifest_sha:
        raise ValueError("Challenge comparison must bind the original training manifest")
    full_rows = load_rows(full_manifest)
    rows = load_rows(manifest)
    known_hashes = {x["image_sha256"] for x in full_rows}
    if any(x["image_sha256"] in known_hashes for x in rows):
        raise ValueError("Challenge reused an exact G5 RGB")
    images, pairs = read_checked_frames(a.data, rows, {xy(x) for x in full_rows})
    score = score_methods(loc.net, loc.templates, a.data, a.roadnet, rows, images)
    truth = np.asarray([xy(r) for r in rows])
    out, examples = {}, []
    for mode in LIVE_METHODS:
        v = model_metrics[mode]
        candidates, first, shortlist = decode_scores(score[mode], loc.centers)
        d = np.linalg.norm(candidates - truth[:, None, :], axis=-1)
        p1, _ = calibrated(first, v["primary_confidence"])
        p5, _ = calibrated(shortlist, v["set_confidence"])
        out[mode] = {factor: paired_metrics(d, p1, p5, ids) for factor, ids in pairs.items()}
        if mode == loc.method:
            for factor, stakes in pairs.items():
                worst = sorted(stakes, key=lambda pair: -(d[pair[1]].min()-d[pair[0]].min()))[:5]
                for base, trial in worst:
                    examples.append(dict(factor=factor, control_id=rows[base]["sample_id"],
                                         perturbed_id=rows[trial]["sample_id"], camera_xy_mm=truth[base].tolist(),
                                         control_nearest_mm=float(d[base].min()), perturbed_nearest_mm=float(d[trial].min()),
                                         control_rgb_sha256=rows[base]["image_sha256"],
                                         perturbed_rgb_sha256=rows[trial]["image_sha256"]))
    controls = Counter(r["ambiguity_region"] for r in rows if r["factor"] == "control")
    doc = dict(status="PASS_CHALLENGE_PROTOCOL", targets=TARGETS, regions=dict(controls),
               frames=len(rows), paired_perturbations=sum(len(p) for p in pairs.values()),
               challenge_manifest_sha256=digest(manifest), full_training_manifest_sha256=loc.manifest_sha,
               checkpoint_sha256=loc.checkpoint_sha, render_log_sha256=digest(a.render_log),
               normalized_challenge_geometry="synthetic, fresh positions; spatial bands can overlap training coverage",
               max_projection_error_px=max(r["camera_metadata"]["projection_check_max_px"] for r in rows),
               primary_output="unlocalized", selected=loc.method, paired=out)
    a.out.mkdir()
    example = next(x for x in examples if x["factor"] == "appearance")
    panel = a.out / "appearance-example.png"
    render_example(a.data, rows, example, panel)
    doc.update(appearance_example_sha256=digest(panel), evaluator_sha256=digest(__file__))
    (a.out / "report.json").write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf8")
    (a.out / "worst_pairs.json").write_text(json.dumps(examples, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: v for k, v in doc.items() if k != "paired"}, indent=2))


if __name__ == "__main__":
    main()
