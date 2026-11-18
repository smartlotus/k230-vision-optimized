"""Measure CameraXY preprocessing drift on the unchanged G5 spatial holdouts."""
import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version
import torch

from board_tensor_bench import fixture
from contract import digest, load_rows, xy
from map_matcher import decode_scores
from model import pixels
from tensor_bench import localizer, write_json
from train import calibrated


EXPERIMENT = Path(__file__).resolve().parent
BENCH = EXPERIMENT / "evidence/G7-native-input-20261117"
INPUT_SHAPE = (3, 360, 640)
OUTPUT_SHAPE = (3, 96, 160)
MODES = ("training_pil", "pil_through_640", "ai2d_through_640")


def resize_ai2d(value):
    """Emulate the measured 4x UINT8 half-pixel resize of planar RGB."""
    if value.shape != INPUT_SHAPE or value.dtype != np.uint8:
        raise ValueError("Expected 640x360 uint8 planar RGB")
    # Each center lies between the second and third pixel of a 4-pixel span.
    total = value[:, 1::4, 1::4].astype(np.uint16)
    total += value[:, 1::4, 2::4]
    total += value[:, 2::4, 1::4]
    total += value[:, 2::4, 2::4]
    out = np.full(OUTPUT_SHAPE, 114, dtype=np.uint8)
    out[:, 3:-3, :] = (total + 2) // 4
    return out


def verify_native_fixtures(checkpoint, manifest):
    original = BENCH / "inputs-after"
    final = BENCH / "final"
    spec_path, report_path = original / "spec.json", final / "native.json"
    spec, report = (json.loads(p.read_text(encoding="utf-8")) for p in (spec_path, report_path))
    expected_names = ["constant", "rgb_ramp", "thin_lines", "checkerboard", "heldout"]
    if (spec["checkpoint_sha256"] != checkpoint or spec["manifest_sha256"] != manifest
            or report["spec_sha256"] != digest(spec_path)
            or report["status"] != "AI2D_AND_DECODE_COMPLETE"
            or report["source_sha256"] != spec["board_sources"]
            or [x["name"] for x in spec["fixtures"]] != expected_names
            or [x["name"] for x in report["preprocessing"]] != expected_names):
        raise ValueError("Frozen native fixture identity differs")
    matches = []
    for item, run in zip(spec["fixtures"], report["preprocessing"]):
        name = item["name"]
        raw = (original / "heldout.input.rgb").read_bytes() if name == "heldout" else fixture(name)
        observed = (final / (name + ".ai2d.rgb")).read_bytes()
        if (item["input_sha256"] != sha256(raw).hexdigest()
                or run["input_sha256"] != item["input_sha256"]
                or run["output"] != name + ".ai2d.rgb"
                or run["output_sha256"] != sha256(observed).hexdigest()):
            raise ValueError(f"Native fixture bytes differ for {name}")
        emulated = resize_ai2d(np.frombuffer(raw, np.uint8).reshape(INPUT_SHAPE)).tobytes()
        if observed != emulated:
            raise ValueError(f"AI2D emulation is not byte exact for {name}")
        matches.append(dict(name=name, matches_bytes=len(observed), sha256=run["output_sha256"]))
    return dict(spec_sha256=digest(spec_path), native_report_sha256=digest(report_path),
                native_firmware=report["firmware"], native_runtime=report["runtime"],
                heldout_source_size=spec["fixtures"][-1]["source_size"], fixtures=matches)


def get_views(data, row):
    path = (data / row["image"]).resolve()
    if not path.is_relative_to(data) or digest(path) != row["image_sha256"]:
        raise ValueError(f"Held-out RGB identity differs: {row['sample_id']}")
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (320, 180):
            raise ValueError(f"Unexpected G5 image mode/size: {row['sample_id']}")
        intermediate = image.resize((640, 360), Image.Resampling.BILINEAR)
        return (pixels(image), pixels(intermediate),
                resize_ai2d(np.asarray(intermediate).transpose(2, 0, 1)))


@torch.inference_mode()
def replay(model, rows, data):
    scores = {mode: [] for mode in MODES}
    pixel_mae = {mode: [] for mode in MODES[1:]}
    for start in range(0, len(rows), 48):
        chunk = rows[start:start + 48]
        batches = [np.stack(items) for items in zip(*(get_views(data, row) for row in chunk))]
        for mode, batch in zip(MODES, batches):
            value = torch.from_numpy(batch).float() / 255
            scores[mode].append(model(value).numpy())
        for mode, batch in zip(MODES[1:], batches[1:]):
            delta = np.abs(batch.astype(np.int16) - batches[0].astype(np.int16))
            pixel_mae[mode].append(delta.mean(axis=(1, 2, 3)))
    return ({mode: np.concatenate(parts) for mode, parts in scores.items()},
            {mode: np.concatenate(parts) for mode, parts in pixel_mae.items()})


def outcome(candidate, first, shortlist, truth, loc):
    distance = np.linalg.norm(candidate - truth[:, None, :], axis=-1)
    prob1, accept = calibrated(first, loc.cal)
    prob5, _ = calibrated(shortlist, loc.set_cal)
    hit1 = distance[:, 0] <= 50
    hit5 = distance.min(1) <= 50
    return hit1, hit5, dict(
        frames=len(truth), top1_hits_50mm=int(hit1.sum()), top5_hits_50mm=int(hit5.sum()),
        top1_median_mm=float(np.median(distance[:, 0])),
        top1_p95_mm=float(np.quantile(distance[:, 0], .95)),
        top5_minimum_median_mm=float(np.median(distance.min(1))),
        top1_brier50=float(np.mean((prob1 - hit1) ** 2)),
        top5_brier50=float(np.mean((prob5 - hit5) ** 2)),
        top5_mean_confidence=float(prob5.mean()),
        primary_xy_frames=int(accept.sum()) if loc.primary_qualified else 0)


def compare(scores, pixel_mae, rows, groups, loc):
    truth = np.asarray([xy(row) for row in rows])
    decoded = {mode: decode_scores(logits, loc.centers) for mode, logits in scores.items()}
    baseline = decoded[MODES[0]][0]
    out = {}
    for group, select in groups.items():
        per_mode = {}
        hits = {}
        for mode, (candidate, first, shortlist) in decoded.items():
            hit1, hit5, report = outcome(candidate[select], first[select], shortlist[select], truth[select], loc)
            hits[mode] = hit1, hit5
            per_mode[mode] = report
            if mode == MODES[0]:
                continue
            delta = np.abs(scores[mode][select] - scores[MODES[0]][select])
            base1, base5 = hits[MODES[0]]
            report.update(
                pixel_mae_u8=float(pixel_mae[mode][select].mean()),
                pixel_frame_mae_p95_u8=float(np.quantile(pixel_mae[mode][select], .95)),
                logit_mae=float(delta.mean()), logit_max=float(delta.max()),
                top1_changed_frames=int(np.any(candidate[select, 0] != baseline[select, 0], axis=1).sum()),
                top5_order_changed_frames=int(np.any(candidate[select] != baseline[select], axis=(1, 2)).sum()),
                top1_hit_losses=int((base1 & ~hit1).sum()), top1_hit_gains=int((~base1 & hit1).sum()),
                top5_hit_losses=int((base5 & ~hit5).sum()), top5_hit_gains=int((~base5 & hit5).sum()))
        out[group] = per_mode
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(4)
    loc = localizer(args.provider)
    data = args.data.resolve()
    manifest = data / "manifest.jsonl"
    if digest(manifest) != loc.manifest_sha or loc.primary_qualified:
        raise ValueError("Benchmark requires the original G5 HOLD checkpoint and manifest")
    native = verify_native_fixtures(loc.checkpoint_sha, loc.manifest_sha)
    full_rows = load_rows(manifest)
    rows = [row for row in full_rows if row["split"] == "test"]
    variants = np.asarray([row["variant"] for row in rows])
    groups = dict(spatial_holdout=variants < 2, appearance_holdout=variants == 2,
                  repeated_straight_holdout=np.asarray(
                      [row["ambiguity_region"] == "repeated_straight" for row in rows]))
    if (len(rows) != 3000 or len({row["image_sha256"] for row in rows}) != len(rows)
            or [int(mask.sum()) for mask in groups.values()] != [2000, 1000, 1488]):
        raise ValueError("The frozen G5 holdout denominators differ")
    for row in rows:
        xy(row)
    scores, pixel_mae = replay(loc.net, rows, data)
    result = compare(scores, pixel_mae, rows, groups, loc)
    metrics_path = args.provider / "metrics.json"
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))[loc.method]["metrics"]
    for group, measured in result.items():
        baseline = measured["training_pil"]
        frozen = saved[group]
        frames = frozen["frames"]
        paired_hits = (round(frozen["within_mm"]["50"] * frames),
                       round(frozen["best_of_k_within50"]["5"] * frames))
        if (baseline["frames"] != frames
                or (baseline["top1_hits_50mm"], baseline["top5_hits_50mm"]) != paired_hits):
            raise ValueError(f"Training PIL baseline did not reproduce G5 hit counts: {group}")
    report = dict(
        status="HOST_PREPROCESSING_REPLAY_COMPLETE", method=loc.method,
        scope="Frozen learned_map CPU comparisons; synthetic imagery, no UVC conversion or KPU accuracy",
        single_point_gate="HOLD", checkpoint_sha256=loc.checkpoint_sha,
        manifest_sha256=loc.manifest_sha, source_sha256=digest(__file__),
        modes=dict(training_pil="Original 320x180 render through model.pixels",
                   pil_through_640="Upsample 320x180 to 640x360 with PIL bilinear, then model.pixels",
                   ai2d_through_640="Same 640x360 intermediate, then verified host AI2D emulation"),
        versions=dict(torch=torch.__version__, pillow=pillow_version, numpy=np.__version__),
        native_tensor_parity=native, metrics=result,
        g5_baseline_hit_counts_reproduced=True, g5_metrics_sha256=digest(metrics_path))
    write_json(args.out, report)
    print(json.dumps(dict(status=report["status"], frames=len(rows), native_fixtures=len(native["fixtures"]),
                          output_sha256=digest(args.out))))


if __name__ == "__main__":
    main()
