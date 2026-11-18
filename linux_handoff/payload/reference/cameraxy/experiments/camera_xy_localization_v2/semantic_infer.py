"""Host semantic CameraXY shortlist and candidate-set confidence research."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from baselines import mask_inputs, semantic, template_map
from contract import TARGETS, digest, load_rows, xy
from train import calibrate, calibrated


def evaluate_candidate_sets(archive, data, seed):
    fields = ("appearance_pair_nominal", "spatial_holdout", "appearance_holdout",
              "repeated_straight_holdout")
    rows = load_rows(data / "manifest.jsonl")
    by_id = {r["sample_id"]: r for r in rows}
    val = np.array([r["sample_id"] for r in rows if r["split"] == "val" and r["variant"] == 0])
    val = np.random.default_rng(seed).permutation(val)
    with np.load(archive, allow_pickle=False) as predictions:
        ids = predictions["sample_ids"]
        example = [by_id[i] for i in ids]
        truth = np.array([xy(r) for r in example])
        if not np.allclose(predictions["truth_mm"], truth, atol=.001, rtol=0):
            raise ValueError("Prediction bundle does not match dataset optical center labels")
        cal_rows = np.isin(ids, val[len(val)//2:])
        test = np.array([r["split"] == "test" for r in example])
        variant = np.array([r["variant"] for r in example])
        repeated = np.array([r["ambiguity_region"] == "repeated_straight" for r in example])
        sets = {}
        for name in ("semantic_oracle", "semantic_roadnet"):
            candidate = predictions[name + "_candidates_mm"]
            score = predictions[name + "_raw_confidence"]
            error = np.linalg.norm(candidate - truth[:, None], axis=2)
            minimum = error.min(1)
            ok = minimum <= 50
            calibration = calibrate(score[cal_rows], ok[cal_rows])
            p, accepted = calibrated(score, calibration)
            result = dict(calibration=calibration)
            for subset, mask in zip(fields, (test & (variant == 0), test & (variant < 2),
                                            test & (variant == 2), test & repeated)):
                result[subset] = dict(frames=int(mask.sum()), candidate_coverage50=float(ok[mask].mean()),
                                      mean_predicted_confidence=float(p[mask].mean()),
                                      brier50=float(np.mean((p[mask] - ok[mask])**2)),
                                      threshold_coverage=float(accepted[mask].mean()))
            sets[name] = result
    return dict(
        dataset_manifest_sha256=digest(data / "manifest.jsonl"),
        prediction_bundle_sha256=digest(archive), confidence_event="any one of five XY candidates within 50 mm",
        seed=seed, primary_xy_status="unlocalized", candidate_sets=sets,
    )


def localize(image, data, layout_path, onnx, report):
    manifest = digest(data / "manifest.jsonl")
    qa = json.loads((data / "qa.json").read_text(encoding="utf8"))
    if report["dataset_manifest_sha256"] != manifest or qa["field_mask_sha256"] != digest(data / "field_mask.png"):
        raise ValueError("Calibration and semantic reference do not match the frozen map input")
    layout = json.loads(layout_path.read_text(encoding="utf8"))
    margin = json.loads((data / "run.json").read_text(encoding="utf8"))["margin"]
    with Image.open(data / "field_mask.png") as world:
        templates = template_map(np.asarray(world), layout, margin)
    with Image.open(image) as frame:
        if frame.width * 9 != frame.height * 16:
            raise ValueError("Expected the recorded OV2735 16:9 geometry")
    masks = mask_inputs(image.parent, [{"image": image.name}], [0], onnx)
    candidates, raw = semantic(templates, masks, "cpu")
    cal = report["candidate_sets"]["semantic_roadnet"]["calibration"]
    p, _ = calibrated(raw, cal)
    return dict(
        status="unlocalized", camera_x_mm=None, camera_y_mm=None, confidence=None,
        candidate_set_confidence=float(p[0]), confidence_event=report["confidence_event"],
        calibration_domain="synthetic pilot spatial validation",
        top_k_camera_xy=[dict(zip(TARGETS, p.tolist())) for p in candidates[0]],
        map_sha256=qa["field_mask_sha256"], roadnet_onnx_sha256=digest(onnx),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--predictions", type=Path)
    modes.add_argument("--image", type=Path)
    for key in ("report", "layout", "roadnet", "out"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--seed", type=int, default=37121)
    a = p.parse_args()
    torch.set_num_threads(4)
    if a.predictions:
        result = evaluate_candidate_sets(a.predictions, a.data, a.seed)
    else:
        if not (a.report and a.layout and a.roadnet):
            p.error("image requires report, layout, and RoadNet ONNX")
        result = localize(a.image, a.data, a.layout, a.roadnet,
                          json.loads(a.report.read_text(encoding="utf8")))
    serialized = json.dumps(result, indent=2, allow_nan=False)
    if a.out:
        with a.out.open("x", encoding="utf8") as output:
            output.write(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
