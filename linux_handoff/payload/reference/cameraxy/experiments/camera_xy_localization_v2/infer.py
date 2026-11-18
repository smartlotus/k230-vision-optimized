"""Host CameraXY inference and independent real OV2735 survey evaluation."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from contract import TARGETS, digest, load_rows, xy
from model import network, pixels, predicted
from train import calibrated, metrics
from train_map import candidate_set_metrics


class Localizer:
    def __init__(self, checkpoint, report):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["targets"] != TARGETS:
            raise ValueError("Expected CameraXY-only checkpoint")
        self.cell_mm = saved["cell_mm"]
        tag = f"grid{self.cell_mm}" if self.cell_mm else "direct"
        self.cal = report[tag]["confidence"]
        self.net = network(self.cell_mm).eval()
        self.net.load_state_dict(saved["weights"], strict=True)
        self.manifest_sha = saved["dataset_manifest_sha256"]
        self.checkpoint_sha = digest(checkpoint)
        if report[tag]["lineage"]["selected_epoch"] != saved["selected_epoch"]:
            raise ValueError("Report and checkpoint selection differ")

    @torch.inference_mode()
    def predict_survey(self, path):
        with Image.open(path) as image:
            if image.width * 9 != image.height * 16:
                raise ValueError("Image must use the recorded 16:9 OV2735 profile")
            value = torch.from_numpy(pixels(image).copy())[None].float() / 255
        cand, raw, _ = predicted(self.net, value, self.cell_mm)
        return cand.numpy()[0], float(raw.item()), None

    def predict_image(self, path):
        candidates, score, _ = self.predict_survey(path)
        return candidates, score

    def output(self, path):
        cand, raw = self.predict_image(path)
        confidence, accepted = calibrated(np.array([raw]), self.cal)
        result = dict(status="localized" if accepted[0] else "unlocalized",
                      confidence=float(confidence[0]), checkpoint_sha256=self.checkpoint_sha,
                      top_k_camera_xy=[dict(zip(TARGETS, p.tolist())) for p in cand])
        result.update(dict(zip(TARGETS, cand[0].tolist() if accepted[0] else (None, None))))
        return result


def survey_label(row):
    """Ground truth only from the camera optical-center survey."""
    measured = row["measurement"]
    if measured["reference"] != "DIRECT_CAMERA_OPTICAL_CENTER_FIELD_MAP_XY":
        raise ValueError("Ground truth must be the measured camera optical center")
    uncertainty = measured["uncertainty_mm"]
    if not (type(uncertainty) in (float, int) and math.isfinite(uncertainty) and uncertainty > 0
            and all(isinstance(measured[key], str) and measured[key].strip()
                    for key in ("method", "record_id"))):
        raise ValueError("Record the direct survey procedure and measured uncertainty")
    if row["camera_metadata"]["sensor"] != "USB_OV2735":
        raise ValueError("Expected OV2735 survey profile")
    return xy(row), uncertainty


def survey_source(camera_metadata):
    source = camera_metadata.get("capture_source", "OPERATOR_OV2735")
    if camera_metadata.get("data_status") == "SYNTHETIC_TEST_FIXTURE_ONLY":
        return "SYNTHETIC_FIXTURE"
    if source not in ("OPERATOR_OV2735", "SYNTHETIC_FIXTURE"):
        raise ValueError("Unknown capture source")
    return source


def real_evaluate(localizer, manifest):
    manifest_sha = digest(manifest)
    rows = load_rows(manifest)
    if not rows:
        raise ValueError("Real test manifest is empty")
    root = manifest.parent.resolve()
    keys = {"sample_id", "image", "image_sha256", "camera_xy_target", "camera_metadata",
            "measurement", "survey_position_id"}
    sample_ids, image_hashes, sources = set(), set(), set()
    positions, position_ids, samples = {}, {}, []
    for row in rows:
        if set(row) != keys or any(not isinstance(row[key], str) or not row[key].strip()
                                   for key in ("sample_id", "survey_position_id", "image")):
            raise ValueError("Invalid real CameraXY manifest row")
        sample = row["sample_id"]
        if sample in sample_ids:
            raise ValueError("Duplicate survey sample ID")
        sample_ids.add(sample)
        pair, uncertainty = survey_label(row)
        sources.add(survey_source(row["camera_metadata"]))
        group = row["survey_position_id"]
        if positions.setdefault(group, pair) != pair:
            raise ValueError("Survey position labels changed between captures")
        if position_ids.setdefault(pair, group) != group:
            raise ValueError("Use the same survey position ID for repeated captures at one XY")
        path = Path(row["image"])
        image = (root / path).resolve()
        image_hash = row["image_sha256"]
        if path.is_absolute() or not image.is_relative_to(root) or digest(image) != image_hash:
            raise ValueError("Survey image path or SHA256 invalid")
        if image_hash in image_hashes:
            raise ValueError("Duplicate survey image bytes")
        image_hashes.add(image_hash)
        samples.append((image, pair, uncertainty))
    if len(sources) != 1:
        raise ValueError("Keep synthetic fixtures and operator captures in separate manifests")

    simulated_fixture = "SYNTHETIC_FIXTURE" in sources
    qualified = bool(getattr(localizer, "primary_qualified", True)) and not simulated_fixture
    primary_cal = localizer.cal if qualified else dict(localizer.cal, threshold=None)
    predictions = [localizer.predict_survey(item[0]) for item in samples]
    candidate_parts, raw, set_raw = zip(*predictions)
    candidates = np.stack(candidate_parts)
    truth = np.asarray([item[1] for item in samples])
    uncertainty = [item[2] for item in samples]
    raw = np.asarray(raw)
    primary, accepted = calibrated(raw, primary_cal)
    distances = np.linalg.norm(candidates - truth[:, None, :], axis=2)
    set_cal = getattr(localizer, "set_cal", None)
    set_metrics, set_probabilities = None, [None] * len(samples)
    if set_cal is not None:
        set_raw = np.asarray(set_raw, dtype=float)
        set_probabilities, _ = calibrated(set_raw, set_cal)
        set_metrics = candidate_set_metrics(truth, candidates, set_raw, set_cal)

    observations = []
    for i, row in enumerate(rows):
        observations.append(dict(
            sample_id=row["sample_id"], survey_position_id=row["survey_position_id"],
            image=row["image"], image_sha256=row["image_sha256"],
            measurement_record_id=row["measurement"]["record_id"],
            uncertainty_mm=uncertainty[i], camera_xy_target=row["camera_xy_target"],
            primary_accepted=bool(accepted[i]), top1_error_mm=float(distances[i, 0]),
            nearest_candidate_error_mm=float(distances[i].min()), confidence=float(primary[i]),
            candidate_set_confidence=None if set_probabilities[i] is None else float(set_probabilities[i]),
            candidate_error_mm=distances[i].tolist(),
            top_k_camera_xy=[dict(zip(TARGETS, point.tolist())) for point in candidates[i]]))
    if digest(manifest) != manifest_sha:
        raise ValueError("Survey manifest changed during evaluation")
    return dict(
        status="SYNTHETIC_SURVEY_FIXTURE" if simulated_fixture else "MEASURED_REAL_OV2735",
        capture_source=next(iter(sources)), contents_sha256=manifest_sha, surveyed_positions=len(positions),
        training_manifest_sha256=localizer.manifest_sha, checkpoint_sha256=localizer.checkpoint_sha,
        primary_qualified=qualified, confidence_domain="frozen synthetic spatial calibration",
        ground_truth_uncertainty_mm=dict(median=float(np.median(uncertainty)), max=max(uncertainty)),
        metrics=metrics(candidates, truth, raw, primary_cal),
        candidate_set_metrics=set_metrics, observations=observations,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--out", type=Path)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path)
    inputs.add_argument("--real-manifest", type=Path)
    a = p.parse_args()
    torch.set_num_threads(2)
    report = json.loads(a.metrics.read_text(encoding="utf8"))
    model = Localizer(a.checkpoint, report)
    result = model.output(a.image) if a.image else real_evaluate(model, a.real_manifest)
    data = json.dumps(result, indent=2, allow_nan=False)
    if a.out:
        with a.out.open("x", encoding="utf8") as file:
            file.write(data + "\n")
    print(data)


if __name__ == "__main__":
    main()
