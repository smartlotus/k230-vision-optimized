"""Load the frozen XY map head for image inference and an independent real survey."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from baselines import mask_inputs, template_map
from contract import HALF_MM, TARGETS, digest
from infer import real_evaluate
from map_matcher import (EMBEDDING_DIM, FrozenMapHead, MapMatcher, analytic_masks_scores,
                         decode_scores)
from model import pixels
from train import calibrated
from train_map import SEMANTIC_LOGIT_SCALE, fused_scores

LIVE_METHODS = ("learned_map", "fused_map", "semantic_roadnet")


class MapLocalizer:
    def __init__(self, checkpoint, metrics_path, result, method=None, data=None, layout=None, roadnet=None):
        self.checkpoint_sha = digest(checkpoint)
        if (result["inference_sha256"] != self.checkpoint_sha
                or result["calibration_metrics_sha256"] != digest(metrics_path)):
            raise ValueError("Frozen map checkpoint and calibration report identity differ")
        state = torch.load(checkpoint, weights_only=True, map_location="cpu")
        self.manifest_sha = state["dataset_manifest_sha256"]
        if (state["targets"] != TARGETS or not state["orientations_collapsed"]
                or self.manifest_sha != result["dataset_manifest_sha256"]
                or state["selected_epoch"] != result["selected_epoch"]):
            raise ValueError("Checkpoint must contain XY-only map inference for this run")
        self.method = method or result["selected_method_by_validation_top5"]
        if self.method not in LIVE_METHODS:
            raise ValueError("Unsupported CameraXY RGB inference method")
        self.centers = np.asarray(state["centers_mm"], dtype=np.float64)
        if (self.centers.ndim != 2 or self.centers.shape[1] != 2 or len(self.centers) < 5
                or not np.isfinite(self.centers).all() or np.abs(self.centers).max() > HALF_MM):
            raise ValueError("Map anchor coordinates outside the FIELD_MAP contract")
        shape = (len(self.centers), state["reference_views"], EMBEDDING_DIM)
        self.net = FrozenMapHead(MapMatcher(), torch.zeros(shape)).eval()
        self.net.load_state_dict(state["state"], strict=True)
        method_metrics = json.loads(metrics_path.read_text(encoding="utf8"))[self.method]
        self.cal = method_metrics["primary_confidence"]
        self.set_cal = method_metrics["set_confidence"]
        self.primary_qualified = (result["full_gate"] == "PASS"
                                  and self.method == result["selected_method_by_validation_top5"])
        if self.method != "learned_map":
            if not (data and layout and roadnet):
                raise ValueError("Supply the selected map, layout and frozen RoadNet ONNX")
            self.roadnet = roadnet
            env = result["semantic_inputs"]
            if (digest(data / "field_mask.png") != env["field_mask_sha256"]
                    or digest(layout) != env["layout_sha256"] or digest(roadnet) != env["roadnet_sha256"]):
                raise ValueError("Semantic inputs differ from the frozen calibration inputs")
            with Image.open(data / "field_mask.png") as img:
                self.templates = template_map(np.asarray(img), json.loads(layout.read_text(encoding="utf8")),
                                              env["margin_mm"])
            if not np.array_equal(self.centers, self.templates[0]):
                raise ValueError("Semantic and learned map use different anchor coordinates")

    @torch.inference_mode()
    def _compute(self, path):
        with Image.open(path) as image:
            if image.width * 9 != image.height * 16:
                raise ValueError("Image must use the recorded 16:9 OV2735 profile")
            value = torch.from_numpy(pixels(image).copy())[None].float()/255
        if self.method == "learned_map":
            return decode_scores(self.net(value).numpy(), self.centers)
        sources = mask_inputs(path.parent, [{"image": path.name}], [0], self.roadnet)
        semantic = analytic_masks_scores(self.templates, sources, "cpu") * SEMANTIC_LOGIT_SCALE
        scores = fused_scores(self.net(value).numpy(), semantic) if self.method == "fused_map" else semantic
        return decode_scores(scores, self.centers)

    def predict_survey(self, path):
        candidates, first, top_set = self._compute(path)
        return candidates[0], float(first[0]), float(top_set[0])

    def predict_image(self, path):
        candidates, score, _ = self.predict_survey(path)
        return candidates, score

    def output(self, path):
        candidates, first, top_set = self._compute(path)
        probability, accepted = calibrated(first, self.cal)
        set_probability, _ = calibrated(top_set, self.set_cal)
        inside = self.primary_qualified and bool(accepted[0])
        report = dict(status="localized" if inside else "unlocalized",
                      method=self.method, confidence=float(probability[0]),
                      primary_qualified=self.primary_qualified,
                      candidate_set_confidence=float(set_probability[0]),
                      confidence_event="position within 50 mm",
                      candidate_set_confidence_event="one of five positions within 50 mm",
                      confidence_domain="synthetic spatial calibration",
                      checkpoint_sha256=self.checkpoint_sha, dataset_manifest_sha256=self.manifest_sha,
                      top_k_camera_xy=[dict(zip(TARGETS, point.tolist())) for point in candidates[0]])
        report.update(dict(zip(TARGETS, candidates[0, 0].tolist() if inside else (None, None))))
        return report


def main():
    parser = argparse.ArgumentParser()
    for key in ("checkpoint", "metrics", "result"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("data", "layout", "roadnet", "out"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--method", choices=LIVE_METHODS)
    input_arg = parser.add_mutually_exclusive_group(required=True)
    input_arg.add_argument("--image", type=Path)
    input_arg.add_argument("--real-manifest", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    localizer = MapLocalizer(args.checkpoint, args.metrics,
                             json.loads(args.result.read_text(encoding="utf8")),
                             args.method, args.data, args.layout, args.roadnet)
    out = (localizer.output(args.image) if args.image else
           dict(real_evaluate(localizer, args.real_manifest), method=localizer.method))
    doc = json.dumps(out, indent=2, allow_nan=False)
    if args.out:
        with args.out.open("x", encoding="utf8") as target:
            target.write(doc + "\n")
    print(doc)


if __name__ == "__main__":
    main()
