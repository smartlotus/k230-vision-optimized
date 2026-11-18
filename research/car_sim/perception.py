"""Evaluate the frozen host ONNX pair on robot-view Blender PNGs.

Example: python -B perception.py --hub <k230-project> --frames <rgb-dir> --output sim.json
Optional truth JSON maps frames by POSIX relative filename under "frames".
Each frame may have "road_mask" (relative to the truth JSON; PNG class IDs 0..6,
255 ignored) and/or "topfaces" (visible labeled centers in input image pixels,
e.g. [{"color_id": 1, "center_xy": [320, 180]}]). Omitting a field leaves that
head unscored; [] explicitly labels a topface-negative image.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


ROAD_DIR = Path("webhelp/K230_RoadNet_Boundary_Sim_Execution_Package/experiments/"
                "roadnet_boundary_sim_v1/evidence/delivery/roadnet-20261117/model")
TOP_DIR = Path("TopFaceNet-V1.1-Complete-a01/models")
ROAD_SHA = "7e6b6d6b0d58b985d34c178299f256ae57fce958b50f07c6e69ee9872665cbb5"
TOP_SHA = "833e5786d3a0385b75ee38638993c56adbcce30165e5f67a02f26ebebd7b0ad4"
ROAD_CLASSES = ("ROAD_GRAY", "FORBIDDEN_YELLOW", "LIGHT_BACKGROUND", "BLUE_START_ZONE",
                "BLACK_OBSTACLE", "MATERIAL_OBJECT", "SPECIAL_MARKING")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_session(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Frozen model SHA-256 mismatch for {path}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def camera_source(bgr):
    height, width = bgr.shape[:2]
    if width * 9 != height * 16 or bgr.shape != (height, width, 3) or bgr.dtype != np.uint8:
        raise ValueError("Expected a uint8 BGR image with 16:9 aspect ratio")
    return cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_LINEAR)


def road_outputs(session, source):
    """Return the unpadded normalized RGB and 7 frozen logits per content pixel."""
    content = cv2.cvtColor(cv2.resize(source, (320, 180), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    padded = cv2.copyMakeBorder(content, 6, 6, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255
    logits = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    if logits.shape != (1, 7, 192, 320) or not np.isfinite(logits).all():
        raise ValueError("Invalid RoadNet output")
    return tensor[0, :, 6:186].transpose(1, 2, 0), logits[0, :, 6:186].transpose(1, 2, 0)


class FrozenPerception:
    def __init__(self, hub):
        hub = Path(hub)
        decoder = Path(__file__).resolve().parents[1] / "topface/task_geometry.py"
        if not decoder.is_file():
            decoder = hub / "worktrees/ops-topface-production/research/topface/task_geometry.py"
        module = importlib.util.spec_from_file_location("topface_task_geometry", decoder)
        self.geometry = importlib.util.module_from_spec(module)
        module.loader.exec_module(self.geometry)
        self.decoder_sha = digest(decoder)
        self.road = load_session(hub / ROAD_DIR / "roadnet.onnx", ROAD_SHA)
        self.top = load_session(hub / TOP_DIR / "topface.onnx", TOP_SHA)

    def observe(self, bgr):
        """Return the 320×180 road class mask and decoded topfaces in 640×360 coordinates."""
        source = camera_source(bgr)
        _, logits = road_outputs(self.road, source)
        labels = logits.argmax(axis=2).astype(np.uint8)
        top_input, _ = self.geometry.preprocess_bgr(source)
        raw = self.top.run(None, {self.top.get_inputs()[0].name: top_input[None]})[0]
        return labels, self.geometry.decode(raw)


def load_road_truth(mask_file):
    with Image.open(mask_file) as image:
        truth = np.asarray(image)
    if truth.ndim != 2 or truth.dtype != np.uint8 or 9 * truth.shape[1] != 16 * truth.shape[0]:
        raise ValueError(f"Expected 16:9 indexed/8-bit grayscale road labels in {mask_file}")
    if not np.isin(truth, [*range(7), 255]).all():
        raise ValueError(f"Invalid road class in {mask_file}")
    return cv2.resize(truth, (320, 180), interpolation=cv2.INTER_NEAREST_EXACT)


def road_confusion(labels, mask_file):
    truth = load_road_truth(mask_file)
    valid = truth != 255
    return np.bincount(truth[valid] * 7 + labels[valid], minlength=49).reshape(7, 7)


def score_centers(expected, predictions, width, height, radius):
    """Maximize the one-to-one correct-color matches in radius, then minimize distance."""
    from scipy.optimize import linear_sum_assignment

    centers, colors = [], []
    for obj in expected:
        xy = np.asarray(obj["center_xy"], dtype=float)
        color = obj["color_id"]
        if (type(color) is not int or color not in range(1, 7) or xy.shape != (2,)
                or not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy >= (width, height))):
            raise ValueError(f"Invalid visible topface label {obj}")
        centers.append(xy * [640 / width, 360 / height])
        colors.append(color)
    errors = []
    if centers and predictions:
        predicted = np.asarray([obj["center_xy_projected_3d"] for obj in predictions])
        distances = np.linalg.norm(np.asarray(centers)[:, None, :] - predicted[None], axis=2)
        same_color = np.equal.outer(colors, [obj["color_id"] for obj in predictions])
        eligible = same_color & (distances <= radius)
        cost = np.where(eligible, distances, (min(distances.shape) + 1) * radius)
        rows, cols = linear_sum_assignment(cost)
        errors = [float(distances[i, j]) for i, j in zip(rows, cols) if eligible[i, j]]
    return {"matched": len(errors), "missed": len(centers) - len(errors),
            "unmatched_predictions": len(predictions) - len(errors), "errors_px": errors}


def run(hub, frames_dir, truth_path=None, radius=12):
    frames_dir = Path(frames_dir)
    files = sorted(p for p in frames_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".png")
    if not files:
        raise ValueError(f"No PNG camera frames in {frames_dir}")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Positive finite match radius required")
    truth = json.loads(truth_path.read_text(encoding="utf-8"))["frames"] if truth_path else {}
    paths = {p.relative_to(frames_dir).as_posix() for p in files}
    if set(truth) - paths:
        raise ValueError(f"Truth references missing frames: {sorted(set(truth) - paths)}")
    models = FrozenPerception(hub)
    reports = []
    confusion = np.zeros((7, 7), dtype=np.int64)
    top_scores = []
    road_frames = 0
    for filename in files:
        bgr = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Invalid camera frame {filename}")
        height, width = bgr.shape[:2]
        labels, decoded = models.observe(bgr)
        rel = filename.relative_to(frames_dir).as_posix()
        row = {"file": rel, "sha256": digest(filename), "input_wh": [width, height],
               "road_class_pixels": np.bincount(labels.ravel(), minlength=7).tolist(),
               "topface": decoded}
        annotation = truth.get(rel, {})
        if "road_mask" in annotation:
            cm = road_confusion(labels, truth_path.parent / annotation["road_mask"])
            confusion += cm
            road_frames += 1
            row["road_confusion"] = cm.tolist()
        if "topfaces" in annotation:
            score = score_centers(annotation["topfaces"], decoded["detections"], width, height, radius)
            row["topface_matching"] = score
            top_scores.append(score)
        reports.append(row)
    intersection = int(confusion[0, 0])
    union = int(confusion[0].sum() + confusion[:, 0].sum() - intersection)
    return {"scope": "Frozen float ONNX inference on supplied rendered camera images",
            "frames_evaluated": len(reports), "backend": ort.__version__, "provider": "CPUExecutionProvider",
            "topface_onnx_sha256": TOP_SHA, "roadnet_onnx_sha256": ROAD_SHA,
            "topface_geometry_sha256": models.decoder_sha, "road_classes": ROAD_CLASSES,
            "road_truth_frames": road_frames, "road_confusion_gt_rows": confusion.tolist() if road_frames else None,
            "gray_road_iou": intersection / union if union else None,
            "topface_truth_frames": len(top_scores), "topface_match_radius_640x360_px": radius,
            "topface_matching_sum": {k: sum(s[k] for s in top_scores) for k in
                                     ("matched", "missed", "unmatched_predictions")} if top_scores else None,
            "frames": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--match-px", type=float, default=12)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path to preserve the previous run")
    report = run(args.hub, args.frames, args.truth, args.match_px)
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(serialized + "\n")
    print(json.dumps({k: report[k] for k in ("frames_evaluated", "road_truth_frames", "gray_road_iou",
                                              "topface_truth_frames", "topface_matching_sum")}))
