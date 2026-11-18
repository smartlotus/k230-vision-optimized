"""Train a simulation-only binary gray-road segmentation head over frozen RoadNet features.

Input JSON: {"frames":[{"group":"episode1","split":"train","rgb":"rgb/1.png",
"mask":"semantic/1.png","pose":"extended"}]}. Use independent episodes for
train, val, and test. RGB and indexed 0..6/255 mask paths are relative to JSON.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from perception import ROAD_DIR, ROAD_SHA, camera_source, digest, load_road_truth, load_session, road_outputs


SPLITS = ("train", "val", "test")
FEATURES = tuple(f"road_logit_{i}" for i in range(7)) + ("R", "G", "B", "chroma", "brightness_sq")
L2 = 0.005
THRESHOLDS = np.linspace(0.05, 1.0, 20)


def features(rgb, logits):
    chroma = np.ptp(rgb, axis=2, keepdims=True)
    brightness_sq = np.square(rgb.mean(axis=2, keepdims=True))
    return np.concatenate((logits, rgb, chroma, brightness_sq), axis=2)


def read_manifest(manifest):
    rows = json.loads(manifest.read_text(encoding="utf-8"))["frames"]
    groups = {}
    buckets = {s: [] for s in SPLITS}
    for row in rows:
        split, group = row["split"], row["group"]
        if split not in SPLITS or not isinstance(group, str) or not group:
            raise ValueError(f"Invalid split/group in {row}")
        if groups.setdefault(group, split) != split:
            raise ValueError(f"Episode {group} crosses splits")
        buckets[split].append(row)
    if not all(buckets.values()):
        raise ValueError("Explicit train, val and test episodes required")
    return buckets


def samples_by_frame(matrix, target, rng, max_count):
    """Equal cap for each binary class per frame; the threshold is fit on full val images."""
    indices = []
    for road in (False, True):
        possible = np.flatnonzero(target == road)
        indices.extend(rng.choice(possible, min(possible.size, max_count // 2), replace=False))
    return matrix[indices], target[indices]


def fit(training, target):
    if np.unique(target).size != 2:
        raise ValueError("Training needs both gray-road and non-road labels")
    mean = training.mean(axis=0, dtype=np.float64)
    std = np.maximum(training.std(axis=0, dtype=np.float64), 1.e-5)
    x = (training - mean) / std

    def objective(p):
        w, bias = p[:-1], p[-1]
        z = np.einsum("ij,j->i", x, w) + bias
        residual = expit(z) - target
        loss = np.mean(np.logaddexp(0, z) - target * z) + L2 * np.dot(w, w) / 2
        gradient = np.einsum("ij,i->j", x, residual) / len(target) + L2 * w
        return loss, np.r_[gradient, residual.mean()]

    fit_result = minimize(objective, np.zeros(x.shape[1] + 1), method="L-BFGS-B",
                          jac=True, options={"maxiter": 150, "ftol": 1.e-10})
    if not fit_result.success or not np.isfinite(fit_result.x).all():
        raise ValueError(f"Logistic fit failed: {fit_result.message}")
    # Compose standardization with the head, so inference requires one dot product.
    weights = fit_result.x[:-1] / std
    bias = fit_result.x[-1] - np.dot(weights, mean)
    return weights, bias, {"iterations": int(fit_result.nit), "loss": float(fit_result.fun)}


def probabilities(rgb, logits, weights, bias):
    return expit(np.einsum("...j,j->...", features(rgb, logits), weights) + bias)


def confusion(target, pred):
    # [TN, FP, FN, TP]; inputs are boolean arrays from valid labels only.
    return np.bincount(target.astype(np.uint8) * 2 + pred, minlength=4)


def metrics(cm):
    tn, fp, fn, tp = map(int, cm)
    def ratio(a, b):
        return a / b if b else None
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "iou": ratio(tp, tp + fp + fn),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "false_drivable_rate": ratio(fp, fp + tn)}


def sum_counts(vectors):
    return np.sum(vectors, axis=0, dtype=np.int64)


def evaluate(rows, threshold):
    frame_scores = []
    accum = {"all": [], "group": {}, "pose": {}}
    for row in rows:
        pair = np.stack((row["baseline"], confusion(row["y"], row["p"] >= threshold)))
        scores = {"rgb": row["rgb"], "group": row["group"], "pose": row["pose"],
                  "frozen": metrics(pair[0]), "adapter": metrics(pair[1])}
        frame_scores.append(scores)
        accum["all"].append(pair)
        for key in ("group", "pose"):
            accum[key].setdefault(row[key], []).append(pair)

    def pack(values):
        frozen, adapter = sum_counts(values)
        return {"frames": len(values), "frozen": metrics(frozen), "adapter": metrics(adapter)}

    return {"all": pack(accum["all"]),
            "by_episode": {k: pack(v) for k, v in accum["group"].items()},
            "by_pose": {k: pack(v) for k, v in accum["pose"].items()}, "frames": frame_scores}


def supported(summary):
    base, adapt = summary["frozen"], summary["adapter"]
    return (base["iou"] is not None and adapt["iou"] is not None
            and base["false_drivable_rate"] is not None
            and adapt["iou"] > base["iou"] and adapt["false_drivable_rate"] <= base["false_drivable_rate"])


def select_threshold(rows):
    frozen = metrics(sum_counts([row["baseline"] for row in rows]))
    candidates = []
    for threshold in THRESHOLDS:
        cm = sum_counts([confusion(row["y"], row["p"] >= threshold) for row in rows])
        score = metrics(cm)
        candidates.append({"threshold": float(threshold), **score})
    # Threshold=1 is the conservative fallback when no other candidate is eligible.
    eligible = [c for c in candidates if c["false_drivable_rate"] <= frozen["false_drivable_rate"]]
    winner = max(eligible or candidates, key=lambda c: (c["iou"], -c["false_drivable_rate"], c["threshold"]))
    return winner["threshold"], candidates


def run(hub, manifest, out, seed=230, per_frame=4096):
    if out.exists():
        raise FileExistsError(f"Select a new experiment directory: {out}")
    if per_frame < 2:
        raise ValueError("At least two training samples per frame required")
    buckets = read_manifest(manifest)
    session = load_session(hub / ROAD_DIR / "roadnet.onnx", ROAD_SHA)
    rng = np.random.default_rng(seed)
    seen = {}
    inventory = []

    def load_split(split):
        for row in buckets[split]:
            image_file = manifest.parent / row["rgb"]
            mask_file = manifest.parent / row["mask"]
            bgr = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Invalid frame {image_file}")
            source = camera_source(bgr)
            source_id = hashlib.sha256(source.tobytes()).hexdigest()
            item = {**row, "rgb_sha256": digest(image_file), "mask_sha256": digest(mask_file),
                    "pixel_sha256_640x360": source_id}
            inventory.append(item)
            if source_id in seen:
                item["excluded"] = "Identical model-input RGB in " + seen[source_id]
                continue
            seen[source_id] = split + "/" + row["rgb"]
            truth = load_road_truth(mask_file)
            valid = truth != 255
            if not valid.any():
                item["excluded"] = "No labeled pixels"
                continue
            rgb, logits = road_outputs(session, source)
            target = truth[valid] == 0
            base = logits[valid].argmax(axis=-1) == 0
            item["class_pixels"] = np.bincount(truth[valid], minlength=7).tolist()
            yield row, rgb, logits, target, valid, confusion(target, base)

    train_x, train_y = [], []
    for _, rgb, logits, y, valid, _ in load_split("train"):
        x, y = samples_by_frame(features(rgb, logits)[valid], y, rng, per_frame)
        train_x.append(x)
        train_y.append(y)
    if not train_x:
        raise ValueError("No usable training observations")
    samples = np.concatenate(train_y)
    weights, bias, fit_info = fit(np.concatenate(train_x), samples)

    def score_split(split):
        scored = []
        for row, rgb, logits, y, valid, baseline in load_split(split):
            p = probabilities(rgb, logits, weights, bias)[valid]
            scored.append({"rgb": row["rgb"], "group": row["group"], "pose": row.get("pose", row.get("state", "unspecified")),
                           "baseline": baseline, "p": p, "y": y})
        if not scored:
            raise ValueError(f"No usable unique {split} frames")
        tn, fp, fn, tp = sum_counts([s["baseline"] for s in scored])
        if min(tp + fn, tn + fp) == 0:
            raise ValueError(f"No class contrast in {split}; both road and non-road required for evaluation")
        return scored

    val_rows = score_split("val")
    threshold, sweep = select_threshold(val_rows)
    validation = evaluate(val_rows, threshold)
    # Test labels are read only after coefficients and decision threshold are fixed.
    test = evaluate(score_split("test"), threshold)
    sim_supported = supported(validation["all"]) and supported(test["all"])
    report = {"task": "Simulation binary gray-road segmentation", "frozen_onnx_sha256": ROAD_SHA,
              "scope": "Rendered episode evaluation; controller and physical vehicle performance unevaluated",
              "split_manifest_sha256": digest(manifest), "seed": seed, "train_max_samples_per_frame": per_frame,
              "used_train_samples": len(samples), "positive_train_samples": int(samples.sum()),
              "features": FEATURES, "regularization": L2, "training": fit_info,
              "selection": "Maximize validation IoU with false-drivable rate no higher than frozen baseline",
              "sim_support_rule": "Strictly greater pooled IoU and no increase in pooled false-drivable rate on both val and test",
              "status": "SUPPORTED_ON_RENDERED_SPLITS" if sim_supported else "RETAIN_FROZEN_BASELINE",
              "threshold": threshold, "threshold_sweep_val": sweep, "validation": validation,
              "test": test, "inputs": inventory}
    out.mkdir(parents=True, exist_ok=False)
    artifact = out / "candidate.npz"
    np.savez(artifact, weights=weights, bias=bias, threshold=threshold, features=np.array(FEATURES),
             frozen_onnx_sha256=ROAD_SHA)
    report["weights_sha256"] = digest(artifact)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--max-samples-per-frame", type=int, default=4096)
    args = parser.parse_args()
    result = run(args.hub, args.manifest, args.out, args.seed, args.max_samples_per_frame)
    print(json.dumps({"status": result["status"], "test": result["test"]["all"]}))
