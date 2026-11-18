"""Compile a frozen CameraXY candidate for hardware smoke tests, without accuracy evaluation."""
import argparse
import json
import os
from pathlib import Path

import torch

from contract import digest, load_rows
from export_k230 import host_onnx, kmodel, tensor
from infer_map import MapLocalizer


def main():
    p = argparse.ArgumentParser()
    for name in ("provider", "data", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    original = json.loads((a.provider / "result.json").read_text(encoding="utf8"))
    model = MapLocalizer(a.provider / "map-inference.pt", a.provider / "metrics.json", original,
                         method="learned_map")
    if digest(a.data / "manifest.jsonl") != model.manifest_sha:
        raise ValueError("Data identity differs from the frozen model")
    rows = load_rows(a.data / "manifest.jsonl")
    train = [r for r in rows if r["split"] == "train" and r["variant"] == 0]
    calibration = [train[i * (len(train)-1) // 31] for i in range(32)]
    probes = [r for r in rows if r["split"] == "test"][:5]
    a.out.mkdir(parents=True, exist_ok=False)
    graph, onnx_runner, numerical = host_onnx(a, model, probes)
    print("ENGINEERING_ONNX_NUMERICAL", json.dumps(numerical), flush=True)
    runtime, compiled = kmodel(a, graph, calibration)
    (a.out / "camera_xy.kmodel").rename(a.out / "engineering.kmodel")
    compiled["filename"] = "engineering.kmodel"
    numeric_pairs = []
    import numpy as np
    for row in probes:
        value = tensor(a.data, row)
        reference, quantized = onnx_runner(value), runtime(value)
        if reference.shape != quantized.shape or not np.isfinite(quantized).all():
            raise ValueError("Candidate KModel ABI mismatch")
        numeric_pairs.append(dict(sample_id=row["sample_id"], image_sha256=row["image_sha256"],
                                  shape=list(quantized.shape), max_abs=float(np.max(np.abs(reference-quantized))),
                                  mean_abs=float(np.mean(np.abs(reference-quantized)))))
    report = dict(schema="CAMERA_XY_ENGINEERING_ONLY_V1", purpose="USER_AUTHORIZED_HARDWARE_SMOKE",
        production_eligible=False, require_unlocalized=True, accuracy_test="NOT_RUN_BY_USER_SCOPE",
        road_quality_test="NOT_RUN_BY_USER_SCOPE", host_scientific_gate=original["full_gate"],
        frozen_selected_method=original["selected_method_by_validation_top5"], method="learned_map",
        checkpoint_sha256=model.checkpoint_sha, training_manifest_sha256=model.manifest_sha,
        provider_result_sha256=digest(a.provider / "result.json"),
        metrics_sha256=digest(a.provider / "metrics.json"),
        identity=dict(input_rgb_uint8_nchw=[1,3,96,160], map_centers_mm=model.centers.tolist(),
                      content_hw=[90,160], padding=[3,3], pad_value=114, normalization="RGB uint8 /255"),
        onnx=numerical, kmodel=compiled, numerical_observations=numeric_pairs,
        quantized_accuracy_acceptance="NOT_EVALUATED", preprocessing_parity="MISMATCH_MEASURED_PREVIOUS_RUN",
        native_status="NOT_RUN")
    with (a.out / "engineering.json").open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("ENGINEERING_COMPILE_COMPLETE", compiled["kmodel_bytes"], compiled["kmodel_sha256"], flush=True)


if __name__ == "__main__":
    main()
