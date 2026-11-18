"""Export a host-accepted CameraXY model and compare inference before K230 testing."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from contract import digest, load_rows, xy
from infer import Localizer
from infer_map import MapLocalizer
from map_matcher import decode_scores
from model import H, W, grid_decode, pixels
from train import metrics


def tensor(data, row, flip=False):
    path = data / row["image"]
    if digest(path) != row["image_sha256"]:
        raise ValueError(f"Source image drift: {path}")
    with Image.open(path) as image:
        value = pixels(image)[None]
    return np.ascontiguousarray(value[..., ::-1, ::-1] if flip else value)


def camera_decode(raw, runner, value, cell):
    if cell:
        candidates, probability, _ = grid_decode(torch.from_numpy(raw), cell)
        return candidates.numpy()[0], probability.numpy()[0, 0]
    a = np.clip(raw[0], -1, 1) * 1200
    b = np.clip(runner(value[..., ::-1, ::-1].copy())[0], -1, 1) * 1200
    return ((a+b)/2)[None], -np.linalg.norm(a-b)


def check_acceptance(a, localizer):
    accepted = json.loads(a.host_acceptance.read_text(encoding="utf8"))
    real = json.loads(a.real_report.read_text(encoding="utf8"))
    if isinstance(localizer, MapLocalizer):
        calibration = accepted["selected_method_by_validation_top5"]
        checkpoint_tag = localizer.method
        valid_recipe = checkpoint_tag == "learned_map" and real.get("method") == checkpoint_tag
    else:
        calibration = accepted["model_selected_by_validation_median"]
        checkpoint_tag = f"grid{localizer.cell_mm}" if localizer.cell_mm else "direct"
        valid_recipe = True
    manifest_sha = digest(a.data / "manifest.jsonl")
    goal = accepted["scientific_goal"]
    real_metrics = real.get("metrics", {})
    coverage = real_metrics.get("coverage")
    median = real_metrics.get("accepted_median_mm")
    p95 = real_metrics.get("accepted_p95_mm")
    real_pass = (
        real.get("capture_source") == "OPERATOR_OV2735"
        and real.get("primary_qualified") is True
        and real_metrics.get("accepted_count", 0) > 0
        and coverage is not None and coverage >= goal["accepted_coverage_min"]
        and median is not None and 0 <= median <= goal["median_mm"]
        and p95 is not None and 0 <= p95 <= goal["p95_mm"]
    )
    checks = (
        accepted["status"] == "COMPUTE_COMPLETE",
        accepted["full_gate"] == "PASS",
        calibration == checkpoint_tag and valid_recipe,
        real.get("status") == "MEASURED_REAL_OV2735" and real_pass,
        real.get("checkpoint_sha256") == localizer.checkpoint_sha,
        real.get("training_manifest_sha256") == localizer.manifest_sha,
        localizer.manifest_sha == manifest_sha == accepted["dataset_manifest_sha256"],
    )
    if not all(checks):
        raise ValueError("Model has not passed host/real validation with the same frozen checkpoint")


def host_onnx(a, model, held_out):
    import onnx
    import onnxruntime as ort
    path = a.out / "camera_xy.onnx"
    with torch.inference_mode():
        torch.onnx.export(model.net, torch.zeros((1, 3, H, W)), path, input_names=["rgb"],
                          output_names=["xy_head"], opset_version=12, dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    def onnx_runner(value):
        return session.run(None, {"rgb": value.astype(np.float32) / 255})[0]

    @torch.inference_mode()
    def torch_runner(value):
        return model.net(torch.from_numpy(value).float() / 255).numpy()

    parity = []
    for row in held_out[::max(1, len(held_out)//24)]:
        value = tensor(a.data, row)
        for flip in (False, True) if isinstance(model, Localizer) and not model.cell_mm else (False,):
            image = value[..., ::-1, ::-1].copy() if flip else value
            expected, actual = torch_runner(image), onnx_runner(image)
            np.testing.assert_allclose(expected, actual, rtol=1e-3, atol=1e-4)
            parity.append(float(np.max(np.abs(expected - actual))))
    return path, onnx_runner, dict(samples=len(parity), max_abs=max(parity),
                                   onnx_sha256=digest(path), onnx_bytes=path.stat().st_size)


def kmodel(a, onnx_path, calibration):
    versions = {name: importlib.metadata.version(name) for name in ("nncase", "nncase-kpu")}
    if set(versions.values()) != {"2.11.0"}:
        raise ValueError("Use the previously qualified K230 nncase pair 2.11.0")
    plugins = str(Path(importlib.util.find_spec("_nncase").origin).parent)
    os.environ["NNCASE_PLUGIN_PATH"] = plugins
    os.environ["PATH"] = plugins + os.pathsep + os.environ["PATH"]
    dll = os.add_dll_directory(plugins) if os.name == "nt" else None
    try:
        import nncase
        options = nncase.CompileOptions()
        settings = dict(target="k230", preprocess=True, input_shape=[1, 3, H, W],
                        input_type="uint8", input_range=[0, 1], swapRB=False,
                        mean=[0, 0, 0], std=[1, 1, 1], input_layout="NCHW",
                        shape_bucket_enable=False, dump_ir=False, dump_asm=False,
                        dump_dir=str(a.out.resolve() / "compiler-dump"))
        for key, value in settings.items():
            setattr(options, key, value)
        compiler = nncase.Compiler(options)
        compiler.import_onnx(onnx_path.read_bytes(), nncase.ImportOptions())
        ptq = nncase.PTQTensorOptions()
        ptq.samples_count = len(calibration)
        ptq.calibrate_method = "NoClip"
        ptq.set_tensor_data([[tensor(a.data, row)] for row in calibration])
        compiler.use_ptq(ptq)
        compiler.compile()
        binary = compiler.gencode_tobytes()
        file = a.out / "camera_xy.kmodel"
        file.write_bytes(binary)
        runtime = nncase.Simulator()
        runtime.load_model(binary)
        if (runtime.inputs_size, runtime.outputs_size, list(runtime.get_input_shape(0))) != (1, 1, [1, 3, H, W]):
            raise ValueError("KModel Simulator input layout drift")

        def runner(value):
            runtime.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(value))
            runtime.run()
            raw = runtime.get_output_tensor(0).to_numpy().copy()
            if not np.isfinite(raw).all():
                raise ValueError("Unexpected KModel non-finite output")
            return raw

        runner.dll_search_path = dll
        return runner, dict(versions=versions, settings=settings, kmodel_bytes=len(binary),
                            kmodel_sha256=digest(file), calibration_ids=[r["sample_id"] for r in calibration])
    except BaseException:
        if dll is not None:
            dll.close()
        raise


def evaluate(runner, dataset, rows, model):
    all_candidates, raw_confidence, times = [], [], []
    for row in rows:
        frame = tensor(dataset, row)
        start = time.perf_counter()
        raw = runner(frame)
        if isinstance(model, MapLocalizer):
            candidates, primary, _ = decode_scores(raw, model.centers)
            cand, score = candidates[0], primary[0]
        else:
            expected = 2 if not model.cell_mm else (2400 // model.cell_mm)**2 * 3
            if raw.shape != (1, expected):
                raise ValueError("Runtime output shape drift")
            cand, score = camera_decode(raw, runner, frame, model.cell_mm)
        times.append((time.perf_counter() - start) * 1000)
        all_candidates.append(cand)
        raw_confidence.append(score)
    result = dict(host_ms=dict(median=float(np.median(times)), p95=float(np.quantile(times, .95))))
    for tag, aref in (("spatial_holdout", np.array([r["variant"] < 2 for r in rows])),
                      ("appearance_holdout", np.array([r["variant"] == 2 for r in rows]))):
        result[tag] = metrics(np.stack(all_candidates)[aref], np.array([xy(r) for r in rows])[aref],
                              np.array(raw_confidence)[aref], model.cal)
    return result


def main():
    p = argparse.ArgumentParser()
    for key in ("data", "checkpoint", "metrics", "out", "host-acceptance", "real-report"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--map-head", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.map_head:
        accepted = json.loads(a.host_acceptance.read_text(encoding="utf8"))
        model = MapLocalizer(a.checkpoint, a.metrics, accepted, method="learned_map")
    else:
        model = Localizer(a.checkpoint, json.loads(a.metrics.read_text(encoding="utf8")))
    check_acceptance(a, model)
    rows = load_rows(a.data / "manifest.jsonl")
    held_out = [r for r in rows if r["split"] == "test"]
    ref = [r for r in rows if r["split"] == "train" and r["variant"] == 0]
    calibration = [ref[i] for i in np.linspace(0, len(ref)-1, min(32, len(ref)), dtype=int)]
    a.out.mkdir()
    graph, onnx_runner, evidence = host_onnx(a, model, held_out)
    identity = dict(input_rgb_uint8_nchw=[1, 3, H, W], content_hw=[90, W], padding=[3, 3],
                    pad_value=114, normalization="uint8 RGB / 255", confidence=model.cal)
    if a.map_head:
        identity.update(map_centers_mm=model.centers.tolist(), candidate_set_confidence=model.set_cal)
    else:
        identity["cell_mm"] = model.cell_mm
    report = dict(
        checkpoint_sha256=model.checkpoint_sha, dataset_manifest_sha256=model.manifest_sha,
        identity=identity,
        onnx=evidence, onnx_eval=evaluate(onnx_runner, a.data, held_out, model),
        native_board_status="NOT_RUN",
    )
    if a.compile:
        kpu_sim, info = kmodel(a, graph, calibration)
        report["kmodel"] = info
        report["kmodel_simulator_eval"] = evaluate(kpu_sim, a.data, held_out, model)
    (a.out / "deployment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
