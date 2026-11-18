"""Verify exact native CameraXY engineering reports; never evaluates correctness."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_runs(runs):
    if not isinstance(runs, list) or not runs:
        raise ValueError("A nonempty native run inventory is required")
    names, hashes = set(), set()
    for item in runs:
        name, value, count = item["file"], item["native_sha256"], item["frames"]
        if (not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name
                or not name.endswith(".json") or name.casefold() in names
                or not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value) or value in hashes
                or type(count) is not int or not 1 <= count <= 3000):
            raise ValueError("Native run identities must be unique with bounded positive integer frame counts")
        names.add(name.casefold())
        hashes.add(value)
    return runs


def verify_report(path, expected_frames, native_sha, deployment_path, sources):
    if sha(path) != native_sha:
        raise ValueError("Native report readback SHA differs")
    report = json.loads(path.read_text(encoding="utf8"))
    deployment = json.loads(deployment_path.read_text(encoding="utf8"))
    count = len(deployment["identity"]["map_centers_mm"])
    if report["status"] != "ENGINEERING_CHAIN_COMPLETE":
        raise ValueError("Native engineering run incomplete")
    for key in ("requested_frames", "attempts", "frames", "ai2d_calls", "kpu_calls", "decoded_frames", "displayed_frames"):
        if type(report[key]) is not int or report[key] != expected_frames:
            raise ValueError("Native stage denominator differs: " + key)
    if (report["capture_none"] or report["cleanup_errors"] or not report["startup_unchanged"]
            or report["startup_before"] != report["startup_after"]):
        raise ValueError("Capture or startup/cleanup verification failed")
    if (report["model_sha256"] != deployment["kmodel"]["kmodel_sha256"]
            or report["model_bytes"] != deployment["kmodel"]["kmodel_bytes"]
            or report["deployment_sha256"] != sha(deployment_path) or report["sources"] != sources):
        raise ValueError("Native model/configuration/source identity differs")
    if report["output_shape"] != [1, count]:
        raise ValueError("Native output shape differs")
    input_desc, output_desc = (ast.literal_eval(report[k]) for k in ("inputs_desc", "outputs_desc"))
    if (input_desc["tensor_size"] != 3 * 96 * 160 or input_desc["tensor_datatype"] != "dt_uint8"
            or output_desc["tensor_size"] != count * 4 or output_desc["tensor_datatype"] != "dt_float32"):
        raise ValueError("Physical KPU tensor ABI differs")
    if (not report["preview_requested"] or report["preview_status"] != "COMPLETE"
            or report["preview_pixel_format"] != "RGB565"):
        raise ValueError("Preview calls not covered by this verification")
    if set(report["timing"]) != {"capture", "rgb", "ai2d", "kpu", "readback", "decode", "preview", "cycle"}:
        raise ValueError("Native timing stages differ")
    for stage in report["timing"].values():
        if stage["samples"] != expected_frames or any(
                not math.isfinite(stage[k]) or stage[k] < 0 for k in ("mean_ms", "p50_ms", "p95_ms")):
            raise ValueError("Native timing samples are incomplete")
    if (report["initial_memory"]["sys_mmz"]["used"] != 0
            or report["final_memory"]["sys_mmz"]["used"] != 0):
        raise ValueError("Native video pool not released")
    answer = report["last_output"]
    if (report["accuracy_test"] != "NOT_RUN_BY_USER_SCOPE"
            or report["road_quality_test"] != "NOT_RUN_BY_USER_SCOPE"
            or report["production_eligible"] is not False
            or answer["status"] != "unlocalized" or answer["camera_x_mm"] is not None
            or answer["camera_y_mm"] is not None or answer["confidence"] is not None
            or answer["candidate_set_confidence"] is not None):
        raise ValueError("Engineering-only output boundary changed")
    if len(answer["top_k_camera_xy"]) != 5 or not all(math.isfinite(v) and abs(v) <= 1200
            for point in answer["top_k_camera_xy"] for v in point.values()):
        raise ValueError("Malformed candidate coordinate structure")
    if not report["last_score_sample"] or not all(math.isfinite(v) for v in report["last_score_sample"]):
        raise ValueError("Malformed finite score sample")
    return dict(file=path.name, frames=report["frames"], processed_fps=report["processed_fps"],
                elapsed_seconds=report["elapsed_seconds"], timing=report["timing"],
                preview_pixel_format=report["preview_pixel_format"],
                initial_memory=report["initial_memory"], final_memory=report["final_memory"],
                native_sha256=native_sha, source_sha256=report["sources"],
                model_sha256=report["model_sha256"], startup_sha256=report["startup_after"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--identities", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    sources = {name:sha(source / name) for name in ("board_usb_smoke.py", "board_probe.py")}
    runs = validate_runs(json.loads(args.identities.read_text(encoding="utf8")))
    verified = [verify_report(args.evidence / item["file"], item["frames"], item["native_sha256"],
                              args.deployment, sources) for item in runs]
    result = dict(status="BOUNDED_NATIVE_ENGINEERING_PASS", runs=verified,
                  completed_frames=sum(row["frames"] for row in verified),
                  accuracy_test="NOT_RUN_BY_USER_SCOPE", road_quality_test="NOT_RUN_BY_USER_SCOPE",
                  tested_method="learned_map", scientific_gate="HOLD", production_eligible=False)
    with args.out.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], runs=len(verified), frames=result["completed_frames"])))


if __name__ == "__main__":
    main()
