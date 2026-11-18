"""Prepare frozen native input tests and verify downloaded CameraXY bench bytes."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from board_tensor_bench import fixture
from contract import digest, load_rows
from infer_map import MapLocalizer
from model import pixels


def write_json(path, value):
    with path.open("x", encoding="utf8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def localizer(provider):
    return MapLocalizer(provider / "map-inference.pt", provider / "metrics.json",
                        json.loads((provider / "result.json").read_text(encoding="utf8")),
                        method="learned_map")


def prepare(a):
    from hashlib import sha256
    torch.set_num_threads(2)
    loc = localizer(a.provider)
    rows = {r["sample_id"]: r for r in load_rows(a.data / "manifest.jsonl")}
    replay = json.loads((Path(__file__).parent / "evidence/G5-host-replay/replay.json").read_text(encoding="utf8"))
    if loc.manifest_sha != digest(a.data / "manifest.jsonl"):
        raise ValueError("Training data identity differs")
    a.out.mkdir(parents=True, exist_ok=False)
    spec = dict(input_shape=[1, 3, 360, 640], output_shape=[1, 3, 96, 160],
                scope="Diagnostic only; no KModel export or primary qualification",
                checkpoint_sha256=loc.checkpoint_sha, manifest_sha256=loc.manifest_sha,
                method=loc.method, selected_method=replay["method"],
                primary_qualified=loc.primary_qualified, centers_mm=loc.centers.tolist(),
                primary_cal=loc.cal, set_cal=loc.set_cal, fixtures=[], score_cases=[],
                board_sources={name: digest(Path(__file__).parent / name)
                               for name in ("board_probe.py", "board_tensor_bench.py")})
    for name in ("constant", "rgb_ramp", "thin_lines", "checkerboard"):
        raw = fixture(name)
        rgb = np.frombuffer(raw, np.uint8).reshape(3, 360, 640).transpose(1, 2, 0)
        golden = pixels(Image.fromarray(rgb)).tobytes()
        (a.out / (name + ".golden")).write_bytes(golden)
        spec["fixtures"].append(dict(name=name, kind="procedural", input_sha256=sha256(raw).hexdigest(),
                                     golden_sha256=sha256(golden).hexdigest()))
    for i, record in enumerate(replay["records"]):
        row = rows[record["sample_id"]]
        path = a.data / row["image"]
        if digest(path) != row["image_sha256"] or row["image_sha256"] != record["image_sha256"]:
            raise ValueError("Held-out RGB drift")
        with Image.open(path) as image:
            value = pixels(image).copy()[None]
            if i == 0:
                source_size = list(image.size)
                rgb = np.asarray(image.convert("RGB").resize((640, 360), Image.Resampling.BILINEAR))
                raw = rgb.transpose(2, 0, 1).copy().tobytes()
                segment_golden = pixels(Image.fromarray(rgb)).tobytes()
                (a.out / "heldout.input.rgb").write_bytes(raw)
                (a.out / "heldout.golden").write_bytes(segment_golden)
                spec["fixtures"].append(dict(name="heldout", kind="frozen_synthetic_image",
                    sample_id=row["sample_id"], original_image_sha256=row["image_sha256"],
                    source_size=source_size, intermediate="Host PIL 640x360 fixture; UVC conversion untested",
                    input_sha256=sha256(raw).hexdigest(), golden_sha256=sha256(segment_golden).hexdigest()))
                if a.shared_fixture_root:
                    spec["fixtures"][-1]["compressed_input_path"] = a.shared_fixture_root.rstrip("/") + "/heldout.rgb.z"
        with torch.inference_mode():
            scores = loc.net(torch.from_numpy(value).float() / 255).numpy()[0]
        spec["score_cases"].append(dict(sample_id=row["sample_id"], image_sha256=row["image_sha256"],
            scores=scores.tolist(), expected=loc.output(path)))
    write_json(a.out / "spec.json", spec)
    print(json.dumps(dict(status="PREPARED", fixtures=len(spec["fixtures"]),
                          score_cases=len(spec["score_cases"]), spec_sha256=digest(a.out / "spec.json"))))


def verify(a):
    torch.set_num_threads(2)
    spec = json.loads((a.inputs / "spec.json").read_text(encoding="utf8"))
    native = json.loads((a.native / "native.json").read_text(encoding="utf8"))
    if native["status"] != "AI2D_AND_DECODE_COMPLETE" or native["spec_sha256"] != digest(a.inputs / "spec.json"):
        raise ValueError("Native run or input identity mismatch")
    if native["cleanup_errors"] or not native["startup_unchanged"] or native["source_sha256"] != spec["board_sources"]:
        raise ValueError("Native source or cleanup/startup gate failed")
    if [r["name"] for r in native["preprocessing"]] != [r["name"] for r in spec["fixtures"]]:
        raise ValueError("Native fixture denominator differs")
    if [r["sample_id"] for r in native["decode"]] != [r["sample_id"] for r in spec["score_cases"]]:
        raise ValueError("Native decoder denominator differs")
    report = dict(status="NATIVE_SEGMENTS_VERIFIED", firmware=native["firmware"], runtime=native["runtime"],
                  camera="NOT_RUN", kpu="NOT_RUN", full_camera_pipeline="NOT_RUN", single_point_gate="HOLD",
                  spec_sha256=digest(a.inputs / "spec.json"), native_sha256=digest(a.native / "native.json"),
                  checkpoint_sha256=spec["checkpoint_sha256"], method=spec["method"],
                  selected_method=spec["selected_method"], preprocessing=[], decoder=[],
                  startup_unchanged=native["startup_unchanged"], cleanup_errors=native["cleanup_errors"])
    for item, measured in zip(spec["fixtures"], native["preprocessing"]):
        actual_path = a.native / measured["output"]
        golden_path = a.inputs / (item["name"] + ".golden")
        if (digest(actual_path) != measured["output_sha256"] or digest(golden_path) != item["golden_sha256"]
                or measured["input_sha256"] != item["input_sha256"] or measured["output_bytes"] != 46080):
            raise ValueError("Tensor byte identity mismatch")
        actual = np.frombuffer(actual_path.read_bytes(), np.uint8).reshape(3, 96, 160)
        golden = np.frombuffer(golden_path.read_bytes(), np.uint8).reshape(3, 96, 160)
        diff = np.abs(actual.astype(np.int16) - golden.astype(np.int16))
        padding_ok = bool(np.all(actual[:, :3] == 114) and np.all(actual[:, -3:] == 114))
        if not padding_ok:
            raise ValueError("Native padding differs from training contract")
        record = dict(name=item["name"], bytes=actual.size, exact=bool(not diff.any()),
                      max_abs=int(diff.max()), mean_abs=float(diff.mean()),
                      differing_bytes=int(np.count_nonzero(diff)), padding_exact=padding_ok,
                      golden_sha256=item["golden_sha256"], native_sha256=measured["output_sha256"],
                      ai2d_us=measured["ai2d_us"])
        if item["name"] == "heldout":
            loc = localizer(a.provider)
            if (loc.checkpoint_sha != spec["checkpoint_sha256"] or loc.manifest_sha != spec["manifest_sha256"]
                    or loc.cal != spec["primary_cal"] or loc.set_cal != spec["set_cal"]
                    or loc.primary_qualified != spec["primary_qualified"]):
                raise ValueError("CPU replay provider differs from the frozen native spec")
            from map_matcher import decode_scores
            from board_probe import decode_map
            with torch.inference_mode():
                scores = loc.net(torch.from_numpy(actual.copy()[None]).float() / 255).numpy()
                golden_scores = loc.net(torch.from_numpy(golden.copy()[None]).float() / 255).numpy()
            choices, first, top_set = decode_scores(scores, loc.centers)
            golden_choices, _, _ = decode_scores(golden_scores, loc.centers)
            record["frozen_model_host_replay_after_native_ai2d"] = dict(
                scope="CPU network replay of native AI2D bytes; no KPU", sample_id=item["sample_id"],
                top_k_camera_xy=choices[0].tolist(), first_mass=float(first[0]), set_mass=float(top_set[0]),
                scores_max_abs_change=float(np.max(np.abs(scores - golden_scores))),
                host_preprocessing_top_k=golden_choices[0].tolist(),
                decoder=decode_map(scores[0].tolist(), loc.centers.tolist(), loc.cal, loc.set_cal))
        report["preprocessing"].append(record)
    for item, measured in zip(spec["score_cases"], native["decode"]):
        expected, actual = item["expected"], measured["output"]
        matching = {key: actual[key] == expected[key]
                    for key in ("status", "camera_x_mm", "camera_y_mm", "top_k_camera_xy")}
        errors = {key: abs(actual[key] - expected[key]) for key in ("confidence", "candidate_set_confidence")}
        passed = all(matching.values()) and max(errors.values()) <= 1e-6
        report["decoder"].append(dict(sample_id=item["sample_id"],
                                     exact_candidate_order=matching["top_k_camera_xy"], passed=passed,
                                     expected=expected, actual=actual, matching=matching,
                                     status=actual["status"], confidence_abs_errors=errors,
                                     decode_us=measured["decode_us"]))
    if native["invalid_decode"] != [dict(case=n, rejected=True) for n in ("truncated", "nonfinite")]:
        raise ValueError("Invalid tensor rejection failed")
    report["invalid_decode"] = native["invalid_decode"]
    report["preprocessing_parity"] = "EXACT" if all(r["exact"] for r in report["preprocessing"]) else "MISMATCH_MEASURED"
    report["decoder_parity"] = "PASS" if all(r["passed"] for r in report["decoder"]) else "FAIL"
    if report["decoder_parity"] == "FAIL":
        report["status"] = "NATIVE_DECODER_MISMATCH"
    write_json(a.out, report)
    print(json.dumps(dict(status=report["status"], preprocessing=report["preprocessing_parity"],
                          fixtures=len(report["preprocessing"]), decoder_cases=len(report["decoder"]))))
    return 0 if report["decoder_parity"] == "PASS" else 2


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("data", "provider", "out"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--shared-fixture-root", help="Reuse an immutable on-board xy input to conserve SD space")
    check = commands.add_parser("verify")
    for name in ("inputs", "native", "provider", "out"):
        check.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    return (prepare if args.command == "prepare" else verify)(args)


if __name__ == "__main__":
    raise SystemExit(main())
