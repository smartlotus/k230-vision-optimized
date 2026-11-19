"""Local subprocess entry point for one R3 campaign case."""

import argparse
from pathlib import Path
import subprocess
import sys

from campaign_ledger import digest, read_json, write_json


SIM = Path(__file__).resolve().parent
TRANSPORT = SIM.parents[1] / "k230_transport_mvp"
sys.path.insert(0, str(TRANSPORT))


def run_command(command, out, name, cwd=None):
    with (out / f"{name}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=False)
    return result.returncode


def baseline(_case, hub, out, _blender):
    from mission_vision import run
    from perception import load_session, ROAD_DIR, ROAD_SHA, TOP_DIR, TOP_SHA

    folder = SIM / "output"
    load_session(hub / ROAD_DIR / "roadnet.onnx", ROAD_SHA)
    model = hub / TOP_DIR / "topface.onnx"
    if digest(model) != TOP_SHA:
        raise ValueError("TopFace ONNX identity mismatch")
    result = run(hub, folder)
    tests = run_command(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests",
         "-p", "test_workflow_session.py", "-v"],
        out, "workflow-tests", TRANSPORT,
    )
    old = read_json(folder / "mission_vision.json")
    write_json(out / "baseline.json", {
        "scores": result["scores"], "reference_scores": old["scores"],
        "equal_scores": result["scores"] == old["scores"], "unit_test_exit_code": tests,
        "source_frames": {p.name: digest(p) for p in sorted((folder / "mission_camera").glob("*.png"))},
        "road_onnx_sha256": ROAD_SHA, "topface_onnx_sha256": TOP_SHA,
        "scope": "Frozen R2 rendered workcell observations and existing workflow tests",
    })
    return verdict(result["scores"] == old["scores"] == {"ON_TARGET": 24, "OFF_TARGET": 0, "HOLD": 0}
                   and tests == 0, "Baseline observations or workflow tests regressed")


def verdict(passed, reason):
    return {"status": "PASS" if passed else "FAIL", "reason": None if passed else reason}


def blender_case(blender, script, hub, out, extra=()):
    command = [str(blender), "-b", "--factory-startup", "--python-exit-code", "1",
               "--python", str(SIM / script), "--", "--hub", str(hub), "--out", str(out), *extra]
    if run_command(command, out, "blender"):
        raise RuntimeError(f"Blender {script} failed. Read the case's blender.log.")


def scenario(case, hub, out, blender):
    import cv2
    import numpy as np
    from core.task_parser import parse_task
    from core.types import ValidationError
    from sim.run_workflow import run_workflow

    payload = case["payload"]
    if payload:
        modules = cv2.QRCodeEncoder_create().encode(payload)
        modules = np.pad(modules, 4, constant_values=255)
        image = cv2.resize(modules, (512, 512), interpolation=cv2.INTER_NEAREST)
    else:
        image = np.full((512, 512), 255, dtype=np.uint8)
    (out / "texture.png").write_bytes(cv2.imencode(".png", image)[1].tobytes())
    blender_case(blender, "render_scenario_qr.py", hub, out, ("--case", str(out / "case.json")))
    pixels = cv2.imdecode(np.fromfile(out / "qr.png", dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        raise ValueError("QR render did not decode as a PNG")
    decoded = cv2.QRCodeDetector().detectAndDecode(pixels)[0]
    state = "HOLD_NO_QR"
    observations = {"expected_payload": payload, "decoded": decoded, "submissions": 0,
                    "camera": "front Blender pinhole", "world_model": "R2 fixed gray field"}
    if decoded:
        try:
            parse_task(decoded)
        except ValidationError as exc:
            state = "HOLD_INVALID_TASK"
            observations["rejection"] = str(exc)
        else:
            result = run_workflow(decoded, start_id=case["start_id"])
            write_json(out / "workflow.json", result)
            summary = result["summary"]
            groups = case["layout_task"].split("+")
            expected_slots = dict(zip(map(int, groups[0]), map(int, groups[1])))
            actual_slots = {
                row["color"]: row["slot"] for row in result["transitions"]
                if row["dest"] == "BUFFER" and row["batch"] == 2
            }
            passed = (result["field_complete"] and summary["status"] == "DONE"
                      and summary["station"] == case["start_id"] and expected_slots == actual_slots
                      and summary["counters"] == {"picks": 12, "placements": 12})
            state = "COMPLETE" if passed else "WORKFLOW_FAILED"
            observations.update(submissions=result["controller_submissions"],
                                expected_stack_slots=expected_slots, actual_stack_slots=actual_slots,
                                counters=summary["counters"], end=summary["station"])
    observations.update(state=state, expected_state=case["expected"])
    if not (out / "workflow.json").exists():
        write_json(out / "workflow.json", {"status": state, "controller_submissions": 0})
    write_json(out / "observations.json", observations)
    return verdict(state == case["expected"] and decoded == payload, "Unexpected QR or task result")


def physics_sweep(_case, hub, out, _blender):
    cad = hub / ("webhelp/camera_xy_v2_audit_20261117/k230_material_block_model_bundle/"
                 "k230_material_block_model/model_validation.json")
    command = [sys.executable, "-B", str(SIM / "stack_contact.py"), "--cad-receipt", str(cad),
               "--output", str(out / "stack_contact.json")]
    if run_command(command, out, "mujoco"):
        raise RuntimeError("Independent MuJoCo trial failed to run")
    actual = read_json(out / "stack_contact.json")
    reference = read_json(SIM / "output" / "stack_contact.json")
    return verdict(actual == reference, "MuJoCo sweep diverges from frozen R2 reference")


def recovery_probe(_case, _hub, out, _blender):
    command = [sys.executable, "-B", "-m", "unittest", "-v", "test_campaign_recovery"]
    code = run_command(command, out, "recovery-tests", SIM)
    return verdict(code == 0, "Recovery probe failed")


def rig_trial(_case, hub, out, blender):
    blender_case(blender, "rig_trial.py", hub, out)
    result = read_json(out / "rig.json")
    return verdict(result["gate_pass"], "Gripper transform, clearance, or occlusion check failed")


def road_observe(_case, hub, out, blender):
    from road_metrics import observation_report

    blender_case(blender, "road_trial.py", hub, out, ("--python", sys.executable, "--mode", "observe"))
    result = observation_report(hub, out / "road_observations.json")
    write_json(out / "metrics.json", result)
    return verdict(result["gate_pass"], "A right-boundary observation missed its expected result")


def drive_trial(_case, hub, out, blender):
    from road_metrics import drive_report

    blender_case(blender, "road_trial.py", hub, out,
                 ("--python", sys.executable, "--mode", "drive", "--case", str(out / "case.json")))
    result = drive_report(hub, out / "road_drive.json")
    write_json(out / "metrics.json", result)
    return verdict(result["gate_pass"], "The paired road-control case missed a development gate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--hub", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blender", type=Path)
    args = parser.parse_args()
    spec = read_json(args.spec)
    worker = spec["worker"]
    if worker == "protocol":
        from sim_protocol_replay import replay

        result = replay(args.out, TRANSPORT, run_command)
    else:
        handler = {"baseline": baseline, "scenarios": scenario, "physics_sweep": physics_sweep,
                   "recovery_probe": recovery_probe, "rig": rig_trial, "road_observe": road_observe,
                   "drive": drive_trial}[worker]
        case = spec["case"]
        write_json(args.out / "case.json", case)
        result = handler(case, args.hub, args.out, args.blender)
    write_json(args.out / "result.json", result)


if __name__ == "__main__":
    main()
