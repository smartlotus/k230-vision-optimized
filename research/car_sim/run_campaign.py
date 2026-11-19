"""Run or resume a bounded PC-only K230 simulation campaign."""

import argparse
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from campaign_ledger import Ledger, TERMINAL, digest, identity, read_json, supervisor_lock, write_json
from scenario_catalog import drive_cases, qr_cases


SIM = Path(__file__).resolve().parent
ROOT = SIM.parents[1]
ASSETS = {
    "field": "worktrees/ops-temporal-camera-v1/docs/temporal_camera/map-correction/field_gray_v1.blend",
    "floor_mask": "worktrees/ops-temporal-camera-v1/docs/temporal_camera/map-correction/field_mask.png",
    "r2_zip": "deliverables/K230_Car_Blender_R2.zip",
    "material": ("webhelp/camera_xy_v2_audit_20261117/k230_material_block_model_bundle/"
                 "k230_material_block_model/material_block.glb"),
    "cad_receipt": ("webhelp/camera_xy_v2_audit_20261117/k230_material_block_model_bundle/"
                    "k230_material_block_model/model_validation.json"),
    "road_onnx": ("webhelp/K230_RoadNet_Boundary_Sim_Execution_Package/experiments/"
                  "roadnet_boundary_sim_v1/evidence/delivery/roadnet-20261117/model/roadnet.onnx"),
    "topface_onnx": "TopFaceNet-V1.1-Complete-a01/models/topface.onnx",
}
REQUIRED = {
    "baseline": ["baseline.json", "workflow-tests.log"],
    "protocol": ["protocol.json", "success.wire", "ack-loss.wire", "timeout.wire",
                 "success.jsonl", "ack-loss.jsonl", "timeout.jsonl", "protocol-tests.log"],
    "scenarios": ["qr.png", "texture.png", "render_state.json", "observations.json", "workflow.json"],
    "physics_sweep": ["stack_contact.json"],
    "recovery_probe": ["recovery-tests.log"],
    "rig": ["rig.json", "rig-r3.blend", "blender.log"],
    "road_observe": ["road_observations.json", "metrics.json", "blender.log"],
    "drive": ["road_drive.json", "metrics.json", "blender.log"],
    "align": ["alignment.json", "vision.log", "blender.log"],
    "grasp": ["grasp.json", "vision.log", "blender.log"],
    "delivery": ["comparison.json", "report.md"],
}
R30_JOBS = {"baseline", "protocol", "scenarios", "recovery_probe", "delivery"}
R31_JOBS = R30_JOBS | {"rig", "road_observe", "drive"}
R32_JOBS = R31_JOBS | {"align", "grasp", "physics_sweep"}
PROFILES = {"r30": R30_JOBS, "r31": R31_JOBS, "r32": R32_JOBS}
BRANCH_CASES = {"scenarios": qr_cases, "drive": drive_cases}
BLENDER_JOBS = {"scenarios", "rig", "road_observe", "drive", "align", "grasp"}


def expand(definition, profile):
    specs = {}
    added = {"id": "recovery_probe", "after": ["baseline"], "suites": ["T00"]}
    for job in [*definition["jobs"][:-1], added, definition["jobs"][-1]]:
        name = job["id"]
        if profile in PROFILES and name not in PROFILES[profile]:
            continue
        if name == "physics_sweep":
            cases = [{"key": name, "seed": 0, "probe": "ideal_r2"},
                     {"key": "physics_gripper", "seed": 0, "probe": "moving_fingers"}]
        else:
            cases = BRANCH_CASES[name]() if name in BRANCH_CASES else [{"key": name, "seed": 0}]
        for case in cases:
            key = case["key"]
            artifacts = ["result.json", *REQUIRED.get(name, [])]
            specs[key] = {
                "key": key, "worker": name, "case": case, "suites": job["suites"],
                "depends": job["after"], "artifacts": artifacts,
                "timeout_seconds": 180 if name == "drive" else 120 if name in BLENDER_JOBS else 60,
            }
    translations = {}
    for spec in specs.values():
        translations.setdefault(spec["worker"], []).append(spec["key"])
    for spec in specs.values():
        spec["depends"] = [key for dep in spec["depends"] for key in translations[dep]]
    return specs


def inventory(paths, root):
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths) if p.is_file()}


def file_id(path):
    return {"path": str(path), "sha256": digest(path)} if path.is_file() else {"path": str(path), "missing": True}


def probe(args, definition, specs):
    packages = {}
    for name in ("numpy", "onnxruntime", "opencv-python", "Pillow", "scipy", "mujoco", "torch", "onnx"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    assets = {key: file_id(args.hub / rel) for key, rel in ASSETS.items()}
    baseline = SIM / "output"
    sources = inventory(
        [*SIM.glob("*.py"), *SIM.glob("*.json"), *SIM.glob("campaigns/*.json"), *SIM.glob("workcell/*.py"),
         *ROOT.glob("k230_transport_mvp/core/*.py"), *ROOT.glob("k230_transport_mvp/ports/*.py"),
         *ROOT.glob("k230_transport_mvp/sim/*.py"), *ROOT.glob("k230_transport_mvp/vision/*.py"),
         *ROOT.glob("k230_transport_mvp/stm32/*.[ch]"), *ROOT.glob("k230_transport_mvp/tests/*.py"),
         *ROOT.glob("research/topface/*.py")],
        ROOT,
    )
    identity_contract = {
        "campaign": args.campaign, "profile": args.profile, "scope": "PC_SIM_ONLY", "transport": "loopback",
        "definition": definition, "jobs": list(specs.values()),
        "sources": sources, "assets": assets,
        "r2_receipts": inventory([baseline / "qr.png", baseline / "mission_truth.json",
                                  baseline / "mission_vision.json", baseline / "stack_contact.json",
                                  *baseline.glob("mission_camera/*.png")], ROOT),
        "runtime": {"python": str(Path(sys.executable).resolve()), "version": sys.version,
                    "packages": packages, "blender": file_id(args.blender) if args.blender else None,
                    "gcc": shutil.which("gcc"), "wsl": shutil.which("wsl.exe"),
                    "wsl_distro": os.environ.get("K230_WSL_DISTRO", "Ubuntu-24.04"),
                    "ffmpeg": shutil.which("ffmpeg")},
        "budget": {"wall_seconds_per_invocation": args.wall_seconds, "minimum_free_mib": args.min_free_mib},
    }
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return {"manifest_id": identity(identity_contract), "contract": identity_contract,
            "source_commit": head.stdout.strip() if head.returncode == 0 else None,
            "initial_disk_free_bytes": shutil.disk_usage(args.out).free, "created_unix_s": time.time()}


def snapshot(ledger, specs, phase):
    jobs = ledger.jobs()
    statuses = Counter(row["status"] for row in jobs)
    attempts = {row["id"]: ledger.attempts(row["id"]) for row in jobs}
    return {"phase": phase, "updated_unix_s": time.time(), "supervisor_pid": os.getpid(),
            "counts": dict(statuses),
            "jobs": [{"key": row["id"], "worker": specs[row["id"]]["worker"],
                      "suites": specs[row["id"]]["suites"], "status": row["status"], "reason": row["reason"],
                      "attempt": row["attempt"], "attempts": len(attempts[row["id"]]),
                      "worker_pid": attempts[row["id"]][-1]["pid"] if row["status"] == "RUNNING"
                                    and attempts[row["id"]] else None,
                      "updated_unix_s": row["updated"]}
                     for row in jobs]}


def report(ledger, specs, definition, directory):
    data = snapshot(ledger, specs, "DELIVERY")
    jobs = [row for row in data["jobs"] if row["worker"] != "delivery"]
    by_suite = {}
    for suite in definition["required_suites"]:
        rows = [row for row in jobs if suite in row["suites"]]
        by_suite[suite] = {"scheduled_cases": len(rows), "verdicts": dict(Counter(row["status"] for row in rows))}
    # A successful probe or a reference run covers a subset of its full suite.
    complete_suites = ["T00"] if by_suite["T00"]["verdicts"] == {"PASS": 2} else []
    missing = [suite for suite in definition["required_suites"] if suite not in complete_suites]
    data.update(suites=by_suite, complete_suites=complete_suites, unverified_suites=missing,
                unattended_acceptance="UNVERIFIED", algorithm_acceptance="UNVERIFIED")
    write_json(directory / "comparison.json", data)
    lines = ["# K230 local continuous simulation", "", "Full-plan acceptance is pending.",
             "", "Unattended recovery and algorithm acceptance remain unverified.", "",
             "| Suite | Scheduled cases | Results |", "| --- | ---: | --- |"]
    for suite, record in by_suite.items():
        lines.append(f"| {suite} | {record['scheduled_cases']} | "
                     + (", ".join(f"{key} {n}" for key, n in record["verdicts"].items()) or "Not run") + " |")
    lines.extend(["", "Case receipts include the precise scope and hashes of the artifacts.", ""])
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(directory / "result.json", {"status": "PASS", "reason": None})


def stop_child(child):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            child.kill()
        else:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def worker(ledger, specs, spec, args, deadline):
    key = spec["key"]
    heartbeat = 30
    attempt, work = ledger.stage(key, heartbeat * 2)
    spec_path = work.parent / "spec.json"
    write_json(spec_path, spec)
    command = [sys.executable, "-B", "-u", str(SIM / "campaign_worker.py"), "--spec", str(spec_path),
               "--hub", str(args.hub), "--out", str(work)]
    if args.blender:
        command.extend(["--blender", str(args.blender)])
    write_json(work.parent / "command.json", command)
    start = time.monotonic()
    last_heartbeat = 0.
    with (work / "worker.log").open("wb") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=os.name != "nt")
        try:
            while child.poll() is None:
                now = time.monotonic()
                if now - last_heartbeat >= heartbeat:
                    ledger.heartbeat(attempt, child.pid, heartbeat * 2)
                    write_json(ledger.root / "status.json", snapshot(ledger, specs, "RUNNING"))
                    last_heartbeat = now
                if (ledger.root / "STOP").exists() or now >= deadline:
                    stop_child(child)
                    ledger.mark(key, "PENDING", "Campaign stopped during case", "STOPPED")
                    return False
                if now - start >= spec["timeout_seconds"]:
                    stop_child(child)
                    raise TimeoutError(f"Case exceeded {spec['timeout_seconds']} s")
                time.sleep(.10)
        finally:
            if child.poll() is None:
                stop_child(child)
    write_json(work / "execution.json", {"wall_seconds": time.monotonic() - start, "exit_code": child.returncode})
    if child.returncode:
        raise RuntimeError(f"Worker exit code {child.returncode}")
    ledger.publish(spec, attempt, work)
    return True


def unavailable(spec, args, manifest):
    name = spec["worker"]
    if name not in REQUIRED:
        return f"{name} has no campaign worker yet"
    runtime = manifest["contract"]["runtime"]
    if name in BLENDER_JOBS and (not args.blender or not args.blender.is_file()):
        return "Blender executable unavailable"
    if name == "protocol" and not (runtime["gcc"] or runtime["wsl"]):
        return "Host C compiler unavailable"
    if name == "physics_sweep" and runtime["packages"]["mujoco"] is None:
        return "MuJoCo unavailable"
    return None


def supervise(args, ledger, specs, definition, manifest):
    deadline = time.monotonic() + args.wall_seconds if args.wall_seconds else float("inf")
    retries = definition["policy"]["infra_retries"]
    while True:
        rows = {row["id"]: row for row in ledger.jobs()}
        pending = [specs[key] for key, row in rows.items() if row["status"] == "PENDING"]
        if not pending:
            return "TERMINAL"
        if (ledger.root / "STOP").exists() or time.monotonic() >= deadline:
            return "STOPPED"
        if shutil.disk_usage(ledger.root).free < args.min_free_mib * 1024 * 1024:
            return "DISK_BUDGET"
        for spec in pending:
            key, name = spec["key"], spec["worker"]
            if name == "delivery":
                if any(k != key and v["status"] not in TERMINAL for k, v in rows.items()):
                    continue
                attempt, work = ledger.stage(key, 60)
                report(ledger, specs, definition, work)
                ledger.publish(spec, attempt, work)
                break
            dependencies = [rows[dep]["status"] for dep in spec["depends"]]
            if any(status not in TERMINAL for status in dependencies):
                continue
            if any(status == "BLOCKED" for status in dependencies):
                ledger.mark(key, "BLOCKED", "A required case has no verified artifacts")
                break
            problem = unavailable(spec, args, manifest)
            if problem:
                ledger.mark(key, "BLOCKED", problem)
                break
            try:
                if not worker(ledger, specs, spec, args, deadline):
                    return "STOPPED"
            except (OSError, RuntimeError, TimeoutError, ValueError, KeyError) as exc:
                infra_count = sum(a["status"] == "INFRA_ERROR" for a in ledger.attempts(key)) + 1
                ledger.mark(key, "PENDING" if infra_count <= retries else "BLOCKED",
                            f"{type(exc).__name__}: {exc}", "INFRA_ERROR")
            break
        else:
            raise ValueError("Campaign dependencies cannot be satisfied")
        write_json(ledger.root / "status.json", snapshot(ledger, specs, "RUNNING"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--profile", choices=(*PROFILES, "full"), default="full")
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--wall-seconds", type=float, default=0)
    parser.add_argument("--min-free-mib", type=int, default=1024)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.campaign):
        parser.error("Use a lowercase campaign ID with up to 64 safe characters")
    if not math.isfinite(args.wall_seconds) or args.wall_seconds < 0 or args.min_free_mib < 0:
        parser.error("Budgets must be nonnegative")
    args.hub, args.out = args.hub.resolve(), args.out.resolve()
    campaign = args.out / args.campaign
    if args.status:
        print(json.dumps(read_json(campaign / "status.json"), indent=2))
        return 0
    if not args.blender:
        standard = Path("C:/Program Files/Blender Foundation/Blender 5.3/blender.exe")
        name = os.environ.get("K230_BLENDER") or shutil.which("blender")
        args.blender = Path(name).resolve() if name else standard if standard.is_file() else None
    else:
        args.blender = args.blender.resolve()
    definition = read_json(SIM / "campaigns" / "full.json")
    specs = expand(definition, args.profile)
    with supervisor_lock(campaign):
        manifest = probe(args, definition, specs)
        location = campaign / "manifest.json"
        if location.exists():
            if not args.resume:
                parser.error("Existing campaign requires --resume")
            if read_json(location)["manifest_id"] != manifest["manifest_id"]:
                parser.error("Campaign contract changed. Use the frozen inputs or a new campaign ID.")
        else:
            write_json(location, manifest)
        ledger = Ledger(campaign)
        try:
            ledger.register(specs.values(), manifest["manifest_id"])
            ledger.recover(specs)
            phase = supervise(args, ledger, specs, definition, manifest)
            result = snapshot(ledger, specs, phase)
            write_json(campaign / "status.json", result)
            print(json.dumps({"phase": phase, "counts": result["counts"], "path": str(campaign)}))
            return 0 if phase == "TERMINAL" and set(result["counts"]) == {"PASS"} else 2
        finally:
            ledger.db.close()


if __name__ == "__main__":
    raise SystemExit(main())
