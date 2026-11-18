"""Exercise workflow receipt gating through the existing local C/V/A byte stream."""

import json
import os
import shutil
import subprocess
import sys

from campaign_ledger import write_json
from core.workflow_session import WorkflowSession
from ports.protocol import StreamParser, encode_ack, encode_control, encode_observation
from sim.camera_sim import CameraSim
from sim.run_workflow import run_workflow


PAYLOAD = "156+123+516+231"


def exchange(inject, out):
    expected = run_workflow(inject=inject)
    session = WorkflowSession(PAYLOAD, "sim-workflow", mode="SIM")
    camera = CameraSim(PAYLOAD, inject)
    parser = StreamParser()
    trace = []
    stream = bytearray()
    now = 0
    seq = 0

    def transmit(data, now_ms):
        stream.extend(data)
        split = len(data) // 2
        if parser.feed(data[:split], now_ms):
            raise ValueError("Frame accepted early")
        packets = parser.feed(data[split:], now_ms + 1)
        if len(packets) != 1:
            raise ValueError("Valid fragmented frame was lost")
        return packets[0]

    for ack, completion in zip(expected["controller_receipts"][::2], expected["controller_receipts"][1::2]):
        requirement = session.requirement()
        observation = None
        if requirement["step"]["action"] in {"PICK", "PLACE"}:
            frame = camera.get_frame(requirement["camera_id"])
            observation = camera.detect(requirement["kind"], frame, requirement["target_id"], requirement["station"])
            now = frame["timestamp_ms"]
            seq += 1
            packet = transmit(encode_observation(seq, observation), now)
            if packet["frame_id"] != observation["frame_id"] or packet["is_simulated"] is not True:
                raise ValueError("Observation identity lost on wire")
        seq += 1
        command_seq = seq
        command = transmit(encode_control(seq, "PING"), now)
        issued = session.issue(observation, now_ms=now)["intent"]
        if not issued or issued["request_id"] != completion["request_id"]:
            raise ValueError("Workflow and controller references disagree")
        before = session.step_index
        for receipt in (ack, dict(ack, status="BUSY"), completion):
            seq += 1
            packet = transmit(encode_ack(seq, receipt["status"], receipt.get("reason"), command_seq), now + 2)
            if packet["command_seq"] != command["seq"]:
                raise ValueError("Receipt references a different command")
            status = session.receive(dict(receipt, seq=seq, status=packet["status"]), now + 3)
            trace.append({
                "wire_seq": seq, "command_seq": command_seq, "request_id": issued["request_id"],
                "action": issued["step"]["action"], "receipt": packet,
                "step_index": session.step_index, "counters": status["counters"], "workflow": status["status"],
            })
            if receipt["status"] != "DONE" and session.step_index != before:
                raise ValueError("A non-terminal receipt changed the material ledger")
        if parser.feed(encode_ack(seq, completion["status"], completion.get("reason"), command_seq), now + 5):
            raise ValueError("Repeated sequence was accepted")
        if completion["status"] == "DONE":
            duplicate = session.receive(dict(completion, seq=seq + 1), now + 5)
            if session.step_index != before + 1 or not duplicate.get("receipt_rejection"):
                raise ValueError("Duplicate DONE re-applied a material movement")
        now += 40

    observed = session.summary()
    fields = ("status", "confirmed_steps", "counters", "station")
    if any(observed[k] != expected["summary"][k] for k in fields):
        raise ValueError("Wire replay differs from independent field/workflow run")
    if inject and (observed["pending"] is None or session.issue(now_ms=now + 5001)["intent"] is not None):
        raise ValueError("An uncertain material action was re-issued")
    tag = inject or "success"
    (out / f"{tag}.wire").write_bytes(stream)
    with (out / f"{tag}.jsonl").open("w", encoding="utf-8") as log:
        for step in trace:
            log.write(json.dumps(step, ensure_ascii=False) + "\n")
    write_json(out / f"{tag}-field.json", {
        "workflow": observed, "field": expected["field"],
        "transitions": expected["transitions"], "controller_submissions": expected["controller_submissions"],
    })
    return {"status": observed["status"], "commands": expected["controller_submissions"],
            "field_transfers": len(expected["transitions"]), "confirmed": observed["confirmed_steps"],
            "frames": seq, "duplicates_rejected": parser.rejected}


def replay(out, transport, run_command):
    results = {kind or "success": exchange(kind, out) for kind in (None, "timeout", "ack-loss")}
    if (results["success"]["status"] != "DONE" or results["timeout"]["field_transfers"] != 0
            or results["ack-loss"]["field_transfers"] != 1):
        raise ValueError("Missing receipt distinguished neither execution state")
    c_dir = transport / "stm32"
    flags = ["-std=c99", "-Wall", "-Wextra", "-Werror"]
    compiler = shutil.which("gcc")
    binary = out.parent / "host_vectors.exe"
    command = [*flags, str(c_dir / "host_vectors.c"), str(c_dir / "protocol.c"), "-lm", "-o", str(binary)]
    tests = ["-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_protocol.py", "-v"]
    if compiler and run_command([compiler, *command], out, "compile-c-native") == 0:
        decoder = [str(binary)]
        test_command = [sys.executable, *tests]
        backend = "native gcc"
    else:
        wsl = shutil.which("wsl.exe")
        if not wsl:
            raise RuntimeError("A working native or WSL GCC is required for host C decode parity")
        distro = os.environ.get("K230_WSL_DISTRO", "Ubuntu-24.04")
        prefix = [wsl, "-d", distro, "--exec"]

        def linux_path(path):
            result = subprocess.run([*prefix, "wslpath", "-a", "-u", str(path)],
                                    capture_output=True, check=True)
            return result.stdout.decode("utf-8").strip()

        command = [*prefix, "gcc", *flags, linux_path(c_dir / "host_vectors.c"),
                   linux_path(c_dir / "protocol.c"), "-lm", "-o", linux_path(binary)]
        if run_command(command, out, "compile-c-wsl"):
            raise RuntimeError("WSL host C parser build failed")
        decoder = [*prefix, linux_path(binary)]
        test_command = [wsl, "-d", distro, "--cd", linux_path(transport), "--exec", "python3", *tests]
        backend = f"WSL {distro} gcc"
    for tag, summary in results.items():
        data = (out / f"{tag}.wire").read_bytes()
        frames = StreamParser().feed(data)
        expected = [f"{item['type']},{item['seq']}" for item in frames]
        parsed = subprocess.run(decoder, input=data, capture_output=True, check=True)
        actual = parsed.stdout.decode("ascii").splitlines()
        if actual != expected or len(actual) != summary["frames"]:
            raise ValueError(f"Python/C parser disagreement: {tag}")
        summary["matched_host_c_frames"] = len(actual)
    tests = run_command(test_command, out, "protocol-tests", transport)
    write_json(out / "protocol.json", {
        "transport": "loopback byte stream", "cases": results, "protocol_test_exit_code": tests,
        "host_c_backend": backend,
        "scope": "Existing v1 C/V/A replay with pick receipts and existing host protocol fault vectors",
    })
    return {"status": "PASS" if tests == 0 else "FAIL", "reason": None if tests == 0 else "Protocol tests failed"}
