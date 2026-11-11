"""Compile/run a synthetic PC-only communication demo; opens no board or UART."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ports.protocol import encode_material_snapshot, encode_ack


def example(frame, count=3, **changes):
    value = dict(version=1, status="OK" if count else "NOT_FOUND", camera_id="CSI2",
                 frame_id=frame, timestamp_ms=frame*100, profile_id="a"*64,
                 source_session="b"*32, width=640, height=360, coord_frame="IMAGE_PX",
                 is_simulated=False, complete=True, reason=None,
                 objects=[dict(color_id=color,u_px=u,v_px=150,semimajor_px=25,
                               semiminor_px=20,angle_rad=0.4,score_hint=None,
                               stable=True,geometry_valid=True)
                          for color,u in [(2,100),(4,220),(5,340)][:count]])
    value.update(changes)
    return value


def export(output):
    compiler = shutil.which("gcc")
    if not compiler and Path("C:/msys64/mingw64/bin/gcc.exe").exists():
        compiler = "C:/msys64/mingw64/bin/gcc.exe"
    if not compiler:
        raise RuntimeError("C compiler unavailable: receiver proof UNKNOWN")
    env = dict(os.environ)
    env["PATH"] = str(Path(compiler).parent)+os.pathsep+env.get("PATH", "")
    values = [example(1),example(2,1),example(3,0),
              example(4,0,status="UNKNOWN",complete=False,reason="SCENE_LIMIT:4"),
              example(5,is_simulated=True),example(6),example(6)]
    inputs = []
    commands = []
    for seq,value in enumerate(values,1):
        now = seq*10
        wire = encode_material_snapshot(seq,value)
        inputs.append(dict(local_ms=now,wire_ascii=wire.decode("ascii"),synthetic=True))
        commands.append("F %d %s\n"%(now,wire.hex()))
    commands += ["F 150 "+encode_ack(8,"PONG").hex()+"\n","P 160\n","P 161\n"]
    sources = ["stm32/protocol.c","stm32/protocol.h","stm32/material_receiver.c",
               "stm32/material_receiver.h","stm32/host_material.c","ports/protocol.py",
               "k230/uart_adapter.py","tests/test_material_protocol.py",
               "tools/export_material_comm_evidence.py"]
    with tempfile.TemporaryDirectory(prefix="k230-material-evidence-") as temp:
        executable = Path(temp)/("receiver.exe" if os.name == "nt" else "receiver")
        command = [compiler,"-std=c99","-Wall","-Wextra","-Werror",
                   *[str(ROOT/name) for name in ("stm32/protocol.c","stm32/material_receiver.c",
                                                "stm32/host_material.c")],"-lm","-o",str(executable)]
        compiled = subprocess.run(command,capture_output=True,text=True,env=env,check=True)
        result = subprocess.run([str(executable)],input="".join(commands),capture_output=True,
                                text=True,env=env,check=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    if ([row["count"] for row in rows] != [3,1,0,0,3,3,3,3,3,0] or
        [row["usable"] for row in rows] != [1,1,0,0,0,1,1,1,1,0] or
        rows[-1]["status"] != 7 or rows[6]["received_ms"] != 60):
        raise RuntimeError("Synthetic receiver trace failed; do not claim host PASS")
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    trace = dict(scope="PC synthetic only; not hardware recognition or serial acceptance",
                 inputs=inputs,driver_commands=commands,receiver_outputs=rows)
    trace_path = output/"synthetic-trace.json"
    trace_path.write_text(json.dumps(trace,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    version = subprocess.run([compiler,"--version"],capture_output=True,text=True,env=env,check=True)
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),status="HOST_SYNTHETIC_PASS",
                  board_access=False,uart_access=False,training_steps=0,
                  compiler=version.stdout.splitlines()[0],compiler_flags=command[1:5],
                  compile_stderr=compiled.stderr,receiver_stderr=result.stderr,
                  receiver_bytes_host=rows[0]["receiver_bytes"],
                  memory_scope="sizeof on current host ABI only; not STM32 memory measurement",
                  trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                  source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
    (output/"verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({key:report[key] for key in ("status","receiver_bytes_host","board_access","uart_access")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    export(parser.parse_args().output)
