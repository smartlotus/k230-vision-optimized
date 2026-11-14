"""Bounded cloud smoke job. Drive admission precedes installation/rendering.

Notebook packages this source from a private, hash-bound input dataset. It does
not train a model or launch the 200-frame preview without its separate gate.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from cloud_store import Drive, SECRET_NAME, archive_directory, atomic_json, sha


def execute(command, log, timeout, cwd=None):
    with Path(log).open("wb") as output:
        result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,
                              timeout=timeout,cwd=cwd,check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed, exit {result.returncode}; see {Path(log).name}")


def main(folder_id, input_root, run_id):
    input_root=Path(input_root)
    root=Path("/kaggle/working")/run_id
    root.mkdir(exist_ok=False)
    started=time.monotonic()
    state={"schema":"k230-kaggle-smoke-v1","run_id":run_id,"status":"STARTED",
           "automatic_replay":False,"optimizer_steps":0,"images_rendered":0}
    try:
        manifest=json.loads((input_root/"INPUT_MANIFEST.json").read_text())
        for member in manifest["files"]:
            file=input_root/member["path"]
            if file.stat().st_size != member["bytes"] or sha(file) != member["sha256"]:
                raise ValueError("Private input dataset bytes differ from frozen manifest")
        environment={"python":sys.version,"platform":platform.platform(),"packages":{}}
        for package in ("torch","torchvision","numpy","opencv-python","blenderproc"):
            try:environment["packages"][package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:environment["packages"][package]=None
        import torch
        environment.update(cuda_available=torch.cuda.is_available(),torch_cuda=torch.version.cuda,
            gpu_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        gpu_command=shutil.which("nvidia-smi")
        if gpu_command:
            execute([gpu_command],root/"nvidia-smi.log",30)
        atomic_json(root/"environment.json",environment)
        # Stop here if the exact runtime has not been granted access to the Secret.
        state["stage"]="DRIVE_ADMISSION"
        atomic_json(root/"STATE.json",state)
        drive=Drive(folder_id)
        drive.verify_folder()
        probe=root/"storage-probe.json"
        atomic_json(probe,{"project":"k230-topface","run_id":run_id,"bytes_source":"cloud producer"})
        state["drive_probe"]=drive.export(probe,run_id+"-storage-probe",root/"DRIVE_PROBE.json")
        # Prepare one exact original CAD asset; retain original bundle as provenance.
        assets=root/"assets"
        assets.mkdir()
        with zipfile.ZipFile(input_root/"cad_bundle.zip") as bundle:
            for name in bundle.namelist():
                if not name.endswith("/"):
                    (assets/Path(name).name).write_bytes(bundle.read(name))
        shutil.copy2(input_root/"hardware_profile.json",root/"hardware_profile.json")
        # Keep preinstalled CUDA and PyTorch. BlenderProc manages Blender's own Python.
        state["stage"]="ENVIRONMENT_INSTALL"
        atomic_json(root/"STATE.json",state)
        execute([sys.executable,"-m","pip","install","blenderproc==2.8.0"],
                root/"install-blenderproc.log",600)
        executable=shutil.which("blenderproc")
        if not executable:raise RuntimeError("Installed BlenderProc CLI not found")
        # Official quickstart uses a separate test object, never the target CAD labels.
        execute([executable,"quickstart"],root/"blenderproc-quickstart.log",900,cwd=root)
        state["stage"]="CAD_RENDER_SMOKE"
        atomic_json(root/"STATE.json",state)
        execute([executable,"run",str(input_root/"render_scene.py"),"--assets",str(assets),
            "--output",str(root/"smoke"),"--count","1","--seed","20261114"],
            root/"render-smoke.log",600)
        from audit_dataset import audit
        report=audit(root/"smoke")
        if report["images"] != 1 or report["instances"] != 1:
            raise RuntimeError("Smoke must contain exactly one intact top")
        state.update(status="SMOKE_PASSED_ARCHIVE_PENDING",images_rendered=1,
                     elapsed_seconds=time.monotonic()-started)
        atomic_json(root/"STATE.json",state)
        archive=root.parent/(run_id+".tar.gz")
        archive_directory(root,archive)
        receipt=drive.export(archive,run_id,root.parent/(run_id+"-DRIVE.json"))
        state.update(status="PASS",drive_archive=receipt,elapsed_seconds=time.monotonic()-started,
                     next_stage="200_FRAME_PREVIEW_REQUIRES_SEPARATE_VERSION_AND_REVIEW")
        # A small final receipt binds the unchanged immutable archive's manifest.
        atomic_json(root.parent/(run_id+"-RESULT.json"),state)
        print(json.dumps(state))
    except Exception as error:
        # Exception text from secret helpers can contain private endpoint details.
        state.update(status="NEEDS_SECRET_OR_TECHNICAL_REPAIR",error_type=type(error).__name__,
                     elapsed_seconds=time.monotonic()-started)
        atomic_json(root/"STATE.json",state)
        print(json.dumps(state))
        raise RuntimeError("Bounded smoke stopped; inspect STATE.json and stage log") from None


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--folder-id",required=True)
    parser.add_argument("--input-root",type=Path,required=True)
    parser.add_argument("--run-id",required=True)
    args=parser.parse_args()
    main(args.folder_id,args.input_root,args.run_id)
