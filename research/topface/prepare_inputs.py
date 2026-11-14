"""Freeze original inputs and generate the first reviewable Kaggle notebook."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
from cloud_store import atomic_json, sha

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DOC=ROOT/"docs/topface"
LENS_SOURCE="https://wiki.lckfb.com/zh-hans/lushan-pi-k230/image-recog/use-sensor.html"
BOARD_SOURCE="https://wiki.lckfb.com/zh-hans/lushan-pi-k230/quick-start.html"
FOLDER="1MwltNmaI24HjvcxnG1w608qui42RAPKf"
DATASET="lotus30/k230-topface-smoke-input-20261114"
KERNEL="lotus30/k230-topfacenet-original-cad-drive-smoke-20261114"


def field(value,unit,status,source,confidence,verification_method):
    return dict(value=value,unit=unit,status=status,source=source,
                confidence=confidence,verification_method=verification_method)


def main(cad,upstream):
    DOC.mkdir(parents=True,exist_ok=True)
    source_files={"sensor_modes":upstream/"k230_transport_mvp/evidence/board/sensor-fps-modes.json",
        "sensor_startup":upstream/"k230_transport_mvp/evidence/board/preview-fps-native-modes.json",
        "capture_metadata":ROOT/"k230_transport_mvp/data/ai_pilot/material-series-20261109-g/capture-03-lab.json",
        "r4":ROOT/"k230_transport_mvp/data/ai_pilot/material_manifest_reviewed_r4_20261110.json",
        "board_preflight":ROOT/"docs/validation/native-fps-20261112/preflight.json",
        "native_smoke_log":ROOT/"docs/validation/native-fps-20261112/smoke01.log"}
    witnesses={name:{"path":str(path),"bytes":path.stat().st_size,"sha256":sha(path)}
               for name,path in source_files.items()}
    modes=json.loads(source_files["sensor_modes"].read_text(encoding="utf-8-sig"))
    capture=json.loads(source_files["capture_metadata"].read_text(encoding="utf-8-sig"))
    r4=json.loads(source_files["r4"].read_text(encoding="utf-8-sig"))
    startup=json.loads(source_files["sensor_startup"].read_text(encoding="utf-8-sig"))
    if not any(row["camera_id"]==2 and "gc2093_csi2" in row["listed"][0] for row in modes["rows"]):
        raise ValueError("Historical CSI2 identity no longer matches the inspected evidence")
    if (capture["camera_id"],capture["width"],capture["height"],capture["is_simulated"])!=(2,640,360,False):
        raise ValueError("Frozen real capture provenance mismatch")
    approved=[image for image in r4["images"] if image.get("review",{}).get("status")=="APPROVED"]
    evidence={"observed_utc":datetime.now(timezone.utc).isoformat(),"witnesses":witnesses,
        "sensor_modes":modes,"startup":startup,"capture_metadata_without_lab_grid":
            {key:value for key,value in capture.items() if key!="lab_grid"},
        "r4_counts":{"images":len(r4["images"]),"approved_images":len(approved),
            "approved_top_boxes":sum(len(image["review"]["objects"]) for image in approved),
            "approved_splits":{s:sum(image["split"]==s for image in approved) for s in ("train","val","test")}},
        "live_hardware_probed_this_run":False,"real_annotations_modified":False}
    atomic_json(DOC/"hardware_evidence.json",evidence)
    historical="historical local board/capture evidence, SHA-bound in hardware_evidence.json; not a fresh board read"
    user="user chat 2026-11-14; height clarified as lens-to-top and small fixed-focus lens confirmed"
    params={}
    def add(name,value,unit,status,source,confidence="high",method=historical):
        params[name]=field(value,unit,status,source,confidence,method)
    add("board","LCKFB-LSPI-K230-1G-CanMV",None,"measured","native_smoke_log")
    add("firmware","v1.8-36-gae2f20d",None,"measured","native_smoke_log")
    add("board_memory",1,"GB","from_datasheet",BOARD_SOURCE,method="board variant specification; not an exhaustive RAM test")
    add("csi_port",2,None,"read_from_config","k230_transport_mvp/k230/camera_roles.py")
    add("stream_channel",0,None,"read_from_config","camera_manager.py default output channel",method="default API channel; fresh channel enumeration pending")
    add("sensor_model","GC2093",None,"measured","sensor_modes")
    add("sensor_native_dimensions",[1920,1080],"px","measured","sensor_startup")
    add("sensor_supported_modes",modes["rows"][-1]["listed"][1],None,"measured","sensor_modes")
    add("width",640,"px","measured","capture_metadata")
    add("height",360,"px","measured","capture_metadata")
    add("fps_configured",30,"fps","read_from_config","sensor_startup")
    add("pixel_format","RGB565",None,"read_from_config","k230_transport_mvp/k230/camera_manager.py")
    add("model_color_order","RGB",None,"read_from_config","k230_transport_mvp/tools/material_preprocess.py")
    add("lens_model","LCKFB small fixed-focus GC2093 module",None,"estimated",user,method="user identification; lens serial/individual optics not measured")
    add("camera_height_mm",[100,400],"mm","estimated",user,method="user operating envelope; vertical lens-to-object-top distance")
    add("camera_plane_angle_deg",[80,100],"deg","estimated",user,method="90 means top-down; signed deviation +/-10 degrees from vertical")
    add("camera_motion","vertical",None,"estimated",user,method="user operating condition; no motion-control commands enabled")
    add("focal_length_mm",4.3,"mm","from_datasheet",LENS_SOURCE,method="small fixed-focus module nominal EFL")
    add("horizontal_fov_deg",65,"deg","from_datasheet",LENS_SOURCE,method="small lens nominal horizontal FOV, not calibration")
    add("vertical_fov_deg",40,"deg","from_datasheet",LENS_SOURCE,method="small lens nominal vertical FOV, not calibration")
    add("diagonal_fov_deg",73,"deg","from_datasheet",LENS_SOURCE,method="small lens nominal diagonal FOV, not calibration")
    add("lens_f_number",2,None,"from_datasheet",LENS_SOURCE,method="small lens nominal aperture")
    add("nominal_focus_distance_mm",700,"mm","from_datasheet",LENS_SOURCE,method="vendor small-lens fixed-focus description; 100-400mm sharpness needs real images")
    add("nominal_distortion_bound",.5,"percent","from_datasheet",LENS_SOURCE,method="vendor distortion specification, not Brown-Conrady coefficients")
    for name,unit in [("lens_serial",None),("fps_actual","fps"),("exposure_us","us"),
        ("gain",None),("white_balance",None),("fx","px"),("fy","px"),("cx","px"),("cy","px"),
        ("distortion_coefficients",None),("camera_roll_deg","deg"),("camera_yaw_deg","deg")]:
        add(name,None,unit,"unknown","not present in inspected evidence","unknown",
            "requires current sensor readback or calibration at actual CSI2 resolution/focus")
    add("model_input_width",224,"px","read_from_config","TopFaceNet execution plan; intended new experiment",method="experiment choice, not a board measurement")
    add("model_input_height",224,"px","read_from_config","TopFaceNet execution plan; intended new experiment",method="experiment choice, not a board measurement")
    add("preprocessing",{"layout":"NCHW","resize":"letterbox","interpolation":"bilinear", "normalization":"uint8 / 255 exactly once","pad":114},None,"read_from_config","material_preprocess.py and planned TopFaceNet contract",method="host reference only; native AI2D equivalence remains unverified")
    profile={"schema":"k230-hardware-profile-v1","parameters":params,
        "simulation_priors":{"hfov_deg":[62,68],"status":"estimated","focal_is_square_pixels":True,
            "principal_point":"image midpoint for smoke only","distortion":"none for smoke geometry qualification",
            "focus_blur":"must be assessed from real 100-400mm images before acceptance"},
        "protected":{"uart":False,"motors":False,"production_save":False,"csi0":"quarantined","csi1":"unchanged"}}
    atomic_json(DOC/"hardware_profile.json",profile)
    atomic_json(DOC/"execution_contract.json",{"schema":"topface-execution-v1",
        "source_plan":"K230_TopFaceNet_Sim2Real_Execution_Plan_2026-11-14.md",
        "stages":[{"id":"A","scope":"Drive admission, environment, official quickstart, 1 original-CAD render and audit", "max_seconds":3600},
            {"id":"B","scope":"200 preview frames and contact sheets; user visual review required"},
            {"id":"C","scope":"1000-3000 grouped synthetic frames after preview gate"},
            {"id":"D","scope":"MobileNetV2 and ShuffleNetV2 TopFaceNet; synthetic and approved-real experiments"},
            {"id":"E","scope":"held-out evaluation, ONNX/nncase preparation, board acceptance separately"}],
        "current_stage":"A_PREPARED_NOT_EXECUTED","training_started":False,
        "drive_folder_id":FOLDER,"drive_remote":"gdrive:k230-topfacenet-20261114",
        "kaggle_input":DATASET,"kaggle_kernel":KERNEL,"secret_name":"DOASTRA_GDRIVE_OAUTH_JSON",
        "approval_boundary":"Associate existing Drive secret in new notebook; no new credential in source",
        "height_reference":"lens center to object top plane, vertical distance", "height_mm":[100,400],
        "plane_angle_deg":[80,100],"lens":"small fixed-focus; nominal H65deg EFL4.3mm",
        "original_cad":"fused single mesh; semantic top faces only", "real_data":"r4 immutable; no real ellipse ground truth",
        "scope_note":"Only stage A is executable in this first notebook; remaining stages are not yet implemented/verified"})
    staged=ROOT/".ops/local/topface-input"
    staged.mkdir(parents=True,exist_ok=True)
    for filename in ["geometry.py","cloud_store.py","render_scene.py","audit_dataset.py","kaggle_job.py"]:
        shutil.copy2(HERE/filename,staged/filename)
    shutil.copy2(cad,staged/"cad_bundle.zip")
    shutil.copy2(DOC/"hardware_profile.json",staged/"hardware_profile.json")
    members=[p for p in sorted(staged.iterdir()) if p.is_file() and p.name not in {"dataset-metadata.json","INPUT_MANIFEST.json"}]
    manifest={"schema":"topface-kaggle-input-v1","files":[{"path":p.name,"bytes":p.stat().st_size,"sha256":sha(p)} for p in members],
              "cad_bundle_original_sha256":sha(cad),"source_git_head_before_changes":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
    atomic_json(staged/"INPUT_MANIFEST.json",manifest)
    atomic_json(DOC/"INPUT_MANIFEST.json",manifest)
    atomic_json(staged/"dataset-metadata.json",{"id":DATASET,"title":"K230 TopFaceNet smoke input 20261114", "licenses":[{"name":"other"}],
        "description":"Private original CAD and K230 stage-A smoke code. Original authors retain their rights; no open redistribution license is granted. This version contains only the stage-A smoke workload."})
    kernel_dir=ROOT/".ops/local/topface-kernel"
    kernel_dir.mkdir(parents=True,exist_ok=True)
    intro="""# K230 TopFaceNet stage A: original CAD + Drive smoke\n\nAssociate the existing `DOASTRA_GDRIVE_OAUTH_JSON` Secret under Add-ons -> Secrets, enable Internet, then Save Version -> Save & Run All once. This stage performs storage admission before installing BlenderProc, renders exactly one original-CAD RGB/top-mask sample, audits projection geometry, uploads the complete result from Kaggle to the K230 Drive folder, and reads it back with SHA-256. It does not train or start the 200-frame preview.\n\nCamera: small fixed-focus GC2093 lens; nominal horizontal FOV 65 degrees. Operating range: 100-400mm vertically above the top plane, 80-100 degrees to the plane. Intrinsics remain uncalibrated.\n"""
    code=f'''import hashlib, json, pathlib, sys, subprocess\ninput_root = pathlib.Path('/kaggle/input/k230-topface-smoke-input-20261114')\nexpected_manifest = {sha(staged/"INPUT_MANIFEST.json")!r}\nassert hashlib.sha256((input_root/'INPUT_MANIFEST.json').read_bytes()).hexdigest() == expected_manifest\nsys.path.insert(0,str(input_root))\nfrom kaggle_job import main\nmain({FOLDER!r}, input_root, 'k230-topface-smoke-20261114-a01')\n'''
    notebook={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"}},
        "cells":[{"cell_type":"markdown","metadata":{},"source":intro.splitlines(True),"id":"intro"},
            {"cell_type":"code","metadata":{},"source":code.splitlines(True),"execution_count":None,"outputs":[],"id":"smoke"}]}
    atomic_json(kernel_dir/"topface_smoke.ipynb",notebook)
    atomic_json(kernel_dir/"kernel-metadata.json",{"id":KERNEL,"title":"K230 TopFaceNet original CAD Drive smoke 20261114",
        "code_file":"topface_smoke.ipynb","language":"python","kernel_type":"notebook", "is_private":True,
        "enable_gpu":True,"enable_tpu":False,"enable_internet":True,"dataset_sources":[DATASET],
        "competition_sources":[],"kernel_sources":[],"model_sources":[]})
    shutil.copy2(kernel_dir/"topface_smoke.ipynb",HERE/"topface_smoke.ipynb")
    print(json.dumps({"dataset_dir":str(staged),"notebook_dir":str(kernel_dir),"input_manifest_sha256":sha(staged/"INPUT_MANIFEST.json"),"evidence":evidence["r4_counts"]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--cad",type=Path,required=True)
    parser.add_argument("--upstream",type=Path,required=True)
    args=parser.parse_args()
    main(args.cad,args.upstream)
