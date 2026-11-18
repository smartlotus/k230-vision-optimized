"""Collect the standalone Blender files, evidence, and actual rebuild inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile


HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from mission_vision import build_round
from perception import ROAD_DIR, TOP_DIR


def pack(hub,destination):
    files={}
    def add(source,archive):
        if not source.is_file():
            raise FileNotFoundError(source)
        files[str(archive).replace("\\","/")]=source
    for name in ("vehicle.blend","competition_round.blend","整轮回放.mp4","live_profile_round.json",
                 "live_challenge_round.json","servo_trials.json","stack_contact.json","adapter/report.json"):
        if not (HERE/"output"/name).is_file():
            raise FileNotFoundError(name)
    for source in HERE.rglob("*"):
        if source.is_file() and "__pycache__" not in source.parts and source.name!=".gitattributes":
            add(source,source.relative_to(PROJECT))
    add(HERE/"README.md","打开说明.md")
    build_round()
    transport=PROJECT/"k230_transport_mvp"
    for module in tuple(sys.modules.values()):
        source=Path(getattr(module,"__file__","") or ".").resolve()
        if source.is_relative_to(transport) and source.suffix==".py":
            add(source,source.relative_to(PROJECT))
    for rel in ("research/topface/native_cad.py","research/topface/task_geometry.py",
                "k230_transport_mvp/data/task_qr.png"):
        add(PROJECT/rel,rel)
    mapdir=Path("worktrees/ops-temporal-camera-v1/docs/temporal_camera/map-correction")
    caddir=Path("webhelp/camera_xy_v2_audit_20261117/k230_material_block_model_bundle/k230_material_block_model")
    for rel in [mapdir/"field_gray_v1.blend",mapdir/"field_mask.png",caddir/"material_block.glb",
                caddir/"model_validation.json",TOP_DIR/"topface.onnx",ROAD_DIR/"roadnet.onnx",
                Path("webhelp/official-2027-task.pdf")]:
        add(hub/rel,rel)
    fingerprints={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sorted(files.items())}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(destination,"x",compression=ZIP_DEFLATED,compresslevel=5) as archive:
        for name,path in sorted(files.items()):
            archive.write(path,name)
        archive.writestr("SHA256.json",json.dumps(fingerprints,ensure_ascii=False,indent=2))
    with ZipFile(destination) as archive:
        for name,expected in fingerprints.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=expected:
                raise ValueError(name)
    print(f"{destination}: {len(files)} verified entries plus SHA256.json")


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--hub",required=True,type=Path)
    cli.add_argument("--zip",required=True,type=Path)
    args=cli.parse_args()
    pack(args.hub.resolve(),args.zip.resolve())
