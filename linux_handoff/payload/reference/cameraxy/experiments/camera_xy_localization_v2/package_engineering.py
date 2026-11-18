"""Create a byte-verified local handoff for an exclusive-owner CameraXY smoke run."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    source = Path(__file__).resolve().parent
    files = {name:(source / name).read_bytes() for name in ("board_usb_smoke.py", "board_probe.py")}
    for name in ("engineering.json", "engineering.kmodel"):
        files[name] = (a.model_dir / name).read_bytes()
    document = json.loads(files["engineering.json"])
    from board_usb_smoke import engineering_contract
    engineering_contract(document)
    if hashlib.sha256(files["engineering.kmodel"]).hexdigest() != document["kmodel"]["kmodel_sha256"]:
        raise ValueError("Engineering model identity differs")
    manifest = {name:hashlib.sha256(data).hexdigest() for name, data in files.items()}
    files["FILES_SHA256.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    files["README.txt"] = (
        "CAMERAXY ENGINEERING ONLY; NO ACCURACY OR ROAD QUALITY ACCEPTANCE.\n"
        "Only the current exclusive COM3/USB owner may upload or run this package.\n"
        "Use a fresh isolated /tmp/xy... or /sdcard/xy... directory. Never change startup files.\n"
        "Verify every member against FILES_SHA256.json after transfer, including the model.\n"
        "Insert that exact directory first in sys.path in a fresh CanMV script session.\n"
        "from board_usb_smoke import run\n"
        "run(ROOT, 'chain30-fresh.json', 'openaicam#openaicam', frames=30, preview=True)\n"
        "For /sdcard/xy... installs, reports go to /data/logs/<install name>/.\n"
        "For /tmp/xy... installs, reports stay in the install directory.\n"
        "Only after full report readback passes, use a fresh filename for a bounded longer run.\n"
        "Read back reports and verify counts, shapes, finite outputs, null primary XY, None confidences,\n"
        "startup hashes, every cleanup error and final memory. Download one fresh preview frame.\n"
        "The prototype tests learned_map. The selected fused_map is not tested by this package.\n"
        "RAM files are volatile. Keep all persistent reports on the host.\n"
    ).encode()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.out, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(a.out) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(files):
            raise ValueError("Handoff archive integrity failure")
        for name, data in files.items():
            if archive.read(name) != data:
                raise ValueError("Handoff member differs: " + name)
    receipt = dict(status="HOST_PACKAGE_BYTES_VERIFIED", archive=str(a.out),
                   sha256=hashlib.sha256(a.out.read_bytes()).hexdigest(), bytes=a.out.stat().st_size,
                   members=len(files), member_sha256=manifest, authority="LOCAL_PACKAGING_ONLY",
                   native_execution="SEE_SEPARATE_NATIVE_RUN_REPORTS")
    with Path(str(a.out) + ".json").open("x", encoding="utf8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
