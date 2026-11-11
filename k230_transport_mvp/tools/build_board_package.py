"""Build a reviewable SD-card bundle without changing a connected board."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(output):
    files = [ROOT / name for name in ('run_board.py','run_multicam_preview.py','run_block_preview.py','run_block_service.py','run_vision_preview.py','run_tuning_panel.py','run_tuned_service.py','run_live_workbench.py')]
    for folder in ('core', 'vision', 'ports', 'k230'):
        files.extend(sorted((ROOT / folder).rglob('*.py')))
    manifest = {}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if '__pycache__' in path.parts:
                continue
            data = path.read_bytes()
            relative = path.relative_to(ROOT).as_posix()
            manifest[relative] = hashlib.sha256(data).hexdigest()
            info = zipfile.ZipInfo('transport_mvp/' + relative, (2026, 10, 31, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
        info = zipfile.ZipInfo('transport_mvp/manifest.json', (2026, 10, 31, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, json.dumps(manifest, indent=2))
    return {'file': str(output), 'files': len(manifest),
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dist'/'transport_mvp.zip')
    print(json.dumps(build(parser.parse_args().output), indent=2))
