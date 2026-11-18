"""Read-only verification of the archived USB engineering evidence."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence/board-20261117'
TRIALS = EVIDENCE / 'usb-engineering-v1'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(git_blobs=False):
    rows = []
    for name, expected in (('uvc10-b', 293), ('road30-b', 305)):
        report = load(EVIDENCE / (name + '.json'))
        assert report['status'] == 'MEASURED'
        assert report['uvc_attempts'] == report['uvc_frames'] == expected
        assert report['uvc_drops'] == 0 and report['cleanup_errors'] == []
        assert report['final_memory']['sys_mmz']['used'] == 0
        rows.append({'run_id': name, 'frames': expected, 'fps': report['scheduler_cycles_fps']})
    for name, expected, shared in (('road-preview15-b', 118, False),
                                  ('both120-b', 749, True), ('both-restart30-b', 187, True)):
        report = load(TRIALS / (name + '.json'))
        intent = load(TRIALS / (name + '-intent.json'))
        script = (TRIALS / (name + '-executed.py')).read_text(encoding='utf8')
        ast.parse(script)
        assert hashlib.sha256(script.encode()).hexdigest() == intent['executed_script_sha256']
        assert report['run_id'] == intent['run_id'] == name
        assert report['status'] == 'ENGINEERING_MEASURED'
        assert report['attempts'] == report['frames'] == expected
        assert report['roadnet_inferences'] == report['postprocess_frames'] == report['displayed_frames'] == expected
        assert report['shared_inferences'] == report['topface_inferences'] == (expected if shared else 0)
        assert report['decoded_content_shape'] == [180, 320]
        assert report['capture_failures'] == 0 and report['cleanup_errors'] == []
        assert report['memory_sampling_failures'] == 0
        assert report['final_memory']['sys_mmz']['used'] == 0
        assert report['accuracy_test'] == report['road_quality_test'] == 'NOT_RUN_BY_USER_SCOPE'
        assert 'USB_ENGINEERING_RESULT:' in (TRIALS / (name + '-terminal.log')).read_text(encoding='utf8')
        rows.append({'run_id': name, 'frames': expected, 'fps': report['fps']})
    preview = load(TRIALS / 'PREVIEW.json')
    assert preview['first']['available'] and preview['second']['available']
    assert preview['second']['frameId'] > preview['first']['frameId']
    assert preview['first']['ageMs'] == preview['second']['ageMs'] == 0
    final = load(TRIALS / 'FINAL_STATE.json')
    assert final['board']['connected'] and final['board']['ready']
    assert final['script_running']['running'] is False
    assert digest(ROOT / 'evidence/delivery/roadnet-20261117/K230_RoadNet_Delivery_20261117-r3.zip') == 'a2b13c74920a1d08eb530a9e2b4e0ecbab4872ed4a8ec6636a3046993477d805'
    for path in (ROOT / 'reference/usb_engineering.py', ROOT / 'reference/run_usb_engineering.py'):
        ast.parse(path.read_text(encoding='utf8'))
    for path in (ROOT / 'README.md', ROOT / 'CURRENT_RUNTIME_FACTS.md', ROOT / 'reference/BOARD_SMOKE.md',
                 ROOT / 'reference/USB_ENGINEERING.md', TRIALS / 'REPORT.md'):
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf8')):
            if '://' not in link and not link.startswith('#'):
                assert (path.parent / link.split('#')[0]).exists(), (path, link)
    manifest = TRIALS / 'FILE_MANIFEST.json'
    checked_blobs = 0
    if manifest.exists():
        for name, row in load(manifest).items():
            path = ROOT / name
            assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], name
            if git_blobs:
                raw = subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)
                assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], name
                checked_blobs += 1
    return {'status': 'USB_ENGINEERING_EVIDENCE_VERIFIED', 'runs': rows,
            'git_blob_files_checked': checked_blobs,
            'accuracy_test': 'NOT_RUN_BY_USER_SCOPE', 'road_quality_test': 'NOT_RUN_BY_USER_SCOPE',
            'scope': 'Actual camera-to-model/output/display/restart counts and cleanup only'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--git-blobs', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(git_blobs=args.git_blobs)))
