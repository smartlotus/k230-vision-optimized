"""Bounded, isolated engineering checks using the existing CanMV MCP client.

Installs the immutable R2 archive with chunk readback; optionally substitutes
current CameraXY source after a documented native fix. Never writes startup,
production profiles, or any other experiment's directory.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'k230_transport_mvp'))
from tools.deploy_native_trial import CanMvClient

CHUNK = 262144
BOARD_SOURCES = ('board_multiroute.py', 'board_temporal.py', 'competition_paths.py',
                 'embedded_filter.py', 'localization_runtime.py')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')
    temp.replace(path)


def idle(client):
    if client.tool('canmv_script_running').get('running') is not False:
        raise RuntimeError('Board busy or unknown; existing session preserved')


def remote_root(value):
    if not re.fullmatch(r'/sdcard/k230_temporal_v1_[a-z0-9_]+', value):
        raise ValueError('A dedicated CameraXY directory is required')
    return value


def install(args, client):
    raw = args.archive.read_bytes()
    external = json.loads(args.archive.with_suffix('.SHA256.json').read_text(encoding='utf8'))
    if sha(raw) != external['sha256'] or len(raw) != external['bytes']:
        raise ValueError('Archive differs from the delivery receipt')
    prefix = 'k230_temporal_v1_gray_' + args.setting + '/'
    with zipfile.ZipFile(args.archive) as bundle:
        manifest = json.loads(bundle.read('SHA256_MANIFEST.json'))
        if set(bundle.namelist()) != set(manifest) | {'SHA256_MANIFEST.json'}:
            raise ValueError('Archive member set differs')
        for name, record in manifest.items():
            data = bundle.read(name)
            if len(data) != record['bytes'] or sha(data) != record['sha256']:
                raise ValueError('Archive member differs: ' + name)
        files = {name[len(prefix):]: bundle.read(name) for name in manifest if name.startswith(prefix)}
    changes = {}
    if args.current_source:
        for name in BOARD_SOURCES:
            data = (Path(__file__).parent / name).read_bytes()
            if data != files[name]:
                changes[name] = {'archive_sha256': sha(files[name]), 'current_sha256': sha(data)}
                files[name] = data
    idle(client)
    if client.tool('canmv_stat_file', {'path': args.remote}).get('exists') is not False:
        raise RuntimeError('New install directory required; existing files preserved')
    jobs = []
    for name, data in files.items():
        target = args.remote + '/' + name
        parts = [target + '.p' + str(i) for i in range((len(data) + CHUNK - 1) // CHUNK)]
        if any(len(p.encode()) >= 64 for p in parts):
            raise ValueError('CanMV file paths must stay below 64 bytes')
        jobs.append({'target': target, 'parts': parts, 'bytes': len(data), 'sha256': sha(data)})
    receipt = dict(status='INSTALLING', setting=args.setting, remote=args.remote,
                   observed_utc=datetime.now(timezone.utc).isoformat(),
                   archive_sha256=sha(raw), verified_archive_members=len(manifest) + 1,
                   source_changes=changes, files=jobs, verified_parts=[], automatic_retry=False)
    save(args.out, receipt)
    if client.tool('canmv_mkdir', {'path': args.remote}).get('success') is not True:
        raise RuntimeError('Directory creation unconfirmed')
    for (name, data), job in zip(files.items(), jobs):
        for index, part in enumerate(job['parts']):
            block = data[index * CHUNK:(index + 1) * CHUNK]
            if client.tool('canmv_write_file', {'path': part, 'encoding': 'base64',
                           'content': base64.b64encode(block).decode()}).get('success') is not True:
                raise RuntimeError('Part write unconfirmed; no retry')
            readback = client.tool('canmv_read_file', {'path': part, 'encoding': 'base64'})
            if base64.b64decode(readback['content'], validate=True) != block:
                raise RuntimeError('Part readback differs')
            receipt['verified_parts'].append({'path': part, 'bytes': len(block), 'sha256': sha(block)})
            save(args.out, receipt)
        print('Verified upload ' + name, flush=True)
    idle(client)
    # The existing installer performs no retry and reopens final files for SHA.
    sys.path.insert(0, str(ROOT / 'research/topface'))
    from deploy_board_trial import join_script
    marker = 'TEMPORAL_INSTALL_' + uuid.uuid4().hex + ':'
    client.tool('canmv_run_script', {'script': join_script(jobs, marker)})
    deadline = time.monotonic() + 45
    while client.tool('canmv_script_running').get('running') is not False:
        if time.monotonic() >= deadline:
            raise TimeoutError('Assembly unknown; do not repeat')
        time.sleep(.5)
    terminal = client.tool('canmv_terminal_output', {'clear': False})['text']
    args.out.with_suffix('.log').write_text(terminal, encoding='utf8')
    lines = [line[len(marker):] for line in terminal.splitlines() if line.startswith(marker)]
    expected = {job['target']: {'bytes': job['bytes'], 'sha256': job['sha256']} for job in jobs}
    if len(lines) != 1 or json.loads(lines[0]) != expected:
        raise RuntimeError('Final stored-file identity unconfirmed')
    receipt.update(status='VERIFIED_BOARD_FILES_ONLY', stored_files=expected)
    save(args.out, receipt)
    print(json.dumps({'status': receipt['status'], 'remote': args.remote, 'files': len(files)}), flush=True)


def run(args, client):
    if not re.fullmatch(r'[a-z0-9_-]+', args.run_id) or not 1 <= args.frames <= 120:
        raise ValueError('Use a unique run ID and 1..120 attempts for this engineering smoke')
    idle(client)
    report = args.remote + '/' + args.run_id + '.json'
    envelope = args.remote + '/' + args.run_id + '-entry.json'
    for path in (report, report + '.partial', envelope):
        if client.tool('canmv_stat_file', {'path': path}).get('exists') is not False:
            raise RuntimeError('Existing report or unknown state; do not repeat')
    script = ("import sys,json,os\nroot=" + repr(args.remote) + "\nsys.path.insert(0,root)\n"
              "entry={'status':'STARTED','root':root,'run_id':" + repr(args.run_id) + "}\n"
              "try:\n from board_multiroute import run\n"
              " result=run(root," + repr(args.run_id) + ",'openaicam#openaicam',frames=" + str(args.frames) +
              ",mode='filtered')\n entry['status']='RETURNED'\n entry['report_status']=result['status']\n"
              "except BaseException as error:\n entry['status']='ENTRY_FAILED'\n"
              " entry['error']=str(error)\n sys.print_exception(error)\n"
              "finally:\n with open(" + repr(envelope) + ",'w') as stream:stream.write(json.dumps(entry))\n"
              " print('TEMPORAL_ENTRY:'+json.dumps(entry))\n")
    args.out.with_suffix('.runner.py').write_text(script, encoding='utf8')
    save(args.out, dict(status='DISPATCHING', report=report, entry=envelope, frames=args.frames,
                        accuracy='NOT_RUN', road_test='NOT_RUN', automatic_retry=False))
    client.tool('canmv_run_script', {'script': script})
    print(json.dumps({'status': 'DISPATCHED_ONLY', 'report': report, 'entry': envelope}), flush=True)


def fetch(args, client):
    idle(client)
    terminal = client.tool('canmv_terminal_output', {'clear': False})['text']
    args.out.with_suffix('.log').write_text(terminal, encoding='utf8')
    collected = {}
    for suffix in ('.json', '-entry.json'):
        path = args.remote + '/' + args.run_id + suffix
        if client.tool('canmv_stat_file', {'path': path}).get('exists') is True:
            value = client.tool('canmv_read_file', {'path': path, 'encoding': 'utf8'})['content']
            collected[suffix] = json.loads(value)
    save(args.out, collected)
    print(json.dumps(collected), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'run', 'fetch'))
    parser.add_argument('--remote', type=remote_root, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--setting', choices=('cnn1', 'tcn4'))
    parser.add_argument('--current-source', action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--frames', type=int, default=10)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve previous evidence: ' + str(args.out))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    client = CanMvClient()
    if client.tool('canmv_board_info').get('connected') is not True:
        raise RuntimeError('K230 connection unconfirmed')
    globals()[args.action](args, client)


if __name__ == '__main__':
    main()
