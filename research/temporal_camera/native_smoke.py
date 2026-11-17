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
RUNTIME_CHUNK = 8192
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
    if not re.fullmatch(r'/(?:sdcard|tmp)/k230_temporal_v1_[a-z0-9_]+', value):
        raise ValueError('A dedicated CameraXY directory is required')
    return value


def execute_marked(client, script, marker, timeout=45):
    idle(client)
    client.tool('canmv_run_script', {'script': script})
    deadline = time.monotonic() + timeout
    while client.tool('canmv_script_running').get('running') is not False:
        if time.monotonic() >= deadline:
            raise TimeoutError('Board operation unknown; do not repeat')
        time.sleep(.5)
    terminal = client.tool('canmv_terminal_output', {'clear': False})['text']
    lines = [line[len(marker):] for line in terminal.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError('Board operation marker unconfirmed: ' + terminal[-1200:])
    return json.loads(lines[0]), terminal


def append_script(part, target, offset, block, marker):
    return ("import os,json,hashlib,binascii\np=" + repr(part) + "\nt=" + repr(target) +
            "\noffset=" + str(offset) + "\n"
            "try:\n size=os.stat(t)[6]\nexcept OSError:\n size=0\n"
            "if size!=offset:raise ValueError('Unexpected assembly offset')\n"
            "with open(p,'rb') as src:\n b=src.read()\n"
            "if len(b)!=" + str(len(block)) + " or binascii.hexlify(hashlib.sha256(b).digest()).decode()!=" +
            repr(sha(block)) + ":raise ValueError('Part identity differs')\n"
            "with open(t,'wb' if offset==0 else 'ab') as dst:dst.write(b)\n"
            "if os.stat(t)[6]!=offset+len(b):raise ValueError('Stored append length differs')\n"
            "os.remove(p)\nprint(" + repr(marker) + "+json.dumps({'bytes':os.stat(t)[6]}))\n")


def runtime_append_script(target, offset, block, marker):
    # MicroPython applications use /tmp directly; the IDE file browser exposes
    # only sdcard/data/udisk for mutations. Never use this for system folders.
    if not re.fullmatch(r'/tmp/k230_temporal_v1_[a-z0-9_]+/[a-zA-Z0-9_.]+', target):
        raise ValueError('Runtime upload is limited to dedicated /tmp test files')
    return ("import os,json,hashlib\nimport ubinascii as binascii\nt=" + repr(target) + "\noffset=" + str(offset) + "\n"
            "try:\n size=os.stat(t)[6]\nexcept OSError:\n size=0\n"
            "if size!=offset:raise ValueError('Unexpected runtime upload offset')\n"
            "b=binascii.a2b_base64(" + repr(base64.b64encode(block)) + ")\n"
            "if len(b)!=" + str(len(block)) + " or binascii.hexlify(hashlib.sha256(b).digest()).decode()!=" +
            repr(sha(block)) + ":raise ValueError('Runtime upload block differs')\n"
            "with open(t,'wb' if offset==0 else 'ab') as f:f.write(b)\n"
            "if os.stat(t)[6]!=offset+len(b):raise ValueError('Runtime stored size differs')\n"
            "print(" + repr(marker) + "+json.dumps({'bytes':os.stat(t)[6]}))\n")


def install(args, client):
    if args.runtime_upload and not args.current_source:
        raise ValueError('Temporary runtime installation requires --current-source')
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
    if args.runtime_upload:
        if not args.remote.startswith('/tmp/k230_temporal_v1_'):
            raise ValueError('Runtime upload requires a dedicated /tmp test root')
        marker = 'TEMPORAL_MKDIR_' + uuid.uuid4().hex + ':'
        script = ("import os,json\np=" + repr(args.remote) + "\n"
                  "try:\n os.stat(p)\nexcept OSError:\n pass\nelse:\n raise ValueError('Existing directory preserved')\n"
                  "os.mkdir(p)\nprint(" + repr(marker) + "+json.dumps({'success':True}))\n")
        created, _ = execute_marked(client, script, marker)
        receipt['transport'] = 'MICROPYTHON_TMP_FILE_IO'
    else:
        created = client.tool('canmv_mkdir', {'path': args.remote})
    receipt['mkdir_result'] = created
    save(args.out, receipt)
    if created.get('success') is not True:
        receipt['status'] = 'DIRECTORY_CREATION_FAILED'
        save(args.out, receipt)
        raise RuntimeError('Directory creation unconfirmed: ' + str(created))
    final_files = {}
    for (name, data), job in zip(files.items(), jobs):
        if args.runtime_upload:
            for offset in range(0, len(data), RUNTIME_CHUNK):
                block = data[offset:offset + RUNTIME_CHUNK]
                marker = 'TEMPORAL_RAM_' + uuid.uuid4().hex + ':'
                value, _ = execute_marked(client, runtime_append_script(job['target'], offset, block, marker), marker)
                if value['bytes'] != offset + len(block):
                    raise RuntimeError('Runtime upload size differs')
                receipt['verified_parts'].append(dict(path=job['target'], offset=offset,
                                                      bytes=len(block), sha256=sha(block)))
                save(args.out, receipt)
                if offset // RUNTIME_CHUNK % 32 == 0:
                    print(json.dumps({'file': name, 'verified_bytes': offset + len(block),
                                      'total_bytes': len(data)}), flush=True)
        else:
            install_file_parts(client, data, job, receipt, args.out)
        marker = 'TEMPORAL_FILE_' + uuid.uuid4().hex + ':'
        script = ("import os,json,hashlib,binascii\nt=" + repr(job['target']) + "\nh=hashlib.sha256()\n"
                  "with open(t,'rb') as f:\n while True:\n  b=f.read(4096)\n  if not b:break\n  h.update(b)\n"
                  "print(" + repr(marker) + "+json.dumps({'bytes':os.stat(t)[6],"
                  "'sha256':binascii.hexlify(h.digest()).decode()}))\n")
        value, terminal = execute_marked(client, script, marker)
        if value != {'bytes': job['bytes'], 'sha256': job['sha256']}:
            raise RuntimeError('Final stored file identity differs')
        final_files[job['target']] = value
        receipt['stored_files'] = final_files
        save(args.out, receipt)
        args.out.with_suffix('.log').write_text(terminal, encoding='utf8')
        print('Verified stored file ' + name, flush=True)
    expected = {job['target']: {'bytes': job['bytes'], 'sha256': job['sha256']} for job in jobs}
    if final_files != expected:
        raise RuntimeError('Final stored-file identity unconfirmed')
    receipt.update(status='VERIFIED_BOARD_FILES_ONLY', stored_files=expected)
    save(args.out, receipt)
    print(json.dumps({'status': receipt['status'], 'remote': args.remote, 'files': len(files)}), flush=True)


def install_file_parts(client, data, job, receipt, output):
        for index, part in enumerate(job['parts']):
            block = data[index * CHUNK:(index + 1) * CHUNK]
            if client.tool('canmv_write_file', {'path': part, 'encoding': 'base64',
                           'content': base64.b64encode(block).decode()}).get('success') is not True:
                raise RuntimeError('Part write unconfirmed; no retry')
            readback = client.tool('canmv_read_file', {'path': part, 'encoding': 'base64'})
            if base64.b64decode(readback['content'], validate=True) != block:
                raise RuntimeError('Part readback differs')
            receipt['verified_parts'].append({'path': part, 'bytes': len(block), 'sha256': sha(block)})
            save(output, receipt)
            marker = 'TEMPORAL_PART_' + uuid.uuid4().hex + ':'
            script = append_script(part, job['target'], index * CHUNK, block, marker)
            value, terminal = execute_marked(client, script, marker)
            if value['bytes'] != index * CHUNK + len(block):
                raise RuntimeError('Append readback differs')
            receipt['last_assembled_bytes'] = value['bytes']
            save(output, receipt)


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


def patch(args, client):
    if not args.remote.startswith('/tmp/k230_temporal_v1_') or args.module not in BOARD_SOURCES:
        raise ValueError('Patch only an owned temporary CameraXY source file')
    if not re.fullmatch(r'[a-z0-9_]+', args.revision):
        raise ValueError('Use a unique ASCII backup revision')
    data = (Path(__file__).parent / args.module).read_bytes()
    target = args.remote + '/' + args.module
    staging = target + '.new'
    backup = target + '.' + args.revision
    idle(client)
    for path in (staging, backup):
        if client.tool('canmv_stat_file', {'path': path}).get('exists') is not False:
            raise RuntimeError('Existing patch artifact preserved')
    for offset in range(0, len(data), RUNTIME_CHUNK):
        marker = 'TEMPORAL_PATCH_' + uuid.uuid4().hex + ':'
        execute_marked(client, runtime_append_script(staging, offset, data[offset:offset + RUNTIME_CHUNK], marker), marker)
    marker = 'TEMPORAL_SWITCH_' + uuid.uuid4().hex + ':'
    script = ("import os,json,hashlib,ubinascii\np=" + repr(staging) + "\nt=" + repr(target) +
              "\nb=" + repr(backup) + "\nwith open(p,'rb') as f:raw=f.read()\n"
              "h=ubinascii.hexlify(hashlib.sha256(raw).digest()).decode()\n"
              "if len(raw)!=" + str(len(data)) + " or h!=" + repr(sha(data)) + ":raise ValueError('Patch differs')\n"
              "with open(t,'rb') as f:old=f.read()\nold_h=ubinascii.hexlify(hashlib.sha256(old).digest()).decode()\n"
              "os.rename(t,b)\nos.rename(p,t)\nprint(" + repr(marker) +
              "+json.dumps({'bytes':len(raw),'sha256':h,'previous_sha256':old_h,'backup':b}))\n")
    result, terminal = execute_marked(client, script, marker)
    save(args.out, result)
    args.out.with_suffix('.log').write_text(terminal, encoding='utf8')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'run', 'fetch', 'patch'))
    parser.add_argument('--remote', type=remote_root, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--setting', choices=('cnn1', 'tcn4'))
    parser.add_argument('--current-source', action='store_true')
    parser.add_argument('--runtime-upload', action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--module', choices=BOARD_SOURCES)
    parser.add_argument('--revision')
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
