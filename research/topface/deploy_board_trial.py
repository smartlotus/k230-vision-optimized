"""Verify and install one immutable TopFace board package in a new SD directory."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
import tomllib
import urllib.parse
import urllib.request
import uuid

from cloud_store import atomic_json, sha
from package_formal_delivery import BOARD_NAMES, verify_board


ROOT = Path(__file__).resolve().parents[2]
CHUNK_BYTES = 262144


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def verified_assets(delivery):
    delivery = delivery.resolve()
    checksum_path = delivery / 'SHA256SUMS.json'
    inventory = read(checksum_path)
    if (inventory['status'] != 'FORMAL_BOARD_TEST_PACKAGE_VERIFIED' or
            inventory['native_board_test'] != 'NOT_RUN' or not 1 <= len(inventory['files']) <= 64):
        raise ValueError('Expected the verified, unmodified formal board-test package')
    seen = set()
    for row in inventory['files']:
        name = row['path']
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or '..' in relative.parts or '\\' in name or
                name != relative.as_posix() or ':' in name or name in seen):
            raise ValueError('Invalid or repeated delivery member')
        seen.add(name)
        path = delivery.joinpath(*relative.parts)
        if (not path.resolve().is_relative_to(delivery) or
                any(parent.is_symlink() for parent in (path, *path.parents) if parent != delivery)):
            raise ValueError('Delivery links and escaped paths are not accepted')
        if not path.is_file() or not 0 <= row['bytes'] <= 64 * 1024**2:
            raise ValueError('Delivery file is missing or outside its size budget')
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Delivery file differs from SHA256SUMS.json')
    required = {'board/' + name for name in BOARD_NAMES}
    if not required.issubset(seen):
        raise ValueError('The delivery inventory omits board files')
    verify_board(delivery / 'board', delivery / 'models')
    identity = read(delivery / 'models/identity.json')
    if identity['checkpoint_sha256'] != inventory['checkpoint_sha256']:
        raise ValueError('The formal checkpoint identity differs')
    assets = []
    for name in sorted(BOARD_NAMES):
        path = delivery / 'board' / name
        if path.stat().st_size > 32 * 1024**2:
            raise ValueError('Board asset exceeds its transfer budget')
        raw = path.read_bytes()
        expected = next(row for row in inventory['files'] if row['path'] == 'board/' + name)
        if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError('Board asset changed during preparation')
        assets.append((name, raw))
    return inventory, sha(checksum_path), assets


def remote_jobs(remote, assets):
    if not re.fullmatch(r'/sdcard/tf_[A-Za-z0-9_]{1,16}', remote):
        raise ValueError('Use a new short /sdcard/tf_NAME trial directory')
    jobs = []
    for name, raw in assets:
        parts = [remote + '/' + name + '.p' + str(index)
                 for index, _ in enumerate(range(0, len(raw), CHUNK_BYTES))]
        paths = [remote + '/' + name, *parts]
        if any(len(path.encode('ascii')) >= 64 for path in paths):
            raise ValueError('Native file protocol paths must be shorter than64 bytes')
        jobs.append({'target': paths[0], 'parts': parts, 'bytes': len(raw),
                     'sha256': hashlib.sha256(raw).hexdigest()})
    return jobs


def client_from_config(config_path):
    # Reuse the project's single-send RPC implementation; bind its configuration
    # explicitly because the desktop instance and legacy CanMV host differ.
    sys.path.insert(0, str(ROOT / 'k230_transport_mvp'))
    from tools.deploy_native_trial import CanMvClient
    config = tomllib.loads(config_path.read_text(encoding='utf8'))
    entry = config['mcp_servers']['canmv-k230']
    if entry.get('enabled') is False:
        raise ValueError('CanMV MCP is disabled')
    parsed = urllib.parse.urlsplit(entry['url'])
    try:
        loopback = parsed.hostname == 'localhost' or ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = False
    if (not loopback or parsed.scheme != 'http' or parsed.path != '/mcp' or
            parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError('An explicit official loopback CanMV MCP is required')
    headers = dict(entry.get('http_headers', {}))
    for key, variable in entry.get('env_http_headers', {}).items():
        if os.getenv(variable):
            headers[key] = os.environ[variable]
    if entry.get('bearer_token_env_var'):
        headers['Authorization'] = 'Bearer ' + os.environ[entry['bearer_token_env_var']]
    if not any(key.lower() == 'authorization' and value for key, value in headers.items()):
        raise ValueError('CanMV MCP authentication is missing')
    client = CanMvClient.__new__(CanMvClient)
    client.url, client.headers, client.sequence = entry['url'], headers, 0
    client.headers.update({'Content-Type': 'application/json',
                           'Accept': 'application/json, text/event-stream'})
    client.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    client.request('initialize', {'protocolVersion': '2024-11-05', 'capabilities': {},
                   'clientInfo': {'name': 'topface-isolated-install', 'version': '1'}})
    return client


def join_script(jobs, marker):
    return ("import os,json,hashlib,binascii\njobs=" + json.dumps(jobs) + "\nverified={}\n"
            "for job in jobs:\n"
            " try:\n  os.stat(job['target'])\n except OSError:\n  pass\n"
            " else:\n  raise ValueError('Existing target preserved')\n"
            " with open(job['target'],'wb') as dst:\n"
            "  for part in job['parts']:\n"
            "   with open(part,'rb') as src:\n"
            "    while True:\n     block=src.read(4096)\n"
            "     if not block:break\n     dst.write(block)\n"
            " h=hashlib.sha256()\n size=0\n"
            " with open(job['target'],'rb') as src:\n"
            "  while True:\n   block=src.read(4096)\n"
            "   if not block:break\n   h.update(block)\n   size+=len(block)\n"
            " value=binascii.hexlify(h.digest()).decode()\n"
            " if size!=job['bytes'] or value!=job['sha256']:raise ValueError('Stored file differs')\n"
            " verified[job['target']]={'bytes':size,'sha256':value}\n"
            "print(" + repr(marker) + "+json.dumps(verified))\n")


def deploy(delivery, remote, output, *, install=False, mcp_config=None, client=None):
    inventory, checksum, assets = verified_assets(delivery)
    jobs = remote_jobs(remote, assets)
    output = output.resolve()
    if output.exists() or output.is_relative_to(delivery.resolve()):
        raise ValueError('Use a new receipt path outside the immutable delivery')
    if client is None:
        if mcp_config is None:
            raise ValueError('Choose the configured CanMV host with --mcp-config')
        client = client_from_config(mcp_config)
    if client.tool('canmv_board_info').get('connected') is not True:
        raise RuntimeError('Physical board is not confirmed connected')
    if client.tool('canmv_script_running').get('running') is not False:
        raise RuntimeError('Board is busy; existing process preserved')
    if client.tool('canmv_stat_file', {'path': remote}).get('exists') is not False:
        raise RuntimeError('The trial directory already exists or its absence is unknown')
    receipt = {'schema': 'topface-isolated-board-install-v1', 'status': 'PREFLIGHT_PASS',
               'observed_utc': datetime.now(timezone.utc).isoformat(),
               'run_id': inventory['run_id'], 'remote_root': remote,
               'delivery_inventory_sha256': checksum, 'checkpoint_sha256': inventory['checkpoint_sha256'],
               'files': jobs, 'verified_parts': [], 'automatic_retry': False,
               'inference_started': False, 'native_accuracy': 'NOT_RUN', 'native_fps': 'NOT_RUN'}
    atomic_json(output, receipt)
    if not install:
        return receipt

    def stage(name, **details):
        receipt.update(status=name, observed_utc=datetime.now(timezone.utc).isoformat(), **details)
        atomic_json(output, receipt)

    try:
        # Readiness and destination absence were confirmed before any SD writes.
        stage('CREATING_DIRECTORY')
        if client.tool('canmv_mkdir', {'path': remote}).get('success') is not True:
            raise RuntimeError('Directory creation outcome is unconfirmed')
        for (_, raw), job in zip(assets, jobs):
            for index, path in enumerate(job['parts']):
                block = raw[index * CHUNK_BYTES:(index + 1) * CHUNK_BYTES]
                expected = hashlib.sha256(block).hexdigest()
                stage('UPLOADING_PART', current_part=path)
                written = client.tool('canmv_write_file', {'path': path, 'encoding': 'base64',
                                      'content': base64.b64encode(block).decode('ascii')})
                if written.get('success') is not True:
                    raise RuntimeError('Part write outcome is unconfirmed')
                readback = client.tool('canmv_read_file', {'path': path, 'encoding': 'base64'})
                if base64.b64decode(readback['content'], validate=True) != block:
                    raise RuntimeError('Board part readback differs')
                receipt['verified_parts'].append({'path': path, 'bytes': len(block), 'sha256': expected})
                stage('PART_VERIFIED')
        if client.tool('canmv_script_running').get('running') is not False:
            raise RuntimeError('Another board script started; do not interrupt it')
        marker = 'TOPFACE_FILES_' + uuid.uuid4().hex + ':'
        stage('JOINING_AND_READING_STORED_FILES', terminal_marker=marker)
        result = client.tool('canmv_run_script', {'script': join_script(jobs, marker)})
        if result.get('success') is False:
            raise RuntimeError('File assembly was refused')
        deadline = time.monotonic() + 30
        while client.tool('canmv_script_running').get('running') is not False:
            if time.monotonic() >= deadline:
                raise TimeoutError('File assembly outcome is unknown; do not restart it')
            time.sleep(.25)
        terminal = client.tool('canmv_terminal_output', {'clear': False})['text']
        lines = [line[len(marker):] for line in terminal.splitlines() if line.startswith(marker)]
        expected = {job['target']: {'bytes': job['bytes'], 'sha256': job['sha256']} for job in jobs}
        if len(lines) != 1 or json.loads(lines[0]) != expected:
            raise RuntimeError('Final stored-file readback is unconfirmed')
        stage('VERIFIED_BOARD_FILES_ONLY', stored_files=expected)
        return receipt
    except Exception as error:
        stage('NEEDS_RECONCILIATION', error_type=type(error).__name__)
        raise RuntimeError('Inspect the receipt and original trial directory; no automatic retry') from None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', type=Path, required=True)
    parser.add_argument('--remote', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mcp-config', type=Path, required=True)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    result = deploy(**vars(args))
    print(json.dumps({'status': result['status'], 'remote_root': result['remote_root'],
                      'files': len(result['files']), 'inference_started': False}))
