"""Transfer one exact Kaggle archive from a hosted runner to Google Drive."""

import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


CHUNK_BYTES = 8 * 1024**2
MAX_ARCHIVE_BYTES = 1024**3
MAX_JSON_BYTES = 48 * 1024
WALL_SECONDS = 600
TRANSIENT = {429, 500, 502, 503, 504}
REDIRECT = {301, 302, 303, 307, 308}
DRIVE_HOST = 'www.googleapis.com'
DRIVE_ID = re.compile(r'[A-Za-z0-9_-]{10,128}')
SHA256 = re.compile(r'[a-f0-9]{64}')


class TransferError(RuntimeError):
    """A sanitized failure safe to report in the hosted-runner log."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def require(condition, reason):
    if not condition:
        raise TransferError(reason)


def endpoint(url):
    require(isinstance(url, str) and 0 < len(url) <= 16384 and
            not any(ord(char) <= 32 or ord(char) >= 127 for char in url), 'invalid_url')
    parsed = urlsplit(url)
    require(parsed.scheme == 'https' and parsed.hostname and
            parsed.netloc == parsed.hostname and not parsed.fragment, 'invalid_endpoint')
    return parsed


def source_endpoint(url):
    parsed = endpoint(url)
    require(parsed.hostname in {'www.kaggle.com', 'www.kaggleusercontent.com', 'storage.googleapis.com'} or
            parsed.hostname.endswith('.storage.googleapis.com'), 'unexpected_kaggle_output_host')


def require_access(spec):
    require(spec['drive_expiry_epoch'] > time.time() + 30, 'drive_access_expired_or_expiring')


def parse_spec(raw):
    require(len(raw.encode()) <= MAX_JSON_BYTES, 'transfer_spec_too_large')
    spec = json.loads(raw)
    require(type(spec) is dict and set(spec) == {
        'source_url', 'upload_url', 'drive_access_token', 'drive_expiry_epoch',
        'archive', 'folder_id', 'drive_file_id', 'run_id', 'prep',
    }, 'invalid_transfer_spec')
    run_id = spec['run_id']
    require(isinstance(run_id,str) and re.fullmatch(r'k230-[a-z0-9-]{8,100}',run_id), 'invalid_run_id')
    require(spec['folder_id']=='1MwltNmaI24HjvcxnG1w608qui42RAPKf', 'wrong_k230_folder')
    archive = spec['archive']
    require(type(archive) is dict and set(archive) in (
        {'name', 'bytes', 'sha256'}, {'name', 'bytes', 'sha256', 'files'},
    ), 'invalid_archive')
    digest = archive['sha256']
    require(isinstance(digest, str) and SHA256.fullmatch(digest), 'invalid_archive_sha256')
    require(archive['name'] == f'{run_id}-{digest[:16]}.tar.gz', 'invalid_archive_name')
    size = archive['bytes']
    require(type(size) is int and 0 < size <= MAX_ARCHIVE_BYTES, 'invalid_archive_size')
    for key in ('folder_id', 'drive_file_id'):
        require(isinstance(spec[key], str) and DRIVE_ID.fullmatch(spec[key]), 'invalid_drive_id')
    require(spec['folder_id'] != spec['drive_file_id'], 'invalid_drive_destination')
    token = spec['drive_access_token']
    require(isinstance(token, str) and re.fullmatch(r'[A-Za-z0-9._~+/=-]{20,4096}', token),
            'invalid_drive_access_token')
    expiry = spec['drive_expiry_epoch']
    require(type(expiry) in (int, float) and math.isfinite(expiry), 'invalid_drive_access_expiry')
    require_access(spec)
    prep = spec['prep']
    require(type(prep) is dict and set(prep) == {
        'ref', 'kernel_id', 'version', 'provider_status', 'source_sha256',
    }, 'invalid_prep_identity')
    require(isinstance(prep['ref'], str) and re.fullmatch(r'lotus30/[a-z0-9][a-z0-9-]{0,127}', prep['ref']),
            'invalid_prep_ref')
    require(type(prep['kernel_id']) is int and prep['kernel_id'] > 0 and
            type(prep['version']) is int and prep['version'] > 0 and prep['provider_status'] == 'COMPLETE',
            'invalid_prep_version')
    require(isinstance(prep['source_sha256'], str) and SHA256.fullmatch(prep['source_sha256']),
            'invalid_prep_source_sha256')
    source_endpoint(spec['source_url'])
    upload = endpoint(spec['upload_url'])
    require(upload.hostname == DRIVE_HOST and upload.path == '/upload/drive/v3/files' and
            len(parse_qs(upload.query).get('upload_id', [])) == 1, 'invalid_drive_upload_session')
    return spec


def request(opener, url, *, method='GET', headers=None, data=None):
    request_headers = {'Accept-Encoding': 'identity'}
    request_headers.update(headers or {})
    try:
        return opener.open(Request(url, method=method, headers=request_headers, data=data), timeout=30)
    except HTTPError as error:
        return error


def source_response(opener, url):
    for _ in range(3):
        source_endpoint(url)
        response = request(opener, url)
        if response.status not in REDIRECT:
            return response
        with response:
            location = response.headers.get('Location')
        require(location, 'kaggle_redirect_missing_location')
        url = urljoin(url, location)
    raise TransferError('kaggle_redirect_limit')


def verify_body(response, archive, target=None):
    require(response.status == 200, f'download_http_{response.status}')
    expected = archive['bytes']
    declared = response.headers.get('Content-Length')
    if declared is not None:
        require(declared == str(expected), 'download_length_mismatch')
    digest = hashlib.sha256()
    count = 0
    while block := response.read(CHUNK_BYTES):
        count += len(block)
        require(count <= expected, 'download_exceeded_bound')
        digest.update(block)
        if target is not None:
            target.write(block)
    require(count == expected and digest.hexdigest() == archive['sha256'], 'download_byte_proof_mismatch')


def read_json(response):
    body = response.read(MAX_JSON_BYTES + 1)
    require(len(body) <= MAX_JSON_BYTES, 'provider_json_exceeded_bound')
    value = json.loads(body)
    require(type(value) is dict, 'invalid_provider_json')
    return value


def upload_archive(opener, spec, path):
    """Recover within the same session after an ambiguous or partial PUT."""
    size = spec['archive']['bytes']
    offset = recoveries = 0
    query = False
    with path.open('rb') as stream:
        while offset < size or query:
            require_access(spec)
            stream.seek(offset)
            body = b'' if query else stream.read(min(CHUNK_BYTES, size - offset))
            require(query or body, 'local_archive_truncated')
            end = offset + len(body)
            content_range = f'bytes */{size}' if query else f'bytes {offset}-{end - 1}/{size}'
            headers = {
                'Content-Type': 'application/gzip',
                'Content-Length': str(len(body)),
                'Content-Range': content_range,
            }
            try:
                with request(opener, spec['upload_url'], method='PUT', headers=headers, data=body) as response:
                    status = response.status
                    if status in (200, 201):
                        require(query or end == size, 'drive_upload_completed_early')
                        require(read_json(response).get('id') == spec['drive_file_id'],
                                'drive_uploaded_id_mismatch')
                        return
                    if status == 308:
                        raw_range = response.headers.get('Range')
                        match = re.fullmatch(r'bytes=0-(\d+)', raw_range or '')
                        require(not raw_range or match, 'invalid_drive_upload_range')
                        confirmed = int(match[1]) + 1 if match else 0
                        require(offset <= confirmed <= (size if query else end) and confirmed < size,
                                'drive_upload_range_mismatch')
                        if query or confirmed > offset:
                            offset, query = confirmed, False
                            continue
                    else:
                        require(status in TRANSIENT, f'drive_upload_http_{status}')
            except (URLError, TimeoutError, ConnectionError, http.client.HTTPException):
                pass
            require(recoveries < 3, 'drive_upload_needs_reconciliation')
            recoveries += 1
            time.sleep(2 ** (recoveries - 1))
            query = True
    raise TransferError('drive_upload_missing_completion')


def readback(opener, spec):
    archive = spec['archive']
    base = f'https://{DRIVE_HOST}/drive/v3/files/{spec["drive_file_id"]}'
    headers = {'Authorization': 'Bearer ' + spec['drive_access_token']}
    require_access(spec)
    fields = 'id,name,parents,size,sha256Checksum,trashed'
    with request(opener, f'{base}?fields={fields}', headers=headers) as response:
        require(response.status == 200, f'drive_metadata_http_{response.status}')
        meta = read_json(response)
    require(meta.get('id') == spec['drive_file_id'] and meta.get('name') == archive['name'] and
            meta.get('parents') == [spec['folder_id']] and meta.get('size') == str(archive['bytes']) and
            meta.get('trashed') is False and meta.get('sha256Checksum', archive['sha256']) == archive['sha256'],
            'drive_metadata_mismatch')
    require_access(spec)
    with request(opener, f'{base}?alt=media', headers=headers) as response:
        verify_body(response, archive)


def transfer(spec, opener):
    archive = spec['archive']
    with tempfile.TemporaryDirectory(prefix='k230-transfer-') as temporary:
        path = Path(temporary) / 'archive.tar.gz'
        with source_response(opener, spec['source_url']) as response, path.open('xb') as target:
            verify_body(response, archive, target)
        upload_archive(opener, spec, path)
        readback(opener, spec)
    uploaded = {key: archive[key] for key in ('name', 'bytes', 'sha256')}
    uploaded.update(file_id=spec['drive_file_id'], provider_readback='FULL_BYTE_SHA256_PASS')
    return dict(run_id=spec['run_id'], prep=spec['prep'], folder_id=spec['folder_id'],
                uploaded=uploaded, status='DRIVE_EXPORT_FULL_BYTE_SHA_COMMITTED',
                automatic_replay=False, provider_mutations=1)


def deadline(*_):
    raise TransferError('transfer_deadline')


def main():
    try:
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(WALL_SECONDS)
        spec = parse_spec(os.environ.pop('TOPFACE_TRANSFER_SPEC', ''))
        result = transfer(spec, build_opener(NoRedirect()))
    except TransferError as error:
        print(f'TOPFACE_TRANSFER_FAILED {error}', file=sys.stderr, flush=True)
        return 1
    except Exception:
        # Provider exceptions can contain signed URLs or request headers.
        print('TOPFACE_TRANSFER_FAILED invalid_input_or_transport_failure', file=sys.stderr, flush=True)
        return 1
    finally:
        signal.alarm(0)
    print('TOPFACE_TRANSFER_RECEIPT ' + json.dumps(result, separators=(',', ':')), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
