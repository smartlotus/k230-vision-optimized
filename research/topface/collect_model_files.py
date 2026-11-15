"""Collect bounded user deliverables after their original cloud archive is accepted."""
import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path

from cloud_store import atomic_json, sha
from collect_compact import download
from prepare_full_shard import FOLDER_ID


def collect(result, publication, token_file, output, receipt):
    saved = json.loads(result.read_text(encoding='utf8'))
    durable = json.loads(publication.read_text(encoding='utf8'))
    if (saved['status'] != 'TRAINED_EXPORT_COMPILE_AND_SIMULATOR_COMPLETE' or
            durable['status'] != 'DRIVE_EXPORT_FULL_BYTE_SHA_COMMITTED' or
            durable['secret_cleanup'] != 'CONFIRMED_DELETED' or
            not durable['independent_metadata_match'] or durable['run_id'] != saved['run_id'] or
            durable['folder_id'] != FOLDER_ID or durable['uploaded']['provider_readback'] != 'FULL_BYTE_SHA256_PASS' or
            any(durable['uploaded'][key] != saved['archive'][key] for key in ('bytes', 'sha256'))):
        raise ValueError('Deliverables require the accepted original conversion archive')
    rows = saved['model_files']
    allowed = {'topface.onnx', 'topface.kmodel', 'identity.json', 'topface-board-bundle.tar.gz'}
    required = {'topface.onnx', 'topface.kmodel', 'identity.json'}
    names = [Path(row['path']).name for row in rows]
    if (len(names) != len(set(names)) or not required <= set(names) <= allowed or
            sum(row['bytes'] for row in rows) > 64*1024**2):
        raise ValueError('Unexpected deliverable inventory or total size')
    for row in rows:
        if row['path'] != saved['run_id']+'-models/'+Path(row['path']).name or not 0 < row['bytes'] <= 32*1024**2:
            raise ValueError('Unexpected deliverable path or size')
    os.environ['KAGGLE_API_TOKEN'] = token_file.read_text(encoding='utf8').strip()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
    output.mkdir(parents=True, exist_ok=True)
    source = durable['prep']
    with api.build_kaggle_client() as client:
        for row in rows:
            target = output/Path(row['path']).name
            if not target.exists():
                raw = download(client, source['ref'], source['version'], row['path'], row['bytes'])
                temporary = target.with_suffix(target.suffix+'.part')
                temporary.write_bytes(raw)
                if temporary.stat().st_size != row['bytes'] or sha(temporary) != row['sha256']:
                    raise ValueError('Saved deliverable bytes differ from the conversion result')
                temporary.replace(target)
            if target.stat().st_size != row['bytes'] or sha(target) != row['sha256']:
                raise ValueError('Existing local deliverable differs; preserve and investigate')
    identity = json.loads((output/'identity.json').read_text(encoding='utf8'))
    if (identity != saved['identity'] or sha(output/'topface.onnx') != identity['onnx_sha256'] or
            sha(output/'topface.kmodel') != identity['kmodel_sha256']):
        raise ValueError('Collected model identity differs')
    if saved['export']['onnx_sha256'] != identity['onnx_sha256']:
        raise ValueError('Export report differs from the collected model')
    atomic_json(output/'RESULT.json', saved['export'])
    value = {'status':'BOUNDED_MODEL_DELIVERABLES_SHA256_PASS', 'run_id':saved['run_id'],
             'observed_utc':datetime.now(timezone.utc).isoformat(), 'source':source,
             'source_result_sha256':sha(result), 'publication_receipt_sha256':sha(publication),
             'local_directory':str(output.resolve()), 'files':rows,
             'purpose':'User-requested deployable models; complete process archive remains in Drive',
             'raw_archive_downloaded_to_host':False, 'board_test':'NOT_RUN'}
    atomic_json(receipt, value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('result', 'publication', 'token-file', 'output', 'receipt'):
        parser.add_argument('--'+key, type=Path, required=True)
    print(json.dumps(collect(**vars(parser.parse_args())), indent=2))
