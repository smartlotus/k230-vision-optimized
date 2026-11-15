"""Evaluate the released formal artifacts on the approved local test fixtures."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from cloud_store import atomic_json, sha
from evaluate_real_r4 import MANIFEST_SHA


HERE = Path(__file__).resolve().parent
REAL_STATUS = 'HISTORICAL_REAL_TEST_EVALUATED'
SEQUENCE_STATUS = 'SYNTHETIC_REDETECTION_MEASURED'
BACKENDS = {'onnx': ('ONNX_RUNTIME', 'onnx_sha256'),
            'simulator': ('NNCASE_SIMULATOR', 'kmodel_sha256')}


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def evaluate(conversion, dataset, sequence_acceptance, output, sequence_candidate=None):
    model_receipt = conversion / 'MODEL_FILES_RECEIPT.json'
    report_receipt = conversion / 'REPORTS_RECEIPT.json'
    models, reports = read(model_receipt), read(report_receipt)
    if (models['status'] != 'BOUNDED_MODEL_DELIVERABLES_SHA256_PASS' or
            reports['status'] != 'BOUNDED_COMPLETE_MODEL_REPORTS_SHA256_PASS' or
            any(models[key] != reports[key] for key in
                ('run_id', 'source', 'source_result_sha256', 'publication_receipt_sha256'))):
        raise ValueError('The saved models and synthetic reports have different publication identities')

    model_dir = Path(models['local_directory']).resolve()
    identity = read(model_dir / 'identity.json')
    exported = read(model_dir / 'RESULT.json')
    config = read(conversion / 'MODEL_ARTIFACT_CONFIG.json')
    saved_result = conversion / f"compact-v{models['source']['version']}" / f"{models['run_id']}-RESULT.json"
    parent = config['parent']
    if (config['phase'] != 'formal_conversion' or config['run_id'] != models['run_id'] or
            parent['kind'] != 'formal_model' or parent['completed_epochs'] < 100 or
            exported['experiment'] != 'formal_full' or exported['status'] != 'PYTORCH_ONNX_PARITY_PASS' or
            parent['checkpoint_sha256'] != identity['checkpoint_sha256']):
        raise ValueError('Local evaluation requires the completed formal model')
    if sha(saved_result) != models['source_result_sha256']:
        raise ValueError('Saved conversion result differs from the publication receipts')
    saved = read(saved_result)
    if saved['export'] != exported or saved['identity'] != identity or saved['model_files'] != models['files']:
        raise ValueError('Local model inventory differs from the exact saved conversion')
    for row in models['files']:
        path = model_dir / Path(row['path']).name
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError(f'Collected formal artifact changed: {path.name}')

    accepted = read(sequence_acceptance)
    sequence_sha = sha(dataset / 'AUDIT_V11.json')
    if (accepted['status'] != 'SYNTHETIC_SEQUENCE_RENDER_QA_AND_DRIVE_ACCEPTED' or
            accepted['dataset_index_sha256'] != sequence_sha):
        raise ValueError('Local sequence index differs from the accepted test fixture')

    output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'HOST_EVALUATION_RECEIPT.json'
    lock_path = output / '.evaluation.lock'
    with lock_path.open('x', encoding='utf8'):
        pass
    try:
        results = {}
        reused = {}
        for backend, (backend_name, digest_key) in BACKENDS.items():
            for kind, script, extra in (
                ('historical_real', 'evaluate_real_r4.py', ['--phase', 'formal_historical_test']),
                ('synthetic_sequence', 'evaluate_redetection.py',
                 ['--phase', 'formal', '--dataset', str(dataset.resolve())]),
            ):
                name = f'{kind}_{backend}'
                target = output / f'{name}.json'
                if name == 'synthetic_sequence_simulator' and not target.exists() and sequence_candidate is not None:
                    candidate = read(sequence_candidate)
                    if (candidate['status'] == SEQUENCE_STATUS and candidate['phase'] == 'formal' and
                            candidate['backend'] == backend_name and candidate['model_sha256'] == identity[digest_key] and
                            candidate['reference_onnx_sha256'] == identity['onnx_sha256'] and
                            candidate['dataset_index_sha256'] == sequence_sha and
                            candidate['sequence_count'] == accepted['sequences'] and
                            candidate['denominators']['frames'] == accepted['images']):
                        atomic_json(target, candidate)
                        reused[name] = {'source_sha256': sha(sequence_candidate),
                                        'kmodel_sha256': identity[digest_key],
                                        'dataset_index_sha256': sequence_sha}
                if not target.exists():
                    print(f'Running {name}', flush=True)
                    with (output / f'{name}.log').open('ab') as log:
                        subprocess.run([sys.executable, str(HERE / script), '--export-dir', str(model_dir),
                                        '--output', str(target.resolve()), '--backend', backend, *extra],
                                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=8 * 3600)
                result = read(target)
                if (result['backend'] != backend_name or result['model_sha256'] != identity[digest_key] or
                        result['reference_onnx_sha256'] != identity['onnx_sha256']):
                    raise ValueError(f'Evaluation refers to a different artifact: {name}')
                if kind == 'historical_real':
                    if (result['status'] != REAL_STATUS or result['manifest_sha256'] != MANIFEST_SHA or
                            result['evaluation_split'] != 'test'):
                        raise ValueError('Historical real evaluation identity differs')
                elif (result['status'] != SEQUENCE_STATUS or result['dataset_index_sha256'] != sequence_sha or
                      result['sequence_count'] != accepted['sequences'] or
                      result['denominators']['frames'] != accepted['images']):
                    raise ValueError('Sequence evaluation did not cover the accepted fixture')
                results[name] = {'name': target.name, 'sha256': sha(target),
                                 'denominators': result['denominators']}

        value = {'status': 'FORMAL_HOST_EVALUATION_COMPLETE',
                 'observed_utc': datetime.now(timezone.utc).isoformat(), 'run_id': models['run_id'],
                 'checkpoint_sha256': identity['checkpoint_sha256'], 'model_receipt_sha256': sha(model_receipt),
                 'synthetic_reports_receipt_sha256': sha(report_receipt),
                 'sequence_acceptance_sha256': sha(sequence_acceptance), 'evaluations': results,
                 'reused_evaluations': reused,
                 'native_K230_test': 'NOT_RUN'}
        atomic_json(receipt, value)
        return value
    finally:
        lock_path.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('conversion', 'dataset', 'sequence-acceptance', 'output'):
        parser.add_argument('--' + argument, type=Path, required=True)
    parser.add_argument('--sequence-candidate', type=Path)
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))
