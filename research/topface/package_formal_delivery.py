"""Assemble the verified formal model, board files, and synthetic test reports."""
import argparse
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from cloud_artifacts import extract_verified_archive
from cloud_store import atomic_json, sha
from collect_model_files import validate_accepted_conversion
from collect_model_reports import manifest, validate_reports


MODEL_NAMES = {'topface-best.pt', 'topface.onnx', 'topface.kmodel', 'identity.json',
               'output_contract.json', 'topface-board-bundle.tar.gz'}
BOARD_NAMES = {'topface_trial.py', 'topface_decode.py', 'topface_pair_capture.py',
               'topface.kmodel', 'identity.json', 'output_contract.json', 'README.md',
               'BENCHMARK_STATUS.json', 'MANIFEST.json'}


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def copy_verified(source, destination, inventory):
    for name, row in inventory.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
        if (target.stat().st_size, sha(target)) != (row['bytes'], row['sha256']):
            raise ValueError(f'Collected file differs from the conversion inventory: {name}')


def verify_board(board, model):
    members = read(board / 'MEMBERS.json')
    if len(members) != len(BOARD_NAMES) or {row['name'] for row in members} != BOARD_NAMES:
        raise ValueError('Board bundle member inventory differs')
    manifest = read(board / 'MANIFEST.json')
    expected = [row for row in members if row['name'] != 'MANIFEST.json']
    if (manifest['status'] != 'DEPLOYMENT_BUNDLE_READY' or manifest['board_test'] != 'NOT_RUN' or
            sorted(manifest['files'], key=lambda row: row['name']) != sorted(expected, key=lambda row: row['name'])):
        raise ValueError('Board package manifest differs from the archive members')
    for name in ('identity.json', 'topface.kmodel', 'output_contract.json'):
        if sha(board / name) != sha(model / name):
            raise ValueError(f'Board package contains a different {name}')
    benchmark = read(board / 'BENCHMARK_STATUS.json')
    if benchmark['status'] != 'NOT_RUN' or benchmark['model_sha256'] != sha(model / 'topface.kmodel'):
        raise ValueError('Board benchmark status refers to a different package')


def package(conversion, publication, dataset_acceptance, output):
    models_path = conversion / 'MODEL_FILES_RECEIPT.json'
    reports_path = conversion / 'REPORTS_RECEIPT.json'
    models, reports = read(models_path), read(reports_path)
    provider = models['source']
    run_id = models['run_id']
    if (models['status'] != 'BOUNDED_MODEL_DELIVERABLES_SHA256_PASS' or
            reports['status'] != 'BOUNDED_COMPLETE_MODEL_REPORTS_SHA256_PASS' or
            any(models[key] != reports[key] for key in
                ('run_id', 'source', 'source_result_sha256', 'publication_receipt_sha256')) or
            models['publication_receipt_sha256'] != sha(publication)):
        raise ValueError('Collected models and reports require matching publication receipts')
    compact = conversion / f"compact-v{provider['version']}"
    saved_path = compact / f'{run_id}-RESULT.json'
    if sha(saved_path) != models['source_result_sha256']:
        raise ValueError('Exact saved conversion result changed')
    saved, durable, accepted = read(saved_path), read(publication), read(dataset_acceptance)
    validate_accepted_conversion(saved, durable)
    if (saved['export']['experiment'] != 'formal_full' or durable['prep'] != provider or
            saved['model_files'] != models['files'] or saved['evaluation_reports'] != reports['files'] or
            accepted['status'] != 'FULL_DATASET_ACCEPTED'):
        raise ValueError('Expected the original formal conversion and accepted Full dataset')

    model_files = {Path(row['path']).name: row for row in models['files']}
    report_files = {row['path']: row for row in reports['files']}
    if len(models['files']) != len(MODEL_NAMES) or set(model_files) != MODEL_NAMES:
        raise ValueError('Formal models require all six delivery files')
    manifest(saved)

    output = output.resolve()
    if output.exists():
        raise FileExistsError('Choose a new local delivery directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.topface-delivery-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'delivery'
        model = stage / 'models'
        evaluation = stage / 'synthetic-test'
        board = stage / 'board'
        copy_verified(Path(models['local_directory']), model, model_files)
        copy_verified(compact / 'evaluation', evaluation, report_files)
        validate_reports(evaluation, saved)
        test = read(evaluation / 'SYNTHETIC_TEST.json')
        if (test['images'] != accepted['split_images']['synthetic_test'] or
                test['dataset_audit_sha256'] != accepted['dataset_index_sha256']):
            raise ValueError('Independent test report does not match the accepted dataset')
        extract_verified_archive(model / 'topface-board-bundle.tar.gz', board,
                                 max_members=len(BOARD_NAMES) + 1, max_expanded_bytes=32 * 1024**2)
        verify_board(board, model)
        (stage / 'README.md').write_text(
            '# TopFaceNet V1.1 上板测试包\n\n'
            '在 CanMV 上使用 board 目录，运行方法见 board/README.md。\n\n'
            'models 目录包含正式权重、ONNX、量化 KModel 和输入输出契约。'
            'synthetic-test 包含独立仿真测试报告。实拍准确率和原生 FPS 需另行测量。\n\n'
            'SHA256SUMS.json 记录本目录各文件的大小与 SHA-256。\n',
            encoding='utf8')
        checksums = {'status': 'FORMAL_BOARD_TEST_PACKAGE_VERIFIED', 'run_id': run_id,
                     'provider': provider, 'checkpoint_sha256': saved['identity']['checkpoint_sha256'],
                     'conversion_archive_sha256': durable['uploaded']['sha256'],
                     'model_receipt_sha256': sha(models_path), 'report_receipt_sha256': sha(reports_path),
                     'held_out_synthetic_images': test['images'], 'native_board_test': 'NOT_RUN',
                     'files': [{'path': path.relative_to(stage).as_posix(),
                                'bytes': path.stat().st_size, 'sha256': sha(path)}
                               for path in sorted(stage.rglob('*')) if path.is_file()]}
        atomic_json(stage / 'SHA256SUMS.json', checksums)
        stage.rename(output)
    return checksums


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('conversion', 'publication', 'dataset-acceptance', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    receipt = package(**vars(args))
    print(json.dumps({'status': receipt['status'], 'output': str(args.output.resolve()),
                      'verified_files': len(receipt['files'])}, indent=2))
