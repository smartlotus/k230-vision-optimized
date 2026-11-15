"""Collect and package the independently published formal board preview."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from cloud_artifacts import extract_verified_archive
from cloud_store import atomic_json, sha
from collect_compact import download
from collect_model_files import validate_accepted_conversion
from package_formal_delivery import BOARD_NAMES, MODEL_NAMES, copy_verified, verify_board


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def collect_preview_reports(conversion, token_file):
    models = read(conversion / 'MODEL_FILES_RECEIPT.json')
    compact = conversion / f"compact-v{models['source']['version']}"
    result = compact / f"{models['run_id']}-RESULT.json"
    if sha(result) != models['source_result_sha256']:
        raise ValueError('Preview saved output changed')
    saved = read(result)
    rows = saved['evaluation_reports']
    if ({row['path'] for row in rows} != {'exported/RESULT.json', 'compiled/RESULT.json'} or
            len(rows) != 2 or any(not 0 < row['bytes'] <= 2 * 1024**2 for row in rows)):
        raise ValueError('Unexpected preview report inventory')
    os.environ['KAGGLE_API_TOKEN'] = token_file.read_text(encoding='utf8').strip()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    directory = compact / 'evaluation'
    with api.build_kaggle_client() as client:
        for row in rows:
            target = directory / row['path']
            if not target.exists():
                raw = download(client, models['source']['ref'], models['source']['version'],
                               saved['run_id'] + '/' + row['path'], row['bytes'])
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix('.json.part')
                temporary.write_bytes(raw)
                if temporary.stat().st_size != row['bytes'] or sha(temporary) != row['sha256']:
                    raise ValueError('Downloaded preview report differs')
                temporary.replace(target)
            if target.stat().st_size != row['bytes'] or sha(target) != row['sha256']:
                raise ValueError('Existing preview report changed; preserve and inspect')
    return directory


def package_preview(conversion, publication, output):
    receipt_path = conversion / 'MODEL_FILES_RECEIPT.json'
    receipt = read(receipt_path)
    compact = conversion / f"compact-v{receipt['source']['version']}"
    result_path = compact / f"{receipt['run_id']}-RESULT.json"
    saved, durable = read(result_path), read(publication)
    config = read(conversion / 'MODEL_ARTIFACT_CONFIG.json')
    if (receipt['status'] != 'BOUNDED_MODEL_DELIVERABLES_SHA256_PASS' or
            sha(result_path) != receipt['source_result_sha256'] or
            sha(publication) != receipt['publication_receipt_sha256'] or durable['prep'] != receipt['source'] or
            config['phase'] != 'formal_board_preview' or config['run_id'] != saved['run_id'] or
            config['parent']['kind'] != 'formal_model' or config['parent']['completed_epochs'] < 100 or
            saved['scope'] != 'FORMAL_BOARD_PREVIEW_ONLY' or saved['full_934_test'] != 'PENDING_ORIGINAL_07B' or
            saved['native_board_test'] != 'NOT_RUN' or saved['export']['experiment'] != 'formal_full' or
            saved['identity']['checkpoint_sha256'] != config['parent']['checkpoint_sha256'] or
            saved['model_files'] != receipt['files']):
        raise ValueError('Preview is not bound to the accepted formal model')
    validate_accepted_conversion(saved, durable)
    model_rows = {Path(row['path']).name: row for row in receipt['files']}
    if len(receipt['files']) != len(MODEL_NAMES) or set(model_rows) != MODEL_NAMES:
        raise ValueError('Preview needs all six model files')
    report_rows = {row['path']: row for row in saved['evaluation_reports']}
    if len(saved['evaluation_reports']) != 2 or set(report_rows) != {'compiled/RESULT.json', 'exported/RESULT.json'}:
        raise ValueError('Preview paired reports are missing')
    evaluation = compact / 'evaluation'
    exported, compiled = read(evaluation / 'exported/RESULT.json'), read(evaluation / 'compiled/RESULT.json')
    if (exported != saved['export'] or exported['status'] != 'PYTORCH_ONNX_PARITY_PASS' or
            exported['actual_train_image_pytorch_ort']['images'] != 12 or
            exported['actual_train_image_pytorch_ort']['pass'] is not True or
            compiled['identity'] != saved['identity'] or len(compiled['calibration_members']) != 64 or
            len(compiled['paired_members']) != 12 or len(compiled['paired_measurements']) != 12 or
            any(row['split'] != 'train' for row in compiled['calibration_members'] + compiled['paired_members'])):
        raise ValueError('Train-only calibration or paired export verification differs')
    output = output.resolve()
    if output.exists():
        raise FileExistsError('Keep every previous preview package unchanged')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.topface-preview-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'package'
        copy_verified(Path(receipt['local_directory']), stage / 'models', model_rows)
        copy_verified(evaluation, stage / 'paired-checks', report_rows)
        board = stage / 'board'
        extract_verified_archive(stage / 'models/topface-board-bundle.tar.gz', board,
                                 max_members=len(BOARD_NAMES) + 1, max_expanded_bytes=32 * 1024**2)
        verify_board(board, stage / 'models')
        (stage / 'README.md').write_text(
            '# TopFaceNet 正式权重上板试用包\n\n'
            '来自本轮完成100轮训练后按验证集选定的best权重。'
            '已执行PyTorch/ONNX验证、64张train量化校准及12张train ORT/Simulator配对。\n\n'
            '原定934张独立测试和1872帧运动评估继续另行执行。'
            '本包不表示独立测试、实机准确率或FPS通过。\n\n'
            '把board目录文件复制到一个新的SD测试目录；保护main.py/boot.py及既有目录。'
            '在CanMV中显式指定这个新目录，例如：\n\n'
            '```python\nimport sys\nsys.path.insert(0,"/sdcard/tf_preview01")\n'
            'from topface_trial import run\nrun(root="/sdcard/tf_preview01",frames=600,display=True)\n```\n\n'
            '硬件配对与FPS需独立实测；UART和运动关闭，不保存生产参数。'
            '每个文件的大小与SHA见SHA256SUMS.json。\n', encoding='utf8')
        value = {'status': 'FORMAL_BOARD_PREVIEW_PACKAGE_VERIFIED', 'scope': 'FORMAL_BOARD_PREVIEW_ONLY',
                 'run_id': saved['run_id'], 'provider': receipt['source'],
                 'checkpoint_sha256': saved['identity']['checkpoint_sha256'],
                 'training_completed_epochs': config['parent']['completed_epochs'],
                 'selected_epoch': exported['epoch'], 'native_board_test': 'NOT_RUN',
                 'full_934_test': 'PENDING_ORIGINAL_07B', 'calibration_train_images': 64,
                 'paired_train_images': 12, 'publication_receipt_sha256': sha(publication),
                 'model_receipt_sha256': sha(receipt_path),
                 'files': [{'path': path.relative_to(stage).as_posix(), 'bytes': path.stat().st_size,
                            'sha256': sha(path)} for path in sorted(stage.rglob('*')) if path.is_file()]}
        atomic_json(stage / 'SHA256SUMS.json', value)
        stage.rename(output)
    return value
