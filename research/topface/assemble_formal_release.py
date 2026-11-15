"""Assemble an offline release after the original conversion and host evaluations."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import zipfile

from cloud_store import atomic_json, sha
from deploy_board_trial import verified_assets


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs/topface/v11'
CONVERSION = DOCS / 'cloud/conversion-formal-a01'
HOST = DOCS / 'reports/formal-host-a01'
HOST_REPORTS = ('historical_real_onnx', 'historical_real_simulator',
                'synthetic_sequence_onnx', 'synthetic_sequence_simulator')
STATIC_EVIDENCE = (
    'audit/PROJECT_AUDIT.md', 'audit/REUSE_MAP.md', 'audit/RISK_AND_UNKNOWN.md',
    'camera/camera_profile.yaml', 'camera/camera_sweep.yaml', 'camera/calibration_gap_report.md',
    'cad/CAD_VALIDATION.md', 'cad/geometry_validation.json',
    'reports/SMOKE_HUMAN_REVIEW.json', 'reports/SMOKE_DRIVE_ACCEPTANCE.json',
    'cloud/pilot-a02/ACCEPTANCE.json', 'cloud/audit-a02/ACCEPTANCE-v4.json',
    'reports/PILOT_METRICS.md', 'reports/REDETECTION_SWEEP_ACCEPTANCE_A01.json',
    'training/FULL_INVENTORY.json', 'training/FULL_DATASET_ACCEPTANCE.json',
    'training/FORMAL_TRAINING_A01.md', 'cloud/formal-a01/FORMAL_CONFIG.json',
    'cloud/formal-a01/ACCEPTANCE.json', 'cloud/formal-a01/SELECTED_MODEL.json',
    'cloud/formal-a01/compact-v1/k230-topface-v11-formal-a01-RESULT.json',
    'cloud/formal-a01/compact-v1/formal_full/RESULT.json',
    'cloud/formal-a01/compact-v1/formal_full/BEST_METRICS.json',
    'cloud/publisher/k230-topface-v11-formal-a01/TRANSFER_RECEIPT.json',
    'reports/BOARD_PAIR_PROTOCOL.md', 'reports/BOARD_TEST_PLAN.md',
    'reports/BOARD_PREVIEW_A01_ACCEPTANCE.json', 'reports/REQUIREMENT_COVERAGE.json',
)


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def release_inputs(package):
    inventory, inventory_sha, _ = verified_assets(package)
    if inventory['status'] != 'FORMAL_BOARD_TEST_PACKAGE_VERIFIED':
        raise ValueError('Full release requires the original 07b package with independent test results')
    identity = read(package / 'models/identity.json')
    selected = read(DOCS / 'cloud/formal-a01/SELECTED_MODEL.json')
    model_receipt = CONVERSION / 'MODEL_FILES_RECEIPT.json'
    report_receipt = CONVERSION / 'REPORTS_RECEIPT.json'
    host_receipt_path = HOST / 'HOST_EVALUATION_RECEIPT.json'
    host_receipt = read(host_receipt_path)
    if (identity['checkpoint_sha256'] != selected['checkpoint_sha256'] or
            host_receipt['status'] != 'FORMAL_HOST_EVALUATION_COMPLETE' or
            host_receipt['checkpoint_sha256'] != identity['checkpoint_sha256'] or
            host_receipt['run_id'] != inventory['run_id'] or
            sha(model_receipt) != host_receipt['model_receipt_sha256'] or
            sha(report_receipt) != host_receipt['synthetic_reports_receipt_sha256'] or
            set(host_receipt['evaluations']) != set(HOST_REPORTS)):
        raise ValueError('Formal package, selected checkpoint and host evaluation do not match')

    synthetic = read(package / 'synthetic-test/SYNTHETIC_TEST.json')
    paired = read(package / 'synthetic-test/PAIRED_SYNTHETIC_TEST.json')
    dataset = read(DOCS / 'training/FULL_DATASET_ACCEPTANCE.json')
    if (synthetic['images'] != 934 or paired['images'] != 934 or
            inventory['held_out_synthetic_images'] != 934 or paired['identity'] != identity or
            synthetic['dataset_audit_sha256'] != dataset['dataset_index_sha256'] or
            paired['dataset_index_sha256'] != dataset['dataset_index_sha256']):
        raise ValueError('The independent test dataset or model identity differs')

    host_reports = {}
    for name in HOST_REPORTS:
        path = HOST / (name + '.json')
        recorded = host_receipt['evaluations'][name]
        result = read(path)
        is_simulator = name.endswith('_simulator')
        model_key = 'kmodel_sha256' if is_simulator else 'onnx_sha256'
        if (recorded['name'] != path.name or recorded['sha256'] != sha(path) or
                result['model_sha256'] != identity[model_key] or
                result['reference_onnx_sha256'] != identity['onnx_sha256'] or
                result['denominators'] != recorded['denominators'] or
                (name.startswith('synthetic_sequence') and result['denominators']['frames'] != 1872)):
            raise ValueError(f'Host report differs from the accepted evaluation: {name}')
        host_reports[name] = result
    publication = DOCS / 'cloud/publisher' / inventory['run_id'] / 'TRANSFER_RECEIPT.json'
    if sha(publication) != read(model_receipt)['publication_receipt_sha256']:
        raise ValueError('Conversion archive receipt changed')
    return inventory_sha, identity, synthetic, paired, host_reports, publication


def drive_objects(publication):
    smoke = read(DOCS / 'reports/SMOKE_DRIVE_ACCEPTANCE.json')
    pilot = read(DOCS / 'cloud/pilot-a02/ACCEPTANCE.json')['drive_archive']
    colorfix = read(DOCS / 'cloud/colorfix-a01/compact-v2/k230-topface-v11-color-counterexamples-a01-DRIVE.json')
    audit = read(DOCS / 'cloud/audit-a02/ACCEPTANCE-v4.json')['drive_archive']
    full = read(DOCS / 'training/FULL_INVENTORY.json')
    if len(full['shards']) != 10 or full['accepted_scenes'] != 10000:
        raise ValueError('Expected the ten accepted Full shards')
    rows = [('Smoke', smoke), ('Pilot', pilot), ('颜色补充', colorfix), ('Pilot 组合审计', audit)]
    rows += [(f"Full 分片 {i}", row['publication']['uploaded']) for i, row in enumerate(full['shards'], 1)]
    rows += [('运动序列', read(DOCS / 'reports/REDETECTION_SWEEP_ACCEPTANCE_A01.json')['drive']),
             ('正式训练', read(DOCS / 'cloud/publisher/k230-topface-v11-formal-a01/TRANSFER_RECEIPT.json')['uploaded']),
             ('正式转换和原始评估', read(publication)['uploaded'])]
    for _, row in rows:
        if row.get('provider_readback', row.get('status')) != 'FULL_BYTE_SHA256_PASS':
            raise ValueError('Drive artifact lacks full readback acceptance')
    return [{'label': label, **{key: row[key] for key in ('name', 'file_id', 'bytes', 'sha256')}}
            for label, row in rows]


def markdown(synthetic, paired, host_reports, objects):
    lines = ['# TopFaceNet V1.1 正式模型交付', '',
             '正式训练 100 轮，按验证集选定第 92 轮权重。', '',
             '| 用途 | 文件 |', '|---|---|',
             '| K230 测试 | [board](board/) 与 [运行说明](board/README.md) |',
             '| 模型及输入输出契约 | [models](models/) |',
             '| 独立静态仿真结果与失败图 | [synthetic-test](synthetic-test/) |',
             '| 历史实拍与仿真运动结果 | [host-evaluation](host-evaluation/) |',
             '| 相机、CAD、审计、训练和复现证据 | [evidence](evidence/) |',
             '| 全部文件校验 | [SHA256SUMS.json](SHA256SUMS.json) |', '',
             '## 934 张独立仿真图', '',
             '| 后端 | mAP@0.5 | mAP@0.5:0.95 | 召回@0.5 | 匹配颜色正确率 |',
             '|---|---:|---:|---:|---:|']
    for name, result in (('PyTorch', synthetic), ('ONNX Runtime', paired['backends']['ort']),
                         ('nncase Simulator', paired['backends']['simulator'])):
        values = [result[k] for k in ('task_mAP50', 'task_mAP50_95', 'class_agnostic_recall50',
                                      'matched_color_accuracy')]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{v:.2%}' for v in values) + ' |')
    lines += ['', '## 历史实拍和仿真运动', '',
              '| 数据及后端 | 图像或帧 | TP | FP | FN |', '|---|---:|---:|---:|---:|']
    labels = ('历史实拍 ONNX', '历史实拍 Simulator', '仿真运动 ONNX', '仿真运动 Simulator')
    for label, name in zip(labels, HOST_REPORTS):
        row = host_reports[name]['denominators']
        lines.append(f"| {label} | {row.get('images', row.get('frames'))} | {row['tp']} | {row['fp']} | {row['fn']} |")
    lines += ['', '实机兼容性、准确率、FPS、稳定性和新拍盲测待用户安排。历史实拍的几何真值缺失。',
              '板测步骤见 [实机测试计划](evidence/reports/BOARD_TEST_PLAN.md)。', '',
              '## 原计划交付索引', '',
              '1. [项目审计](evidence/audit/PROJECT_AUDIT.md)',
              '2. [复用地图](evidence/audit/REUSE_MAP.md)',
              '3. [风险与未知](evidence/audit/RISK_AND_UNKNOWN.md)',
              '4. [相机资料](evidence/camera/camera_profile.yaml)',
              '5. [仿真相机范围](evidence/camera/camera_sweep.yaml)',
              '6. [CAD 验证](evidence/cad/CAD_VALIDATION.md)',
              '7. [仿真生成源代码](source.zip)，入口在 research/topface/render_native.py。',
              '8. [Smoke 预览](evidence/previews/index.html)和 [完整归档索引](DRIVE_ARTIFACTS.json)。',
              '9. [Pilot 数据与诊断](evidence/cloud/pilot-a02/ACCEPTANCE.json)和 [补充数据审计](evidence/cloud/audit-a02/ACCEPTANCE-v4.json)。',
              '10. [Full 数据验收](evidence/training/FULL_DATASET_ACCEPTANCE.json)和 [十个分片](DRIVE_ARTIFACTS.json)。',
              '11. [训练代码](source.zip)和 [正式权重](models/topface-best.pt)。',
              '12. [训练报告](evidence/training/FORMAL_TRAINING_A01.md)、[配置](evidence/cloud/formal-a01/FORMAL_CONFIG.json)和 [原日志归档](DRIVE_ARTIFACTS.json)。',
              '13. [独立静态仿真](synthetic-test/)，[运动序列与历史实拍](host-evaluation/)。',
              '14. [ONNX](models/topface.onnx)和 [KModel](models/topface.kmodel)。',
              '15. [输入输出契约](models/output_contract.json)。',
              '16. [K230 包](board/)和 [板测计划](evidence/reports/BOARD_TEST_PLAN.md)，原生结果待测。',
              '17. [相机标定缺口](evidence/camera/calibration_gap_report.md)和 [板端配对步骤](evidence/reports/BOARD_PAIR_PROTOCOL.md)。', '',
              '## 已校验的 Drive 归档', '']
    for row in objects:
        lines.append(f"- {row['label']}　{row['bytes']:,} 字节。"
                     f" [Drive 原件](https://drive.google.com/file/d/{row['file_id']}/view)")
    lines += ['', '归档的文件名、大小和 SHA-256 见 [DRIVE_ARTIFACTS.json](DRIVE_ARTIFACTS.json)。', '']
    return '\n'.join(lines)


def assemble(package, output):
    package, output = package.resolve(), output.resolve()
    archive = output.parent / (output.name + '.zip')
    if output.exists() or archive.exists() or not output.parent.is_dir():
        raise FileExistsError('Choose a new release directory and ZIP in an existing parent')
    package_sha, identity, synthetic, paired, host_reports, publication = release_inputs(package)
    objects = drive_objects(publication)
    with TemporaryDirectory(prefix='.topface-release-', dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        source_inventory = read(package / 'SHA256SUMS.json')
        if sha(package / 'SHA256SUMS.json') != package_sha:
            raise ValueError('Original board package inventory changed')
        for row in source_inventory['files']:
            relative = Path(row['path'])
            if relative.parts[0] not in ('models', 'board', 'synthetic-test'):
                continue
            source = package / relative
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if target.stat().st_size != row['bytes'] or sha(target) != row['sha256']:
                raise ValueError(f'Original package file changed: {relative}')
        for relative in STATIC_EVIDENCE:
            target = staging / 'evidence' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(DOCS / relative, target)
        shutil.copytree(DOCS / 'previews', staging / 'evidence/previews')
        shutil.copyfile(publication, staging / 'evidence/CONVERSION_DRIVE_RECEIPT.json')
        for name in ('MODEL_FILES_RECEIPT.json', 'REPORTS_RECEIPT.json'):
            shutil.copyfile(CONVERSION / name, staging / 'evidence' / name)
        host_output = staging / 'host-evaluation'
        host_output.mkdir()
        for name in (*HOST_REPORTS, 'HOST_EVALUATION_RECEIPT'):
            shutil.copyfile(HOST / (name + '.json'), host_output / (name + '.json'))
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        subprocess.run(['git', 'archive', '--format=zip', '-o', str(staging / 'source.zip'), revision,
                        'research/topface'], cwd=ROOT, check=True)
        atomic_json(staging / 'DRIVE_ARTIFACTS.json', objects)
        (staging / 'README.md').write_text(markdown(synthetic, paired, host_reports, objects), encoding='utf8')
        files = {path.relative_to(staging).as_posix(): path for path in sorted(staging.rglob('*')) if path.is_file()}
        receipt = {'status': 'FORMAL_MODEL_AND_OFFLINE_EVALUATION_RELEASE_VERIFIED',
                   'source_tree_commit': revision, 'model_identity': identity, 'board_package_sha256': package_sha,
                   'native_board_test': 'USER_DEFERRED_TEST', 'files': [
                       {'path': name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
                       for name, path in files.items()]}
        atomic_json(staging / 'SHA256SUMS.json', receipt)
        files['SHA256SUMS.json'] = staging / 'SHA256SUMS.json'
        building_zip = Path(temporary) / archive.name
        with zipfile.ZipFile(building_zip, 'x', compression=zipfile.ZIP_DEFLATED) as zipped:
            for name, path in files.items():
                zipped.write(path, f'{output.name}/{name}')
        with zipfile.ZipFile(building_zip) as zipped:
            for name, path in files.items():
                if sha256(zipped.read(f'{output.name}/{name}')).hexdigest() != sha(path):
                    raise ValueError(f'ZIP member failed verification: {name}')
        zip_sha = sha(building_zip)
        staging.rename(output)
        building_zip.rename(archive)
    return {'status': receipt['status'], 'directory': str(output), 'zip': str(archive),
            'zip_sha256': zip_sha, 'files': len(files), 'native_board_test': receipt['native_board_test']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    print(json.dumps(assemble(**vars(parser.parse_args())), indent=2))
