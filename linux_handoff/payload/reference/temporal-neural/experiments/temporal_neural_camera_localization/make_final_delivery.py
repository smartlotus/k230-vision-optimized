"""Extend the verified long-window delivery with completed compiler experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from evaluate_neural import SETTINGS, digest, read, verified_graph


ROOT = Path(__file__).resolve().parent


def verify_backend(backend, paired_path, collection, metrics, parent, setting, full_sha, trained_sha, derivative):
    report_path = backend / 'KMODEL_REPORT.json'
    compiled = read(report_path)
    paired = read(paired_path)
    route, window = SETTINGS[setting]
    expected = dict(status='COMPILE_AND_SIMULATOR_PASS', route=route, window=window,
                    input_shape=[1, window * 3, 96, 160], input_type='uint8',
                    float_onnx_sha256=metrics['onnx_sha256'], checkpoint_sha256=metrics['best_sha256'],
                    dataset_manifest_sha256=full_sha, parent_sha256=parent, evaluation_split='test',
                    evaluated_frames=250, full_evaluation_split_frames=250)
    if (any(compiled.get(key) != value for key, value in expected.items())
            or compiled.get('training_manifest_sha256', full_sha) != trained_sha
            or paired.get('status') != 'NEURAL_KMODEL_PAIRED_TEST_VERIFIED'
            or paired.get('setting') != setting or paired.get('test_frames') != 250
            or paired.get('source') != collection['source']
            or paired.get('dataset_manifest_sha256') != full_sha
            or paired.get('training_manifest_sha256') != trained_sha
            or paired.get('raw_onnx_sha256') != metrics['onnx_sha256']
            or paired.get('simulator_report_sha256') != digest(report_path)
            or paired.get('simulator_scores_sha256') != digest(backend / 'SIMULATOR_SCORES.npz')):
        raise ValueError('Paired compiler report identity differs from the verified trained model')
    for name, key in (('camera_xy.kmodel', 'kmodel_sha256'), ('camera_xy_uint8.onnx', 'uint8_onnx_sha256')):
        if compiled.get(key) != digest(backend / name) or paired.get(key) != compiled[key]:
            raise ValueError('Compiled delivery member differs from paired evidence')
    lowering = compiled.get('onnx_lowering', {})
    if (lowering.get('source_sha256') != metrics['onnx_sha256']
            or lowering.get('derivative_sha256') != digest(backend / derivative)
            or lowering.get('original_paired_frames') != 250):
        raise ValueError('ONNX derivative differs from the source and full-parity evidence')


def build(out, receipt, gru_models, gru_data):
    base = read(ROOT / 'LONG_OFFLINE_DELIVERY.json')
    base_path = Path(base['path'])
    if out.exists() or digest(base_path) != base['sha256']:
        raise ValueError('Expected the verified base archive and a new final path')
    original_sha = read(ROOT / 'GRU_UNROLLED_FLOAT_PARITY.json')['dataset_manifest_sha256']
    if digest(gru_data / 'manifest.jsonl') != original_sha:
        raise ValueError('Original GRU holdout identity differs')
    collection = read(gru_models / 'MODEL_COLLECTION.json')
    files = {}
    prefix = 'source/experiments/' + ROOT.name + '/'
    omitted = {'FINAL_OFFLINE_DELIVERY.json', 'FINAL_OFFLINE_PACKAGE_REPLAY.json'}
    for path in ROOT.rglob('*'):
        rel = path.relative_to(ROOT)
        if path.is_file() and not any(p in ('.runtime', '__pycache__', 'gmodel_dump_dir') for p in rel.parts) and path.name not in omitted:
            files[prefix + rel.as_posix()] = path
    files['source/experiments/camera_xy_localization_v2/contract.py'] = ROOT / 'evidence/frozen-renderers/long-a05-camera/contract.py'
    for setting in ('gru4', 'gru8'):
        _, _, _, metrics, parent, _ = verified_graph(gru_models, collection, setting, original_sha)
        for name in ('best.pt', 'camera_xy.onnx', 'METRICS.json', 'history.json'):
            files['models/original-gru/' + setting + '/' + name] = gru_models / setting / name
        backend = ROOT / '.runtime' / (setting + '-unrolled-a16-test')
        verify_backend(backend, ROOT / (setting.upper() + '_UNROLLED_KMODEL_TEST.json'), collection,
                       metrics, parent, setting, original_sha, original_sha, 'camera_xy_unrolled.onnx')
        trial = read(ROOT / (setting.upper() + '_UNROLLED_BOARD_TRIAL.json'))
        trial_path = Path(trial['archive'])
        if (trial['status'] != 'NEURAL_BOARD_TRIAL_ARCHIVE_VERIFIED' or trial['setting'] != setting
                or trial['archive_sha256'] != digest(trial_path)
                or trial['kmodel_sha256'] != digest(backend / 'camera_xy.kmodel')):
            raise ValueError('GRU board trial differs from the paired compiled model')
        files['trials/' + trial_path.name] = trial_path
        for name in ('camera_xy_unrolled.onnx', 'camera_xy_uint8.onnx', 'camera_xy.kmodel', 'KMODEL_REPORT.json', 'SIMULATOR_SCORES.npz'):
            files['models/compiled-' + setting + '/' + name] = backend / name
    files['models/original-gru/MODEL_COLLECTION.json'] = gru_models / 'MODEL_COLLECTION.json'
    for name in ('manifest.jsonl', 'QA.json'):
        files['data/original-gru/' + name] = gru_data / name
    for line in (gru_data / 'manifest.jsonl').read_text(encoding='utf8').splitlines():
        row = json.loads(line)
        for kind in ('image', 'mask'):
            path = gru_data / row[kind + '_path']
            if digest(path) != row[kind + '_sha256']:
                raise ValueError('Original GRU dataset member differs')
            files['data/original-gru/' + row[kind + '_path']] = path
    mixture = ROOT / '.runtime/mixture-a06-models'
    trained = read(ROOT / 'cloud/mixture-a06/SUBMISSION_INTENT.json')['input']['training_manifest_sha256']
    mixture_collection = read(mixture / 'MODEL_COLLECTION.json')
    _, _, _, mixture_metrics, mixture_parent, _ = verified_graph(mixture, mixture_collection, 'tcn8', trained)
    for name in ('best.pt', 'camera_xy.onnx', 'METRICS.json', 'history.json'):
        files['models/mixture-a06/tcn8/' + name] = mixture / 'tcn8' / name
    files['models/mixture-a06/MODEL_COLLECTION.json'] = mixture / 'MODEL_COLLECTION.json'
    conversion = read(ROOT / 'MIXTURE_A06_CONVERSION_STATUS.json')
    if conversion['status'] == 'TRAINED_MODEL_STABLE_COMPILE_AND_FULL_TEST_PASS':
        backend = ROOT / '.runtime/mixture-a06-tcn8-stable-a16-test'
        full = read(ROOT / 'GEOMETRY_PILOT_QA.json')['manifest_sha256']
        verify_backend(backend, ROOT / 'MIXTURE_A06_TCN8_KMODEL_TEST.json', mixture_collection,
                       mixture_metrics, mixture_parent, 'tcn8', full, trained, 'camera_xy_stable.onnx')
        for name in ('camera_xy_stable.onnx', 'camera_xy_uint8.onnx', 'camera_xy.kmodel', 'KMODEL_REPORT.json', 'SIMULATOR_SCORES.npz'):
            files['models/compiled-mixture/' + name] = backend / name
    elif conversion['status'] != 'TRAINED_MODEL_COMPILER_EXPERIMENT_CLOSED_WITH_FAILURE':
        raise ValueError('Mixture compiler experiment must have a verified outcome')
    exp = prefix.rstrip('/')
    start = (
        '# K230 时序神经网络离线交付\n\n'
        '本包包含正确田字形地图、RGB 序列、单帧/时序模型、训练血统、固定窗口运行时、'
        '编译模型、同帧评测与独立 SD 试验入口。比赛从随机一个蓝区出发，另一个蓝区为终点。\n\n'
        '**精度仍为 HOLD，原生设备验收未执行。** 原计划的 500 mm P95 门限未通过，'
        '10k–20k 全量训练继续等待 Pilot 条件通过。A06 连续空间头未升级为主模型。\n\n'
        f'状态及统一人工接入清单：{exp}/OFFLINE_STATUS.json、{exp}/HUMAN_INPUTS.md。'
        'TCN8/TCN16 独立板端入口分别在 board 与 board-tcn16。'
        'GRU4/GRU8 的独立板端试验 ZIP 在 trials 中，各包带 RUN.md，仍需实机验收。\n\n'
        '安装源码目录中的 requirements_host.txt 后，可在包根复验：\n\n'
        '```powershell\n'
        f'python {exp}/evaluate_neural_simulator.py --data data/original-gru --models models/original-gru --backend models/compiled-gru4 --setting gru4 --out gru4-test.json\n'
        f'python {exp}/evaluate_neural_simulator.py --data data/original-gru --models models/original-gru --backend models/compiled-gru8 --setting gru8 --out gru8-test.json\n'
        f'python {exp}/evaluate_long.py --data data/long-spatial --plan {exp}/LONG_SPATIAL_PLAN.json --reference data/geometry-a03 --field assets/field_mask.png --baseline models/geometry-a03-models --candidate models/mixture-a06 --setting tcn8 --out mixture-spatial.json\n'
        f'python -m unittest discover -s {exp} -p "test_neural*.py"\n'
        '```\n\n'
        '需要重渲染长空间路线时，使用 Blender 5.3.0 和包内冻结的依赖。输出目录须为新目录：\n\n'
        '```powershell\n'
        f'blender -b -t 2 --python {exp}/render_challenge.py -- --plan {exp}/LONG_SPATIAL_PLAN.json '
        '--blend assets/field_gray_v1.blend --cad assets/material_block.glb '
        f'--roadnet {exp}/evidence/frozen-renderers/long-a05-roadnet '
        f'--camera-renderer {exp}/evidence/frozen-renderers/long-a05-camera/render.py --out rerender-long\n'
        '```\n\n'
        'TCN16 量化与长轨迹复验另见 LONG_WINDOW_README.md。所有模拟器时间均为宿主时间。'
        'GRU 原循环节点导入失败和 A06 直接转换的 NaN 记录继续保留；等价展开另存派生模型，原始权重未改。\n'
    ).encode('utf8')
    extra = {'START_HERE.md': start}
    manifest = {}
    partial = out.with_suffix('.zip.partial')
    try:
        with zipfile.ZipFile(base_path) as old, zipfile.ZipFile(partial, 'x', zipfile.ZIP_DEFLATED) as new:
            base_members = json.loads(old.read('SHA256SUMS.json'))
            long_readme = old.read('LONG_WINDOW_README.md')
            if hashlib.sha256(long_readme).hexdigest() != base_members['LONG_WINDOW_README.md']['sha256']:
                raise ValueError('Long-window readme differs')
            previous = '当前连续空间输出头回合 A06 还在 Kaggle 排队，来源和提交身份随源码保留。'
            text = long_readme.decode('utf8')
            if text.count(previous) != 1:
                raise ValueError('Expected one historical queued-run sentence')
            extra['LONG_WINDOW_README.md'] = text.replace(previous, '连续空间输出头回合 A06 已完成，结果和模型随包保留，当前定位精度仍为 HOLD。').encode('utf8')
            for name, identity in base_members.items():
                if name in files or name in extra:
                    continue
                content = old.read(name)
                if len(content) != identity['bytes'] or hashlib.sha256(content).hexdigest() != identity['sha256']:
                    raise ValueError('Base member differs from accepted identity')
                new.writestr(name, content)
                manifest[name] = identity
            for name, path in files.items():
                new.write(path, name)
                manifest[name] = dict(bytes=path.stat().st_size, sha256=digest(path))
            for name, content in extra.items():
                new.writestr(name, content)
                manifest[name] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            new.writestr('SHA256SUMS.json', json.dumps(manifest, indent=2))
        with zipfile.ZipFile(partial) as package:
            if set(package.namelist()) != set(manifest) | {'SHA256SUMS.json'}:
                raise ValueError('Final inventory differs')
            for name, identity in manifest.items():
                with package.open(name) as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if actual != identity['sha256'] or package.getinfo(name).file_size != identity['bytes']:
                    raise ValueError('Final member full readback failed')
        partial.rename(out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    value = dict(status='FINAL_NEURAL_OFFLINE_ARCHIVE_FULL_READBACK_PASS', path=str(out.resolve()),
                 bytes=out.stat().st_size, sha256=digest(out), members=len(manifest) + 1,
                 base_archive_sha256=base['sha256'], quality='HOLD', native='NOT_RUN',
                 human_input_list=prefix + 'HUMAN_INPUTS.md', mixture_conversion=conversion['status'])
    receipt.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')
    print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('out', 'receipt', 'gru-models', 'gru-data'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.out, args.receipt, args.gru_models, args.gru_data)
