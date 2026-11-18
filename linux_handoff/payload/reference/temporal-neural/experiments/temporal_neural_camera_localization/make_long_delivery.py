"""Deliver the complete verified A03 bundle plus the causal16 source and artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from evaluate_neural import digest, read


ROOT = Path(__file__).resolve().parent


def build(out, record_path):
    prior = read(ROOT / 'OFFLINE_DELIVERY.json')
    prior_archive = Path(prior['path'])
    if digest(prior_archive) != prior['sha256'] or out.exists():
        raise ValueError('Expected the original verified base archive and a new output path')
    trial = read(ROOT / 'LONG_A05_BOARD_TRIAL.json')
    trial_path = Path(trial['archive'])
    if digest(trial_path) != trial['archive_sha256']:
        raise ValueError('Long-window trial differs from receipt')
    files = {}
    prefix = 'source/experiments/' + ROOT.name + '/'
    for path in ROOT.rglob('*'):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(part in ('.runtime', '__pycache__') for part in rel.parts):
            continue
        files[prefix + rel.as_posix()] = path
    for name in ('camera_xy.onnx', 'best.pt', 'METRICS.json', 'history.json', 'MODEL_COLLECTION.json'):
        rel = name if name == 'MODEL_COLLECTION.json' else 'tcn16/' + name
        files['models/long-a05-models/' + rel] = ROOT / '.runtime/long-a05-models' / rel
    backend = ROOT / '.runtime/long-a05-tcn16-a16-test'
    for name in ('camera_xy_uint8.onnx', 'camera_xy.kmodel', 'KMODEL_REPORT.json', 'SIMULATOR_SCORES.npz'):
        files['models/compiled-tcn16/' + name] = backend / name
    data = ROOT / '.runtime/long-spatial-a05'
    for name in ('manifest.jsonl', 'QA.json'):
        files['data/long-spatial/' + name] = data / name
    for line in (data / 'manifest.jsonl').read_text(encoding='utf8').splitlines():
        row = json.loads(line)
        for kind in ('image', 'mask'):
            path = data / row[kind + '_path']
            if digest(path) != row[kind + '_sha256']:
                raise ValueError('Long-spatial input changed')
            files['data/long-spatial/' + row[kind + '_path']] = path
    extra = {}
    with zipfile.ZipFile(trial_path) as package:
        extra.update({'board-tcn16/' + name: package.read(name) for name in package.namelist()})
    exp = prefix.rstrip('/')
    extra['LONG_WINDOW_README.md'] = (
        '# 时序神经网络长窗口离线接续\n\n'
        '本包含原已验证 A03 完整数据和 TCN8 板端试验，以及新增 TCN16、240 帧长轨迹和配套报告。'
        '定位精度仍为 HOLD。当前连续空间输出头回合 A06 还在 Kaggle 排队，来源和提交身份随源码保留。\n\n'
        '从包根运行以下命令复验长窗口量化结果。\n\n'
        '```powershell\n'
        f'python {exp}/evaluate_neural_simulator.py --data data/geometry-a03 --models models/long-a05-models '
        '--backend models/compiled-tcn16 --setting tcn16 --out long-quantized.json\n'
        f'python {exp}/evaluate_long.py --data data/long-spatial --plan {exp}/LONG_SPATIAL_PLAN.json '
        '--reference data/geometry-a03 --field assets/field_mask.png --baseline models/geometry-a03-models '
        '--candidate models/long-a05-models --out long-spatial.json\n'
        '```\n\n'
        'TCN16 板端试验目录为 board-tcn16。时间配置和逐帧记录说明见该目录 RUN.md。'
        'KModel Simulator 在宿主计算机执行，记录的时延属于宿主模拟器。'
        '真实设备、安装视野和光心真值待人工接入。\n').encode()
    manifest = {}
    partial = out.with_suffix('.zip.partial')
    stream = partial.open('xb')
    try:
        with zipfile.ZipFile(prior_archive) as old, stream, zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as new:
            old_manifest = json.loads(old.read('SHA256SUMS.json'))
            for name, receipt in old_manifest.items():
                if name in files or name in extra:
                    continue
                content = old.read(name)
                if (len(content), hashlib.sha256(content).hexdigest()) != (receipt['bytes'], receipt['sha256']):
                    raise ValueError('Base archive member differs')
                new.writestr(name, content)
                manifest[name] = receipt
            for name, path in files.items():
                new.write(path, name)
                manifest[name] = dict(bytes=path.stat().st_size, sha256=digest(path))
            for name, content in extra.items():
                new.writestr(name, content)
                manifest[name] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            new.writestr('SHA256SUMS.json', json.dumps(manifest, indent=2))
        with zipfile.ZipFile(partial) as package:
            if set(package.namelist()) != set(manifest) | {'SHA256SUMS.json'}:
                raise ValueError('Expanded delivery member list differs')
            for name, identity in manifest.items():
                with package.open(name) as source:
                    sha = hashlib.file_digest(source, 'sha256').hexdigest()
                if sha != identity['sha256'] or package.getinfo(name).file_size != identity['bytes']:
                    raise ValueError('Expanded delivery member readback failed')
        partial.rename(out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    value = dict(status='NEURAL_LONG_OFFLINE_DELIVERY_READBACK_PASS', path=str(out.resolve()),
                 bytes=out.stat().st_size, sha256=digest(out), members=len(manifest) + 1,
                 base_archive_sha256=prior['sha256'], scientific_quality='HOLD', native='NOT_RUN',
                 pending_run=read(ROOT / 'cloud/mixture-a06/KERNEL_SAVE_RESPONSE.json'))
    record_path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    build(args.out, args.receipt)
