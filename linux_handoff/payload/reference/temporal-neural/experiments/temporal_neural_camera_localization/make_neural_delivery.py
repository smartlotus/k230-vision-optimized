"""Archive the neural offline sources, exact data, model files and board entry."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from evaluate_neural import digest, read, verified_graph


ROOT = Path(__file__).resolve().parent


def create(out, data, assets, reference, board_archive, roadnet, cad):
    if out.exists():
        raise FileExistsError(out)
    audit = read(ROOT / 'GEOMETRY_PILOT_QA.json')
    report = read(ROOT / 'GEOMETRY_A03_TCN8_KMODEL_TEST.json')
    trial = read(ROOT / 'GEOMETRY_A03_BOARD_TRIAL_R2.json')
    if (digest(data / 'manifest.jsonl') != audit['manifest_sha256']
            or digest(board_archive) != trial['archive_sha256']
            or report['kmodel_sha256'] != trial['kmodel_sha256']):
        raise ValueError('Delivery data or board KModel differs from the verified artifacts')
    models = ROOT / '.runtime/geometry-a03-models'
    collection = read(models / 'MODEL_COLLECTION.json')
    for setting in ('cnn1', 'tcn8'):
        verified_graph(models, collection, setting, report['training_manifest_sha256'])
    files = {}
    for file in ROOT.rglob('*'):
        relative = file.relative_to(ROOT)
        if not file.is_file() or any(part in ('.runtime', '__pycache__') for part in relative.parts):
            continue
        if file.name == 'OFFLINE_DELIVERY.json':
            continue
        files['source/experiments/' + ROOT.name + '/' + relative.as_posix()] = file
    for name in ('model.py', 'board_probe.py', 'render.py'):
        files['source/experiments/camera_xy_localization_v2/' + name] = ROOT.parent / 'camera_xy_localization_v2' / name
    for prefix in ('geometry-a03-models', 'appearance-a02-models'):
        base = ROOT / '.runtime' / prefix
        for file in base.rglob('*'):
            if file.is_file() and file.name in ('best.pt', 'camera_xy.onnx', 'history.json', 'METRICS.json', 'MODEL_COLLECTION.json'):
                files['models/' + prefix + '/' + file.relative_to(base).as_posix()] = file
    compiled = ROOT / '.runtime/geometry-a03-tcn8-a16-test'
    for name in ('KMODEL_REPORT.json', 'SIMULATOR_SCORES.npz', 'camera_xy.kmodel', 'camera_xy_uint8.onnx'):
        files['models/compiled/' + name] = compiled / name
    for name in ('field_gray_v1.blend', 'field_mask.png', 'field_preview.png'):
        files['assets/' + name] = assets / name
    render_identity = read(data / 'QA.json')
    if digest(roadnet / 'blender_scene.py') != render_identity['roadnet_renderer_sha256'] or digest(cad) != render_identity['cad_sha256']:
        raise ValueError('Frozen rendering helper or CAD differs from the geometry dataset')
    for name in ('blender_scene.py', 'sim.py', 'configs/field_layout.json'):
        files['assets/roadnet/' + name] = roadnet / name
    files['assets/material_block.glb'] = cad
    if digest(reference) != read(ROOT / 'cloud/geometry-a03/SUBMISSION_INTENT.json')['parent_sha256']:
        raise ValueError('Pretrained reference differs')
    # Training is performed from the pinned private Kaggle input. The complete
    # trained inference weights are included above for offline replay.
    for name in ('manifest.jsonl', 'QA.json'):
        files['data/geometry-a03/' + name] = data / name
    rows = [json.loads(line) for line in (data / 'manifest.jsonl').read_text(encoding='utf8').splitlines()]
    for row in rows:
        for kind in ('image', 'mask'):
            path = data / row[kind + '_path']
            if digest(path) != row[kind + '_sha256']:
                raise ValueError('Geometry pilot data member changed')
            files['data/geometry-a03/' + row[kind + '_path']] = path
    reserved = ROOT / '.runtime/strict-appearance-a05'
    for name in ('manifest.jsonl', 'QA.json'):
        files['data/strict-appearance/' + name] = reserved / name
    for line in (reserved / 'manifest.jsonl').read_text(encoding='utf8').splitlines():
        row = json.loads(line)
        for kind in ('image', 'mask'):
            file = reserved / row[kind + '_path']
            if digest(file) != row[kind + '_sha256']:
                raise ValueError('Reserved appearance member changed')
            files['data/strict-appearance/' + row[kind + '_path']] = file
    extra = {}
    with zipfile.ZipFile(board_archive) as board:
        for name in board.namelist():
            if '/' in name or '\\' in name:
                raise ValueError('Board package contains a non-flat member')
            extra['board/' + name] = board.read(name)
    experiment = 'source/experiments/' + ROOT.name
    start = (
        '# K230 时序神经网络离线包\n\n'
        '本包提供 CNN/TCN 模型、完整几何 Pilot、场地、Kaggle 记录、主机源码和 CanMV 试验入口。'
        '模型精度为 HOLD。真实 OV2735 和 K230 原生验收待进行。'
        '当前 TCN-8 浮点和 KModel 在 250 帧本地测试上的 P95 都约 2.21 米。\n\n'
        f'逐项证据见 [{experiment}/OFFLINE_STATUS.json]({experiment}/OFFLINE_STATUS.json)，'
        f'实物阶段统一清单见 [{experiment}/HUMAN_INPUTS.md]({experiment}/HUMAN_INPUTS.md)。\n\n'
        '在包根目录安装依赖并运行以下命令。编译需要另外安装 requirements_compiler.txt。\n\n'
        '```powershell\n'
        f'python -m pip install -r {experiment}/requirements_host.txt\n'
        f'python -m unittest discover -s {experiment} -v\n'
        f'python {experiment}/evaluate_neural_simulator.py --data data/geometry-a03 '
        '--models models/geometry-a03-models --backend models/compiled --setting tcn8 --out reproduced-kmodel.json\n'
        '```\n\n'
        '实拍入口 replay_capture.py 接受逐帧 RGB 索引，参数和格式见统一清单。'
        'board 目录中的 RUN.md 给出原生试验命令。'
        'support 目录包含云操作依赖，可使用其 collect_models.py 恢复精确命名的云权重。'
        'cloud 下的 run.ipynb 冻结了当时的训练源码，复现实验使用其中对应版本。\n'
    ).encode()
    extra['START_HERE.md'] = start
    inventory = {name: dict(bytes=path.stat().st_size, sha256=digest(path)) for name, path in files.items()}
    inventory.update({name: dict(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()) for name, blob in extra.items()})
    extra['SHA256SUMS.json'] = (json.dumps(inventory, indent=2, sort_keys=True) + '\n').encode()
    partial = out.with_suffix('.zip.partial')
    stream = partial.open('xb')
    try:
        with stream, zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as package:
            for name, file in sorted(files.items()):
                package.write(file, name)
            for name, blob in sorted(extra.items()):
                package.writestr(name, blob)
        with zipfile.ZipFile(partial) as package:
            if set(package.namelist()) != set(inventory) | {'SHA256SUMS.json'}:
                raise ValueError('Neural delivery member list differs')
            for name, entry in inventory.items():
                with package.open(name) as member:
                    sha = hashlib.file_digest(member, 'sha256').hexdigest()
                if sha != entry['sha256'] or package.getinfo(name).file_size != entry['bytes']:
                    raise ValueError('Neural delivery member readback failed')
        partial.rename(out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    receipt = dict(status='NEURAL_OFFLINE_DELIVERY_FULL_READBACK_PASS', path=str(out.resolve()),
                   bytes=out.stat().st_size, sha256=digest(out), members=len(inventory) + 1,
                   model_quality='HOLD', native='NOT_RUN', complete_project=False,
                   full_data_manifest_sha256=audit['manifest_sha256'], kmodel_sha256=report['kmodel_sha256'])
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('out', 'data', 'assets', 'reference', 'board', 'roadnet', 'cad', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    receipt = create(args.out, args.data, args.assets, args.reference, args.board, args.roadnet, args.cad)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8', newline='\n')
