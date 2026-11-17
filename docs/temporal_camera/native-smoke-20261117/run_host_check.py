"""Actual released ONNX -> CPU filter smoke, with simulated images only."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'research/temporal_camera'))
from replay_camera import load_routes, replay

OUT = Path(__file__).resolve().parent
DELIVERY = Path('C:/Users/28389/Desktop/k230-project/deliverables')
DATA = DELIVERY / 'K230_Official_Gray_Map_And_1100_Frame_Data_R1.zip'
MODELS = DELIVERY / 'K230_Official_Gray_Map_Pilot_A06_Models.zip'
BOARD = DELIVERY / 'K230_Official_Gray_Map_A_TCN4_Board_Bench_R2.zip'


def verify_archive(path):
    receipt = json.loads(path.with_suffix('.SHA256.json').read_text(encoding='utf8'))
    raw = path.read_bytes()
    if len(raw) != receipt['bytes'] or hashlib.sha256(raw).hexdigest() != receipt['sha256']:
        raise ValueError('Delivery archive differs: ' + path.name)
    return receipt['sha256']


def main():
    if (OUT / 'host-model-replay.json').exists() or (OUT / 'simulated-fixture').exists():
        raise FileExistsError('Preserve completed host smoke; do not repeat')
    sources = {p.name: verify_archive(p) for p in (DATA, MODELS, BOARD)}
    fixture = OUT / 'simulated-fixture'
    fixture.mkdir()
    members = []
    with zipfile.ZipFile(DATA) as bundle:
        inventory = json.loads(bundle.read('SHA256_MANIFEST.json'))
        names = sorted(n for n in inventory if '/rgb/' in n and n.endswith('.png'))[:4]
        if len(names) != 4:
            raise ValueError('Expected four delivered simulated RGB images')
        for index, name in enumerate(names):
            raw = bundle.read(name)
            record = inventory[name]
            digest = hashlib.sha256(raw).hexdigest()
            if len(raw) != record['bytes'] or digest != record['sha256']:
                raise ValueError('Dataset image differs')
            filename = f'frame{index}.png'
            (fixture / filename).write_bytes(raw)
            members.append(dict(archive_member=name, fixture_name=filename, sha256=digest))
    rows = []
    sequence = [0, 1, None, None, None, None, None, 2, 3]
    for index, frame in enumerate(sequence):
        row = dict(trajectory_id='simulated-engineering-smoke', frame_index=index,
                   timestamp_s=index * .04, image_path=None)
        if frame is not None:
            row.update(image_path=members[frame]['fixture_name'], image_sha256=members[frame]['sha256'])
        rows.append(row)
    manifest = fixture / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
    routes, anchors, source = load_routes(MODELS, BOARD, ['cnn1', 'tcn4'])
    report = replay(manifest, OUT / 'host-model-replay.json', routes, anchors, source, 'simulated')
    summaries = {}
    for name, result in report['routes'].items():
        summary = result['summary']
        assert (summary['capture_attempts'], summary['observed_frames'], summary['missing_captures']) == (9, 4, 5)
        assert summary['lost_updates'] == 1 and summary['global_recovery_events'] == 1
        assert summary['history_resets'] == (1 if name == 'tcn4' else 0)
        assert summary['surveyed_filtered']['n'] == 0 and summary['start_zone_trials'] == 0
        frames = result['trajectories']['simulated-engineering-smoke']['frames']
        assert all(frame['position_valid'] is False for frame in frames)
        summaries[name] = summary
    persisted = json.loads((OUT / 'host-model-replay.json').read_text(encoding='utf8'))
    assert persisted == report
    result = dict(status='HOST_MODEL_FILTER_REPORT_READBACK_PASS', input_source='SIMULATED_DELIVERY_IMAGES',
                  archives=sources, images=members, routes=summaries, json_report_readback='PASS',
                  native_k230='NOT_RUN', physical_camera_frames=0, accuracy='NOT_RUN', road_test='NOT_RUN',
                  onnx_runtime_provider='CPUExecutionProvider', quantized_kmodel_runtime='NOT_RUN')
    (OUT / 'HOST_MODEL_PIPELINE.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
