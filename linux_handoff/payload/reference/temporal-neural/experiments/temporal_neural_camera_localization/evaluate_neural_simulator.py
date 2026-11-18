"""Verify and compare a compiled fixed-window model on the entire paired test split."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_neural import SETTINGS, TARGETS, digest, load_sequences, model_manifest_identity, read, statistics, verified_graph


def run(data, models, backend, setting, out):
    torch.set_num_threads(2)
    sequences, manifest_sha, field_sha = load_sequences(data)
    collection = read(models / 'MODEL_COLLECTION.json')
    route, window = SETTINGS[setting]
    trained = model_manifest_identity(data, models, setting, manifest_sha)
    _, _, centers, calibration, parent, _ = verified_graph(models, collection, setting, trained)
    report_path = backend / 'KMODEL_REPORT.json'
    compiled = read(report_path)
    expected = dict(status='COMPILE_AND_SIMULATOR_PASS', route=route, window=window, input_type='uint8',
                    input_shape=[1, window * 3, 96, 160], dataset_manifest_sha256=manifest_sha,
                    parent_sha256=parent, float_onnx_sha256=calibration['onnx_sha256'],
                    checkpoint_sha256=calibration['best_sha256'], evaluation_split='test')
    if any(compiled[k] != v for k, v in expected.items()):
        raise ValueError('Simulator lineage or fixed-window graph differs')
    if compiled.get('training_manifest_sha256', manifest_sha) != trained:
        raise ValueError('Simulator training-side manifest differs')
    for filename, key in (('camera_xy_uint8.onnx', 'uint8_onnx_sha256'), ('camera_xy.kmodel', 'kmodel_sha256')):
        if digest(backend / filename) != compiled[key]:
            raise ValueError('Compiled model member differs')
    training = {row['sample_id'] for line in (data / 'manifest.jsonl').read_text(encoding='utf8').splitlines()
                if (row := json.loads(line))['split'] == 'train'}
    ptq = compiled['ptq']
    if not set(ptq['ids']) <= training or ptq['samples'] != len(set(ptq['ids'])):
        raise ValueError('Quantization must use training samples only')
    rows = [row for segment, _ in sequences for row in segment]
    ids = [row['sample_id'] for row in rows]
    pairs = compiled['pairs']
    arrays_path = backend / 'SIMULATOR_SCORES.npz'
    with np.load(arrays_path, allow_pickle=False) as arrays:
        scores = arrays['logits']
        if (arrays['sample_ids'].tolist() != ids or [pair['sample_id'] for pair in pairs] != ids
                or scores.shape != (len(ids), len(centers)) or not np.isfinite(scores).all()
                or compiled['evaluated_frames'] != len(ids) or compiled['full_evaluation_split_frames'] != len(ids)):
            raise ValueError('Simulator must cover exactly the frozen test endpoints')
    xyz = centers[scores.argmax(1)]
    truth = np.asarray([[row['camera_xy_target'][key] for key in TARGETS] for row in rows])
    errors = np.linalg.norm(xyz - truth, axis=1)
    if not np.allclose(errors, [pair['kmodel_error_mm'] for pair in pairs], atol=1e-6, rtol=0):
        raise ValueError('Per-frame Simulator camera XY disagrees with report')
    temperature = calibration['temperature']
    probability = 1 / np.exp((scores - scores.max(axis=1, keepdims=True)) / temperature).sum(axis=1)
    quantized = []
    for i, row in enumerate(rows):
        jump = float(np.linalg.norm(xyz[i] - xyz[i - 1])) if row['frame_index'] else 0.
        quantized.append(dict(error_mm=float(errors[i]), confidence=float(probability[i]), jump_mm=jump))
    def errors_summary(values):
        return dict(median_mm=float(np.median(values)), p95_mm=float(np.quantile(values, .95)))

    warmed = np.array([row['frame_index'] >= window - 1 for row in rows])
    float_errors = np.array([pair['onnx_error_mm'] for pair in pairs])
    status = dict(status='NEURAL_KMODEL_PAIRED_TEST_VERIFIED', setting=setting, test_frames=len(ids),
                  profile=dict(window=window, input_shape=compiled['input_shape'], input_dtype='uint8',
                               normalization=compiled['normalization'], compiler=compiled['compiler'], ptq=ptq['activation']),
                  source=collection['source'], dataset_manifest_sha256=manifest_sha, field_mask_sha256=field_sha,
                  training_manifest_sha256=trained,
                  quantized=statistics(quantized), float_onnx=errors_summary(float_errors),
                  common_warm=dict(frames=int(warmed.sum()), quantized=errors_summary(errors[warmed]),
                                   float_onnx=errors_summary(float_errors[warmed])),
                  anchor_agreement=float(np.mean([pair['anchor_same'] for pair in pairs])),
                  simulator_host_p95_ms=float(np.quantile([pair['host_simulator_seconds'] * 1000 for pair in pairs], .95)),
                  simulator_report_sha256=digest(report_path), simulator_scores_sha256=digest(arrays_path),
                  kmodel_sha256=compiled['kmodel_sha256'], uint8_onnx_sha256=compiled['uint8_onnx_sha256'],
                  raw_onnx_sha256=compiled['float_onnx_sha256'], native_k230='NOT_RUN')
    out.write_text(json.dumps(status, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps({k: status[k] for k in ('setting', 'test_frames', 'quantized', 'float_onnx', 'anchor_agreement')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('data', 'models', 'backend', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--setting', choices=SETTINGS, required=True)
    args = parser.parse_args()
    run(args.data, args.models, args.backend, args.setting, args.out)
