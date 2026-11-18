"""Compile a verified packed-window ONNX graph and simulate every held-out endpoint."""
import argparse
from importlib import metadata, util
import json
import os
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import torch

from evaluate_neural import SETTINGS, TARGETS, clip, digest, load_sequences, model_manifest_identity, read, verified_graph


def add_uint8_input(float_path, input_name, output, shape):
    model = onnx.load(float_path)
    graph = model.graph
    entry = 'rgb_window_uint8'
    prefix = [
        helper.make_node('Cast', [entry], ['camera_rgb_f32'], to=TensorProto.FLOAT),
        helper.make_node('Div', ['camera_rgb_f32', 'camera_rgb_255'], [input_name]),
    ]
    graph.input[0].CopyFrom(helper.make_tensor_value_info(entry, TensorProto.UINT8, shape))
    for index, node in enumerate(prefix):
        graph.node.insert(index, node)
    graph.initializer.append(helper.make_tensor('camera_rgb_255', TensorProto.FLOAT, [], [255.]))
    onnx.checker.check_model(model)
    onnx.save(model, output)
    return entry


def inputs(sequences, window):
    for rows, frames in sequences:
        for i, row in enumerate(rows):
            yield row, np.rint(clip(frames, i, window, 'chronological') * 255).astype(np.uint8)


def run(data, models, setting, out, lower_gru=False, stable_lse=False):
    torch.set_num_threads(2)
    if lower_gru and stable_lse:
        raise ValueError('Choose one explicitly verified ONNX lowering')
    out = out.resolve()
    route, window = SETTINGS[setting]
    train, manifest, _ = load_sequences(data, 'train')
    test, test_manifest, _ = load_sequences(data)
    if manifest != test_manifest:
        raise ValueError('Trajectory manifest changed between train and test reads')
    collection = read(models / 'MODEL_COLLECTION.json')
    training_manifest = model_manifest_identity(data, models, setting, manifest)
    float_session, float_input, centers, metrics, parent, _ = verified_graph(models, collection, setting, training_manifest)
    shape = [1, window * 3, 96, 160]
    out.mkdir(parents=True, exist_ok=False)
    float_path = models / setting / 'camera_xy.onnx'
    lowering = None
    if lower_gru:
        if route != 'gru':
            raise ValueError('Fixed GRU lowering requires an original GRU setting')
        from lower_fixed_gru import lower
        float_path = out / 'camera_xy_unrolled.onnx'
        lowering = lower(models / setting / 'camera_xy.onnx', float_path, window)
    elif stable_lse:
        from stable_logsumexp import lower
        float_path = out / 'camera_xy_stable.onnx'
        lowering = lower(models / setting / 'camera_xy.onnx', float_path)
    if lowering is not None:
        lowering.update(source_sha256=metrics['onnx_sha256'], derivative_sha256=digest(float_path))
    uint8_path = out / 'camera_xy_uint8.onnx'
    uint8_input = add_uint8_input(float_path, float_input, uint8_path, shape)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    quant_source = ort.InferenceSession(str(uint8_path), sess_options=options,
                                        providers=['CPUExecutionProvider'])
    calibration = [(row, rgb) for i, (row, rgb) in enumerate(inputs(train, window)) if i % 5 == 0]
    for _, rgb in calibration[:3]:
        original, = float_session.run(None, {float_input: rgb.astype(np.float32) / 255})
        wrapped, = quant_source.run(None, {uint8_input: rgb})
        np.testing.assert_allclose(original, wrapped, atol=1e-4, rtol=1e-3)

    versions = {p: metadata.version(p) for p in ('nncase', 'nncase-kpu')}
    if set(versions.values()) != {'2.11.0'}:
        raise ValueError('Use the qualified K230 compiler pair')
    plugins = str(Path(util.find_spec('_nncase').origin).parent)
    os.environ['NNCASE_PLUGIN_PATH'] = plugins
    os.environ['PATH'] = plugins + os.pathsep + os.environ['PATH']
    dll = os.add_dll_directory(plugins) if os.name == 'nt' else None
    import nncase

    os.chdir(out)
    opts = nncase.CompileOptions()
    opts.target = 'k230'
    opts.preprocess = False
    opts.input_type = 'uint8'
    opts.input_shape = shape
    opts.dump_ir = opts.dump_asm = False
    compiler = nncase.Compiler(opts)
    compiler.import_onnx(uint8_path.read_bytes(), nncase.ImportOptions())
    ptq = nncase.PTQTensorOptions()
    ptq.samples_count = len(calibration)
    ptq.quant_type = 'int16'
    ptq.w_quant_type = 'uint8'
    ptq.calibrate_method = 'NoClip'
    ptq.set_tensor_data([[rgb] for _, rgb in calibration])
    compiler.use_ptq(ptq)
    compiler.compile()
    path = out / 'camera_xy.kmodel'
    binary = compiler.gencode_tobytes()
    path.write_bytes(binary)
    simulator = nncase.Simulator()
    simulator.load_model(binary)
    scores, pairs = [], []
    lowering_max_abs = 0.
    for row, rgb in inputs(test, window):
        float_scores, = quant_source.run(None, {uint8_input: rgb})
        if lowering is not None:
            original, = float_session.run(None, {float_input: rgb.astype(np.float32) / 255})
            np.testing.assert_allclose(float_scores, original, atol=1e-4, rtol=1e-3)
            lowering_max_abs = max(lowering_max_abs, float(np.max(abs(float_scores - original))))
        simulator.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(rgb))
        start = time.perf_counter()
        simulator.run()
        elapsed = time.perf_counter() - start
        quant = simulator.get_output_tensor(0).to_numpy().copy()
        if quant.shape != float_scores.shape or not np.isfinite(quant).all():
            raise ValueError('Simulator output differs from finite XY logit contract')
        target = np.asarray([row['camera_xy_target'][key] for key in TARGETS])
        qa, fa = int(quant.argmax()), int(float_scores.argmax())
        scores.append(quant[0])
        pairs.append(dict(sample_id=row['sample_id'], anchor_same=qa == fa,
                          onnx_error_mm=float(np.linalg.norm(centers[fa] - target)),
                          kmodel_error_mm=float(np.linalg.norm(centers[qa] - target)),
                          host_simulator_seconds=elapsed))
        if len(pairs) % 25 == 0:
            print(json.dumps(dict(evaluated=len(pairs))), flush=True)
    np.savez_compressed(out / 'SIMULATOR_SCORES.npz', sample_ids=np.array([p['sample_id'] for p in pairs]),
                        logits=np.stack(scores))
    report = dict(status='COMPILE_AND_SIMULATOR_PASS', route=route, window=window, input_shape=shape,
                  input_type='uint8', compiler=versions, dataset_manifest_sha256=manifest, parent_sha256=parent,
                  training_manifest_sha256=training_manifest,
                  float_onnx_sha256=metrics['onnx_sha256'], checkpoint_sha256=metrics['best_sha256'],
                  uint8_onnx_sha256=digest(uint8_path), kmodel_sha256=digest(path), kmodel_bytes=path.stat().st_size,
                  normalization='Cast and divide by 255 inside graph; compiler preprocessing disabled',
                  ptq=dict(activation='int16', samples=len(calibration), ids=[row['sample_id'] for row, _ in calibration]),
                  evaluation_split='test', evaluated_frames=len(pairs), pairs=pairs,
                  full_evaluation_split_frames=sum(len(rows) for rows, _ in test), native_k230='NOT_RUN')
    if lowering is not None:
        report['onnx_lowering'] = dict(**lowering, original_paired_frames=len(pairs), max_abs=lowering_max_abs)
    (out / 'KMODEL_REPORT.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    if dll is not None:
        dll.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('data', 'models', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--setting', choices=SETTINGS, required=True)
    parser.add_argument('--lower-gru', action='store_true')
    parser.add_argument('--stable-lse', action='store_true')
    args = parser.parse_args()
    run(args.data, args.models, args.setting, args.out, args.lower_gru, args.stable_lse)
