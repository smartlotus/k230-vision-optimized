"""Qualify the 16-frame causal graph before committing its cloud training run."""
import argparse
from importlib import metadata, util
import json
import os
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch

from compile_neural import add_uint8_input
from evaluate_neural import digest
from train_neural import CameraXY


def run(parent, out, window=16, head_type='linear'):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(117)
    state = torch.load(parent, weights_only=True, map_location='cpu')
    model = CameraXY(window, state['state'], head_type).eval()
    rgb = torch.rand(1, 3 * window, 96, 160)
    with torch.inference_mode():
        expected = model(rgb).numpy()
    graph = out / 'tcn16.onnx'
    torch.onnx.export(model, rgb, str(graph), input_names=['rgb_window'],
                      output_names=['location_logits'], opset_version=13, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(graph), sess_options=options, providers=['CPUExecutionProvider'])
    actual, = session.run(None, {'rgb_window': rgb.numpy()})
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
    u8 = out / 'tcn16_uint8.onnx'
    name = add_uint8_input(graph, 'rgb_window', u8, [1, 3 * window, 96, 160])
    session = ort.InferenceSession(str(u8), sess_options=options, providers=['CPUExecutionProvider'])
    pixel = np.rint(rgb.numpy() * 255).astype(np.uint8)
    original, = session.run(None, {name: pixel})
    plugin = str(Path(util.find_spec('_nncase').origin).parent)
    os.environ['NNCASE_PLUGIN_PATH'] = plugin
    os.environ['PATH'] = plugin + os.pathsep + os.environ['PATH']
    dll = os.add_dll_directory(plugin) if os.name == 'nt' else None
    import nncase

    os.chdir(out)
    opts = nncase.CompileOptions()
    opts.target, opts.preprocess, opts.input_type = 'k230', False, 'uint8'
    opts.input_shape = [1, 3 * window, 96, 160]
    opts.dump_asm = opts.dump_ir = False
    compiler = nncase.Compiler(opts)
    compiler.import_onnx(u8.read_bytes(), nncase.ImportOptions())
    ptq = nncase.PTQTensorOptions()
    ptq.samples_count = 3
    ptq.quant_type, ptq.w_quant_type = 'int16', 'uint8'
    ptq.calibrate_method = 'NoClip'
    ptq.set_tensor_data([[pixel] for _ in range(3)])
    compiler.use_ptq(ptq)
    compiler.compile()
    path = out / 'tcn16.kmodel'
    binary = compiler.gencode_tobytes()
    path.write_bytes(binary)
    sim = nncase.Simulator()
    sim.load_model(binary)
    sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(pixel))
    start = time.perf_counter()
    sim.run()
    quantized = sim.get_output_tensor(0).to_numpy()
    if quantized.shape != expected.shape or not np.isfinite(quantized).all():
        raise ValueError('Long-window KModel interface failed')
    record = dict(status='TCN_OPERATOR_AND_SIMULATOR_PASS', input_shape=[1, 3 * window, 96, 160],
                  output_shape=list(expected.shape), scope='RANDOM_HEAD_OPERATOR_SMOKE',
                  head_type=head_type, dilation=model.dilations, receptive_frames=window, parent_sha256=digest(parent),
                  source_sha256=digest(Path(__file__)), model_source_sha256=digest(Path(__file__).with_name('train_neural.py')),
                  torch_ort_max_abs=float(np.max(abs(actual - expected))),
                  ort_simulator_max_abs=float(np.max(abs(quantized - original))),
                  compiler={p: metadata.version(p) for p in ('nncase', 'nncase-kpu')},
                  onnx_sha256=digest(graph), uint8_onnx_sha256=digest(u8), kmodel_sha256=digest(path),
                  kmodel_bytes=len(binary), host_simulator_s=time.perf_counter() - start, native_k230='NOT_RUN')
    (out / 'SMOKE.json').write_text(json.dumps(record, indent=2), encoding='utf8')
    print(json.dumps(record), flush=True)
    if dll is not None:
        dll.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--window', type=int, choices=(8, 16), default=16)
    parser.add_argument('--head', choices=('linear', 'mixture'), default='linear')
    args = parser.parse_args()
    run(args.parent.resolve(), args.out, args.window, args.head)
