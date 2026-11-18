"""Actual export/ORT and isolated nncase compile/Simulator smoke; random weights."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_one(root, route):
    plugin = str(Path(importlib.util.find_spec('_nncase').origin).parent)
    os.environ['NNCASE_PLUGIN_PATH'] = plugin
    os.environ['PATH'] = plugin + os.pathsep + os.environ.get('PATH', '')
    dll = os.add_dll_directory(plugin) if os.name == 'nt' else None
    import nncase
    report = json.loads((root / f'{route}-onnx.json').read_text())
    shape = report['input_shape']
    work = root / (route + '-compiler')
    work.mkdir(exist_ok=False)
    os.chdir(work)
    opts = nncase.CompileOptions()
    opts.target = 'k230'
    opts.preprocess = False
    opts.input_shape = shape
    opts.input_type = 'float32'
    opts.shape_bucket_enable = False
    opts.dump_ir = False
    opts.dump_asm = False
    opts.dump_dir = str(work / 'dump')
    compiler = nncase.Compiler(opts)
    compiler.import_onnx((root / f'{route}.onnx').read_bytes(), nncase.ImportOptions())
    rng = np.random.default_rng(117)
    calibration = [rng.random(shape, dtype=np.float32) for _ in range(5)]
    ptq = nncase.PTQTensorOptions()
    ptq.samples_count = len(calibration)
    ptq.calibrate_method = 'NoClip'
    ptq.quant_type, ptq.w_quant_type = 'uint8', 'uint8'
    ptq.set_tensor_data([[x] for x in calibration])
    compiler.use_ptq(ptq)
    compiler.compile()
    binary = compiler.gencode_tobytes()
    path = root / f'{route}.kmodel'
    path.write_bytes(binary)
    runtime = nncase.Simulator()
    runtime.load_model(binary)
    runtime.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(calibration[0]))
    start = time.perf_counter()
    runtime.run()
    latency = time.perf_counter() - start
    import onnxruntime as ort
    session = ort.InferenceSession(str(root / f'{route}.onnx'), providers=['CPUExecutionProvider'])
    expected = session.run(None, {'rgb_window': calibration[0]})
    pairs = []
    for i, ref in enumerate(expected):
        actual = runtime.get_output_tensor(i).to_numpy()
        if actual.shape != ref.shape or not np.isfinite(actual).all():
            raise ValueError('Simulator shape/nonfinite output')
        pairs.append({'shape': list(actual.shape), 'max_abs': float(abs(actual-ref).max()),
                      'mae': float(abs(actual-ref).mean())})
    write(root / f'{route}-compile.json', {'status': 'COMPILE_AND_SIMULATOR_PASS',
          'scope': report.get('scope','OPERATOR_COMPATIBILITY_SMOKE_ONLY'),
          'compiler': {p: importlib.metadata.version(p) for p in ('nncase', 'nncase-kpu')},
          'input_type': 'float32', 'preprocess': False, 'normalization': 'external RGB/255 ONCE',
          'onnx_sha256': report['onnx_sha256'], 'kmodel_sha256': sha(path),
          'kmodel_bytes': len(binary), 'host_simulator_seconds': latency, 'pairs': pairs,
          'native_k230': 'NOT RUN'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--compile-one')
    ap.add_argument('--height', type=int, default=64)
    ap.add_argument('--width', type=int, default=96)
    ap.add_argument('--model-module', default='models')
    args = ap.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.compile_one:
        compile_one(root, args.compile_one)
        return
    import torch
    import onnx
    import onnxruntime as ort
    from importlib import import_module
    CameraXYNet=import_module(args.model_module).CameraXYNet
    torch.set_num_threads(2)
    results = []
    for route, t in [('cnn', 1), ('gru', 4), ('tcn', 4), ('gru', 8), ('tcn', 8)]:
        name = f'{route}{t}'
        torch.manual_seed(117)
        model = CameraXYNet(route, t).eval()
        x = torch.rand(1, 3*t, args.height, args.width)
        path = root / f'{name}.onnx'
        torch.onnx.export(model, x, str(path), input_names=['rgb_window'],
                          output_names=['location_logits', 'xy_refinement'],
                          opset_version=13, dynamo=False)
        graph = onnx.load(path)
        onnx.checker.check_model(graph)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])
        with torch.inference_mode():
            ref = [v.numpy() for v in model(x)]
        actual = session.run(None, {'rgb_window': x.numpy()})
        for a, b in zip(actual, ref):
            np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-5)
        report = {'status': 'ONNX_ORT_PASS', 'scope': 'UNTRAINED_OPERATOR_SMOKE',
                  'route': route, 'window': t, 'parameters': sum(p.numel() for p in model.parameters()),
                  'input_shape': list(x.shape), 'output_shapes': [list(v.shape) for v in actual],
                  'opset': 13, 'operators': sorted({n.op_type for n in graph.graph.node}),
                  'onnx_sha256': sha(path),
                  'torch_ort_max_abs': [float(abs(a-b).max()) for a,b in zip(actual, ref)],
                  'versions': {p: importlib.metadata.version(p) for p in ('torch','onnx','onnxruntime')}}
        write(root / f'{name}-onnx.json', report)
        log = root / f'{name}-compile.log'
        with log.open('w', encoding='utf8') as f:
            try:
                run = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--out', str(root),
                                      '--compile-one', name], stdout=f, stderr=subprocess.STDOUT, timeout=240)
                code = run.returncode
            except subprocess.TimeoutExpired:
                code = 'TIMEOUT_240s'
        report['compile_exit_code'] = code
        if not (root / f'{name}-compile.json').exists():
            write(root / f'{name}-compile.json', {'status': 'FAILED', 'exit_code': code,
                  'log': log.name, 'native_k230': 'NOT RUN'})
        results.append(report)
        write(root / 'SUMMARY.json', results)
        print(json.dumps({'route': name, 'onnx': 'PASS', 'compile_exit_code': code}), flush=True)


if __name__ == '__main__':
    main()
