"""Record actual fixed-window GRU importer results in an isolated process."""
import argparse
from importlib import metadata, util
import json
import os
from pathlib import Path
import time

import numpy as np

from evaluate_neural import clip, digest, load_sequences, read, verified_graph


def main(data, models, setting, out):
    sequences, manifest, _ = load_sequences(data)
    collection = read(models / 'MODEL_COLLECTION.json')
    session, input_name, _, metrics, _, _ = verified_graph(models, collection, setting, manifest)
    window = int(setting[-1])
    rgb = clip(sequences[0][1], 7, window, 'chronological')
    float_output, = session.run(None, {input_name: rgb})
    report = dict(status='NNCASE_IMPORT_STARTED', setting=setting, input_shape=list(rgb.shape),
                  onnx_sha256=metrics['onnx_sha256'], best_sha256=metrics['best_sha256'],
                  dataset_manifest_sha256=manifest, saved_version=collection['source'],
                  compiler={p: metadata.version(p) for p in ('nncase', 'nncase-kpu')},
                  onnx_runtime_finite=bool(np.isfinite(float_output).all()), native_k230='NOT_RUN')
    out.mkdir(parents=True, exist_ok=False)
    proof = out / 'REPORT.json'
    proof.write_text(json.dumps(report, indent=2), encoding='utf8')
    plugin = str(Path(util.find_spec('_nncase').origin).parent)
    os.environ['NNCASE_PLUGIN_PATH'] = plugin
    os.environ['PATH'] = plugin + os.pathsep + os.environ['PATH']
    dll = os.add_dll_directory(plugin) if os.name == 'nt' else None
    import nncase

    os.chdir(out)
    options = nncase.CompileOptions()
    options.target = 'k230'
    options.preprocess = False
    options.input_shape = list(rgb.shape)
    options.input_type = 'float32'
    options.dump_ir = options.dump_asm = False
    try:
        compiler = nncase.Compiler(options)
        compiler.import_onnx((models / setting / 'camera_xy.onnx').read_bytes(), nncase.ImportOptions())
        ptq = nncase.PTQTensorOptions()
        ptq.quant_type, ptq.w_quant_type = 'int16', 'uint8'
        train, _, _ = load_sequences(data, 'train')
        calibration = [clip(frames, 7, window, 'chronological') for _, frames in train[:5]]
        ptq.samples_count = len(calibration)
        ptq.set_tensor_data([[value] for value in calibration])
        compiler.use_ptq(ptq)
        compiler.compile()
        binary = compiler.gencode_tobytes()
        (out / 'camera_xy.kmodel').write_bytes(binary)
        simulator = nncase.Simulator()
        simulator.load_model(binary)
        simulator.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(rgb))
        before = time.perf_counter()
        simulator.run()
        actual = simulator.get_output_tensor(0).to_numpy()
        if not np.isfinite(actual).all() or actual.shape != float_output.shape:
            raise ValueError('GRU Simulator shape or output invalid')
        report.update(status='COMPILE_AND_SIMULATOR_PASS', host_simulator_seconds=time.perf_counter() - before,
                      max_abs=float(np.max(abs(actual - float_output))), kmodel_sha256=digest(out / 'camera_xy.kmodel'))
    except Exception as error:
        report.update(status='COMPILE_FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        proof.write_text(json.dumps(report, indent=2), encoding='utf8')
        if dll is not None:
            dll.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('data', 'models', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--setting', choices=('gru4', 'gru8'), required=True)
    args = parser.parse_args()
    main(args.data.resolve(), args.models.resolve(), args.setting, args.out.resolve())
