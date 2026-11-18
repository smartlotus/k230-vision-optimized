"""Build a readback-verified, isolated board trial for a verified neural KModel."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from evaluate_neural import digest, read, verified_graph


ROOT = Path(__file__).resolve().parent


def bench_source(source):
    start = source.index('def decode_map(')
    end = source.index('\n\ndef run(', start)
    source = source[:start] + (
        'def decode_map(values, centers, primary_cal, set_cal):\n'
        '    return decode_logits(values, centers, primary_cal["temperature"], primary_cal["threshold"])\n'
    ) + source[end:]
    replacements = {
        'import math\n': 'import math\nfrom neural_window import FrameWindow\nfrom neural_decode import decode_logits\n',
        'from reference.board_smoke import Latency, sample_memory': 'from board_support import Latency, sample_memory',
        'from k230.topface_adapter import digest': 'from board_support import digest',
        'from k230.topface_service import resources': 'from board_support import resources',
        '    needs_flip = map_centers is None and not identity["cell_mm"]':
            '    needs_flip = False\n    profile = deployment["neural_window"]\n    window = profile["length"]\n'
            '    if len(map_centers) != 2304:\n        raise ValueError("Expected 2304 camera XY anchors")',
        'identity["input_rgb_uint8_nchw"] != [1, 3, 96, 160]':
            'identity["input_rgb_uint8_nchw"] != [1, 3 * window, 96, 160]',
        '    ai2d = builder = kpu = backing = dst = None':
            '    ai2d = builder = kpu = backing = dst = history = window_input = None',
        '    started = None': '    started = None\n    trace = None',
        '    with open(model_root + "/deployment.json") as f:':
            '    for suffix in (".frames.jsonl", ".frames.jsonl.partial"):\n'
            '        try:\n            os.stat(report_path + suffix)\n'
            '        except OSError:\n            pass\n'
            '        else:\n            raise ValueError("Neural frame trace already exists")\n'
            '    with open(model_root + "/deployment.json") as f:',
        '        sample_memory(result, resources, "initial")':
            '        trace = open(report_path + ".frames.jsonl.partial", "w")\n'
            '        sample_memory(result, resources, "initial")',
        '        kpu.set_input_tensor(0, dst)':
            '        history = FrameWindow(np, window, profile["sample_us"], profile["max_gap_us"], time.ticks_diff)\n'
            '        window_input = nn.from_numpy(history.tensor)\n'
            '        kpu.set_input_tensor(0, window_input)\n'
            '        result.update(qualification="EXPERIMENTAL_ACCURACY_HOLD", window=window, '
            'sample_us=profile["sample_us"], capture_decimated=0, resets=0)\n'
            '        latencies["neural_window"] = Latency()',
        '            packed = image.to_rgb888(x_size=640, y_size=360)':
            '            if not history.due(captured):\n'
            '                result["capture_decimated"] += 1\n'
            '                latencies["cycle"].add(time.ticks_diff(time.ticks_us(), cycle))\n'
            '                continue\n'
            '            packed = image.to_rgb888(x_size=640, y_size=360)',
        '                kpu.run()':
            '                history.append(backing, captured)\n'
            '                result["resets"] = history.resets\n'
            '                result["last_window_valid_frames"] = history.count\n'
            '                after_pack = time.ticks_us()\n'
            '                latencies["neural_window"].add(time.ticks_diff(after_pack, before_kpu))\n'
            '                before_kpu = after_pack\n'
            '                kpu.run()',
        '            result["last_prediction"] = answer':
            '            result["last_prediction"] = answer\n'
            '            trace.write(json.dumps(dict(capture_index=index, capture_ticks_us=captured,\n'
            '                captured_since_start_us=time.ticks_diff(captured, started),\n'
            '                window_timestamps_us=history.tensor_timestamps_us,\n'
            '                window_valid_frames=history.count, prediction=answer)) + "\\n")\n'
            '            if index % 10 == 0:\n                trace.flush()',
        '        gc.collect()':
            '        if trace is not None:\n'
            '            try:\n                trace.close()\n'
            '                os.rename(report_path + ".frames.jsonl.partial", report_path + ".frames.jsonl")\n'
            '                result["frames_path"] = report_path + ".frames.jsonl"\n'
            '            except Exception as exc:\n'
            '                result["cleanup_errors"].append(["frame_trace", str(exc)])\n'
            '                result["status"] = "CLEANUP_UNCONFIRMED"\n'
            '        gc.collect()',
    }
    for old, new in replacements.items():
        expected = 2 if old == '    ai2d = builder = kpu = backing = dst = None' else 1
        if source.count(old) != expected:
            raise ValueError('CameraXY bench source changed: ' + old)
        source = source.replace(old, new)
    compile(source, 'neural_board_probe.py', 'exec')
    return source


def run(out, backend, models, proof, profile_path):
    report = read(proof)
    source_collection = read(models / 'MODEL_COLLECTION.json')
    compiler_proof = backend / 'KMODEL_REPORT.json'
    if (report['status'] != 'NEURAL_KMODEL_PAIRED_TEST_VERIFIED' or
            report['source'] != source_collection['source'] or
            digest(compiler_proof) != report['simulator_report_sha256']):
        raise ValueError('Board trial requires the paired neural Simulator test')
    setting = report['setting']
    _, _, centers, metrics, _, _ = verified_graph(models, source_collection, setting,
          report.get('training_manifest_sha256', report['dataset_manifest_sha256']))
    bin_path = backend / 'camera_xy.kmodel'
    uint8 = backend / 'camera_xy_uint8.onnx'
    if digest(bin_path) != report['kmodel_sha256'] or digest(uint8) != report['uint8_onnx_sha256']:
        raise ValueError('Compiled model bytes differ')
    board_base = ROOT.parent / 'camera_xy_localization_v2/board_probe.py'
    window = report['profile']['window']
    profile = read(profile_path)
    if (profile['window'] != window or profile['best_sha256'] != metrics['best_sha256']
            or profile['temperature'] != metrics['temperature']
            or profile['training_manifest_sha256'] != report.get('training_manifest_sha256', report['dataset_manifest_sha256'])
            or profile['threshold'] is not None):
        raise ValueError('Model requires its own verified HOLD runtime profile')
    unknown = dict(bin_edges=[], probabilities=[0.], threshold=None, temperature=metrics['temperature'])
    deployment = dict(identity=dict(input_rgb_uint8_nchw=[1, 3 * window, 96, 160], map_centers_mm=centers.tolist(),
                                    cell_mm=50, confidence=unknown, candidate_set_confidence=unknown),
                      neural_window=dict(length=window, sample_us=profile['sample_us'], max_gap_us=profile['max_gap_us']),
                      kmodel=dict(kmodel_sha256=digest(bin_path)), saved_version=source_collection['source'],
                      qualification='EXPERIMENTAL_ACCURACY_HOLD', native_k230='NOT_RUN')
    binaries = {'camera_xy.kmodel': bin_path, 'camera_xy_uint8.onnx': uint8,
                'camera_xy.onnx': models / setting / 'camera_xy.onnx', 'best.pt': models / setting / 'best.pt',
                'KMODEL_REPORT.json': compiler_proof, 'PAIRED_TEST.json': proof,
                'neural_window.py': ROOT / 'neural_window.py', 'board_support.py': ROOT / 'board_support.py',
                'neural_decode.py': ROOT / 'neural_decode.py', 'RUNTIME_PROFILE.json': profile_path}
    board_files = {name: path.read_bytes() for name, path in binaries.items()}
    board_files['neural_board_probe.py'] = bench_source(board_base.read_text(encoding='utf8')).encode()
    board_files['deployment.json'] = (json.dumps(deployment, indent=2) + '\n').encode()
    board_files['RUN.md'] = (
        '# K230 时序神经网络试验\n\n'
        '当前为精度 HOLD。试验程序输出候选，已验收的位置字段保持空值。'
        f'输入为最近 {window} 张按时间顺序采样的 RGB 帧。窗口节奏绑定 RUNTIME_PROFILE.json 的仿真训练时间戳。\n\n'
        '解压到独立 SD 卡目录，确认设备空闲。从 CanMV 解释器运行以下代码。\n\n'
        '```python\n'
        'import sys\nsys.path.insert(0, "/sdcard/temporal_neural_trial")\nimport neural_board_probe\n'
        'neural_board_probe.run("/sdcard/temporal_neural_trial", "/sdcard/neural_bench_a01.json", '
        '"实际 OV2735 USB ID", frames=1200)\n'
        '```\n\n'
        '`frames` 限制采集尝试次数。结果中的 `frames` 是完成了神经网络推理的样本数。'
        '报告记录采集与推理、窗口打包、跳采和长间隔重置、内存及完整周期。'
        '同名 .frames.jsonl 文件逐帧记录采集时间、窗口内时间及候选，可与实测 XY 对齐。\n').encode()
    inventory = {name: dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
                 for name, value in board_files.items()}
    board_files['MANIFEST.json'] = (json.dumps(inventory, indent=2) + '\n').encode()
    if out.exists():
        raise FileExistsError(out)
    partial = out.with_suffix('.zip.partial')
    stream = partial.open('xb')
    try:
        with stream, zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as package:
            for name, data in board_files.items():
                package.writestr(name, data)
        with zipfile.ZipFile(partial) as package:
            if set(package.namelist()) != set(board_files) or any(
                    package.read(name) != data for name, data in board_files.items()):
                raise ValueError('Neural board trial archive readback failed')
        partial.rename(out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return dict(status='NEURAL_BOARD_TRIAL_ARCHIVE_VERIFIED', archive=str(out.resolve()), archive_sha256=digest(out),
                archive_bytes=out.stat().st_size, members=len(board_files), setting=setting,
                kmodel_sha256=deployment['kmodel']['kmodel_sha256'], upstream_board_source_sha256=digest(board_base),
                checkpoint_sha256=metrics['best_sha256'], native_k230='NOT_RUN', accuracy='HOLD')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('out', 'backend', 'models', 'proof', 'profile', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.out, args.backend, args.models, args.proof, args.profile)
    args.receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps(result))
