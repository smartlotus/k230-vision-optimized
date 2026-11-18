"""Replay archived native AI2D bytes and PIL tensors through the same TCN-8."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

from capture_pixels import ai2d_640
from compare_capture import load_model_pair, read
from neural_decode import decode_logits
from replay_capture import sha as digest


FIXTURES = ('constant', 'rgb_ramp', 'thin_lines', 'checkerboard', 'heldout')
FRAME_SHAPE = (3, 96, 160)
WINDOW = 8


def archive_member(directory, inventory, name):
    path = directory / name
    if digest(path) != inventory[name]:
        raise ValueError(f'Archived native evidence differs: {name}')
    return path


def load_fixtures(directory):
    inventory = read(directory / 'SHA256SUMS.json')
    spec_path = archive_member(directory, inventory, 'inputs-after/spec.json')
    native_path = archive_member(directory, inventory, 'final/native.json')
    spec, native = read(spec_path), read(native_path)
    if (native['status'] != 'AI2D_AND_DECODE_COMPLETE' or native['spec_sha256'] != digest(spec_path)
            or native['source_sha256'] != spec['board_sources'] or native['cleanup_errors']
            or native['camera'] != 'NOT_RUN' or native['kpu'] != 'NOT_RUN' or not native['startup_unchanged']
            or spec['input_shape'] != [1, 3, 360, 640] or spec['output_shape'] != [1, *FRAME_SHAPE]):
        raise ValueError('Unexpected archived native AI2D input contract')
    if (tuple(row['name'] for row in spec['fixtures']) != FIXTURES
            or tuple(row['name'] for row in native['preprocessing']) != FIXTURES):
        raise ValueError('The archived fixture denominator differs')
    frames, reports = {}, {}
    for fixture, measured in zip(spec['fixtures'], native['preprocessing'], strict=True):
        name = fixture['name']
        golden_path = archive_member(directory, inventory, f'inputs-after/{name}.golden')
        actual_path = archive_member(directory, inventory, f'final/{name}.ai2d.rgb')
        if (fixture['input_sha256'] != measured['input_sha256']
                or digest(golden_path) != fixture['golden_sha256']
                or digest(actual_path) != measured['output_sha256']
                or measured['output'] != actual_path.name
                or measured['output_bytes'] != actual_path.stat().st_size):
            raise ValueError(f'PIL or native tensor receipt differs: {name}')
        golden, actual = (np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(FRAME_SHAPE)
                          for path in (golden_path, actual_path))
        for frame in (golden, actual):
            if not (np.all(frame[:, :3] == 114) and np.all(frame[:, -3:] == 114)):
                raise ValueError(f'Letterbox padding differs: {name}')
        delta = np.abs(golden.astype(np.int16) - actual.astype(np.int16))
        frames[name] = (golden, actual)
        reports[name] = dict(values=delta.size, different=int(np.count_nonzero(delta)),
                             mean_abs=float(delta.mean()), max_abs=int(delta.max()),
                             input_sha256=fixture['input_sha256'], pil_sha256=fixture['golden_sha256'],
                             native_sha256=measured['output_sha256'])
    source = dict(native_report_sha256=digest(native_path), spec_sha256=digest(spec_path),
                  firmware=native['firmware'], runtime=native['runtime'], board_sources=spec['board_sources'])
    return frames, reports, source


def compare_outputs(pil, ai2d, centers, temperature):
    outputs = [decode_logits(logits.tolist(), centers, temperature) for logits in (pil, ai2d)]
    candidates = [answer['top_k_camera_xy'] for answer in outputs]
    positions = [tuple(candidate[0][key] for key in ('camera_x_mm', 'camera_y_mm')) for candidate in candidates]
    delta = np.abs(ai2d - pil)
    return dict(output_values=len(delta), logit_mean_abs=float(delta.mean()), logit_max_abs=float(delta.max()),
                first_candidate_shift_mm=math.dist(*positions), ordered_top5_equal=candidates[0] == candidates[1],
                pil_top5=candidates[0], native_ai2d_top5=candidates[1],
                pil_peak_mass=outputs[0]['visual_peak_mass'], native_ai2d_peak_mass=outputs[1]['visual_peak_mass'])


def run(evidence, models, profile_path, out):
    source, profile, model_metrics = load_model_pair(models, profile_path)
    metrics = model_metrics['tcn8']
    if metrics['best_sha256'] != profile['best_sha256'] or metrics['temperature'] != profile['temperature']:
        raise ValueError('Model and runtime profile differ')
    model = (models / 'tcn8/camera_xy.onnx').read_bytes()
    if hashlib.sha256(model).hexdigest() != metrics['onnx_sha256']:
        raise ValueError('Original TCN-8 graph changed after lineage verification')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(model, sess_options=options, providers=['CPUExecutionProvider'])
    input_, = session.get_inputs()
    output, = session.get_outputs()
    if input_.shape != [1, 24, 96, 160] or input_.type != 'tensor(float)' or output.shape != [1, 2304]:
        raise ValueError('Unexpected TCN-8 graph interface')
    axis = range(-1175, 1200, 50)
    centers = [[x, y] for y in axis for x in axis]
    frames, fixture_reports, native = load_fixtures(evidence)
    cases = {}
    for fixture in FIXTURES:
        for positions in (tuple(range(WINDOW)), (0,), (WINDOW - 1,)) if fixture == 'heldout' else (tuple(range(WINDOW)),):
            label = fixture + ('_all' if len(positions) == WINDOW else f'_slot_{positions[0]}')
            golden, actual = frames[fixture]
            pil = np.concatenate([golden] * WINDOW)[None]
            mixed = np.concatenate([actual if i in positions else golden for i in range(WINDOW)])[None]
            outputs = [session.run(None, {input_.name: value.astype(np.float32) / 255})[0][0]
                       for value in (pil, mixed)]
            cases[label] = dict(compare_outputs(*outputs, centers, metrics['temperature']),
                                fixture=fixture, slots_replaced=list(positions))
    result = dict(status='ARCHIVED_NATIVE_AI2D_TCN8_HOST_REPLAY_COMPLETE', native_evidence=native,
                  model_source=source, onnx_sha256=metrics['onnx_sha256'], checkpoint_sha256=metrics['best_sha256'],
                  profile_sha256=digest(profile_path), evaluator_sha256=digest(Path(__file__)),
                  input_shape=[1, 24, 96, 160], fixtures=fixture_reports, cases=cases,
                  localization_accuracy='UNSCORED_NO_OPTICAL_CENTER_GT', temporal_motion='NOT_RUN_REPEATED_STILL_INPUT',
                  native_camera='NOT_RUN', native_temporal_kpu='NOT_RUN', qualification='HOLD')
    with out.open('x', encoding='utf8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({label: (entry['first_candidate_shift_mm'], entry['logit_max_abs'],
                              entry['ordered_top5_equal']) for label, entry in cases.items()}))


def verify_resize(evidence, out):
    from importlib.util import module_from_spec, spec_from_file_location

    frames, _, source = load_fixtures(evidence)
    inventory = read(evidence / 'SHA256SUMS.json')
    original = Path(__file__).resolve().parents[1] / 'camera_xy_localization_v2/board_tensor_bench.py'
    if digest(original) != source['board_sources']['board_tensor_bench.py']:
        raise ValueError('Procedural fixture source differs from the measured run')
    module_spec = spec_from_file_location('archived_board_tensor_bench', original)
    module = module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    spec = read(evidence / 'inputs-after/spec.json')
    results = []
    for fixture in spec['fixtures']:
        name = fixture['name']
        if name == 'heldout':
            path = archive_member(evidence, inventory, 'inputs-after/heldout.input.rgb')
            raw = path.read_bytes()
        else:
            raw = module.fixture(name)
        if hashlib.sha256(raw).hexdigest() != fixture['input_sha256']:
            raise ValueError(f'Original AI2D input differs: {name}')
        host = ai2d_640(np.frombuffer(raw, np.uint8).reshape(3, 360, 640))
        actual = frames[name][1]
        if not np.array_equal(host, actual):
            raise ValueError(f'Host AI2D mirror differs from measured output: {name}')
        results.append(dict(name=name, exact_values=host.size, input_sha256=fixture['input_sha256'],
                            output_sha256=hashlib.sha256(host.tobytes()).hexdigest()))
    result = dict(status='HOST_MIRROR_MATCHES_ARCHIVED_NATIVE_AI2D_BYTES', native_evidence=source,
                  implementation_sha256=digest(Path(__file__).with_name('capture_pixels.py')),
                  fixtures=results, exact_values=sum(row['exact_values'] for row in results),
                  scope='Fixed 640x360 planar uint8 intermediate to 160x96 planar uint8 output',
                  uvc_decode='NOT_RUN', native_temporal_kpu='NOT_RUN', real_accuracy='NOT_RUN')
    with out.open('x', encoding='utf8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('evidence', 'models', 'profile', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--resize-out', type=Path)
    args = parser.parse_args()
    run(args.evidence, args.models, args.profile, args.out)
    if args.resize_out is not None:
        print(json.dumps(verify_resize(args.evidence, args.resize_out)))
