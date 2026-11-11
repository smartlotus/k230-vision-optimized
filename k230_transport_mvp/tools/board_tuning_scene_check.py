"""Fresh scene gate with same-frame reference trajectories; never saves tuning."""
import json
import os
import sys

ROOT = '/sdcard/transport_mvp/'


def validate_request(request, profile):
    from k230.tuning.schema import validate
    validate(profile)
    required = {'attempt', 'scene_id', 'profile_file', 'profile_id', 'colors', 'centers', 'reference_id'}
    if set(request) != required:
        raise ValueError('Explicit scene gate request required')
    for key in ('attempt', 'scene_id', 'profile_file'):
        value = request[key]
        stem = value[:-5] if key == 'profile_file' and isinstance(value, str) and value.endswith('.json') else value
        if key == 'profile_file' and (not isinstance(value, str) or not value.endswith('.json')):
            raise ValueError('JSON profile filename required')
        if not isinstance(stem, str) or not 1 <= len(stem) <= 90 or any(
                c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in stem):
            raise ValueError('Invalid ' + key)
    colors = request['colors']
    if not isinstance(colors, list) or not colors or len(set(colors)) != len(colors) or any(
            type(cid) is not int or not 1 <= cid <= 6 for cid in colors):
        raise ValueError('Unique color IDs1..6 required')
    ref = request['reference_id']
    if type(ref) is not int or not 1 <= ref <= 6:
        raise ValueError('Reference color ID required')
    if request['profile_id'] != profile['profile_id'] or profile['locked']:
        raise ValueError('Unchanged unlocked candidate required')
    b = profile['binding']
    if (b['width'], b['height']) != (640, 360) or b['pose_id'] is not None:
        raise ValueError('Explicit uncalibrated640x360 required')
    ids = set(colors + [ref])
    if set(request['centers']) != {str(cid) for cid in ids}:
        raise ValueError('Manual centers for each tested/reference color required')
    for cid in ids:
        point = request['centers'][str(cid)]
        if not isinstance(point, list) or len(point) != 2 or any(type(v) is not int for v in point):
            raise ValueError('Integer manual center required')
        if not 12 <= point[0] < 628 or not 12 <= point[1] < 348:
            raise ValueError('Manual center outside source image')
        if profile['contexts']['TOP_BUFFER']['thresholds'][str(cid)] is None:
            raise ValueError('Measured threshold required')


def gate(rows, colors, camera_errors, initialization_errors, invalid_reason):
    return not invalid_reason and not camera_errors and not initialization_errors and set(rows) == {str(c) for c in colors} and all(
        r['frames'] == 100 and r['found'] == 100 and r['stable'] >= 95 and
        r['inside_manual_face_region'] == 100 and r['detector_errors'] == 0 for r in rows.values())


def main():
    if sys.implementation.name != 'micropython':
        raise SystemExit('Use CanMV on K230.')
    import time
    sys.path.insert(0, ROOT)
    sys.path.insert(0, ROOT + 'checks')
    from board_tuning_green_compare import save_report
    from k230.tuning.runtime import TuningApp
    from k230.tuning.schema import clone, compile_config
    from k230.vision_adapter import VisionAdapter
    with open(ROOT + 'scene_check_request.json') as stream:
        request = json.load(stream)
    name = request.get('profile_file', '')
    # Validate the filename before using it to read any board path.
    if not isinstance(name, str) or not name.endswith('.json') or not name[:-5] or any(
            c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name[:-5]):
        raise ValueError('Invalid profile filename')
    with open(ROOT + name) as stream:
        profile = json.load(stream)
    validate_request(request, profile)
    report = ROOT + 'scene_check_' + request['attempt'] + '.json'
    for name in (report, report + '.tmp'):
        try:
            os.stat(name)
        except OSError as error:
            if error.errno != 2:
                raise
        else:
            raise RuntimeError('Consumed attempt; preserve prior evidence')
    app = TuningApp()
    s = app.session
    s.draft = clone(profile)
    s.dirty = True
    s.apply()
    s.select('TOP', 'BUFFER')
    colors = request['colors']
    current = 0
    s.target_id = colors[current]
    ref_adapter = VisionAdapter(compile_config(profile))
    guard = (clone(s.active), clone(s.draft), s.slot, s.apply_serial)
    rows = {}
    last = -1
    done = False
    try:
        app.setup()
        app.panel.live_requested = True
        started = time.ticks_ms()
        while True:
            app.tick()
            if done:
                continue
            elapsed = time.ticks_diff(time.ticks_ms(), started)
            invalid = None
            if (s.active, s.draft, s.slot, s.apply_serial) != guard or (
                    s.mode, s.station, s.target_id) != ('TOP', 'BUFFER', colors[current]) or (
                    not app.panel.live_requested or app.panel.frozen):
                invalid = 'TEST_CONTEXT_CHANGED'
            if elapsed >= 90000:
                invalid = 'TIMEOUT'
            if not invalid and elapsed >= 2500 and app.frame is not None and app.frame_id != last:
                last = app.frame_id
                packet = dict(app.frame_metadata)
                obs = s.observation
                cid = colors[current]
                if (packet.get('camera_id') != 1 or packet.get('is_simulated') is not False or
                        obs is None or obs.get('target_id') != cid or obs.get('frame_id') != last):
                    invalid = 'NATIVE_FRAME_IDENTITY_MISMATCH'
                else:
                    packet['image'] = app.frame.copy()
                    reference = ref_adapter.detect('TOP', packet, request['reference_id'], 'BUFFER')
                    row = rows.setdefault(str(cid), {'frames': 0, 'found': 0, 'stable': 0,
                        'inside_manual_face_region': 0, 'detector_errors': 0, 'samples': []})
                    row['frames'] += 1
                    row['found'] += obs['u_px'] is not None
                    row['stable'] += bool(obs['stable'])
                    row['detector_errors'] += obs['status'] == 'ERROR'
                    cx, cy = request['centers'][str(cid)]
                    row['inside_manual_face_region'] += obs['u_px'] is not None and abs(obs['u_px']-cx) <= 12 and abs(obs['v_px']-cy) <= 12
                    row['samples'].append({'frame_id': last, 'observation': clone(obs), 'reference': reference})
                    if row['frames'] == 100:
                        print('SCENE_COLOR_RESULT:' + json.dumps({k: v for k, v in row.items() if k != 'samples'}) + ':color=' + str(cid))
                        current += 1
                        if current < len(colors):
                            s.target_id = colors[current]
                            s.observation = None
                            s.adapter.reset_session()
            if invalid or current == len(colors):
                passed = gate(rows, colors, app.capture_errors, app.errors, invalid)
                result = {'test': 'REAL_CSI1_SCENE_FRESH_FRAME_GATE', 'source': 'REAL_CAMERA',
                    'camera_id': 1, 'is_simulated': False, 'request': request, 'profile': profile,
                    'frame_denominator': sum(r['frames'] for r in rows.values()),
                    'requested_frames': 100 * len(colors), 'rows': rows,
                    'complete': current == len(colors) and invalid is None,
                    'invalid_reason': invalid, 'passed_current_scene_gate': bool(passed),
                    'camera_errors': app.capture_errors, 'initialization_errors': app.errors,
                    'elapsed_ms': elapsed, 'saved_generation': None, 'motors_enabled': False,
                    'manual_region_half_width_px': 12,
                    'limit': 'Sequential per-color; same-frame reference trajectory is diagnostic only. Manual regions are coarse sanity, not independent accuracy, H or robustness acceptance.'}
                try:
                    save_report(result, report)
                    print('SCENE_GATE_RESULT:' + json.dumps({'report': report, 'passed': bool(passed), 'complete': result['complete'], 'frames': result['frame_denominator']}))
                    app.panel.message = '当前场景检查通过，参数未保存' if passed else '当前场景未通过，参数未保存'
                except OSError as error:
                    print('SCENE_GATE_REPORT_FAILED:' + repr(error))
                    app.panel.message = '报告写入失败，不能声明完成'
                s.target_id = 4
                s.observation = None
                s.adapter.reset_session()
                app.panel.page = 'PREVIEW'
                done = True
    finally:
        app.close()


if __name__ == '__main__':
    main()
