"""Explicitly armed same-frame green comparison; leaves the live panel running."""
import json
import os
import sys

ROOT = '/sdcard/transport_mvp/'
REQUEST = ROOT + 'green_compare_request.json'


def comparison_profiles(profile):
    from k230.tuning.schema import clone, validate
    validate(profile)
    profiles = {'original': clone(profile)}
    for name, lower_l, lower_b in (('L41_B17', 41, 17), ('L40_B15', 40, 15)):
        candidate = clone(profile)
        candidate['profile_id'] = profile['profile_id'] + '-' + name
        candidate['revision'] += 1
        threshold = candidate['contexts']['TOP_BUFFER']['thresholds']['4']
        if threshold is None:
            raise ValueError('Green threshold required')
        threshold[0], threshold[4] = lower_l, lower_b
        validate(candidate)
        profiles[name] = candidate
    measured = clone(profiles['L41_B17'])
    measured['profile_id'] = profile['profile_id'] + '-L41_A30_B17'
    measured['contexts']['TOP_BUFFER']['thresholds']['4'][3] = -30
    validate(measured)
    profiles['L41_A30_B17'] = measured
    return profiles


def validate_request(request, profile):
    if set(request) != {'attempt', 'scene_id', 'profile', 'profile_file', 'face_region', 'operator_ready'}:
        raise ValueError('Explicit scene/profile/face region request required')
    for key in ('attempt', 'scene_id'):
        value = request[key]
        if not isinstance(value, str) or not 1 <= len(value) <= 60 or any(
                c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in value):
            raise ValueError('Invalid ' + key)
    if request['operator_ready'] is not True or request['profile'] != profile['profile_id']:
        raise ValueError('Operator readiness or profile identity differs')
    profile_filename(request)
    region = request['face_region']
    if not isinstance(region, list) or len(region) != 4 or any(type(v) is not int for v in region):
        raise ValueError('Integer source-pixel face region required')
    x, y, w, h = region
    b = profile['binding']
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > b['width'] or y + h > b['height']:
        raise ValueError('Face region outside source image')
    if (b['width'], b['height']) != (640, 360):
        raise ValueError('This comparison requires a new 640x360 scene')


def profile_filename(request):
    name = request['profile_file']
    if not isinstance(name, str) or not name.endswith('.json') or not name[:-5] or any(
            c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name[:-5]):
        raise ValueError('Profile must be a local JSON filename')
    return name


def save_report(result, path):
    """Single writer; preserve existing or partial evidence and verify final bytes."""
    for existing in (path, path + '.tmp'):
        try:
            os.stat(existing)
        except OSError as error:
            if error.errno != 2:
                raise
        else:
            raise OSError('Existing report preserved: ' + existing)
    raw = json.dumps(result)
    temporary = path + '.tmp'
    with open(temporary, 'w') as stream:
        stream.write(raw)
    with open(temporary) as stream:
        if stream.read() != raw:
            raise OSError('Temporary report readback differs')
    if hasattr(os, 'sync'):
        os.sync()
    os.rename(temporary, path)
    if hasattr(os, 'sync'):
        os.sync()
    with open(path) as stream:
        if stream.read() != raw:
            raise OSError('Final report readback differs')


def main():
    if sys.implementation.name != 'micropython':
        raise SystemExit('Use CanMV on K230.')
    import time
    sys.path.insert(0, ROOT)
    from k230.tuning.runtime import TuningApp
    from k230.tuning.schema import clone, compile_config
    from k230.vision_adapter import VisionAdapter
    with open(REQUEST) as stream:
        request = json.load(stream)
    with open(ROOT + profile_filename(request)) as stream:
        profile = json.load(stream)
    validate_request(request, profile)
    report = ROOT + 'green_pair100_' + request['attempt'] + '.json'
    for path in (report, report + '.tmp'):
        try:
            os.stat(path)
        except OSError as error:
            if error.errno != 2:
                raise
        else:
            raise RuntimeError('Attempt already used; no replay')
    profiles = comparison_profiles(profile)
    adapters = {key: VisionAdapter(compile_config(value)) for key, value in profiles.items()}
    app = TuningApp()
    app.session.draft = clone(profile)
    app.session.dirty = True
    app.session.apply()
    app.session.select('TOP', 'BUFFER')
    app.session.target_id = 4
    guard = (clone(app.session.active), clone(app.session.draft), app.session.slot, app.session.apply_serial)
    rows = []
    summary = {key: {'frames': 0, 'found': 0, 'stable': 0, 'inside_face_region': 0,
                     'errors': 0} for key in profiles}
    done = False
    last = -1
    try:
        app.setup()
        app.panel.live_requested = True
        started = time.ticks_ms()
        while True:
            app.tick()
            if done:
                continue
            elapsed = time.ticks_diff(time.ticks_ms(), started)
            s = app.session
            changed = (s.active, s.draft, s.slot, s.apply_serial) != guard or (
                s.mode, s.station, s.target_id) != ('TOP', 'BUFFER', 4) or (
                not app.panel.live_requested or app.panel.frozen)
            timed_out = elapsed >= 60000
            if not changed and not timed_out and elapsed >= 2500 and app.frame is not None and app.frame_id != last:
                packet = dict(app.frame_metadata)
                if packet['camera_id'] != 1 or packet['is_simulated'] is not False:
                    raise ValueError('Native CSI1 required')
                last = app.frame_id
                packet['image'] = app.frame.copy()
                results = {}
                for key, adapter in adapters.items():
                    obs = adapter.detect('TOP', packet, 4, 'BUFFER')
                    results[key] = obs
                    r = summary[key]
                    r['frames'] += 1
                    r['errors'] += obs['status'] == 'ERROR'
                    if obs['u_px'] is not None:
                        r['found'] += 1
                        r['stable'] += bool(obs['stable'])
                        x, y, w, h = request['face_region']
                        r['inside_face_region'] += x <= obs['u_px'] < x+w and y <= obs['v_px'] < y+h
                rows.append({'frame_id': last, 'results': results})
            if changed or timed_out or len(rows) == 100:
                result = {'test': 'REAL_CSI1_GREEN_SAME_FRAME_COMPARISON', 'request': request,
                    'source': 'REAL_CAMERA', 'camera_id': 1, 'is_simulated': False,
                    'requested_frames': 100, 'frame_denominator': len(rows), 'complete': len(rows) == 100 and not changed,
                    'configuration_changed': changed, 'timed_out': timed_out,
                    'profiles': profiles, 'summary': summary, 'rows': rows,
                    'camera_errors': app.capture_errors, 'initialization_errors': app.errors,
                    'saved_generation': None, 'motors_enabled': False,
                    'limit': 'Current scene only; manual face region is coarse, not independent accuracy or H. No automatic candidate selection or save.'}
                try:
                    save_report(result, report)
                    print('GREEN_PAIR_RESULT:' + json.dumps({'report': report, 'complete': result['complete'], 'summary': summary}))
                    app.panel.message = '同帧对照已记录，未自动保存参数'
                except OSError as error:
                    print('GREEN_PAIR_REPORT_FAILED:' + repr(error))
                    app.panel.message = '报告写入失败，不能声明对照完成'
                done = True
    finally:
        app.close()


if __name__ == '__main__':
    main()
