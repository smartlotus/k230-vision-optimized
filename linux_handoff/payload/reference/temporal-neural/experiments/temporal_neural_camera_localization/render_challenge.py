"""Blender entry for the frozen neural challenge on the corrected field."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import bpy


PROFILES = {
    'nominal': dict(key_rgb=(1., .96, .91), fill_rgb=(1., 1., 1.), exposure=0., key_gain=1., fill_gain=1.),
    'warm': dict(key_rgb=(1., .77, .58), fill_rgb=(1., .88, .68), exposure=-.7, key_gain=.75, fill_gain=.85),
    'cool': dict(key_rgb=(.67, .83, 1.), fill_rgb=(.8, .9, 1.), exposure=.35, key_gain=.85, fill_gain=1.1),
    'warm_reserved': dict(key_rgb=(1., .63, .4), fill_rgb=(1., .76, .51), exposure=-1.0, key_gain=.9, fill_gain=.8),
    'cool_reserved': dict(key_rgb=(.46, .7, 1.), fill_rgb=(.59, .8, 1.), exposure=-.1, key_gain=1.0, fill_gain=.95),
}
TARGETS = ('camera_x_mm', 'camera_y_mm')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def render(args):
    plan = json.loads(args.plan.read_text(encoding='utf8'))
    if digest(args.blend) != plan['geometry_sha256'] or plan['status'] not in (
            'NEURAL_CHALLENGE_FROZEN_BEFORE_SCORING', 'GEOMETRY_PILOT_FROZEN_BEFORE_SCORING'):
        raise ValueError('Incorrect frozen schedule or corrected geometry')
    sys.path.insert(0, str(args.roadnet.resolve()))
    import blender_scene as road
    import sim

    spec = importlib.util.spec_from_file_location('camera_xy_v2_renderer', args.camera_renderer)
    camera_render = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(camera_render)
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    size = (160, 90)
    if bpy.data.objects['2.4 m square'].get('class_id') != 0:
        raise ValueError('Expected the gray road perimeter scene')
    parts = camera_render.existing_scene(road, size)
    nominal_exposure = parts.scene.view_settings.exposure
    camera_render.use_cad(args.cad, parts.blocks)
    board = camera_render.add_qr_board(road, parts)
    for scene in (parts.scene, parts.labels):
        for key in ('use_stamp', 'use_stamp_date', 'use_stamp_time', 'use_stamp_render_time', 'use_stamp_filename'):
            if hasattr(scene.render, key):
                setattr(scene.render, key, False)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'rgb').mkdir()
    (out / 'masks').mkdir()
    count = 0
    pending = out / 'manifest.jsonl.partial'
    with pending.open('x', encoding='utf8', newline='\n') as stream:
        for trajectory in plan['trajectories']:
            start_x, start_y = trajectory['frames'][0]['xy']
            item = dict(group=trajectory['id'], variant=0, provenance=trajectory['seed'], x=start_x, y=start_y)
            props, seed, _ = camera_render.render_record(item, sim)
            props['lighting_id'] = 'nominal'
            props['obstacle_instances'] = [
                obstacle for obstacle in props['obstacle_instances']
                if min(math.dist(obstacle['center_xy'], [v / 1000 for v in f['xy']])
                       for f in trajectory['frames']) > .21
            ]
            for variant in trajectory['variants']:
                seq = trajectory['id'] + '-' + variant
                profile = PROFILES[variant]
                for i, frame in enumerate(trajectory['frames']):
                    name = f'{seq}-{i:03d}'
                    x, y = frame['xy']
                    rec = dict(props, center=[x / 1000, y / 1000, trajectory['height_mm'] / 1000],
                               camera_yaw_deg=frame['yaw_deg'], camera_pitch_deg=trajectory['pitch_deg'],
                               camera_roll_deg=trajectory['roll_deg'], diag_fov_deg=trajectory['diagonal_fov_deg'],
                               image_path=f'rgb/{name}.png', mask_path=f'masks/{name}.png')
                    rec['turntable_phase_rad'] += (math.tau * rec['turntable_direction'] * frame['timestamp_s']
                                                   / rec['turntable_period_s'])
                    camera_render.apply(road, parts, rec, size, board)
                    parts.key.data.color = profile['key_rgb']
                    parts.key.data.energy *= profile['key_gain']
                    parts.fill.data.color = profile['fill_rgb']
                    parts.fill.data.energy *= profile['fill_gain']
                    parts.scene.view_settings.exposure = nominal_exposure + profile['exposure']
                    bx, by, bz = parts.camera.matrix_world.translation
                    if max(abs(bx * 1000 - x), abs(by * 1000 - y)) > .001:
                        raise ValueError('Rendered optical center differs from schedule')
                    road.render(parts, rec, out, size)
                    row = dict(sample_id=name, trajectory_id=seq, pose_group_id=trajectory['id'],
                               appearance_profile_id=f'{seed}-{variant}',
                               frame_index=i, split=trajectory.get('split', 'test'), challenge=trajectory['group'],
                               appearance=variant,
                               timestamp_s=frame['timestamp_s'], camera_xy_target=dict(zip(TARGETS, (bx * 1000, by * 1000))),
                               distance_to_original_train_path_mm=frame['distance_to_original_train_path_mm'],
                               image_path=rec['image_path'], image_sha256=rec['image_sha256'],
                               mask_path=rec['mask_path'], mask_sha256=rec['mask_sha256'],
                               metadata=dict(camera_to_world_blender=rec['camera_to_world_blender'],
                                             intrinsics_pinhole=rec['intrinsics_pinhole'], appearance_seed=seed,
                                             start_zone_id=trajectory['start_zone_id'],
                                             finish_zone_id=trajectory['finish_zone_id']))
                    stream.write(json.dumps(row, separators=(',', ':')) + '\n')
                    count += 1
                stream.flush()
                print(f'NEURAL_CHALLENGE_FRAMES={count}', flush=True)
    pending.rename(out / 'manifest.jsonl')
    if count != plan.get('expected_frames', count):
        raise ValueError('Render did not cover all planned frames')
    identity = dict(status='RENDER_COMPLETE', frames=count, manifest_sha256=digest(out / 'manifest.jsonl'),
                    plan_sha256=digest(args.plan), geometry_sha256=digest(args.blend), cad_sha256=digest(args.cad),
                    field_mask_sha256=plan['field_mask_sha256'], original_pilot_manifest_sha256=plan['pilot_manifest_sha256'],
                    camera_v2_renderer_sha256=digest(args.camera_renderer), roadnet_renderer_sha256=digest(Path(road.__file__)),
                    source_sha256=digest(Path(__file__)), blender=bpy.app.version_string,
                    nominal_exposure_ev=nominal_exposure, exposure_offsets_relative=True, profiles=PROFILES)
    (out / 'QA.json').write_text(json.dumps(identity, indent=2) + '\n', encoding='utf8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('plan', 'blend', 'cad', 'roadnet', 'camera-renderer', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    render(parser.parse_args(sys.argv[sys.argv.index('--') + 1:]))
