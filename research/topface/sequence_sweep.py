"""Frozen test-only six-color recovery and vertical-camera trajectories."""
import argparse
import json
from pathlib import Path

import numpy as np

from cloud_store import atomic_json
from geometry import camera_pose, intrinsics, pixel_to_top
from scene_config import COLORS, camera_partition, make_config
from sequence_config import FRAMES, SCENARIOS


BASE_SEED = 20261119
SCHEMA = 'topface-redetection-sweep-plan-v1.1'


def build_plan():
    cameras = []
    for height in (100, 175, 250, 325, 400):
        for angle in (80, 85, 90, 95, 100):
            for hfov in (62, 65, 68):
                if camera_partition(height, angle, hfov)[1] == 'synthetic_test':
                    cameras.append([height, angle, hfov])
    # This additional inferred camera fills the missing height and angle while
    # retaining the existing deterministic camera-group partition.
    cameras.append([325, 95, 63.4])
    available_heights = [tick / 10 for tick in range(1000, 4001)
                         if camera_partition(tick / 10, 90, 65)[1] == 'synthetic_test']
    vertical_heights = [min(available_heights, key=lambda h: abs(h - target))
                        for target in np.linspace(100, 400, FRAMES)]
    rows = []
    for camera in cameras:
        for color in COLORS:
            for event in SCENARIOS:
                rows.append({'camera': camera, 'color_id': color, 'event': event})
    for color in COLORS:
        for event in ('camera_vertical_rise', 'camera_vertical_fall'):
            rows.append({'camera': None, 'color_id': color, 'event': event})
    return {'schema': SCHEMA, 'base_seed': BASE_SEED, 'count': len(rows) * FRAMES,
            'frames_per_sequence': FRAMES, 'sequences': rows, 'cameras': cameras,
            'vertical_heights_mm': vertical_heights, 'vertical_hfov_deg': 65,
            'scope': 'Discrete inferred test cameras and synthetic trajectories; real calibration pending',
            'scenes': [{'sequence': group, 'frame': frame}
                       for group in range(len(rows)) for frame in range(FRAMES)]}


def planned_config(plan, output_index):
    if plan != build_plan():
        raise ValueError('Sequence plan differs from the frozen test-only protocol')
    row = plan['scenes'][output_index]
    group, frame = row['sequence'], row['frame']
    sequence = plan['sequences'][group]
    event, color = sequence['event'], sequence['color_id']
    config = make_config(BASE_SEED + group, 61)
    vertical = event.startswith('camera_vertical_')
    if vertical:
        heights = plan['vertical_heights_mm']
        height = heights[frame if event.endswith('rise') else FRAMES - 1 - frame]
        angle, hfov = 90, plan['vertical_hfov_deg']
    else:
        height, angle, hfov = sequence['camera']
    camera_group, split = camera_partition(height, angle, hfov)
    if split != 'synthetic_test':
        raise ValueError('Recovery fixture camera belongs to a learning partition')
    pose = camera_pose(height / 1000, angle)
    camera = intrinsics(640, 360, hfov)
    interval = 4 <= frame < 8
    pixel = (320, 180)
    if event == 'in_frame_motion':
        pixel = (160 + min(frame, 7) * 39, 180)
    elif event == 'exit_and_reentry':
        pixel = (-100, 180) if interval else (320, 180)
    position = np.array([0., 0., 0.]) if vertical else pixel_to_top(pixel, camera, pose)
    obj = config['objects'][0]
    obj.update(color_id=color, color_material=COLORS[color][0],
               position_m=[float(position[0]), float(position[1]), 0.])
    hidden = interval and event in ('occlusion', 'exit_and_reentry')
    config.update(camera_intrinsics=camera.tolist(), camera_pose_world_from_cv=pose.tolist(),
                  camera_height_above_top_mm=height, plane_angle_deg=angle, hfov_deg=hfov,
                  camera_group_id=camera_group, split=split, expected_instance_count=1,
                  hard_negative=False, occluder=interval and event == 'occlusion',
                  fully_occluded=interval and event == 'occlusion', occluder_box_m=[.06, .07, .008],
                  expected_visible=not hidden,
                  recovery_window_start=8 if event in ('occlusion', 'exit_and_reentry') else 0,
                  sequence_group_id=f'sweep-{group:03d}-{event}', sequence_frame=frame,
                  sequence_event=event, sequence_length=FRAMES, evaluation_protocol=SCHEMA,
                  exposure_stops=config['exposure_stops'] + .006 * frame,
                  scene_group_id=f'SG-{BASE_SEED}-{output_index:06d}', scene_index=output_index,
                  random_seed=int(np.random.SeedSequence([BASE_SEED, output_index, 0x5E0]).generate_state(1)[0]))
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve the original frozen sequence plan')
    value = build_plan()
    atomic_json(args.output, value)
    print(json.dumps({'count': value['count'], 'sequences': len(value['sequences'])}))
