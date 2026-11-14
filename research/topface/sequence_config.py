"""Frozen CSI2-like sequences for re-detection tests after model selection."""
import numpy as np

from geometry import pixel_to_top
from scene_config import make_config


BASE_SEED = 20261118
SCENARIOS = ('occlusion', 'in_frame_motion', 'exit_and_reentry')
FRAMES = 12


def sequence_config(frame_index):
    scenario_index, frame = divmod(frame_index, FRAMES)
    scenario = SCENARIOS[scenario_index]
    config = make_config(BASE_SEED+scenario_index, 61)
    if config['split'] != 'synthetic_test' or len(config['objects']) != 1:
        raise ValueError('Sequence anchor must have one synthetic test target')
    obj = config['objects'][0]
    K = np.asarray(config['camera_intrinsics'])
    pose = np.asarray(config['camera_pose_world_from_cv'])
    interval = frame in range(4, 8)
    if scenario == 'occlusion':
        pixel = (320, 180)
        config.update(occluder=interval, fully_occluded=interval,
                      occluder_box_m=[.06,.07,.008])
    elif scenario == 'in_frame_motion':
        pixel = (160+min(frame,7)*39,180)
    else:
        pixel = (320,180) if frame<4 else ((-100,180) if interval else (250+20*(frame-8),180))
    xy = pixel_to_top(pixel,K,pose)
    obj['position_m'] = [float(xy[0]),float(xy[1]),0.]
    config.update(sequence_group_id=f'redetection-{scenario}', sequence_frame=frame, sequence_event=scenario,
                  exposure_stops=config['exposure_stops']+.006*frame,
                  random_seed=int(np.random.SeedSequence([BASE_SEED, 0x5E0, frame_index]).generate_state(1)[0]),
                  scene_group_id=f'SG-{BASE_SEED}-{frame_index:06d}', scene_index=frame_index,
                  hard_negative=False, occluder=scenario=='occlusion' and interval,
                  expected_visible=not(interval and scenario!='in_frame_motion'),
                  recovery_window_start=8 if scenario!='in_frame_motion' else 4)
    return config
