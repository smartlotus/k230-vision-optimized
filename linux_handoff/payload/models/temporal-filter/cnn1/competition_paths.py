"""Continuous optical-center paths through both official starting areas."""
import math
import random


FIELD_SPAN_MM = 2400
ZONE_CENTERS = ((1050, 1050), (1050, -1050))
ZONE_HALF_WIDTH = 150


def seed_start_belief(filter_, margin_mm=100):
    """Equal mass per area, with a soft allowance for unknown camera offset."""
    if not math.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError('Invalid optical-center start margin')
    belief = [0.] * len(filter_.valid)
    for cx, cy in ZONE_CENTERS:
        indices = []
        for i, valid in enumerate(filter_.valid):
            if not valid:
                continue
            y, x = divmod(i, filter_.w)
            px = filter_.origin[0] + (x + .5) * filter_.cell
            py = filter_.origin[1] + (y + .5) * filter_.cell
            if max(abs(px-cx), abs(py-cy)) <= ZONE_HALF_WIDTH + margin_mm:
                indices.append(i)
        if not indices:
            raise ValueError('No valid cells in a competition start area')
        for i in indices:
            belief[i] += .5 / len(indices)
    filter_.initialize(belief)


def point_at(points, lengths, t):
    distance = t * sum(lengths)
    for i, length in enumerate(lengths):
        if distance <= length or i==len(lengths)-1:
            u = min(1., distance / length)
            return tuple(a + (b-a)*u for a,b in zip(points[i],points[i+1]))
        distance -= length


def schedule(stage):
    count = {'smoke': 4, 'pilot': 40, 'full': 400}[stage]
    for trajectory in range(count):
        seed = {'smoke': 264000, 'pilot': 265000, 'full': 266000}[stage] + trajectory
        rng = random.Random(seed)
        direction = 1 if trajectory%2==0 else -1
        family = trajectory//2 % 5
        pairing = (trajectory//10 + family) % 4
        split = ('train', 'train', 'val', 'test')[trajectory%4] if stage=='smoke' else (
            'train' if pairing<2 else 'val' if pairing==2 else 'test')
        outer = rng.uniform(865, 1000)
        lateral = rng.uniform(-45, 45)
        start = (rng.uniform(965, 1100), direction*rng.uniform(965, 1100))
        goal = (rng.uniform(965, 1100), -direction*rng.uniform(965, 1100))
        path = [start, (outer, direction*outer)]
        if family==0:
            path += [(outer, -direction*outer)]
        elif family==1:
            path += [(outer,lateral), (lateral,lateral), (lateral,-direction*outer)]
        elif family==2:
            path += [(lateral,direction*outer), (lateral,-direction*outer)]
        elif family==3:
            path += [(-outer,direction*outer), (-outer,-direction*outer)]
        else:
            path += [(outer,lateral), (-outer,lateral), (-outer,-direction*outer)]
        path += [(outer,-direction*outer),goal] if family else [goal]
        lengths = [math.dist(a,b) for a,b in zip(path,path[1:])]
        tr = f'gray-{stage}-tr{trajectory:04d}'
        yaw = rng.uniform(-180,180)
        height = rng.uniform(373,427)
        fov = rng.uniform(92,100)
        tilt = (rng.uniform(-1.5,1.5),rng.uniform(-1.5,1.5))
        # The rate below is simulation-only. No measured hard speed gate uses it.
        duration = sum(lengths) / rng.uniform(350,550)
        lighting = ('nominal','low_light','shadow','reflection')[(trajectory//2)%4]
        for frame in range(25):
            t = frame / 24
            u = t*t if family==1 else t
            if family==4:
                u = (t if t<.35 else .35 if t<.5 else t-.15) / .85
            x,y = point_at(path,lengths,u)
            name = f'{tr}-{frame:03d}'
            block = f'{int((x+1200)//400)}:{int((y+1200)//400)}'
            scenario = ('repeated_road','variable_speed','intersection','turn','stop_start')[family]
            yield {'sample_id':name,'trajectory_id':tr,'scene_id':tr,'pose_group_id':tr,
                   'spatial_block_id':block,'split':split,'seed':seed,'scenario':scenario,
                   'evaluation_partition':'trajectory_holdout','appearance_profile_id':tr,
                   'lighting_profile':lighting,'frame_index':frame,'timestamp_s':duration*t,
                   'camera_xy_target':{'camera_x_mm':x,'camera_y_mm':y},
                   'metadata':{'camera_z_mm':height,'camera_yaw_deg':yaw+30*math.sin(2*math.pi*u),
                               'camera_pitch_deg':tilt[0],'camera_roll_deg':tilt[1],
                               'diagonal_fov_deg':fov,'start_zone_id':1 if direction==1 else 2,
                               'goal_zone_id':2 if direction==1 else 1},
                   'image_path':f'rgb/{name}.png','mask_path':f'masks/{name}.png'}
