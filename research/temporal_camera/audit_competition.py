"""Verify rendered masks and continuous coverage against the corrected map."""
import argparse
import collections
import json
import math
from pathlib import Path

from PIL import Image

from dataset import audit, build_sequences, digest, write_jsonl


def check(data, field_mask):
    rows = [json.loads(line) for line in (data/'manifest.jsonl').read_text().splitlines()]
    report, groups = audit(rows,data)
    resolution = tuple(json.loads((data/'RENDER_IDENTITY.json').read_text())['resolution'])
    if resolution not in ((160,90),(640,360)):
        raise ValueError('Unknown competition sensor resolution')
    with Image.open(field_mask) as field:
        field.load()
    def label(x,y):
        return field.getpixel((int((x/3000+.5)*field.width),int((.5-y/3000)*field.height)))
    for x,y,expected in ((850,850,0),(850,-850,0),(-850,850,0),(-850,-850,0),
                         (0,0,0),(1050,1050,3),(1050,-1050,3),(425,425,1),(-1160,50,2),(-50,-1160,2)):
        if label(x,y)!=expected:
            raise ValueError(f'Unexpected map class at {(x,y)}: {label(x,y)} != {expected}')
    geometry_counts = collections.Counter()
    starts = collections.Counter()
    scenarios = collections.Counter()
    for trajectory in groups.values():
        if len(trajectory)!=25:
            raise ValueError('Trajectories must have 25 ordered frames')
        xy = [(r['camera_xy_target']['camera_x_mm'],r['camera_xy_target']['camera_y_mm']) for r in trajectory]
        if label(*xy[0])!=3 or label(*xy[-1])!=3 or xy[0][1]*xy[-1][1]>=0:
            raise ValueError('Expected opposite blue start and goal areas')
        starts[f"{trajectory[0]['split']}/blue{trajectory[0]['metadata']['start_zone_id']}"] += 1
        scenarios[f"{trajectory[0]['split']}/{trajectory[0]['scenario']}"] += 1
        for left,right in zip(xy,xy[1:]):
            steps = max(1,math.ceil(math.dist(left,right)/10))
            for i in range(steps+1):
                p = tuple(a+(b-a)*i/steps for a,b in zip(left,right))
                code = label(*p)
                geometry_counts[code] += 1
                if code not in (0,3):
                    raise ValueError(f'Optical projection left gray/blue at {p}: class {code}')
    hash_owners = {}
    for row in rows:
        original_split = hash_owners.setdefault(row['image_sha256'], row['split'])
        if original_split!=row['split']:
            raise ValueError('Identical RGB crosses a split')
        with Image.open(data/row['image_path']) as rgb:
            if rgb.size!=resolution:
                raise ValueError('RGB resolution differs from the rendering identity')
        mask = data/row['mask_path']
        if not mask.resolve().is_relative_to(data.resolve()) or digest(mask)!=row['mask_sha256']:
            raise ValueError('Mask identity differs')
        with Image.open(mask) as image:
            if image.mode!='L' or image.size!=resolution or not set(image.tobytes())<={0,1,2,3,4,5,6,255}:
                raise ValueError('Mask class or image contract differs')
    report.update(field_mask_sha256=digest(field_mask),start_zones_by_split=dict(starts),
                  trajectory_scenarios_by_split=dict(scenarios),identical_rgb_cross_split=0,
                  map_class_counts_sampled_at_most_10mm=dict(geometry_counts),
                  mask_frames_verified=len(rows),resolution=resolution,manifest_sha256=digest(data/'manifest.jsonl'),
                  scope='Camera optical projections in simulation; competition chassis clearance not assessed')
    for length in (4,8):
        write_jsonl(data/f'sequences-T{length}.jsonl',build_sequences(rows,length))
    (data/'QA.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data',type=Path)
    parser.add_argument('field_mask',type=Path)
    args = parser.parse_args()
    check(args.data,args.field_mask)
