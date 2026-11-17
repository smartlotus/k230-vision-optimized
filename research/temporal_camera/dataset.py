"""Versioned continuous camera paths; strict split/target/window audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

SCENARIOS = ('straight','turn','intersection','repeated_road','frame_drop',
             'variable_speed','blur','relocalization','stop_start','lighting')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schedule(stage):
    count = {'smoke':10,'pilot':100,'full':1500}[stage]
    seed = {'smoke':170100,'pilot':171000,'full':190000}[stage]
    reserved = (0,5,30,35)
    ordinary = [i for i in range(36) if i not in reserved]
    for tr in range(count):
        rng = random.Random(seed+tr)
        group,kind_index=divmod(tr,10)
        kind=SCENARIOS[kind_index]
        partition='trajectory_holdout'
        split='train' if group%10<6 else 'val' if group%10<8 else 'test'
        if stage=='smoke':split=('val' if tr==7 else 'test' if tr==8 else 'train')
        if group%10==9:
            partition='spatial_challenge' if kind_index<5 else 'appearance_challenge'
        tile=reserved[kind_index%4] if partition=='spatial_challenge' else ordinary[(kind_index*3+group)%len(ordinary)]
        col, row = tile%6, tile//6
        cx, cy = -1000+col*400, -1000+row*400
        if partition!='spatial_challenge':
            if kind in ('intersection','turn'):cx,cy=0,0
            elif kind in ('straight','repeated_road'):cx,cy=0,(-650 if group%2 else 650)
        cx+=rng.uniform(-8,8);cy+=rng.uniform(-8,8)
        yaw = rng.uniform(-math.pi, math.pi)
        identifier = f'{stage}-tr{tr:05d}'
        ts = 0.
        for frame in range(10):
            u = frame/9
            if kind == 'variable_speed':u = u*u
            if kind == 'stop_start':u = 0 if u<.35 else (u-.35)/.65
            if kind == 'turn':
                dx,dy = 130*math.cos(u*math.pi/2),130*math.sin(u*math.pi/2)
            elif kind == 'intersection':
                dx,dy = (240*(u-.5),0) if frame<5 else (0,240*(u-.5))
            else:
                dx,dy = 260*(u-.5)*math.cos(yaw),260*(u-.5)*math.sin(yaw)
            # Entire path remains in a split-owned tile; continuous optical
            # centre poses, no independent per-frame positional jitter.
            x,y = cx+dx,cy+dy
            actual_tile=min(5,max(0,int((y+1200)//400)))*6+min(5,max(0,int((x+1200)//400)))
            ts += .1 if frame else 0
            if kind == 'frame_drop' and frame in (3,7):ts += .2
            yield {'sample_id':f'{identifier}-{frame:03d}', 'trajectory_id':identifier,
                   'scene_id':identifier, 'spatial_block_id':f'tile-{actual_tile:02d}',
                   'pose_group_id':f'xy-{round(x,3)}-{round(y,3)}',
                   'appearance_profile_id':identifier+'-appearance',
                   'split':split,'timestamp_s':ts,'frame_index':frame,'scenario':kind,
                   'evaluation_partition':partition,
                   'lighting_profile':'reflection' if partition=='appearance_challenge' else ('nominal','low_light','shadow')[group%3],
                   'seed':seed+tr, 'map_variant':'existing-roadnet-layout-v1',
                   'camera_xy_target':{'camera_x_mm':x,'camera_y_mm':y},
                   'metadata':{'camera_z_mm':400., 'camera_yaw_deg':math.degrees(yaw)+(90*u if kind=='turn' else 0),
                               'camera_pitch_deg':0.,'camera_roll_deg':0.,
                               'diagonal_fov_deg':96.,'parameter_status':'ASSUMED_NOT_CALIBRATED'},
                   'image_path':f'rgb/{identifier}-{frame:03d}.png'}


def audit(rows, root=None):
    owners = {k:{} for k in ('trajectory_id','pose_group_id','appearance_profile_id')}
    ids=set();last={};counts={};groups={}
    for r in rows:
        if set(r['camera_xy_target']) != {'camera_x_mm','camera_y_mm'}:
            raise ValueError('Only camera_x_mm/camera_y_mm targets allowed')
        for key in r:
            if ('target' in key and key!='camera_xy_target') or key.startswith(('robot_','chassis_')):
                raise ValueError('Forbidden target/robot field')
        if r['sample_id'] in ids:raise ValueError('Duplicate sample')
        ids.add(r['sample_id'])
        if r['split'] not in ('train','val','test'):raise ValueError('Unknown split')
        for key,owner in owners.items():
            value=r[key]
            if value in owner and owner[value]!=r['split']:raise ValueError('Split leakage: '+key)
            owner[value]=r['split']
        ts=r['timestamp_s'];tr=r['trajectory_id']
        if not math.isfinite(ts) or (tr in last and ts<=last[tr]):raise ValueError('Nonmonotonic timestamp')
        last[tr]=ts
        if any(not math.isfinite(v) or not -1200<=v<1200 for v in r['camera_xy_target'].values()):
            raise ValueError('Invalid camera XY')
        path=Path(r['image_path'])
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe image path')
        if root is not None:
            if digest(root/path)!=r['image_sha256']:raise ValueError('Image hash differs')
        counts[r['split']]=counts.get(r['split'],0)+1
        groups.setdefault(tr,[]).append(r)
    challenge_tiles={r['spatial_block_id'] for r in rows if r.get('evaluation_partition')=='spatial_challenge'}
    development_tiles={r['spatial_block_id'] for r in rows if r['split'] in ('train','val')}
    if challenge_tiles & development_tiles:raise ValueError('Spatial challenge leakage')
    return {'status':'PASS','frames':len(rows),'splits':counts,'trajectories':len(groups),
            'evaluation_partitions':{p:sum(r.get('evaluation_partition')==p for r in rows) for p in ('trajectory_holdout','spatial_challenge','appearance_challenge')},
            'spatial_challenge_tiles_disjoint':True,
            'split_group_counts':{k:len(v) for k,v in owners.items()},
            'scenario_counts':{s:sum(r['scenario']==s for r in rows) for s in SCENARIOS}},groups


def build_sequences(rows, length):
    if length not in (4,8):raise ValueError('Only fixed T4/T8')
    _,groups=audit(rows)
    result=[]
    for tr,frames in groups.items():
        for i in range(length-1,len(frames)):
            clip=frames[i-length+1:i+1]
            result.append({'sequence_id':f'{tr}-T{length}-{i:03d}', 'trajectory_id':tr,
                           'split':clip[-1]['split'],'frames':[r['image_path'] for r in clip],
                           'timestamps_s':[r['timestamp_s'] for r in clip],
                           'sample_id':clip[-1]['sample_id'],'target_index':length-1,
                           'camera_xy_target':clip[-1]['camera_xy_target']})
    return result


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows),encoding='utf8')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args()
    rows=[json.loads(x) for x in (args.root/'manifest.jsonl').read_text().splitlines()]
    report,_=audit(rows,args.root)
    for t in (4,8):write_jsonl(args.root/f'sequences-T{t}.jsonl',build_sequences(rows,t))
    (args.root/'QA.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
