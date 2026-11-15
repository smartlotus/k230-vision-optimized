"""Independent upright-CAD negatives for the twenty empty-train camera groups."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from cloud_store import atomic_json
from geometry import pixel_to_top,project,top_boundary
from scene_config import COLORS,camera_partition,make_config

SCHEMA='topface-body-only-plan-v1.1'
ROLE='BODY_ONLY_TOP_OUTSIDE'
EDGES=('left','right','top','bottom')


def outside_margin(boundary,edge):
    return {'left':-float(boundary[:,0].max()),'right':float(boundary[:,0].min())-640.,
            'top':-float(boundary[:,1].max()),'bottom':float(boundary[:,1].min())-360.}[edge]


def body_area(xy,K,pose):
    # Planning only. Actual visible fused-CAD object-ID pixels are audited later.
    import cv2
    angles=np.linspace(0,2*np.pi,128,endpoint=False)
    ring=np.column_stack([xy[0]+.025*np.cos(angles),xy[1]+.025*np.sin(angles)])
    points=np.vstack([np.column_stack([ring,np.full(len(ring),z)]) for z in (0.,.03)])
    hull=cv2.convexHull(project(points,K,pose).astype(np.float32))
    rectangle=np.array([[0,0],[640,0],[640,360],[0,360]],np.float32)
    return float(cv2.intersectConvexConvex(hull,rectangle)[0])


def build_plan(seed=20261123):
    representatives={}
    for index in range(10000):
        config=make_config(20261117,index)
        if config['split']=='train' and not config['objects']:
            representatives.setdefault(config['camera_group_id'],index)
    if len(representatives)!=20:raise ValueError('Frozen Full empty-train camera groups changed')
    scenes=[]
    for group_index,(group,source_index) in enumerate(sorted(representatives.items())):
        for color in range(1,7):
            config=make_config(seed,source_index+600*(color-1))
            if config['split']!='train' or config['camera_group_id']!=group:
                raise ValueError('Negative source must keep its original train camera partition')
            rng=np.random.default_rng(np.random.SeedSequence([seed,group_index,color,0xB0D]))
            edge=EDGES[(group_index+color-1)%4]
            along=float(rng.uniform(100,260) if edge in ('left','right') else rng.uniform(160,480))
            K=np.asarray(config['camera_intrinsics']);pose=np.asarray(config['camera_pose_world_from_cv'])
            chosen=None
            for distance in range(8,302,2):
                uv={'left':[-distance,along],'right':[640+distance,along],
                    'top':[along,-distance],'bottom':[along,360+distance]}[edge]
                xy=pixel_to_top(uv,K,pose)[:2]
                boundary=project(top_boundary(xy),K,pose)
                margin=outside_margin(boundary,edge)
                if margin<4.:continue
                visible_body=body_area(xy,K,pose)
                if visible_body<512.:continue
                chosen=(xy,margin,visible_body);break
            if chosen is None:raise ValueError(f'No safe body-only placement for {group}/{edge}')
            xy,margin,visible_body=chosen
            config.update(scene_index=len(scenes),scene_group_id=f'BODY-NEG-{seed}-{len(scenes):04d}',
                expected_instance_count=1,max_instances=5,objects=[{'instance_id':1,'color_id':color,
                    'color_material':COLORS[color][0],'position_m':[float(xy[0]),float(xy[1]),0.],
                    'yaw_rad':float(rng.uniform(0,2*np.pi)),'roughness':float(rng.uniform(.35,.85)),
                    'brightness':float(rng.uniform(.88,1.08))}],occluder=False,hard_negative=False,
                negative_role=ROLE,negative_edge=edge,top_outside_margin_px=margin,
                planned_body_projection_area_px2=visible_body,source_layout_index=source_index,
                color_policy='all six colors within each matched train camera group; no visible top')
            scenes.append(config)
    return {'schema':SCHEMA,'base_seed':seed,'count':len(scenes),'scenes':scenes,
        'source_scope':'INDEPENDENT_SYNTHETIC_TRAIN_ONLY','camera_groups':sorted(representatives),
        'purpose':'top fully offscreen while original upright fused CAD remains partly visible',
        'geometry_changed':False,'original_full_archives_changed':False}


def planned_config(plan,index):
    if plan['schema']!=SCHEMA:raise ValueError('Explicit body-only source plan required')
    config=copy.deepcopy(plan['scenes'][index])
    group,split=camera_partition(config['camera_height_above_top_mm'],config['plane_angle_deg'],config['hfov_deg'])
    if (split!='train' or config['split']!=split or config['camera_group_id']!=group or
            config['negative_role']!=ROLE or config['expected_instance_count']!=1 or len(config['objects'])!=1 or
            config['occluder'] or config['hard_negative'] or config['base_seed']!=plan['base_seed']):
        raise ValueError('Body-only source must remain one upright original object in a train camera group')
    item=config['objects'][0]
    boundary=project(top_boundary(item['position_m'][:2]),np.asarray(config['camera_intrinsics']),
                     np.asarray(config['camera_pose_world_from_cv']))
    if outside_margin(boundary,config['negative_edge'])<4.:
        raise ValueError('Source top is not fully outside with the frozen safety margin')
    return config


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261123)
    parser.add_argument('--qualification',action='store_true');args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve the original negative scene plan')
    plan=build_plan(args.seed)
    if args.qualification:plan['scenes']=plan['scenes'][:1];plan['count']=1
    atomic_json(args.output,plan)
    print(json.dumps({'count':plan['count'],'seed':plan['base_seed'],'train_camera_groups':len(plan['camera_groups'])}))
