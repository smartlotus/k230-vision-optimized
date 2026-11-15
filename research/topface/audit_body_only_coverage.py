"""Read-only analytic check of complete top exit in the frozen Full layout."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import time

import numpy as np

from cloud_store import atomic_json,sha
from geometry import project,top_boundary
from scene_config import make_config


def clip_image(polygon):
    p=np.asarray(polygon,dtype=np.float64)
    for axis,bound,sign in ((0,0.,1),(0,640.,-1),(1,0.,1),(1,360.,-1)):
        if not len(p):return p
        previous=np.roll(p,1,axis=0)
        inside=(p[:,axis]-bound)*sign>=0
        cross=np.roll(inside,1)!=inside
        if inside.all():continue
        output=np.empty((len(p),2,2));used=np.zeros((len(p),2),dtype=bool)
        if cross.any():
            fraction=(bound-previous[cross,axis])/(p[cross,axis]-previous[cross,axis])
            intersection=previous[cross]+fraction[:,None]*(p[cross]-previous[cross])
            intersection[:,axis]=bound;output[cross,0]=intersection;used[cross,0]=True
        output[inside,1]=p[inside];used[inside,1]=True;p=output[used]
    return p


def area(p):
    return (float(abs(np.dot(p[:,0],np.roll(p[:,1],-1))-
                      np.dot(p[:,1],np.roll(p[:,0],-1))))*.5 if len(p)>2 else 0.)


def scan(seed=20261117,count=10000):
    started=time.perf_counter();counts=Counter();partial_heights=Counter();empty_train=Counter()
    groups=set();partial_groups=Counter();minimum=(float('inf'),None)
    for index in range(count):
        config=make_config(seed,index);group=config['camera_group_id'];groups.add(group)
        counts['scenes']+=1;counts['split_'+config['split']]+=1
        if not config['objects']:
            counts['world_empty_'+config['split']]+=1
            if config['split']=='train':empty_train[group]+=1
        K=np.asarray(config['camera_intrinsics']);pose=np.asarray(config['camera_pose_world_from_cv'])
        for item in config['objects']:
            counts['world_instances']+=1
            boundary=project(top_boundary(item['position_m'][:2],512),K,pose)
            intersection=area(clip_image(boundary))
            if intersection<minimum[0]:minimum=(intersection,{'scene_index':index,'instance_id':item['instance_id'],
                'camera_group':group,'physical_top_center':project([[*item['position_m'][:2],.06]],K,pose)[0].tolist()})
            counts['top_projection_no_intersection']+=int(intersection<=1e-9)
            partial=bool((boundary[:,0]<0).any() or (boundary[:,0]>640).any() or
                         (boundary[:,1]<0).any() or (boundary[:,1]>360).any())
            if partial:
                counts['partially_or_fully_outside']+=1;partial_groups[group]+=1
                partial_heights[str(config['camera_height_above_top_mm'])]+=1
    return {'status':'FROZEN_FULL_LAYOUT_GEOMETRY_AUDITED','observed_utc':datetime.now(timezone.utc).isoformat(),
        'base_seed':seed,'indices':[0,count-1],'statistics':dict(counts),'camera_groups':len(groups),
        'partial_counts_by_height_mm':dict(partial_heights),'partial_counts_by_camera_group':dict(sorted(partial_groups.items())),
        'minimum_projected_top_intersection_px2':minimum[0],'minimum_case':minimum[1],
        'empty_train_slots_by_camera_group':dict(sorted(empty_train.items())),
        'method':'512-vertex inscribed r15mm top disk at z60mm, float64 convex polygon image clipping; no occluder suppression',
        'interpretation':'If every top disk intersects the image, no top-fully-offscreen/body-visible sample exists regardless of base shape.',
        'limitations':['Analytic top disk using original CAD dimensions, not a fresh mesh raster audit.',
            'Partial truncation differs from complete top exit; occlusion is not counted as exit.'],
        'proposed_followup':'Independent audited train-only body-only negative background augmentation; not yet implemented or qualified',
        'source_sha256':{name:sha(Path(__file__).parent/name) for name in ('scene_config.py','geometry.py')},
        'elapsed_seconds':time.perf_counter()-started}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve original layout coverage audit')
    result=scan();atomic_json(args.output,result)
    print({key:result[key] for key in ('status','statistics','minimum_projected_top_intersection_px2')})
