"""Read-only diagnostics of a hash-bound V1.1 rendered dataset.

Perceptual similarities are review candidates, not proof of duplicate scenes
or leakage. Empty/negative scenes are reported separately. Never changes labels.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np

from cloud_store import atomic_json,sha


def stats(values):
    if not values:return {'n':0}
    data=np.asarray(values,dtype=float)
    return {'n':len(values),'min':float(data.min()),'p05':float(np.quantile(data,.05)),
            'median':float(np.median(data)),'p95':float(np.quantile(data,.95)),'max':float(data.max())}


def perceptual_hash(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    coeff=cv2.dct(cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32))[:8,:8].copy()
    coeff[0,0]=0
    bits=(coeff>np.median(coeff)).ravel()
    return int.from_bytes(np.packbits(bits).tobytes(),'big')


def diagnose(root,expected_count=1000,max_seconds=600):
    root=Path(root).resolve();deadline=time.monotonic()+max_seconds
    audit_path=root/'AUDIT_V11.json';audit=json.loads(audit_path.read_text(encoding='utf8'))
    if audit['status']!='PASS' or audit['images']!=expected_count:
        raise ValueError('Expected exact count and PASS from original label QA')
    scenes=audit['scenes']
    if len(scenes)!=expected_count:raise ValueError('QA row denominator mismatch')
    counts=Counter();colors=Counter();splits=Counter();visible_colors=Counter()
    visibility=Counter();split_groups=defaultdict(set);camera_groups=defaultdict(set)
    values=defaultdict(list);images=[];rgb_hashes=set();label_hashes=[]
    for row in scenes:
        if time.monotonic()>deadline:raise TimeoutError('Diagnostic time budget exhausted')
        directory=root/row['scene']
        if not directory.resolve().is_relative_to(root):raise ValueError('Scene path outside dataset')
        label_path=directory/'labels.json';rgb_path=directory/'rgb.png'
        if sha(label_path)!=row['label_sha256'] or sha(rgb_path)!=row['rgb_sha256']:
            raise ValueError('Original QA member identity changed')
        if row['rgb_sha256'] in rgb_hashes:raise ValueError('Exact duplicate RGB')
        rgb_hashes.add(row['rgb_sha256']);label_hashes.append(row['label_sha256'])
        label=json.loads(label_path.read_text(encoding='utf8'))
        if len(label['instances'])!=label['expected_instance_count']:
            raise ValueError('World-instance annotation denominator differs')
        counts[label['expected_instance_count']]+=1;splits[label['split']]+=1
        split_groups[label['scene_group_id']].add(label['split'])
        camera_groups[label['camera_group_id']].add(label['split'])
        image=cv2.imread(str(rgb_path))
        if image is None or image.shape!=(360,640,3):raise ValueError('Unexpected RGB shape')
        images.append({'scene':row['scene'],'split':row['split'],'expected_count':label['expected_instance_count'],
                       'hash':perceptual_hash(image)})
        for item in label['instances']:
            color=str(item['color_id']);colors[color]+=1
            if item['color_id'] not in range(1,7):raise ValueError('Color ID outside approved1..6 mapping')
            visibility['world_instances']+=1
            for flag in ('center_valid','center_visible','geometry_valid','truncated','occluded'):
                visibility[flag]+=int(item[flag])
            if item['center_visible'] and not item['center_valid']:
                raise ValueError('Visible center cannot be geometrically invalid')
            values['visible_fraction'].append(item['visible_fraction'])
            values['physical_conic_center_distance_px'].append(float(np.linalg.norm(
                np.asarray(item['center_xy_projected_3d'])-np.asarray(item['ellipse_conic_center']))))
            ellipse=item['ellipse_full'];values['ellipse_minor_major_ratio'].append(ellipse['semi_minor']/ellipse['semi_major'])
            box=item['bbox_xyxy_visible']
            if box is None:
                visibility['no_visible_bbox_retained_in_labels']+=1
                continue
            width,height=box[2]-box[0],box[3]-box[1]
            if min(width,height)<=0:raise ValueError('Invalid visible bbox')
            visibility['observable_bboxes']+=1;visible_colors[color]+=1
            values['visible_bbox_width_px'].append(width);values['visible_bbox_height_px'].append(height)
            values[f'color{color}_bbox_width_px'].append(width)
            if not item['truncated']:
                values['untruncated_bbox_short_side_px'].append(min(width,height))
                if label['camera_height_above_top_mm']==400:
                    values['400mm_untruncated_bbox_short_side_px'].append(min(width,height))
    if any(len(v)!=1 for v in split_groups.values()) or any(len(v)!=1 for v in camera_groups.values()):
        raise ValueError('Scene/camera group crosses splits')
    near=Counter();examples=[]
    for i,one in enumerate(images):
        if time.monotonic()>deadline:raise TimeoutError('Near-duplicate diagnostic budget exhausted')
        for two in images[i+1:]:
            distance=(one['hash']^two['hash']).bit_count()
            if distance>3:continue
            key='both_empty' if not one['expected_count'] and not two['expected_count'] else 'contains_target'
            near[key]+=1
            if one['split']!=two['split']:
                near['cross_split_'+key]+=1
                if len(examples)<30:
                    examples.append({'a':one['scene'],'b':two['scene'],'hamming':distance,'kind':key})
    per_input={}
    for width,height in ((224,128),(320,192),(384,224)):
        gain=min(width/640,height/360);rw,rh=round(640*gain),round(360*gain)
        per_input[f'{width}x{height}']={'source_wh':[640,360],'resized_wh':[rw,rh],
            'padding_tblr':[(height-rh)//2,height-rh-(height-rh)//2,(width-rw)//2,width-rw-(width-rw)//2],
            'gain':gain,'pad_value':114,'color':'BGR','normalization':'official NanoDet mean/std exactly once',
            '400mm_untruncated_short_side_px':stats([x*gain for x in values['400mm_untruncated_bbox_short_side_px']])}
    return {'schema':'topface-dataset-diagnostics-v1.1','status':'INTEGRITY_AND_GROUP_CHECKS_PASS',
        'observed_utc':datetime.now(timezone.utc).isoformat(),'images':len(scenes),
        'audit_sha256':sha(audit_path),'label_sha256_sequence':label_hashes,
        'world_instances':visibility['world_instances'],'count_histogram':dict(counts),
        'world_color_histogram':dict(colors),'observable_color_histogram':dict(visible_colors),
        'split_images':dict(splits),'scene_groups':len(split_groups),'camera_groups':len(camera_groups),
        'scene_and_camera_group_cross_split_count':0,'exact_duplicate_rgb':0,
        'visibility_denominators':dict(visibility),'distributions':{k:stats(v) for k,v in values.items()},
        'per_input':per_input,'perceptual_similarity':{'method':'64-bit grayscale DCTpHash <=3 Hamming; not a leakage verdict',
            'candidate_counts':dict(near),'cross_split_examples_max30':examples,
            'interpretation':'Labels retain all0..5 world objects, including unobservable/truncated targets. Similar empty images or similar renders are not automatically discarded; inspect target-bearing cross-split candidates.'},
        'training_started':False,'real_accuracy_verified':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--expected-count',type=int,default=1000)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Immutable diagnostic output required')
    result=diagnose(args.root,args.expected_count)
    atomic_json(args.output,result)
    print(json.dumps({k:result[k] for k in ('status','images','world_instances','split_images','visibility_denominators','per_input')},indent=2))
