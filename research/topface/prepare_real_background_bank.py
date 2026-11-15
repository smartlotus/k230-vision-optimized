"""Extract no-top background crops exclusively from approved historical train frames."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np

from background_augmentation import BackgroundBank,expanded,overlap
from cloud_store import atomic_json,sha


ROOT=Path(__file__).resolve().parents[2]
R4_SHA256='6d21a2e9d0a251c1161ee3b63345909d64ad1f8cda6d3f9be0e0c0454843e198'


def prepare(output,report):
    project=ROOT/'k230_transport_mvp'
    sys.path.insert(0,str(project))
    from tools.material_dataset import review_gate,verified_rgb
    path=project/'data/ai_pilot/material_manifest_reviewed_r4_20261110.json'
    if sha(path)!=R4_SHA256:
        raise ValueError('The approved source manifest differs')
    if output.exists() or report.exists():
        raise FileExistsError('Preserve the existing immutable crop bank')
    output.mkdir(parents=True)
    sources=[];members=[]
    for record in json.loads(path.read_text(encoding='utf8'))['images']:
        if record.get('split')!='train':
            continue
        reasons,objects=review_gate(record)
        if reasons:
            continue
        image=np.ascontiguousarray(verified_rgb(project,record)[:,:,::-1])
        boxes=[]
        for item in objects:
            x,y,w,h=item['bbox_xywh'];boxes.append([x,y,x+w,y+h])
        source={'image_id':record['image_id'],'split':'train','scene_group':record['scene_group_id'],
                'native_sha256':record['native_sha256'],'pixel_sha256':record['rgb_pixel_sha256'],
                'approved_top_boxes_xyxy':boxes,'review_status':'APPROVED_ALL_VISIBLE_TOPS_CHECKED'}
        sources.append(source)
        candidates=[]
        for y in sorted(set([*range(0,265,64),264])):
            for x in sorted(set([*range(0,545,64),544])):
                rectangle=[x,y,x+96,y+96]
                if any(overlap(rectangle,expanded(box,12)) for box in boxes):
                    continue
                patch=image[y:y+96,x:x+96]
                gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
                score=float(np.std(gray))
                candidates.append((score,x,y,patch))
        if len(candidates)<8:
            raise ValueError('Insufficient safely reviewed background crops')
        for score,x,y,patch in sorted(candidates,key=lambda row:(-row[0],row[2],row[1]))[:16]:
            name=f'patch-{len(members)+1:03d}.png'
            if not cv2.imwrite(str(output/name),patch):
                raise RuntimeError('Background crop encoding failed')
            members.append({'name':name,'bytes':(output/name).stat().st_size,'sha256':sha(output/name),
                'shape_hwc':list(patch.shape),'source_image_id':record['image_id'],
                'source_xyxy':[x,y,x+96,y+96],'texture_std':score})
    if len(sources)!=4 or len(members)!=64:
        raise ValueError('Expected exactly four eligible real-train frames and64 crops')
    manifest={'schema':'topface-reviewed-background-bank-v1.1','source_scope':'APPROVED_REAL_TRAIN_ONLY',
        'source_manifest_sha256':R4_SHA256,'source_exclusion_margin_px':12,'sources':sources,'members':members,
        'augmentation_policy':{'probability':.35,'max_patches':2,'scale_range':[.75,1.25],
            'target_exclusion_margin_px':12,'placement_attempts':24,'feather_border_px':6},
        'geometry_supervision':'Original synthetic CAD labels only; source real center/ellipse remain UNKNOWN'}
    atomic_json(output/'MANIFEST.json',manifest)
    bank=BackgroundBank(output)
    protected=[{'bbox_xyxy_visible':[240,100,400,260]}]
    changed=0;operations=0
    for seed in range(100):
        original=np.full((360,640,3),127,np.uint8)
        augmented,rectangles=bank.apply(original.copy(),protected,np.random.default_rng(seed))
        if not np.array_equal(augmented[88:272,228:412],original[88:272,228:412]):
            raise ValueError('Background augmentation changed a protected target region')
        again,repeat=bank.apply(original.copy(),protected,np.random.default_rng(seed))
        if not np.array_equal(augmented,again) or repeat!=rectangles:
            raise ValueError('Background augmentation is not deterministic')
        changed+=int(not np.array_equal(augmented,original));operations+=len(rectangles)
    archive=output.with_suffix('.bin')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
        for item in sorted(output.iterdir()):bundle.write(item,item.name)
    value={'status':'REAL_TRAIN_BACKGROUND_BANK_QUALIFIED','source_images':4,'source_crops':64,
        'source_scope':'APPROVED_REAL_TRAIN_ONLY','source_manifest_sha256':R4_SHA256,
        'archive':{'name':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive)},
        'manifest_sha256':sha(output/'MANIFEST.json'),'identity':bank.identity,
        'qualification':{'seeds':100,'protected_target_pixels_unchanged':True,'deterministic':True,
            'images_with_paste':changed,'patches_pasted':operations},
        'reason':'Pilot real-val false positives occurred on mechanical fixtures/cables. Train-only reviewed background crops supply targeted negative textures.',
        'real_val_or_test_used_for_training':False,'optimizer_updates':0}
    atomic_json(report,value)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())),indent=2))
