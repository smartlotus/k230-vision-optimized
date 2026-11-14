"""Bounded cloud-side review artifacts for original pHash candidates."""
import json
from pathlib import Path
import cv2
import numpy as np
from cloud_store import atomic_json,sha


def build(root,diagnostics,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    pairs=diagnostics['perceptual_similarity']['cross_split_examples_max30']
    evidence=[];pages=[];tiles=[]
    for number,pair in enumerate(pairs):
        images=[];labels=[]
        for scene in (pair['a'],pair['b']):
            directory=(root/scene).resolve()
            if not directory.is_relative_to(root.resolve()):raise ValueError('Unsafe similarity scene path')
            label=json.loads((directory/'labels.json').read_text(encoding='utf8'))
            image=cv2.imread(str(directory/'rgb.png'))
            labels.append(label);images.append(image)
        one,two=labels
        pixel_mae=float(np.mean(np.abs(images[0].astype(np.float32)-images[1].astype(np.float32))))
        comparisons={'scene_group_different':one['scene_group_id']!=two['scene_group_id'],
            'seed_different':one['random_seed']!=two['random_seed'],
            'camera_group_different':one['camera_group_id']!=two['camera_group_id'],
            'rgb_sha_different':sha(root/pair['a']/'rgb.png')!=sha(root/pair['b']/'rgb.png')}
        evidence.append({**pair,'identity_checks':comparisons,'mean_absolute_rgb_byte_difference':pixel_mae,
            'a':{'scene':pair['a'],'split':one['split'],'scene_group':one['scene_group_id'],'seed':one['random_seed'],
                 'camera_group':one['camera_group_id'],'camera_intrinsics':one['camera_intrinsics'],
                 'camera_pose_world_from_cv':one['camera_pose_world_from_cv'],'objects':[{'id':i['instance_id'],
                     'color_id':i['color_id'],'object_pose':i['object_pose']} for i in one['instances']]},
            'b':{'scene':pair['b'],'split':two['split'],'scene_group':two['scene_group_id'],'seed':two['random_seed'],
                 'camera_group':two['camera_group_id'],'camera_intrinsics':two['camera_intrinsics'],
                 'camera_pose_world_from_cv':two['camera_pose_world_from_cv'],'objects':[{'id':i['instance_id'],
                     'color_id':i['color_id'],'object_pose':i['object_pose']} for i in two['instances']]}})
        halves=[]
        for image,label,scene in zip(images,labels,(pair['a'],pair['b'])):
            small=cv2.resize(image,(320,180),interpolation=cv2.INTER_AREA)
            panel=np.zeros((208,320,3),np.uint8);panel[28:]=small
            cv2.putText(panel,scene+' '+label['split'],(3,18),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
            halves.append(panel)
        tiles.append(np.concatenate(halves,axis=1))
        if len(tiles)==8 or number==len(pairs)-1:
            name='near-similarity-%02d.jpg'%(len(pages)+1)
            cv2.imwrite(str(output/name),np.concatenate(tiles,axis=0),[cv2.IMWRITE_JPEG_QUALITY,80])
            pages.append({'name':name,'bytes':(output/name).stat().st_size,'sha256':sha(output/name)});tiles=[]
    report={'schema':'topface-near-similarity-review-input-v1.1','status':'EVIDENCE_READY_NOT_AUTOMATIC_PASS',
        'pairs':evidence,'previews':pages,'all_listed_identity_checks_different':all(all(p['identity_checks'].values()) for p in evidence),
        'candidate_counts':diagnostics['perceptual_similarity']['candidate_counts'],
        'scope':'independent seed/scene/camera/pixel witnesses plus bounded actual-image previews; pHash similarity is not a leakage verdict',
        'original_images_or_labels_modified':False}
    atomic_json(output/'NEAR_SIMILARITY_EVIDENCE.json',report);return report
