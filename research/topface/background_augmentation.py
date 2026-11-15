"""Paste reviewed real-train background crops outside every synthetic top box."""
import json
from pathlib import Path
import re
import zipfile

import cv2
import numpy as np

from cloud_store import sha


def overlap(one,two):
    return one[0]<two[2] and two[0]<one[2] and one[1]<two[3] and two[1]<one[3]


def expanded(box,margin):
    return [box[0]-margin,box[1]-margin,box[2]+margin,box[3]+margin]


def unpack(archive,destination,archive_sha256,manifest_sha256):
    if sha(archive)!=archive_sha256:
        raise ValueError('Real background bank archive differs from its frozen input')
    destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        members=bundle.infolist()
        if len(members)>129 or sum(row.file_size for row in members)>8*1024**2:
            raise ValueError('Background bank exceeds its input bounds')
        if len({row.filename for row in members})!=len(members):
            raise ValueError('Background bank contains duplicate names')
        for member in members:
            if not re.fullmatch(r'(MANIFEST\.json|patch-[0-9]{3}\.png)',member.filename) or member.is_dir():
                raise ValueError('Unsafe background bank member')
            path=destination/member.filename
            with path.open('xb') as stream:stream.write(bundle.read(member))
    if sha(destination/'MANIFEST.json')!=manifest_sha256:
        raise ValueError('Background manifest differs from the frozen training policy')
    return destination


class BackgroundBank:
    def __init__(self,root):
        root=Path(root)
        manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf8'))
        if manifest['source_scope']!='APPROVED_REAL_TRAIN_ONLY' or not manifest['members']:
            raise ValueError('Background crops require approved train-only provenance')
        sources={row['image_id']:row for row in manifest['sources']}
        if len(sources)!=len(manifest['sources']) or any(row['split']!='train' for row in sources.values()):
            raise ValueError('Validation or test images cannot supply training background crops')
        self.patches=[]
        for member in manifest['members']:
            source=sources[member['source_image_id']]
            if any(overlap(member['source_xyxy'],expanded(box,manifest['source_exclusion_margin_px']))
                   for box in source['approved_top_boxes_xyxy']):
                raise ValueError('A background crop intersects an approved source target')
            if not re.fullmatch(r'patch-[0-9]{3}\.png',member['name']):
                raise ValueError('Unexpected background crop name')
            path=root/member['name']
            if path.stat().st_size!=member['bytes'] or sha(path)!=member['sha256']:
                raise ValueError('Background crop bytes differ from the reviewed source bank')
            patch=cv2.imread(str(path))
            if patch is None or patch.dtype!=np.uint8 or list(patch.shape)!=member['shape_hwc']:
                raise ValueError('Background crop pixel contract differs')
            self.patches.append(patch)
        self.policy=manifest['augmentation_policy']
        self.identity={'manifest_sha256':sha(root/'MANIFEST.json'),
            'source_manifest_sha256':manifest['source_manifest_sha256'],
            'source_scope':manifest['source_scope'],'sources':sorted(sources),
            'patch_count':len(self.patches),'policy':self.policy,
            'supervision':'Existing synthetic labels only; no real geometric truth inferred'}

    def apply(self,image,instances,rng):
        if image.shape!=(360,640,3) or image.dtype!=np.uint8:
            raise ValueError('Expected a source-size synthetic RGB image in BGR order')
        operations=[]
        if rng.random()>=self.policy['probability']:
            return image,operations
        protected=[expanded(row['bbox_xyxy_visible'],self.policy['target_exclusion_margin_px'])
                   for row in instances if row['bbox_xyxy_visible'] is not None]
        for _ in range(int(rng.integers(1,self.policy['max_patches']+1))):
            patch=self.patches[int(rng.integers(len(self.patches)))]
            factor=float(rng.uniform(*self.policy['scale_range']))
            width,height=max(8,round(patch.shape[1]*factor)),max(8,round(patch.shape[0]*factor))
            if width>640 or height>360:
                continue
            for _ in range(self.policy['placement_attempts']):
                left=int(rng.integers(0,640-width+1));top=int(rng.integers(0,360-height+1))
                rectangle=[left,top,left+width,top+height]
                if any(overlap(rectangle,box) for box in protected):
                    continue
                resized=cv2.resize(patch,(width,height),interpolation=cv2.INTER_LINEAR)
                x=np.minimum(np.arange(width),width-1-np.arange(width))/6
                y=np.minimum(np.arange(height),height-1-np.arange(height))/6
                alpha=(np.clip(y,0,1)[:,None]*np.clip(x,0,1)[None,:])[:,:,None]
                region=image[top:top+height,left:left+width]
                region[:]=np.clip(region*(1-alpha)+resized*alpha,0,255).astype(np.uint8)
                operations.append(rectangle)
                break
        return image,operations
