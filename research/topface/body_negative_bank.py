"""Audited whole-image body-only train augmentation; world instances are retained."""
from collections import Counter,defaultdict
import copy
import json
from pathlib import Path
import re
import shutil
import zipfile

import cv2
import numpy as np

from cloud_store import atomic_json,sha
from geometry import project,top_boundary
from scene_config import camera_partition

ROLE='BODY_ONLY_TOP_OUTSIDE'
POLICY={'probability':.25,'eligibility':'ORIGINAL_WORLD_COUNT_ZERO_ONLY',
        'camera_group_match':'EXACT','real_patch_after_replacement':False}


def unpack(archive,destination,archive_sha256,manifest_sha256):
    if sha(archive)!=archive_sha256:raise ValueError('Body-only bank archive changed')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        members=bundle.infolist()
        if len(members)>1025 or sum(row.file_size for row in members)>128*1024**2:
            raise ValueError('Body-only input exceeds its frozen bounds')
        if len({row.filename for row in members})!=len(members):raise ValueError('Duplicate body bank member')
        for member in members:
            if member.is_dir() or not re.fullmatch(r'(MANIFEST\.json|negative-[0-9]{3}-(rgb\.png|labels\.json|top\.png|body\.png))',member.filename):
                raise ValueError('Unsafe body-only bank member')
            (destination/member.filename).write_bytes(bundle.read(member))
    if sha(destination/'MANIFEST.json')!=manifest_sha256:raise ValueError('Body-only manifest changed')
    return destination


class BodyOnlyBank:
    def __init__(self,root):
        self.root=Path(root).resolve()
        manifest=json.loads((self.root/'MANIFEST.json').read_text(encoding='utf8'))
        if (manifest['source_scope']!='AUDITED_SYNTHETIC_TRAIN_ONLY' or manifest['policy']!=POLICY or
                not manifest['members'] or manifest['cad']['original_topology_preserved'] is not True):
            raise ValueError('Body-only augmentation requires the frozen original-CAD train policy')
        self.groups=defaultdict(list);self.members=[]
        for member in manifest['members']:
            for row in member['files'].values():
                path=self.root/row['name']
                if (not re.fullmatch(r'negative-[0-9]{3}-(rgb\.png|labels\.json|top\.png|body\.png)',row['name']) or
                        path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']):
                    raise ValueError('Body-only source bytes differ from the audited bank')
            label=json.loads((self.root/member['files']['labels']['name']).read_text(encoding='utf8'))
            group,split=camera_partition(label['camera_height_above_top_mm'],label['plane_angle_deg'],label['hfov_deg'])
            if (split!='train' or label['split']!='train' or group!=label['camera_group_id'] or
                    group!=member['camera_group'] or label['negative_role']!=ROLE or label['expected_instance_count']!=1 or
                    len(label['instances'])!=1 or label.get('occluder') or label.get('hard_negative')):
                raise ValueError('Body-only source must retain one world object and train-only camera identity')
            item=label['instances'][0]
            if item['bbox_xyxy_visible'] is not None or item['visible_area_px']!=0 or item['geometry_valid']:
                raise ValueError('A visible top cannot be substituted as a background negative')
            transform=np.asarray(item['object_pose']);points=top_boundary([0.,0.])
            world=(np.column_stack([points,np.ones(len(points))])@transform.T)[:,:3]
            boundary=project(world,np.asarray(label['camera_intrinsics']),np.asarray(label['camera_pose_world_from_cv']))
            edge=label['negative_edge']
            margins={'left':-float(boundary[:,0].max()),'right':float(boundary[:,0].min())-640.,
                     'top':-float(boundary[:,1].max()),'bottom':float(boundary[:,1].min())-360.}
            if edge not in margins or margins[edge]<4.:
                raise ValueError('Body-only top projection intersects the image or safety margin')
            top=cv2.imread(str(self.root/member['files']['top']['name']),0)
            body=cv2.imread(str(self.root/member['files']['body']['name']),0)
            image=cv2.imread(str(self.root/member['files']['rgb']['name']))
            if (top is None or body is None or image is None or top.shape!=(360,640) or body.shape!=(360,640) or
                    image.shape!=(360,640,3) or np.any(top>=128) or int((body>=128).sum())<512):
                raise ValueError('Body-only object-ID/top masks or source pixels violate the audited contract')
            entry={'member':member,'label':label,'image':image}
            self.members.append(entry);self.groups[group].append(entry)
        if len({row['member']['scene'] for row in self.members})!=len(self.members):
            raise ValueError('Duplicate body-only source identity')
        self.identity={'manifest_sha256':sha(self.root/'MANIFEST.json'),'source_audit_sha256':manifest['source_audit_sha256'],
            'source_scope':manifest['source_scope'],'policy':POLICY,'members':len(self.members),
            'camera_groups':sorted(self.groups),'supervision':'One retained world instance; zero observable top GT'}

    def maybe_replace(self,image,label,row,rng):
        if row['split']!='train':raise ValueError('Body-only replacement is restricted to training')
        if label['instances'] or row['camera_group'] not in self.groups or rng.random()>=POLICY['probability']:
            return image,label,None
        entry=self.groups[row['camera_group']][int(rng.integers(len(self.groups[row['camera_group']])))]
        source=entry['member'];effective=copy.deepcopy(entry['label'])
        event={'base_scene':row['scene'],'base_group':row['group'],'base_world_count':0,
            'source_scene':source['scene'],'source_group':effective['scene_group_id'],
            'source_seed':effective['random_seed'],'camera_group':row['camera_group'],
            'effective_world_count':1,'observable_gt':0,'bank_manifest_sha256':self.identity['manifest_sha256'],
            'source_rgb_sha256':source['files']['rgb']['sha256'],'source_label_sha256':source['files']['labels']['sha256']}
        return entry['image'].copy(),effective,event


def build(rendered,output,archive):
    rendered,output,archive=map(Path,(rendered,output,archive))
    if output.exists() or archive.exists():raise FileExistsError('Preserve the original negative bank')
    audit=json.loads((rendered/'AUDIT_V11.json').read_text(encoding='utf8'))
    cad=json.loads((rendered/'cad_import.json').read_text(encoding='utf8'))
    if audit['status']!='PASS' or audit['images']!=120 or cad['vertices']!=9761 or cad['polygons']!=19518:
        raise ValueError('Full body-only bank requires all120 original-CAD frames audited')
    output.mkdir(parents=True)
    members=[];colors=Counter();edges=Counter();body_counts=[]
    for index,row in enumerate(audit['scenes']):
        directory=rendered/row['scene']
        if sha(directory/'rgb.png')!=row['rgb_sha256'] or sha(directory/'labels.json')!=row['label_sha256']:
            raise ValueError('Rendered source changed after label audit')
        for slot in range(1,6):
            mask=cv2.imread(str(directory/f'visible-{slot}.png'),0)
            if mask is None or np.any(mask>=128):raise ValueError('Any visible top invalidates a body-only negative')
        label=json.loads((directory/'labels.json').read_text(encoding='utf8'))
        colors[str(label['instances'][0]['color_id'])]+=1;edges[label['negative_edge']]+=1
        body_counts.append(int((cv2.imread(str(directory/'body-1.png'),0)>=128).sum()))
        files={}
        for kind,source,suffix in (('rgb','rgb.png','rgb.png'),('labels','labels.json','labels.json'),
                                   ('top','visible-1.png','top.png'),('body','body-1.png','body.png')):
            target=output/f'negative-{index:03d}-{suffix}';shutil.copy2(directory/source,target)
            files[kind]={'name':target.name,'bytes':target.stat().st_size,'sha256':sha(target)}
        members.append({'scene':label['scene_group_id'],'camera_group':row['camera_group'],'files':files})
    manifest={'schema':'topface-body-only-bank-v1.1','source_scope':'AUDITED_SYNTHETIC_TRAIN_ONLY',
        'policy':POLICY,'source_audit_sha256':sha(rendered/'AUDIT_V11.json'),'cad':cad,'members':members}
    atomic_json(output/'MANIFEST.json',manifest)
    bank=BodyOnlyBank(output)
    if len(bank.groups)!=20 or set(colors)!=set(map(str,range(1,7))) or set(edges)!={'left','right','top','bottom'}:
        raise ValueError('Negative bank camera/color/edge coverage differs')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.iterdir()):bundle.write(path,path.name)
    return {'status':'BODY_ONLY_TRAIN_BANK_MASK_AND_GEOMETRY_QUALIFIED','identity':bank.identity,
        'manifest_sha256':sha(output/'MANIFEST.json'),'archive':{'bytes':archive.stat().st_size,'sha256':sha(archive)},
        'color_counts':dict(colors),'edge_counts':dict(edges),'body_mask_pixels':{'min':min(body_counts),'max':max(body_counts)},
        'source_audit_sha256':sha(rendered/'AUDIT_V11.json'),'original_full_archives_changed':False,
        'training_effect_verified':False}
