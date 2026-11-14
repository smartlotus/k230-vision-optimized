"""Join immutable cloud datasets without changing original labels/splits."""
import json
from pathlib import Path
import shutil

from cloud_store import atomic_json,sha


def combine(sources,destination):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=False)
    groups=set();seeds=set();pixels=set();camera_splits={};rows=[];parents=[]
    for prefix,source,archive_sha256 in sources:
        if not prefix.isalnum():raise ValueError('Safe explicit source prefix required')
        source=Path(source).resolve();audit_path=source/'AUDIT_V11.json'
        audit=json.loads(audit_path.read_text(encoding='utf8'))
        if audit['status']!='PASS':raise ValueError('Cannot combine failed source QA')
        parents.append({'prefix':prefix,'archive_sha256':archive_sha256,'audit_sha256':sha(audit_path),'images':audit['images']})
        for row in audit['scenes']:
            directory=(source/row['scene']).resolve()
            if not directory.is_relative_to(source):raise ValueError('Unsafe source scene')
            label_path=directory/'labels.json'
            if sha(label_path)!=row['label_sha256'] or sha(directory/'rgb.png')!=row['rgb_sha256']:
                raise ValueError('Source bytes changed before merge')
            label=json.loads(label_path.read_text(encoding='utf8'))
            if label['scene_group_id'] in groups or label['random_seed'] in seeds or row['rgb_sha256'] in pixels:
                raise ValueError('Duplicated scene/seed/pixels across parent datasets')
            groups.add(label['scene_group_id']);seeds.add(label['random_seed']);pixels.add(row['rgb_sha256'])
            camera=label['camera_group_id'];split=label['split']
            if camera in camera_splits and camera_splits[camera]!=split:raise ValueError('Camera group crosses parent splits')
            camera_splits[camera]=split
            scene=prefix+'/'+row['scene'];target=destination/scene
            # The cloud has the full source archive and original members. Copy
            # into the composite view; do not relabel source files or use links
            # resolving outside the audited dataset root.
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(directory,target)
            rows.append({**row,'scene':scene,'source_dataset':prefix})
    result={'schema':'topface-immutable-composite-dataset-v1.1','status':'PASS','images':len(rows),
        'instances':sum(r['instances'] for r in rows),'scenes':rows,'parents':parents,
        'unique_scene_seeds':len(seeds),'unique_scene_groups':len(groups),
        'camera_group_cross_split_count':0,'original_labels_modified':False,'optimizer_updates':0}
    atomic_json(destination/'AUDIT_V11.json',result);return result
