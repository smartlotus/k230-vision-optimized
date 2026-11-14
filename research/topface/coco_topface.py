"""Export the audited visible top boxes in COCO detection format."""
from collections import defaultdict
import json
from pathlib import Path

from cloud_store import atomic_json
from scene_config import COLORS


def export(root, audit):
    root = Path(root)
    categories = [{'id': key, 'name': value[0], 'supercategory': 'top_face'}
                  for key, value in COLORS.items()]
    splits = defaultdict(lambda: {'images': [], 'annotations': [], 'categories': categories})
    annotation_id = 0
    for image_id, row in enumerate(audit['scenes'], start=1):
        scene = row['scene']
        label = json.loads((root/scene/'labels.json').read_text(encoding='utf8'))
        split = splits[row['split']]
        split['images'].append({'id': image_id, 'file_name': scene+'/rgb.png',
                                'width': label['width'], 'height': label['height']})
        for instance in label['instances']:
            box = instance['bbox_xyxy_visible']
            if box is None:
                continue
            x1, y1, x2, y2 = box
            annotation_id += 1
            split['annotations'].append({'id': annotation_id, 'image_id': image_id,
                                         'category_id': instance['color_id'],
                                         'bbox': [x1, y1, x2-x1, y2-y1],
                                         'area': instance['visible_area_px'], 'iscrowd': 0,
                                         'source_instance_id': instance['instance_id']})
    for split, data in splits.items():
        atomic_json(root/f'instances_{split}.json', data)
