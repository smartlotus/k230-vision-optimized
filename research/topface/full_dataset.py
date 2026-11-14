"""Materialize one hash-verified Full dataset view from frozen shard outputs."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from cloud_artifacts import extract_verified_archive
from cloud_store import atomic_json, sha
from scene_config import camera_partition


def load_shards(shards, mount_root, destination, total_scenes, base_seed,
                splits=('train', 'val'), deadline=None):
    mount_root = Path(mount_root).resolve()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    rows, sources = [], []
    seen_seeds, seen_pixels = set(), set()
    counts = Counter()
    camera_splits = {}
    next_index = 0
    for shard in sorted(shards, key=lambda value: value['start_index']):
        start, count = shard['start_index'], shard['count']
        if start != next_index or count < 1 or start + count > total_scenes:
            raise ValueError('Full archive scene ranges must be contiguous and unique')
        prefix = f'shard-{start:05d}'
        receipt = shard['publication']
        archive = shard['archive']
        if (receipt['status'] != 'DRIVE_EXPORT_FULL_BYTE_SHA_COMMITTED' or
                receipt['run_id'] != shard['run_id'] or
                any(archive[key] != receipt['uploaded'][key] for key in ('bytes', 'sha256'))):
            raise ValueError('Full shard lacks matching publication evidence')
        matches = list(mount_root.rglob(archive['name']))
        if len(matches) != 1:
            raise ValueError('Expected exactly one mounted archive per Full shard')
        path = matches[0].resolve()
        if (not path.is_relative_to(mount_root) or path.stat().st_size != archive['bytes'] or
                sha(path, deadline) != archive['sha256']):
            raise ValueError('Mounted Full archive differs from its accepted bytes')

        def select(bundle, _):
            with bundle.extractfile('pilot/AUDIT_V11.json') as stream:
                index = json.load(stream)
            if index['status'] != 'PASS' or index['images'] != count or len(index['scenes']) != count:
                raise ValueError('Full shard audit denominator differs')
            names = {'pilot/AUDIT_V11.json'}
            for offset, item in enumerate(index['scenes']):
                scene_index = start + offset
                scene = f'scene-{scene_index:05d}'
                if item['scene'] != scene or item['group'] != f'SG-{base_seed}-{scene_index:06d}':
                    raise ValueError('Scene is outside the frozen Full range')
                seed = int(np.random.SeedSequence([base_seed,scene_index]).generate_state(1)[0])
                if item['seed'] != seed:
                    raise ValueError('Full scene seed differs from its registered global index')
                if item['seed'] in seen_seeds or item['rgb_sha256'] in seen_pixels:
                    raise ValueError('Duplicate scene seed or RGB across Full shards')
                seen_seeds.add(item['seed'])
                seen_pixels.add(item['rgb_sha256'])
                split = item['split']
                camera = item['camera_group']
                heights, angle, field = camera.split('-')
                if camera_partition(float(heights[1:]), float(angle[1:]), float(field[1:])) != (camera, split):
                    raise ValueError('Full camera group does not match its assigned split')
                if camera in camera_splits and camera_splits[camera] != split:
                    raise ValueError('Full camera group crosses splits')
                camera_splits[camera] = split
                counts[split] += 1
                if split in splits:
                    names.update((f'pilot/{scene}/rgb.png', f'pilot/{scene}/labels.json'))
                rows.append({**item, 'scene': f'{prefix}/pilot/{scene}'})
            return names

        extracted = extract_verified_archive(path, destination/prefix, select=select)
        sources.append({'start_index': start, 'count': count, 'archive_sha256': archive['sha256'],
                        'member_manifest_sha256': extracted['member_manifest_sha256']})
        next_index += count
    if next_index != total_scenes or not all(counts[split] for split in ('train', 'val', 'synthetic_test')):
        raise ValueError('Full dataset coverage differs from the approved inventory')
    index = {'schema': 'topface-full-dataset-v1.1', 'status': 'PASS',
             'images': len(rows), 'instances': sum(row['instances'] for row in rows),
             'base_seed': base_seed, 'split_images': dict(counts), 'sources': sources,
             'camera_groups': len(camera_splits), 'scenes': rows}
    atomic_json(destination/'AUDIT_V11.json', index)
    return index
