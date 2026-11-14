"""Verify one completed render shard and add it to the Full inventory."""
import argparse
import json
from pathlib import Path

from cloud_store import atomic_json, sha


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def accept(frozen, result, publication, inventory):
    intent=read(frozen/'SUBMISSION_INTENT.json')
    config=read(frozen/'FULL_RENDER_CONFIG.json')
    receipt=read(publication)
    computed=read(result)
    run_id=intent['run_id']
    if computed['status']!='COMPUTE_AND_QA_PASS' or computed['images_rendered']!=config['count']:
        raise ValueError('Full shard compute or label audit did not pass')
    if not computed['device']['gpu_render_verified'] or computed['device']['successful_frames']!=config['count']:
        raise ValueError('Full shard GPU confirmation is missing')
    if (computed['run_id'],computed['start_index'],computed['seed'],computed['input_manifest_sha256']) != (
        run_id,config['start_index'],config['seed'],intent['input_manifest_sha256']
    ):
        raise ValueError('Rendered content differs from the frozen Full scene range')
    if (receipt['run_id']!=run_id or receipt['status']!='DRIVE_EXPORT_FULL_BYTE_SHA_COMMITTED' or
            receipt['secret_cleanup']!='CONFIRMED_DELETED' or not receipt['independent_metadata_match'] or
            receipt['prep']['source_sha256']!=intent['notebook_sha256'] or
            any(computed['archive'][key]!=receipt['uploaded'][key] for key in ('bytes','sha256'))):
        raise ValueError('Full shard publication differs from the rendered artifact')
    entry={'run_id':run_id,'start_index':config['start_index'],'count':config['count'],
           'seed':config['seed'],'archive':computed['archive'],'publication':receipt,
           'saved_result_sha256':sha(result),'instances':computed['instances'],
           'elapsed_seconds':computed['elapsed_seconds']}
    value=read(inventory) if inventory.exists() else {'schema':'topface-full-inventory-v1.1',
        'total_planned':config['total_scenes'],'seed':config['seed'],'shards':[]}
    if (value['seed'],value['total_planned'])!=(config['seed'],config['total_scenes']):
        raise ValueError('Shard belongs to a different Full dataset')
    start,end=entry['start_index'],entry['start_index']+entry['count']
    for old in value['shards']:
        if old==entry:
            return value
        if start<old['start_index']+old['count'] and old['start_index']<end:
            raise ValueError('A previously accepted shard already covers this range')
    value['shards'].append(entry)
    value['shards'].sort(key=lambda shard:shard['start_index'])
    value['accepted_scenes']=sum(shard['count'] for shard in value['shards'])
    atomic_json(inventory,value)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('frozen','result','publication','inventory'):
        parser.add_argument('--'+key,type=Path,required=True)
    value=accept(**vars(parser.parse_args()))
    print(json.dumps({'accepted_scenes':value['accepted_scenes'],'planned':value['total_planned']}))
