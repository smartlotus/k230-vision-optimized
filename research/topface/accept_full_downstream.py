"""Accept provenance for Full audit and the single formal training experiment."""
import argparse
import json
import math
from pathlib import Path
import re

from cloud_store import atomic_json, sha
from publisher.plan_saved_output import plan
from prepare_full_shard import FOLDER_ID
from prepare_full_downstream import LEARNING_FILES


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def verified(frozen, collected, publication):
    intent = read(frozen/'SUBMISSION_INTENT.json')
    durable = read(publication)
    observation = frozen/f"OBSERVATION-v{durable['prep']['version']}.json"
    result_path = collected/(intent['run_id']+'-RESULT.json')
    expected = plan(frozen, observation, result_path)
    result = read(result_path)
    if (durable['status'] != 'DRIVE_EXPORT_FULL_BYTE_SHA_COMMITTED' or
            durable['secret_cleanup'] != 'CONFIRMED_DELETED' or not durable['independent_metadata_match'] or
            durable['folder_id'] != FOLDER_ID or durable['automatic_replay'] is not False or
            durable['uploaded']['provider_readback'] != 'FULL_BYTE_SHA256_PASS' or
            any(durable[key] != expected[key] for key in ('run_id', 'prep')) or
            any(durable['uploaded'][key] != expected['archive'][key] for key in ('name', 'bytes', 'sha256'))):
        raise ValueError('Exact Full downstream result lacks matching durable publication')
    return intent, result, durable


def accept_audit(frozen, collected, publication, inventory, review, output):
    if output.exists():
        raise FileExistsError('Preserve the accepted Full dataset record')
    intent, result, durable = verified(frozen, collected, publication)
    config = read(frozen/'FULL_AUDIT_CONFIG.json')
    stock = read(inventory)
    digest = sha(inventory)
    if (result['status'] != 'FULL_AUDIT_COMPLETE' or result['optimizer_updates'] != 0 or
            result['images'] != config['scenes'] or stock['accepted_scenes'] != config['scenes'] or
            result['shard_inventory_sha256'] != digest or config['shard_inventory_sha256'] != digest or
            config['shards'] != sorted(stock['shards'], key=lambda row:row['start_index']) or
            result['integrity_status'] != 'INTEGRITY_AND_GROUP_CHECKS_PASS' or
            result['scene_and_camera_group_cross_split_count'] != 0 or result['exact_duplicate_rgb'] != 0 or
            not all(result['color_coverage'].values())):
        raise ValueError('Full audit, inventory or conditional data checks differ')
    evidence = collected/'NEAR_SIMILARITY_EVIDENCE.json'
    similarity = read(evidence)
    reviewed = read(review)
    if (reviewed['status'] != 'REVIEWED_INDEPENDENT_SCENES' or reviewed['evidence_sha256'] != sha(evidence) or
            reviewed['previews'] != similarity['previews'] or
            result['similarity_review_evidence']['previews'] != similarity['previews'] or
            result['similarity_review_evidence']['pairs'] != len(similarity['pairs']) or
            len(reviewed['pairs']) != len(similarity['pairs']) or
            not similarity['all_listed_identity_checks_different']):
        raise ValueError('Full similarity evidence requires a matching actual review')
    for actual, checked in zip(similarity['pairs'], reviewed['pairs']):
        if (checked['a'], checked['b'], checked['kind']) != (actual['a']['scene'], actual['b']['scene'], actual['kind']):
            raise ValueError('Reviewed similarity pair differs from the saved evidence')
    for preview in similarity['previews']:
        path=collected/preview['name']
        if (not path.resolve().is_relative_to(collected.resolve()) or
                path.stat().st_size!=preview['bytes'] or sha(path)!=preview['sha256']):
            raise ValueError('Reviewed similarity preview bytes differ')
    value = {'schema':'topface-full-data-acceptance-v1.1', 'status':'FULL_DATASET_ACCEPTED',
             'run_id':intent['run_id'], 'shard_inventory_sha256':digest,
             'dataset_index_sha256':result['dataset_index_sha256'], 'images':result['images'],
             'instances':result['instances'], 'split_images':result['split_images'],
             'source_shards':result['source_shards'], 'color_coverage':result['color_coverage'],
             'similarity_review_sha256':sha(review), 'archive':result['archive'],
             'publication_receipt_sha256':sha(publication), 'provider':durable['prep']}
    atomic_json(output, value)
    return value


def formal_evidence(frozen, collected, publication):
    intent, result, durable = verified(frozen, collected, publication)
    config = read(frozen/'FORMAL_CONFIG.json')
    input_manifest=read(frozen/'INPUT_MANIFEST.json')
    learning_names=set(LEARNING_FILES)|{'nanodet-source.bin','nanodet-pretrained.bin'}
    learning_sources={row['path']:row['sha256'] for row in input_manifest['files'] if row['path'] in learning_names}
    config_members=[row for row in input_manifest['files'] if row['path']=='FORMAL_CONFIG.json']
    if (sha(frozen/'INPUT_MANIFEST.json')!=intent['input_manifest_sha256'] or
            set(learning_sources)!=learning_names or config['training_source_sha256']!=learning_sources or
            len(config_members)!=1 or sha(frozen/'FORMAL_CONFIG.json')!=config_members[0]['sha256']):
        raise ValueError('Formal learning sources differ from the frozen input manifest')
    trace = read(collected/'formal_full/RESULT.json')
    metrics = read(collected/'formal_full/BEST_METRICS.json')
    compact=result['compact_training_files']
    if {row['name'] for row in compact}!={'RESULT.json','BEST_METRICS.json'} or len(compact)!=2:
        raise ValueError('Formal compact inventory differs')
    for row in compact:
        path=collected/'formal_full'/row['name']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Formal compact report differs from the saved exact bytes')
    experiment=config['experiment']
    completed=result['completed_epochs']
    checkpoints=trace['checkpoints']
    if ({row['name'] for row in checkpoints}!={'best.pt','last.pt'} or len(checkpoints)!=2 or
            any(not re.fullmatch(r'[a-f0-9]{64}',row['sha256']) or row['bytes']<=0 for row in checkpoints)):
        raise ValueError('Formal best and last checkpoint inventory differs')
    best, = [row for row in checkpoints if row['name']=='best.pt']
    schedule=trace['schedule']
    batches=math.ceil(config['dataset_acceptance']['split_images']['train']/trace['batch_size'])
    if (trace['kind'] != 'formal_full' or result['status']!=trace['status'] or
            not 0 < completed <= experiment['max_epochs'] or
            result['experiment_id'] != config['experiment']['experiment_id'] or
            trace['formal_identity']!=experiment['experiment_id'] or result['session']!=config['session'] or
            trace['dataset_audit_sha256'] != config['dataset_acceptance']['dataset_index_sha256'] or
            trace['training_scene_count'] != config['dataset_acceptance']['split_images']['train'] or
            trace['validation_scene_count'] != config['dataset_acceptance']['split_images']['val'] or
            any(trace[key]!=experiment[key] for key in ('width','height','batch_size','max_epochs','ellipse_shape_loss')) or
            trace['seed']!=experiment['training_seed'] or
            trace['real_background_augmentation']!=(config.get('real_background_bank') or {}).get('identity') or
            trace['best_checkpoint_sha256']!=best['sha256'] or not 1<=trace['best_epoch']<=completed or
            schedule['epoch_batches']!=batches or schedule['target_updates']!=experiment['max_epochs']*batches or
            result['optimizer_updates']!=completed*batches or
            metrics['images'] != trace['validation_scene_count'] or
            any(result[key] != trace[key] for key in ('optimizer_updates', 'completed_epochs', 'best_epoch', 'checkpoints'))):
        raise ValueError('Formal completion or validation selection differs from its frozen experiment')
    parent=config['parent']
    if (parent['kind']=='formal_session' and
            (trace.get('resumed_from_sha256')!=parent['checkpoint_sha256'] or
             parent['experiment_config']!=experiment or
             not completed>parent['completed_epochs'])):
        raise ValueError('Formal session does not extend the exact accepted epoch checkpoint')
    return intent,result,durable,config,trace,metrics


def accept_formal_session(frozen, collected, publication, output, model_output):
    if output.exists() or model_output.exists():
        raise FileExistsError('Preserve the accepted epoch session and resume parent')
    intent,result,durable,config,trace,metrics=formal_evidence(frozen,collected,publication)
    if (result['status']!='FORMAL_SESSION_BOUNDARY' or result.get('resume_eligible') is not True or
            not config['parent'].get('completed_epochs',0)<result['completed_epochs']<config['experiment']['max_epochs']):
        raise ValueError('Only a complete, progressing, unfinished epoch boundary is resumable here')
    last,=[row for row in trace['checkpoints'] if row['name']=='last.pt']
    best,=[row for row in trace['checkpoints'] if row['name']=='best.pt']
    parent={'kind':'formal_session','experiment':'formal_full','experiment_id':result['experiment_id'],
        'experiment_config':config['experiment'],'session':result['session'],
        'completed_epochs':result['completed_epochs'],'optimizer_updates':result['optimizer_updates'],
        'checkpoint_sha256':last['sha256'],'last_checkpoint':last,'best_checkpoint':best,
        'dataset_index_sha256':config['dataset_acceptance']['dataset_index_sha256'],
        'real_background_bank':config.get('real_background_bank'),
        'training_source_sha256':config['training_source_sha256'],
        'schedule':trace['schedule'],'runtime':trace['runtime'],
        'training_input':{'ref':intent['dataset'],'version':intent['dataset_version'],
                          'manifest_sha256':intent['input_manifest_sha256']},
        'archive':result['archive'],'publication':durable}
    value={'status':'FORMAL_EPOCH_SESSION_COMPUTE_AND_ARCHIVE_VERIFIED','run_id':intent['run_id'],
        'experiment_id':result['experiment_id'],'session':result['session'],
        'completed_epochs':result['completed_epochs'],'optimizer_updates':result['optimizer_updates'],
        'resume_parent':parent,'publication_receipt_sha256':sha(publication),
        'formal_training_complete':False,'bitwise_cuda_resume_verified':False}
    atomic_json(model_output,parent)
    atomic_json(output,value)
    return value


def accept_formal(frozen, collected, publication, output, model_output):
    if output.exists() or model_output.exists():
        raise FileExistsError('Preserve the formal experiment acceptance and selected weights')
    intent,result,durable,config,trace,metrics=formal_evidence(frozen,collected,publication)
    if result['status']!='FINITE_TRAINING_COMPLETE' or result['completed_epochs']<100:
        raise ValueError('Formal final acceptance requires completed training, not an epoch session')
    best, = [row for row in trace['checkpoints'] if row['name'] == 'best.pt']
    value = {'status':'FORMAL_TRAINING_COMPUTE_AND_ARCHIVE_VERIFIED', 'run_id':intent['run_id'],
             'experiment_id':result['experiment_id'], 'optimizer_updates':result['optimizer_updates'],
             'completed_epochs':result['completed_epochs'], 'best_epoch':result['best_epoch'],
             'selected_checkpoint':best, 'validation_selection':trace['validation_selection'],
             'validation_metrics':metrics, 'agent_diagnostic_gate':result['gate'],
             'gate_interpretation':'Recorded diagnostic checks; final task quality is reported from frozen evaluation',
             'publication_receipt_sha256':sha(publication), 'archive':result['archive'], 'provider':durable['prep']}
    atomic_json(model_output, {'kind':'formal_model', 'experiment':'formal_full',
        'experiment_id':result['experiment_id'], 'session':result['session'],
        'completed_epochs':result['completed_epochs'], 'checkpoint_sha256':best['sha256'],
        'training_input':{'ref':intent['dataset'],'version':intent['dataset_version'],
                          'manifest_sha256':intent['input_manifest_sha256']},
        'real_background_bank':config.get('real_background_bank'),
        'archive':result['archive'], 'publication':durable})
    atomic_json(output, value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=('audit', 'formal','formal-session'))
    for key in ('frozen', 'collected', 'publication', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    for key in ('inventory', 'review', 'model-output'):
        parser.add_argument('--'+key, type=Path)
    arguments = vars(parser.parse_args())
    phase = arguments.pop('phase')
    if phase == 'audit':
        arguments.pop('model_output')
        value = accept_audit(**arguments)
    else:
        arguments.pop('inventory'); arguments.pop('review')
        value = (accept_formal if phase=='formal' else accept_formal_session)(**arguments)
    print(json.dumps({'status':value['status'], 'run_id':value['run_id']}, indent=2))
