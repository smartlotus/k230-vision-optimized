"""Record the Pilot decision from saved metrics and a verified Drive archive."""
import argparse
import json
from pathlib import Path

import numpy as np

from cloud_store import atomic_json, sha
from train_experiment import pilot_gate, sanity_gate


KINDS = ('overfit', 'pilot_pretrained', 'pilot_random_control')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def accept(frozen, collected, publication, output, parent_output=None):
    if output.exists() or (parent_output is not None and parent_output.exists()):
        raise FileExistsError('Preserve the accepted Pilot decision')
    intent = read(frozen/'SUBMISSION_INTENT.json')
    run_id = intent['run_id']
    receipt = read(publication)
    summary = read(collected/(run_id+'-RESULT.json'))
    observation = read(frozen/f"OBSERVATION-v{receipt['prep']['version']}.json")
    source = read(frozen/f"EXECUTED_SOURCE-v{receipt['prep']['version']}.json")
    if (receipt['status'] != 'DRIVE_EXPORT_FULL_BYTE_SHA_COMMITTED' or
            receipt['secret_cleanup'] != 'CONFIRMED_DELETED' or not receipt['independent_metadata_match']):
        raise ValueError('Pilot must have a verified Drive receipt')
    if not source['exact_source_cells_match'] or intent['input_manifest_sha256'] != summary['input_manifest_sha256']:
        raise ValueError('Saved Pilot execution differs from the frozen input')
    if (observation['status'], observation['ref'], observation['kernel_id'], observation['version']) != (
        'COMPLETE', receipt['prep']['ref'], receipt['prep']['kernel_id'], receipt['prep']['version']
    ) or receipt['prep']['source_sha256'] != intent['notebook_sha256']:
        raise ValueError('Pilot provider identity differs from the archive receipt')
    if receipt['run_id'] != run_id or summary['run_id'] != run_id or any(
        summary['archive'][key] != receipt['uploaded'][key] for key in ('bytes', 'sha256')
    ):
        raise ValueError('Pilot archive and result do not refer to the same bytes')
    result = {'status':'PILOT_COMPUTE_AND_ARCHIVE_VERIFIED','run_id':run_id,
              'provider':receipt['prep'],'publication_receipt_sha256':sha(publication),
              'archive':summary['archive'],'compute_status':summary['status'],
              'optimizer_updates':summary['optimizer_updates'], 'full_render_admission':'HOLD',
              'evaluation_scope':'SYNTHETIC_VALIDATION', 'experiments':[]}
    if summary['status']=='FINITE_PILOT_TRAINING_SEQUENCE_COMPLETE':
        if [item['kind'] for item in summary['experiments']] != list(KINDS):
            raise ValueError('The three frozen Pilot experiments are required')
        traces = {}
        for experiment in summary['experiments']:
            kind = experiment['kind']
            metrics = read(collected/kind/'BEST_METRICS.json')
            trace = read(collected/kind/'RESULT.json')
            if trace['optimizer_updates'] != experiment['optimizer_updates']:
                raise ValueError('Pilot trace update count differs from its summary')
            gate = sanity_gate(metrics) if kind=='overfit' else pilot_gate(metrics)
            if gate != experiment['gate']:
                raise ValueError('Pilot gate cannot be reproduced from saved metrics')
            traces[kind] = trace
            numbers = {key:metrics[key] for key in ('images','task_mAP50','task_mAP50_95',
                'matched_color_accuracy','class_agnostic_recall50','errors','count_groups','denominators')}
            result['experiments'].append({'kind':kind,'gate':gate,'best_epoch':experiment['best_epoch'],
                                          'optimizer_updates':experiment['optimizer_updates'],
                                          'validation_score':experiment['best_validation_score'],
                                          'metrics':numbers})
        pre, random = (traces[kind] for kind in KINDS[1:])
        for key in ('width','height','batch_size','seed','max_epochs','learning_rate','loss_weights',
                    'training_scenes','validation_scenes','dataset_audit_sha256'):
            if pre[key] != random[key]:
                raise ValueError(f'Pilot control differs in {key}')
        if pre['max_epochs'] != 20 or pre['epoch'] != 20 or random['epoch'] != 20:
            raise ValueError('Pilot controls did not finish the same 20-epoch budget')
        if sum(trace['optimizer_updates'] for trace in traces.values()) != summary['optimizer_updates']:
            raise ValueError('Pilot update denominator differs')
        result['controlled_comparison'] = 'SAME_DATA_AND_OPTIMIZER_BUDGET'
        result['measured_pretrained_epoch_seconds'] = {
            label: float(np.median([row[label] for row in pre['epoch_records']]))
            for label in ('training_wall_seconds','validation_wall_seconds')
        }
        candidates = [row for row in result['experiments'][1:] if row['gate']['pass']]
        if result['experiments'][0]['gate']['pass'] and candidates:
            chosen = max(candidates, key=lambda row:row['validation_score'])
            identity = next(row for row in traces[chosen['kind']]['checkpoints'] if row['name']=='best.pt')
            result.update(full_render_admission='PASS', selected_experiment=chosen['kind'],
                          selected_checkpoint=identity)
            if parent_output is not None:
                atomic_json(parent_output, {'kind':'pilot','experiment':chosen['kind'],
                    'checkpoint_sha256':identity['sha256'],'archive':summary['archive'],'publication':receipt})
    atomic_json(output,result)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    for name in ('frozen','collected','publication','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--parent-output',type=Path)
    result = accept(**vars(parser.parse_args()))
    print(json.dumps({key: result.get(key) for key in ('status','optimizer_updates',
          'selected_experiment','full_render_admission')},indent=2))
