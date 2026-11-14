"""Verify the focused Pilot refinement and preserve the original comparison."""
import argparse
import json
from pathlib import Path

from cloud_store import atomic_json, sha
from train_experiment import pilot_gate


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def accept(frozen, publication, original, output, parent_output):
    if output.exists() or parent_output.exists():
        raise FileExistsError('Preserve the existing geometry decision')
    intent=read(frozen/'SUBMISSION_INTENT.json')
    receipt=read(publication)
    collected=frozen/f"compact-v{receipt['prep']['version']}"
    summary=read(collected/(intent['run_id']+'-RESULT.json'))
    if (receipt['status']!='DRIVE_EXPORT_FULL_BYTE_SHA_COMMITTED' or
            receipt['secret_cleanup']!='CONFIRMED_DELETED' or not receipt['independent_metadata_match'] or
            receipt['prep']['source_sha256']!=intent['notebook_sha256'] or
            receipt['run_id']!=intent['run_id'] or summary['run_id']!=intent['run_id'] or
            summary['input_manifest_sha256']!=intent['input_manifest_sha256'] or
            any(receipt['uploaded'][key]!=summary['archive'][key] for key in ('bytes','sha256'))):
        raise ValueError('Refinement computation and publication identities differ')
    previous=read(original/'ACCEPTANCE-v1.json')
    source=read(original/'GEOMETRY_REFINEMENT_PARENT.json')
    output_record={'status':'REFINEMENT_COMPUTE_AND_ARCHIVE_VERIFIED','provider':receipt['prep'],
                   'publication_receipt_sha256':sha(publication),'archive':summary['archive'],
                   'parent_checkpoint_sha256':source['checkpoint_sha256'],
                   'full_render_admission':'HOLD','original_optimizer_updates':previous['optimizer_updates'],
                   'refinement_optimizer_updates':summary['optimizer_updates'],
                   'compute_status':summary['status'],'evaluation_scope':'SYNTHETIC_VALIDATION'}
    if summary['status']=='GEOMETRY_REFINEMENT_COMPLETE':
        pre=read(original/'compact-v1/pilot_pretrained/RESULT.json')
        directory=collected/'pilot_geometry_refine'
        metrics=read(directory/'BEST_METRICS.json')
        baseline=read(directory/'BASELINE.json')
        trace=read(directory/'RESULT.json')
        experiment,=summary['experiments']
        if (trace['initialization']['parent_sha256']!=source['checkpoint_sha256'] or
                trace['optimizer_updates']!=summary['optimizer_updates'] or
                trace['optimizer_updates']!=experiment['optimizer_updates']):
            raise ValueError('Refinement checkpoint lineage or optimizer count differs')
        for key in ('dataset_audit_sha256','training_scenes','validation_scenes','width','height'):
            if trace[key]!=pre[key]:
                raise ValueError(f'Refinement differs from the original data and input in {key}')
        prior=read(original/'compact-v1/pilot_pretrained/BEST_METRICS.json')
        comparable=('task_mAP50','task_mAP50_95','matched_color_accuracy','class_agnostic_recall50')
        if any(abs(baseline[key]-prior[key])>1e-4 for key in comparable):
            raise ValueError('Refinement baseline does not reproduce the frozen Pilot model')
        gate=pilot_gate(metrics)
        if gate!=experiment['gate']:
            raise ValueError('Refinement gate differs from the saved metrics')
        output_record.update(gate=gate,best_epoch=experiment['best_epoch'],
                             baseline={key:baseline[key] for key in (*comparable,'angle_diagnostics')},
                             metrics={key:metrics[key] for key in (*comparable,'denominators','errors','count_groups','angle_diagnostics')})
        if gate['pass']:
            identity=next(item for item in experiment['checkpoints'] if item['name']=='best.pt')
            output_record.update(full_render_admission='PASS',selected_experiment='pilot_geometry_refine',
                                 selected_checkpoint=identity)
            atomic_json(parent_output,{'kind':'pilot','experiment':'pilot_geometry_refine',
                'checkpoint_sha256':identity['sha256'],'archive':summary['archive'],'publication':receipt})
    atomic_json(output,output_record)
    return output_record


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('frozen','publication','original','output','parent-output'):
        parser.add_argument('--'+name,type=Path,required=True)
    result=accept(**vars(parser.parse_args()))
    print(json.dumps({key:result.get(key) for key in ('status','compute_status','best_epoch',
          'refinement_optimizer_updates','full_render_admission')},indent=2))
