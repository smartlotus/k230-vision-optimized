"""Independent 02 dataset audit / 03 finite Pilot training cloud workers.

The worker is prepared before its inputs exist, but refuses execution without
the original 1000-scene compute/storage acceptance. No Full admission here.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time

from cloud_artifacts import fetch_archive,extract_verified_archive
from cloud_store import Drive,archive_directory,atomic_json,sha
from native_job import verify_inputs


def main(input_root,run_id,phase,max_seconds=7200):
    if phase not in ('audit','training') or not run_id.replace('-','').isalnum():
        raise ValueError('Explicit finite downstream identity/phase required')
    work=Path('/kaggle/working')
    if not work.exists():raise ValueError('Downstream producer must run inside Kaggle')
    input_root=Path(input_root).resolve();manifest=verify_inputs(input_root)
    config=json.loads((input_root/'DOWNSTREAM_CONFIG.json').read_text(encoding='utf8'))
    if (config['run_id'],config['phase'])!=(run_id,phase):raise ValueError('Frozen downstream identity mismatch')
    started=time.monotonic();deadline=started+max_seconds
    compute_deadline=deadline-600
    directory=work/run_id;directory.mkdir(exist_ok=False)
    result=directory/'result';result.mkdir()
    state={'schema':'topface-downstream-v1.1','run_id':run_id,'phase':phase,'status':'STARTED',
        'input_manifest_sha256':sha(input_root/'INPUT_MANIFEST.json'),'automatic_replay':False,
        'started_utc':datetime.now(timezone.utc).isoformat(),'max_seconds':max_seconds,
        'optimizer_updates':0,'board_run':False,'real_evaluation':False}
    drive=None
    def stage(name):
        state['stage']=name;state['elapsed_seconds']=time.monotonic()-started
        atomic_json(result/'STATE.json',state)
        print('TOPFACE_DOWNSTREAM_STAGE',json.dumps({k:state[k] for k in ('run_id','stage','elapsed_seconds')}),flush=True)
    try:
        stage('STORAGE_AND_PARENT_ADMISSION')
        if shutil.disk_usage(work).free<8*1024**3:raise ValueError('Cloud working disk below registered8GiB cap')
        drive=Drive(config['folder_id'],deadline=deadline);drive.verify_folder()
        atomic_json(result/'probe.json',{'run_id':run_id,'phase':phase,'producer':'Kaggle'})
        state['storage_probe']=drive.export(result/'probe.json',run_id+'-probe',result/'DRIVE_PROBE.json')
        state['admission_marker']=drive.reserve_run(run_id,state['input_manifest_sha256'])
        parent=config['pilot_compute']
        if parent['status']!='COMPUTE_AND_QA_PASS' or parent['images_rendered']!=1000 or parent['durable_status']!='FULL_BYTE_SHA256_PASS':
            raise ValueError('Exact1000-scene Pilot compute/QA/durable acceptance required')
        if parent['run_id']!=config['pilot_run_id']:raise ValueError('Pilot parent identity mismatch')
        stage('FETCH_AND_VERIFY_CLOUD_PILOT')
        receipt=parent['drive_archive']
        fetch_archive(drive,receipt,directory/'pilot.tar.gz')
        state['parent_members']=extract_verified_archive(directory/'pilot.tar.gz',directory/'pilot-source')
        pilot=directory/'pilot-source/pilot'
        audit=json.loads((pilot/'AUDIT_V11.json').read_text(encoding='utf8'))
        if audit['status']!='PASS' or audit['images']!=1000:raise ValueError('Parent label audit mismatch')
        state['pilot_archive_sha256']=receipt['sha256'];state['pilot_audit_sha256']=sha(pilot/'AUDIT_V11.json')
        expected_count=1000
        if config.get('supplement_compute'):
            supplement=config['supplement_compute'];extra_receipt=supplement['drive_archive']
            if supplement['status']!='COMPUTE_AND_QA_PASS' or supplement['images_rendered']!=240 or supplement['durable_status']!='FULL_BYTE_SHA256_PASS':
                raise ValueError('Original240-scene supplement compute/QA/storage acceptance required')
            fetch_archive(drive,extra_receipt,directory/'supplement.tar.gz')
            state['supplement_members']=extract_verified_archive(directory/'supplement.tar.gz',directory/'supplement-source')
            from combine_datasets import combine
            composite=directory/'composite-pilot'
            combined=combine([('primary',pilot,receipt['sha256']),
                ('colorfix',directory/'supplement-source/pilot',extra_receipt['sha256'])],composite)
            if combined['images']!=1240:raise ValueError('Composite Pilot denominator differs')
            pilot=composite;expected_count=1240
            state['supplement_archive_sha256']=extra_receipt['sha256']
            state['composite_audit_sha256']=sha(pilot/'AUDIT_V11.json')
        source=result/'source';source.mkdir()
        for member in manifest['files']:
            path=input_root/member['path']
            if path.suffix in ('.py','.json','.yml','.yaml'):
                target=source/member['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        stage('INDEPENDENT_DATASET_DIAGNOSTICS')
        from dataset_diagnostics import diagnose
        diagnostics=diagnose(pilot,expected_count);atomic_json(result/'DATASET_DIAGNOSTICS.json',diagnostics)
        # Similarity is a review candidate. Do not silently delete frames or
        # turn grayscale similarity into a fabricated provenance violation.
        near=diagnostics['perceptual_similarity']['candidate_counts'].get('cross_split_contains_target',0)
        if phase=='audit':
            conditional=diagnostics['conditional_color_coverage']['all_split_count_pairs_have_observable_six']
            state.update(status='AUDIT_COMPLETE',images=expected_count,diagnostics={k:diagnostics[k] for k in (
                'images','world_instances','split_images','visibility_denominators','per_input','perceptual_similarity',
                'conditional_color_coverage','world_color_by_split_and_count')},
                training_admission=('NEEDS_COLOR_COUNT_COVERAGE' if not conditional else
                    'NEEDS_SIMILARITY_REVIEW' if near else 'INTEGRITY_AND_GROUP_PASS'))
        else:
            audit_acceptance=config['audit_acceptance']
            if audit_acceptance['pilot_archive_sha256']!=receipt['sha256'] or audit_acceptance['training_admission']!='PASS':
                raise ValueError('Independent02 audit acceptance missing/different')
            if not diagnostics['conditional_color_coverage']['all_split_count_pairs_have_observable_six']:
                raise ValueError('Count/color confounding coverage is still incomplete')
            if config.get('supplement_compute') and audit_acceptance.get('supplement_archive_sha256')!=state['supplement_archive_sha256']:
                raise ValueError('Independent02 acceptance does not bind supplemental archive')
            if near and not audit_acceptance.get('similarity_review_passed'):
                raise ValueError('Target-bearing cross-split similarity review required')
            import torch
            if not torch.cuda.is_available():raise ValueError('PyTorch CUDA unavailable before optimizer admission')
            state['environment']={'torch':str(torch.__version__),'cuda':torch.version.cuda,
                'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                'training_device':'cuda:0','amp':False}
            from train_experiment import train,choose_overfit
            stage('OVERFIT_TRAIN_SANITY')
            selection=choose_overfit(pilot)
            atomic_json(result/'OVERFIT_SELECTION.json',selection)
            experiments=[]
            overfit=train(pilot,result/'overfit',input_root/'nanodet-pretrained.bin',kind='overfit',
                max_epochs=200,max_updates=1000,deadline=compute_deadline,selected_scenes=selection['scenes'])
            experiments.append(overfit);state['optimizer_updates']+=overfit['optimizer_updates']
            if not overfit['gate']['pass']:
                state['status']='OVERFIT_GATE_FAILED_NO_PILOT_TRAINING'
            else:
                stage('PILOT_PRETRAINED_20_EPOCHS')
                pretrained=train(pilot,result/'pilot_pretrained',input_root/'nanodet-pretrained.bin',
                    kind='pilot_pretrained',max_epochs=20,deadline=compute_deadline)
                experiments.append(pretrained);state['optimizer_updates']+=pretrained['optimizer_updates']
                if pretrained['status']!='FINITE_TRAINING_COMPLETE':
                    state['status']='PILOT_BUDGET_STOP_NO_AUTOMATIC_REPLAY'
                else:
                    stage('PILOT_RANDOM_CONTROL_20_EPOCHS')
                    random=train(pilot,result/'pilot_random_control',input_root/'nanodet-pretrained.bin',
                        kind='pilot_random_control',max_epochs=20,deadline=compute_deadline)
                    experiments.append(random);state['optimizer_updates']+=random['optimizer_updates']
                    state['status']='FINITE_PILOT_TRAINING_SEQUENCE_COMPLETE' if random['status']=='FINITE_TRAINING_COMPLETE' else 'CONTROL_BUDGET_STOP'
            state['experiments']=[{k:e.get(k) for k in ('kind','status','optimizer_updates','epoch','gate',
                'best_epoch','best_validation_score','step_seconds','elapsed_seconds','checkpoints')} for e in experiments]
            state['full_admitted']=False
    except Exception as error:
        state.update(status='FAILED_PRESERVE_ORIGINAL_COMPUTE',error_type=type(error).__name__)
        updates=0
        for path in result.glob('*/RESULT.json'):
            value=json.loads(path.read_text(encoding='utf8'));updates+=value.get('optimizer_updates',0)
        state['optimizer_updates']=max(state['optimizer_updates'],updates)
    finally:
        compact=work/(run_id+'-RESULT.json')
        state['elapsed_seconds']=time.monotonic()-started;state['durable_status']='ARCHIVE_PENDING'
        atomic_json(result/'STATE.json',state);atomic_json(compact,state)
        try:
            archive=work/(run_id+'.tar.gz')
            archive_directory(result,archive,deadline=deadline)
            state['archive']={'name':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive,deadline)}
            atomic_json(compact,state)
            if drive is not None:
                state['drive_archive']=drive.export(archive,run_id,work/(run_id+'-DRIVE.json'))
                state['durable_status']='FULL_BYTE_SHA256_PASS'
                if state.get('admission_marker'):
                    try:drive.finish_run(state['admission_marker'],state['status'])
                    except Exception:state['admission_marker_final_update']='PENDING_RECONCILIATION'
            else:state['durable_status']='NO_STORAGE_ADMISSION'
        except Exception as error:
            state.update(durable_status='UNACCEPTED_KEEP_SAVED_OUTPUT',publication_error_type=type(error).__name__)
        state['elapsed_seconds']=time.monotonic()-started;atomic_json(compact,state)
        print('TOPFACE_DOWNSTREAM_RESULT',json.dumps(state),flush=True)
    return state
