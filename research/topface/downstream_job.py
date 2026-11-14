"""Independent 02 dataset audit / 03 finite Pilot training cloud workers.

The worker is prepared before its inputs exist, but refuses execution without
the original 1000-scene compute/storage acceptance. No Full admission here.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

from cloud_artifacts import fetch_archive,extract_verified_archive
from cloud_store import Drive,archive_directory,atomic_json,sha
from native_job import verify_inputs


def train_phases(pilot,result,input_root,config,deadline,stage,state):
    from train_experiment import train,choose_overfit
    checkpoint=input_root/'nanodet-pretrained.bin'
    refinement=config.get('geometry_refinement')
    if refinement:
        from formal_job import unpack_session
        stage('PILOT_GEOMETRY_REFINEMENT')
        source=unpack_session(Path('/kaggle/input').resolve(),result.parent/'parent',refinement,
                              'pilot_pretrained',('best.pt',),deadline)
        refined=train(pilot,result/'pilot_geometry_refine',checkpoint,kind='pilot_geometry_refine',
            max_epochs=refinement['max_epochs'],deadline=deadline,initial_checkpoint=source/'best.pt',
            initial_checkpoint_sha256=refinement['checkpoint_sha256'],ellipse_shape_loss=True)
        state['status']=('GEOMETRY_REFINEMENT_COMPLETE' if refined['status']=='FINITE_TRAINING_COMPLETE'
                         else 'GEOMETRY_REFINEMENT_BUDGET_STOP')
        return [refined]

    stage('OVERFIT_TRAIN_SANITY')
    selection=choose_overfit(pilot)
    atomic_json(result/'OVERFIT_SELECTION.json',selection)
    overfit=train(pilot,result/'overfit',checkpoint,kind='overfit',max_epochs=200,
                  max_updates=1000,deadline=deadline,selected_scenes=selection['scenes'])
    if not overfit['gate']['pass']:
        state['status']='OVERFIT_GATE_FAILED_NO_PILOT_TRAINING'
        return [overfit]
    stage('PILOT_PRETRAINED_20_EPOCHS')
    pretrained=train(pilot,result/'pilot_pretrained',checkpoint,kind='pilot_pretrained',max_epochs=20,deadline=deadline)
    if pretrained['status']!='FINITE_TRAINING_COMPLETE':
        state['status']='PILOT_BUDGET_STOP_NO_AUTOMATIC_REPLAY'
        return [overfit,pretrained]
    stage('PILOT_RANDOM_CONTROL_20_EPOCHS')
    random=train(pilot,result/'pilot_random_control',checkpoint,kind='pilot_random_control',max_epochs=20,deadline=deadline)
    state['status']='FINITE_PILOT_TRAINING_SEQUENCE_COMPLETE' if random['status']=='FINITE_TRAINING_COMPLETE' else 'CONTROL_BUDGET_STOP'
    return [overfit,pretrained,random]


def mount_training_source(input_root,directory):
    bundle=input_root/'nanodet-source.bin'
    if not bundle.is_file():
        # Older frozen inputs contain the vendor directory directly.
        return
    with zipfile.ZipFile(bundle) as archive:
        members=archive.infolist()
        if len(members)>64 or sum(item.file_size for item in members)>2*1024**2:
            raise ValueError('NanoDet source bundle exceeds expected size')
        for item in members:
            path=Path(item.filename)
            if path.parts[:2]!=('vendor','nanodet_core') or '..' in path.parts:
                raise ValueError('Unexpected path in NanoDet source bundle')
        archive.extractall(directory)
    sys.path.insert(0,str(directory))


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
    offline=config.get('storage_route')=='kaggle_saved_output_authorized_publisher'
    scratch=Path('/kaggle/temp') if Path('/kaggle/temp').is_dir() else Path('/tmp')
    directory=(scratch if offline else work)/run_id;directory.mkdir(exist_ok=False)
    result=directory/'result';result.mkdir()
    state={'schema':'topface-downstream-v1.1','run_id':run_id,'phase':phase,'status':'STARTED',
        'input_manifest_sha256':sha(input_root/'INPUT_MANIFEST.json'),'automatic_replay':False,
        'started_utc':datetime.now(timezone.utc).isoformat(),'max_seconds':max_seconds,
        'optimizer_updates':0,'board_run':False,'real_evaluation':False}
    drive=None
    def mounted_archive(expected):
        root=Path('/kaggle/input').resolve()
        candidates=list(root.rglob(expected['archive']['name']))
        if len(candidates)!=1:raise ValueError('Exact source archive mount is missing/ambiguous')
        path=candidates[0].resolve();receipt=expected['drive_archive']
        if not path.is_relative_to(root) or (path.stat().st_size,sha(path,deadline))!=(receipt['bytes'],receipt['sha256']):
            raise ValueError('Mounted saved archive does not match accepted source identity')
        return path
    def stage(name):
        state['stage']=name;state['elapsed_seconds']=time.monotonic()-started
        atomic_json(result/'STATE.json',state)
        print('TOPFACE_DOWNSTREAM_STAGE',json.dumps({k:state[k] for k in ('run_id','stage','elapsed_seconds')}),flush=True)
    try:
        stage('STORAGE_AND_PARENT_ADMISSION')
        if shutil.disk_usage(work).free<8*1024**3:raise ValueError('Cloud working disk below registered8GiB cap')
        if offline and shutil.disk_usage(scratch).free<4*1024**3:
            raise ValueError('Insufficient cloud scratch space for verified source extraction')
        if offline:
            admission=config['publisher_admission']
            if admission['status']!='DRIVE_EXPORT_FULL_BYTE_SHA_COMMITTED' or admission['folder_id']!=config['folder_id'] or admission['secret_cleanup']!='CONFIRMED_DELETED':
                raise ValueError('Authorized independent publisher route is not qualified')
            host_admission=config['host_run_admission']
            if host_admission['run_id']!=run_id or host_admission['automatic_replay'] is not False:
                raise ValueError('Frozen single-writer host admission missing')
            state['storage_route']='API-only compute, exactKaggleSavedOutput -> authorizedprivateActions -> Drive'
            state['publisher_route_qualification']=admission['github']
            state['host_run_admission']=host_admission
        else:
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
        parent_archive=mounted_archive(parent) if offline else fetch_archive(drive,receipt,directory/'pilot.tar.gz')
        state['parent_members']=extract_verified_archive(parent_archive,directory/'pilot-source')
        pilot=directory/'pilot-source/pilot'
        audit=json.loads((pilot/'AUDIT_V11.json').read_text(encoding='utf8'))
        if audit['status']!='PASS' or audit['images']!=1000:raise ValueError('Parent label audit mismatch')
        state['pilot_archive_sha256']=receipt['sha256'];state['pilot_audit_sha256']=sha(pilot/'AUDIT_V11.json')
        expected_count=1000
        if config.get('supplement_compute'):
            supplement=config['supplement_compute'];extra_receipt=supplement['drive_archive']
            if supplement['status']!='COMPUTE_AND_QA_PASS' or supplement['images_rendered']!=240 or supplement['durable_status']!='FULL_BYTE_SHA256_PASS':
                raise ValueError('Original240-scene supplement compute/QA/storage acceptance required')
            extra_archive=mounted_archive(supplement) if offline else fetch_archive(drive,extra_receipt,directory/'supplement.tar.gz')
            state['supplement_members']=extract_verified_archive(extra_archive,directory/'supplement-source')
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
        from near_similarity import build as build_similarity
        similarity=build_similarity(pilot,diagnostics,result/'similarity-review')
        state['similarity_review_evidence']={'pairs':len(similarity['pairs']),
            'all_listed_identity_checks_different':similarity['all_listed_identity_checks_different'],
            'previews':similarity['previews']}
        # Similarity is a review candidate. Do not silently delete frames or
        # turn grayscale similarity into a fabricated provenance violation.
        near=diagnostics['perceptual_similarity']['candidate_counts'].get('cross_split_contains_target',0)
        if phase=='audit':
            if offline:
                review=work/run_id/'result'/'similarity-review'
                review.parent.mkdir(parents=True,exist_ok=False)
                shutil.copytree(result/'similarity-review',review)
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
            if offline and audit_acceptance.get('composite_audit_sha256')!=state.get('composite_audit_sha256'):
                raise ValueError('Training composite differs from the accepted composite index')
            if near and not audit_acceptance.get('similarity_review_passed'):
                raise ValueError('Target-bearing cross-split similarity review required')
            mount_training_source(input_root,result/'source')
            import torch
            if not torch.cuda.is_available():raise ValueError('PyTorch CUDA unavailable before optimizer admission')
            state['environment']={'torch':str(torch.__version__),'cuda':torch.version.cuda,
                'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                'training_device':'cuda:0','amp':False}
            experiments=train_phases(pilot,result,input_root,config,compute_deadline,stage,state)
            state['optimizer_updates']=sum(experiment['optimizer_updates'] for experiment in experiments)
            state['experiments']=[{k:e.get(k) for k in ('kind','status','optimizer_updates','epoch','gate',
                'best_epoch','best_validation_score','step_seconds','elapsed_seconds','checkpoints')} for e in experiments]
            if offline:
                for experiment in experiments:
                    name=experiment['kind']
                    target=work/run_id/'result'/name
                    target.mkdir(parents=True,exist_ok=False)
                    report_names=('BEST_METRICS.json','RESULT.json')
                    if name=='pilot_geometry_refine':report_names+=('BASELINE.json',)
                    for report_name in report_names:
                        report=result/name/report_name
                        if report.is_file():shutil.copy2(report,target/report_name)
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
            else:state['durable_status']='PUBLISHER_PENDING_EXACT_SAVED_OUTPUT' if offline else 'NO_STORAGE_ADMISSION'
        except Exception as error:
            state.update(durable_status='UNACCEPTED_KEEP_SAVED_OUTPUT',publication_error_type=type(error).__name__)
        state['elapsed_seconds']=time.monotonic()-started;atomic_json(compact,state)
        print('TOPFACE_DOWNSTREAM_RESULT',json.dumps(state),flush=True)
    return state
