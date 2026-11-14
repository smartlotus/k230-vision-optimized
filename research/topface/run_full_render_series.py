"""Submit, observe and archive the ten authorized V1.1 Full render shards."""
import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

from accept_full_shard import accept
from cloud_store import atomic_json
from collect_compact import collect
from create_prepared_dataset import create
from prepare_full_shard import prepare, TOTAL_SCENES, SHARD_SCENES
from publisher.publish_completed import publish
from save_prepared_kernel import main as submit


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def run(token_file, cad, source_commit, poll_seconds=60):
    token_file = token_file.resolve()
    os.environ['KAGGLE_API_TOKEN'] = token_file.read_text(encoding='utf8').strip()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
    cloud = ROOT/'docs/topface/v11/cloud'
    inventory = ROOT/'docs/topface/v11/training/FULL_INVENTORY.json'
    progress_path = ROOT/'docs/topface/v11/training/FULL_SERIES_PROGRESS.json'
    lock = ROOT/'.ops/local/topface-full-series.lock'
    with lock.open('x', encoding='ascii') as stream:
        stream.write(str(os.getpid()))
    state = {'status':'RUNNING','started_utc':datetime.now(timezone.utc).isoformat(),
             'total_planned':TOTAL_SCENES,'accepted_scenes':0,'pid':os.getpid()}

    def stage(name, **fields):
        state.update(stage=name, observed_utc=datetime.now(timezone.utc).isoformat(), **fields)
        atomic_json(progress_path,state)
        print('TOPFACE_FULL_SERIES '+json.dumps(state),flush=True)

    try:
        for shard in range(1, TOTAL_SCENES//SHARD_SCENES+1):
            run_id = f'k230-topface-v11-full-{shard:02d}-a01'
            if inventory.exists() and any(row['run_id']==run_id for row in read(inventory)['shards']):
                state['accepted_scenes'] = read(inventory)['accepted_scenes']
                continue
            frozen = cloud/f'full-{shard:02d}-a01'
            staging = ROOT/f'.ops/local/topface-full-{shard:02d}-a01-input'
            kernel_dir = staging.with_name(staging.name+'-kernel')
            stage('PREPARE_SHARD',shard=shard,run_id=run_id)
            if not (frozen/'SUBMISSION_INTENT.json').exists():
                prepare(shard,staging,cad,ROOT/'docs/topface/v11/training/G4_READINESS.json',
                        cloud/'publisher/k230-topface-v11-training-a02/TRANSFER_RECEIPT.json')
            intent = read(frozen/'SUBMISSION_INTENT.json')
            if not (frozen/'DATASET_SAVE_ATTEMPT.json').exists():
                stage('CREATE_PRIVATE_INPUT')
                create(staging,frozen,token_file)
            stage('WAIT_INPUT_READY')
            while True:
                status = json.loads(api.dataset_status(intent['dataset'],format='json(status,current_version_number)'))
                if status['status'].lower()=='ready':
                    if status['current_version_number']!=1:
                        raise ValueError('Full input version changed')
                    break
                if status['status'].lower() not in ('creating','processing'):
                    raise ValueError('Full input failed to prepare')
                time.sleep(poll_seconds)
            if not (frozen/'KERNEL_SAVE_RESPONSE.json').exists():
                stage('SAVE_AND_RUN')
                submit(kernel_dir,frozen,token_file,execute=True)
            response = read(frozen/'KERNEL_SAVE_RESPONSE.json')
            version = response['versionNumber']
            kernel_id = response['kernelId']
            owner,slug = intent['kernel'].split('/')
            observation_path = frozen/f'OBSERVATION-v{version}.json'
            while True:
                request = ApiGetKernelSessionStatusRequest()
                request.user_name = owner
                request.kernel_slug = slug
                request.version_label = f'v{version}'
                try:
                    with api.build_kaggle_client() as client:
                        status = client.kernels.kernels_api_client.get_kernel_session_status(request).status.name
                except Exception as error:
                    stage('OBSERVATION_RETRY',error_type=type(error).__name__)
                    time.sleep(poll_seconds)
                    continue
                atomic_json(observation_path,{'ref':intent['kernel'],'version':version,
                    'kernel_id':kernel_id,'status':status,'observed_utc':datetime.now(timezone.utc).isoformat()})
                stage('KAGGLE_'+status,version=version,kernel_id=kernel_id)
                if status=='COMPLETE':
                    break
                if status not in ('RUNNING','QUEUED','INITIALIZING','PENDING'):
                    raise RuntimeError('Full render reached a failure state. Preserve and inspect the original output.')
                time.sleep(poll_seconds)
            collected = frozen/f'compact-v{version}'
            stage('COLLECT_COMPACT_RESULT')
            saved = collected/(run_id+'-RESULT.json')
            for attempt in range(5):
                collect(intent['kernel'],version,run_id,collected,token_file,publisher_route=True)
                if saved.is_file():
                    break
                if attempt < 4:
                    time.sleep(poll_seconds)
            if not saved.is_file():
                raise RuntimeError('Exact saved result is unavailable; preserve the completed render')
            stage('PUBLISH_SAVED_ARCHIVE')
            while True:
                publication = publish(frozen,observation_path,saved,token_file,source_commit)
                state['publication_status'] = publication.get('status')
                if publication['status']=='DRIVE_EXPORT_FULL_BYTE_SHA_COMMITTED':
                    break
                if publication['status'] not in ('SENT_ONCE','in_progress','queued','AWAITING_ORIGINAL_RUN',
                                                 'ORIGINAL_DISPATCH_RECONCILED'):
                    raise RuntimeError('Cloud publication stopped. Reconcile this original transfer.')
                time.sleep(poll_seconds)
            stage('ACCEPT_SHARD')
            transfer = cloud/'publisher'/run_id/'TRANSFER_RECEIPT.json'
            accepted = accept(frozen,saved,transfer,inventory)
            state['accepted_scenes'] = accepted['accepted_scenes']
            stage('SHARD_ACCEPTED')
        state['status'] = 'FULL_RENDER_SERIES_ACCEPTED'
        stage('COMPLETE')
    except Exception as error:
        state.update(status='NEEDS_RECONCILIATION',error_type=type(error).__name__,
                     stack_locations=[{'file':Path(frame.filename).name,'function':frame.name,'line':frame.lineno}
                                      for frame in traceback.extract_tb(error.__traceback__)[-5:]])
        stage('STOPPED_PRESERVE_EXISTING_RUN')
        raise RuntimeError('Inspect the exact preserved series state before continuing') from None
    finally:
        lock.unlink()
    return state


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token-file',type=Path,required=True)
    parser.add_argument('--cad',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--poll-seconds',type=int,default=60)
    args = parser.parse_args()
    if not 30<=args.poll_seconds<=300:
        raise ValueError('Poll interval must be 30 to 300 seconds')
    os.chdir(ROOT)
    print(json.dumps(run(**vars(args)),indent=2))
