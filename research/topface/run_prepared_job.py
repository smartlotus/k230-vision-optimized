"""Continue one frozen job through exact-version observation and publication."""
import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import time

from cloud_store import atomic_json
from collect_compact import collect
from create_prepared_dataset import create
from publisher.publish_completed import publish
from save_prepared_kernel import main as submit


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def run(frozen, staging, token_file, source_commit, similarity=False, training=False, poll_seconds=60):
    frozen, staging = frozen.resolve(), staging.resolve()
    intent = read(frozen/'SUBMISSION_INTENT.json')
    run_id = intent['run_id']
    progress = frozen/'HOST_PROGRESS.json'
    lock = frozen.parents[4]/'.ops/local'/(run_id+'.host.lock')
    descriptor = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,str(os.getpid()).encode('ascii'))
    os.environ['KAGGLE_API_TOKEN'] = token_file.read_text(encoding='utf8').strip()
    state = {'run_id':run_id, 'status':'RUNNING', 'pid':os.getpid(), 'automatic_replay':False}

    def stage(name, **values):
        state.update(stage=name, observed_utc=datetime.now(timezone.utc).isoformat(), **values)
        atomic_json(progress,state)
        print('TOPFACE_PREPARED_JOB '+json.dumps(state),flush=True)

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi(); api.authenticate()
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
        if not (frozen/'DATASET_SAVE_ATTEMPT.json').exists():
            stage('CREATE_PRIVATE_INPUT')
            create(staging,frozen,token_file)
        while not (frozen/'KERNEL_SAVE_RESPONSE.json').exists():
            if (frozen/'KERNEL_SAVE_ATTEMPT.json').exists():
                raise RuntimeError('Original Save response is uncertain; reconcile without another Save')
            stage('WAIT_INPUT_READY')
            status = json.loads(api.dataset_status(intent['dataset'],format='json(status,current_version_number)'))
            if status['status'].lower() == 'ready':
                if status['current_version_number'] != intent['dataset_version']:
                    raise ValueError('Prepared input version changed')
                stage('SAVE_AND_RUN')
                submit(staging.with_name(staging.name+'-kernel'),frozen,token_file,execute=True)
                break
            if status['status'].lower() not in ('creating','processing'):
                raise RuntimeError('Prepared input is unavailable; preserve original input identity')
            time.sleep(poll_seconds)
        response = read(frozen/'KERNEL_SAVE_RESPONSE.json')
        ref = response['ref'].removeprefix('/code/')
        version = response['versionNumber']
        owner, slug = ref.split('/')
        observation = frozen/f'OBSERVATION-v{version}.json'
        deadline = time.monotonic()+intent['provider_max_seconds']+1800
        while True:
            request = ApiGetKernelSessionStatusRequest()
            request.user_name, request.kernel_slug = owner, slug
            request.version_label = f'v{version}'
            try:
                with api.build_kaggle_client() as client:
                    status = client.kernels.kernels_api_client.get_kernel_session_status(request).status.name
            except Exception as error:
                stage('OBSERVATION_RETRY',error_type=type(error).__name__)
                if time.monotonic() >= deadline:
                    raise RuntimeError('Exact-version observation deadline reached') from None
                time.sleep(poll_seconds)
                continue
            atomic_json(observation,{'ref':ref,'version':version,'kernel_id':response['kernelId'],
                'status':status,'observed_utc':datetime.now(timezone.utc).isoformat()})
            stage('KAGGLE_'+status,ref=ref,version=version,kernel_id=response['kernelId'])
            if status == 'COMPLETE':
                break
            if status not in ('RUNNING','QUEUED','INITIALIZING','PENDING') or time.monotonic()>=deadline:
                raise RuntimeError('Inspect the original saved compute state; no automatic replacement')
            time.sleep(poll_seconds)
        collected = frozen/f'compact-v{version}'
        for attempt in range(5):
            try:
                stage('COLLECT_COMPACT')
                collect(ref,version,run_id,collected,token_file,publisher_route=True,
                        similarity=similarity,training=training)
                if (collected/(run_id+'-RESULT.json')).is_file():
                    break
            except Exception as error:
                stage('COMPACT_READ_RETRY',error_type=type(error).__name__)
            if attempt==4:
                raise RuntimeError('Saved compact results remain unavailable; computation stays closed')
            time.sleep(poll_seconds)
        for _ in range(25):
            stage('PUBLISH_ORIGINAL_ARCHIVE')
            receipt = publish(frozen,observation,collected/(run_id+'-RESULT.json'),token_file,source_commit)
            if receipt['status'] == 'DRIVE_EXPORT_FULL_BYTE_SHA_COMMITTED':
                state['status'] = 'COMPUTE_AND_PUBLICATION_VERIFIED'
                stage('COMPLETE')
                return receipt
            if receipt['status'] not in ('SENT_ONCE','in_progress','queued','AWAITING_ORIGINAL_RUN','ORIGINAL_DISPATCH_RECONCILED'):
                raise RuntimeError('Reconcile the original cloud transfer')
            time.sleep(poll_seconds)
        raise RuntimeError('Publication observation budget reached; preserve the original transfer')
    except Exception as error:
        state.update(status='NEEDS_RECONCILIATION',error_type=type(error).__name__)
        stage('STOPPED_PRESERVE_ORIGINAL_IDENTITIES')
        raise RuntimeError('Read HOST_PROGRESS and reconcile existing identities before continuing') from None
    finally:
        os.close(descriptor)
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('frozen','staging','token-file'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--similarity',action='store_true')
    parser.add_argument('--training',action='store_true')
    parser.add_argument('--poll-seconds',type=int,default=60)
    args = parser.parse_args()
    if not 30 <= args.poll_seconds <= 300:
        raise ValueError('Poll interval must be between 30 and 300 seconds')
    print(json.dumps(run(**vars(args)),indent=2))
