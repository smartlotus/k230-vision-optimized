"""Start the frozen CPU audit after all ten original render shards are accepted."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from cloud_store import atomic_json
from prepare_full_downstream import prepare
from prepare_full_shard import TOTAL_SCENES
from run_prepared_job import run


ROOT=Path(__file__).resolve().parents[2]


def coordinate(token_file,source_commit):
    local=ROOT/'.ops/local'
    lock=local/'topface-full-audit-after-render.lock'
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,str(os.getpid()).encode('ascii'))
    progress=local/'topface-full-audit-after-render.json'
    inventory=ROOT/'docs/topface/v11/training/FULL_INVENTORY.json'
    cloud=ROOT/'docs/topface/v11/cloud'
    frozen=cloud/'full-audit-a01'
    staging=local/'topface-full-audit-a01-input'
    state={'status':'RUNNING','pid':os.getpid(),'automatic_replay':False}

    def stage(name,**fields):
        state.update(stage=name,observed_utc=datetime.now(timezone.utc).isoformat(),**fields)
        atomic_json(progress,state)
        print('TOPFACE_FULL_AUDIT_CONTINUATION '+json.dumps(state),flush=True)

    try:
        deadline=time.monotonic()+26*3600
        last_accepted=None
        while True:
            accepted=json.loads(inventory.read_text(encoding='utf8'))['accepted_scenes'] if inventory.exists() else 0
            if accepted!=last_accepted:
                stage('WAIT_FULL_INVENTORY',accepted_scenes=accepted,total_planned=TOTAL_SCENES)
                last_accepted=accepted
            if accepted==TOTAL_SCENES:
                break
            if accepted>TOTAL_SCENES or time.monotonic()>=deadline:
                raise RuntimeError('Reconcile the preserved Full inventory; do not replace render jobs')
            time.sleep(60)
        if not (frozen/'SUBMISSION_INTENT.json').exists():
            stage('FREEZE_FULL_AUDIT')
            prepare('full-audit','a01',staging,inventory,
                    ROOT/'docs/topface/v11/training/G4_READINESS.json',
                    cloud/'publisher/k230-topface-v11-training-a02/TRANSFER_RECEIPT.json')
        stage('RUN_OR_CONTINUE_ORIGINAL_CPU_AUDIT')
        run(frozen,staging,token_file,source_commit,similarity=True)
        response=json.loads((frozen/'KERNEL_SAVE_RESPONSE.json').read_text(encoding='utf8'))
        result=frozen/f"compact-v{response['versionNumber']}/k230-topface-v11-full-audit-a01-RESULT.json"
        computed=json.loads(result.read_text(encoding='utf8'))
        if computed['status']!='FULL_AUDIT_COMPLETE':
            raise RuntimeError('Full audit computation needs inspection; preserve the existing archive')
        state['status']='AUDIT_READY_FOR_AGENT_VISUAL_REVIEW'
        stage('REVIEW_SAVED_SIMILARITY_PREVIEWS',frozen=str(frozen),version=response['versionNumber'])
    except Exception as error:
        state.update(status='NEEDS_RECONCILIATION',error_type=type(error).__name__)
        stage('STOPPED_PRESERVE_EXISTING_RUNS')
        raise RuntimeError('Inspect the original Full audit continuation state') from None
    finally:
        os.close(descriptor);lock.unlink()
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--token-file',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    print(json.dumps(coordinate(**vars(parser.parse_args())),indent=2))
