"""Publish the exact saved archive for an already-completed Kaggle run."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from publisher.control import dispatch, folder_for, observe, prepare, run_lock
from publisher.plan_saved_output import plan
from cloud_store import atomic_json


def publish(frozen, observation, result, token_file, source_commit):
    requested = plan(frozen, observation, result)
    run_id = requested['run_id']
    transfer = folder_for(run_id)
    frozen_plan = frozen/'PUBLISHER_PLAN.json'
    with run_lock(run_id):
        if frozen_plan.exists():
            if json.loads(frozen_plan.read_text(encoding='utf8')) != requested:
                raise ValueError('The existing publisher plan binds a different artifact')
        else:
            atomic_json(frozen_plan, requested)
        binding_path = transfer/'BINDING.json'
        if not binding_path.exists():
            prepare(frozen_plan, token_file, source_commit)
        else:
            binding = json.loads(binding_path.read_text(encoding='utf8'))
            if any(binding[key] != requested[key] for key in ('run_id', 'archive', 'prep', 'source_archive_path')):
                raise ValueError('Existing transfer binding differs from the completed Kaggle run')
        if (transfer/'TRANSFER_RECEIPT.json').exists() or (transfer/'DISPATCHED.json').exists():
            return observe(run_id)
        if (transfer/'DISPATCH_ATTEMPT.json').exists():
            result = observe(run_id)
            if result['status']!='AWAITING_ORIGINAL_RUN':
                return result
            return dispatch(run_id)
        return dispatch(run_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('frozen','observation','result','token-file'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    try:
        print(json.dumps(publish(**vars(parser.parse_args())), indent=2))
    except Exception as error:
        raise SystemExit(f'Publication requires reconciliation: {type(error).__name__}') from None
