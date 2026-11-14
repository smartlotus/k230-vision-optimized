"""Independently audit the complete Full simulation before formal training."""
import json
from pathlib import Path
import time

from cloud_store import archive_directory, atomic_json, sha
from dataset_diagnostics import diagnose
from full_dataset import load_shards
from native_job import verify_inputs
from near_similarity import build


def run(input_root, run_id, max_seconds):
    input_root = Path(input_root)
    verify_inputs(input_root)
    config = json.loads((input_root/'FULL_AUDIT_CONFIG.json').read_text(encoding='utf8'))
    if config['run_id'] != run_id:
        raise ValueError('Full audit identity differs from the notebook')
    started = time.monotonic()
    deadline = started + max_seconds
    working = Path('/kaggle/working')
    scratch = Path('/kaggle/temp') if Path('/kaggle/temp').is_dir() else Path('/tmp')
    if not working.is_dir():
        raise ValueError('Full audit requires the Kaggle working directory')
    dataset = scratch/run_id/'dataset'
    output = working/run_id
    output.mkdir(exist_ok=False)
    state = {'schema': 'topface-full-audit-v1.1', 'status': 'STARTED', 'run_id': run_id,
             'optimizer_updates': 0, 'input_manifest_sha256': sha(input_root/'INPUT_MANIFEST.json')}
    try:
        print('TOPFACE_FULL_AUDIT_STAGE VERIFY_SHARDS', flush=True)
        index = load_shards(config['shards'], '/kaggle/input', dataset, config['scenes'],
                            config['seed'], ('train', 'val', 'synthetic_test'), deadline-300)
        print('TOPFACE_FULL_AUDIT_STAGE CHECK_DATASET', flush=True)
        diagnostics = diagnose(dataset, config['scenes'], max_seconds=max(1, int(deadline-time.monotonic()-180)))
        similarity = build(dataset, diagnostics, output/'similarity-review')
        atomic_json(output/'DATASET_DIAGNOSTICS.json', diagnostics)
        state.update(status='FULL_AUDIT_COMPLETE', images=index['images'], instances=index['instances'],
                     shard_inventory_sha256=config['shard_inventory_sha256'],
                     split_images=index['split_images'], dataset_index_sha256=sha(dataset/'AUDIT_V11.json'),
                     integrity_status=diagnostics['status'],
                     scene_and_camera_group_cross_split_count=diagnostics['scene_and_camera_group_cross_split_count'],
                     exact_duplicate_rgb=diagnostics['exact_duplicate_rgb'],
                     camera_groups=diagnostics['camera_groups'],
                     color_coverage=diagnostics['conditional_color_coverage'],
                     cross_split_candidates=diagnostics['perceptual_similarity']['candidate_counts'],
                     similarity_review_evidence={'pairs':len(similarity['pairs']), 'previews':similarity['previews']},
                     source_shards=index['sources'])
    except Exception as error:
        state.update(status='FULL_AUDIT_FAILED', error_type=type(error).__name__)
    finally:
        state['elapsed_seconds'] = time.monotonic()-started
        atomic_json(output/'STATE.json', state)
        try:
            archive = working/(run_id+'.tar.gz')
            archive_directory(output, archive, deadline)
            state['archive'] = {'name': archive.name, 'bytes': archive.stat().st_size,
                                'sha256': sha(archive, deadline)}
            state['durable_status'] = 'PUBLISHER_PENDING_EXACT_SAVED_OUTPUT'
        except Exception as error:
            state.update(durable_status='UNACCEPTED_KEEP_SAVED_OUTPUT', publication_error_type=type(error).__name__)
        atomic_json(working/(run_id+'-RESULT.json'), state)
        print('TOPFACE_FULL_AUDIT_RESULT', json.dumps(state), flush=True)
    return state
