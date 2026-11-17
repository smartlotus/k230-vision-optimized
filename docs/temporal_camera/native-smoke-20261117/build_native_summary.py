"""Summarize existing native receipts without rerunning a closed attempt."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUNS = ('c1-10b', 'c1-60c', 'c1-10d', 'c1-10e', 'c1-10f')
rows = []
for name in RUNS:
    path = OUT / (name + '-readback.json')
    data = json.loads(path.read_text(encoding='utf8'))
    report, entry = data['.json'], data['-entry.json']
    assert report['status'] == 'MEASURED' and entry['status'] == 'RETURNED'
    assert report['frames'] == report['requested_frames'] and report['dropped'] == 0
    assert report['cleanup_errors'] == [] and report['memory_sampling_failures'] == 0
    assert report['unlocalized'] == report['frames']
    assert report['last_prediction']['position_valid'] is False
    assert all(report['latency'][stage]['samples'] == report['frames']
               for stage in ('capture', 'ai2d', 'kpu', 'filter', 'cycle'))
    if name in ('c1-10e', 'c1-10f'):
        assert report['final_memory']['sys_mmz']['used'] == 0
    rows.append(dict(run_id=entry['run_id'], frames=report['frames'], attempts=report['requested_frames'],
                     dropped=report['dropped'], fps=report['camera_to_result_fps'],
                     seconds=report['wall_seconds'], cleanup_errors=report['cleanup_errors'],
                     final_mmz_used=report['final_memory']['sys_mmz']['used'],
                     receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
post = json.loads((OUT / 'POSTFLIGHT.json').read_text(encoding='utf8'))
assert post['status'] == 'PASS' and post['final_mmz'][1] == 0
assert len(post['files']) == 9 and all(r['matches'] for r in post['files'].values())
assert len(post['other_tmp_files']) == 10 and all(r['matches'] for r in post['other_tmp_files'].values())
assert post['startup']['sha256'] == 'c0b89b0b48864ff51e0e1d9a07082d306dbaf301aa4b6c95432ea6e04dbc640c'
assert not post['boot_present']
source = ROOT / 'research/temporal_camera/board_multiroute.py'
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_sha == post['files']['/tmp/k230_temporal_v1_c1r3/board_multiroute.py']['sha256']
short = json.loads((OUT / 'c1-60c-readback.json').read_text(encoding='utf8'))['.json']
summary = dict(
    status='BOUNDED_NATIVE_CNN1_FILTER_ENGINEERING_PASS',
    observed_utc=datetime.now(timezone.utc).isoformat(),
    tested_route='a06 CNN1 KModel + CPU Bayesian temporal grid filter',
    source_sha256=source_sha,
    model_sha256=short['kmodel_sha256'], model_bytes=short['model_bytes'],
    firmware='v1.8-36-gae2f20d', nncase_runtime=short['nncase_runtime'],
    remote_root='/tmp/k230_temporal_v1_c1r3',
    native_runs=rows, full_pipeline_frames=sum(r['frames'] for r in rows),
    full_pipeline_capture_attempts=sum(r['attempts'] for r in rows),
    dropped=sum(r['dropped'] for r in rows),
    startup_entry_failure=dict(run_id='c1_10a', status='ENTRY_FAILED', captured_frames=0,
                              cause='Added tuple startswith check unsupported by MicroPython', fixed=True),
    short_60_frame_measurement=dict(fps=short['camera_to_result_fps'],
                                    wall_seconds=short['wall_seconds'], latency=short['latency']),
    source_and_model_readback_files=9, preserved_other_tmp_files_verified=10,
    final_mmz_used=0, final_script_running=False,
    startup_unchanged=True, boot_unchanged_absent=True,
    real_camera_accuracy='NOT_RUN', road_test='NOT_RUN', long_soak='NOT_RUN',
    native_tcn_gru='NOT_RUN', native_pixel_equivalence='NOT_RUN',
    position_valid=False, qualification='EXPERIMENTAL_ACCURACY_HOLD',
    known_storage_issue='SD write returned30(read-only); testing used explicitly isolated /tmp and host readback',
    approved_cleanup_directory='/tmp/xyusb1117e',
    backup='All19 files fully downloaded and size/SHA verified; only9files in approved e directory deleted',
    transport='Builtin ubinascii, bytes input,8192-byte blocks; whole stored files reopened for SHA',
    prior_64k_transport_attempt='UNKNOWN; one controlled reconnect, not replayed',
    production_deployment=False)
assert summary['full_pipeline_frames'] == summary['full_pipeline_capture_attempts'] == 100
(OUT / 'NATIVE_SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf8')
print(json.dumps({k: summary[k] for k in ('status', 'full_pipeline_frames', 'dropped', 'final_mmz_used',
                                        'final_script_running', 'source_sha256')}, indent=2))
