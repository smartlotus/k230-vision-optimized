"""Reproduce fault exits, package checks and the coarse-version delivery audit."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
from sim.run_mission import FAULTS
from tools.build_board_package import build

ROOT=Path(__file__).resolve().parents[1]


def main():
    faults=[]
    for fault in FAULTS:
        destination=ROOT/'evidence'/'faults'/(fault+'.json')
        command=[sys.executable,'-X','utf8','-m','sim.run_mission','--task','156+123+516+231',
                 '--inject',fault,'--output',str(destination)]
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        data=json.loads(destination.read_text(encoding='utf-8'))
        if result.returncode!=1 or data['summary']['status']!='ERROR':
            raise ValueError('Fault was not rejected: '+fault)
        faults.append({'fault':fault,'exit_code':result.returncode,'summary':data['summary'],
                       'evidence':destination.relative_to(ROOT).as_posix()})
    (ROOT/'evidence'/'fault-matrix.json').write_text(json.dumps(faults,indent=2),encoding='utf-8')
    package=build(ROOT/'dist'/'transport_mvp.zip')
    with tempfile.TemporaryDirectory(prefix='kmvp-package-') as temporary:
        repeated=build(Path(temporary)/'repeat.zip')
        if package['sha256']!=repeated['sha256']:
            raise ValueError('Package rebuild differs')
        with zipfile.ZipFile(package['file']) as archive:
            manifest=json.loads(archive.read('transport_mvp/manifest.json'))
            for name,digest in manifest.items():
                raw=archive.read('transport_mvp/'+name)
                if hashlib.sha256(raw).hexdigest()!=digest:
                    raise ValueError('Archive content hash mismatch: '+name)
                ast.parse(raw.decode('utf-8'))
            archive.extractall(temporary)
        result=subprocess.run([sys.executable,str(Path(temporary)/'transport_mvp/run_board.py')],
                               capture_output=True,text=True)
        if result.returncode==0 or 'CanMV on K230' not in result.stderr:
            raise ValueError('Board entry did not reject desktop Python')
        preview=subprocess.run([sys.executable,str(Path(temporary)/'transport_mvp/run_multicam_preview.py')],
                               capture_output=True,text=True)
        if preview.returncode==0 or 'CanMV' not in preview.stderr:
            raise ValueError('Multicam entry did not reject desktop Python')
        for entry in ('run_block_preview.py','run_block_service.py','run_vision_preview.py','run_tuning_panel.py','run_tuned_service.py','run_live_workbench.py'):
            guarded=subprocess.run([sys.executable,str(Path(temporary)/'transport_mvp'/entry)],capture_output=True,text=True)
            if guarded.returncode==0 or 'CanMV' not in guarded.stderr:
                raise ValueError('Block entry did not reject desktop Python: '+entry)
    package['file']='dist/transport_mvp.zip'
    package.update(source_manifest=manifest,host_syntax_check='PASS',archive_hash_check='PASS',
                   reproducible_rebuild='PASS',pc_entry_guard='PASS')
    (ROOT/'evidence'/'package.json').write_text(json.dumps(package,indent=2),encoding='utf-8')
    required=['DEVELOPMENT_PLAN.md','README.md','STATUS.md','HUMAN_ACTIONS.md','DECISIONS.md',
              'contracts/schema.md','contracts/protocol_samples.txt','HANDOFF_CORE_SIM.md',
              'HANDOFF_VISION.md','HANDOFF_COMMS.md','data/spec/source-manifest.json',
              'evidence/validation.json','evidence/validation.log','evidence/mission-official.json',
              'evidence/mission-capacity1.json','evidence/photos/photo_analysis.json',
              'evidence/board/receipt.json','evidence/board/attempt01-camera-timeout.json',
              'evidence/board/attempt02-camera-smoke.json','evidence/board/runtime-smoke.json',
              'evidence/board/qr-fixture-report.json','evidence/board/native-fixtures-attempt01.json',
              'evidence/board/native-fixtures-attempt02.json','evidence/ring-candidate-replay.json',
              'evidence/board/native-fixtures-reconnect.json','evidence/board/lifecycle-reconnect.json',
              'evidence/board/deployment-reconnect.json','evidence/board/capture-soak-reconnect.json',
              'evidence/board/entry-reconnect.json','evidence/board/reconnect-summary.json',
              'evidence/board/three-camera-probe.json','evidence/board/three-camera-qualification.json',
              'evidence/board/three-camera-live-receipt.json','evidence/board/three-camera-current-diagnostic.json',
              'evidence/board/three-camera-isolation-failure.json','evidence/board/three-camera-after-reset.json',
              'LIVE_BLOCK_PLAN.md','BENCH_TEST_GUIDE.md','k230/block_profile.py','vision/block_detector.py',
              'evidence/board/block-service-check.json','evidence/board/block-live-six-attempt02.json',
              'evidence/board/block-six-label-evaluation.json','evidence/board/qr-live-task-attempt02.json',
              'evidence/board/block-v7-600.json','evidence/board/black-density-diagnostic.json',
              'evidence/board/block-manual-check-user-report.json','evidence/mission-from-live-qr-sim.json',
              'evidence/board/block-v9-600.json','evidence/board/black-miss-probe300.json',
              'evidence/board/exposure-comparison.json',
              'VISUAL_COMPLETION.md','FULL_WORKFLOW_PLAN.md','run_vision_preview.py',
              'evidence/board/block-v9-long-soak.json','evidence/board/block-v10-600.json',
              'evidence/board/vision-modes-attempt01.json','evidence/board/vision-modes-attempt02.json',
              'evidence/board/top-round-fixture.json','evidence/board/native-fixtures-visual-workbench.json',
              'evidence/board/visual-deployment-readback.json','evidence/board/workflow-default-refusal.json',
              'evidence/workflow-official.json','evidence/workflow-fault-matrix.json',
              'evidence/board/block-v10-continuous-attempt01.json',
              'evidence/board/block-stability-1800.json','evidence/board/block-stability-1800-raw.json',
              'HANDOFF_CAMERA_ROLE_SWAP.md','evidence/board/role-swap-final-deployment.json',
              'evidence/board/role-swap-after-power-cycle-probe.json',
              'evidence/board/role-swap-isolation-each-run.json',
              'evidence/board/camera-format-diagnostic-after-power.json','evidence/board/camera-recheck-no-swap01.json','evidence/board/csi1-role-recovery600.json',
              'HANDOFF_CSI1_TOP_LIVE.md','evidence/board/capture-top-final-readback.json',
              'evidence/board/native-capture-io-guard.json','evidence/board/preview-storage-exit-attempt01.json',
              'evidence/board/csi1-top-native-fresh-attempt01.json','evidence/board/csi1-top-native-fresh-attempt02.json',
              'evidence/board/csi1-top-native-fresh-attempt04.json','evidence/board/fresh-black-top-component-diagnostic.json',
              'data/visual_profiles/csi1-topbench03-current-scene.json',
              'ON_DEVICE_TUNING_PLAN.md','TUNING_PANEL_GUIDE.md','run_tuning_panel.py','run_tuned_service.py',
              'evidence/board/tuning-panel-deployment-20261102.json','evidence/board/tuning-config-native-20261102.json',
              'evidence/board/standalone-startup-20261102.json',
              'HANDOFF_TOP640_20261102.md','k230/tuning/sampling.py','tests/test_tuning_sampling.py',
              'evidence/board/tuning-sampling-deployment-20261102.json',
              'evidence/board/csi1-top640-angle01-fresh600-attempt01.json',
              'evidence/board/csi1-top640-angle02-fresh600-attempt01.json','evidence/board/csi1-top640-angle02-black-pair100.json']
    for name in required:
        if not (ROOT/name).is_file():
            raise ValueError('Missing delivery artifact: '+name)
    official=json.loads((ROOT/'evidence/mission-official.json').read_text(encoding='utf-8'))
    summary=official['summary']
    if (summary['status']!='DONE' or summary['counters']!={'picks':12,'placements':12}
            or not summary['field_complete'] or summary['station']!='START_1'):
        raise ValueError('Official mission final ledger invalid')
    stacks=[s['slot'] for s in official['transitions'] if s['batch']==2 and s['dest']=='BUFFER']
    if stacks!=[2,1,3]:
        raise ValueError('Same-color stacking mismatch')
    validation=json.loads((ROOT/'evidence/validation.json').read_text(encoding='utf-8'))
    if validation['exit_code']!=0:
        raise ValueError('Host tests failed')
    # Preserve historical receipts; an opt-in live delivery has its own exact source binding.
    live_receipt=ROOT/'evidence/board/live-20261103/deployment.json'
    deployed=json.loads((live_receipt if live_receipt.exists() else ROOT/'evidence/board/tuning-sampling-deployment-20261102.json').read_text(encoding='utf-8'))
    actual={r['file']:r['actual'] for r in deployed['files'] if r['match']}
    if actual!=manifest or deployed['matched_files']!=len(manifest):
        raise ValueError('Current production package differs from native byte readback')
    if deployed['package_sha256']!=package['sha256']:raise ValueError('Current deployed package identity differs')
    new_angle=json.loads((ROOT/'evidence/board/csi1-top640-angle01-fresh600-attempt01.json').read_text(encoding='utf8'))
    if (new_angle['source']!='REAL_CAMERA' or new_angle['camera_id']!=1 or not new_angle['complete']
            or new_angle['frame_denominator']!=600 or new_angle['passed_current_scene_gate']
            or new_angle['saved_generation'] is not None or new_angle['motors_enabled']):
        raise ValueError('New-angle failure/source or refusal evidence changed')
    live_top=json.loads((ROOT/'evidence/board/csi1-top-native-fresh-attempt04.json').read_text(encoding='utf-8'))
    if not live_top['complete'] or live_top['source']!='REAL_CAMERA' or live_top['camera_id']!=1 or live_top['motors_enabled']:
        raise ValueError('Real CSI1 TOP source/denominator missing')
    if set(live_top['per_color'])!={str(c) for c in range(1,7)}:raise ValueError('Real TOP six-color report required')
    for row in live_top['per_color'].values():
        obs=row['last']
        if (row['frames']<100 or row['pixels_found']!=row['frames'] or row['errors']
                or not 0<row['stable_frames']<=row['pixels_found'] or obs['camera_id']!=1
                or obs['is_simulated'] or obs['x_mm'] is not None or obs['y_mm'] is not None
                or obs['target_pose_valid']):raise ValueError('Real TOP availability/refusal check failed')
    modes=json.loads((ROOT/'evidence/board/vision-modes-attempt02.json').read_text(encoding='utf-8'))
    if not modes['check_passed'] or set(modes['per_mode_frames'])!={'COLOR','QR','RING','TOP','LANE','IDLE'}:
        raise ValueError('Native visual mode check incomplete')
    top=json.loads((ROOT/'evidence/board/top-round-fixture.json').read_text(encoding='utf-8'))
    if top['passed']!=6 or not top['mismatch_rejected'] or not top['is_simulated']:
        raise ValueError('Simulated native TOP fixture failed or lost source identity')
    workflow=json.loads((ROOT/'evidence/workflow-official.json').read_text(encoding='utf-8'))
    if (workflow['summary']['status']!='DONE' or not workflow['field_complete']
            or workflow['summary']['counters']!={'picks':12,'placements':12}
            or workflow['controller_submissions']!=31 or len(workflow['controller_receipts'])!=62):
        raise ValueError('Nonblocking simulation incomplete')
    matrix=json.loads((ROOT/'evidence/workflow-fault-matrix.json').read_text(encoding='utf-8'))
    if len(matrix)!=8 or any(r['status']=='DONE' or r['field_complete'] for r in matrix):
        raise ValueError('Nonblocking simulation fault completed unexpectedly')
    stability=json.loads((ROOT/'evidence/board/block-stability-1800.json').read_text(encoding='utf-8'))
    if (not stability['check_completed'] or stability['frames']!=1800
            or stability['replayed_frames_ignored'] or stability['source']!='REAL_CAMERA'
            or not 0<=stability['all_six_unique_stable_frames']<=stability['all_six_unique_eligible_frames']<=1800):
        raise ValueError('Static stability evidence denominator/source invalid')
    for row in stability['per_color'].values():
        if (not 0<=row['unique_stable_frames']<=row['unique_eligible_frames']<=row['visible_frames']<=1800
                or row['eligible_centers']['samples']!=row['unique_eligible_frames']
                or row['stable_centers']['samples']!=row['unique_stable_frames']):
            raise ValueError('Static stability subset accounting invalid')
    panel=json.loads((ROOT/'evidence/board/tuning-config-native-20261102.json').read_text(encoding='utf8'))
    if (panel['cameras_started'] is not False or panel['motors_enabled'] is not False
            or panel['optimizer_updates']!=0 or panel['checks']['geometry']!=[640,360]
            or any(panel['checks'][k] is not True for k in ('slot_restored','locked_slot_guard',
                'draft_not_implicitly_applied','parameter_restored','service_uses_saved_sha'))):
        raise ValueError('Native manual-panel configuration checks incomplete')
    startup=json.loads((ROOT/'evidence/board/standalone-startup-20261102.json').read_text(encoding='utf8'))
    if not startup['created_new_main_py'] or not startup['executable_AST_equal_ignoring_module_docstring'] or startup['camera_default_enabled']:
        raise ValueError('Standalone startup installation evidence incomplete')
    requirements=[
        {'id':'S0','requirement':'Source inventory, development plan and frozen contracts','state':'DONE','evidence':['data/spec/source-manifest.json','DEVELOPMENT_PLAN.md','contracts/schema.md']},
        {'id':'S1','requirement':'Complete two-batch ledger and original start return','state':'SIM_PASS','evidence':['evidence/mission-official.json','evidence/mission-capacity1.json','tests/test_mission_sim.py']},
        {'id':'V1','requirement':'QR validation, 3 fresh frames and once-per-session submission','state':'HISTORICAL_REAL_CSI1_LIVE_PASS / CURRENT_CSI2_TARGET_PENDING','evidence':['tests/test_vision.py','tests/test_task_preview.py','tests/test_camera_roles.py','evidence/board/qr-fixture-report.json','evidence/board/qr-live-task-attempt02.json'],'live':'Historical real CSI1 QR generated the31-step plan. Current scanner is CSI2 after explicit role swap; historical receipt retains CSI1 identity, physical steps zero. New CSI2 live QR acceptance pending.'},
        {'id':'V2','requirement':'6-color configurable candidates and bbox center','state':'HISTORICAL_CSI2_BENCH_PASS / CURRENT_CSI1_CAPTURE_RECOVERED_TARGETS_PENDING / ROBUSTNESS_PENDING','evidence':['k230/block_profile.py','vision/block_detector.py','evidence/board/block-v10-600.json','evidence/board/block-v9-long-soak.json','evidence/board/block-six-label-evaluation.json','evidence/board/block-service-check.json','evidence/board/black-miss-probe300.json','evidence/board/exposure-comparison.json'],'live':'v10 side-view observation window 0,35,320,150: 600-frame bench about27.36 detections/s, all six visible600 and stable597. Black size floor excludes measured background shadow, specific to this distance. Original v9 long-soak poor stability remains preserved. Current-scene availability is not general accuracy. Distance/light, TOP and metric calibration pending.'},
        {'id':'V3','requirement':'Stable local window with honest unknown turntable phase/omega','state':'HOST_PASS','evidence':['vision/tracker.py','tests/test_vision.py']},
        {'id':'V4','requirement':'Concentric-circle clustering and explicit station ID mapping','state':'HOST_PASS / NEW_BOARD_FILE_PASS / ORIGINAL_FILE_ID_UNKNOWN','evidence':['vision/ring.py','tests/test_ring_replay.py','evidence/board/native-fixtures-reconnect.json','evidence/board/native-fixtures-attempt02.json'],'live':'HW_PENDING: no measured real ring samples or direction; only explicitly simulated file geometry verified'},
        {'id':'V5','requirement':'Bound homography and independent validation; no uncalibrated MM','state':'HOST_PASS','evidence':['vision/mapping.py','tools/calibrate_homography.py','tests/test_tools.py','tests/test_safety.py'],'live':'HW_PENDING: H null, no motion enabled'},
        {'id':'V6','requirement':'Auxiliary lane and separate top plane','state':'HISTORICAL_320x240_TOP_PASS / NEW_640x360_BLACK_FAILED / LANE_REAL_TARGET_PENDING','evidence':['vision/lane.py','vision/top.py','k230/visual_inspector.py','k230/vision_adapter.py','evidence/board/csi1-top-native-fresh-attempt04.json','evidence/board/csi1-top-native-fresh-attempt01.json','data/visual_profiles/csi1-topbench03-current-scene.json','evidence/board/csi1-top640-angle01-fresh600-attempt01.json'],'live':'Historical320x240 CSI1 six-color616frames availability retained. New640x360 sequential600:five colors100found98stable each; black22found3stable with body-center issue, candidate unsaved. No independent accuracy or H; actual lane and broader TOP pose/light acceptance pending.'},
        {'id':'S2','requirement':'Bounded K230 camera and runtime with correct failure states','state':'REAL_CAPTURE_PASS / REAL_RUNTIME_PASS / REAL_ENTRY_PASS','evidence':['evidence/board/capture-soak-reconnect.json','evidence/board/lifecycle-reconnect.json','evidence/board/entry-reconnect.json'],'limit':'Live target recognition deferred; not full Gate2 acceptance'},
        {'id':'S3','requirement':'Staged cameras and bounded UART / STM32 parser','state':'HOST_PASS / HISTORICAL_MULTI_PASS / CURRENT_CSI1_CAPTURE_RECOVERED_TARGETS_PENDING / CSI0_QUARANTINED','evidence':['evidence/board/three-camera-qualification.json','evidence/board/camera-format-diagnostic-after-power.json','evidence/board/camera-recheck-no-swap01.json','evidence/board/csi1-role-recovery600.json','tests/test_protocol.py'],'live':'Latest user confirms no component swap:CSI1/2 each30/30 frames; actual role entry CSI1 main600 andCSI2 auxiliary101 frames,zero capture errors. Video quality and target recognition remain pending; earlier0-frame failures retained. CSI0 quarantined; UART/STM32 pending.'},
        {'id':'S4','requirement':'Default refusal of unapproved, simulated, stale and mismatched motion','state':'HOST_PASS','evidence':['ports/motion_port.py','tests/test_safety.py'],'live':'No real motion authorization or claims'},
        {'id':'S5','requirement':'Runnable package, instructions, manual worksheets and evidence','state':'DELIVERED_COARSE_VERSION','evidence':['README.md','HUMAN_ACTIONS.md','DECISIONS.md','evidence/package.json']},
        {'id':'UI01','requirement':'Standalone manual tuning with correct aspect and persistent profiles',
         'state':'HOST_AND_NATIVE_DEMO_PASS / INSTALLED_STARTUP / PHYSICAL_COLD_BOOT_TOUCH_PENDING',
         'evidence':['TUNING_PANEL_GUIDE.md','tests/test_tuning_system.py','evidence/board/tuning-config-native-20261102.json',
            'evidence/board/standalone-startup-20261102.json'],
         'limit':'New640x360 angle02: black paired100 improved83/57 to100/98 afterL17; latest600 five colors100/98 butgreen95/83, groupnot saved. Physical panel alignment and disconnect/power-cycle remain unverified.'}
    ]
    audit={'scope':'Coarse software DoD in ai01.md: complete SIM plus testable hardware adapters and explicit manual inputs; not real competition acceptance',
           'requirements':requirements,'artifacts':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in required},
           'visual_stage':{'entry':'run_tuning_panel.py','legacy_visual_entry':'run_vision_preview.py','current_profile':'MANUAL_UNVERIFIED_640x360',
               'object_camera_id':1,'front_camera_id':2,'current_CSI1_capture':'REAL_640x360_600_FRESH_ZERO_CAPTURE_ERRORS',
               'native_mode_check':'CURRENT_CSI1_MAIN_CAPTURE600_CSI2_AUX101_PASS_TARGETS_PENDING','exact_deployed_sources':len(actual),
               'native_TOP_fixture':'SIMULATED_FILE_ONLY','live_TOP':'NEW_640x360_BLACK_IMPROVED_GREEN95_FOUND83_STABLE_GROUP_UNSAVED',
               'live_RING_LANE':'PENDING_REAL_TARGETS',
               'user_setup_changes_deferred':False,'next_input':'NONE_USER_WRAPPING_UP_ACCOUNT_SWITCH_NEXT_GREEN_PAIRED_TEST',
               'camera_preview_enabled':True,'new_experiments_paused_for_account_switch':True,'tuning_scope':'LCD_TOUCH_PROFILE_AND_NEW_640x360_REAL_TOP_TUNING',
               'handoff':'HANDOFF_TOP640_20261102.md'},
           'manual_real_world_work_remains':True,'faults_rejected':len(faults),
           'official_second_buffer_slots':stacks,'host_test_exit_code':validation['exit_code']}
    (ROOT/'evidence'/'requirements-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({'delivery_audit':'PASS_WITH_EXPLICIT_HW_PENDING','faults_rejected':len(faults),
                      'board_source_files':len(manifest),'package_sha256':package['sha256']}))


if __name__=='__main__':
    main()
