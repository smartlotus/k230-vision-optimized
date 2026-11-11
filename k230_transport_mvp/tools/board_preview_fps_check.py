"""Four finite live TOP benchmarks, then restore the normal continuous preview."""
import sys,json,gc
if sys.implementation.name!='micropython':raise SystemExit('Use CanMV on K230.')
ROOT='/sdcard/transport_mvp/'
sys.path.insert(0,ROOT)
from run_block_preview import main
from k230.capture_io import CaptureIO
profile=json.load(open(ROOT+'vision_pixel_profile.json'))
try:task=json.load(open(ROOT+'confirmed_task_receipt.json'))
except OSError:task=None
storage=CaptureIO();reports=[];aborted=False
for fps,interval in ((30,33),(30,67),(60,33),(60,67)):
    name='preview-fps-%d-%d' % (fps,interval)
    print('FPS_CASE_START:'+name)
    report=main(max_frames=600,report_path=ROOT+name+'.json',inspection_mode='TOP',
        inspection_target=5,inspection_station='BUFFER',pixel_profile=profile,
        confirmed_task=task,qr_enabled=False,capture_path=None,save_samples=False,
        sensor_fps=fps,display_interval_ms=interval)
    reports.append(report)
    summary={'test':'REAL_CSI1_TOP_FPS_MATRIX','source':'REAL_CAMERA','requested_cases':4,
        'completed_cases':sum(r['frames']==600 and r['fatal'] is None for r in reports),
        'reports':reports,'motors_enabled':False,'target_id':5,
        'scope':'One operator-held static bench scene. Pixel presence is not accuracy;60fps exposure/color acceptance remains pending.'}
    storage.save_json(summary,ROOT+'preview_fps_matrix01.json')
    if report['frames']<600 and report['fatal'] is None:
        aborted=True;break
    if report['fatal'] is not None:break
    gc.collect()
if not aborted:
    print('FPS_CHECK_RESTORE_CONTINUOUS')
    main(inspection_mode='TOP',inspection_target=5,inspection_station='BUFFER',pixel_profile=profile,
        control_path=ROOT+'vision_control.json',confirmed_task=task,display_interval_ms=33,sensor_fps=30,
        capture_path=None,report_path=ROOT+'top_metrics_baseline.json')
