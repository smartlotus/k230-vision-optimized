"""Unified pixel-only five-mode workbench. Edit MODE or use the control file."""
import sys
if sys.implementation.name!='micropython':raise SystemExit('Use CanMV on K230 for the visual workbench.')
sys.path.insert(0,'/sdcard/transport_mvp')
import json
from run_block_preview import main
MODE='COLOR'
TARGET_ID=1
STATION='RAW'
CONTROL_PATH='/sdcard/transport_mvp/vision_control.json'
try:
    with open('/sdcard/transport_mvp/vision_pixel_profile.json') as stream:profile=json.load(stream)
except OSError:profile=None
RESTORE_CONFIRMED_TASK=True
task=None
if RESTORE_CONFIRMED_TASK:
    try:
        with open('/sdcard/transport_mvp/confirmed_task_receipt.json') as stream:task=json.load(stream)
    except OSError:pass
main(inspection_mode=MODE,inspection_target=TARGET_ID,inspection_station=STATION,
     pixel_profile=profile,control_path=CONTROL_PATH,confirmed_task=task,
     report_path='/sdcard/transport_mvp/vision_workbench_latest.json')
