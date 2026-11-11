"""Dual-camera task-controlled observation using current user-assigned roles."""
import sys
if sys.implementation.name != 'micropython':
    raise SystemExit('Use CanMV on K230; run_block_service is the dual-camera observation entry.')
if '/sdcard/transport_mvp' not in sys.path:
    sys.path.insert(0,'/sdcard/transport_mvp')
from k230.block_profile import make_config
from k230.main import main
main(make_config())
