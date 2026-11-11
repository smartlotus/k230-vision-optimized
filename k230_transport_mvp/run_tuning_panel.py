"""Standalone LCD parameter panel. No camera, training or motion starts at boot."""
import sys
if sys.implementation.name!='micropython':raise SystemExit('Use CanMV on K230 for the touch panel.')
if '/sdcard/transport_mvp' not in sys.path:sys.path.insert(0,'/sdcard/transport_mvp')
from k230.tuning.runtime import main
main()
