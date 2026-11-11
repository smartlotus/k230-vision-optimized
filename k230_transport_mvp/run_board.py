"""Run explicitly in CanMV after copying transport_mvp/ to the SD card."""
import sys

if sys.implementation.name != 'micropython':
    raise SystemExit('Board entry: run with CanMV on K230; for PC use python -m sim.run_mission.')

if '/sdcard/transport_mvp' not in sys.path:
    sys.path.insert(0, '/sdcard/transport_mvp')

from k230.main import main
main()
