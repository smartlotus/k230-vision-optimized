"""Observation service using the board panel's saved, locked profile; no motors."""
import sys
if sys.implementation.name!='micropython':raise SystemExit('Use CanMV on K230 for the saved-profile service.')
if '/sdcard/transport_mvp' not in sys.path:sys.path.insert(0,'/sdcard/transport_mvp')
from k230.tuning.integration import load_locked_config
from k230.main import main
main(load_locked_config())
